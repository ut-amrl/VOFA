import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio
import torch
import torch.nn.functional as F
from model import *
from utils.buffer import ExperienceBuffer
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder
from envs import *


class PPO:

    def __init__(self, cfg, test=False):
        self.test = test
        self.cfg = cfg
        if self.test:
            self.cfg["basic"]["headless"] = False
            self.cfg["env"]["num_envs"] = 1
            self.cfg.setdefault("camera", {})["view_depth"] = True
        # prepare the environment
        self._set_seed()
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)

        self.device = self.cfg["basic"]["rl_device"]
        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.model = eval(self.cfg["algorithm"]["model_class"])(
            self.env.num_actions,
            self.env.num_prop,
            self.env.hist_len,
            self.env.num_privileged_obs,
            self.cfg["algorithm"]["init_logstd"]
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._load()

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.hist_len, self.env.num_prop))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)
        self.buffer.add_buffer("next_obses", (self.env.hist_len, self.env.num_prop))
        self.buffer.add_buffer("next_privileged_obses", (self.env.num_privileged_obs,))

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load(self):
        if not self.cfg["basic"]["checkpoint"]:
            return
        if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
            self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        # do not load logstd
        model_dict["model"].pop("logstd")
        self.model.load_state_dict(model_dict["model"], strict=False)
        try:
            self.env.curriculum_prob = model_dict["curriculum"]
        except Exception as e:
            print(f"Failed to load curriculum: {e}")
        try:
            self.optimizer.load_state_dict(model_dict["optimizer"])
        except Exception as e:
            print(f"Failed to load optimizer: {e}")

    def train(self):
        self.recorder = Recorder(self.cfg)
        obs, infos = self.env.reset()
        # initialize at random episode length
        if self.cfg["runner"]["random_init_episode_length"]:
            max_episode_length = np.ceil(self.cfg["rewards"]["episode_length_s"] / self.env.dt)
            self.env.episode_length_buf = torch.randint(0, int(max_episode_length), (self.env.num_envs,), device=self.device)

        next_obs = None
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        for it in range(self.cfg["basic"]["max_iterations"]):
            # within horizon_length, env.step() is called with same act
            with torch.no_grad():
                for n in range(self.cfg["runner"]["horizon_length"]):
                    obs, infos = self.env.handle_reset()
                    privileged_obs = infos["privileged_obs"].to(self.device)

                    self.buffer.update_data("obses", n, obs)
                    self.buffer.update_data("privileged_obses", n, privileged_obs)
                    dist = self.model.act(obs, privileged_obs)  # teacher policy needs privileged obs
                    act = dist.sample()
                    next_obs, rew, done, infos = self.env.step(act)
                    next_obs, rew, done = next_obs.to(self.device).clone(), rew.to(self.device).clone(), done.to(self.device).clone()
                    next_privileged_obs = infos["privileged_obs"].to(self.device).clone()
                    self.buffer.update_data("actions", n, act)
                    self.buffer.update_data("rewards", n, rew)
                    self.buffer.update_data("dones", n, done)
                    self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                    self.buffer.update_data("next_obses", n, next_obs)
                    self.buffer.update_data("next_privileged_obses", n, next_privileged_obs)
                    ep_info = {"reward": rew}
                    ep_info.update(infos["rew_terms"])
                    self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))
                
                values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
                next_values = self.model.est_value(self.buffer["next_obses"], self.buffer["next_privileged_obses"])

                # special handling for time outs
                self.buffer["rewards"][self.buffer["time_outs"]] += next_values[self.buffer["time_outs"]] * self.cfg["algorithm"]["gamma"]
                advantages = discount_values(
                    self.buffer["rewards"],
                    self.buffer["dones"] | self.buffer["time_outs"],
                    values,
                    next_values,
                    self.cfg["algorithm"]["gamma"],
                    self.cfg["algorithm"]["lam"],
                )
                returns = values + advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # store old action log probs
                old_dist = self.model.act(self.buffer["obses"], self.buffer["privileged_obses"])
                old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

            mean_value_loss = 0
            mean_actor_loss = 0
            mean_bound_loss = 0
            mean_entropy = 0
            for n in range(self.cfg["runner"]["mini_epochs"]):
                values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
                value_loss = F.mse_loss(values, returns)

                dist = self.model.act(self.buffer["obses"], self.buffer["privileged_obses"])
                actions_log_prob = dist.log_prob(self.buffer["actions"]).sum(dim=-1)
                actor_loss = surrogate_loss(old_actions_log_prob, actions_log_prob, advantages)

                bound_loss = torch.clip(dist.loc - 1.0, min=0.0).square().mean() + torch.clip(dist.loc + 1.0, max=0.0).square().mean()

                entropy = dist.entropy().sum(dim=-1)

                loss = (
                    value_loss
                    + actor_loss
                    + self.cfg["algorithm"]["bound_coef"] * bound_loss
                    + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                with torch.no_grad():
                    if self.cfg["algorithm"]["adaptive_lr"]:
                        kl = torch.sum(
                            torch.log(dist.scale / old_dist.scale)
                            + 0.5 * (torch.square(old_dist.scale) + torch.square(dist.loc - old_dist.loc)) / torch.square(dist.scale)
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate
                    else:
                        kl_mean = 0.0

                mean_value_loss += value_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_bound_loss += bound_loss.item()
                mean_entropy += entropy.mean()

            mean_value_loss /= self.cfg["runner"]["mini_epochs"]
            mean_actor_loss /= self.cfg["runner"]["mini_epochs"]
            mean_bound_loss /= self.cfg["runner"]["mini_epochs"]
            mean_entropy /= self.cfg["runner"]["mini_epochs"]

            self.recorder.record_statistics(
                {
                    "value_loss": mean_value_loss,
                    "actor_loss": mean_actor_loss,
                    "bound_loss": mean_bound_loss,
                    "entropy": mean_entropy,
                    "kl_mean": kl_mean,
                    "lr": self.learning_rate
                },
                it,
            )

            if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                self.recorder.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    it + 1,
                )
            print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))

    def play(self):
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        obss = []
        if self.cfg["viewer"]["record_video"]:
            if self.cfg["basic"]["checkpoint"]:
                video_folder = os.path.dirname(self.cfg["basic"]["checkpoint"])
                video_folder = os.path.join(video_folder, "..", "videos")
            else:
                video_folder = "videos"
            name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            video_folder = os.path.join(video_folder, name)
            os.makedirs(video_folder, exist_ok=True)
            record_time = self.cfg["viewer"]["record_interval"]
        while True:
            with torch.no_grad():
                obs, infos = self.env.handle_reset()
                obs = obs.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                dist = self.model.act(obs, privileged_obs)
                act = dist.loc
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                obss.append(obs.cpu().numpy()[0, :])
            if self.cfg["viewer"]["record_video"]:
                record_time -= self.env.dt
                print("Remaining record time: {:.2f}".format(record_time), end="\r")
                if record_time < 0:
                    record_time += self.cfg["viewer"]["record_interval"]
                    self.interrupt = False
                    signal.signal(signal.SIGINT, self.interrupt_handler)
                    with imageio.get_writer(os.path.join(video_folder, "video.mp4"), fps=int(1.0 / self.env.dt)) as self.writer:
                        for frame in self.env.camera_frames:
                            self.writer.append_data(frame)
                    self.writer.close()
                    # save the video
                    print("Video saved to {}".format(os.path.join(video_folder, "video.mp4")))

                    # TODO: make this obs plot robust to different joint configurations
                    import matplotlib.pyplot as plt
                    # plot the obs1
                    obss = np.array(obss)
                    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
                    for i in range(11):
                        axes[i // 6, i % 6].plot(obss[:, i])
                    plt.savefig(os.path.join(video_folder, "obs1.png"))
                    plt.close()

                    # plot the obs2
                    obss = np.array(obss)
                    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
                    for i in range(12):
                        axes[i // 6, i % 6].plot(obss[:, i + 11])
                    plt.savefig(os.path.join(video_folder, "obs2.png"))
                    plt.close()

                    # plot the obs3
                    obss = np.array(obss)
                    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
                    for i in range(12):
                        axes[i // 6, i % 6].plot(obss[:, i + 23])
                    plt.savefig(os.path.join(video_folder, "obs3.png"))
                    plt.close()

                    # plot the obs4
                    obss = np.array(obss)
                    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
                    for i in range(12):
                        axes[i // 6, i % 6].plot(obss[:, i + 35])
                    plt.savefig(os.path.join(video_folder, "obs4.png"))
                    plt.close()
                    
                    if self.interrupt:
                        raise KeyboardInterrupt
                    
                    break
                    signal.signal(signal.SIGINT, signal.default_int_handler)

    def viz_joint(self):
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        obss = []
        dof_targets = []
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        total_timestep = int(record_time / self.env.dt)
        T = total_timestep // self.env.num_actions
        curr_time_step = 0
        while True:
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.loc
                act = torch.zeros_like(act)
                joint_id = min(total_timestep // T, self.env.num_actions - 1)
                if curr_time_step % T == 0:
                    print(f"Joint {joint_id}")
                act[:, joint_id] = 0.5 * 4 * torch.sin(torch.tensor([curr_time_step % T / T * 2 * np.pi]))
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                obss.append(obs.cpu().numpy()[0, :])
                dof_targets.append(self.env.last_dof_targets.cpu().numpy()[0, :])
            if self.cfg["viewer"]["record_video"]:
                record_time -= self.env.dt
                if record_time < 0:
                    record_time += self.cfg["viewer"]["record_interval"]
                    self.interrupt = False
                    signal.signal(signal.SIGINT, self.interrupt_handler)
                    with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as self.writer:
                        for frame in self.env.camera_frames:
                            self.writer.append_data(frame)

                    import matplotlib.pyplot as plt
                    obss = np.array(obss)
                    dof_targets = np.array(dof_targets)
                    # plot the obs1
                    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
                    for i in range(11):
                        axes[i // 6, i % 6].plot(obss[:, i])
                    plt.savefig(os.path.join("videos", name.replace(".mp4", "_obs1.png")))
                    plt.close()

                    # plot the obs2
                    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
                    for i in range(12):
                        axes[i // 6, i % 6].plot(obss[:, i + 11], label="dof_pos")
                        axes[i // 6, i % 6].plot(dof_targets[:, i + 11], label="dof_target")
                    plt.savefig(os.path.join("videos", name.replace(".mp4", "_obs2.png")))
                    plt.close()

                    # plot the obs3
                    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
                    for i in range(12):
                        axes[i // 6, i % 6].plot(obss[:, i + 23])
                    plt.savefig(os.path.join("videos", name.replace(".mp4", "_obs3.png")))
                    plt.close()

                    # plot the obs4
                    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
                    for i in range(12):
                        axes[i // 6, i % 6].plot(obss[:, i + 35])
                    plt.savefig(os.path.join("videos", name.replace(".mp4", "_obs4.png")))
                    plt.close()
                    
                    if self.interrupt:
                        raise KeyboardInterrupt
                    
                    break
                    signal.signal(signal.SIGINT, signal.default_int_handler)

    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True

    def export_policy(self, export_path):
        scripted_policy = torch.jit.script(self.model)
        # print("EXPORT methods:", [m.name for m in scripted_policy._c.get_methods()])
        # Expect to see: ['forward', 'act', 'est_value'] (or whatever you exported)
        print(scripted_policy.code)  # quick eyeball that 'def act' exists
        scripted_policy.save(export_path)
        print(f"Exported policy to {export_path}")
