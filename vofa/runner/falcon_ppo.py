import os
import glob
import time
import imageio
import signal
import numpy as np
import torch
from runner.ppo import PPO
from model import *
from envs import *

# The only difference is that 
class FalconPPO(PPO):
    def __init__(self, cfg, test=False):
        super().__init__(cfg, test)
        self.model = eval(self.cfg["algorithm"]["model_class"])(
            self.env.model_num_actions,
            self.env.num_prop,
            self.env.hist_len,
            self.env.num_privileged_obs,
            self.cfg["algorithm"]["init_logstd"]
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._load_mod()
        self.buffer.add_buffer("actions", (self.env.model_num_actions,))

    def _load(self):
        pass

    def _load_mod(self):
        if not self.cfg["basic"]["checkpoint"]:
            return
        if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
            self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        # do not load logstd
        if "logstd" in model_dict["model"]:
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
            
    def export_policy(self, export_path):
        scripted_policy = torch.jit.script(self.model)
        # print("EXPORT methods:", [m.name for m in scripted_policy._c.get_methods()])
        # Expect to see: ['forward', 'act', 'est_value'] (or whatever you exported)
        print(scripted_policy.code)  # quick eyeball that 'def act' exists
        scripted_policy.save(export_path)
        print(f"Exported policy to {export_path}")
        
    def evaluate(self):
        print("evaluating")
        obs, info = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = info["privileged_obs"].to(self.device)
        save_name = self.cfg["evaluation"].get("eval_save_name", "tmp.txt")
        
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
                ended = self.env.evaluate_step(save_name)
                if ended:
                    break
                
class FalconCamEvaluate(FalconPPO):
    def evaluate(self):
        obs, info = self.env.reset()
        obs = obs.to(self.device)
        depth = info["depth"].to(self.device)
        privileged_obs = info["privileged_obs"].to(self.device)
        
        save_name = self.cfg["evaluation"].get("eval_save_name", "tmp.txt")
        while True:
            with torch.no_grad():
                obs, infos = self.env.handle_reset()
                obs = obs.to(self.device)
                depth = infos["depth"].to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                dist = self.model.act(obs, depth, privileged_obs)
                act = dist.loc
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                ended = self.env.evaluate_step(save_name)
                if ended:
                    break

    def play(self):
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        depth = infos["depth"].to(self.device)
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
        # obs, infos = self.env.reset()
        print("Start Playing")
        while True:
            with torch.no_grad():
                obs, infos = self.env.handle_reset()
                obs = obs.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                depth = infos["depth"].to(self.device)
                dist = self.model.act(obs, depth, privileged_obs)
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