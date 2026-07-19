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
import torch.nn as nn
import torch.nn.functional as F

from model import *
from utils.buffer import ExperienceBuffer
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder
from envs import *
from runner.ppo import PPO


PRIV_OBS_KEY = "privileged_obs"
TIME_OUTS_KEY = "time_outs"
REW_TERMS_KEY = "rew_terms"


class DAgger(PPO):
    """DAgger runner with frozen teacher, per-env beta mixing, and aggregated BC ."""

    def __init__(self, cfg, test=False):
        self.test = test
        self.cfg = cfg

        # prepare the environment
        self._set_seed()
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)

        # devices / lr
        self.device = self.cfg["basic"]["rl_device"]
        self.learning_rate = self.cfg["algorithm"]["learning_rate"]

        # student
        self.model = eval(self.cfg["algorithm"]["model_class"])(
            self.env.num_actions,
            self.env.num_prop,
            self.env.hist_len,
            self.env.num_privileged_obs
        ).to(self.device)

        # # BC optimizer
        # self.optimizer = torch.optim.Adam(
        #     list(self.model.actor.parameters()) + [self.model.logstd] + list(self.model.hist_encoder.parameters()),
        #     lr=self.learning_rate
        # )
        self._load()

        # teacher
        if self.cfg["basic"]["expert_path"] is None:
            raise ValueError("DAgger requires expert_path in config. Set basic.expert_path to a valid checkpoint.")
        self._load_and_freeze_teacher(self.cfg["basic"]["expert_path"])

        # rollout buffer
        self._init_buffer()

        # DAgger dataset aggregation
        self._dagger_obs_chunks = []
        self._dagger_act_chunks = []
        self.max_dagger_dataset_size = self.cfg["algorithm"].get("max_dagger_dataset_size", 100000)

        # loss
        loss_type = self.cfg["algorithm"].get("loss_type", "huber")
        self.loss_scale = self.cfg["algorithm"].get("loss_scale", 10.0)
        huber_delta = self.cfg["algorithm"].get("huber_delta", 0.1)
        if loss_type == "mse":
            self.behavior_loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.behavior_loss_fn = lambda x, y: nn.functional.huber_loss(x, y, delta=huber_delta)
        elif loss_type == "l1":
            self.behavior_loss_fn = nn.functional.l1_loss
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        # beta schedule
        self.beta_decay_iterations = self.cfg["algorithm"].get("beta_decay_iterations", 2000)
        self.beta_start = self.cfg["algorithm"].get("beta_start", 1.0)
        self.beta_end = self.cfg["algorithm"].get("beta_end", 0.0)

    def train(self):
        self.recorder = Recorder(self.cfg)
        obs, infos = self.env.reset()

        # initialize at random episode length
        if self.cfg["runner"]["random_init_episode_length"]:
            max_episode_length = np.ceil(self.cfg["rewards"]["episode_length_s"] / self.env.dt)
            self.env.episode_length_buf = torch.randint(0, int(max_episode_length), (self.env.num_envs,), device=self.device)

        obs = obs.to(self.device)
        privileged_obs = infos[PRIV_OBS_KEY].to(self.device)
        expert_obs = infos.get("expert_obs", obs).to(self.device)

        for it in range(self.cfg["basic"]["max_iterations"]):
            # horizon collection
            step_obs_list = []
            step_expert_actions_list = []
            intervention_rates = []

            for n in range(self.cfg["runner"]["horizon_length"]):
                # store before stepping
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                step_obs_list.append(obs.detach().cpu())

                with torch.no_grad():
                    # no privileged inputs for student
                    student_act = self.model.act(obs, privileged_obs=None, hist_encoding=True).mean
                    # privileged inputs for teacher
                    expert_act = self.expert_policy.act(expert_obs, privileged_obs).mean
                    step_expert_actions_list.append(expert_act.detach().cpu())

                # beta mixing
                beta = self._beta(it)
                use_expert = (torch.rand(self.env.num_envs, device=self.device) < beta)
                exec_act = torch.where(use_expert.unsqueeze(-1), expert_act, student_act)
                intervention_rates.append(use_expert.float().mean().item())

                # env step
                obs, rew, done, infos = self.env.step(exec_act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos[PRIV_OBS_KEY].to(self.device)
                expert_obs = infos.get("expert_obs", obs).to(self.device)

                # episode stats
                self.buffer.update_data("actions", n, student_act)
                self.buffer.update_data("expert_actions", n, expert_act)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos[TIME_OUTS_KEY].to(self.device))
                ep_info = {"reward": rew}
                ep_info.update(infos[REW_TERMS_KEY])
                self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))

            # dagger dataset aggregation
            if len(step_obs_list) == 0 or len(step_expert_actions_list) == 0:
                print(f"Warning: No data collected in iteration {it}, skipping aggregation")
                continue

            try:
                h_obs = torch.stack(step_obs_list, dim=0).reshape(-1, self.env.hist_len, self.env.num_prop)
                h_expert_actions = torch.stack(step_expert_actions_list, dim=0).reshape(-1, self.env.model_num_actions)
            except Exception as e:
                print(f"Error during stacking in iteration {it}: {e}")
                continue

            self._dagger_obs_chunks.append(h_obs)
            self._dagger_act_chunks.append(h_expert_actions)

            # exact cap by total samples
            while self._get_total_samples() > self.max_dagger_dataset_size:
                self._dagger_obs_chunks.pop(0)
                self._dagger_act_chunks.pop(0)

            if len(self._dagger_obs_chunks) == 0:
                print("Warning: No aggregated data available for training")
                continue

            # tensors from aggregated dataset
            obses = torch.cat(self._dagger_obs_chunks, dim=0).to(self.device)
            expert_actions = torch.cat(self._dagger_act_chunks, dim=0).to(self.device)

            num_samples = obses.shape[0]
            num_minibatch = self.cfg["runner"].get("num_minibatch", 5)
            mini_epochs = self.cfg["runner"].get("mini_epochs", 5)

            # validation split
            val_size = min(1000, num_samples // 10)
            val_loss = torch.tensor(0.0, device=self.device)
            if val_size > 0:
                val_indices = torch.randperm(num_samples, device=self.device)[:val_size]
                val_obs = obses[val_indices]
                val_expert_actions = expert_actions[val_indices]
                with torch.no_grad():
                    val_act = self.model.act(val_obs, privileged_obs=None, hist_encoding=True).mean
                    val_loss = self.behavior_loss_fn(val_act, val_expert_actions)

            print("Epoch ({}/{}):".format(it + 1, self.cfg["basic"]["max_iterations"]))

            # BC training on aggregated dataset
            total_behavior_loss = 0
            for e in range(mini_epochs):
                indices = torch.randperm(num_samples, device=self.device)
                epoch_behavior_loss = 0

                for mb_idx in torch.chunk(indices, num_minibatch):
                    mb_obses = obses[mb_idx]
                    mb_expert_actions = expert_actions[mb_idx]

                    act = self.model.act(mb_obses, privileged_obs=None, hist_encoding=True).mean
                    behavior_loss = self.behavior_loss_fn(act, mb_expert_actions) * self.loss_scale

                    self.optimizer.zero_grad()
                    behavior_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.actor.parameters()) + list(self.model.hist_encoder.parameters()) + [self.model.logstd], 100.0
                    )
                    self.optimizer.step()

                    epoch_behavior_loss += behavior_loss.item()

                mean_epoch_loss = epoch_behavior_loss / num_minibatch
                total_behavior_loss += mean_epoch_loss
                print("Mini-epoch {}: {:.6f}".format(e, mean_epoch_loss))

            mean_behavior_loss = total_behavior_loss / mini_epochs
            avg_intervention_rate = np.mean(intervention_rates) if intervention_rates else 0.0

            log_dict = {
                "mean_behavior_loss": mean_behavior_loss,
                "val_behavior_loss": val_loss.item(),
                "beta_scheduled": beta,
                "intervention_rate": avg_intervention_rate,
                "dataset_size": num_samples
            }
            self.recorder.record_statistics(log_dict, it)

            if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                self.recorder.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    it + 1,
                )
                print("Dataset size: {}, Beta: {:.3f}, Intervention: {:.3f}".format(
                    num_samples, beta, avg_intervention_rate))

    #helpers
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
            self.cfg["basic"]["checkpoint"] = sorted(
                glob.glob(os.path.join("logs", "**/*.pth"), recursive=True),
                key=os.path.getmtime
            )[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)
        try:
            self.env.curriculum_prob = model_dict["curriculum"]
        except Exception as e:
            print(f"Failed to load curriculum: {e}")
        # Guard optimizer state loading for BC training
        load_optimizer_for_bc = self.cfg["algorithm"].get("load_optimizer_for_bc", False)
        if load_optimizer_for_bc:
            try:
                self.optimizer.load_state_dict(model_dict["optimizer"])
            except Exception as e:
                print(f"Failed to load optimizer: {e}")
        else:
            print("Skipping optimizer state loading for BC training (fresh Adam optimizer)")

    def _load_and_freeze_teacher(self, expert_path):
        expert_folder = os.path.dirname(os.path.dirname(expert_path))  # /expert_folder/nn/model_xxx.pth
        self.expert_cfg = yaml.load(open(os.path.join(expert_folder, "config.yaml"), "r"), Loader=yaml.FullLoader)
        self.expert_policy = eval(self.expert_cfg["algorithm"]["model_class"])(
            self.env.num_actions,
            self.env.num_prop,
            self.env.hist_len,
            self.env.num_privileged_obs
        ).to(self.device)
        sd = torch.load(expert_path, map_location=self.device, weights_only=True)
        self.expert_policy.load_state_dict(sd["model"])
        self.expert_policy.eval()
        for param in self.expert_policy.parameters():
            param.requires_grad = False

    def _init_buffer(self):
        H = self.cfg["runner"]["horizon_length"]
        self.buffer = ExperienceBuffer(H, self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.model_num_actions,))
        self.buffer.add_buffer("expert_actions", (self.env.model_num_actions,))
        self.buffer.add_buffer("obses", (self.env.hist_len, self.env.num_prop))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

    def _get_total_samples(self):
        return sum(chunk.shape[0] for chunk in self._dagger_obs_chunks)

    def _beta(self, it):
        # linear decay from beta_start to beta_end over beta_decay_iterations
        progress = min(it / max(1, self.beta_decay_iterations), 1.0)
        return max(0.0, self.beta_start + (self.beta_end - self.beta_start) * progress)