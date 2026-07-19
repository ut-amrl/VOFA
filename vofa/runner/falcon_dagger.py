import os
import glob
import torch
from runner.dagger import DAgger, PRIV_OBS_KEY, REW_TERMS_KEY, TIME_OUTS_KEY
from model import *
from envs import *
import yaml
import time
import math
import numpy as np
from utils.recorder import Recorder
from envs import *

class FalconDAgger(DAgger):

    def __init__(self, cfg, test=False):
        super().__init__(cfg, test)
        self.model = eval(self.cfg["algorithm"]["model_class"])(
            self.env.model_num_actions,
            self.env.num_prop,
            self.env.hist_len,
            self.env.num_privileged_obs,
            self.cfg["algorithm"]["init_logstd"]
        ).to(self.device)

        self.trainable_params = (
            list(self.model.actor.parameters()) +
            list(self.model.img_encoder.parameters()) +
            list(self.model.img_seq_proj.parameters()) +
            list(self.model.hist_encoder.parameters())
        )

        self.optimizer = torch.optim.Adam(
            self.trainable_params,
            lr=self.learning_rate
        )

        # Build a dict mapping parameter object → name
        param_to_name = {}
        for name, p in self.model.named_parameters():
            param_to_name[p] = name

        print("---- Trainable Params Breakdown ----")
        total = 0

        for p in self.trainable_params:
            name = param_to_name.get(p, "<NOT FOUND>")
            print(f"{name:50s}  shape={tuple(p.shape)}  numel={p.numel()}")
            total += p.numel()

        print("Total trainable params:", total)
        print("---- End of Trainable Params Breakdown ----")

        self._load_mod()
        self._load_and_freeze_teacher_mod(self.cfg["basic"]["expert_path"])
        self._dagger_depth_obs_chunks = []

    def _load(self):
        pass
    
    def _load_and_freeze_teacher(self, expert_path):
        pass

    def _load_mod(self):
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
            
    def _load_and_freeze_teacher_mod(self, expert_path):
        expert_folder = os.path.dirname(os.path.dirname(expert_path))  # /expert_folder/nn/model_xxx.pth
        self.expert_cfg = yaml.load(open(os.path.join(expert_folder, "config.yaml"), "r"), Loader=yaml.FullLoader)
        self.expert_policy = eval(self.expert_cfg["algorithm"]["model_class"])(
            self.expert_cfg["env"]["model_num_actions"],
            self.expert_cfg["env"]["num_proprioceptive"],
            self.expert_cfg["env"]["hist_len"],
            self.expert_cfg["env"]["num_privileged_obs"]
        ).to(self.device)
        sd = torch.load(expert_path, map_location=self.device, weights_only=True)
        self.expert_policy.load_state_dict(sd["model"])
        self.expert_policy.eval()
        for param in self.expert_policy.parameters():
            param.requires_grad = False

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
        depth_obs = infos["depth"].to(self.device)

        for it in range(self.cfg["basic"]["max_iterations"]):
            # horizon collection
            step_obs_list = []
            step_depth_obs_list = []
            step_expert_actions_list = []
            intervention_rates = []

            for n in range(self.cfg["runner"]["horizon_length"]):
                obs, infos = self.env.handle_reset()
                obs = obs.to(self.device)
                expert_obs = infos.get("expert_obs", obs).to(self.device)
                depth_obs = infos["depth"].to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                # store before stepping
                # self.buffer.update_data("obses", n, obs)
                # self.buffer.update_data("privileged_obses", n, privileged_obs)
                step_obs_list.append(obs.detach().clone())
                step_depth_obs_list.append(depth_obs.detach().clone())

                with torch.no_grad():
                    # no privileged inputs for student
                    student_act = self.model.act(obs, depth_obs, privileged_obs=None, hist_encoding=True).mean
                    # privileged inputs for teacher
                    expert_act = self.expert_policy.act(expert_obs, privileged_obs).mean
                    step_expert_actions_list.append(expert_act.detach().clone())

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
                depth_obs = infos["depth"].to(self.device)

                # episode stats
                # self.buffer.update_data("actions", n, student_act)
                # self.buffer.update_data("expert_actions", n, expert_act)
                # self.buffer.update_data("rewards", n, rew)
                # self.buffer.update_data("dones", n, done)
                # self.buffer.update_data("time_outs", n, infos[TIME_OUTS_KEY].to(self.device))
                ep_info = {"reward": rew}
                ep_info.update(infos[REW_TERMS_KEY])
                self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))

            # dagger dataset aggregation
            if len(step_obs_list) == 0 or len(step_expert_actions_list) == 0 or len(step_depth_obs_list) == 0:
                print(f"Warning: No data collected in iteration {it}, skipping aggregation")
                continue

            try:
                h_obs = torch.stack(step_obs_list, dim=0).reshape(-1, self.env.hist_len, self.env.num_prop)
                h_expert_actions = torch.stack(step_expert_actions_list, dim=0).reshape(-1, self.env.model_num_actions)
                h_depth_obs = torch.stack(step_depth_obs_list, dim=0).reshape(-1, self.env.camera_cfg.buffer_len, self.env.camera_cfg.resized_height, self.env.camera_cfg.resized_width)
            except Exception as e:
                print(f"Error during stacking in iteration {it}: {e}")
                continue

            self._dagger_obs_chunks.append(h_obs)
            self._dagger_act_chunks.append(h_expert_actions)
            self._dagger_depth_obs_chunks.append(h_depth_obs)

            # exact cap by total samples
            while self._get_total_samples() > self.max_dagger_dataset_size:
                self._dagger_obs_chunks.pop(0)
                self._dagger_act_chunks.pop(0)
                self._dagger_depth_obs_chunks.pop(0)

            if len(self._dagger_obs_chunks) == 0:
                print("Warning: No aggregated data available for training")
                continue
            
            # tensors from aggregated dataset
            obses = torch.cat(self._dagger_obs_chunks, dim=0)
            depth_obses = torch.cat(self._dagger_depth_obs_chunks, dim=0)
            expert_actions = torch.cat(self._dagger_act_chunks, dim=0)

            num_samples = obses.shape[0]
            mini_epochs = self.cfg["runner"].get("mini_epochs", 5)
            num_minibatch = self.cfg["runner"].get("num_minibatch", 5)

            # validation split
            val_size = min(1000, num_samples // 10)
            val_loss = torch.tensor(0.0, device=self.device)

            # single permutation for this epoch
            perm = torch.randperm(num_samples, device=self.device)
            val_indices = perm[:val_size]
            train_indices = perm[val_size:]
            if val_size > 0:
                val_obs = obses[val_indices]
                val_depth_obs = depth_obses[val_indices]
                val_expert_actions = expert_actions[val_indices]
                with torch.no_grad():
                    val_act = self.model.act(val_obs, val_depth_obs, privileged_obs=None, hist_encoding=True).mean
                    val_loss = self.behavior_loss_fn(val_act, val_expert_actions)

            print("Epoch ({}/{}):".format(it + 1, self.cfg["basic"]["max_iterations"]))
            
            # BC training on aggregated dataset
            total_behavior_loss = 0
            for e in range(mini_epochs):
                epoch_behavior_loss = 0
                
                shuffled_train_indices = train_indices[torch.randperm(len(train_indices), device=self.device)]
                for mb_idx in torch.chunk(shuffled_train_indices, num_minibatch):
                    mb_obses = obses[mb_idx]
                    mb_depth_obses = depth_obses[mb_idx]
                    mb_expert_actions = expert_actions[mb_idx]

                    act = self.model.act(mb_obses, mb_depth_obses, privileged_obs=None, hist_encoding=True).mean
                    behavior_loss = self.behavior_loss_fn(act, mb_expert_actions) * self.loss_scale

                    self.optimizer.zero_grad()
                    behavior_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, 100.0)
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