import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import wandb
import yaml


class Recorder:

    def __init__(self, cfg):
        self.cfg = cfg
        name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.dir = os.path.join("logs", cfg["basic"]["task"], name)
        os.makedirs(self.dir)
        self.model_dir = os.path.join(self.dir, "nn")
        os.mkdir(self.model_dir)
        self.writer = SummaryWriter(os.path.join(self.dir, "summaries"))
        if self.cfg["runner"]["use_wandb"]:
            wandb.init(
                project="T1",
                dir=self.dir,
                name=os.path.join(cfg["basic"]["task"], name),
                notes=self.cfg["basic"]["description"],
                config=self.cfg,
            )

        self.episode_statistics = {}
        self.last_episode = {}
        self.last_episode["steps"] = []
        self.episode_steps = None

        with open(os.path.join(self.dir, "config.yaml"), "w") as file:
            yaml.dump(self.cfg, file)

    def record_episode_statistics(self, done, ep_info, it, write_record=False):
        # --- NEW: per-step reward logging (minimal & non-invasive) ---
        if write_record:
            env_dt = self.cfg["sim"]["dt"] * self.cfg["control"]["decimation"]
            for key, value in ep_info.items():
                r_mean = value.mean().item() / env_dt
                self._record_scalar(f"reward_step/{key}", r_mean, it)

        # ------------------------------------------------------------
        if self.episode_steps is None:
            self.episode_steps = torch.zeros_like(done, dtype=int)
        else:
            self.episode_steps += 1
        for val in self.episode_steps[done]:
            self.last_episode["steps"].append(val.item())
        self.episode_steps[done] = 0

        for key, value in ep_info.items():
            if self.episode_statistics.get(key) is None:
                self.episode_statistics[key] = torch.zeros_like(value)
            self.episode_statistics[key] += value
            if self.last_episode.get(key) is None:
                self.last_episode[key] = []
            for done_value in self.episode_statistics[key][done]:
                self.last_episode[key].append(done_value.item())
            self.episode_statistics[key][done] = 0

        if write_record:
            for key in self.last_episode.keys():
                path = ("" if key == "steps" or key == "reward" else "episode/") + key
                if len(self.last_episode[key]) == 0:
                    continue
                value = self._mean(self.last_episode[key])
                self._record_scalar(path, value, it)
                self.last_episode[key].clear()

    def record_statistics(self, statistics, it):
        for key, value in statistics.items():
            self._record_scalar(key, float(value), it)

    def save(self, model_dict, it):
        path = os.path.join(self.model_dir, "model_{}.pth".format(it))
        print("Saving model to {}".format(path))
        torch.save(model_dict, path)

    def _mean(self, data):
        if len(data) == 0:
            return 0.0
        else:
            return sum(data) / len(data)

    def _record_scalar(self, key, value, it):
        self.writer.add_scalar(key, value, it)
        if self.cfg["runner"]["use_wandb"]:
            wandb.log({key: value}, step=it)
