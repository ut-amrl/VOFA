from envs.t1_falcon_push_obj_cam_sim2sim_one_env import T1VOFATeacher, T1VOFAStudent, GOAL_WIDTH
from isaacgym import gymtorch
from utils.utils import apply_randomization
import torch


class T1FalconPushCamEvalMixin:
    """
    Shared evaluation behavior for teacher and camera/student eval envs.
    """

    def init_falcon(self, cfg):
        super().init_falcon(cfg)
        self.eval_steps = 0
        self.success_step_buf = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.final_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.evaluation_cfg = self.cfg.get("evaluation", {})
        self.reset_counter = torch.zeros(self.num_envs, device=self.device)
        self.has_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_counter += 1
        warm_up_steps = self.evaluation_cfg.get("warm_up_steps", 10)
        early_mask = self.reset_counter < warm_up_steps

        self.terminated_buf |= torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.terminated_buf |= self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.terminated_buf |= self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        
        if early_mask.any():
            env_ids = early_mask.nonzero(as_tuple=False).squeeze(-1)
            self.reset_buf[env_ids] = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)[env_ids]
            self.reset_buf[env_ids] |= self.root_states[env_ids, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
            self.reset_buf[env_ids] |= self.base_pos[env_ids, 2] - self.terrain.terrain_heights(self.base_pos[env_ids]) < self.cfg["rewards"]["terminate_height"]

            local_reset_mask = self.reset_buf[env_ids].bool()
            if local_reset_mask.any():
                reset_ids = env_ids[local_reset_mask]
                self.reset_counter.index_fill_(0, reset_ids, 0)
                self.terminated_buf.index_fill_(0, reset_ids, False)
                self.success_step_buf.index_fill_(0, reset_ids, -1)
                self.final_success_buf.index_fill_(0, reset_ids, False)
                self.has_touched.index_fill_(0, reset_ids, False)

    def evaluate_step(self, save_name):
        obj = self.ball_pos[:, :2]
        goal = self.goal_pos[:, :2]
        dist = torch.norm(obj - goal, dim=-1)

        self.eval_steps += 1
        thr = self.evaluation_cfg.get("success_distance_threshold", GOAL_WIDTH / 2)
        interval = self.evaluation_cfg.get("evaluation_interval", 100)
        total_steps = self.evaluation_cfg.get("total_evaluation_steps", 1500)
        streak_N = self.evaluation_cfg.get("final_success_streak", 100)

        raw_success = dist < thr
        raw_term = self.terminated_buf
        contact_buf = self.ball_contact_forces[:, :2] # filter out z direction force
        contact_norm = torch.norm(contact_buf, dim=-1)
        contact_flag = (contact_norm > 1e-4).float() # good

        self.has_touched |= contact_flag.bool()
        touched = self.has_touched.clone()

        # stop counting an env after it reaches final success
        active = ~self.final_success_buf
        
        # update streak only for active envs
        self.success_step_buf[active & ~raw_success] = -1
        self.success_step_buf[active &  raw_success] += 1

        # latch final success after N successful steps (absorbing)
        self.final_success_buf |= (self.success_step_buf >= streak_N)
        active = ~self.final_success_buf
        final_success = self.final_success_buf

        # buckets (mutually exclusive; cover all envs)
        term = active & raw_term
        suc_in_prog = active & ~raw_term & (self.success_step_buf >= 0)
        alive = active & ~raw_term & (self.success_step_buf == -1)

        term_touch = term & touched
        term_not_touch = term & ~touched
        alive_touch = alive & touched
        alive_not_touch = alive & ~touched

        # global rates (divide by num_envs)
        final_success_rate = final_success.float().mean().item()
        termination_rate = term.float().mean().item()
        term_touched_rate = term_touch.float().mean().item()
        term_not_touched_rate = term_not_touch.float().mean().item()
        suc_in_prog_rate = suc_in_prog.float().mean().item()
        alive_touched_rate = alive_touch.float().mean().item()
        alive_not_touched_rate = alive_not_touch.float().mean().item()

        bucket_sum = (final_success_rate + suc_in_prog_rate + termination_rate +
                    alive_touched_rate + alive_not_touched_rate)
        if self.eval_steps % interval == 0:
            print(f"Step {self.eval_steps} (global / num_envs={self.num_envs}):")
            print(f"  Final success (absorbing)      : {final_success_rate:.2%}")
            print(f"  Success-in-progress            : {suc_in_prog_rate:.2%}")
            print(f"  Alive & Touched                : {alive_touched_rate:.2%}")
            print(f"  Alive & Not touched            : {alive_not_touched_rate:.2%}")
            print(f"---------------------------------------")
            print(f"  Termination (total)            : {termination_rate:.2%}")
            print(f"  Terminated & Touched           : {term_touched_rate:.2%}")
            print(f"  Terminated & Not touched       : {term_not_touched_rate:.2%}")
            print(f"  Bucket sum (~1.0)              : {bucket_sum:.4f}\n")

        if self.eval_steps > total_steps:
            log_path = save_name
            with open(log_path, "w") as f:
                f.write(f"Step {self.eval_steps} (global / num_envs={self.num_envs}):\n")
                f.write(f"  Final success (absorbing)      : {final_success_rate:.2%}\n")
                f.write(f"  Success-in-progress            : {suc_in_prog_rate:.2%}\n")
                f.write(f"  Alive & Touched                : {alive_touched_rate:.2%}\n")
                f.write(f"  Alive & Not touched            : {alive_not_touched_rate:.2%}\n")
                f.write(f"---------------------------------------\n")
                f.write(f"  Termination (total)            : {termination_rate:.2%}\n")
                f.write(f"  Terminated & Touched           : {term_touched_rate:.2%}\n")
                f.write(f"  Terminated & Not touched       : {term_not_touched_rate:.2%}\n")
                f.write(f"  Bucket sum (~1.0)              : {bucket_sum:.4f}\n\n")
            return True
        return False

    def _push_robots(self):
        # randomly kick ball
        if self.evaluation_cfg.get("test_perturb_ball", False):
            perturb_box_env_idx = self.episode_length_buf == 175
            if perturb_box_env_idx.any():
                self.ball_states[:, 7:9] += apply_randomization(self.ball_states[:, 7:9], self.cfg["randomization"].get("kick_ball_lin_vel")) * 5
                print("perturbing ball", perturb_box_env_idx, self.ball_states[:, 7:9])
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))
        super()._push_robots()


class TeacherPolicyEval(T1FalconPushCamEvalMixin, T1VOFATeacher):
    """
    Evaluation env for the teacher/original observation stack.
    """
    pass


class StudentPolicyEval(T1FalconPushCamEvalMixin, T1VOFAStudent):
    """
    Evaluation env for the camera-depth DAgger/student observation stack.
    """
    pass


class T1FalconPushCamDemo(T1VOFATeacher):
    def __init__(self, cfg):
        super().__init__(cfg)
        # This is one demo with 3 waypoints: triangle
        # triangle
        # self.waypoints = torch.tensor([
        #     [4.0, 1.5],
        #     [4.0, -1.5],
        #     [1.5, 0.0]
        # ], device=self.device)

        # word: 8
        self.waypoints = torch.tensor([
            [3.0, 0],
            [5.0, 0],
            [5.0, -4.0],
            [5.0, 0.0],
            [3.0, 0.0],
            [3.0, -2.0],
        ], device=self.device)
        self.curr_waypoint = 0
        self.goal_states[:, 0:2] = self.waypoints[self.curr_waypoint]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))
        self.curr_waypoint += 1
        

    def compute_success(self):
        obj = self.ball_pos[:, :2]
        goal = self.goal_pos[:, :2]
        dist = torch.norm(obj - goal, dim=-1)

        thr = self.evaluation_cfg.get("success_distance_threshold", GOAL_WIDTH / 2)
        streak_N = self.evaluation_cfg.get("final_success_streak", 100)
        thr = 0.5
        streak_N = 50

        raw_success = dist < thr

        # update streak only for active envs
        self.success_step_buf[~raw_success] = -1
        self.success_step_buf[raw_success] += 1

        # latch final success after N successful steps (absorbing)
        return self.success_step_buf >= streak_N
    

    def _check_termination(self):
        """Check if environments need to be reset"""
        super()._check_termination()
        if self.compute_success():
            print("Demo success!")
            if self.curr_waypoint >= self.waypoints.shape[0]:
                self.dt = 1000000
                return 
            self.goal_states[:, 0:2] = self.waypoints[self.curr_waypoint]
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))
            self.curr_waypoint += 1
