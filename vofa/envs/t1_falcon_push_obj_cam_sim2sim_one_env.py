import os
import cv2
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
    quat_mul,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis
)

assert gymtorch

import torch
from torch import Tensor
import random

from collections import deque
import numpy as np
from .base_task import BaseTask
from omegaconf import OmegaConf
from hydra.utils import instantiate
from utils.utils import apply_randomization, apply_perception_noise, calc_heading
import torch.nn.functional as F

# FIELD_LENGTH = 8.0  # [m]
FIELD_LENGTH = 6.0
FIELD_WIDTH = 10.0  # [m]
ROBOT_INIT_LENGTH = 1.0
ROBOT_INIT_WIDTH = 1.0
GOAL_WIDTH = 0.6  # [m]

import xml.etree.ElementTree as ET

def get_box_size_from_urdf(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Find the first collision box (adjust if you have multiple)
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            geom = collision.find("geometry")
            if geom is None:
                continue
            box = geom.find("box")
            if box is not None:
                size_str = box.attrib["size"]  # e.g. "0.4 0.3 0.2"
                L, W, H = [float(x) for x in size_str.split()]
                return L, W, H

    raise RuntimeError(f"No <collision><geometry><box> found in {urdf_path}")

import math

@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor, w_last: bool=True) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions of shape (..., 4).
        w_last: If True, the real part of the quaternion is last.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if w_last:
        i, j, k, r = torch.unbind(quaternions, -1)
    else:
        r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

@torch.jit.script
def calc_heading_quat(q: Tensor, w_last: bool=True) -> Tensor:
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q, w_last)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q

def quat_mul_custom(q1: gymapi.Quat, q2: gymapi.Quat) -> gymapi.Quat:
    return gymapi.Quat(
        q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
        q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
        q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
        q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z,
    )
    
def quat_from_euler_rpy(roll, pitch, yaw) -> gymapi.Quat:
    # ZYX (yaw-pitch-roll)
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return gymapi.Quat(
        sr*cp*cy - cr*sp*sy,  # x
        cr*sp*cy + sr*cp*sy,  # y
        cr*cp*sy - sr*sp*cy,  # z
        cr*cp*cy + sr*sp*sy,  # w
    )

class TaskSampler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = cfg["env"]["num_envs"]
        self.sampling_strategy = cfg["task_sampler"]["sampling_strategy"]
        self.device = cfg["basic"]["sim_device"]
        if self.cfg["env"].get("H_range"):
            self.box_height = self.cfg["env"]["H_range"][1]
        else:
            obj_name = cfg["env"]["obj_name"]
            obj_asset_root = "resources/assets"
            size = get_box_size_from_urdf(obj_asset_root + "/" + obj_name)
            self.box_height = size[2]

        assert self.sampling_strategy in [
            "uniform_ball_pos",  # sample ball pos uniformly in predefined ball pos region
            "around_robot",  # sample ball pos in some region around robot
            "evaluation",  # set robot around ball
        ]

        if self.sampling_strategy == "uniform_ball_pos":
            raise NotImplementedError("uniform_ball_pos not implemented yet")
            # sample the center of the ball pos region
            self.ball_pos_center = torch.zeros(self.num_envs, 3, device=self.device)
            # spans front half field
            self.ball_pos_center[:, 0] = torch.rand(self.num_envs, device=self.device) * FIELD_LENGTH * 0.45
            self.ball_pos_center[:, 1] = (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0) * FIELD_WIDTH * 0.45
        elif self.sampling_strategy == "evaluation":
            raise NotImplementedError("evaluation not implemented yet")
            # 2D meshgrid with an interval of 1m
            ball_pos_center_x = torch.tensor(self.cfg["task_sampler"]["evaluation_ball_pos_center_x"], device=self.device)
            ball_pos_center_y = torch.tensor(self.cfg["task_sampler"]["evaluation_ball_pos_center_y"], device=self.device)
            self.ball_pos_center_x, self.ball_pos_center_y = torch.meshgrid(ball_pos_center_x, ball_pos_center_y, indexing="ij")
            self.task_id2ball_pos = torch.stack([
                self.ball_pos_center_x.flatten(),
                self.ball_pos_center_y.flatten(),
                torch.zeros_like(self.ball_pos_center_x.flatten())
            ], dim=-1)
            # repeat by num_repeats
            self.num_tasks = self.task_id2ball_pos.shape[0]
            self.ball_pos_center = self.task_id2ball_pos.repeat(self.cfg["task_sampler"]["num_repeats"], 1)
            self.env_id2task_idx = torch.arange(self.num_tasks, device=self.device).repeat(self.cfg["task_sampler"]["num_repeats"])
            assert self.ball_pos_center.shape[0] == self.num_envs, f"{self.ball_pos_center.shape[0]} != {self.num_envs}"
            
    def sample_task(self, env_ids, root_state: Tensor, env_origins: Tensor):
        """ Takes in env_ids (B,) and root_states (B, 13) and returns the new
        ball_states (B, 13) and root_states (B, 13)
        """
        if self.sampling_strategy == "uniform_ball_pos":
            raise NotImplementedError("uniform_ball_pos not implemented yet")
            ball_pos = self.ball_pos_center[env_ids].clone()
            ball_pos[:, 2] = 0.52
            # circle around the center
            ball_range = apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.cfg["task_sampler"].get("init_ball_r"),
            )
            ball_angle = apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.cfg["task_sampler"].get("init_ball_angle"),
            )
            ball_pos[:, 0] += ball_range * torch.cos(ball_angle)
            ball_pos[:, 1] += ball_range * torch.sin(ball_angle)
            ball_pos[:, :2] += env_origins[:, :2]

        elif self.sampling_strategy == "around_robot":
            ball_pos = root_state[:, :3].clone()
            _, _, robot_yaw = get_euler_xyz(root_state[:, 3:7])
            ball_pos[:, 2] = self.box_height / 2 + 0.02 # changed this to match the real box height
            # circle around the robot
            ball_range = apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.cfg["task_sampler"].get("init_ball_r"),
            )
            ball_angle = apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.cfg["task_sampler"].get("init_ball_angle"),
            )

            if self.cfg["task_sampler"].get("fix_ball_angle", False):
                ball_angle = ball_angle + robot_yaw
            ball_pos[:, 0] += ball_range * torch.cos(ball_angle)
            ball_pos[:, 1] += ball_range * torch.sin(ball_angle)

        elif self.sampling_strategy == "evaluation":
            raise NotImplementedError("evaluation not implemented yet")
            ball_pos = self.ball_pos_center[env_ids].clone()
            ball_pos[:, :2] += env_origins[:, :2]
            ball_pos[:, 2] = 0.2
            # circle around the center
            robot_range = apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.cfg["task_sampler"].get("init_ball_r"),
            )
            robot_angle = apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.cfg["task_sampler"].get("init_ball_angle"),
            )
            root_state[:, :2] = ball_pos[:, :2].clone()
            root_state[:, 0] += robot_range * torch.cos(robot_angle)
            root_state[:, 1] += robot_range * torch.sin(robot_angle)
            # randomize yaw
            quat_offset = quat_from_euler_xyz(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                torch.rand(len(env_ids), device=self.device) * (2 * torch.pi)
            )
            root_state[:, 3:7] = quat_mul(root_state[:, 3:7], quat_offset)

        return ball_pos, root_state


class T1VOFATeacher(BaseTask):
    """
    Environment with observation stack as history of previous observations. Used to RMA/ROA training 
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_falcon(cfg)
        # hacky way to deal with initial reset issue: initially the robot will crash
        obs, infos = self.reset()
        self.step(self.last_falcon_actions)
        obs, infos = self.reset()

    def init_falcon(self, cfg):
        self.hydra_config = OmegaConf.create(cfg)
        self.falcon_history_length = self.hydra_config.falcon_obs.history_length
        self.falcon_actor_obs_num = self.hydra_config.falcon_obs.actor_obs_num
        self.falcon_obs_scales = self.hydra_config.falcon_obs.obs_scales
        self.falcon_obs_buf_tensor = torch.zeros(self.num_envs, self.falcon_actor_obs_num * self.falcon_history_length, device=self.device)
        self.falcon_ref_upper_dof_pos = torch.zeros(self.num_envs, 10, device=self.device)
        self.goal_locations = torch.zeros(self.num_envs, 3, device=self.device)
        self.falcon_upper_dof_indices = torch.arange(0, 10, device=self.device)  # first 10 dofs are upper body dofs
        self.falcon_rounding_error = torch.tensor(1e3, device=self.device)
        self.falcon_original_obj2goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.falcon_last_obj2goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.falcon_original_robot2obj_dist = torch.zeros(self.num_envs, device=self.device)
        self.falcon_prev_position_fov = torch.zeros(self.num_envs, 2, device=self.device)
        self.camera_global_counter = 0
        self.falcon_prev_ee_reward = torch.zeros(self.num_envs, device=self.device)
        self.model_num_actions = self.cfg["env"]["model_num_actions"] 
        self.success_step_buf = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long) # for early termination
        self.evaluation_cfg = self.cfg.get("evaluation", {}) # for early termination
        
        self.desired_upper_residual = torch.zeros_like(self.falcon_ref_upper_dof_pos)
        for joint_name, target_pos in self.cfg["rewards"]["target_dof_pos"].items():
            joint_idx = self.dof_names.index(joint_name)
            default_joint_pos = self.default_dof_pos[:, joint_idx]
            desired_joint_pos = target_pos - default_joint_pos
            self.desired_upper_residual[:, joint_idx] = desired_joint_pos # hacky: assume the upper starts from 0

        self.last_falcon_actions = torch.zeros(self.num_envs, self.model_num_actions, device=self.device)
        self.falcon_policy = instantiate(self.hydra_config.falcon_model, device=self.device)
        self.falcon_policy.eval()

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        
        # dgp init
        self.obj_mass = torch.zeros(self.num_envs, device=self.device)
        self.obj_dims = torch.zeros(self.num_envs, 3, device=self.device)  # length, width, height
        self.cam_marker_handles = []

        # load robot asset
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        # load ball asset
        obj_asset_root = "resources/assets"
        ball_asset_options = gymapi.AssetOptions()
        ball_asset_options.density = 1
        # large_obj
        # ball_asset = self.gym.load_asset(self.sim, obj_asset_root, obj_name, ball_asset_options) # used to be large_obj.urdf
        if self.cfg["env"].get("H_range"):
            L = np.random.uniform(self.cfg["env"]["L_range"][0], self.cfg["env"]["L_range"][1])
            W = np.random.uniform(self.cfg["env"]["W_range"][0], self.cfg["env"]["W_range"][1])
            H = np.random.uniform(self.cfg["env"]["H_range"][0], self.cfg["env"]["H_range"][1])
        else:
            obj_name = self.cfg["env"]["obj_name"] # "rectangle.urdf" or "box_real.urdf"
            L, W, H = get_box_size_from_urdf(obj_asset_root + "/" + obj_name)
            print("L, W, H:", L, W, H)
        ball_asset = self.gym.create_box(self.sim, L, W, H, ball_asset_options)

        # load goal asset
        goal_asset_options = gymapi.AssetOptions()
        goal_asset_options.fix_base_link = True
        goal_asset = self.gym.load_asset(self.sim, obj_asset_root, "goal.urdf", goal_asset_options)

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        # Validate head joints exist and store indices
        head_joints = ["AAHead_yaw", "Head_pitch"]
        for joint_name in head_joints:
            if joint_name not in self.dof_names:
                raise ValueError(f"Expected head joint '{joint_name}' not found in DOF names: {self.dof_names}")

        # Store head joint indices for proper indexing
        self.head_yaw_idx = self.dof_names.index("AAHead_yaw")
        self.head_pitch_idx = self.dof_names.index("Head_pitch")

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names = body_names
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])
        self.ball_indice = self.base_indice + self.num_bodies  # ball is added as a rigid body after the robot
        self.goal_indice = self.ball_indice + 1  # goal is added as a rigid body after the ball

        # Find head link index
        # Ensure 'head_name' is defined in asset_cfg in your YAML (e.g., "H2")
        self.head_link_name = asset_cfg["head_name"]
        self.head_link_indice = self.gym.find_asset_rigid_body_index(robot_asset, self.head_link_name)
        if self.head_link_indice == -1:
            raise ValueError(f"Head link named '{self.head_link_name}' as specified in asset_cfg['head_name'] not found in URDF. Please check your config and the URDF file.")

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        ball_init_state_list = (
            self.cfg["init_state"]["ball_pos"] + [0, 0, 0, 1] + self.cfg["init_state"]["ball_lin_vel"] + [0, 0, 0]
        )
        self.ball_init_state = to_torch(ball_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.ball_handles = []
        self.goal_handles = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        
        # handle camera config
        camera_cfg = self.cfg.get("camera", None)
        if camera_cfg is not None:
            self.camera_cfg = OmegaConf.create(self.cfg["camera"])
            self.use_camera = self.camera_cfg.get("use_camera", False)
            self.show_camera = self.camera_cfg.get("show_camera", False)
            # set up depth camera buffer
            self.depth_buffer = torch.zeros(self.num_envs,
                                            self.camera_cfg.buffer_len,
                                            self.camera_cfg.resized_width,
                                            self.camera_cfg.resized_height, device=self.device)
        else:
            self.use_camera = False
            self.show_camera = False
        self.cam_handles = []
        self.cam_tensors = []

        # compute max aggregate size
        robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        robot_num_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        ball_num_bodies  = self.gym.get_asset_rigid_body_count(ball_asset)
        ball_num_shapes  = self.gym.get_asset_rigid_shape_count(ball_asset)
        goal_num_bodies  = self.gym.get_asset_rigid_body_count(goal_asset)
        goal_num_shapes  = self.gym.get_asset_rigid_shape_count(goal_asset)

        max_agg_bodies = robot_num_bodies + ball_num_bodies + goal_num_bodies
        max_agg_shapes = robot_num_shapes + ball_num_shapes + goal_num_shapes
        if self.use_camera and self.show_camera:
            print("Adding camera marker to aggregate")
            max_agg_bodies += 1
            max_agg_shapes += 1
            
        for i in range(self.num_envs):
            print("Creating env ", i)
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # build env
            self.gym.begin_aggregate(env_handle, max_agg_bodies, max_agg_shapes, True)

            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)
            ball_start_pose = gymapi.Transform()
            ball_start_pose.p = gymapi.Vec3(pos[0] + 0.2, pos[1] + 0.2, 0.2)  # place the ball slightly above the ground and infront of the robot
            goal_start_pose = gymapi.Transform()
            if self.cfg["task_sampler"].get("goal_pos", None):
                goal_pos_x = float(np.random.uniform(self.cfg["task_sampler"]["goal_pos"][0], self.cfg["task_sampler"]["goal_pos"][1]))
            else:
                goal_pos_x = 4
            goal_start_pose.p = gymapi.Vec3(pos[0] + goal_pos_x, pos[1], 0.0)  # place the goal at the edge of the field

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)

            # TODO: randomize ball rigid body properties
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_start_pose, "ball", i, 0, 0)
            goal_handle = self.gym.create_actor(env_handle, goal_asset, goal_start_pose, "goal", self.num_envs + 10, 0, 0) # special group for goal to avoid collisions
            
            # (Optional) randomize ALL ball bodies/shapes
            ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            assert len(ball_body_props) == 1, "Expected only one rigid body for the ball."
            if self.cfg["randomization"].get("box_mass") is not None:
                box_mass_range = self.cfg["randomization"]["box_mass"]
            else:
                box_mass_range = (3.0, 5.0)  # align with visualmimic
            if self.cfg["randomization"].get("box_friction") is not None:
                box_friction_range = self.cfg["randomization"]["box_friction"]
            else:
                box_friction_range = (0.5, 2.0)  # align with visualmimic
            for j in range(len(ball_body_props)):
                ball_body_props[j].mass = float(np.random.uniform(box_mass_range[0], box_mass_range[1])) # align with visualmimic
                self.obj_mass[i] = ball_body_props[j].mass
            self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_body_props, True)

            ball_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, ball_handle)
            for j in range(len(ball_shape_props)):
                ball_shape_props[j].friction = float(np.random.uniform(box_friction_range[0], box_friction_range[1])) # align with visualmimic
            self.gym.set_actor_rigid_shape_properties(env_handle, ball_handle, ball_shape_props)

            self.obj_dims[i, 0] = L
            self.obj_dims[i, 1] = W
            self.obj_dims[i, 2] = H    

            if self.num_envs == 1:
                # Set visual color for the goal
                goal_color = gymapi.Vec3(212/255.0, 106/255.0, 106/255.0)

                num_bodies = self.gym.get_actor_rigid_body_count(env_handle, goal_handle)
                for rb in range(num_bodies):
                    self.gym.set_rigid_body_color(
                        env_handle,
                        goal_handle,
                        rb,
                        gymapi.MESH_VISUAL,
                        goal_color
                    )

                # Set visual color for the box (ball)
                kraft_color = gymapi.Vec3(196/255.0, 154/255.0, 108/255.0)

                num_bodies = self.gym.get_actor_rigid_body_count(env_handle, ball_handle)
                for rb in range(num_bodies):
                    self.gym.set_rigid_body_color(
                        env_handle,
                        ball_handle,
                        rb,
                        gymapi.MESH_VISUAL,
                        kraft_color
                    )

                # limb_color = gymapi.Vec3(69/255.0, 90/255.0, 115/255.0)
                limb_color = gymapi.Vec3(58/255.0, 74/255.0, 99/255.0)
                torso_color = gymapi.Vec3(230/255.0, 232/255.0, 235/255.0)

                # color every rigid body (quick check)
                num_bodies = self.gym.get_actor_rigid_body_count(env_handle, actor_handle)
                for rb in range(num_bodies):
                    if rb in [0, 11]:
                        self.gym.set_rigid_body_color(env_handle, actor_handle, rb, gymapi.MESH_VISUAL, torso_color)
                    else:
                        self.gym.set_rigid_body_color(env_handle, actor_handle, rb, gymapi.MESH_VISUAL, limb_color)

            # attach camera sensor
            self.attach_camera(env_handle, actor_handle)
            
            # end build env
            self.gym.end_aggregate(env_handle)
            
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.ball_handles.append(ball_handle)
            self.goal_handles.append(goal_handle)

    def _create_cam_marker(self, env_handle, color=(1.0, 0.0, 0.0)):
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True

        # Capsule is typically aligned with local +Y in many IsaacGym primitives.
        # We'll handle axis alignment in _update_cam_marker by applying a fixed rotation.
        radius = 0.01
        half_length = 0.15
        marker_asset = self.gym.create_capsule(self.sim, radius, half_length, opts)

        marker_pose = gymapi.Transform()
        marker_pose.p = gymapi.Vec3(0, 0, 0)

        collision_group = 32
        collision_filter = 0  # collide with nothing
        marker_actor = self.gym.create_actor(env_handle, marker_asset, marker_pose, "cam_marker", collision_group, collision_filter, 0)

        # extra safety: kill collision filter on shape props
        shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, marker_actor)
        for p in shape_props:
            p.filter = 0
            p.contact_offset = 0.0
            p.rest_offset = 0.0
            p.friction = 0.0
            p.restitution = 0.0
        self.gym.set_actor_rigid_shape_properties(env_handle, marker_actor, shape_props)

        self.gym.set_rigid_body_color(env_handle, marker_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
        return marker_actor

    def _update_cam_marker(self, env_i, env_handle, cam_handle):
        cam_tf = self.gym.get_camera_transform(self.sim, env_handle, cam_handle)
        self.cam_marker_states[env_i, 0:3] = torch.tensor(
            [cam_tf.p.x, cam_tf.p.y, cam_tf.p.z],
            device=self.cam_marker_states.device,
            dtype=self.cam_marker_states.dtype,
        )
        self.cam_marker_states[env_i, 3:7] = torch.tensor(
            [cam_tf.r.x, cam_tf.r.y, cam_tf.r.z, cam_tf.r.w],
            device=self.cam_marker_states.device,
            dtype=self.cam_marker_states.dtype,
        )
        self.cam_marker_states[env_i, 7:13] = 0.0  # zero velocities

        # push this one actor to sim TODO hardcoded
        idx32 = torch.tensor([3], device=self.cam_marker_states.device, dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states_all),   # raw (total_actors, 13) tensor
            gymtorch.unwrap_tensor(idx32),
            1,
        )

    def attach_camera(self, env_handle, actor_handle):
        if not self.use_camera:
            return

        # --- Camera properties ---
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.camera_cfg.width
        camera_props.height = self.camera_cfg.height
        camera_props.enable_tensors = self.camera_cfg.enable_tensors
        camera_props.horizontal_fov = self.camera_cfg.horizontal_fov
        camera_props.near_plane = self.camera_cfg.near_clip
        camera_props.far_plane = self.camera_cfg.far_clip

        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        self.cam_handles.append(camera_handle)
        
        # --- nominal extrinsics ---
        x0, y0, z0 = 0.05, 0.0, 0.1

        # --- noise (uniform) ---
        dx, dy, dz = np.random.uniform(-0.03, 0.03, size=3)  # 30mm each axis
        yaw   = np.random.uniform(-5, 5) * math.pi / 180.0   # about +Z
        pitch = np.random.uniform(-5, 5) * math.pi / 180.0   # about +Y
        roll  = np.random.uniform(-2, 2) * math.pi / 180.0   # about +X

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(x0 + dx, y0 + dy, z0 + dz)

        q_base  = gymapi.Quat(0, 0, 0, 1)
        q_noise = quat_from_euler_rpy(roll, pitch, yaw)
        local_transform.r = quat_mul_custom(q_base, q_noise)  # equals q_noise here
        
        # --- Attach to H2 link ---
        head_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "H2")
        assert head_handle != -1, "Rigid body 'H2' not found on actor."

        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            head_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )

        if self.camera_cfg.enable_tensors:
            cam_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handle, camera_handle, gymapi.IMAGE_DEPTH
            )
            self.cam_tensors.append(gymtorch.wrap_tensor(cam_tensor))
            
        if self.show_camera:
            marker_actor = self._create_cam_marker(env_handle, color=(1, 0, 0))
            self.cam_marker_handles.append(marker_actor)

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True
                )
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = max(1.0, np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.terrain_heights(self.env_origins)

    def _init_buffers(self):
        self.num_prop = self.cfg["env"]["num_proprioceptive"]
        self.hist_len = self.cfg["env"]["hist_len"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.hist_len, self.num_prop, dtype=torch.float, device=self.device)

        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}
        self.extras["goal_hit"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # robot wrapper tensors
        self.root_states_all = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, -1, 13)
        self.root_states = self.root_states_all[:, 0, :]
        self.robot_init_state = self.root_states.clone()
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)[:, :self.num_bodies, :]  # shape: num_envs, num_bodies, xyz axis
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, -1, 13)[:, :self.num_bodies, :]  # shape: num_envs, num_bodies, 13 (xyz pos, quat, xyz vel, ang vel)
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        # ball wrapper tensors
        self.ball_states = self.root_states_all[:, 1, :]
        self.ball_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)[:, self.ball_indice, :]  # shape: num_envs, 1, 3
        self.ball_pos = self.ball_states[:, 0:3]
        self.ball_quat = self.ball_states[:, 3:7]
        self.ball_lin_vel = self.ball_states[:, 7:10]
        self.ball_ang_vel = self.ball_states[:, 10:13]

        # goal wrapper tensors
        self.goal_states = self.root_states_all[:, 2, :]
        self.goal_pos = self.goal_states[:, 0:3]
        
        # camera marker
        if self.use_camera and self.show_camera:
            self.cam_marker_states = self.root_states_all[:, 3, :]

        # Head wrapper tensors
        self.head_pos = self.body_states[:, self.head_link_indice, 0:3]
        self.head_quat = self.body_states[:, self.head_link_indice, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.curriculum_prob = torch.zeros(
            1 + 2 * self.cfg["commands"]["lin_vel_levels"],
            1 + 2 * self.cfg["commands"]["ang_vel_levels"],
            dtype=torch.float,
            device=self.device,
        )
        self.curriculum_prob[self.cfg["commands"]["lin_vel_levels"], self.cfg["commands"]["ang_vel_levels"]] = 1.0
        self.env_curriculum_level = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0
        if self.show_camera: # hardcoded
            self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies + 2 + 1, 3, dtype=torch.float, device=self.device)
            self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies + 2 + 1, 3, dtype=torch.float, device=self.device)
        else:
            self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies + 2, 3, dtype=torch.float, device=self.device)
            self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies + 2, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]

        # kick specific states
        self.robot_ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)  # ball's position in robot's frame
        self.robot_ball_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)  # ball velocity in robot frame
        self.robot_goal_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)  # goal's position in robot's frame
        self.robot_ball_pos_yaw2d = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)  # ball's position in robot's frame
        self.robot_ball_vel_yaw2d = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)  # ball velocity in robot frame
        self.robot_goal_pos_yaw2d = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)  # goal's position in robot's frame
        self.head_forward_vec_local = to_torch([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1) # Assuming X is forward for the head
        
        self.perception_update_interval = self.cfg["env"]["perception_update_interval"]
        self.next_ball_pos_update_step = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * self.perception_update_interval
        self.next_goal_pos_update_step = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * self.perception_update_interval

        # prepare task sampler
        self.task_sampler = TaskSampler(self.cfg)  # tasks represented by ball and robot positions resampled at reset or ball out of bounds

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        self.penalty_reward_scales = self.cfg["rewards"]["penalty_scales"].copy()
        for key in list(self.penalty_reward_scales.keys()):
            scale = self.penalty_reward_scales[key]
            if scale == 0:
                self.penalty_reward_scales.pop(key)
            else:
                self.penalty_reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        self.penalty_reward_functions = []
        self.penalty_reward_names = []
        for name, scale in self.penalty_reward_scales.items():
            self.penalty_reward_names.append(name)
            name = "_reward_" + name
            self.penalty_reward_functions.append(getattr(self, name))
        self.penalty_schedule = self.cfg["rewards"]["penalty_schedule"]

    def reset(self):
        """Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._resample_commands()
        self._compute_observations()

        # initialize the ball and goal pos update steps
        self.next_ball_pos_update_step = torch.randint(0, self.perception_update_interval, (self.num_envs,), device=self.device) + 1
        self.next_goal_pos_update_step = torch.randint(0, self.perception_update_interval, (self.num_envs,), device=self.device) + 1
        # clear the observation history
        self.obs_buf = torch.zeros(self.num_envs, self.hist_len, self.num_prop, dtype=torch.float, device=self.device)

        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._update_curriculum(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0
        self.falcon_prev_position_fov[env_ids] = self.base_pos[env_ids, :2]
        self.falcon_prev_ee_reward[env_ids] = 0.0
        self.obs_buf[env_ids] = 0.0 # reset observation history, 12/06 previously was not reset
        self.actions[env_ids] = 0.0 # reset actions, 12/22 previously was not reset
        self.last_falcon_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0 # reset actions, 12/22 previously was not reset
        self.success_step_buf[env_ids] = -1 # reset success step buffer
        if self.use_camera:
            self.depth_buffer[env_ids] = 0.0 # reset depth buffer

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32) * 3  # two actors: robot and ball
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        # robot root states
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, 0] += (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * ROBOT_INIT_LENGTH * self.cfg["task_sampler"]["random_init_pos"]
        self.root_states[env_ids, 1] += (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * ROBOT_INIT_WIDTH * self.cfg["task_sampler"]["random_init_pos"]
        self.root_states[env_ids, 2] += self.terrain.terrain_heights(self.root_states[env_ids, :2])
        
        if self.cfg["task_sampler"].get("robot_yaw_min") is not None:
            yaw_low = self.cfg["task_sampler"]["robot_yaw_min"]
            yaw_high = self.cfg["task_sampler"]["robot_yaw_max"]
            if self.cfg["task_sampler"]["robot_yaw_mirror"] and torch.rand(1).item() < 0.5:
                yaw_low, yaw_high = -yaw_high, -yaw_low
        else:
            yaw_low = -self.cfg["task_sampler"]["robot_yaw_range"]
            yaw_high = self.cfg["task_sampler"]["robot_yaw_range"]
        yaw = yaw_low + (yaw_high - yaw_low) * torch.rand(
            len(env_ids), device=self.device
        )
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            yaw
        )
        # self.root_states[env_ids, 7:9] = apply_randomization(
        #     torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
        #     self.cfg["randomization"].get("init_base_lin_vel_xy"),
        # )
        self.robot_init_state[env_ids] = self.root_states[env_ids].clone()

        self._reset_ball(env_ids)

    def _reset_ball(self, env_ids):
        # sample ball pos
        self.ball_states[env_ids, :3], self.root_states[env_ids, :] = \
            self.task_sampler.sample_task(env_ids, self.root_states[env_ids], self.env_origins[env_ids])
        # set ball orientation
        # yaw = (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi) - torch.pi
        yaw = torch.rand(len(env_ids), device=self.device) * 2 * self.cfg["task_sampler"]["ball_yaw_range"] - self.cfg["task_sampler"]["ball_yaw_range"] / 2.0
        self.ball_states[env_ids, 3:7] = torch.stack([torch.zeros_like(yaw), 
                                                      torch.zeros_like(yaw), 
                                                      torch.sin(yaw/2), 
                                                      torch.cos(yaw/2)], dim=-1)

        self.ball_states[env_ids, 7:13] = torch.zeros(len(env_ids), 6, dtype=torch.float, device=self.device)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))
        
        # store original distance from object to goal for curriculum
        ball_to_goal = self.goal_pos[env_ids, :2] - self.ball_pos[env_ids, :2]
        ball_goal_dist = torch.norm(ball_to_goal, dim=1)
        self.falcon_original_obj2goal_dist[env_ids] = ball_goal_dist
        self.falcon_last_obj2goal_dist[env_ids] = ball_goal_dist

        robot_ball_pos = self.ball_pos[env_ids, :2] - self.base_pos[env_ids, :2]
        robot_ball_dist = torch.norm(robot_ball_pos, dim=1)
        self.falcon_original_robot2obj_dist[env_ids] = robot_ball_dist

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
        raise NotImplementedError("We should not reach here")
        out_x_min = self.root_states[:, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size
        self.root_states[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size
        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))
            self._refresh_feet_state()

    def _resample_commands(self):
        # not useful, remove later
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        if self.cfg["commands"]["curriculum"]:
            self._resample_curriculum_commands(env_ids)
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                self.cfg["commands"]["lin_vel_x"][0], self.cfg["commands"]["lin_vel_x"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(
                self.cfg["commands"]["lin_vel_y"][0], self.cfg["commands"]["lin_vel_y"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(
                self.cfg["commands"]["ang_vel_yaw"][0], self.cfg["commands"]["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        still_envs = env_ids[torch.randperm(len(env_ids))[: int(self.cfg["commands"]["still_proportion"] * len(env_ids))]]
        self.commands[still_envs, :] = 0.0
        self.gait_frequency[still_envs] = 0.0
        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / self.dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

    def _update_curriculum(self, env_ids):
        if not self.cfg["commands"]["curriculum"]:
            return
        success = self.episode_length_buf[env_ids] > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) * (
            1 - self.cfg["commands"]["episode_length_toler"]
        )
        success &= torch.abs(self.filtered_lin_vel[env_ids, 0] - self.commands[env_ids, 0]) < self.cfg["commands"]["lin_vel_x_toler"]
        success &= torch.abs(self.filtered_lin_vel[env_ids, 1] - self.commands[env_ids, 1]) < self.cfg["commands"]["lin_vel_y_toler"]
        success &= torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2]) < self.cfg["commands"]["ang_vel_yaw_toler"]
        for i in range(len(env_ids)):
            if success[i]:
                x = self.env_curriculum_level[env_ids[i], 0] + self.cfg["commands"]["lin_vel_levels"]
                y = self.env_curriculum_level[env_ids[i], 1] + self.cfg["commands"]["ang_vel_levels"]
                self.curriculum_prob[x, y] += self.cfg["commands"]["update_rate"]
                if x > 0:
                    self.curriculum_prob[x - 1, y] += self.cfg["commands"]["update_rate"]
                if x < self.curriculum_prob.shape[0] - 1:
                    self.curriculum_prob[x + 1, y] += self.cfg["commands"]["update_rate"]
                if y > 0:
                    self.curriculum_prob[x, y - 1] += self.cfg["commands"]["update_rate"]
                if y < self.curriculum_prob.shape[1] - 1:
                    self.curriculum_prob[x, y + 1] += self.cfg["commands"]["update_rate"]
        self.curriculum_prob.clamp_(max=1.0)

    def _resample_curriculum_commands(self, env_ids):
        grid_idx = torch.multinomial(self.curriculum_prob.flatten(), len(env_ids), replacement=True)
        lin_vel_level = grid_idx % self.curriculum_prob.shape[1] - self.cfg["commands"]["lin_vel_levels"]
        ang_vel_level = grid_idx // self.curriculum_prob.shape[1] - self.cfg["commands"]["ang_vel_levels"]
        self.env_curriculum_level[env_ids, 0] = lin_vel_level
        self.env_curriculum_level[env_ids, 1] = ang_vel_level
        self.mean_lin_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 0]).float())
        self.mean_ang_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 1]).float())
        self.max_lin_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 0]))
        self.max_ang_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 1]))
        self.commands[env_ids, 0] = (
            lin_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["lin_vel_x_resolution"]
        self.commands[env_ids, 1] = (
            torch.abs(lin_vel_level)
            * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
            * self.cfg["commands"]["lin_vel_y_resolution"]
        )
        self.commands[env_ids, 2] = (
            ang_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["ang_vel_resolution"]

    def falcon_step(self, actions):
        # TODO: we may need to implement some clipping mechanism to make sure the actions are in valid range
        """
        1. actions torch.Size([1, 23])
        2. base_ang_vel torch.Size([1, 3])
        3. command_ang_vel torch.Size([1, 1])
        4. command_base_height torch.Size([1, 1])
        5. command_lin_vel torch.Size([1, 2])
        6. command_stand torch.Size([1, 1])
        7. command_waist_dofs torch.Size([1, 3])
        8. dof_pos torch.Size([1, 23])
        9. dof_vel torch.Size([1, 23])
        10. projected_gravity torch.Size([1, 3])
        11. ref_upper_dof_pos torch.Size([1, 10])
        """
        self.last_falcon_actions[:] = actions.clone()

        command_ang_vel = actions[:, 0:1].clone()
        command_base_height = 0.62 + torch.zeros_like(command_ang_vel)  # desired base height
        command_lin_vel = actions[:, 1:3].clone() # zero linear velocity command
        command_waist_dofs = torch.zeros([self.num_envs, 3], device=self.device)
        
        # print("command_ang_vel", command_ang_vel)
        # print("command_lin_vel", command_lin_vel)
        # print("self.base_ang_vel", self.base_ang_vel)
        # case 1
        upper_ref = self.desired_upper_residual.clone()
        
        # case 2
        upper_ref += actions[:, 3:13].clone()

        ref_upper_dof_pos = upper_ref + self.default_dof_pos[:, self.falcon_upper_dof_indices] # predicting residual upper 

        obs_curr_dict = {}
        obs_curr_dict["actions"] = self.last_actions.clone() 
        obs_curr_dict["base_ang_vel"] = self.base_ang_vel.clone()
        obs_curr_dict["command_ang_vel"] = command_ang_vel
        obs_curr_dict["command_base_height"] = command_base_height
        obs_curr_dict["command_lin_vel"] = command_lin_vel
        obs_curr_dict["command_stand"] = torch.ones([self.num_envs, 1], device=self.device)
        obs_curr_dict["command_waist_dofs"] = command_waist_dofs
        obs_curr_dict["dof_pos"] = self.dof_pos.clone() - self.default_dof_pos
        obs_curr_dict["dof_vel"] = self.dof_vel.clone()
        obs_curr_dict["projected_gravity"] = self.projected_gravity.clone()
        obs_curr_dict["ref_upper_dof_pos"] = ref_upper_dof_pos
        self.falcon_ref_upper_dof_pos[:] = obs_curr_dict["ref_upper_dof_pos"]

        tmp_obs_dict = {}
        for obs_key in obs_curr_dict.keys():
            obs_scale = self.falcon_obs_scales[obs_key] # this scaled is applied in the original codebase
            curr_obs_scaled = obs_curr_dict[obs_key] * obs_scale
            tmp_obs_dict[obs_key] = curr_obs_scaled

        obs_keys = sorted(tmp_obs_dict.keys())
        curr_obs_tensor = torch.cat([tmp_obs_dict[key] for key in obs_keys], dim=-1)
        assert curr_obs_tensor.shape[1] == self.falcon_actor_obs_num, f"Current obs tensor shape is not {self.falcon_actor_obs_num}"
        self.falcon_obs_buf_tensor = torch.cat((self.falcon_obs_buf_tensor[:, self.falcon_actor_obs_num:self.falcon_actor_obs_num*self.falcon_history_length], curr_obs_tensor), dim=-1)
        
        decoupled_actions = []
        for actor_name in sorted(self.falcon_policy.actors.keys())[::-1]:
            actor_model = self.falcon_policy.actors[actor_name]
            decoupled_actions.append(actor_model(self.falcon_obs_buf_tensor))
        actions = torch.cat(decoupled_actions, dim=1)

        return actions
    
    def step(self, actions):
        actions = self.falcon_step(actions)
        # pre physics step
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        
        # ----------- Integrate with FALCON -----------
        pd_tar = self.cfg["control"]["action_scale"] * self.actions
        pd_tar[:, self.falcon_upper_dof_indices] += (self.falcon_ref_upper_dof_pos - self.default_dof_pos[:, self.falcon_upper_dof_indices])
        dof_targets = self.default_dof_pos + pd_tar

        # perform physics step
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
            
        if self.use_camera and self.show_camera:
            print("updating cam marker")
            for env_i, env_handle in enumerate(self.envs):
                cam_handle = self.cam_handles[env_i]
                self._update_cam_marker(env_i, env_handle, cam_handle)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        # compute camera data
        self.update_depth_buffer()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self._refresh_feet_state()
        # robot goal pos computed from 2D heading
        heading = calc_heading(self.base_quat, w_last=True)
        q_yaw_2d = torch.stack([
            torch.zeros_like(heading),              # x
            torch.zeros_like(heading),              # y
            torch.sin(heading / 2),                 # z
            torch.cos(heading / 2)                  # w
        ], dim=-1).to(self.device)
        # update kick states
        self.robot_ball_pos_yaw2d[:] = quat_rotate_inverse(q_yaw_2d, self.ball_pos - self.base_pos)
        self.robot_ball_vel_yaw2d[:] = quat_rotate_inverse(q_yaw_2d, self.ball_lin_vel - self.base_lin_vel)
        self.robot_goal_pos_yaw2d[:] = quat_rotate_inverse(q_yaw_2d, self.goal_pos - self.base_pos)
        
        self.robot_ball_pos[:] = quat_rotate_inverse(self.base_quat, self.ball_pos - self.base_pos)
        self.robot_ball_vel[:] = quat_rotate_inverse(self.base_quat, self.ball_lin_vel - self.base_lin_vel)
        self.robot_goal_pos[:] = quat_rotate_inverse(self.base_quat, self.goal_pos - self.base_pos)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        # self._kick_robots() # disable this because my policy learns weird stuff
        self._push_robots()
        self._check_termination()
        self._compute_reward()

        self._compute_observations() # get all next obs

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def handle_reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)
        self._teleport_robot()
        self._teleport_ball()
        self._resample_commands()

        self._compute_observations(env_ids)
        return self.obs_buf, self.extras
    
    def get_obs(self):
        robot_root_pos = self.base_pos
        robot_rot = self.base_quat
        obj_root_pos = self.ball_pos
        obj_rot = self.ball_quat

        obj_pos_local = self.robot_ball_pos 
        
        right_ee_index = self.body_names.index('right_hand_link')
        left_ee_index  = self.body_names.index('left_hand_link')

        right_ee_pos = self.body_states[:, right_ee_index, :3]
        left_ee_pos  = self.body_states[:, left_ee_index, :3]
        
        rel_right_ee_robot_root_pos = right_ee_pos - robot_root_pos
        right_ee_pos_local = quat_rotate_inverse(robot_rot, rel_right_ee_robot_root_pos)
        right_ee_obj_pos = right_ee_pos_local - obj_pos_local # first field
        rel_left_ee_robot_root_pos = left_ee_pos - robot_root_pos
        left_ee_pos_local = quat_rotate_inverse(robot_rot, rel_left_ee_robot_root_pos)
        left_ee_obj_pos = left_ee_pos_local - obj_pos_local # first field

        q_bRo = quat_mul(quat_conjugate(robot_rot), obj_rot)
        R_bRo = quaternion_to_matrix(q_bRo).reshape(-1, 9)

        upper_dof_pos_diff = (self.dof_pos - self.default_dof_pos)[:, self.falcon_upper_dof_indices]
        upper_dof_vel = self.dof_vel[:, self.falcon_upper_dof_indices]

        obj_goal_pos_local = self.robot_goal_pos[:, :2]
        
        actor_obs = torch.cat([
            right_ee_obj_pos, # 3
            left_ee_obj_pos, # 3
            R_bRo, # 9
            upper_dof_pos_diff, # 10
            upper_dof_vel, # 10
            self.base_lin_vel, # 3
            self.base_ang_vel, # 3
            self.projected_gravity, # 3
            obj_goal_pos_local, # 2
            self.last_falcon_actions # 13
        ], dim=1)


        # priviledged information
        bvo = quat_rotate_inverse(robot_rot, self.ball_lin_vel)
        bwo = quat_rotate_inverse(robot_rot, self.ball_ang_vel)
        
        contact_buf = self.ball_contact_forces[:, :2] # filter out z direction force
        contact_norm = torch.norm(contact_buf, dim=-1, keepdim=True)
        contact_flag = (contact_norm > 1e-4).float() # good
        
        bpcom = quat_rotate_inverse(robot_rot, obj_root_pos)

        critic_obs = torch.cat([
            contact_flag, # 1
            bpcom, # 3
            self.obj_mass.unsqueeze(-1), # 1
            self.obj_dims, # 3
            bvo, # 3
            bwo # 3
        ], dim=1)
        
        return actor_obs, critic_obs
    
    def resize_image(self, depth_image, out_h, out_w):
        """
        depth_image: (H, W) or (B, H, W)
        Returns resized depth image with shape (out_h, out_w) or (B, out_h, out_w)
        """

        # If single image, add batch + channel dims
        single = False
        if depth_image.ndim == 2:
            depth_image = depth_image[None, None, ...]  # (1,1,H,W)
            single = True
        elif depth_image.ndim == 3:
            depth_image = depth_image[:, None, ...]     # (B,1,H,W)

        # Resize (bilinear)
        depth_resized = F.interpolate(
            depth_image, 
            size=(out_h, out_w), 
            mode='bilinear',
            align_corners=False
        )

        # Remove extra dims if single
        if single:
            depth_resized = depth_resized[0, 0]
        else:
            depth_resized = depth_resized[:, 0]

        return depth_resized

    def process_depth_image_batch(self, depth_batch: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of depth images.
        Args:
            depth_batch: (N, H, W) float32 tensor on GPU
        Returns:
            (N, H, W) processed tensor
        """
        N = depth_batch.shape[0]

        # Add uniform noise: shape (N, 1, 1) → broadcast over HxW
        noise = self.camera_cfg.dis_noise * 2 * (torch.rand(N, 1, 1, device=depth_batch.device) - 0.5)
        depth_batch = depth_batch + noise

        # Clip depth values
        depth_batch = torch.clamp(
            depth_batch, -self.camera_cfg.far_clip, -self.camera_cfg.near_clip
        )
        
        depth_batch = self.resize_image(depth_batch, self.camera_cfg.resized_height, self.camera_cfg.resized_width)
        # Normalize in batch
        normalized_depth_batch = self.normalize_depth_image(depth_batch)  # should support (N, H, W)
        if self.camera_cfg.get("view_depth", False):
            self.view_perception_depth(normalized_depth_batch) # only for evaluation
            # input()
        
        return normalized_depth_batch
    
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.camera_cfg.near_clip) / (self.camera_cfg.far_clip - self.camera_cfg.near_clip)  - 0.5
        return depth_image
        
    def view_perception_depth(self, normalized_img):
        depth_img = (normalized_img[0].cpu().numpy() + 0.5) * 255
        depth_uint8 = depth_img.astype(np.uint8)

        # Resize
        depth_resized = cv2.resize(depth_uint8, (256, 256), interpolation=cv2.INTER_NEAREST)

        # --- Add this: convert grayscale to heatmap ---
        colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_JET)
        # Other options: COLORMAP_TURBO, COLORMAP_INFERNO, COLORMAP_MAGMA, etc.

        # Show
        cv2.imshow("Depth View (Env 0)", colored)
        cv2.waitKey(1)
    
    def update_depth_buffer(self):
        if not self.use_camera:
            return

        self.camera_global_counter += 1
        if self.camera_global_counter % self.camera_cfg.update_interval != 0:
            return
        self.camera_global_counter = 0
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)

        if self.camera_cfg.enable_tensors:
            self.gym.start_access_image_tensors(self.sim)
            cam_tensors = torch.stack(self.cam_tensors)
            processed_depth = self.process_depth_image_batch(cam_tensors)  # [N, H, W]
            # for i in range(128):
            #     cv2.imwrite(f"Depth Image{i}.png", ((processed_depth[i].cpu().numpy() + 0.5) * 255).astype(np.uint8))
            #     print("Saved depth image")
            # input()

            self.depth_buffer[:, :-1] = self.depth_buffer[:, 1:].clone()
            self.depth_buffer[:, -1] = processed_depth.to(self.device).clone()

            # seg_tensors = torch.stack(self.seg_tensors)
            # # cv2.imwrite("Seg Image.png", ((seg_tensors[0].cpu().numpy()) * 255).astype(np.uint8))
            # self.seg_buffer[:, :-1] = self.seg_buffer[:, 1:].clone()
            # self.seg_buffer[:, -1] = seg_tensors.to(self.device).clone()
            self.gym.end_access_image_tensors(self.sim)
        else:
            depth_images_np = []

            for i in range(self.num_envs):
                cam = self.cam_handles[i]
                env = self.envs[i]
                depth = self.gym.get_camera_image(self.sim, env, cam, gymapi.IMAGE_DEPTH)
                depth_images_np.append(depth)
                
            # Stack and move to GPU in one shot
            depth_batch_np = np.stack(depth_images_np)  # shape: (num_envs, H, W)
            depth_batch = torch.from_numpy(depth_batch_np).to(self.device)  # single bulk transfer
            depth_batch = self.process_depth_image_batch(depth_batch)
            # cv2.imwrite("Depth Image.png", ((depth_batch[0].cpu().numpy() + 0.5) * 255).astype(np.uint8))
            self.depth_buffer[:] = torch.cat([self.depth_buffer[:, 1:], depth_batch.unsqueeze(1)], dim=1)
        # self.view_perception_seg() # only for evaluation
        # self.update_camera_info() # no need to use this for now since we are using _ball_within_fov
    
    def compute_angle_object_goal(self):
        goal_pos = self.goal_pos[:, :2]  # (N, 2)
        obj_pos = self.ball_pos[:, :2]
        robot_pos = self.base_pos[:, :2]

        # Vectors: object → robot and object → goal
        vec_a = robot_pos - obj_pos  # (N, 2)
        vec_b = goal_pos - obj_pos   # (N, 2)

        # Normalize each vector
        vec_a = vec_a / (vec_a.norm(dim=-1, keepdim=True) + 1e-8)
        vec_b = vec_b / (vec_b.norm(dim=-1, keepdim=True) + 1e-8)

        # Cosine of angle between vectors
        dot_product = torch.sum(vec_a * vec_b, dim=-1).clamp(-1.0, 1.0)
        angle_rad = torch.acos(dot_product)

        # Normalize angle to [0, 1]
        angle_norm = 1 - angle_rad / torch.pi  # 0 = correct opposite side, 1 = incorrect same side
        return angle_norm

    def _teleport_ball(self):
        # if ball is outside the field, teleport it to a random position on the field
        if not self.cfg["env"]["teleport_ball"]:
            return
        raise NotImplementedError("We should not reach here")
        ball_outside_field = torch.logical_or(
            torch.abs(self.ball_states[:, 0] - self.env_origins[:, 0]) > FIELD_LENGTH / 2,
            torch.abs(self.ball_states[:, 1] - self.env_origins[:, 1]) > FIELD_WIDTH / 2,
        )
        if torch.any(ball_outside_field):
            env_ids = ball_outside_field.nonzero(as_tuple=False).flatten()
            self._reset_ball(env_ids)

    def _kick_robots(self):
        raise NotImplementedError("We should not reach here")
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 7:10] = apply_randomization(self.root_states[:, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 10:13] = apply_randomization(self.root_states[:, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))

        # randomly kick ball
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_ball_interval_s"] / self.dt) == 0:
            self.ball_states[:, 7:10] = apply_randomization(self.ball_states[:, 7:10], self.cfg["randomization"].get("kick_ball_lin_vel"))
            self.ball_states[:, 10:13] = apply_randomization(self.ball_states[:, 10:13], self.cfg["randomization"].get("kick_ball_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        self.feet_contact[:] = torch.any(
            (feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos) < 0.01).reshape(
                self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]
            ),
            dim=2,
        )

    def compute_success(self):
        obj = self.ball_pos[:, :2]
        goal = self.goal_pos[:, :2]
        dist = torch.norm(obj - goal, dim=-1)

        thr = self.evaluation_cfg.get("success_distance_threshold", GOAL_WIDTH / 2)
        streak_N = self.evaluation_cfg.get("final_success_streak", 100)

        raw_success = dist < thr

        # update streak only for active envs
        self.success_step_buf[~raw_success] = -1
        self.success_step_buf[raw_success] += 1

        # latch final success after N successful steps (absorbing)
        return self.success_step_buf >= streak_N
    
    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf |= self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.reset_buf |= self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        self.time_out_buf = self.episode_length_buf >= np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) # here timeout will be reset
        self.reset_buf |= self.time_out_buf
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time
        self.extras["time_outs"] = self.time_out_buf
        
        if self.cfg["env"].get("early_termination", False):
            success = self.compute_success()
            self.reset_buf |= success
            self.extras["time_outs"] |= success

        # reset when ball outside field
        if self.cfg["env"].get("reset_ball_outside_field", False):
            raise NotImplementedError
            ball_outside_field = torch.logical_or(
                torch.abs(self.ball_states[:, 0] - self.env_origins[:, 0]) > FIELD_LENGTH / 2,
                torch.abs(self.ball_states[:, 1] - self.env_origins[:, 1]) > FIELD_WIDTH / 2,
            )
            self.reset_buf |= ball_outside_field
            # check hit goal
            ball_goal_dist = torch.norm(self.ball_states[:, :2] - self.goal_pos[:, :2], dim=-1)
            self.extras["goal_hit"] = torch.logical_and(
                ball_outside_field,
                ball_goal_dist < GOAL_WIDTH / 2
            )

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew

        penalty_stage = min(max((self.common_step_counter / 24. - self.penalty_schedule[2]), 0) / self.penalty_schedule[3], 1)
        penalty_coef = (
            self.penalty_schedule[1] * penalty_stage + self.penalty_schedule[0] * (1 - penalty_stage)
        )  # linear interpolation between penalty_schedule[0] and penalty_schedule[1]
        for i in range(len(self.penalty_reward_functions)):
            name = self.penalty_reward_names[i]
            rew = self.penalty_reward_functions[i]() * self.penalty_reward_scales[name]
            self.rew_buf += rew * penalty_coef
            self.extras["rew_terms"][name] = rew

        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _ball_within_fov(self, env_ids, fov_half_angle=1.05):
        # Calculate head forward vector in world frame
        head_forward_world = quat_rotate(self.head_quat[env_ids], self.head_forward_vec_local[env_ids])

        # Calculate vector from head to ball in world frame
        head_to_ball_world = self.ball_pos[env_ids] - self.head_pos[env_ids]
        head_to_ball_distance = torch.norm(head_to_ball_world, dim=-1, keepdim=True)
        
        # Normalize vectors
        head_forward_world_norm = head_forward_world / (torch.norm(head_forward_world, dim=-1, keepdim=True) + 1e-8)
        head_to_ball_world_norm = head_to_ball_world / (head_to_ball_distance + 1e-8)

        # Cosine of the angle between the two vectors (dot product of normalized vectors)
        cos_angle = torch.sum(head_forward_world_norm * head_to_ball_world_norm, dim=-1)
        
        # Angle in radians (acos is numerically unstable around 1 and -1, clamp to avoid NaN)
        angle = torch.acos(torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)) # angle is between 0 and pi

        # Camera field of view constraint (assume 60 degree FOV = ~1.05 radians)
        # TODO: use real FOV from the camera, which should be a rectangle
        # fov_half_angle = 1.05  # ~60 degrees FOV
        within_fov = angle <= fov_half_angle
        return within_fov, angle
    
    def _compute_ball_tilt(self):
        # [num_envs, 4]
        ball_quat = self.ball_states[:, 3:7]

        # local up in object frame
        up_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        # rotate local up into world frame
        up_world = quat_apply(ball_quat, up_local)   # [num_envs, 3]

        # angle between object's up and world up
        cos_theta = up_world[:, 2]                   # dot(up_world, [0,0,1]) = z component
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)                # [num_envs], in radians (0 = perfectly upright)

        return theta
    
    def compute_odom(self, env_ids=None):
        """Get odom return x, y, z, theta 
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        heading = calc_heading(self.base_quat[env_ids], w_last=True)
        return self.base_pos[env_ids, :], heading

    def _compute_observations(self, env_ids=None):
        """Computes observations"""
        env_is_none = False
        if env_ids is None:
            env_is_none = True
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if len(env_ids) == 0:
            return
        actor_obs, critic_obs = self.get_obs()
        actor_obs = actor_obs[env_ids]
        critic_obs = critic_obs[env_ids]
        
        if self.use_camera:
            self.extras["depth"] = self.depth_buffer.clone()
            
        # priviledge only for teacher and critic
        self.privileged_obs_buf[env_ids] = critic_obs

        # update the observation history
        if env_is_none:
            self.obs_buf[env_ids, :, :] = torch.cat([self.obs_buf[env_ids, 1:, :], actor_obs.unsqueeze(1)], dim=1)
        else:
            # we don't want to move the history for all envs when only a subset is updated (handle_reset case)
            self.obs_buf[env_ids, -1, :] = actor_obs

        self.extras["privileged_obs"] = self.extras["critic"] = self.privileged_obs_buf

    def get_odom_observations(self):
        return self.obs_buf[:, :, :-4]

    def get_observations(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._compute_observations(env_ids)
        return self.obs_buf, self.extras

    # ------------ reward functions ------------
    def _reward_object_goal_reaching_gaussian(self):
        # dynamic goal reaching rewards r1 reward
        obj_root_pos = self.ball_pos[:, :2]
        obj_goal_pos = self.goal_pos[:, :2]

        # Compute current robot-to-object distance
        curr_dist = torch.norm(obj_root_pos - obj_goal_pos, dim=-1)
        square_dist = curr_dist ** 2
        
        sigma2 = self.cfg["rewards"].get("obj_goal_reaching_gaussian_sigma2", 10)
        reward = torch.exp(-square_dist / sigma2)

        reward = torch.where(curr_dist < GOAL_WIDTH / 2, torch.full_like(curr_dist, 2.0), reward)
        return reward
    
    def _reward_object_to_goal_reaching_velocity(self):
        # positions (x, y)
        obj_pos = self.ball_pos[:, :2]        # [N, 2]
        goal_pos = self.goal_pos[:, :2]       # [N, 2]

        # direction from object to goal
        p_og = goal_pos - obj_pos             # [N, 2]

        # object linear velocity in xy
        v_o = self.ball_lin_vel[:, :2]        # [N, 2]

        speed = v_o.norm(dim=-1, keepdim=True)  # [N]
        min_speed = self.cfg["rewards"].get("obj_goal_min_speed", 0.01)
    
        # normalize to unit vectors
        v_hat = v_o / (v_o.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        p_hat = p_og / (p_og.norm(dim=-1, keepdim=True).clamp(min=1e-6))

        v_hat = torch.where(speed < min_speed, torch.zeros_like(v_hat), v_hat)

        # dot product (alignment) term
        align = (v_hat * p_hat).sum(dim=-1)       # [N]

        sigma2 = self.cfg["rewards"].get("obj_goal_velocity_sigma2", 0.5)
        # final reward
        reward = torch.exp((align - 1.0) / sigma2)

        # dynamic goal reaching rewards r1 reward
        obj_root_pos = self.ball_pos[:, :2]
        obj_goal_pos = self.goal_pos[:, :2]
        # Compute current robot-to-object distance
        curr_dist = torch.norm(obj_root_pos - obj_goal_pos, dim=-1)
        reward = torch.where(curr_dist < GOAL_WIDTH / 2, torch.zeros_like(reward), reward)
        return reward
    
    def _reward_ee_to_object_reaching(self):
        obj_pos = self.ball_pos[:, :2]
        right_ee_index = self.body_names.index('right_hand_link')
        left_ee_index  = self.body_names.index('left_hand_link')

        right_ee_pos = self.body_states[:, right_ee_index, :2]
        left_ee_pos  = self.body_states[:, left_ee_index, :2]
        
        right_vec = right_ee_pos - obj_pos
        left_vec  = left_ee_pos  - obj_pos

        right_square_dist = torch.sum(torch.square(right_vec), dim=-1)
        left_square_dist  = torch.sum(torch.square(left_vec), dim=-1)

        sigma2 = self.cfg["rewards"].get("ee_object_sigma2", 2)
        e1 = torch.exp(-right_square_dist / sigma2)
        e2 = torch.exp(-left_square_dist / sigma2)

        reward = 2 * e1 * e2 / (e1 + e2 + 1e-6) # harmonic mean
        
        # set this reward to 0 if not align with goal
        close_obj_dist = self.cfg["rewards"].get("align_robot_goal_close_obj_dist", 1.0)
        misaligned_angle = self.cfg["rewards"].get("align_robot_goal_misaligned_angle", 0.2)
        angle_norm = self.compute_angle_object_goal()
        bad_angle = angle_norm > misaligned_angle  # ~150 deg
        close = (self.base_pos[:, :2] - obj_pos).norm(dim=-1) < close_obj_dist
        mask_zero = close & bad_angle
        reward = torch.where(mask_zero, torch.zeros_like(reward), reward)

        # dynamic goal reaching rewards r1 reward
        obj_root_pos = self.ball_pos[:, :2]
        obj_goal_pos = self.goal_pos[:, :2]
        # Compute current robot-to-object distance
        curr_dist = torch.norm(obj_root_pos - obj_goal_pos, dim=-1)
        reward = torch.where(curr_dist < GOAL_WIDTH / 2, self.falcon_prev_ee_reward, reward)

        self.falcon_prev_ee_reward[:] = reward
        return reward

    def _reward_head_tracks_object(self):
        # Camera field of view constraint (assume 60 degree FOV = ~1.05 radians)
        fov_half_angle = self.cfg["rewards"].get("head_tracking_fov", 1.05)  # ~60 degrees FOV
        within_fov, angle = self._ball_within_fov(torch.arange(self.num_envs, device=self.device), fov_half_angle=fov_half_angle)
        
        # Angular tracking reward - tighter sigma for better precision
        sigma2 = self.cfg["rewards"].get("head_tracking_sigma2", 0.5)
        
        angular_reward = torch.exp(-torch.square(angle) / sigma2)

        # Combined reward: angular accuracy * FOV constraint (no distance scaling)
        reward = angular_reward * within_fov.float()
        
        # add guards for far object
        obj_pos = self.ball_pos[:, :2]
        robot_pos = self.base_pos[:, :2]
        far_obj_dist = self.cfg["rewards"].get("far_obj_dist", 2.5)
        far = (robot_pos - obj_pos).norm(dim=-1) > far_obj_dist
        
        return torch.where(far, torch.zeros_like(reward), reward)
    
    def _reward_align_robot_goal(self):
        obj_pos = self.ball_pos[:, :2]
        robot_pos = self.base_pos[:, :2]
        angle_norm = self.compute_angle_object_goal()

        # Square penalty (higher reward when on the other side)
        sigma2 = self.cfg["rewards"].get("align_robot_goal_sigma2", 0.25)
        reward = torch.exp(-torch.square(angle_norm) / sigma2)

        # Zero-reward condition: when the robot is too far from the object
        far_obj_dist = self.cfg["rewards"].get("far_obj_dist", 2.5)
        far = (robot_pos - obj_pos).norm(dim=-1) > far_obj_dist
        reward = torch.where(far, torch.zeros_like(reward), reward)
        return reward

    # ------------ penalty helper functions ------------
    def _reward_default_upper_joints(self):
        # try to make this into zero
        total_joint_reward = torch.sum(torch.square(5 * self.last_falcon_actions[:, 3:13]), dim=-1) # TODO: hardcoded
        return total_joint_reward
            
    def _reward_object_balance(self):
        # penalty
        theta = self._compute_ball_tilt()          # [num_envs]
        theta_lim = self.cfg["rewards"].get("theta_lim", 0.1)  # radians

        # cost: only penalize tilt beyond limit
        excess_tilt = torch.clamp_min(theta - theta_lim, 0.0)  # [num_envs]
        tilt_cost = excess_tilt ** 2

        return tilt_cost
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        lower_cmd = self.actions[:, :3]
        upper_cmd = self.actions[:, 3:]
        last_lower_cmd = self.last_actions[:, :3]
        last_upper_cmd = self.last_actions[:, 3:]

        diff_lower  = torch.square(lower_cmd  - last_lower_cmd)
        diff_upper = torch.square(upper_cmd - last_upper_cmd)

        lower_w = self.cfg["rewards"].get("action_rate_lower_weight", 2.0)
        upper_w = self.cfg["rewards"].get("action_rate_upper_weight", 0.5)

        reward = lower_w * torch.sum(diff_lower, dim=-1) + upper_w * torch.sum(diff_upper, dim=-1)

        return reward



class T1VOFAStudent(T1VOFATeacher):
    def __init__(self,cfg):
        self.expert_obs = torch.zeros(
            cfg["env"]["num_envs"], 
            cfg["env"]["hist_len"], 
            cfg["env"]["num_expert_obs"], dtype=torch.float, device=cfg["basic"]["rl_device"])
        super().__init__(cfg)

    def _compute_observations(self, env_ids=None):
        """Computes observations"""
        env_is_none = False
        if env_ids is None:
            env_is_none = True
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if len(env_ids) == 0:
            return
        actor_obs, critic_obs = self.get_obs()
        actor_obs = actor_obs[env_ids]
        critic_obs = critic_obs[env_ids]
        
        if self.use_camera:
            self.extras["depth"] = self.depth_buffer.clone()
            
        # priviledge only for teacher and critic
        self.privileged_obs_buf[env_ids] = critic_obs

        # student obs
        robot_ball_pos = apply_perception_noise(
                self.robot_ball_pos_yaw2d[env_ids],
                self.robot_ball_vel_yaw2d[env_ids],
                self.dt,
                c_vel=self.cfg["noise"].get("ball_pos").get("c_vel"),
                c_min=self.cfg["noise"].get("ball_pos").get("c_min")
            )
        
        robot_goal_pos = apply_perception_noise( 
            self.robot_goal_pos_yaw2d[env_ids],
            self.base_lin_vel[env_ids],
            self.dt,
            c_vel=self.cfg["noise"].get("goal_pos").get("c_vel"),
            c_min=self.cfg["noise"].get("goal_pos").get("c_min")
        )
        projected_gravity = apply_randomization(self.projected_gravity[env_ids], self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"]
        base_ang_vel = apply_randomization(self.base_ang_vel[env_ids], self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"]
        dof_pos = apply_randomization(self.dof_pos[env_ids] - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"]
        dof_vel = apply_randomization(self.dof_vel[env_ids], self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"]
        prop_buf = torch.cat(
                (   
                    projected_gravity,
                    base_ang_vel,
                    dof_pos,
                    dof_vel,
                    self.last_falcon_actions[env_ids],
                    robot_goal_pos[:, :2] # TODO: don't normalize goal pos for camera version # temporary sim2real to remove
                ),
                dim=-1,
        )

        # update the observation history
        if env_is_none:
            self.obs_buf[env_ids, :, :] = torch.cat([self.obs_buf[env_ids, 1:, :], prop_buf.unsqueeze(1)], dim=1)
            self.expert_obs[env_ids, :, :] = torch.cat([self.expert_obs[env_ids, 1:, :], actor_obs.unsqueeze(1)], dim=1)
        else:
            # we don't want to move the history for all envs when only a subset is updated (handle_reset case)
            self.obs_buf[env_ids, -1, :] = prop_buf
            self.expert_obs[env_ids, -1, :] = actor_obs

        self.extras["privileged_obs"] = self.extras["critic"] = self.privileged_obs_buf
        self.extras["expert_obs"] = self.expert_obs

    def compute_success(self):
        obj = self.ball_pos[:, :2]
        goal = self.goal_pos[:, :2]
        dist = torch.norm(obj - goal, dim=-1)

        thr = self.evaluation_cfg.get("success_distance_threshold", GOAL_WIDTH / 2)
        streak_N = self.evaluation_cfg.get("final_success_streak", 100)

        raw_success = dist < thr

        # update streak only for active envs
        self.success_step_buf[~raw_success] = -1
        self.success_step_buf[raw_success] += 1

        # latch final success after N successful steps (absorbing)
        return self.success_step_buf >= streak_N

    # below are overriding the depth image processing functions
    def resize_image(self, depth_image, out_h, out_w):
        """
        depth_image: (H, W) or (B, H, W)
        Returns resized depth image with shape (out_h, out_w) or (B, out_h, out_w)
        """

        # If single image, add batch + channel dims
        single = False
        if depth_image.ndim == 2:
            depth_image = depth_image[None, None, ...]  # (1,1,H,W)
            single = True
        elif depth_image.ndim == 3:
            depth_image = depth_image[:, None, ...]     # (B,1,H,W)

        # Resize (nearest)
        depth_resized = F.interpolate(
            depth_image, 
            size=(out_h, out_w), 
            mode='nearest',
        )

        # Remove extra dims if single
        if single:
            depth_resized = depth_resized[0, 0]
        else:
            depth_resized = depth_resized[:, 0]

        return depth_resized

    def normalize_depth_image(self, depth_image):
        depth_image = (depth_image - self.camera_cfg.near_clip) / (self.camera_cfg.far_clip - self.camera_cfg.near_clip) # [0, 1]
        return 1 - depth_image # near is 1 and far is 0
    
    def compute_ground_start_row(self, x: torch.Tensor, thresh: float, min_count: int) -> torch.Tensor:
        """
        x: (N,H,W) depth-like image.
        Returns:
        ground_start: (N,) int64. The first row index y where count(x[y, :] > thresh) >= min_count.
                        If no row meets condition, returns H (meaning "no ground found").
        """
        device = x.device
        N, H, W = x.shape

        # If x is normalized [0,1], convert thresh from "50" (uint8-ish) to 50/255
        x_max = x.detach().amax()
        if x_max <= 1.5:
            thresh_eff = thresh / 255.0
        else:
            thresh_eff = thresh

        # count per row: number of pixels above threshold
        # counts: (N,H)
        counts = (x > thresh_eff).sum(dim=2)

        # row condition: (N,H) boolean
        cond = counts >= min_count

        # Find first row that satisfies cond. If none, set to H.
        y_idx = torch.arange(H, device=device).view(1, H).expand(N, H)  # (N,H)
        ground_start = torch.where(cond, y_idx, torch.full_like(y_idx, H)).min(dim=1).values  # (N,)

        return ground_start

    def augment_rects_above_ground(
        self,
        x: torch.Tensor,
        ground_start: torch.Tensor,   # (N,) int64
        n_patches: int,
        min_hfrac: float,
        max_hfrac: float,
        min_wfrac: float,
        max_wfrac: float,
        val_lo: float,
        val_hi: float,
    ) -> torch.Tensor:
        """
        x: (N,H,W) normalized depth in [0,1], 0=far, 1=near
        ground_start: (N,) first ground row index; rectangles will only be placed in rows < ground_start
        """
        device, dtype = x.device, x.dtype
        N, H, W = x.shape

        ys = torch.arange(H, device=device).view(1, H, 1)  # (1,H,1)
        xs = torch.arange(W, device=device).view(1, 1, W)  # (1,1,W)

        # topH per sample: how many rows available above ground
        topH = ground_start.clamp(min=1, max=H)  # (N,)

        n_patches = random.randint(1, n_patches + 1)

        for _ in range(n_patches):
            # Sample rectangle sizes; height cannot exceed topH for each sample.
            rh = ((min_hfrac + (max_hfrac - min_hfrac) * torch.rand(N, device=device)) * H).long().clamp(min=1, max=H)
            rw = ((min_wfrac + (max_wfrac - min_wfrac) * torch.rand(N, device=device)) * W).long().clamp(min=1, max=W)
            rh = torch.minimum(rh, topH)  # ensure fits in available top region

            # Sample y0 uniformly in [0, topH - rh] per sample
            max_y0 = (topH - rh).clamp(min=0)  # (N,)
            y0 = (torch.rand(N, device=device) * (max_y0.to(dtype) + 1.0)).long()

            # Sample x0 uniformly in [0, W - rw]
            max_x0 = (W - rw).clamp(min=0)
            x0 = (torch.rand(N, device=device) * (max_x0.to(dtype) + 1.0)).long()

            y0v = y0.view(N, 1, 1)
            x0v = x0.view(N, 1, 1)
            y1v = (y0 + rh).view(N, 1, 1)
            x1v = (x0 + rw).view(N, 1, 1)

            rect_mask = ((ys >= y0v) & (ys < y1v) & (xs >= x0v) & (xs < x1v)).to(dtype)  # (N,H,W)

            # --- blocky noise ---
            block_ds = 4  # 3–5 works well for 32x32
            h2 = max(1, H // block_ds)
            w2 = max(1, W // block_ds)

            lowres = torch.rand((N, 1, h2, w2), device=device, dtype=dtype)
            lowres = torch.nn.functional.interpolate(
                lowres, size=(H, W), mode="nearest"
            ).squeeze(1)

            rect_values = val_lo + (val_hi - val_lo) * lowres

            # apply only inside rectangle
            x = x * (1.0 - rect_mask) + rect_values * rect_mask

        return x.clamp(0, 1)
    
    def augment_corr_noise(self, x, aug_corr_downsample: int, aug_corr_amp: float):
        # low-res noise then upsample -> correlated
        device, dtype = x.device, x.dtype
        N, H, W = x.shape

        ds = int(aug_corr_downsample)
        h2, w2 = max(4, H // ds), max(4, W // ds)
        noise = torch.randn((N, 1, h2, w2), device=device, dtype=x.dtype)
        noise = F.interpolate(noise, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        amp = float(aug_corr_amp)  # +/- 0.03 in normalized units
        x = (x + amp * noise).clamp(0, 1)
        return x 

    def augment_dropout(self, x: torch.Tensor, drop_prob: float):
        device, dtype = x.device, x.dtype
        N, H, W = x.shape

        drop_prob = float(drop_prob)
        keep = (torch.rand((N, H, W), device=device) > drop_prob).to(x.dtype)

        x = x * keep
        return x 

    def augment_depth_normalized(self, x: torch.Tensor):
        """
        x: (N,H,W) in [0,1], 1=near, 0=far
        returns: x_aug, valid_mask (N,H,W) float {0,1}
        """
        device = x.device
        
        cfg = self.camera_cfg
        p_patch   = cfg.get("p_patch", 0.75)
        p_corr    = cfg.get("p_corr", 0.75)
        p_dropout = cfg.get("p_dropout", 1.0)

        ground_start = self.compute_ground_start_row(
            x,
            thresh=self.camera_cfg.get("ground_row_threshold", 50.0),
            min_count=self.camera_cfg.get("ground_row_min_count", 6),
        )

        if torch.rand(1, device=device) <= p_patch:
            x = self.augment_rects_above_ground(
                x,
                ground_start,
                n_patches=cfg.get("aug_n_patches", 3),
                min_hfrac=cfg.get("aug_patch_min_hfrac", 0.10),
                max_hfrac=cfg.get("aug_patch_max_hfrac", 0.35),
                min_wfrac=cfg.get("aug_patch_min_wfrac", 0.05),
                max_wfrac=cfg.get("aug_patch_max_wfrac", 0.14),
                val_lo=cfg.get("aug_patch_val_lo", 0.01),
                val_hi=cfg.get("aug_patch_val_hi", 0.23)
            )
        if torch.rand(1, device=device) <= p_corr:
            x = self.augment_corr_noise(
                x, 
                aug_corr_downsample=cfg.get("aug_corr_downsample", 16), 
                aug_corr_amp=cfg.get("aug_corr_amp", 0.03)
            )

        if torch.rand(1, device=device) <= p_dropout:
            # drop prob should be uniformly sampled 
            drop_prob = random.uniform(
                cfg.get("aug_dropout_prob_min", 0.003), 
                cfg.get("aug_dropout_prob_max", 0.03)
            )
            x = self.augment_dropout(
                x,
                drop_prob=drop_prob
            )

        return x

    def process_depth_image_batch(self, depth_batch: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of depth images.
        Args:
            depth_batch: (N, H, W) float32 tensor on GPU
        Returns:
            (N, H, W) processed tensor
        """
        N, H, W = depth_batch.shape

        # Add uniform noise: shape (N, 1, 1) → broadcast over HxW
        depth_batch = -depth_batch
        
        depth_batch = torch.nan_to_num(depth_batch, nan=self.camera_cfg.far_clip, posinf=self.camera_cfg.far_clip, neginf=0.0)
        # TODO add noise later
        # background_mask = depth_batch >= self.camera_cfg.far_clip - 1e-3
        # foreground_mask = ~background_mask
        noise = self.camera_cfg.dis_noise * 2 * (torch.rand(N, 1, 1, device=depth_batch.device) - 0.5)
        depth_batch = depth_batch + noise
        
        depth_batch = self.resize_image(depth_batch, self.camera_cfg.resized_height, self.camera_cfg.resized_width)
        
        # Clip depth values
        depth_batch = torch.clamp(
            depth_batch, self.camera_cfg.near_clip, self.camera_cfg.far_clip
        )
        
        # Normalize in batch
        normalized_depth_batch = self.normalize_depth_image(depth_batch)  # should support (N, H, W)
        if self.camera_cfg.get("apply_noise", False):
            normalized_depth_batch = self.augment_depth_normalized(normalized_depth_batch)
        
        if self.camera_cfg.get("view_depth", False):    
            self.view_perception_depth(normalized_depth_batch) # only for evaluation
            # input()
        
        return normalized_depth_batch
    
    def view_perception_depth(self, normalized_img):
        depth_img = normalized_img[0].cpu().numpy() * 255
        depth_uint8 = depth_img.astype(np.uint8)

        # Resize
        depth_resized = cv2.resize(depth_uint8, (256, 256), interpolation=cv2.INTER_NEAREST)

        # --- Add this: convert grayscale to heatmap ---
        colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_JET)
        # Other options: COLORMAP_TURBO, COLORMAP_INFERNO, COLORMAP_MAGMA, etc.

        # Show
        # cv2.imwrite(f"depth_view_step_{self.common_step_counter}.png", colored)
        cv2.imshow("Depth View (Env 0)", colored)
        cv2.waitKey(1)
