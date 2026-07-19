import torch
from torch import Tensor
import numpy as np

from isaacgym.torch_utils import (
    quat_rotate,
    quat_from_angle_axis
)

@torch.jit.script
def calc_heading(q: Tensor, w_last: bool) -> Tensor:
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def calc_heading_quat(q: Tensor, w_last: bool) -> Tensor:
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q, w_last)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q

@torch.jit.script
def calc_heading_quat_inv(q: Tensor, w_last: bool = False) -> Tensor:
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q, w_last)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q

def apply_randomization(tensor, params, return_noise=False):
    if params == None:
        return tensor

    if params["distribution"] == "gaussian":
        mu, var = params["range"]
        noise = torch.randn_like(tensor) if isinstance(tensor, torch.Tensor) else np.random.randn()
        noise_val = mu + var * noise
    elif params["distribution"] == "uniform":
        lower, upper = params["range"]
        noise = torch.rand_like(tensor) if isinstance(tensor, torch.Tensor) else np.random.rand()
        noise_val = lower + (upper - lower) * noise
    else:
        raise ValueError(f"Invalid randomization distribution: {params['distribution']}")

    if params["operation"] == "additive":
        result = tensor + noise_val
    elif params["operation"] == "scaling":
        result = tensor * noise_val
    else:
        raise ValueError(f"Invalid randomization operation: {params['operation']}")

    if return_noise:
        return result, noise
    else:
        return result

def discount_values(rewards, dones, values, next_values, gamma, lam):
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(advantages[-1, :])
    for t in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[t, :].float()
        delta = rewards[t, :] + gamma * next_nonterminal * next_values[t] - values[t]
        advantages[t, :] = last_advantage = delta + gamma * lam * next_nonterminal * last_advantage
    return advantages

def surrogate_loss(old_actions_log_prob, actions_log_prob, advantages, e_clip=0.2):
    ratio = torch.exp(actions_log_prob - old_actions_log_prob)
    surrogate = -advantages * ratio
    surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - e_clip, 1.0 + e_clip)
    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
    return surrogate_loss

@torch.jit.script
def velocity_dependent_noise(
    p_dot: torch.Tensor,
    dt: float,
    c_vel: float = 5.0,
    c_min: float = 0.01,
) -> torch.Tensor:
    """Velocity-dependent SDE noise used for head position.

    Args:
        p_dot: (B, 3) head velocity vectors in m/s.
        dt:     simulation timestep (seconds).
        c_vel:  velocity scaling constant.
        c_min:  minimum noise floor.

    Returns:
        (B, 3) noise to be added to the head position.
    """

    # |p_dot| (batch, 1)
    speed = torch.norm(p_dot, dim=1, keepdim=True)

    # scale ~ (|v| / c_vel + c_min) * sqrt(dt)
    noise_scale = (speed / c_vel + c_min) * (dt ** 0.5)

    # Wiener increment dW ~ N(0, I)
    dW = torch.normal(mean=0.0, std=1.0, size=p_dot.size(), device=p_dot.device)

    return noise_scale * dW

def apply_perception_noise(pos, vel, dt, c_vel=0.2, c_min=1.5):
    pos_noise = velocity_dependent_noise(vel, dt, c_vel, c_min)
    # print(f"Pos noise: {torch.norm(pos_noise, dim=1)}")
    pos_new = pos.clone()
    pos_new += pos_noise
    return pos_new