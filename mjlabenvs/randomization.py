"""Native teacher reset sampling and persistent world-frame wrench state."""

from __future__ import annotations

import math

import torch

from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul, random_orientation


def _scale_workspace_bounds(
    mins: torch.Tensor, maxs: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale workspace bounds about their center."""
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins) * scale
    return center - half, center + half


def sample_absolute_goal_pose(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
    n_envs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample position and orientation inside the workspace."""
    mins_t = torch.as_tensor(mins, device=device, dtype=torch.float32)
    maxs_t = torch.as_tensor(maxs, device=device, dtype=torch.float32)
    lo, hi = _scale_workspace_bounds(mins_t, maxs_t, scale)
    pos = lo + (hi - lo) * torch.rand(n_envs, 3, device=device)
    return pos, random_orientation(n_envs, device=device)


def sample_delta_goal_pose(
    prev_pos: torch.Tensor,              # (N, 3)
    prev_quat_wxyz: torch.Tensor,         # (N, 4)
    delta_distance: float,
    delta_rotation_degrees: float,
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb the previous goal by a bounded random walk."""
    device = prev_pos.device
    n = prev_pos.shape[0]

    mins_t = torch.as_tensor(mins, device=device, dtype=torch.float32)
    maxs_t = torch.as_tensor(maxs, device=device, dtype=torch.float32)
    lo, hi = _scale_workspace_bounds(mins_t, maxs_t, scale)

    pos_noise = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * delta_distance
    new_pos = torch.clamp(prev_pos + pos_noise, lo, hi)

    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=-1)
    angle = (torch.rand(n, device=device) * 2.0 - 1.0) * delta_rotation_degrees * (
        math.pi / 180.0
    )
    dq = quat_from_angle_axis(angle, axis)
    new_quat = quat_mul(dq, prev_quat_wxyz)
    return new_pos, new_quat


from simtoolreal_shared.actions import sample_log_uniform


class SimToolRealReset:
    def __init__(self, cfg, joint_limits, default_joint_pos, device):
        self.cfg, self.device = cfg, device
        self.lower, self.upper = joint_limits.unbind(-1)
        self.default_joint_pos = default_joint_pos
        if cfg.fixed_trajectory_file:
            raise ValueError('This port implements the native teacher recipe, not the fixed-trajectory ablation')

    def goal(self, n, previous_pose=None):
        cfg = self.cfg
        if cfg.fixed_goal_pose is not None:
            return torch.tensor(cfg.fixed_goal_pose, device=self.device, dtype=torch.float32).expand(n, -1).clone()
        if previous_pose is None or cfg.goal_sampling_type == 'absolute':
            pos, quat = sample_absolute_goal_pose(cfg.target_volume_mins, cfg.target_volume_maxs,
                cfg.target_volume_region_scale, n, self.device)
        elif cfg.goal_sampling_type == 'delta':
            pos, quat = sample_delta_goal_pose(previous_pose[:, :3], previous_pose[:, 3:7],
                cfg.delta_goal_distance, cfg.delta_rotation_degrees, cfg.target_volume_mins,
                cfg.target_volume_maxs, cfg.target_volume_region_scale)
        else:
            raise ValueError(f'Unknown goal sampling mode: {cfg.goal_sampling_type}')
        return torch.cat([pos, quat], dim=-1)

    def full(self, n):
        cfg = self.cfg
        default = self.default_joint_pos.expand(n, -1)
        scale = torch.zeros_like(default)
        scale[:, :7] = cfg.reset_dof_pos_random_interval_arm
        scale[:, 7:] = cfg.reset_dof_pos_random_interval_fingers
        sampled = self.lower + (self.upper - self.lower) * torch.rand_like(default)
        joint_pos = torch.lerp(default, sampled, scale).clamp(self.lower, self.upper)
        joint_vel = torch.empty_like(default).uniform_(-cfg.reset_dof_vel_random_interval,
                                                       cfg.reset_dof_vel_random_interval)
        table_z = cfg.table_reset_z + torch.empty(n, device=self.device).uniform_(
            -cfg.table_reset_z_range, cfg.table_reset_z_range)
        table_pos = torch.zeros(n, 3, device=self.device)
        table_pos[:, 2] = table_z
        rx, ry = cfg.table_reset_xy_range_m
        if rx > 0 or ry > 0:
            table_pos[:, 0] = torch.empty(n, device=self.device).uniform_(-rx, rx)
            table_pos[:, 1] = torch.empty(n, device=self.device).uniform_(-ry, ry)
        if cfg.table_reset_yaw_range_deg > 0:
            yaw = torch.empty(n, device=self.device).uniform_(-1., 1.) * cfg.table_reset_yaw_range_deg * (torch.pi / 180.)
            half = yaw * .5
            w, z = torch.cos(half), torch.sin(half)
            table_quat = torch.stack([w, torch.zeros_like(w), torch.zeros_like(w), z], dim=-1)
        else:
            table_quat = torch.tensor([1., 0., 0., 0.], device=self.device).expand(n, -1)
        table_pose = torch.cat([table_pos, table_quat], dim=-1)
        if cfg.fixed_start_pose is not None:
            object_pose = torch.tensor(cfg.fixed_start_pose, device=self.device, dtype=torch.float32).expand(n, -1).clone()
        else:
            noise = torch.empty(n, 3, device=self.device).uniform_(-1., 1.)
            object_pos = torch.stack([noise[:, 0] * cfg.reset_position_noise_x,
                noise[:, 1] * cfg.reset_position_noise_y,
                table_z + cfg.table_object_z_offset + noise[:, 2] * cfg.reset_position_noise_z], dim=-1)
            object_pose = torch.cat([object_pos, random_orientation(n, device=self.device)], dim=-1)
        return dict(joint_pos=joint_pos, joint_vel=joint_vel,
                    table_pose=table_pose, goal_pose=self.goal(n),
                    object_state=torch.cat([object_pose, torch.zeros(n, 6, device=self.device)], dim=-1))


class SimToolRealWrench:
    _fields = ('_object_forces', '_object_torques', '_random_force_prob', '_random_torque_prob')

    def __init__(self, cfg, object_mass, step_dt, device):
        self.cfg, self.device, self.step_dt = cfg, device, step_dt
        self.num_envs = len(object_mass)
        self._object_mass = object_mass.reshape(self.num_envs, 1)
        self._object_forces = torch.zeros(self.num_envs, 1, 3, device=device)
        self._object_torques = torch.zeros_like(self._object_forces)
        self._random_force_prob = sample_log_uniform(cfg.force_prob_range, self.num_envs).to(device)
        self._random_torque_prob = sample_log_uniform(cfg.torque_prob_range, self.num_envs).to(device)

    def reset(self, ids):
        self._object_forces[ids] = 0
        self._object_torques[ids] = 0
        self._random_force_prob[ids] = sample_log_uniform(self.cfg.force_prob_range, ids.numel()).to(self.device)
        self._random_torque_prob[ids] = sample_log_uniform(self.cfg.torque_prob_range, ids.numel()).to(self.device)

    def step(self, lifted):
        dr = self.cfg
        if dr.force_decay > 0:
            self._object_forces *= dr.force_decay ** (self.step_dt / dr.force_decay_interval)
        else:
            self._object_forces.zero_()
        if dr.torque_decay > 0:
            self._object_torques *= dr.torque_decay ** (self.step_dt / dr.torque_decay_interval)
        else:
            self._object_torques.zero_()
        force_fire = (torch.rand(self.num_envs, device=self.device) < self._random_force_prob).view(-1, 1, 1)
        torque_fire = (torch.rand(self.num_envs, device=self.device) < self._random_torque_prob).view(-1, 1, 1)
        mass = self._object_mass.unsqueeze(-1)
        new_force = torch.randn(self.num_envs, 1, 3, device=self.device) * mass * dr.force_scale
        new_torque = torch.randn(self.num_envs, 1, 3, device=self.device) * mass * dr.torque_scale
        self._object_forces = torch.where(force_fire, new_force, self._object_forces)
        self._object_torques = torch.where(torque_fire, new_torque, self._object_torques)
        if dr.force_only_when_lifted:
            self._object_forces *= lifted.float().view(-1, 1, 1)
        if dr.torque_only_when_lifted:
            self._object_torques *= lifted.float().view(-1, 1, 1)
        return self._object_forces, self._object_torques

    def get_state(self):
        return {name: getattr(self, name).clone() for name in self._fields}

    def set_state(self, state):
        for name in self._fields:
            getattr(self, name).copy_(state[name])
