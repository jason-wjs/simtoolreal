"""Original teacher observations on canonical mjlab scene tensors.

Scene tensors use local positions, wxyz rotations and world-frame COM velocities.
The policy retains the original xyzw convention and delayed noisy actor inputs.
"""
import math
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul, convert_quat

NUM_JOINTS: int = 29

NUM_FINGERTIPS: int = 5

NUM_KEYPOINTS: int = 4

PALM_CENTER_OFFSET: tuple[float, float, float] = (-0.0, -0.02, 0.16)

FINGERTIP_OFFSET: tuple[float, float, float] = (0.02, 0.002, 0.0)

KEYPOINT_CORNERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (-1, -1, -1),
)

OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos": NUM_JOINTS,
    "joint_vel": NUM_JOINTS,
    "prev_action_targets": NUM_JOINTS,
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "object_rot": 4,
    "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,  # 15
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,  # 12
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,  # 12
    "object_scales": 3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,  # 5
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
}

def compute_obs_dim(field_list) -> int:
    """Return total tensor dim for an ordered list of obs field names."""
    return sum(OBS_FIELD_SIZES[f] for f in field_list)

def _stack_obs_dict(obs_dict: dict[str, torch.Tensor], field_list) -> torch.Tensor:
    """Concatenate named tensors in config order."""
    return torch.cat(
        [obs_dict[f].reshape(obs_dict[f].shape[0], -1) for f in field_list],
        dim=-1,
    )

def _perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Apply random-axis rotation noise to wxyz quaternions."""
    n = q_wxyz.shape[0]
    axis = torch.nn.functional.normalize(
        torch.randn(n, 3, device=q_wxyz.device), dim=-1
    )
    angle = torch.empty(n, device=q_wxyz.device).uniform_(
        -max_deg, max_deg
    ) * (math.pi / 180.0)
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, q_wxyz)

def _apply_local_offset(
    pos_w: torch.Tensor,
    rot_wxyz: torch.Tensor,
    offset: tuple[float, float, float],
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    """Apply one local-frame offset to batched world poses."""
    offset_t = torch.as_tensor(offset, device=pos_w.device, dtype=pos_w.dtype)
    offset_t = offset_t.expand(*batch_shape, 3)
    shifted = quat_apply(
        rot_wxyz.reshape(-1, 4), offset_t.reshape(-1, 3)
    ).reshape(*batch_shape, 3)
    return pos_w + shifted

def _keypoints_world(
    center_pos: torch.Tensor,    # (N, 3)
    center_rot: torch.Tensor,    # (N, 4) wxyz
    kp_offsets: torch.Tensor,    # (N, K, 3)
) -> torch.Tensor:
    """Rotate + translate object-frame keypoints."""
    n_envs, k, _ = kp_offsets.shape
    rot_r = center_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    offsets_r = kp_offsets.reshape(-1, 3)
    return center_pos.unsqueeze(1) + quat_apply(rot_r, offsets_r).reshape(n_envs, k, 3)

def _sample_delay(
    queue: torch.Tensor,
    values: torch.Tensor,
    env,
    flush: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push current values into a rolling queue and sample per-env delay."""
    if flush is not None and flush.any():
        queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)

    queue = torch.roll(queue, shifts=1, dims=1)
    queue[:, 0, :] = values
    idx = torch.randint(0, queue.shape[1], (env.num_envs,), device=env.device)
    delayed = queue[torch.arange(env.num_envs, device=env.device), idx]
    return queue, delayed

class SimToolRealObservations:
    def __init__(self, cfg, object_scales, joint_limits, device):
        self.cfg, self.device = cfg, device
        self.num_envs = len(object_scales)
        self._joint_lower_canon, self._joint_upper_canon = joint_limits.unbind(-1)
        self._object_scale_per_env = object_scales
        rew, dr = cfg.reward, cfg.domain_randomization
        corners = torch.tensor(KEYPOINT_CORNERS, device=device, dtype=torch.float32)
        self._keypoint_offsets = corners.unsqueeze(0) * (
            object_scales * rew.object_base_size * rew.keypoint_scale * .5).unsqueeze(1)
        self._keypoint_offsets_fixed = (corners * (.5 * rew.keypoint_scale *
            torch.tensor(rew.fixed_size, device=device)).unsqueeze(0)).unsqueeze(0).expand(self.num_envs,-1,-1).contiguous()
        self._object_scale_multiplier = torch.ones_like(object_scales)
        self._object_state_queue = torch.zeros(self.num_envs, max(1, dr.object_state_delay_max), 13, device=device)
        self._obs_queue = torch.zeros(self.num_envs, max(1, dr.obs_delay_max), compute_obs_dim(cfg.obs.obs_list), device=device)

    def reset(self, ids, scale_multiplier):
        self._object_scale_multiplier[ids] = scale_multiplier
        self._object_state_queue[ids] = 0
        self._obs_queue[ids] = 0

    def distances(self, state):
        obj, goal = state['object_pose'], state['goal_pose']
        offsets = self._keypoint_offsets_fixed if self.cfg.reward.fixed_size_keypoint_reward else self._keypoint_offsets
        object_keypoints = _keypoints_world(obj[:, :3], obj[:, 3:7], offsets)
        goal_keypoints = _keypoints_world(goal[:, :3], goal[:, 3:7], offsets)
        fingertip_distances = torch.norm(state['fingertip_pose'][:, :, :3] - obj[:, None, :3], dim=-1)
        keypoint_distances = torch.norm(object_keypoints - goal_keypoints, dim=-1).max(dim=-1).values
        return fingertip_distances, keypoint_distances

    def build(self, state, progress, previous_targets, reward):
        dr = self.cfg.domain_randomization
        joint_pos = 2. * (state['joint_pos'] - self._joint_lower_canon) / (self._joint_upper_canon - self._joint_lower_canon) - 1.
        joint_vel, prev_targets_canon = state['joint_vel'], previous_targets
        palm_rot, palm_vel = state['palm_pose'][:, 3:7], state['palm_velocity']
        palm_pos = _apply_local_offset(state['palm_pose'][:, :3], palm_rot, PALM_CENTER_OFFSET, (self.num_envs,))
        ft_pos = _apply_local_offset(state['fingertip_pose'][:, :, :3], state['fingertip_pose'][:, :, 3:7], FINGERTIP_OFFSET, (self.num_envs, NUM_FINGERTIPS))
        obj_pos, obj_rot = state['object_pose'][:, :3], state['object_pose'][:, 3:7]
        obj_vel = state['object_velocity']
        goal_pos, goal_rot = state['goal_pose'][:, :3], state['goal_pose'][:, 3:7]
        episode_start = (progress.episode_length_buf == 0) & (progress._successes == 0)
        if dr.use_object_state_delay_noise:
            values = torch.cat([obj_pos, obj_rot, obj_vel], dim=-1)
            self._object_state_queue, delayed = _sample_delay(self._object_state_queue, values, self, flush=episode_start)
            noisy_obj_pos = delayed[:, :3] + torch.randn_like(delayed[:, :3]) * dr.object_state_xyz_noise_std
            noisy_obj_rot = _perturb_quat(delayed[:, 3:7], dr.object_state_rotation_noise_degrees)
            noisy_obj_vel = delayed[:, 7:13]
        else:
            noisy_obj_pos, noisy_obj_rot, noisy_obj_vel = obj_pos, obj_rot, obj_vel
        kp_offsets = self._keypoint_offsets * self._object_scale_multiplier.unsqueeze(1)
        obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
        goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)
        noisy_obj_kp = _keypoints_world(noisy_obj_pos, noisy_obj_rot, kp_offsets)

        noisy_goal_kp = goal_kp

        keypoints_rel_palm_clean = obj_kp - palm_pos.unsqueeze(1)
        keypoints_rel_palm_noisy = noisy_obj_kp - palm_pos.unsqueeze(1)
        keypoints_rel_goal_clean = obj_kp - goal_kp
        keypoints_rel_goal_noisy = noisy_obj_kp - noisy_goal_kp

        fingertip_pos_rel_palm = (
            ft_pos - palm_pos.unsqueeze(1)
        )  # (N, 5, 3)

        object_scales_obs = self._object_scale_per_env * self._object_scale_multiplier

        # Policy obs use legacy Isaac Gym xyzw; internal math stays wxyz.
        palm_rot_xyzw = convert_quat(palm_rot, to="xyzw")
        obj_rot_xyzw = convert_quat(obj_rot, to="xyzw")
        noisy_obj_rot_xyzw = convert_quat(noisy_obj_rot, to="xyzw")

        obs_clean: dict[str, torch.Tensor] = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "prev_action_targets": prev_targets_canon,
            "palm_pos": palm_pos,
            "palm_rot": palm_rot_xyzw,
            "palm_vel": palm_vel,
            "object_rot": obj_rot_xyzw,
            "object_vel": obj_vel,
            "fingertip_pos_rel_palm": fingertip_pos_rel_palm,
            "keypoints_rel_palm": keypoints_rel_palm_clean,
            "keypoints_rel_goal": keypoints_rel_goal_clean,
            "object_scales": object_scales_obs,
            "closest_keypoint_max_dist": progress._closest_keypoint_max_dist.unsqueeze(-1),
            "closest_fingertip_dist": progress._closest_fingertip_dist,
            "lifted_object": progress._lifted_object.float().unsqueeze(-1),
            "progress": torch.log(progress.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
            "successes": torch.log(progress._successes.float() + 1.0).unsqueeze(-1),
            "reward": (reward * 0.01).unsqueeze(-1),
        }

        obs_noisy = dict(obs_clean)
        obs_noisy["object_rot"] = noisy_obj_rot_xyzw
        obs_noisy["object_vel"] = noisy_obj_vel
        obs_noisy["keypoints_rel_palm"] = keypoints_rel_palm_noisy
        obs_noisy["keypoints_rel_goal"] = keypoints_rel_goal_noisy
        if dr.joint_velocity_obs_noise_std > 0:
            obs_noisy["joint_vel"] = (
                joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
            )

        state_tensor = _stack_obs_dict(obs_clean, self.cfg.obs.state_list)
        policy_tensor = _stack_obs_dict(obs_noisy, self.cfg.obs.obs_list)

        if dr.use_obs_delay:
            self._obs_queue, policy_tensor = _sample_delay(self._obs_queue, policy_tensor, self, flush=episode_start)

        clip = self.cfg.obs.clamp_abs_observations
        policy_tensor = policy_tensor.clamp(-clip, clip)
        state_tensor = state_tensor.clamp(-clip, clip)

        return {"policy": policy_tensor, "critic": state_tensor}

    def get_state(self):
        return {name: getattr(self, name).clone() for name in
                ('_object_scale_multiplier', '_object_state_queue', '_obs_queue')}

    def set_state(self, state):
        for name in ('_object_scale_multiplier', '_object_state_queue', '_obs_queue'):
            getattr(self, name).copy_(state[name])
