"""Native Lab teacher rewards, goal counters, and tolerance curriculum.

Geometry is supplied in local scene coordinates. The task calls ``dones``
before ``rewards`` and writes new goal poses for the returned goal-reset IDs.
This preserves the native ordering, including its sentinel reset before reward.
"""
import torch

from simtoolreal_shared.rewards import (
    lifting_reward, distance_delta_reward, keypoint_reward, action_penalty,
    update_near_goal_steps, reach_goal_bonus,
)


class SimToolRealProgress:
    _tensor_fields = (
        '_lifted_object', '_closest_keypoint_max_dist', '_closest_fingertip_dist',
        '_successes', '_near_goal_steps', '_prev_episode_successes', '_object_init_z',
        '_keypoints_max_dist', '_curr_fingertip_distances', '_near_goal', '_is_success',
        'episode_length_buf',
    )
    _scalar_fields = ('_current_success_tolerance', '_frame_counter', '_last_curriculum_update')

    def __init__(self, cfg, num_envs, device, max_episode_length):
        self.cfg, self.num_envs, self.device = cfg, num_envs, device
        self.max_episode_length = max_episode_length
        self._arm_joint_ids = torch.arange(7, device=device)
        self._hand_joint_ids = torch.arange(7, 29, device=device)
        self._lifted_object = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._closest_keypoint_max_dist = torch.full((num_envs,), -1., device=device)
        self._closest_fingertip_dist = torch.full((num_envs, 5), -1., device=device)
        self._successes = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._near_goal_steps = torch.zeros_like(self._successes)
        self._prev_episode_successes = torch.zeros_like(self._successes)
        self.episode_length_buf = torch.zeros_like(self._successes)
        self._object_init_z = torch.full((num_envs,),
            cfg.reset.table_reset_z + cfg.reset.table_object_z_offset, device=device)
        self._keypoints_max_dist = torch.zeros(num_envs, device=device)
        self._curr_fingertip_distances = torch.zeros(num_envs, 5, device=device)
        self._near_goal = torch.zeros_like(self._lifted_object)
        self._is_success = torch.zeros_like(self._lifted_object)
        self._current_success_tolerance = cfg.termination.success_tolerance
        self._frame_counter = self._last_curriculum_update = 0

    def _clear_goal(self, ids):
        self._closest_keypoint_max_dist[ids] = -1.
        self._closest_fingertip_dist[ids] = -1.
        self._near_goal_steps[ids] = 0

    def reset(self, ids, object_init_z):
        self._prev_episode_successes[ids] = self._successes[ids]
        self._clear_goal(ids)
        self._lifted_object[ids] = False
        self._successes[ids] = 0
        self.episode_length_buf[ids] = 0
        self._object_init_z[ids] = object_init_z

    def update_curriculum(self):
        self._frame_counter += 1
        term = self.cfg.termination
        if self._frame_counter - self._last_curriculum_update >= term.tolerance_curriculum_interval:
            if self._prev_episode_successes.float().mean().item() >= term.tolerance_curriculum_success_threshold:
                tolerance = self._current_success_tolerance * term.tolerance_curriculum_increment
                self._current_success_tolerance = max(min(tolerance, term.success_tolerance),
                                                      term.target_success_tolerance)
                self._last_curriculum_update = self._frame_counter
        if term.eval_success_tolerance is not None:
            self._current_success_tolerance = float(term.eval_success_tolerance)

    def dones(self, object_z, fingertip_distances, keypoints_max_dist):
        """Called after physics and the caller's episode-length increment."""
        self.update_curriculum()
        self._curr_fingertip_distances = fingertip_distances
        self._keypoints_max_dist = keypoints_max_dist
        self._closest_keypoint_max_dist = torch.where(self._closest_keypoint_max_dist < 0,
            keypoints_max_dist, self._closest_keypoint_max_dist)
        self._closest_fingertip_dist = torch.where(self._closest_fingertip_dist < 0,
            fingertip_distances, self._closest_fingertip_dist)
        term = self.cfg.termination
        self._near_goal = keypoints_max_dist <= self._current_success_tolerance * self.cfg.reward.keypoint_scale
        self._near_goal_steps = update_near_goal_steps(self._near_goal, self._near_goal_steps,
                                                      term.force_consecutive_near_goal_steps)
        self._is_success = self._near_goal_steps >= term.success_steps
        self._successes = self._successes + self._is_success.long()
        goal_ids = self._is_success.nonzero(as_tuple=False).squeeze(-1)
        self._clear_goal(goal_ids)
        self.episode_length_buf[goal_ids] = 0
        fall = object_z < .1
        max_successes = self._successes >= term.max_consecutive_successes if term.max_consecutive_successes > 0 else torch.zeros_like(fall)
        hand_far = fingertip_distances.max(dim=-1).values > 1.5
        truncated = self.episode_length_buf >= self.max_episode_length
        self._termination_reasons = dict(fall=fall, max_successes=max_successes,
                                        hand_far=hand_far, timeout=truncated)
        return fall | max_successes | hand_far, truncated, goal_ids

    def rewards(self, object_z, joint_vel):
        rew, term = self.cfg.reward, self.cfg.termination
        lift, bonus_lift, self._lifted_object = lifting_reward(object_z, self._object_init_z,
            self._lifted_object, rew.lifting_bonus_threshold, rew.lifting_bonus, rew.lifting_rew_scale)
        finger, self._closest_fingertip_dist = distance_delta_reward(self._curr_fingertip_distances,
            self._closest_fingertip_dist, self._lifted_object, rew.distance_delta_rew_scale)
        keypoint, self._closest_keypoint_max_dist = keypoint_reward(self._keypoints_max_dist,
            self._closest_keypoint_max_dist, self._lifted_object, rew.keypoint_rew_scale)
        arm, hand = action_penalty(joint_vel, self._arm_joint_ids, self._hand_joint_ids,
            rew.kuka_actions_penalty_scale, rew.hand_actions_penalty_scale)
        bonus = reach_goal_bonus(self._near_goal, self._is_success, rew.reach_goal_bonus,
                                 term.success_steps, term.force_consecutive_near_goal_steps)
        reward = lift + bonus_lift + finger + keypoint + arm + hand + bonus
        self._reward_terms = dict(fingertip_delta_rew=finger, lifting_rew=lift,
            lift_bonus_rew=bonus_lift, keypoint_rew=keypoint, kuka_actions_penalty=arm,
            hand_actions_penalty=hand, bonus_rew=bonus, total_reward=reward)
        return reward

    def get_state(self):
        state = {name: getattr(self, name).clone() for name in self._tensor_fields}
        state.update({name: getattr(self, name) for name in self._scalar_fields})
        return state

    def set_state(self, state):
        for name in self._tensor_fields:
            getattr(self, name).copy_(state[name])
        for name in self._scalar_fields:
            setattr(self, name, state[name])
