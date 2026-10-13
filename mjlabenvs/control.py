"""Original SimToolReal action pipeline with canonical-order mjlab buffers."""
import torch

from simtoolreal_shared.actions import apply_action_pipeline


class SimToolRealControl:
    def __init__(self, cfg, joint_limits, default_joint_pos, num_envs, device, step_dt):
        self.cfg, self.num_envs, self.device = cfg, num_envs, device
        self.step_dt = step_dt
        # The shared native routine names its internal order "Lab". This port
        # keeps that order canonical, and the scene maps canonical targets to
        # actual native actuator indices at its boundary.
        self._perm_canon_to_lab = torch.arange(29, device=device)
        self._arm_joint_ids = torch.arange(7, device=device)
        self._hand_joint_ids = torch.arange(7, 29, device=device)
        self._arm_lower, self._arm_upper = joint_limits[:7].unbind(-1)
        self._hand_lower, self._hand_upper = joint_limits[7:].unbind(-1)
        self._cur_targets = default_joint_pos.expand(num_envs, -1).clone()
        self._prev_targets = self._cur_targets.clone()
        self._action_queue = torch.zeros(num_envs, max(1, cfg.domain_randomization.action_delay_max),
                                         29, device=device)

    def reset(self, env_ids, joint_pos):
        self._cur_targets[env_ids] = joint_pos
        self._prev_targets[env_ids] = joint_pos
        self._action_queue[env_ids] = 0

    def targets(self, actions, episode_lengths, successes):
        self.episode_length_buf, self._successes = episode_lengths, successes
        apply_action_pipeline(self, actions)
        return self._cur_targets

    def get_state(self):
        return {name: getattr(self, name).clone() for name in
                ('_cur_targets', '_prev_targets', '_action_queue')}

    def set_state(self, state):
        for name in ('_cur_targets', '_prev_targets', '_action_queue'):
            getattr(self, name).copy_(state[name])
