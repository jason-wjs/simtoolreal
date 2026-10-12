"""Run the original Lab training CLI with checkpoint/first-rollout state probes.

Use distinct Hydra output directories for fresh and resumed invocations. This
records state only; it does not restore physics or change task/training behavior.
"""
import json
from pathlib import Path

import torch
import numpy as np
from rl_games.algos_torch.a2c_continuous import A2CAgent


def snapshot(agent):
    env = agent.vec_env.env.env.unwrapped
    values = {'joint_pos': env.robot.data.joint_pos, 'joint_vel': env.robot.data.joint_vel,
              'physx_joint_pos': env.robot.root_physx_view.get_dof_positions(),
              'physx_joint_vel': env.robot.root_physx_view.get_dof_velocities(),
              'physx_object_pose_xyzw': env.object.root_physx_view.get_transforms(),
              'physx_object_velocity': env.object.root_physx_view.get_velocities(),
              'object_root': env.object.data.root_state_w, 'goal_root': env.goal_viz.data.root_state_w,
              'targets': env._cur_targets, 'previous_targets': env._prev_targets,
              'episode_length': env.episode_length_buf, 'object_delay_queue': env._object_state_queue,
              'obs_delay_queue': env._obs_queue, 'action_delay_queue': env._action_queue,
              'successes': env._successes, 'curriculum_frame': torch.as_tensor(env._frame_counter),
              'tolerance': torch.as_tensor(env._current_success_tolerance)}
    return {k: v.detach().cpu().clone() for k, v in values.items()}


def differences(expected, actual):
    return {k: {'equal': torch.equal(v, actual[k]),
                'max_abs_difference': float((v.double() - actual[k].double()).abs().max())}
            for k, v in expected.items()}


def physics_readback_differences(state):
    root = state['object_root']
    expected = {'joint_pos': state['joint_pos'], 'joint_vel': state['joint_vel'],
                'object_pose': torch.cat((root[:, :3], root[:, 4:7], root[:, 3:4]), dim=1),
                'object_velocity': root[:, 7:13]}
    actual = {'joint_pos': state['physx_joint_pos'], 'joint_vel': state['physx_joint_vel'],
              'object_pose': state['physx_object_pose_xyzw'],
              'object_velocity': state['physx_object_velocity']}
    return differences(expected, actual)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


original_get = A2CAgent.get_full_state_weights
original_set = A2CAgent.set_full_state_weights
original_play = A2CAgent.play_steps


def get_state(self):
    state = original_get(self)
    state['reproduction_native_state'] = snapshot(self)
    return state


def set_state(self, weights, set_epoch=True):
    before = snapshot(self)
    original_set(self, weights, set_epoch=set_epoch)
    after = snapshot(self)
    # Production checkpoints have task state, but no diagnostic-only snapshot.
    saved_native = weights.get('reproduction_native_state')
    report = {'saved_env_state_is_none': weights['env_state'] is None,
              'saved_native_probe_present': saved_native is not None,
              'saved_vs_restored_native': differences(saved_native, after) if saved_native is not None else None,
              'restored_scene_vs_physx': physics_readback_differences(after),
              'before_vs_after_restore_native': differences(before, after),
              'saved_observation_restored_exactly': all(torch.equal(v, self.obs[k]) for k, v in weights['obs'].items()),
              'epoch': self.epoch_num, 'frame': self.frame,
              'scope': 'Native task state snapshots around actual agent restore; diagnostic adds no physics setters.'}
    if weights['env_state'] is not None:
        actual_state = self.vec_env.get_env_state()
        report['task_state_exact'] = {k: equal_tree(v, actual_state[k]) for k, v in weights['env_state'].items()}
    actual_learner = original_get(self)
    keys = ['model', 'optimizer', 'assymetric_vf_nets', 'central_value_training', 'scaler',
            'rnn_states', 'dones', 'obs', 'current_rewards', 'current_shaped_rewards',
            'current_lengths', 'epoch', 'frame']
    report['learner_state_exact'] = {k: equal_tree(weights[k], actual_learner[k]) for k in keys}
    path = Path(self.config['train_dir'])
    torch.save({'saved': saved_native, 'before': before, 'after': after}, path / 'resume_native_states.pt')
    (path / 'resume_probe.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    self._reproduction_restored_state = after


def play_steps(self):
    if hasattr(self, '_reproduction_restored_state'):
        path = Path(self.config['train_dir'])
        report = differences(self._reproduction_restored_state, snapshot(self))
        (path / 'first_rollout_native_changes.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        del self._reproduction_restored_state
    return original_play(self)


A2CAgent.get_full_state_weights = get_state
A2CAgent.set_full_state_weights = set_state
A2CAgent.play_steps = play_steps

if __name__ == '__main__':
    import faulthandler
    import signal
    import runpy
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'isaacsimenvs/train.py'), run_name='__main__')
