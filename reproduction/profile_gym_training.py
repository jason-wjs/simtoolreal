"""Measure native Gym training and report its actual checkpoint restoration.

The original task, learning updates and restore implementation are unchanged.
"""
from isaacgym import gymapi  # Must precede torch.

import copy
import hashlib
import json
from pathlib import Path
import runpy
import time

import numpy as np
import torch
from rl_games.algos_torch.a2c_continuous import A2CAgent


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return bool(a == b)


original_epoch = A2CAgent.train_epoch
original_restore = A2CAgent.set_full_state_weights
original_action = A2CAgent.get_action_values
original_step = A2CAgent.env_step
original_get = A2CAgent.get_full_state_weights


def static_scene_digest(env):
    """Read all actors in every native environment; this recipe has static properties."""
    assert not env.randomize and env.good_reset_boundary == 0
    assert not env.should_load_initial_states and not env.cfg['env']['useFixedGoalStates']
    assert not env.enable_camera_sensors
    digest = hashlib.sha256()
    shape_fields = ('compliance', 'contact_offset', 'filter', 'friction', 'rest_offset',
                    'restitution', 'rolling_friction', 'thickness', 'torsion_friction')
    for instance in env.envs:
        for actor in range(env.gym.get_actor_count(instance)):
            row = {'name': env.gym.get_actor_name(instance, actor), 'bodies': [], 'shapes': []}
            for p in env.gym.get_actor_rigid_body_properties(instance, actor):
                row['bodies'].append([p.mass, p.flags, p.com.x, p.com.y, p.com.z,
                    *[getattr(getattr(p.inertia, axis), component) for axis in 'xyz' for component in 'xyz']])
            row['shapes'] = [[getattr(p, key) for key in shape_fields]
                            for p in env.gym.get_actor_rigid_shape_properties(instance, actor)]
            dofs = env.gym.get_actor_dof_properties(instance, actor)
            row['dofs'] = {name: dofs[name].tolist() for name in dofs.dtype.names}
            digest.update(json.dumps(row, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def get_state(self):
    env = self.vec_env.env
    if not hasattr(self, '_static_scene_digest'):
        self._static_scene_digest = static_scene_digest(env)
    state = original_get(self)
    state['reproduction_static_scene_digest'] = self._static_scene_digest
    return state


def record_nonfinite(self, stage):
    env = self.vec_env.env
    bad = {}
    for name, value in vars(env).items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            mask = ~torch.isfinite(value)
            if mask.any():
                bad[name] = {'shape': list(value.shape), 'count': int(mask.sum()),
                             'first_indices': mask.nonzero()[:10].cpu().tolist()}
    report = {'stage': stage, 'steps_after_restore': self._restore_probe_steps,
              'bad_task_tensors': bad,
              'observation_finite': {k: bool(torch.isfinite(v).all()) for k, v in self.obs.items()}}
    (Path(self.config['train_dir']) / 'first_nonfinite.json').write_text(json.dumps(report, indent=2) + '\n')


def action(self, *args, **kwargs):
    if hasattr(self, '_first_restored_task'):
        actual = self.vec_env.env.get_env_state()
        check = {k: equal_tree(v, actual[k]) for k, v in self._first_restored_task.items()}
        (Path(self.config['train_dir']) / 'first_rollout_check.json').write_text(json.dumps(check, indent=2) + '\n')
        del self._first_restored_task
        assert all(check.values()), 'Task state changed before first restored action'
    try:
        return original_action(self, *args, **kwargs)
    except RuntimeError:
        if hasattr(self, '_restore_probe_steps'):
            record_nonfinite(self, 'policy_action')
        raise


def step(self, *args, **kwargs):
    result = original_step(self, *args, **kwargs)
    if hasattr(self, '_restore_probe_steps'):
        self._restore_probe_steps += 1
        if not all(bool(torch.isfinite(v).all()) for v in result[0].values()):
            self.obs = result[0]
            record_nonfinite(self, 'environment_step')
            raise RuntimeError('Nonfinite observation immediately after native environment step')
    return result


def train_epoch(self):
    torch.cuda.synchronize(self.ppo_device)
    start = time.perf_counter()
    result = original_epoch(self)
    torch.cuda.synchronize(self.ppo_device)
    elapsed = time.perf_counter() - start
    env = self.vec_env.env
    record = {'epoch': self.epoch_num, 'unix_time': time.time(),
              'num_envs': self.num_actors, 'horizon': self.horizon_length,
              'cohorts': self.num_actors // self.intr_coef_block_size,
              'object_pool_size': len(env.object_asset_files),
              'frames': self.curr_frames, 'epoch_seconds': elapsed,
              'frames_per_second': self.curr_frames / elapsed,
              'success_tolerance': float(env.success_tolerance),
              'mean_successes': float(env.successes.float().mean()),
              'losses_finite': all(bool(torch.isfinite(x).all()) for values in result[4:9] for x in values)}
    with (Path(self.config['train_dir']) / 'epoch_profile.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    assert record['losses_finite']
    return result


def restore(self, weights, set_epoch=True):
    missing = {'obs_queue', 'action_queue', 'object_state_queue', 'checkpoint_rng',
               'object_scale_noise_multiplier', '_tyler_curriculum_scale'} - weights['env_state'].keys()
    if missing:
        raise ValueError(f'Checkpoint lacks delay history required for training continuation: {sorted(missing)}')
    # The native task mutates its input dictionary, so retain the saved state
    # before calling it. This is evidence only; do not compensate for differences.
    expected = copy.deepcopy(weights)
    original_restore(self, weights, set_epoch=set_epoch)
    actual = self.get_full_state_weights()
    keys = ['model', 'optimizer', 'assymetric_vf_nets', 'central_value_training',
            'rnn_states', 'dones', 'obs', 'current_rewards', 'current_shaped_rewards',
            'current_lengths', 'epoch', 'frame', 'scaler', 'trackers', 'last_mean_rewards']
    report = {'epoch': self.epoch_num, 'frame': self.frame,
              'learner_state_exact': {k: equal_tree(expected[k], actual[k]) for k in keys},
              'saved_task_fields_exact': {k: equal_tree(v, actual['env_state'][k])
                                         for k, v in expected['env_state'].items()},
              'static_scene_exact': expected['reproduction_static_scene_digest'] == actual['reproduction_static_scene_digest'],
              'native_readback': {},
              'scope': 'Original camera-free teacher recipe with generic physical DR disabled; static properties of every native actor, saved task/RNG/learner state and refreshed root/DOF tensors. No hidden physics cache or bitwise future-trajectory claim.'}
    env = self.vec_env.env
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    for key in ['dof_state', 'root_state_tensor']:
        report['native_readback'][key] = float((getattr(env, key).cpu() - expected['env_state'][key].cpu()).abs().max())
    (Path(self.config['train_dir']) / 'resume_probe.json').write_text(json.dumps(report, indent=2) + '\n')
    self._restore_probe_steps = 0
    self._first_restored_task = expected['env_state']
    assert report['static_scene_exact'] and all(report['saved_task_fields_exact'].values())
    assert all(report['learner_state_exact'].values())
    assert all(v <= 1e-6 for v in report['native_readback'].values())


A2CAgent.train_epoch = train_epoch
A2CAgent.set_full_state_weights = restore
A2CAgent.get_action_values = action
A2CAgent.env_step = step
A2CAgent.get_full_state_weights = get_state

if __name__ == '__main__':
    runpy.run_module('isaacgymenvs.train', run_name='__main__')
