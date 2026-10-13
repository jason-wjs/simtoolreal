"""Measure native Gym training and report its actual checkpoint restoration.

The original task, learning updates and restore implementation are unchanged.
"""
from isaacgym import gymapi  # Must precede torch.

import copy
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
    return a == b


original_epoch = A2CAgent.train_epoch
original_restore = A2CAgent.set_full_state_weights
original_action = A2CAgent.get_action_values
original_step = A2CAgent.env_step


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
    # The native task mutates its input dictionary, so retain the saved state
    # before calling it. This is evidence only; do not compensate for differences.
    expected = copy.deepcopy(weights)
    original_restore(self, weights, set_epoch=set_epoch)
    actual = self.get_full_state_weights()
    keys = ['model', 'optimizer', 'assymetric_vf_nets', 'central_value_training',
            'rnn_states', 'dones', 'obs', 'current_rewards', 'current_shaped_rewards',
            'current_lengths', 'epoch', 'frame']
    report = {'epoch': self.epoch_num, 'frame': self.frame,
              'learner_state_exact': {k: equal_tree(expected[k], actual[k]) for k in keys},
              'saved_task_fields_exact': {k: equal_tree(v, actual['env_state'][k])
                                         for k, v in expected['env_state'].items()},
              'scope': 'Stored-field restore audit only. Does not establish completeness of task/DR/RNG serialization or hidden physics cache continuity.'}
    (Path(self.config['train_dir']) / 'resume_probe.json').write_text(json.dumps(report, indent=2) + '\n')
    self._restore_probe_steps = 0


A2CAgent.train_epoch = train_epoch
A2CAgent.set_full_state_weights = restore
A2CAgent.get_action_values = action
A2CAgent.env_step = step

if __name__ == '__main__':
    runpy.run_module('isaacgymenvs.train', run_name='__main__')
