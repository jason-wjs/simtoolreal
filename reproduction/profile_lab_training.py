"""Profile the original Lab training CLI without changing rollout/update code."""
import json
from pathlib import Path
import time

import torch
from rl_games.algos_torch.a2c_continuous import A2CAgent

original_epoch = A2CAgent.train_epoch


def train_epoch(self):
    torch.cuda.synchronize(self.ppo_device)
    torch.cuda.reset_peak_memory_stats(self.ppo_device)
    start = time.perf_counter()
    result = original_epoch(self)
    torch.cuda.synchronize(self.ppo_device)
    elapsed = time.perf_counter() - start
    env = self.vec_env.env.env.unwrapped
    record = {
        'epoch': self.epoch_num, 'unix_time': time.time(),
        'num_envs': self.num_actors, 'horizon': self.horizon_length,
        'cohort_size': self.intr_coef_block_size,
        'cohorts': self.num_actors // self.intr_coef_block_size,
        'object_pool_size': len(env._object_urdf_paths),
        'actor_minibatch_size': self.minibatch_size,
        'critic_minibatch_size': self.central_value_net.minibatch_size,
        'frames': self.curr_frames, 'synchronized_epoch_seconds': elapsed,
        'frames_per_second': self.curr_frames / elapsed,
        'author_step_seconds': result[0], 'author_play_seconds': result[1],
        'author_update_seconds': result[2],
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.ppo_device),
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(self.ppo_device),
        'mean_successes': float(env._successes.float().mean()),
        'success_tolerance': env._current_success_tolerance,
    }
    record['losses_finite'] = all(bool(torch.isfinite(x).all()) for values in result[4:9] for x in values)
    with (Path(self.config['train_dir']) / 'epoch_profile.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    assert record['losses_finite']
    return result


A2CAgent.train_epoch = train_epoch

if __name__ == '__main__':
    import faulthandler
    import signal
    import runpy
    # Cooperative setup sampling works in containers that prohibit ptrace.
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'isaacsimenvs/train.py'), run_name='__main__')
