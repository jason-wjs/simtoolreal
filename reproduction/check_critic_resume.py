"""Exercise real SAPG critic updates across the agent's disk checkpoint boundary.

CPU-only synthetic batches isolate learner resume; this is not simulator resume
or a training benchmark. Keep the author's full networks and six SAPG cohorts.
"""
import argparse
import copy
import json
from pathlib import Path

import gym
import numpy as np
import torch
import yaml
from rl_games.torch_runner import Runner, _restore


class LearnerOnlyEnv:
    def get_env_info(self):
        return {'observation_space': gym.spaces.Box(-np.inf, np.inf, (140,), dtype=np.float32),
                'state_space': gym.spaces.Box(-np.inf, np.inf, (140,), dtype=np.float32),
                'action_space': gym.spaces.Box(-1, 1, (29,), dtype=np.float32), 'agents': 1}

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        assert state is None


def agent(output):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml').read_text())
    cfg = config['params']['config']
    cfg.update(device='cpu', device_name='cpu', mixed_precision=False,
               num_actors=6, expl_coef_block_size=1, minibatch_size=96,
               full_experiment_name='0_critic_resume', train_dir=str(output), vec_env=LearnerOnlyEnv())
    cfg['central_value_config']['minibatch_size'] = 96
    runner = Runner()
    runner.load(config)
    result = runner.algo_factory.create(runner.algo_name, base_name='resume_audit', params=runner.params)
    result.init_tensors()
    return result


def max_tensor_difference(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    first = agent(args.output)
    cv = first.central_value_net
    torch.manual_seed(123)
    obs = torch.randn(96, 141)
    obs[:, -1] = first.intr_reward_coef_embd[:, 0].repeat(16)
    batch = {'obs': obs, 'old_values': torch.zeros(96, 1),
             'returns': torch.randn(96, 1), 'actions': torch.zeros(96, 29),
             'dones': torch.zeros(96), 'rnn_masks': None}
    cv.update_dataset(copy.deepcopy(batch))
    losses = [cv.train_net(), cv.train_net()]
    first.epoch_num, first.frame = cv.epoch_num, cv.frame
    # Exercise a nondefault lr so loading only Adam's lr is insufficient.
    cv.lr = 3e-5
    cv.update_lr(cv.lr)
    first.save(str(args.output / 'checkpoint'))
    expected_optimizer = copy.deepcopy(cv.optimizer.state_dict())
    expected_counters = {'epoch': cv.epoch_num, 'frame': cv.frame, 'lr': cv.lr}
    expected_model = copy.deepcopy(cv.state_dict())
    expected_next_loss = cv.train_net()
    expected_next_model = copy.deepcopy(cv.state_dict())
    expected_next_optimizer = copy.deepcopy(cv.optimizer.state_dict())
    resumed = agent(args.output)
    resumed.restore(str(args.output / 'checkpoint.pth'))
    restored = resumed.central_value_net
    loaded_model_difference = max_tensor_difference(expected_model, restored.state_dict())
    loaded_optimizer_equal = equal_tree(expected_optimizer, restored.optimizer.state_dict())
    counters = {'epoch': restored.epoch_num, 'frame': restored.frame, 'lr': restored.lr}
    restored.update_dataset(copy.deepcopy(batch))
    next_loss = restored.train_net()
    next_difference = max_tensor_difference(expected_next_model, restored.state_dict())
    next_optimizer_equal = equal_tree(expected_next_optimizer, restored.optimizer.state_dict())
    # Legacy author checkpoints must remain usable as explicit initialization,
    # but must not silently claim restoration of optimizer history they lack.
    legacy = torch.load(args.output / 'checkpoint.pth', weights_only=False)
    legacy[0].pop('central_value_training', None)
    torch.save(legacy, args.output / 'legacy.pth')
    legacy_rejected = False
    try:
        resumed.restore(str(args.output / 'legacy.pth'))
    except ValueError as error:
        assert 'central critic optimizer/counters' in str(error)
        legacy_rejected = True
    _restore(resumed, {'checkpoint': str(args.output / 'legacy.pth'), 'checkpoint_load_mode': 'weights'})
    legacy_weights_equal = equal_tree(expected_model, restored.state_dict())
    report = {'torch': torch.__version__, 'device': 'cpu', 'cohorts': 6,
              'scope': 'Original full SAPG actor/critic architecture and actual agent disk save/restore; two synthetic critic epochs followed by identical third epoch. Six actors with block size 1, AMP off. No physics, rollout or actor-gradient equivalence claim.',
              'warmup_losses': losses, 'loaded_model_max_abs_difference': loaded_model_difference,
              'loaded_optimizer_equal': loaded_optimizer_equal, 'expected_counters': expected_counters,
              'loaded_counters': counters, 'next_loss_continuous': expected_next_loss,
              'next_loss_resumed': next_loss, 'next_model_max_abs_difference': next_difference,
              'next_optimizer_equal': next_optimizer_equal,
              'legacy_resume_rejected': legacy_rejected, 'legacy_explicit_weights_equal': legacy_weights_equal}
    report['passed'] = (loaded_model_difference == 0 and loaded_optimizer_equal and counters == expected_counters
                        and expected_next_loss == next_loss and next_difference == 0 and next_optimizer_equal
                        and legacy_rejected and legacy_weights_equal)
    (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    first.writer.close()
    resumed.writer.close()
    print(json.dumps(report, indent=2))
    assert report['passed'], 'Critic optimizer/counters or next update did not resume exactly'


if __name__ == '__main__':
    main()
