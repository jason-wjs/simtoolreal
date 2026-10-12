"""Load an actual Lab checkpoint/config through the deployment player on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from deployment.rl_player import RlPlayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    player = RlPlayer(140, 29, str(args.config), str(args.checkpoint), 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)[0]
    actual = player.player.model.state_dict()
    assert actual.keys() == checkpoint['model'].keys()
    assert all(torch.equal(actual[k], value) for k, value in checkpoint['model'].items())
    observations = torch.linspace(-0.1, 0.1, 140).reshape(1, 140)
    sequences = []
    for _ in range(2):
        player.reset()
        with torch.no_grad():
            actions = torch.stack([player.get_normalized_action(observations) for _ in range(4)])
        assert actions.shape == (4, 1, 29) and torch.isfinite(actions).all()
        assert actions.abs().max() <= 1
        sequences.append(actions)
    assert torch.equal(*sequences)
    result = {
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'checkpoint_frame': int(checkpoint['frame']),
        'model_tensor_count': len(actual), 'model_matches_checkpoint': True,
        'finite_bounded_actions': True, 'recurrent_reset_repeat_exact': True,
        'actions': sequences[0].tolist(),
        'scope': 'Original deployment player; CPU, 140 observations and 29 actions. Real checkpoint/config loading and recurrent inference only; no task success claim.',
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'actions'}, indent=2))


if __name__ == '__main__':
    main()
