"""Run the native SimToolReal SAPG recipe on its concrete mjlab task.

Use an external finite GPU guard; this entry point bounds optimizer epochs.
The real rollout/physics and separate-process continuation gates must pass
before scheduling formal training.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'rl_games'), str(REPO)]


def namespace(value):
    return SimpleNamespace(**{key: namespace(item) for key,item in value.items()}) if isinstance(value,dict) else value


def prepare_runner(data, task, output, epochs):
    from rl_games.common import vecenv, env_configurations
    from rl_games.torch_runner import Runner
    from mjlabenvs.rlgames import SimToolRealVecEnv
    from simtoolreal_shared.rlgames_observer import EnvStatsAlgoObserver

    agent = copy.deepcopy(data['agent'])
    cfg = agent['params']['config']
    assert not cfg['multi_gpu']
    assert cfg['num_actors'] == task.num_envs
    assert cfg['num_actors'] // cfg['expl_coef_block_size'] == 6
    assert cfg['num_actors'] % cfg['expl_coef_block_size'] == 0
    cfg.update(train_dir=str(output), max_epochs=epochs)
    bounds = agent['params']['env']
    adapter = SimToolRealVecEnv(task, bounds['clip_observations'], bounds['clip_actions'])
    vecenv.register('SimToolRealMjlab', lambda config_name,num_actors,**kw: adapter)
    env_configurations.register(cfg['env_name'], dict(vecenv_type='SimToolRealMjlab',env_creator=lambda **kw: adapter))
    runner = Runner(EnvStatsAlgoObserver())
    runner.load(agent)
    runner.reset()
    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Original resolved or interpolated Lab env+agent YAML')
    parser.add_argument('--robot-assets', type=Path, required=True)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True, help='Absolute final epoch, including restored epochs')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf
    from mjlabenvs.task import SimToolRealTask

    data = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg = data['agent']['params']['config']
    seed = int(data['agent']['params']['seed'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    args.output.mkdir(parents=True, exist_ok=False)
    # Save the exact resolved inputs before expensive scene construction.
    data['agent']['params']['config'].update(train_dir=str(args.output.resolve()),max_epochs=args.epochs)
    (args.output/'config.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    task = SimToolRealTask(namespace(data['env']),args.robot_assets,args.tools,
                          cfg['num_actors'],cfg['device'])
    runner = prepare_runner(data,task,args.output.resolve(),args.epochs)
    (args.output/'launch.json').write_text(json.dumps(dict(seed=seed,num_envs=task.num_envs,
        cohorts=6,object_pool=1200,max_epoch=args.epochs,
        checkpoint=str(args.checkpoint) if args.checkpoint else None),indent=2)+'\n')
    runner.run(dict(train=True,play=False,checkpoint=str(args.checkpoint) if args.checkpoint else None,
                    checkpoint_load_mode='resume'))


if __name__ == '__main__':
    main()
