"""Record native released-policy commands, then replay on GPU/CPU tensor pipelines.

Reuses the official evaluator's scene overrides. Seed42 and removal of display,
wall-clock pacing, and video make this a diagnostic, not an unchanged benchmark.
"""
from isaacgym import gymapi, gymtorch
import torch
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from omegaconf import OmegaConf
from dextoolbench import eval_isaacgym as evaluator
from deployment.isaac.isaac_env import create_env_from_cfg, merge_cfg_with_default_config
from deployment.rl_player_utils import read_cfg_omegaconf
from deployment.rl_player import RlPlayer
from check_gym_pipeline_replay import scene_properties


def array(tensor):
    return tensor.detach().cpu().tolist()


def snapshot(env):
    for name in ['actor_root', 'dof', 'rigid_body']:
        getattr(env.gym, 'refresh_' + name + '_state_tensor')(env.sim)
    assert all(torch.isfinite(t).all() for t in [env.root_state_tensor, env.dof_state, env.rigid_body_states])
    return dict(root=array(env.root_state_tensor), dof=array(env.dof_state), bodies=array(env.rigid_body_states))


def joint_excess(row, env):
    q = torch.tensor(row['dof'])[:, 0]
    return float(torch.maximum(env.arm_hand_dof_lower_limits.cpu() - q,
                               q - env.arm_hand_dof_upper_limits.cpu()).clamp_min(0).max())


class CommandRecorder:
    """Read-only interception of the native task's concrete physics writes."""
    def __init__(self, env):
        self.env, self.native, self.pending, self.rows = env, env.gym, [], []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def record(self, name, sim, value, indices=None, count=None):
        command = dict(name=name, value=array(gymtorch.wrap_tensor(value)))
        if indices is not None:
            command['indices'] = array(gymtorch.wrap_tensor(indices))
            command['count'] = count
        self.pending.append(command)
        if indices is None:
            return getattr(self.native, name)(sim, value)
        return getattr(self.native, name)(sim, value, indices, count)

    def set_dof_position_target_tensor(self, sim, value):
        return self.record('set_dof_position_target_tensor', sim, value)

    def set_dof_position_target_tensor_indexed(self, sim, value, indices, count):
        return self.record('set_dof_position_target_tensor_indexed', sim, value, indices, count)

    def set_actor_root_state_tensor_indexed(self, sim, value, indices, count):
        return self.record('set_actor_root_state_tensor_indexed', sim, value, indices, count)

    def set_dof_state_tensor_indexed(self, sim, value, indices, count):
        return self.record('set_dof_state_tensor_indexed', sim, value, indices, count)

    def apply_rigid_body_force_tensors(self, sim, forces, torques, space):
        # Official evaluation disables external forces/impulses. Fail if that changes.
        assert all(t is None or not torch.count_nonzero(gymtorch.wrap_tensor(t)) for t in [forces, torques])
        return self.native.apply_rigid_body_force_tensors(sim, forces, torques, space)

    def simulate(self, sim):
        commands, self.pending = self.pending, []
        self.native.simulate(sim)
        self.native.fetch_results(sim, True)
        self.rows.append(dict(step=len(self.rows), commands=commands, **snapshot(self.env)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['capture', 'gpu_replay', 'cpu_replay'], required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert (args.source is None) == (args.mode == 'capture')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    source = json.loads(args.source.read_text()) if args.source else None

    def create_scene(config_path, device, headless, overrides):
        cfg = merge_cfg_with_default_config(read_cfg_omegaconf(config_path, device))
        cfg.task.sim.physx.use_gpu = True
        overrides = dict(overrides)
        overrides.update({'pipeline': 'cpu' if args.mode == 'cpu_replay' else 'gpu',
                          'task.sim.physx.contact_collection': 0 if args.mode == 'capture' else 1})
        return create_env_from_cfg(cfg, headless=headless, overrides=overrides)

    class Runner:
        def __init__(self, env, config_path, checkpoint_path, **unused):
            self.env, self.config_path, self.checkpoint_path = env, Path(config_path), Path(checkpoint_path)

        def run_eval(self, num_episodes):
            env = self.env
            assert num_episodes == 1 and env.num_envs == 1 and env.control_freq_inv == 1
            assert not env.randomize and not env.privileged_actions
            assert env.force_scale == env.torque_scale == env.lin_vel_impulse_scale == env.ang_vel_impulse_scale == 0
            properties = scene_properties(env)
            initial = snapshot(env)
            config = OmegaConf.to_container(OmegaConf.create(env.cfg), resolve=True)
            metadata = dict(mode=args.mode, properties=properties, config=config,
                initial=initial, dt=env.sim_params.dt, substeps=env.sim_params.substeps,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scene_helper_sha256=hashlib.sha256(Path(__file__).with_name('check_gym_pipeline_replay.py').read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest() if args.source else None,
                external_wrenches='Zero; official evaluation disables all force and velocity impulse scales.',
                scope='Released policy on short_screwdriver/spin_horizontal, up to600policy steps plus2warmups. Seed42, no GUI/video/pacing. Fresh-process command replay, not benchmark success or physical acceptance.')
            if source:
                assert properties == source['properties']
                assert metadata['config_sha256'] == source['config_sha256']
                assert metadata['checkpoint_sha256'] == source['checkpoint_sha256']
                root = torch.tensor(source['initial']['root'], dtype=torch.float32, device=env.device)
                dof = torch.tensor(source['initial']['dof'], dtype=torch.float32, device=env.device)
                assert env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(root))
                assert env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(dof))
                actual = snapshot(env)
                assert torch.max(torch.abs(torch.tensor(actual['root']) - root.cpu())) < 1e-7
                assert actual['dof'] == source['initial']['dof']
                metadata['initial'] = actual
                rows, deep_streak = [], 0
                for recorded in source['trace']:
                    retained = []
                    for command in recorded['commands']:
                        value = torch.tensor(command['value'], dtype=torch.float32, device=env.device)
                        retained.append(value)
                        method = getattr(env.gym, command['name'])
                        if 'indices' in command:
                            indices = torch.tensor(command['indices'], dtype=torch.int32, device=env.device)
                            retained.append(indices)
                            assert method(env.sim, gymtorch.unwrap_tensor(value), gymtorch.unwrap_tensor(indices), command['count'])
                        else:
                            assert method(env.sim, gymtorch.unwrap_tensor(value))
                    env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
                    row = dict(step=len(rows), **snapshot(env))
                    if args.mode == 'cpu_replay':
                        contacts = env.gym.get_env_rigid_contacts(env.envs[0])
                        row['contacts'] = [{k: int(c[k]) if k in ['body0', 'body1'] else float(c[k])
                                           for k in ['body0', 'body1', 'initialOverlap', 'minDist']} for c in contacts]
                    rows.append(row)
                    row['joint_excess_rad'] = joint_excess(row, env)
                    deep_streak = deep_streak + 1 if any(c['initialOverlap'] < -.005 for c in row.get('contacts', [])) else 0
                    if row['joint_excess_rad'] > .05 or deep_streak >= 12:
                        metadata['physical_review_required'] = 'Joint excess>0.05rad or12samples of raw mesh contact field below-5mm'
                        break
                metadata.update(trace=rows, steps=len(rows), source_steps=source['steps'])
            else:
                player = RlPlayer(140, 29, str(self.config_path), str(self.checkpoint_path), 'cuda:0', 1)
                recorder = CommandRecorder(env); env.gym = recorder
                zero = torch.zeros((1, 29), device=env.device)
                env.step(zero)  # EvalRunner.__init__ performs this first warmup.
                player.reset()
                obs, _, _, _ = env.step(zero)  # _run_episode then performs the second.
                actions = []
                done = False
                for _ in range(600):
                    if any(joint_excess(row, env) > .05 for row in recorder.rows[-2:]):
                        metadata['physical_review_required'] = 'Joint excess>0.05rad'
                        break
                    action = player.get_normalized_action(obs['obs'], deterministic_actions=True)
                    assert torch.isfinite(action).all()
                    actions.append(array(action))
                    obs, _, ended, _ = env.step(action)
                    done = bool(ended[0])
                    if done:
                        break
                assert not recorder.pending
                metadata.update(trace=recorder.rows, steps=len(recorder.rows), actions=actions,
                    policy_steps=len(actions), episode_done=done,
                    goal_pct=100 * int(env.successes[0].item()) / env.max_consecutive_successes)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metadata, allow_nan=False) + '\n')
            env.gym.destroy_sim(env.sim)
            print('POLICY_REPLAY_COMPLETE', args.mode, metadata['steps'], flush=True)

    evaluator.create_env = create_scene
    evaluator.EvalRunner = Runner
    evaluator.tyro.cli = lambda _: evaluator.EvalArgs(object_category='screwdriver', object_name='short_screwdriver',
        task_name='spin_horizontal', config_path=Path('pretrained_policy/config.yaml'),
        checkpoint_path=Path('pretrained_policy/model.pth'))
    evaluator.main()


if __name__ == '__main__':
    main()
