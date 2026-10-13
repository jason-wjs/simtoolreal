"""Finite native-scene replay separating contact collection from tensor pipeline.

Uses the native scene builder, fixed evaluation-style reset and recorded PD targets.
No policy inference, rewards, resets during rollout, or claims of policy acceptance.
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
from deployment.isaac.isaac_env import create_env_from_cfg, merge_cfg_with_default_config
from deployment.rl_player_utils import read_cfg_omegaconf


def scene_properties(env):
    rows = []
    shape_fields = ('compliance', 'contact_offset', 'filter', 'friction', 'rest_offset',
                    'restitution', 'rolling_friction', 'thickness', 'torsion_friction')
    instance = env.envs[0]
    for actor in range(env.gym.get_actor_count(instance)):
        row = dict(name=env.gym.get_actor_name(instance, actor), bodies=[],
                   body_names=env.gym.get_actor_rigid_body_names(instance, actor))
        for p in env.gym.get_actor_rigid_body_properties(instance, actor):
            row['bodies'].append([p.mass, p.flags, p.com.x, p.com.y, p.com.z,
                *[getattr(getattr(p.inertia, axis), component) for axis in 'xyz' for component in 'xyz']])
        row['shapes'] = [[getattr(p, k) for k in shape_fields]
                        for p in env.gym.get_actor_rigid_shape_properties(instance, actor)]
        dofs = env.gym.get_actor_dof_properties(instance, actor)
        row['dofs'] = {k: dofs[k].tolist() for k in dofs.dtype.names}
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['gpu_off', 'gpu_contacts', 'cpu_contacts'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    assert (args.source is None) == (args.mode == 'gpu_off')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    trajectory = Path('dextoolbench/trajectories/screwdriver/short_screwdriver/spin_horizontal.json')
    traj = json.loads(trajectory.read_text()); traj['start_pose'][2] += .03
    overrides = {
        'pipeline': 'cpu' if args.mode == 'cpu_contacts' else 'gpu',
        'task.env.objectName': 'short_screwdriver', 'task.env.numEnvs': 1,
        'task.env.envSpacing': .4, 'task.env.capture_video': False,
        'task.env.enableCameraSensors': False,
        'task.env.resetPositionNoiseX': 0., 'task.env.resetPositionNoiseY': 0.,
        'task.env.resetPositionNoiseZ': 0., 'task.env.randomizeObjectRotation': False,
        'task.env.resetDofPosRandomIntervalFingers': 0.,
        'task.env.resetDofPosRandomIntervalArm': 0., 'task.env.resetDofVelRandomInterval': 0.,
        'task.env.tableResetZRange': 0., 'task.env.tableResetZ': .38,
        'task.env.asset.table': 'urdf/table_narrow.urdf',
        'task.env.objectScaleNoiseMultiplierRange': [1., 1.],
        'task.env.useFixedInitObjectPose': True, 'task.env.objectStartPose': traj['start_pose'],
        'task.env.useFixedGoalStates': True, 'task.env.fixedGoalStates': traj['goals'],
        'task.env.startArmHigher': True, 'task.env.useActionDelay': False,
        'task.env.useObsDelay': False, 'task.env.useObjectStateDelayNoise': False,
        'task.env.forceScale': 0., 'task.env.torqueScale': 0.,
        'task.env.linVelImpulseScale': 0., 'task.env.angVelImpulseScale': 0.,
        'task.sim.physx.contact_collection': 0 if args.mode == 'gpu_off' else 1,
    }
    cfg = merge_cfg_with_default_config(read_cfg_omegaconf('pretrained_policy/config.yaml', 'cuda:0'))
    cfg.task.sim.physx.use_gpu = True
    env = create_env_from_cfg(cfg, headless=True, overrides=overrides)
    assert env.num_envs == 1 and not env.randomize
    assert env.sim_params.physx.use_gpu
    assert env.device == ('cpu' if args.mode == 'cpu_contacts' else 'cuda:0')
    env.reset_idx(torch.tensor([0], dtype=torch.long, device=env.device))
    properties = scene_properties(env)
    # The native reset fills the tensors and defers simulator submission. Submit
    # identical full tensors in every mode, before any physics step.
    if args.source:
        source = json.loads(args.source.read_text())
        assert source['properties'] == properties, 'Native scene properties differ'
        initial = source['initial']; targets = source['targets']
    else:
        initial = dict(root=env.root_state_tensor.cpu().tolist(), dof=env.dof_state.cpu().tolist())
        q = env.dof_state[:, 0].detach().cpu().numpy()
        lower = env.arm_hand_dof_lower_limits.cpu().numpy()
        upper = env.arm_hand_dof_upper_limits.cpu().numpy()
        targets = []
        for step in range(120):
            target = q.copy()
            # Hold the arm; bounded smooth finger excursion exercises native drives.
            target[7:] += .05 * np.sin(2 * np.pi * step / 120)
            targets.append(np.clip(target, lower, upper).tolist())
    root = torch.tensor(initial['root'], dtype=torch.float32, device=env.device)
    dof = torch.tensor(initial['dof'], dtype=torch.float32, device=env.device)
    env.root_state_tensor.copy_(root); env.dof_state.copy_(dof)
    assert env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(root))
    assert env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(dof))
    env.gym.refresh_actor_root_state_tensor(env.sim); env.gym.refresh_dof_state_tensor(env.sim)
    initial_actual = dict(root=env.root_state_tensor.cpu().tolist(), dof=env.dof_state.cpu().tolist())
    assert initial_actual == initial, 'Initial native readback mismatch'
    rows = []
    for step, value in enumerate(targets):
        target = torch.tensor(value, dtype=torch.float32, device=env.device)
        assert env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(target))
        env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        assert all(torch.isfinite(t).all() for t in [env.root_state_tensor, env.dof_state, env.rigid_body_states])
        row = dict(step=step, root=env.root_state_tensor.cpu().tolist(),
                   dof=env.dof_state.cpu().tolist(), bodies=env.rigid_body_states.cpu().tolist())
        if args.mode == 'cpu_contacts':
            contacts = env.gym.get_env_rigid_contacts(env.envs[0])
            row['contacts'] = [{k: int(c[k]) if k in ['body0', 'body1'] else float(c[k])
                                for k in ['body0', 'body1', 'initialOverlap', 'minDist']} for c in contacts]
        rows.append(row)
    config = OmegaConf.to_container(OmegaConf.create(env.cfg), resolve=True)
    result = dict(mode=args.mode, steps=120, dt=env.sim_params.dt, substeps=env.sim_params.substeps,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(Path('pretrained_policy/config.yaml').read_bytes()).hexdigest(),
        trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest() if args.source else None,
        properties=properties, config=config, initial=initial_actual, targets=targets, trace=rows,
        scope='One native short-screwdriver scene,120 open-loop steps, original dt/substeps. '
              'Contact collection and tensor pipeline are separate interventions. No policy physical acceptance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n')
    env.gym.destroy_sim(env.sim)
    print('REPLAY_COMPLETE', args.mode, flush=True)


if __name__ == '__main__':
    main()
