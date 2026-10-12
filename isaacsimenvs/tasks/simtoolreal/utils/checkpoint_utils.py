"""Checkpoint state for the native, single-GPU SimToolReal teacher task.

Scene poses/velocities and task buffers are restored; internal PhysX solver
caches are not serialized, so bitwise future physics is not promised.
"""
import random

import numpy as np
import torch


BUFFERS = (
    '_cur_targets', '_prev_targets', '_object_scale_multiplier', '_lifted_object',
    '_closest_keypoint_max_dist', '_closest_fingertip_dist', '_successes', '_near_goal_steps',
    '_prev_episode_successes', '_object_init_z', '_table_z_per_env', '_object_state_queue',
    '_obs_queue', '_action_queue', '_random_force_prob', '_random_torque_prob',
    '_object_forces', '_object_torques', '_keypoints_max_dist', '_curr_fingertip_distances',
    '_near_goal', '_is_success', 'reward_buf', 'episode_length_buf', 'reset_buf',
    'reset_terminated', 'reset_time_outs',
)
COUNTERS = ('_current_success_tolerance', '_frame_counter', '_last_curriculum_update',
            '_sim_step_counter', 'common_step_counter')


def get_env_state(env):
    if env.cfg.student_obs.enabled or env.cfg.reset.fixed_trajectory_file:
        raise ValueError('Task checkpoint currently supports the original teacher recipe, not camera/student or fixed-trajectory ablations')
    return {
        'scene': env.scene.get_state(),
        'buffers': {name: getattr(env, name).clone() for name in BUFFERS},
        'counters': {name: getattr(env, name) for name in COUNTERS},
        'joint_names': list(env.robot.data.joint_names),
        'object_asset_indices': env._object_asset_index_per_env.clone(),
        'object_scales': env._object_scale_per_env.clone(),
        'joint_velocity_target': env.robot.data.joint_vel_target.clone(),
        'joint_effort_target': env.robot.data.joint_effort_target.clone(),
        'materials': {name: getattr(env, name).root_physx_view.get_material_properties().clone()
                      for name in ('robot', 'object', 'table')},
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()},
    }


def set_env_state(env, state):
    if state is None:
        raise ValueError('Checkpoint has no SimToolReal environment state; use checkpoint_load_mode=weights for fresh initialization')
    assert state['joint_names'] == list(env.robot.data.joint_names)
    assert torch.equal(state['object_asset_indices'].to(env.device), env._object_asset_index_per_env)
    assert torch.equal(state['object_scales'].to(env.device), env._object_scale_per_env)
    for name in BUFFERS:
        value = state['buffers'][name].to(env.device)
        assert value.shape == getattr(env, name).shape, name
        getattr(env, name).copy_(value)
    for name in COUNTERS:
        setattr(env, name, state['counters'][name])
    scene = {kind: {name: {key: value.to(env.device) for key, value in fields.items()}
                    for name, fields in assets.items()} for kind, assets in state['scene'].items()}
    env.scene.reset_to(scene)
    # reset_to assumes joint targets equal measured q/qd. Preserve actual drives.
    env.robot.set_joint_position_target(env._cur_targets)
    env.robot.set_joint_velocity_target(state['joint_velocity_target'].to(env.device))
    env.robot.set_joint_effort_target(state['joint_effort_target'].to(env.device))
    env.object.set_external_force_and_torque(env._object_forces, env._object_torques, is_global=True)
    indices = torch.arange(env.num_envs, dtype=torch.int64, device='cpu')
    for name, materials in state['materials'].items():
        getattr(env, name).root_physx_view.set_material_properties(materials.cpu(), indices)
    env.scene.write_data_to_sim()
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['rng']['cuda']])
