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


def disable_thumb_index_contacts(env):
    """Diagnostic only: remove three active thumb/index pairs, verify other filters unchanged."""
    instance, actor = env.envs[0], 0
    names = env.gym.get_actor_rigid_body_names(instance, actor)
    spans = env.gym.get_actor_rigid_body_shape_indices(instance, actor)
    properties = env.gym.get_actor_rigid_shape_properties(instance, actor)
    before = [p.filter for p in properties]
    assert not any(mask & (1 << 30) for mask in before)
    owners = {}
    selected = {'left_thumb_PP', 'left_thumb_DP', 'left_index_MP', 'left_index_DP'}
    assert selected.issubset(names)
    for name, span in zip(names, spans):
        for i in range(span.start, span.start + span.count):
            owners[i] = name
            if name in selected:
                properties[i].filter |= 1 << 30
    env.gym.set_actor_rigid_shape_properties(instance, actor, properties)
    after = [p.filter for p in env.gym.get_actor_rigid_shape_properties(instance, actor)]
    assert after == [p.filter for p in properties]
    changed = set()
    for i in range(len(before)):
        for j in range(i):
            if bool(before[i] & before[j]) != bool(after[i] & after[j]):
                assert not before[i] & before[j] and after[i] & after[j]
                changed.add(tuple(sorted((owners[i], owners[j]))))
    # Native filters already suppress thumb_PP/index_MP; do not count it
    # as an intervention or change any additional pair.
    expected = {('left_index_DP','left_thumb_PP'),
                ('left_index_DP','left_thumb_DP'), ('left_index_MP','left_thumb_DP')}
    assert changed == expected, changed
    return dict(removed_pairs=sorted(changed), before_filters=before, after_filters=after,
                scope='Contact-causality probe; disabling physical contacts is not a repair.')


class CommandRecorder:
    """Read-only interception of the native task's concrete physics writes."""
    def __init__(self, env):
        self.env, self.native, self.pending, self.rows = env, env.gym, [], []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def record(self, name, sim, value, indices=None, count=None):
        command = dict(name=name, value=array(gymtorch.wrap_tensor(value)))
        if indices is not None:
            # Native reset code also submits count0 no-ops; Gym cannot wrap an
            # empty descriptor back into torch. Preserve the original call.
            command['indices'] = array(gymtorch.wrap_tensor(indices)) if count else []
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
    parser.add_argument('--config-path', type=Path, default=Path('pretrained_policy/config.yaml'))
    parser.add_argument('--checkpoint-path', type=Path, default=Path('pretrained_policy/model.pth'))
    parser.add_argument('--match-released-velocity-noise', action='store_true',
                        help='Diagnostic: set task joint-velocity observation noise to released value0.01')
    parser.add_argument('--disable-thumb-index-contact', action='store_true')
    parser.add_argument('--position-iterations', type=int, choices=[8, 32, 128],
                        help='Diagnostic native solver convergence; retain collisions and timestep')
    parser.add_argument('--restart-after', type=int, choices=[77, 219],
                        help='Restore a saved contact-entry state, then replay its next32commands')
    parser.add_argument('--switch-iterations-after', type=int, choices=[219],
                        help='Diagnostic: preserve native history, request live solver change before next32commands')
    args = parser.parse_args()
    assert (args.source is None) == (args.mode == 'capture')
    assert not args.disable_thumb_index_contact or args.mode == 'gpu_replay'
    assert args.position_iterations is None or (args.mode in ('capture','gpu_replay') and not args.disable_thumb_index_contact)
    assert args.restart_after is None or (args.mode == 'gpu_replay' and not args.disable_thumb_index_contact)
    assert args.switch_iterations_after is None or (args.mode=='gpu_replay' and args.position_iterations in (8,128)
        and args.restart_after is None and not args.disable_thumb_index_contact)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    source = json.loads(args.source.read_text()) if args.source else None

    def create_scene(config_path, device, headless, overrides):
        cfg = merge_cfg_with_default_config(read_cfg_omegaconf(config_path, device))
        cfg.task.sim.physx.use_gpu = True
        overrides = dict(overrides)
        if args.match_released_velocity_noise:
            overrides['task.env.jointVelocityObsNoiseStd'] = 0.01
        if args.position_iterations is not None and args.switch_iterations_after is None:
            overrides['task.sim.physx.num_position_iterations'] = args.position_iterations
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
            params = env.gym.get_sim_params(env.sim)
            runtime_solver = dict(dt=params.dt, substeps=params.substeps,
                use_gpu_pipeline=params.use_gpu_pipeline, gpu_physx=params.physx.use_gpu,
                solver_type=params.physx.solver_type,
                position_iterations=params.physx.num_position_iterations,
                velocity_iterations=params.physx.num_velocity_iterations,
                contact_offset=params.physx.contact_offset, rest_offset=params.physx.rest_offset)
            assert params.physx.num_position_iterations == config['sim']['physx']['num_position_iterations']
            metadata = dict(mode=args.mode, properties=properties, config=config,
                runtime_solver=runtime_solver,
                initial=initial, dt=env.sim_params.dt, substeps=env.sim_params.substeps,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scene_helper_sha256=hashlib.sha256(Path(__file__).with_name('check_gym_pipeline_replay.py').read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest(),
                config_path=str(self.config_path.resolve()), checkpoint_path=str(self.checkpoint_path.resolve()),
                source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest() if args.source else None,
                external_wrenches='Zero; official evaluation disables all force and velocity impulse scales.',
                scope='Specified checkpoint on short_screwdriver/spin_horizontal, up to600policy steps plus2warmups. Seed42, no GUI/video/pacing. Fresh-process command replay, not benchmark success or physical acceptance.')
            if source:
                assert properties == source['properties']
                assert metadata['config_sha256'] == source['config_sha256']
                assert metadata['checkpoint_sha256'] == source['checkpoint_sha256']
                if args.disable_thumb_index_contact:
                    metadata['contact_intervention'] = disable_thumb_index_contacts(env)
                    modified = scene_properties(env)
                    expected = json.loads(json.dumps(properties))
                    for row, mask in zip(expected[0]['shapes'], metadata['contact_intervention']['after_filters']):
                        row[2] = mask
                    assert modified == expected
                    metadata['modified_properties'] = modified
                requested_initial = source['initial'] if args.restart_after is None else source['trace'][args.restart_after]
                replay_rows = source['trace'] if args.restart_after is None else source['trace'][args.restart_after+1:args.restart_after+33]
                if args.switch_iterations_after is not None:
                    replay_rows = source['trace'][:args.switch_iterations_after+33]
                if args.restart_after is not None:
                    assert len(replay_rows) == 32 and requested_initial['step'] == args.restart_after
                    metadata['restart_after'] = args.restart_after
                    metadata['restart_scope'] = 'Restore saved root/DOF states and previous PD target. Solver/contact history is not restored; compare8-iteration restart with uninterrupted source before equivalence claims.'
                    previous_target = next(c['value'] for c in requested_initial['commands'] if c['name']=='set_dof_position_target_tensor')
                    initial_target = torch.tensor(previous_target,dtype=torch.float32,device=env.device)
                    assert env.gym.set_dof_position_target_tensor(env.sim,gymtorch.unwrap_tensor(initial_target))
                    metadata['initial_pd_target'] = previous_target
                root = torch.tensor(requested_initial['root'], dtype=torch.float32, device=env.device)
                dof = torch.tensor(requested_initial['dof'], dtype=torch.float32, device=env.device)
                assert env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(root))
                assert env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(dof))
                actual = snapshot(env)
                requested_root, actual_root = root.cpu().double(), torch.tensor(actual['root'], dtype=torch.float64)
                qa, qb = requested_root[:, 3:7], actual_root[:, 3:7]
                na, nb = qa.norm(dim=-1, keepdim=True), qb.norm(dim=-1, keepdim=True)
                assert (na - 1).abs().max() < 1e-6 and (nb - 1).abs().max() < 1e-6
                qa, qb = qa / na, qb / nb
                angle = 4 * torch.asin(torch.minimum((qa-qb).norm(dim=-1), (qa+qb).norm(dim=-1)).clamp(max=2) / 2)
                error = (requested_root - actual_root).abs()
                initial_errors = dict(position_m=float(error[:, :3].max()), rotation_rad=float(angle.max()),
                                      quaternion_component=float(error[:, 3:7].max()), velocity_component=float(error[:, 7:].max()))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.with_suffix('.initial.json').write_text(json.dumps(dict(
                    requested=requested_initial, actual=actual, source_sha256=metadata['source_sha256'],
                    script_sha256=metadata['script_sha256'], errors=initial_errors)) + '\n')
                assert initial_errors['position_m'] < 1e-7 and initial_errors['rotation_rad'] < 1e-6
                assert initial_errors['velocity_component'] == 0
                assert actual['dof'] == requested_initial['dof']
                if args.restart_after is not None:
                    metadata['initial_body_position_component_error_m'] = float(np.max(np.abs(
                        np.asarray(actual['bodies'])[:, :3] - np.asarray(requested_initial['bodies'])[:, :3])))
                metadata['initial'] = actual
                metadata['initial_readback_errors'] = initial_errors
                rows, deep_streak = [], 0
                for recorded in replay_rows:
                    if args.switch_iterations_after is not None and recorded['step']==args.switch_iterations_after+1:
                        before = snapshot(env)
                        original = source['trace'][args.switch_iterations_after]
                        assert all(before[k]==original[k] for k in ('root','dof','bodies'))
                        params = env.gym.get_sim_params(env.sim)
                        params.physx.num_position_iterations = args.position_iterations
                        env.gym.set_sim_params(env.sim,params)
                        readback = env.gym.get_sim_params(env.sim)
                        switch = dict(after_step=args.switch_iterations_after,
                            requested_position_iterations=args.position_iterations,
                            readback_position_iterations=readback.physx.num_position_iterations,
                            states_before_exact=True, states_after_exact=snapshot(env)==before)
                        metadata['live_solver_switch'] = switch
                        args.output.parent.mkdir(parents=True,exist_ok=True)
                        args.output.with_suffix('.switch.json').write_text(json.dumps(switch)+'\n')
                        assert switch['states_after_exact']
                        assert readback.physx.num_position_iterations==args.position_iterations, switch
                    retained = []
                    for command in recorded['commands']:
                        value = torch.tensor(command['value'], dtype=torch.float32, device=env.device)
                        retained.append(value)
                        method = getattr(env.gym, command['name'])
                        if 'indices' in command:
                            indices = torch.tensor(command['indices'], dtype=torch.int32, device=env.device)
                            retained.append(indices)
                            applied = method(env.sim, gymtorch.unwrap_tensor(value), gymtorch.unwrap_tensor(indices), command['count'])
                            assert applied or command['count'] == 0, (recorded['step'], command['name'], command['count'])
                        else:
                            assert method(env.sim, gymtorch.unwrap_tensor(value))
                    env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
                    row = dict(step=recorded['step'], **snapshot(env))
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
                actions, observations, goal_counts = [], [], []
                done = False
                for _ in range(600):
                    if any(joint_excess(row, env) > .05 for row in recorder.rows[-2:]):
                        metadata['physical_review_required'] = 'Joint excess>0.05rad'
                        break
                    observations.append(array(obs['obs']))
                    action = player.get_normalized_action(obs['obs'], deterministic_actions=True)
                    assert torch.isfinite(action).all()
                    actions.append(array(action))
                    obs, _, ended, _ = env.step(action)
                    goal_counts.append(int(env.successes[0].item()))
                    done = bool(ended[0])
                    if done:
                        break
                assert not recorder.pending
                metadata.update(trace=recorder.rows, steps=len(recorder.rows), actions=actions, observations=observations,
                    goal_counts=goal_counts, total_goal_waypoints=env.max_consecutive_successes,
                    policy_steps=len(actions), episode_done=done,
                    goal_pct=100 * int(env.successes[0].item()) / env.max_consecutive_successes)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metadata, allow_nan=False) + '\n')
            env.gym.destroy_sim(env.sim)
            print('POLICY_REPLAY_COMPLETE', args.mode, metadata['steps'], flush=True)

    evaluator.create_env = create_scene
    evaluator.EvalRunner = Runner
    evaluator.tyro.cli = lambda _: evaluator.EvalArgs(object_category='screwdriver', object_name='short_screwdriver',
        task_name='spin_horizontal', config_path=args.config_path,
        checkpoint_path=args.checkpoint_path)
    evaluator.main()


if __name__ == '__main__':
    main()
