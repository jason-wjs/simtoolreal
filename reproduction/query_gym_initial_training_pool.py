"""Full original reset-pool native contact audit; no task or policy rollout."""
from isaacgym import gymapi, gymtorch
import argparse
import hashlib
import json
from pathlib import Path
import torch
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from deployment.isaac.isaac_env import create_env_from_cfg, merge_cfg_with_default_config
from deployment.rl_player_utils import read_cfg_omegaconf
from check_gym_saved_pose_contacts import pose_error
from profile_gym_training import static_scene_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    trace = json.loads((source / 'native/initial_training_trace.json').read_text())
    assert trace['num_envs'] == 24576 and len(trace['asset_sha256']) == 1200
    assets = [source / 'native/trace_assets' / (str(i) + '.urdf') for i in range(1200)]
    for path, expected in zip(assets, trace['asset_sha256']):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    SimToolReal._main_object_assets_and_scales = lambda self, object_asset_root, tmp_assets_dir: (assets, [(1., 1., 1.)] * 1200, [False] * 1200)
    cfg = merge_cfg_with_default_config(read_cfg_omegaconf(str(source / 'native/.hydra/config.yaml'), 'cuda:0'))
    cfg.task.sim.physx.use_gpu = True
    env = create_env_from_cfg(cfg, headless=True, overrides={
        'pipeline': 'cpu', 'graphics_device_id': -1, 'task.env.numEnvs': 24576,
        'task.sim.dt': 1e-6, 'task.sim.substeps': 1, 'task.sim.physx.contact_collection': 1})
    report = dict(status='running', num_envs=env.num_envs, calibration=[],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes={}, readbacks=[], scope='Saved-pose native initialOverlap queries at all original24576 reset tuples; same1200 assets/all-world properties/grid, CPU tensors plus GPU PhysX. No policy or training. Two saved worlds calibrate body reconstruction; no full-pool original GPU body-pose capture exists.')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def load(name, label):
        path = source / name / (label + '.pt')
        report['source_hashes'][name + '/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return torch.load(path, map_location='cpu')

    def install(state):
        roots, dofs = state['native_root'].clone(), state['native_dof'].clone()
        roots[:, 7:] = 0.; dofs[:, 1] = 0.
        targets = dofs[:, 0].contiguous()
        assert env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(roots))
        assert env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(dofs))
        assert env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
        for kind in ('actor_root', 'dof', 'rigid_body'):
            getattr(env.gym, 'refresh_' + kind + '_state_tensor')(env.sim)
        path = args.output.parent / ('readback_' + str(len(report['readbacks'])) + '.pt')
        torch.save(dict(requested_root=roots, actual_root=env.root_state_tensor.clone(),
            requested_dof=dofs, actual_dof=env.dof_state.clone()), path)
        delta = (env.root_state_tensor - roots).abs()
        worst = int(delta.argmax())
        report['readbacks'].append(dict(file=path.name, max_root_error=float(delta.max()),
            worst_actor=worst // 13, worst_component=worst % 13,
            max_dof_error=float((env.dof_state - dofs).abs().max())))
        assert torch.equal(env.dof_state, dofs)
        assert torch.allclose(env.root_state_tensor, roots, atol=1e-6, rtol=0)
        return roots, dofs, targets

    try:
        assert env.device == 'cpu' and env.num_envs == 24576
        assert static_scene_digest(env) == trace['native_scene_digest']
        report['native_scene_digest_exact'] = True
        # Same original scene/grid. Calibrate against the two actually retained
        # post-integration GPU body poses; pre-reset body caches are stale.
        for name in ('native', 'candidate'):
            state = load(name, 'after_env_step')
            installed = install(state)
            actual = env.rigid_body_states[trace['selected_worlds']].tolist()
            expected = state['selected_bodies'].tolist()
            errors = [pose_error(a, b) for a, b in zip(actual, expected)]
            report['calibration'].append(dict(source=name, errors=errors,
                actual_body_poses=actual, expected_body_poses=expected))
            assert all(e['position_m'] <= 1e-5 and e['rotation_rad'] <= 1e-4 for e in errors), errors
            env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
        state = load('native', 'pre_integrate')
        roots, dofs, targets = install(state)
        env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
        rows = []
        for i, instance in enumerate(env.envs):
            contacts = [{k: int(c[k]) if k in ('body0', 'body1') else float(c[k])
                         for k in ('body0', 'body1', 'initialOverlap', 'minDist')}
                        for c in env.gym.get_env_rigid_contacts(instance)]
            assert all(-1 <= c[k] < 33 for c in contacts for k in ('body0', 'body1'))
            rows.append(dict(world=i, asset_index=i % 1200, contacts=contacts))
        report['per_world'] = rows
        clear = roots.clone()
        clear[state['object_indices'].long(), 2] += 3.
        assert env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(clear))
        assert env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(dofs))
        assert env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
        env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
        counts = [sum(30 in (int(c['body0']), int(c['body1']))
                      for c in env.gym.get_env_rigid_contacts(instance)) for instance in env.envs]
        report['clear_object_control_counts'] = counts
        assert not any(counts)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='stopped', error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
