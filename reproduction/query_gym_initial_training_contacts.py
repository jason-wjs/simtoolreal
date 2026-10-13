"""Native saved-pose queries for the two observed worst training worlds."""
from isaacgym import gymapi, gymtorch
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from deployment.isaac.isaac_env import create_env_from_cfg, merge_cfg_with_default_config
from deployment.rl_player_utils import read_cfg_omegaconf
from check_gym_pipeline_replay import scene_properties
from check_gym_policy_replay import snapshot
from check_gym_saved_pose_contacts import pose_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    trace = json.loads((source / 'native/initial_training_trace.json').read_text())
    worlds = [12965, 15208]
    assert trace['selected_worlds'] == worlds
    assets = [source / 'native/trace_assets' / (str(w % 1200) + '.urdf') for w in worlds]
    for w, path in zip(worlds, assets):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == trace['asset_sha256'][w % 1200]
    # Scale descriptors feed task keypoints/observations, not actor geometry.
    # No task step or policy is run; native body/shape/DOF properties are checked.
    SimToolReal._main_object_assets_and_scales = lambda self, object_asset_root, tmp_assets_dir: (assets, [(1., 1., 1.)] * 2, [False] * 2)
    cfg = merge_cfg_with_default_config(read_cfg_omegaconf(str(source / 'native/.hydra/config.yaml'), 'cuda:0'))
    cfg.task.sim.physx.use_gpu = True
    native = gymapi.acquire_gym()

    class OriginMatchedGym:
        def __init__(self):
            self.created = 0
            self.selected = 0

        def __getattr__(self, name):
            return getattr(native, name)

        def create_env(self, sim, lower, upper, num_per_row):
            # Preserve the original grid indices and bounds to avoid changing
            # float32 world-coordinate precision during local pose calibration.
            original_row = int(math.sqrt(trace['num_envs']))
            while self.created < worlds[self.selected]:
                native.create_env(sim, lower, upper, original_row)
                self.created += 1
            instance = native.create_env(sim, lower, upper, original_row)
            self.created += 1
            self.selected += 1
            return instance

    gymapi.acquire_gym = lambda: OriginMatchedGym()
    env = create_env_from_cfg(cfg, headless=True, overrides={
        'pipeline': 'cpu', 'graphics_device_id': -1, 'task.env.numEnvs': 2,
        'task.sim.dt': 1e-6, 'task.sim.substeps': 1, 'task.sim.physx.contact_collection': 1})
    report = dict(status='running', cases=[], worlds=worlds,
        query_env_origins=[[v.x, v.y, v.z] for v in [env.gym.get_env_origin(e) for e in env.envs]],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Two original training assets/properties, CPU tensor pipeline plus GPU PhysX. Post-integration GPU body poses calibrate reconstruction before common pre-integration root/DOF poses are queried. Tiny-step raw initialOverlap, not a dynamics replay.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        assert env.device == 'cpu' and env.num_envs == 2
        assert [scene_properties(env, i) for i in range(2)] == [trace['selected_properties'][str(w)] for w in worlds]
        report['native_properties_exact'] = True
        for name, label in [('native', 'after_env_step'), ('candidate', 'after_env_step'), ('native', 'pre_integrate')]:
            path = source / name / (label + '.pt')
            state = torch.load(path, map_location='cpu')
            assert state['native_root'].shape == (24576 * 4, 13)
            roots = state['native_root'].view(24576, 4, 13)[worlds].reshape(8, 13).clone()
            dofs = state['native_dof'].view(24576, 29, 2)[worlds].reshape(58, 2).clone()
            roots[:, 7:] = 0.; dofs[:, 1] = 0.
            target = dofs[:, 0].contiguous()
            assert env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(roots))
            assert env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(dofs))
            assert env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(target))
            before = snapshot(env)
            assert torch.equal(torch.tensor(before['dof']), dofs)
            assert torch.allclose(torch.tensor(before['root']), roots, atol=1e-6, rtol=0)
            errors = [pose_error(before['bodies'][i], state['selected_bodies'][i].tolist()) for i in range(2)]
            if label == 'after_env_step':
                report.setdefault('calibration_attempts', []).append(dict(source=name, errors=errors,
                    actual_body_poses=before['bodies'], expected_body_poses=state['selected_bodies'].tolist()))
                assert all(e['position_m'] <= 1e-5 and e['rotation_rad'] <= 1e-4 for e in errors), errors
            env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
            contacts = []
            for instance in env.envs:
                rows = env.gym.get_env_rigid_contacts(instance)
                contacts.append([{k: int(c[k]) if k in ('body0', 'body1') else float(c[k])
                                  for k in ('body0', 'body1', 'initialOverlap', 'minDist')} for c in rows])
            assert all(-1 <= c[k] < 33 for rows in contacts for c in rows for k in ('body0', 'body1'))
            clear = roots.clone(); clear[[1, 5], 2] += 3.
            assert env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(clear))
            assert env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(dofs))
            env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
            counts = [sum(30 in (int(c['body0']), int(c['body1'])) for c in env.gym.get_env_rigid_contacts(instance)) for instance in env.envs]
            assert counts == [0, 0]
            report['cases'].append(dict(source=name, label=label, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_body_errors=errors, contacts=contacts, clear_object_control_counts=counts,
                query_body_poses=before['bodies']))
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='stopped', error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
