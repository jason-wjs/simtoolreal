"""Two native Lab physics steps from one recorded mjlab reset; no learner."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--tools', type=Path, required=True)
parser.add_argument('--world', type=int, choices=[461, 42, 728, 969, 127, 1475, 1330, 0], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--clear-impact', action='store_true', help='Replay the saved13-step clear-start CPU target sequence.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import numpy as np
import torch
from isaacsim.core.simulation_manager import SimulationManager
from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
from isaacsimenvs.tasks.simtoolreal.utils.audit_utils import AuditedSimToolRealEnv
from simtoolreal_shared.native_state_audit import lab_snapshot
from simtoolreal_shared.robot_constants import JOINT_NAMES_CANONICAL

initial = torch.load(args.run / 'native/initial.pt', map_location='cpu', weights_only=False)
final = torch.load(args.run / 'native/final.pt', map_location='cpu', weights_only=False)
saved = torch.load(args.run / 'native/task_state.pt', map_location='cpu', weights_only=False)
assert not initial['object_velocity'].any()
assert not saved['wrench']['_object_forces'].any() and not saved['wrench']['_object_torques'].any()
world = args.world
record = json.loads((args.tools / 'manifest.json').read_text())['tools'][world % 1200]
cfg = SimToolRealEnvCfg()
cfg.scene.num_envs = 1
cfg.assets.object_urdf = str((args.tools / record['path']).resolve())
cfg.assets.object_scale = tuple(record['scale'])
cfg.domain_randomization.force_scale = cfg.domain_randomization.torque_scale = 0.
assert cfg.decimation == 2 and abs(cfg.sim.dt - .00833333333) < 1e-10
args.output.mkdir(parents=True, exist_ok=False)
env = AuditedSimToolRealEnv(cfg)
env.reset()

def tensor(value):
    return value.unsqueeze(0).to(env.device)

perm = env._perm_canon_to_lab
q = tensor(initial['joint_pos'][world])[:, perm]
qd = tensor(initial['joint_vel'][world])[:, perm]
targets = tensor(saved['control']['_cur_targets'][world])[:, perm]
target_sequence = saved['control']['_cur_targets'][world].unsqueeze(0)
if args.clear_impact:
    cpu_path = args.run / 'clear_reset_dynamics_cpu.json'
    cpu_report = json.loads(cpu_path.read_text())
    for name, digest in cpu_report['source_hashes'].items():
        assert digest == hashlib.sha256((args.run / 'native' / name).read_bytes()).hexdigest()
    case = next(c for c in cpu_report['cases'] if c['world'] == world and c['mode'] == 'native120hz')
    target_sequence = torch.tensor(case['targets_canonical'], dtype=torch.float32)
    assert target_sequence.shape == (13, 29)
    torch.testing.assert_close(target_sequence[0], saved['control']['_cur_targets'][world], atol=1e-6, rtol=0)
env.robot.write_joint_state_to_sim(q, qd)
for label, asset in [('object', env.object), ('table', env.table), ('goal', env.goal_viz)]:
    state = torch.zeros((1, 13), device=env.device)
    state[:, :7] = tensor(initial[label + '_pose'][world])
    asset.write_root_state_to_sim(state)
env.robot.set_joint_position_target(targets)
env.scene.write_data_to_sim()
SimulationManager.get_physics_sim_view().update_articulations_kinematic()
env.scene.update(0)
# Read native tensors rather than trusting only the asset's write-through cache.
view = env.robot.root_physx_view
np.testing.assert_allclose(view.get_dof_positions().cpu(), q.cpu(), atol=1e-6, rtol=0)
np.testing.assert_allclose(view.get_dof_velocities().cpu(), qd.cpu(), atol=1e-6, rtol=0)
for label, asset in [('object', env.object), ('table', env.table), ('goal', env.goal_viz)]:
    native = asset.root_physx_view.get_transforms().cpu().numpy()[0]
    expected = initial[label + '_pose'][world].numpy()
    np.testing.assert_allclose(native[:3], expected[:3], atol=1e-6, rtol=0)
    assert abs(float(np.dot(native[[6, 3, 4, 5]], expected[3:]))) > 1 - 1e-6
palm_error = float((env.robot.data.body_link_pose_w[0, env._palm_body_id, :3].cpu()
                    - initial['palm_pose'][world, :3]).abs().max())
assert palm_error < 1e-5, palm_error
env._cur_targets[:] = targets
before = lab_snapshot(env, torch.zeros(1, cfg.observation_space, device=env.device),
                      torch.zeros(1, 29, device=env.device))
(args.output / 'initial_native.json').write_text(json.dumps(before, indent=2) + '\n')
# Runtime material tensors and resolved USD bindings are different evidence:
# the tensor API exposes coefficients but not native friction-combine enums.
from pxr import UsdPhysics, UsdShade
import omni.usd
stage = omni.usd.get_context().get_stage()
settings = dict(material_tensors={}, collision_bindings=[], physics_attributes=[])
for label, asset in [('robot', env.robot), ('object', env.object), ('table', env.table)]:
    settings['material_tensors'][label] = asset.root_physx_view.get_material_properties().cpu().tolist()
for prim in stage.Traverse():
    if not str(prim.GetPath()).startswith('/World'):
        continue
    attrs = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
             if a.GetName().startswith(('physx', 'physics:')) and a.Get() is not None}
    if attrs:
        settings['physics_attributes'].append(dict(path=str(prim.GetPath()),
            schemas=list(prim.GetAppliedSchemas()), values=attrs))
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
        settings['collision_bindings'].append(dict(path=str(prim.GetPath()),
            material_path=str(material.GetPath()) if material else None,
            relationship_path=str(relationship.GetPath()) if relationship else None))
settings['scope'] = ('Read-only native per-shape static/dynamic/restitution tensors and runtime '
    'USD resolved physics bindings/schema attributes. Native tensor material overrides '
    'need not appear in USD. No direct native combine-mode getter or friction-force identification.')
(args.output / 'runtime_contact_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
env.audit_control_step = 0
env.audit_action = np.zeros(29, dtype=np.float32)
physics_steps = 0
stop_reason = None
deep_streak = 0
for control_step, canonical_targets in enumerate(target_sequence):
    env.audit_control_step = control_step
    env.robot.set_joint_position_target(tensor(canonical_targets)[:, perm])
    for substep in [1, 2]:
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(cfg.sim.dt)
        env.audit_substep = substep
        env._capture_audit_substep()
        physics_steps += 1
        contacts = env.audit_contacts[-1]
        depth = float(np.maximum(-contacts[:, -1], 0).max()) if len(contacts) else 0.
        deep_streak = deep_streak + 1 if depth > .005 else 0
        q_actual = env.audit_rows[-1]['joint_pos']
        limit_excess = np.maximum(env.audit_joint_limits[:, 0] - q_actual,
                                  q_actual - env.audit_joint_limits[:, 1]).clip(min=0).max()
        if args.clear_impact and (deep_streak >= 12 or limit_excess > .05):
            stop_reason = 'physical_review_required'
            break
    if stop_reason:
        break
env.write_audit(args.output)
actual = env.robot.data.joint_pos[0, env._perm_lab_to_canon].cpu()
limits = env.robot.root_physx_view.get_dof_limits()[0, env._perm_lab_to_canon.cpu()].cpu()
violation = torch.maximum(limits[:, 0] - actual, actual - limits[:, 1]).clamp_min(0)
gpu = final['joint_pos'][world]
gpu_violation = torch.maximum(limits[:, 0] - gpu, gpu - limits[:, 1]).clamp_min(0)
report = dict(world=world, tool=record, physics_steps=physics_steps, physics_dt=cfg.sim.dt,
    planned_physics_steps=len(target_sequence)*2, stop_reason=stop_reason,
    initial_palm_max_abs_error_m=palm_error, native_initial_state_matches=True,
    canonical_joint_names=JOINT_NAMES_CANONICAL, targets_canonical=targets[0, env._perm_lab_to_canon].cpu().tolist(),
    lab_final_joint_pos=actual.tolist(), mjlab_final_joint_pos=gpu.tolist(),
    lab_max_joint_violation_rad=float(violation.max()),
    mjlab_max_joint_violation_against_lab_limits_rad=float(gpu_violation.max()),
    lab_final_object_state=env.object.data.root_state_w[0].cpu().tolist(),
    input_hashes={name: hashlib.sha256((args.run / 'native' / name).read_bytes()).hexdigest()
                  for name in ['initial.pt', 'final.pt', 'task_state.pt']},
    scope='Matched initial q/qd, object/table/goal link poses and held actuator targets; '
          'native Lab assets/PD/solver versus saved mjlab first-step output. '
          'No policy, task reward, resampling, automatic reset or solver tuning. '
          'Native contact audit covers object/robot and object/table only. '
          'Backend-specific physical parameters are retained and require separate alignment audit.')
if args.clear_impact:
    report['target_sequence_canonical'] = target_sequence.tolist()
    report['target_sequence_report_sha256'] = hashlib.sha256(cpu_path.read_bytes()).hexdigest()
    report['scope'] = ('Clear-start diagnostic: exact saved initial states and13-step native zero-action target sequence; '
        'native Lab120Hz physics,60Hz control, original assets/drives/solver, no external wrench. '
        'No policy/reward/automatic reset or training; contact recorder covers object/robot and object/table. '
        'Early stop on12 consecutive>5mm contact samples or>0.05rad joint violation; physical acceptance remains separate.')
(args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
# Avoid known headless Kit teardown stalls after all evidence has been flushed.
import os
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
