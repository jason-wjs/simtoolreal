"""Original tool pool in three native mjlab topology groups, one policy batch."""
from functools import partial
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.entity import EntityCfg, VariantEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from mjlabenvs.assets import object_spec
from mjlabenvs.robot import robot_entity
from simtoolreal_shared.robot_constants import (
    FINGERTIP_LINK_NAMES, JOINT_NAMES_CANONICAL, PALM_BODY_NAME,
)

PHYSICS_DT = 0.00833333333
DECIMATION = 2
# MuJoCo integration state for this fixed model (no plugins or history sensors).
INTEGRATION_FIELDS = ('time', 'qpos', 'qvel', 'act', 'history', 'qacc_warmstart',
                      'ctrl', 'qfrc_applied', 'xfrc_applied', 'eq_active',
                      'mocap_pos', 'mocap_quat', 'userdata')


def _group_physics(group):
    # These runtime fields distinguish the contact/limit variants actually
    # exercised by the reproduction probes; asset hashes do not capture them.
    model = group.sim.wp_model
    return dict(timestep=wp.to_torch(model.opt.timestep),
                geom_solref=wp.to_torch(model.geom_solref),
                jnt_solref=wp.to_torch(model.jnt_solref))


def table_spec():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='table', mocap=True, pos=[0, 0, .38])
    body.add_geom(name='table_box', type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[.475/2, .4/2, .3/2], friction=[.5, 0, 0], condim=3)
    spec.worldbody.add_geom(name='ground', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .1], friction=[.5, 0, 0], condim=3)
    return spec


def goal_spec():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name='goal', mocap=True)
    return spec


def topology(path):
    return tuple('capsule' if c.find('geometry/cylinder') is not None else list(c.find('geometry'))[0].tag
                 for c in ET.parse(path).getroot().findall('link/collision'))


class _Group:
    def __init__(self, robot_assets, tool_root, records, pool_indices, assignment, device):
        local_variant = {pool_id: i for i, pool_id in enumerate(pool_indices)}
        variants = {f'tool_{i}': partial(object_spec, tool_root / records[i]['path'])
                    for i in pool_indices}
        variant_assignment = [local_variant[i] for i in assignment]
        self.scene = Scene(SceneCfg(num_envs=len(assignment), env_spacing=0, entities={
            'robot': robot_entity(robot_assets),
            'object': VariantEntityCfg(variants=variants,
                assignment=lambda count: variant_assignment,
                init_state=EntityCfg.InitialStateCfg(pos=(0, 0, .63))),
            'table': EntityCfg(spec_fn=table_spec),
            'goal': EntityCfg(spec_fn=goal_spec),
        }), device)
        self.sim = Simulation(len(assignment), SimulationCfg(nconmax=256, njmax=1024,
            mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=20, ls_iterations=50,
                            cone='elliptic')), device=device, spec=self.scene.spec,
            variant_info=self.scene.collect_variant_info())
        self.primitive_geometry_audit = self._write_primitive_variants(
            tool_root, records, pool_indices, variant_assignment)
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot, self.object = self.scene['robot'], self.scene['object']
        self.table, self.goal = self.scene['table'], self.scene['goal']
        self.joint_ids, _ = self.robot.find_joints(JOINT_NAMES_CANONICAL, preserve_order=True)
        self.tip_ids, _ = self.robot.find_bodies(FINGERTIP_LINK_NAMES, preserve_order=True)
        self.palm_id = self.robot.find_bodies(PALM_BODY_NAME)[0][0]

    def _write_primitive_variants(self, tool_root, records, pool_indices, assignment):
        # mjlab v1.6's variant builder scatters geometry fields only for mesh
        # slots. Its per-world inertias already vary, but primitives otherwise
        # retain the first variant's dimensions and local pose. These arrays
        # are already per-world: write in place to preserve captured pointers.
        models = [object_spec(tool_root / records[i]['path']).compile() for i in pool_indices]
        gids = [self.sim.mj_model.geom(f'object/collision_{i}').id for i in range(models[0].ngeom)]
        fields = ('geom_size', 'geom_rbound', 'geom_aabb', 'geom_pos', 'geom_quat')
        indices = np.asarray(assignment)
        report = {'worlds': len(assignment), 'variants': len(models), 'before_max_absolute_error': {}}
        for name in fields:
            values = np.stack([getattr(model, name) for model in models])[indices].astype(np.float32)
            if name == 'geom_aabb':
                values = values.reshape(len(assignment), len(gids), 2, 3)
            expected = torch.as_tensor(values, device=self.sim.device)
            target = getattr(self.sim.model, name)
            report['before_max_absolute_error'][name] = float((target[:, gids] - expected).abs().max())
            target[:, gids] = expected
            self.sim.default_model_fields[name][:, gids] = expected
            torch.testing.assert_close(target[:, gids], expected, atol=0, rtol=0)
        bid = self.sim.mj_model.body('object/object_root').id
        for name in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_invweight0'):
            expected = torch.as_tensor(np.stack([getattr(model, name)[model.body('object_root').id]
                for model in models])[indices].astype(np.float32), device=self.sim.device)
            torch.testing.assert_close(getattr(self.sim.model, name)[:, bid], expected, atol=0, rtol=0)
        report['actual_geometry_and_inertias_passed'] = True
        return report

    def state(self):
        data = self.robot.data
        return {
            'joint_pos': data.joint_pos[:, self.joint_ids],
            'joint_vel': data.joint_vel[:, self.joint_ids],
            'object_pose': self.object.data.root_link_pose_w,
            'object_velocity': self.object.data.root_com_vel_w,
            'goal_pose': self.goal.data.root_link_pose_w,
            'table_pose': self.table.data.root_link_pose_w,
            'palm_pose': data.body_link_pose_w[:, self.palm_id],
            'palm_velocity': data.body_com_vel_w[:, self.palm_id],
            'fingertip_pose': data.body_link_pose_w[:, self.tip_ids],
        }


class SimToolRealScene:
    """Keep global env-i→tool-i%1200 while simulating exact primitive topologies."""
    def __init__(self, robot_assets: Path, tool_root: Path, num_envs: int, device: str):
        assert num_envs >= 1200, 'Native pool coverage requires at least 1200 environments'
        self.num_envs, self.device = num_envs, device
        self.asset_identity = {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in [('tools', tool_root / 'manifest.json'),
                               ('robot_xml', robot_assets / 'robot.xml'),
                               ('robot_urdf', robot_assets / 'robot.urdf')]}
        records = json.loads((tool_root / 'manifest.json').read_text())['tools']
        assert len(records) == 1200
        kinds = [topology(tool_root / r['path']) for r in records]
        self.assignment = torch.arange(num_envs, device=device) % len(records)
        self.object_scales = torch.tensor([records[i % 1200]['scale'] for i in range(num_envs)],
                                         dtype=torch.float32, device=device)
        self.groups = []
        for kind in dict.fromkeys(kinds):
            pool_ids = [i for i, k in enumerate(kinds) if k == kind]
            global_ids = [i for i in range(num_envs) if kinds[i % 1200] == kind]
            group = _Group(robot_assets, tool_root, records, pool_ids,
                           [i % 1200 for i in global_ids], device)
            ids = torch.tensor(global_ids, device=device, dtype=torch.long)
            self.groups.append((ids, group))
        assert len(self.groups) == 3
        group = self.groups[0][1]
        self.default_joint_pos = group.robot.data.default_joint_pos[0, group.joint_ids].clone()
        self.joint_limits = group.robot.data.joint_pos_limits[0, group.joint_ids].clone()
        self.object_mass = torch.empty(num_envs, device=device)
        for ids, group in self.groups:
            bid = group.sim.mj_model.body('object/object_root').id
            self.object_mass[ids] = group.sim.model.body_mass[:, bid]

    def reset(self, env_ids, joint_pos, joint_vel, object_state, table_pose, goal_pose):
        # Inputs are indexed by the supplied global env ids, not topology order.
        lookup = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        lookup[env_ids] = torch.arange(env_ids.numel(), device=self.device)
        for ids, group in self.groups:
            source = lookup[ids]
            local = (source >= 0).nonzero(as_tuple=True)[0]
            if not local.numel():
                continue
            source = source[local]
            group.sim.reset(local)
            group.scene.reset(local)
            group.robot.write_joint_state_to_sim(joint_pos[source], joint_vel[source],
                                                 joint_ids=group.joint_ids, env_ids=local)
            group.robot.set_joint_position_target(joint_pos[source], joint_ids=group.joint_ids,
                                                    env_ids=local[:, None])
            group.object.write_root_state_to_sim(object_state[source], env_ids=local)
            group.table.write_mocap_pose_to_sim(table_pose[source], env_ids=local)
            group.goal.write_mocap_pose_to_sim(goal_pose[source], env_ids=local)
            group.scene.write_data_to_sim()
            group.sim.forward()
            group.scene.update(0)

    def write_goals(self, env_ids, poses):
        lookup = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        lookup[env_ids] = torch.arange(env_ids.numel(), device=self.device)
        for ids, group in self.groups:
            source = lookup[ids]
            local = (source >= 0).nonzero(as_tuple=True)[0]
            if local.numel():
                group.goal.write_mocap_pose_to_sim(poses[source[local]], env_ids=local)
                group.sim.forward()

    def step(self, targets, forces, torques):
        for ids, group in self.groups:
            group.robot.set_joint_position_target(targets[ids], joint_ids=group.joint_ids)
            group.object.write_external_wrench_to_sim(forces[ids], torques[ids])
            for _ in range(DECIMATION):
                group.scene.write_data_to_sim()
                group.sim.step()
                group.scene.update(PHYSICS_DT)
            group.sim.forward()

    def state(self):
        result = {}
        for ids, group in self.groups:
            for name, values in group.state().items():
                if name not in result:
                    result[name] = torch.empty((self.num_envs, *values.shape[1:]),
                                              dtype=values.dtype, device=self.device)
                result[name][ids] = values
        return result

    def get_state(self):
        return dict(assets=self.asset_identity,
            physics=dict(physics_dt=PHYSICS_DT, decimation=DECIMATION),
            groups=[dict(ids=ids.clone(),
            physics={name: value.clone() for name, value in _group_physics(group).items()},
            integration={name: getattr(group.sim.data, name).clone() for name in INTEGRATION_FIELDS})
            for ids, group in self.groups])

    def validate_state(self, state):
        """Reject a different physics recipe before mutating any task state."""
        assert state['assets'] == self.asset_identity
        assert len(state['groups']) == len(self.groups)
        if 'physics' not in state:
            raise ValueError('Checkpoint lacks runtime physics identity; use a newly captured checkpoint')
        if state['physics'] != dict(physics_dt=PHYSICS_DT, decimation=DECIMATION):
            raise ValueError('Checkpoint scene timestep/decimation mismatch')
        for index, ((ids, group), saved) in enumerate(zip(self.groups, state['groups'])):
            assert torch.equal(ids, saved['ids'].to(self.device))
            for name, value in _group_physics(group).items():
                if not torch.equal(value, saved['physics'][name].to(value.device)):
                    raise ValueError(f'Checkpoint physics mismatch: group {index} {name}')

    def set_state(self, state, targets, forces, torques):
        self.validate_state(state)
        for (ids, group), saved in zip(self.groups, state['groups']):
            assert torch.equal(ids, saved['ids'].to(self.device))
            group.sim.reset()
            group.scene.reset()
            group.robot.set_joint_position_target(targets[ids], joint_ids=group.joint_ids)
            group.object.write_external_wrench_to_sim(forces[ids], torques[ids])
            group.scene.write_data_to_sim()
            for name in INTEGRATION_FIELDS:
                getattr(group.sim.data, name).copy_(saved['integration'][name].to(self.device))
            # Rebuild derived kinematics and contacts from the restored inputs.
            # Preserve warmstart after forward, which may run the solver.
            group.sim.forward()
            group.sim.data.qacc_warmstart.copy_(saved['integration']['qacc_warmstart'].to(self.device))
            group.scene.update(0)
