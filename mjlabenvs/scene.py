"""Original tool pool in three native mjlab topology groups, one policy batch."""
from functools import partial
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import torch
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
    return tuple(list(c.find('geometry'))[0].tag
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
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot, self.object = self.scene['robot'], self.scene['object']
        self.table, self.goal = self.scene['table'], self.scene['goal']
        self.joint_ids, _ = self.robot.find_joints(JOINT_NAMES_CANONICAL, preserve_order=True)
        self.tip_ids, _ = self.robot.find_bodies(FINGERTIP_LINK_NAMES, preserve_order=True)
        self.palm_id = self.robot.find_bodies(PALM_BODY_NAME)[0][0]

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
            group.table.write_root_link_pose_to_sim(table_pose[source], env_ids=local)
            group.goal.write_root_link_pose_to_sim(goal_pose[source], env_ids=local)
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
                group.goal.write_root_link_pose_to_sim(poses[source[local]], env_ids=local)
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
