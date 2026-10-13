"""Native mjlab actuator configuration for the fixed Sharpa-left/iiwa14 robot."""
from functools import partial
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from simtoolreal_shared.robot_constants import (
    ARM_DEFAULT_JOINT_POS, ARM_JOINT_DAMPING, ARM_JOINT_STIFFNESS,
    FINGERTIP_LINK_NAMES, HAND_JOINT_ARMATURE, HAND_JOINT_DAMPING,
    HAND_JOINT_STIFFNESS, JOINT_NAMES_CANONICAL,
)


def robot_spec(path: Path):
    spec = mujoco.MjSpec.from_file(str(path))
    root = ET.parse(path.with_suffix('.urdf')).getroot()
    fixed_parents = {j.find('child').get('link'): j.find('parent').get('link')
                     for j in root.findall('joint') if j.get('type') == 'fixed'}
    for body in spec.bodies:
        representative = body.name
        while representative in fixed_parents:
            representative = fixed_parents[representative]
        if body.name != 'world':
            body.gravcomp = 1  # Native Lab disables gravity on robot bodies.
        for geom in body.geoms:
            if geom.contype or geom.conaffinity:
                geom.friction = [1.5 if representative in FINGERTIP_LINK_NAMES else 0.5, 0, 0]
                geom.condim = 3
    return spec


def robot_entity(asset_directory: Path):
    """Consume the audited URDF/MJCF pair; retain canonical joint order."""
    root = ET.parse(asset_directory / 'robot.urdf').getroot()
    limits = {j.get('name'): float(j.find('limit').get('effort'))
              for j in root.findall('joint') if j.get('type') == 'revolute'}
    assert tuple(limits) == JOINT_NAMES_CANONICAL
    stiffness = {**ARM_JOINT_STIFFNESS, **HAND_JOINT_STIFFNESS}
    damping = {**ARM_JOINT_DAMPING, **HAND_JOINT_DAMPING}
    return EntityCfg(
        spec_fn=partial(robot_spec, asset_directory / 'robot.xml'),
        init_state=EntityCfg.InitialStateCfg(pos=(0, 0.8, 0),
            joint_pos={**ARM_DEFAULT_JOINT_POS, **{name: 0 for name in HAND_JOINT_STIFFNESS}}),
        articulation=EntityArticulationInfoCfg(actuators=tuple(
            BuiltinPositionActuatorCfg(target_names_expr=(name,), stiffness=stiffness[name],
                damping=damping[name], effort_limit=limits[name],
                armature=HAND_JOINT_ARMATURE.get(name, 0), frictionloss=0)
            for name in JOINT_NAMES_CANONICAL), soft_joint_pos_limit_factor=1.0))
