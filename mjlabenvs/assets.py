"""Import native Lab tool collisions and explicit URDF inertias."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def origin(element):
    node = element.find('origin')
    if node is None:
        return np.zeros(3), np.eye(3)
    position = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
    rotation = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return position, rotation


def object_spec(path: Path, friction: float = 0.5):
    root = ET.parse(path).getroot()
    links = root.findall('link')
    assert len(links) == 1 and not root.findall('joint'), path
    link = links[0]
    inertial = link.find('inertial')
    center, rotation = origin(inertial)
    values = inertial.find('inertia').attrib
    inertia = np.array([[float(values['ixx']), float(values['ixy']), float(values['ixz'])],
                        [float(values['ixy']), float(values['iyy']), float(values['iyz'])],
                        [float(values['ixz']), float(values['iyz']), float(values['izz'])]])
    inertia = rotation @ inertia @ rotation.T
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='object_root', explicitinertial=True,
        mass=float(inertial.find('mass').get('value')), ipos=center,
        fullinertia=[inertia[0, 0], inertia[1, 1], inertia[2, 2],
                     inertia[0, 1], inertia[0, 2], inertia[1, 2]])
    body.add_freejoint(name='object_free')
    for index, collision in enumerate(link.findall('collision')):
        position, rotation = origin(collision)
        quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        geometry = list(collision.find('geometry'))
        assert len(geometry) == 1
        primitive = geometry[0]
        if primitive.tag == 'box':
            kind = mujoco.mjtGeom.mjGEOM_BOX
            size = np.fromstring(primitive.get('size'), sep=' ') / 2
        elif primitive.tag == 'cylinder':
            # Lab's URDF converter replaces collision cylinders by capsules.
            # USD height is the cylindrical spine, excluding hemispheres.
            kind = mujoco.mjtGeom.mjGEOM_CAPSULE
            size = [float(primitive.get('radius')), float(primitive.get('length')) / 2, 0]
        else:
            raise ValueError(f'Unexpected original tool primitive: {primitive.tag}')
        body.add_geom(name=f'collision_{index}', type=kind, pos=position, quat=quat, size=size,
                      mass=0, friction=[friction, 0, 0], condim=3, rgba=[0.55, 0.27, 0.07, 1])
    return spec


def audit_object(path: Path):
    """Check source inertias and Lab's cylinder-to-capsule collision import."""
    link = ET.parse(path).getroot().find('link')
    inertial = link.find('inertial')
    model = object_spec(path).compile()
    body = model.body('object_root')
    center, rotation = origin(inertial)
    values = inertial.find('inertia').attrib
    source = np.array([[float(values['ixx']), float(values['ixy']), float(values['ixz'])],
                       [float(values['ixy']), float(values['iyy']), float(values['iyz'])],
                       [float(values['ixz']), float(values['iyz']), float(values['izz'])]])
    source = rotation @ source @ rotation.T
    compiled_rotation = Rotation.from_quat(body.iquat[[1, 2, 3, 0]]).as_matrix()
    compiled = compiled_rotation @ np.diag(body.inertia) @ compiled_rotation.T
    np.testing.assert_allclose(body.mass, float(inertial.find('mass').get('value')), rtol=1e-12)
    np.testing.assert_allclose(body.ipos, center, atol=1e-12)
    np.testing.assert_allclose(compiled, source, rtol=1e-10, atol=1e-12)
    collisions = link.findall('collision')
    assert model.ngeom == len(collisions) and model.nq == 7 and model.nv == 6
    for index, collision in enumerate(collisions):
        position, rotation = origin(collision)
        np.testing.assert_allclose(model.geom_pos[index], position, atol=1e-12)
        actual_rotation = Rotation.from_quat(model.geom_quat[index][[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(actual_rotation, rotation, atol=1e-12)
        primitive = list(collision.find('geometry'))[0]
        if primitive.tag == 'box':
            assert model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX
            np.testing.assert_allclose(model.geom_size[index], np.fromstring(primitive.get('size'), sep=' ')/2)
        else:
            assert model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CAPSULE
            np.testing.assert_allclose(model.geom_size[index, :2], [float(primitive.get('radius')), float(primitive.get('length'))/2])
    return {'mass': float(body.mass[0]), 'collision_count': len(collisions),
            'collision_cylinders_replaced_by_capsules': sum(c.find('geometry/cylinder') is not None for c in collisions),
            'max_inertia_error': float(np.abs(compiled-source).max()), 'passed': True}
