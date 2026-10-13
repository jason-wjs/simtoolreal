"""Compare every compiled mjlab tool with its actual native Lab USD export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--lab-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from mjlabenvs.assets import object_spec

    native = json.loads(args.lab_audit.read_text())
    assert native['num_tools'] == 1200
    worst_inertia, worst_geometry = 0., 0.
    for row in native['tools']:
        path = args.tools / row['urdf']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['urdf_sha256']
        model = object_spec(path).compile()
        body = model.body('object_root')
        np.testing.assert_allclose(body.mass[0], row['mass'], rtol=2e-6)
        np.testing.assert_allclose(body.ipos, row['com'], atol=1e-7, rtol=0)
        q = np.array(row['principal_axes_wxyz'])
        r = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        inertia = r @ np.diag(row['diagonal_inertia']) @ r.T
        r = Rotation.from_quat(body.iquat[[1,2,3,0]]).as_matrix()
        actual = r @ np.diag(body.inertia) @ r.T
        np.testing.assert_allclose(actual, inertia, rtol=2e-5, atol=1e-10)
        worst_inertia = max(worst_inertia,float(np.abs(actual-inertia).max()))
        assert model.ngeom == len(row['collisions'])
        for index, geom in enumerate(row['collisions']):
            matrix = np.array(geom['transform_column_convention'])
            scale = np.linalg.norm(matrix[:3,:3], axis=0)
            rotation = matrix[:3,:3] / scale
            compiled = Rotation.from_quat(model.geom_quat[index][[1,2,3,0]]).as_matrix()
            np.testing.assert_allclose(compiled, rotation, atol=1e-6, rtol=0)
            np.testing.assert_allclose(model.geom_pos[index], matrix[:3,3], atol=1e-7, rtol=0)
            if geom['type'] == 'Cube':
                assert model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX
                size = geom['size'] * scale / 2
            else:
                assert geom['type'] == 'Capsule' and geom['axis'] == 'Z'
                assert model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CAPSULE
                np.testing.assert_allclose(scale,1,atol=1e-6)
                size = np.array([geom['radius'],geom['height']/2,0])
            np.testing.assert_allclose(model.geom_size[index],size,atol=1e-7,rtol=0)
            worst_geometry = max(worst_geometry,float(np.abs(model.geom_size[index]-size).max()))
    report = {'passed': True, 'tools': 1200, 'native_collision_types': native['collision_types'],
              'max_inertia_absolute_error': worst_inertia, 'max_geometry_size_error_m': worst_geometry,
              'lab_audit_sha256': hashlib.sha256(args.lab_audit.read_bytes()).hexdigest(),
              'scope': 'Compiled collision types,dimensions,local transforms,mass,COM,full inertia agree with actual baked Lab USD. Float32 USD precision allowed. Does not establish solver/contact behavior equivalence.'}
    with args.output.open('x') as stream:
        stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
