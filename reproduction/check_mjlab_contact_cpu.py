"""CPU isolation of tool/table impact; no robot, training or CUDA context."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import mujoco
    import numpy as np
    from mjlab.sim import MujocoCfg
    from mjlabenvs.assets import object_spec
    from mjlabenvs.scene import PHYSICS_DT, table_spec

    records = json.loads((args.tools / 'manifest.json').read_text())['tools']
    rows = []
    for tool in [1139, 954]:
        for subdivisions in [1, 4]:
            spec = table_spec()
            spec.attach(object_spec(args.tools / records[tool]['path']), prefix='object/',
                        frame=spec.worldbody.add_frame())
            model = spec.compile()
            MujocoCfg(timestep=PHYSICS_DT/subdivisions, iterations=20, ls_iterations=50,
                      cone='elliptic').apply(model)
            data = mujoco.MjData(model)
            data.qpos[:] = [0, 0, .9, 1, 0, 0, 0]
            data.mocap_pos[0] = [0, 0, .38]
            data.mocap_quat[0] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            pairs = {}
            tail = []
            for step in range(480*subdivisions):
                mujoco.mj_step(model, data)
                assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                depth = 0.
                for contact in data.contact:
                    penetration = max(0., -float(contact.dist))
                    names = [model.geom(int(g)).name for g in contact.geom]
                    key = ' <-> '.join(sorted(names))
                    pairs[key] = max(pairs.get(key, 0.), penetration)
                    depth = max(depth, penetration)
                if step >= 420*subdivisions:
                    tail.append(depth)
            rows.append({'tool_index': tool, 'asset': records[tool]['path'],
                         'subdivisions': subdivisions, 'dt': model.opt.timestep,
                         'geom_solref': model.geom_solref.tolist(),
                         'pair_max_depth_m': pairs, 'last_half_second_max_depth_m': max(tail),
                         'final_qpos': data.qpos.tolist(), 'final_qvel': data.qvel.tolist()})
    report = {'scope': 'CPU tool/table impact isolation. Same imported primitives/inertias and mjlab solver settings; robot omitted, table z=.38, identity object pose z=.9. Not a replay of the full GPU scene.',
              'cases': rows}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
