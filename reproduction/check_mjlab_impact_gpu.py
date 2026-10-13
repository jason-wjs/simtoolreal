"""Same two CPU impact cases in native mjlab, with the robot omitted."""
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
    import torch
    import warp as wp
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
    from mjlabenvs.assets import object_spec
    from mjlabenvs.scene import PHYSICS_DT, table_spec

    records = json.loads((args.tools / 'manifest.json').read_text())['tools']
    rows = []
    for tool in [1139, 954]:
        scene = Scene(SceneCfg(num_envs=64, env_spacing=0, entities={
            'object': EntityCfg(spec_fn=partial(object_spec, args.tools / records[tool]['path'])),
            'table': EntityCfg(spec_fn=table_spec)}), 'cuda:0')
        sim = Simulation(64, SimulationCfg(nconmax=256, njmax=1024,
            mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=20, ls_iterations=50, cone='elliptic')),
            device='cuda:0', spec=scene.spec)
        scene.initialize(sim.mj_model, sim.model, sim.data)
        obj = torch.zeros(64, 13, device='cuda:0')
        obj[:, 2], obj[:, 3] = .9, 1
        table = torch.zeros(64, 7, device='cuda:0')
        table[:, 2], table[:, 3] = .38, 1
        scene['object'].write_root_state_to_sim(obj)
        scene['table'].write_mocap_pose_to_sim(table)
        sim.forward()
        max_depth, tail_depth = 0., 0.
        for step in range(480):
            scene.write_data_to_sim()
            sim.step()
            scene.update(PHYSICS_DT)
            count = int(wp.to_torch(sim.wp_data.nacon)[0])
            assert count <= sim.wp_data.naconmax
            depth = 0.
            if count:
                distances = wp.to_torch(sim.wp_data.contact.dist)[:count]
                assert bool(torch.isfinite(distances).all())
                depth = float((-distances).clamp_min(0).max())
            max_depth = max(max_depth, depth)
            if step >= 420:
                tail_depth = max(tail_depth, depth)
        sim.forward()
        state = torch.cat([scene['object'].data.root_link_pose_w,
                           scene['object'].data.root_com_vel_w], dim=-1)
        assert bool(torch.isfinite(state).all())
        rows.append({'tool_index': tool, 'num_envs': 64, 'max_depth_m': max_depth,
                     'last_half_second_max_depth_m': tail_depth,
                     'final_state_first_world': state[0].tolist(),
                     'final_state_world_spread': (state.max(0).values-state.min(0).values).tolist(),
                     'geom_solref': sim.mj_model.geom_solref.tolist()})
    result = {'scope': 'Native mjlab two isolated tool/table cases,64 identical worlds each,dt1/120,480physics steps. Same CPU impact initial pose and solver settings; robot omitted.', 'cases': rows}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
