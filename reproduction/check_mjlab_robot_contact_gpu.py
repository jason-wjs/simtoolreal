"""Full robot+single-tool GPU control, without VariantEntityCfg."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--robot-assets', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import warp as wp
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
    from mjlabenvs.assets import object_spec
    from mjlabenvs.robot import robot_entity
    from mjlabenvs.scene import PHYSICS_DT, table_spec
    from simtoolreal_shared.robot_constants import JOINT_NAMES_CANONICAL, PALM_BODY_NAME, FINGERTIP_LINK_NAMES

    initial = torch.load(args.initial,map_location='cuda:0',weights_only=False)
    records = json.loads((args.tools/'manifest.json').read_text())['tools']
    rows = []
    for env_id in [1139,42,226]:
        scene = Scene(SceneCfg(num_envs=64,env_spacing=0,entities={
            'robot':robot_entity(args.robot_assets),
            'object':EntityCfg(spec_fn=partial(object_spec,args.tools/records[env_id]['path'])),
            'table':EntityCfg(spec_fn=table_spec)}),'cuda:0')
        sim = Simulation(64,SimulationCfg(nconmax=256,njmax=1024,
            mujoco=MujocoCfg(timestep=PHYSICS_DT,iterations=20,ls_iterations=50,cone='elliptic')),
            device='cuda:0',spec=scene.spec)
        scene.initialize(sim.mj_model,sim.model,sim.data)
        ids=torch.arange(64,device='cuda:0')
        sim.reset(ids)
        scene.reset(ids)
        robot=scene['robot']
        joint_ids,_=robot.find_joints(JOINT_NAMES_CANONICAL,preserve_order=True)
        q=initial['joint_pos'][env_id].expand(64,-1).clone()
        robot.write_joint_state_to_sim(q,torch.zeros_like(q),joint_ids=joint_ids,env_ids=ids)
        robot.set_joint_position_target(q,joint_ids=joint_ids)
        obj=torch.zeros(64,13,device='cuda:0')
        obj[:,:7]=initial['object_pose'][env_id]
        scene['object'].write_root_state_to_sim(obj,env_ids=ids)
        table=initial['table_pose'][env_id].expand(64,-1).clone()
        scene['table'].write_mocap_pose_to_sim(table,env_ids=ids)
        scene.write_data_to_sim()
        sim.forward()
        scene.update(0)
        palm=robot.find_bodies(PALM_BODY_NAME)[0][0]
        tips=robot.find_bodies(FINGERTIP_LINK_NAMES,preserve_order=True)[0]
        torch.testing.assert_close(robot.data.body_link_pose_w[0,palm,:3],initial['palm_pose'][env_id,:3],atol=1e-6,rtol=0)
        torch.testing.assert_close(robot.data.body_link_pose_w[0,tips,:3],initial['fingertip_pose'][env_id,:,:3],atol=1e-6,rtol=0)
        q_trace,v_trace,depth_trace=[],[],[]
        max_depth,tail_depth,joint_violation=0.,0.,0.
        limits=robot.data.joint_pos_limits[0,joint_ids]
        for step in range(480):
            scene.write_data_to_sim()
            sim.step()
            scene.update(PHYSICS_DT)
            count=int(wp.to_torch(sim.wp_data.nacon)[0])
            assert count<=sim.wp_data.naconmax
            depth=0.
            if count:
                distances=wp.to_torch(sim.wp_data.contact.dist)[:count]
                assert bool(torch.isfinite(distances).all())
                depth=float((-distances).clamp_min(0).max())
            max_depth=max(max_depth,depth)
            if step>=420:tail_depth=max(tail_depth,depth)
            pos=robot.data.joint_pos[:,joint_ids]
            joint_violation=max(joint_violation,float(torch.maximum(limits[:,0]-pos,pos-limits[:,1]).clamp_min(0).max()))
            assert bool(torch.isfinite(sim.data.qpos[:]).all()) and bool(torch.isfinite(sim.data.qvel[:]).all())
            q_trace.append(sim.data.qpos[0].cpu().numpy().copy())
            v_trace.append(sim.data.qvel[0].cpu().numpy().copy())
            depth_trace.append(depth)
            if (step+1)%2==0:sim.forward()
        np.savez(args.output/f'world{env_id}.npz',qpos=np.asarray(q_trace),qvel=np.asarray(v_trace),depth=np.asarray(depth_trace))
        rows.append({'global_env':env_id,'num_envs':64,'initial_fk_matches':True,
                     'max_depth_m':max_depth,'last_half_second_max_depth_m':tail_depth,
                     'max_joint_limit_violation_rad':joint_violation,
                     'final_object_pose':scene['object'].data.root_link_pose_w[0].tolist()})
    report={'scope':'Native mjlab fullrobot+singletool,64 identical worlds for each CPU case. No variants; same initial states/PD/solver/timing,including policy-boundary forward. Depth maximum across64worlds; trace records world0.', 'cases':rows}
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
