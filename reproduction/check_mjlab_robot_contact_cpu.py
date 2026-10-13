"""CPU MuJoCo control for selected worlds from the actual full mjlab scene check."""
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
    parser.add_argument('--worlds', type=int, nargs='+', default=[1139,42,226])
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    import mujoco
    import numpy as np
    import torch
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg
    from mjlabenvs.assets import object_spec
    from mjlabenvs.robot import robot_entity
    from mjlabenvs.scene import PHYSICS_DT, table_spec
    from simtoolreal_shared.robot_constants import JOINT_NAMES_CANONICAL, PALM_BODY_NAME, FINGERTIP_LINK_NAMES

    initial = torch.load(args.initial, map_location='cpu', weights_only=False)
    records = json.loads((args.tools/'manifest.json').read_text())['tools']
    rows = []
    for env_id in args.worlds:
        scene = Scene(SceneCfg(num_envs=1, env_spacing=0, entities={
            'robot': robot_entity(args.robot_assets),
            'object': EntityCfg(spec_fn=partial(object_spec,args.tools/records[env_id]['path'])),
            'table': EntityCfg(spec_fn=table_spec)}), 'cpu')
        model = scene.spec.compile()
        MujocoCfg(timestep=PHYSICS_DT, iterations=20, ls_iterations=50, cone='elliptic').apply(model)
        data = mujoco.MjData(model)
        joints = [model.joint('robot/'+name).id for name in JOINT_NAMES_CANONICAL]
        qadr = model.jnt_qposadr[joints]
        q = initial['joint_pos'][env_id].numpy()
        data.qpos[qadr] = q
        obj = model.joint('object/object_free').qposadr[0]
        data.qpos[obj:obj+7] = initial['object_pose'][env_id].numpy()
        table = model.body('table/table').mocapid[0]
        data.mocap_pos[table] = initial['table_pose'][env_id,:3].numpy()
        data.mocap_quat[table] = initial['table_pose'][env_id,3:].numpy()
        for actuator in range(model.nu):
            joint = model.actuator_trnid[actuator,0]
            data.ctrl[actuator] = q[joints.index(joint)]
        mujoco.mj_forward(model,data)
        palm = data.xpos[model.body('robot/'+PALM_BODY_NAME).id]
        tips = np.array([data.xpos[model.body('robot/'+name).id] for name in FINGERTIP_LINK_NAMES])
        np.testing.assert_allclose(palm,initial['palm_pose'][env_id,:3],atol=1e-6,rtol=0)
        np.testing.assert_allclose(tips,initial['fingertip_pose'][env_id,:,:3],atol=1e-6,rtol=0)
        initial_contacts = [{'geoms':[model.geom(int(g)).name or model.body(int(model.geom_bodyid[g])).name for g in c.geom],
                             'depth_m':max(0.,-float(c.dist))} for c in data.contact]
        pairs, tail_depth, joint_violation = {}, 0., 0.
        q_trace, v_trace, depth_trace = [], [], []
        for step in range(0 if args.snapshot_only else 480):
            mujoco.mj_step(model,data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            joint_violation = max(joint_violation,float(np.maximum(
                model.jnt_range[joints,0]-data.qpos[qadr],data.qpos[qadr]-model.jnt_range[joints,1]).max()))
            for contact in data.contact:
                depth = max(0.,-float(contact.dist))
                if step>=420:
                    tail_depth=max(tail_depth,depth)
                if depth <= .005:
                    continue
                names=[model.geom(int(g)).name or model.body(int(model.geom_bodyid[g])).name for g in contact.geom]
                key=' <-> '.join(sorted(names))
                row=pairs.setdefault(key,{'max_depth_m':0.,'samples':0})
                row['samples']+=1
                if depth>row['max_depth_m']:
                    row.update(max_depth_m=depth,physics_step=step+1)
            q_trace.append(data.qpos.copy())
            v_trace.append(data.qvel.copy())
            depth_trace.append(max([0.] + [-float(c.dist) for c in data.contact]))
            if (step+1) % 2 == 0:
                mujoco.mj_forward(model,data)
        np.savez(args.output.with_name(args.output.stem+f'-world{env_id}.npz'),
                 qpos=np.asarray(q_trace),qvel=np.asarray(v_trace),depth=np.asarray(depth_trace))
        rows.append({'global_env':env_id,'tool':records[env_id]['path'],
                     'initial_contacts':initial_contacts,'snapshot_only':args.snapshot_only,
                     'initial_fk_matches_gpu':True,'deep_pairs':pairs,
                     'last_half_second_max_depth_m':tail_depth,
                     'max_joint_limit_violation_rad':joint_violation,
                     'final_object_pose':data.qpos[obj:obj+7].tolist()})
    report={'scope':'CPU MuJoCo selected complete robot/tool/table worlds. Same native mjlab entity builders,PD controls,solver settings and saved initial states; forward repeated after every two physics steps like the GPU scene. Initial palm/fingertip FK checked against GPU. Independent trajectories, not bitwise replay.', 'cases':rows}
    with args.output.open('x') as stream:
        stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
