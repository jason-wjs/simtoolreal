"""Finite API capability check: contacting box and airborne negative control.

This tests native data availability, not learned-policy physical quality.
"""
import argparse
import json
from pathlib import Path

from isaacgym import gymapi, gymtorch
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['gpu', 'cpu_pipeline_gpu_physx', 'cpu'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1 / 120
    params.substeps = 1
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = args.mode == 'gpu'
    params.physx.use_gpu = args.mode != 'cpu'
    params.physx.contact_collection = gymapi.ContactCollection.CC_LAST_SUBSTEP
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    box = gym.create_box(sim, .1, .1, .1, gymapi.AssetOptions())
    envs = []
    for i, z in enumerate([.04, 1.0]):
        env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 2)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, z)
        gym.create_actor(env, box, pose, 'box', i, 0)
        envs.append(env)
    gym.prepare_sim(sim)
    forces = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
    for _ in range(30):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    gym.refresh_net_contact_force_tensor(sim)
    rows = []
    for label, env in zip(['contact', 'airborne'], envs):
        contacts = gym.get_env_rigid_contacts(env)
        print('NATIVE_CONTACT_ARRAY', label, contacts.dtype, flush=True)
        rows.append({'scene': label, 'contact_count': len(contacts),
                     'dtype_fields': list(contacts.dtype.names),
                     'min_distances': contacts['minDist'].tolist()})
    result = {'mode': args.mode, 'steps': 30, 'dt': params.dt,
              'net_contact_force': forces.cpu().numpy().tolist(), 'scenes': rows,
              'point_query_supported': args.mode != 'gpu',
              'min_dist_meaning': 'Solver target distance, not signed separation or penetration depth.',
              'scope': 'Native API availability only. Empty unsupported contact data is not a no-penetration result.'}
    assert torch.isfinite(forces).all()
    assert abs(forces[0, 2].item() - 9.81) < 0.1, 'Contact positive control failed'
    assert forces[1].abs().max().item() < 1e-5, 'Airborne negative control failed'
    assert rows[1]['contact_count'] == 0
    if args.mode != 'gpu':
        assert rows[0]['contact_count'] > 0, 'Supported point query returned no contacts'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
