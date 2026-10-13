"""CPU-pipeline PhysX contact calibration against known box/plane gaps."""
import argparse
import hashlib
import json
from pathlib import Path

from isaacgym import gymapi
import numpy as np


def vector(value):
    return np.array([value[k] for k in ['x', 'y', 'z']], dtype=float)


def box_gap(states, box_body, support_top):
    pose = states['pose'][box_body]
    q = pose['r']
    x, y, z, w = [float(q[k]) for k in ['x', 'y', 'z', 'w']]
    row_z = np.array([2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)])
    return float(pose['p']['z']) - .05 * float(np.abs(row_z).sum()) - support_top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-physx', action='store_true')
    args = parser.parse_args()
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1 / 120
    params.substeps = 1
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = args.gpu_physx
    params.physx.num_threads = 2
    params.physx.contact_collection = gymapi.ContactCollection.CC_LAST_SUBSTEP
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    assert sim is not None
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    box = gym.create_box(sim, .1, .1, .1, gymapi.AssetOptions())
    fixed = gymapi.AssetOptions()
    fixed.fix_base_link = True
    support = gym.create_box(sim, .4, .4, .3, fixed)
    cases = []
    for support_kind in ['plane', 'box']:
        for label, gap, angle in [('penetrating', -.01, 0.), ('small_gap', .005, 0.),
                                  ('rotated_penetrating', -.01, .35), ('rotated_gap', .005, .35),
                                  ('separated', .1, 0.)]:
            env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 3)
            support_top = 0. if support_kind == 'plane' else .3
            support_body = -1
            if support_kind == 'box':
                support_pose = gymapi.Transform()
                support_pose.p = gymapi.Vec3(0, 0, .15)
                support_actor = gym.create_actor(env, support, support_pose, 'support', len(cases), 0)
                support_body = gym.get_actor_rigid_body_index(env, support_actor, 0, gymapi.DOMAIN_ENV)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, support_top + .05 * (abs(np.sin(angle)) + abs(np.cos(angle))) + gap)
            pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), angle)
            actor = gym.create_actor(env, box, pose, 'box', len(cases), 0)
            cases.append(dict(label=label, support_kind=support_kind, support_top_m=support_top,
                              support_body=support_body,
                              box_body=gym.get_actor_rigid_body_index(env, actor, 0, gymapi.DOMAIN_ENV),
                              expected_initial_gap_m=gap, angle_rad=angle, env=env, actor=actor, pose=pose))
    gym.prepare_sim(sim)
    # Scene initialization can change initial overlapping poses. Install the known test poses
    # afterwards and verify their actual geometry before measuring one step.
    before = []
    for c in cases:
        state = gym.get_actor_rigid_body_states(c['env'], c['actor'], gymapi.STATE_ALL).copy()
        pose = c.pop('pose')
        state['pose']['p'][0] = (pose.p.x, pose.p.y, pose.p.z)
        state['pose']['r'][0] = (pose.r.x, pose.r.y, pose.r.z, pose.r.w)
        state['vel']['linear'][0] = (0, 0, 0)
        state['vel']['angular'][0] = (0, 0, 0)
        gym.set_actor_rigid_body_states(c['env'], c.pop('actor'), state, gymapi.STATE_ALL)
        actual = gym.get_env_rigid_body_states(c['env'], gymapi.STATE_ALL).copy()
        if c['support_body'] >= 0:
            assert abs(float(actual['pose']['p']['z'][c['support_body']]) + .15 - c['support_top_m']) < 1e-6
        assert abs(box_gap(actual, c['box_body'], c['support_top_m']) - c['expected_initial_gap_m']) < 1e-6
        before.append(actual)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    result = []
    for c, pre in zip(cases, before):
        env = c.pop('env')
        post = gym.get_env_rigid_body_states(env, gymapi.STATE_ALL).copy()
        contacts = gym.get_env_rigid_contacts(env)
        rows = []
        for contact in contacts:
            if {int(contact['body0']), int(contact['body1'])} != {c['box_body'], c['support_body']}:
                continue
            normal = vector(contact['normal'])
            record = {name: float(contact[name]) for name in ['minDist', 'initialOverlap', 'lambda']}
            record.update(body0=int(contact['body0']), body1=int(contact['body1']), normal=normal.tolist())
            for name in ['localPos0', 'localPos1', 'offset0', 'offset1']:
                record[name] = vector(contact[name]).tolist()
            rows.append(record)
        if c['label'] != 'separated':
            assert len(rows) > 0, c
        if c['label'] == 'separated':
            assert len(rows) == 0, rows
        c.update(pre_geometric_gap_m=box_gap(pre, c['box_body'], c['support_top_m']), post_geometric_gap_m=box_gap(post, c['box_body'], c['support_top_m']),
                 minimum_initialOverlap_m=min((r['initialOverlap'] for r in rows), default=None),
                 contacts=rows, before_box_z=float(pre['pose']['p']['z'][c['box_body']]),
                 after_box_z=float(post['pose']['p']['z'][c['box_body']]))
        if rows:
            c['calibration_error_m'] = abs(c['minimum_initialOverlap_m'] - c['pre_geometric_gap_m'])
            assert c['calibration_error_m'] < 1e-6, c
        result.append(c)
    payload = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   calibration_passed=True, mode='cpu_pipeline_gpu_physx' if args.gpu_physx else 'cpu_pipeline_cpu_physx',
                   steps=1, gravity=[0, 0, 0], dt=params.dt,
                   box_size=[.1, .1, .1], cases=result,
                   scope='Known geometric box/plane and box/box fixtures with CPU tensor pipeline only. '
                         'Negative initialOverlap agrees with pre-step penetration for these cases. '
                         'Not a post-step distance, GPU-pipeline measurement or policy physical-quality acceptance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2), flush=True)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
