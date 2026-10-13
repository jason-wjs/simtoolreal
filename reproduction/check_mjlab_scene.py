"""Bounded full-tool-pool scene rollout, before task learning is implemented."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-assets', type=Path, required=True)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import warp as wp
    from mjlabenvs.scene import SimToolRealScene, PHYSICS_DT, DECIMATION

    torch.manual_seed(42)
    start = time.monotonic()
    scene = SimToolRealScene(args.robot_assets, args.tools, 1536, 'cuda:0')
    report = {'num_envs': scene.num_envs, 'object_pool': 1200,
              'group_sizes': [len(ids) for ids, _ in scene.groups],
              'physics_dt': PHYSICS_DT, 'decimation': DECIMATION,
              'setup_seconds': time.monotonic()-start, 'steps': 240,
              'contact_samples': 0, 'max_penetration_m': 0.,
              'max_joint_limit_violation_rad': 0., 'all_finite': True}
    for _, group in scene.groups:
        def step_and_audit(original=group.sim.step, group=group):
            original()
            data = group.sim.wp_data
            count = int(wp.to_torch(data.nacon)[0])
            assert count <= data.naconmax, 'Native contact capacity exceeded'
            if count:
                distances = wp.to_torch(data.contact.dist)[:count]
                assert bool(torch.isfinite(distances).all())
                report['contact_samples'] += count
                report['max_penetration_m'] = max(report['max_penetration_m'],
                                                 float((-distances).clamp_min(0).max()))
        group.sim.step = step_and_audit
    n = scene.num_envs
    ids = torch.arange(n, device=scene.device)
    q = scene.default_joint_pos.expand(n, -1).clone()
    objects = torch.zeros(n, 13, device=scene.device)
    objects[:, 2], objects[:, 3] = .9, 1
    tables = torch.zeros(n, 7, device=scene.device)
    tables[:, 2] = .38 + torch.empty(n, device=scene.device).uniform_(-.01, .01)
    tables[:, 3] = 1
    goals = objects[:, :7].clone()
    scene.reset(ids, q, torch.zeros_like(q), objects, tables, goals)
    initial = scene.state()
    torch.testing.assert_close(initial['joint_pos'], q, atol=1e-6, rtol=0)
    torch.testing.assert_close(initial['object_pose'], objects[:, :7], atol=1e-6, rtol=0)
    torch.testing.assert_close(initial['table_pose'], tables, atol=1e-6, rtol=0)
    torch.save(initial, args.output / 'initial.pt')
    forces = torch.zeros(n, 1, 3, device=scene.device)
    rollout = time.monotonic()
    for index in range(report['steps']):
        scene.step(q, forces, forces)
        state = scene.state()
        assert all(bool(torch.isfinite(v).all()) for v in state.values())
        joint = state['joint_pos']
        violation = torch.maximum((scene.joint_limits[:, 0]-joint).clamp_min(0),
                                   (joint-scene.joint_limits[:, 1]).clamp_min(0))
        report['max_joint_limit_violation_rad'] = max(report['max_joint_limit_violation_rad'], float(violation.max()))
        if index % 30 == 0:
            print(json.dumps({'step': index+1, 'max_depth_m': report['max_penetration_m']}), flush=True)
    torch.cuda.synchronize()
    report['rollout_seconds'] = time.monotonic()-rollout
    report['elapsed_seconds'] = time.monotonic()-start
    report['physical_review_required'] = report['max_penetration_m'] > .005 or report['max_joint_limit_violation_rad'] > .01
    report['scope'] = 'All1200 original tool assets and1536 worlds in3topology groups. Native primitive geometry, robot PD hold, randomized table z, diagnostic freefall from0.9m. Native contact depths sampled every physics step. No task reward, learning, or accepted physical reproduction claim.'
    torch.save(state, args.output / 'final.pt')
    (args.output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
