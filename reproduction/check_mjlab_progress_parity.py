"""CPU lifecycle parity against native Lab teacher code at commit 1c41d4c.

Identity rotations and zero keypoint offsets isolate reward/counter ordering
from quaternion geometry. This does not validate the scene or pose encoding.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import yaml
    from mjlabenvs.progress import SimToolRealProgress

    repo = Path(__file__).resolve().parents[1]
    data = yaml.safe_load(args.config.read_text())['env']
    cfg = NS(**{k: NS(**data[k]) for k in ('reward', 'termination', 'reset')})
    reference = {'torch': torch, '__name__': 'native_progress_reference'}

    def load_functions(filename, selected=None):
        code = subprocess.check_output(['git', 'show',
            f'1c41d4c:isaacsimenvs/tasks/simtoolreal/utils/{filename}.py'], cwd=repo, text=True)
        tree = ast.parse(code)
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and (selected is None or node.name in selected)]
        # Only the local helper import is replaced by the same frozen module's
        # helper. No lifecycle arithmetic is changed in the reference.
        for node in tree.body:
            node.body = [line for line in node.body if not isinstance(line, ast.ImportFrom)]
        exec(compile(tree, filename, 'exec'), reference)

    load_functions('reward_utils')
    load_functions('obs_utils', {'compute_intermediate_values', '_keypoints_world'})
    load_functions('termination_utils')
    load_functions('reset_utils', {'_clear_goal_trackers'})
    reference['quat_apply'] = lambda quat, vector: vector  # All test quaternions are identity.

    def reset_goal(env, ids):
        reference['_clear_goal_trackers'](env, ids)
        env.goal_ids = ids.clone()
    reference['reset_goal_trackers'] = reset_goal

    torch.manual_seed(42)
    n = 8
    scenarios = []
    for consecutive in (True, False):
        for eval_tolerance in (None, cfg.termination.target_success_tolerance):
            case_cfg = copy.deepcopy(cfg)
            case_cfg.termination.force_consecutive_near_goal_steps = consecutive
            case_cfg.termination.eval_success_tolerance = eval_tolerance
            port = SimToolRealProgress(case_cfg, n, 'cpu', 600)
            native = NS(**copy.deepcopy(port.__dict__))
            native._fingertip_body_ids = list(range(5))
            native._keypoint_offsets = torch.zeros(n, 4, 3)
            native._keypoint_offsets_fixed = torch.zeros(n, 4, 3)
            origins = torch.randn(n, 3)
            native.scene = NS(env_origins=origins)
            native.object = NS(data=NS())
            native.goal_viz = NS(data=NS())
            native.robot = NS(data=NS())
            # Exercise both rejected and accepted curriculum updates at its
            # actual interval without spending thousands of dummy steps.
            native._frame_counter = port._frame_counter = case_cfg.termination.tolerance_curriculum_interval - 1
            for step in range(100):
                if step == 2:
                    successes = int(case_cfg.termination.tolerance_curriculum_success_threshold) + 1
                    native._prev_episode_successes[:] = successes
                    port._prev_episode_successes[:] = successes
                if step == 40:
                    ids = torch.tensor([1, 3, 6])
                    init_z = torch.tensor([.5, .7, .9])
                    port.reset(ids, init_z)
                    native._prev_episode_successes[ids] = native._successes[ids]
                    reference['_clear_goal_trackers'](native, ids)
                    native._lifted_object[ids] = False
                    native._successes[ids] = 0
                    native.episode_length_buf[ids] = 0
                    native._object_init_z[ids] = init_z
                if step == 50:
                    state = port.get_state()
                    port = SimToolRealProgress(case_cfg, n, 'cpu', 600)
                    port.set_state(state)
                native.episode_length_buf += 1
                port.episode_length_buf += 1
                native.episode_length_buf[5] = port.episode_length_buf[5] = 600
                if step == 30:
                    native._successes[7] = port._successes[7] = case_cfg.termination.max_consecutive_successes
                obj = torch.zeros(n, 3)
                obj[:, 2] = torch.linspace(.05, 1., n)
                goal = obj.clone()
                goal[:, 0] += torch.tensor([.01, .4, .005, .1, 0., .5, .02, .3])
                if step % 5 == 0:
                    goal[2, 0] += .4  # Break the near-goal streak.
                ft = obj[:, None] + torch.rand(n, 5, 3) * .2
                ft[4, 0, 0] += 2.  # Hand-far termination.
                identity = torch.tensor([1., 0., 0., 0.]).expand(n, -1)
                native.object.data.root_pos_w = obj + origins
                native.object.data.root_quat_w = identity
                native.goal_viz.data.root_pos_w = goal + origins
                native.goal_viz.data.root_quat_w = identity
                native.robot.data.body_state_w = torch.cat([
                    ft + origins[:, None], identity[:, None].expand(-1, 5, -1), torch.zeros(n, 5, 6)], -1)
                native.robot.data.joint_vel = torch.randn(n, 29)
                reference['update_tolerance_curriculum'](native)
                reference['compute_intermediate_values'](native)
                native.goal_ids = torch.empty(0, dtype=torch.long)
                expected_done = reference['compute_terminations'](native)
                # Scene-local z uses exactly the same world-origin subtraction.
                z = (native.object.data.root_pos_w - origins)[:, 2]
                actual_done = port.dones(z, native._curr_fingertip_distances.clone(),
                                         native._keypoints_max_dist.clone())
                for actual, expected in zip(actual_done, (*expected_done, native.goal_ids)):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                expected_reward = reference['compute_rewards'](native)
                actual_reward = port.rewards(z, native.robot.data.joint_vel)
                torch.testing.assert_close(actual_reward, expected_reward, atol=0, rtol=0)
                for name in port._tensor_fields:
                    torch.testing.assert_close(getattr(port, name), getattr(native, name), atol=0, rtol=0)
                for name in port._scalar_fields:
                    assert getattr(port, name) == getattr(native, name), name
                for name in native._reward_terms:
                    torch.testing.assert_close(port._reward_terms[name], native._reward_terms[name], atol=0, rtol=0)
            scenarios.append({'consecutive': consecutive, 'eval_tolerance': eval_tolerance, 'passed': True})
    report = {'passed': True, 'reference_commit': '1c41d4c', 'steps_per_case': 100,
              'environments': n, 'exact_float32_rewards_trackers_and_terminations': True,
              'cases': scenarios, 'geometry_scope': 'identity quaternion fixture; precomputed native distances',
              'checks': ['goal reset before reward', 'one-shot lift', 'partial reset', 'full counter restore',
                         'curriculum rejection and shrink', 'eval tolerance pin', 'fall, hand-far, max-success, timeout']}
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
