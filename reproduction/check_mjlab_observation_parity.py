"""CPU teacher observation parity with frozen native code and installed Lab math."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--lab-math', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import yaml
    from mjlabenvs.observations import SimToolRealObservations, OBS_FIELD_SIZES
    from mjlabenvs.progress import SimToolRealProgress
    from simtoolreal_shared.rewards import update_near_goal_steps

    # Compile the actual installed Isaac Lab math functions independently of
    # mjlab. A real source file lets TorchScript preserve its native decorators.
    math_src = args.lab_math.read_text()
    names = {'normalize', 'quat_apply', 'quat_mul', 'quat_from_angle_axis', 'convert_quat'}
    nodes = [n for n in ast.parse(math_src).body if isinstance(n, ast.FunctionDef) and n.name in names]
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'native_observation_math.py'
        path.write_text('import torch\nimport numpy as np\nfrom typing import Literal\n' + '\n'.join(ast.unparse(n) for n in nodes))
        spec = importlib.util.spec_from_file_location('native_observation_math', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        reference = {name: getattr(module, name) for name in names}

    repo = Path(__file__).resolve().parents[1]
    code = subprocess.check_output(['git', 'show',
        '0eb3edc:isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py'], cwd=repo, text=True)
    tree = ast.parse(code)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'isaaclab.utils.math')]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'compute_intermediate_values':
            node.body = [line for line in node.body if not isinstance(line, ast.ImportFrom)]
    reference['update_near_goal_steps'] = update_near_goal_steps
    exec(compile(tree, 'native-observation-reference', 'exec'), reference)
    data = yaml.safe_load(args.config.read_text())['env']
    cfg = NS(**{key: NS(**data[key]) for key in ('reward', 'reset', 'termination', 'obs', 'domain_randomization')})
    torch.manual_seed(42)
    n = 8
    scales = torch.rand(n, 3) + .5
    limits = torch.stack([-torch.rand(29)-.1, torch.rand(29)+.1], dim=-1)
    cases = []
    for case in ('native_recipe', 'clean', 'all_fields_noise', 'scaled_reward_keypoints'):
        c = copy.deepcopy(cfg)
        if case == 'clean':
            c.domain_randomization.use_object_state_delay_noise = False
            c.domain_randomization.use_obs_delay = False
        if case == 'all_fields_noise':
            c.obs.obs_list = c.obs.state_list = list(OBS_FIELD_SIZES)
            c.domain_randomization.joint_velocity_obs_noise_std = .01
        if case == 'scaled_reward_keypoints':
            c.reward.fixed_size_keypoint_reward = False
        port = SimToolRealObservations(c, scales, limits, 'cpu')
        progress = SimToolRealProgress(c, n, 'cpu', 600)
        native = NS(**copy.deepcopy(port.__dict__))
        native._perm_lab_to_canon = torch.randperm(29)
        native._palm_body_id, native._fingertip_body_ids = 0, list(range(1, 6))
        native.scene = NS(env_origins=torch.randn(n, 3))
        native.robot, native.object, native.goal_viz = NS(data=NS()), NS(data=NS()), NS(data=NS())
        native._object_scale_multiplier = torch.empty(n, 3).uniform_(.9, 1.1)
        port.reset(torch.arange(n), native._object_scale_multiplier)
        max_error = 0.
        for step in range(60):
            if step == 20:
                progress.episode_length_buf[0], progress._successes[0] = 0, 1
            if step == 30:
                ids = torch.tensor([1, 3])
                scale = torch.empty(2, 3).uniform_(.9, 1.1)
                port.reset(ids, scale)
                native._object_scale_multiplier[ids] = scale
                native._object_state_queue[ids] = native._obs_queue[ids] = 0
                progress.reset(ids, torch.ones(2)*.6)
            if step == 40:
                saved = port.get_state()
                port = SimToolRealObservations(c, scales, limits, 'cpu')
                port.set_state(saved)
            def poses(count):
                value = torch.randn(n, count, 7)
                value[:, :, 3:7] = torch.nn.functional.normalize(value[:, :, 3:7], dim=-1)
                return value
            body = torch.cat([poses(6), torch.randn(n, 6, 6)], dim=-1)
            obj, goal = poses(1)[:, 0], poses(1)[:, 0]
            native.robot.data.body_state_w = body
            native.robot.data.joint_pos = torch.randn(n, 29)
            native.robot.data.joint_vel = torch.randn(n, 29)
            native._prev_targets = torch.randn(n, 29)
            native.object.data.root_pos_w, native.object.data.root_quat_w = obj[:, :3], obj[:, 3:7]
            native.object.data.root_lin_vel_w, native.object.data.root_ang_vel_w = torch.randn(n, 3), torch.randn(n, 3)
            native.goal_viz.data.root_pos_w, native.goal_viz.data.root_quat_w = goal[:, :3], goal[:, 3:7]
            for name in progress._tensor_fields + progress._scalar_fields:
                setattr(native, name, copy.deepcopy(getattr(progress, name)))
            native.reward_buf = torch.randn(n) * 1000
            state = dict(joint_pos=native.robot.data.joint_pos[:, native._perm_lab_to_canon],
                joint_vel=native.robot.data.joint_vel[:, native._perm_lab_to_canon],
                object_pose=torch.cat([obj[:, :3]-native.scene.env_origins, obj[:, 3:7]], -1),
                object_velocity=torch.cat([native.object.data.root_lin_vel_w, native.object.data.root_ang_vel_w], -1),
                goal_pose=torch.cat([goal[:, :3]-native.scene.env_origins, goal[:, 3:7]], -1),
                palm_pose=torch.cat([body[:, 0, :3]-native.scene.env_origins, body[:, 0, 3:7]], -1),
                palm_velocity=body[:, 0, 7:13],
                fingertip_pose=torch.cat([body[:, 1:, :3]-native.scene.env_origins[:, None], body[:, 1:, 3:7]], -1))
            reference['compute_intermediate_values'](native)
            actual_dist = port.distances(state)
            for a, b in zip(actual_dist, (native._curr_fingertip_distances, native._keypoints_max_dist)):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=0)
            # Observations consume the tracker state after done/reward hooks.
            for name in ('_closest_keypoint_max_dist', '_closest_fingertip_dist'):
                setattr(progress, name, getattr(native, name).clone())
            rng = torch.get_rng_state()
            expected = reference['build_observations'](native)
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            actual = port.build(state, progress, native._prev_targets[:, native._perm_lab_to_canon], native.reward_buf)
            assert torch.equal(torch.get_rng_state(), expected_rng)
            for name in ('policy', 'critic'):
                error = float((actual[name]-expected[name]).abs().max())
                max_error = max(max_error, error)
                torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=0)
            for name in ('_object_state_queue', '_obs_queue'):
                torch.testing.assert_close(getattr(port, name), getattr(native, name), atol=2e-6, rtol=0)
            progress.episode_length_buf += 1
        cases.append(dict(case=case, passed=True, steps=60, max_absolute_observation_error=max_error,
                          actor_dim=actual['policy'].shape[1], critic_dim=actual['critic'].shape[1]))
    report = dict(passed=True, reference_commit='0eb3edc', cases=cases,
        lab_math_sha256=hashlib.sha256(args.lab_math.read_bytes()).hexdigest(),
        checks=['nonidentity rotations', 'nonzero scene origins', 'canonical joint permutation',
                'raw fingertip reward versus offset pad observation', 'keypoint geometry',
                'actor/critic field order', 'identical RNG advancement', 'delay queues',
                'goal-hit retains queue', 'partial episode reset', 'separate-object state restore'],
        scope='CPU tensors; does not establish live scene velocity frames or full task acceptance')
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
