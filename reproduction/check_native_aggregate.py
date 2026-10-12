"""Measure original SimToolReal shape capacities and exercise the native allocator.

Inventory all original tools by default. --fixture TOOL creates one native scene;
--round-capacity repairs only the known Preview 4 aggregate bitmap allocation.
This is an allocation regression fixture, not policy or physical-quality testing.
"""
import argparse
import json
from pathlib import Path

from isaacgym import gymapi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', choices=['short_screwdriver', 'flat_spatula'])
    parser.add_argument('--round-capacity', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'assets'
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1 / 60
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 8
    params.physx.num_velocity_iterations = 0
    params.physx.contact_offset = 0.002
    params.physx.rest_offset = 0
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    robot_options = gymapi.AssetOptions()
    robot_options.fix_base_link = True
    robot_options.collapse_fixed_joints = True
    robot_options.disable_gravity = True
    robot_options.thickness = 0.001
    robot_options.angular_damping = robot_options.linear_damping = 0.01
    robot_options.use_physx_armature = True
    robot_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    robot = gym.load_asset(sim, str(root), 'urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf', robot_options)
    table_options = gymapi.AssetOptions()
    table_options.disable_gravity = True
    table_options.fix_base_link = True
    table = gym.load_asset(sim, str(root), 'urdf/table_narrow.urdf', table_options)
    tools = sorted((root / 'urdf/dextoolbench').glob('*/*/*_decomposed.urdf'))
    assert len(tools) == 12
    rows = []
    for path in tools:
        name = path.parent.name
        if args.fixture and name != args.fixture:
            continue
        options = gymapi.AssetOptions()
        options.collapse_fixed_joints = True
        options.replace_cylinder_with_capsule = True
        obj = gym.load_asset(sim, str(path.parent), path.name, options)
        options.disable_gravity = True
        goal = gym.load_asset(sim, str(path.parent), path.name, options)
        assets = [robot, obj, table, goal]
        shapes = [gym.get_asset_rigid_shape_count(a) for a in assets]
        bodies = [gym.get_asset_rigid_body_count(a) for a in assets]
        original = sum(shapes)
        capacity = ((original + 30) // 32) * 32 + 1 if args.round_capacity else original
        row = {'object': name, 'actor_order': ['robot', 'object', 'table', 'goal'],
               'shapes': shapes, 'bodies': bodies, 'original_capacity': original,
               'requested_capacity': capacity, 'bitmap_allocated_bytes': 4*((capacity+31)//32),
               'bitmap_cleared_bytes': (capacity+31)//8}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.fixture:
            env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 1)
            gym.begin_aggregate(env, sum(bodies), capacity, True)
            for label, asset, xyz, group, mask in zip(
                row['actor_order'], assets, [(0,.8,0),(0,0,.63),(0,0,.38),(-.35,-.06,.71)],
                [0,0,0,1], [-1,0,-1,0]):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(*xyz)
                gym.create_actor(env, asset, pose, label, group, mask, 0)
            gym.end_aggregate(env)
            gym.prepare_sim(sim)
            print('NATIVE_AGGREGATE_SCENE_CREATED', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2)+'\n')
    gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
