"""Read-only native parameters and first-policy-input snapshots for Gym/Lab."""
import json
from pathlib import Path


def array(tensor):
    return tensor.detach().cpu().numpy().tolist()


def write_snapshot(output_dir, snapshot):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "native_state.json").write_text(json.dumps(snapshot, indent=2) + "\n")


def gym_snapshot(env, obs, action):
    gym, handle, robot = env.gym, env.envs[0], env.robots[0]
    props = gym.get_actor_dof_properties(handle, robot)
    names = gym.get_actor_dof_names(handle, robot)
    joints = {name: {key: props[key][i].item() for key in props.dtype.names}
              for i, name in enumerate(names)}
    actors = {}
    for label, actor in [("robot", robot), ("object", env.objects[0])]:
        bodies = gym.get_actor_rigid_body_properties(handle, actor)
        body_names = gym.get_actor_rigid_body_names(handle, actor)
        actors[label] = {name: {
            "mass": float(p.mass),
            "com_position": [float(getattr(p.com, axis)) for axis in "xyz"],
            "inertia_raw": [[float(getattr(getattr(p.inertia, row), col)) for col in "xyz"] for row in "xyz"],
        } for name, p in zip(body_names, bodies)}
        actors[label + "_shape_count"] = gym.get_actor_rigid_shape_count(handle, actor)
    return {"backend": "gym", "joint_names": names, "joints": joints, "actors": actors,
            "joint_pos": array(env.arm_hand_dof_pos[0]), "joint_vel": array(env.arm_hand_dof_vel[0]),
            "targets": array(env.cur_targets[0]), "observation": array(obs[0]), "action": array(action[0]),
            "object_state_xyzw": array(env.root_state_tensor[env.object_indices[0]]),
            "goal_state_xyzw": array(env.goal_states[0]),
            "inertia_note": "Raw Gym RigidBodyProperties inertia; compare frames before numerical cross-engine inertia comparison."}


def lab_snapshot(env, obs, action):
    view = env.robot.root_physx_view
    fields = {"stiffness": view.get_dof_stiffnesses(), "damping": view.get_dof_dampings(),
              "effort": view.get_dof_max_forces(), "velocity": view.get_dof_max_velocities(),
              "friction": view.get_dof_friction_coefficients(), "armature": view.get_dof_armatures()}
    friction_properties = view.get_dof_friction_properties()[0]
    limits = view.get_dof_limits()[0]
    names = list(env.robot.joint_names)
    joints = {name: {key: float(value[0, i]) for key, value in fields.items()} for i, name in enumerate(names)}
    for i, name in enumerate(names):
        joints[name].update(lower=float(limits[i, 0]), upper=float(limits[i, 1]),
                            friction_properties_static_dynamic_viscous=array(friction_properties[i]))
    actors = {}
    for label, asset, body_names in [("robot", env.robot, env.robot.body_names),
                                      ("object", env.object, [env.object.root_physx_view.prim_paths[0].split("/")[-1]])]:
        native = asset.root_physx_view
        masses = native.get_masses().reshape(-1)
        coms = native.get_coms().reshape(-1, 7)
        inertias = native.get_inertias().reshape(-1, 3, 3)
        actors[label] = {name: {"mass": float(masses[i]), "com_position": array(coms[i, :3]),
                                "com_rotation_xyzw": array(coms[i, 3:]), "inertia_raw": array(inertias[i])}
                         for i, name in enumerate(body_names)}
        actors[label + "_shape_count"] = native.max_shapes
    def xyzw(state):
        value = array(state)
        return value[:3] + value[4:7] + [value[3]] + value[7:]
    return {"backend": "lab", "joint_names": names, "joints": joints, "actors": actors,
            "joint_pos": array(env.robot.data.joint_pos[0]), "joint_vel": array(env.robot.data.joint_vel[0]),
            "targets": array(env._cur_targets[0]), "observation": array(obs[0]), "action": array(action[0]),
            "object_state_xyzw": xyzw(env.object.data.root_state_w[0]),
            "goal_state_xyzw": xyzw(env.goal_viz.data.root_state_w[0]),
            "inertia_note": "Raw PhysX tensor inertia and COM transform; compare frames before numerical cross-engine inertia comparison."}
