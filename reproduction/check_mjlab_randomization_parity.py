"""CPU reset and persistent-wrench parity against frozen native teacher code."""
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
    from mjlab.utils.lab_api.math import random_orientation
    from mjlabenvs.randomization import SimToolRealReset, SimToolRealWrench
    repo = Path(__file__).resolve().parents[1]
    ref = dict(torch=torch, random_orientation=random_orientation)
    def source(name):
        return subprocess.check_output(['git','show',f'8e18764:isaacsimenvs/tasks/simtoolreal/utils/{name}.py'],cwd=repo,text=True)
    exec(compile(source('goal_sampling').replace('from isaaclab.utils.math import','from mjlab.utils.lab_api.math import'),'native-goal','exec'),ref)
    tree = ast.parse(source('reset_utils'))
    tree.body = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
                 {'_randomize_robot_dof_state','_reset_table_pose','_reset_object_pose','_reset_goal_pose'}]
    exec(compile(tree,'native-reset','exec'),ref)
    exec(compile(source('action_utils'),'native-wrench','exec'),ref)
    data = yaml.safe_load(args.config.read_text())['env']
    reset_cfg, dr = NS(**data['reset']), NS(**data['domain_randomization'])
    torch.manual_seed(42)
    n = 8
    limits = torch.stack([-torch.rand(29)-.1,torch.rand(29)+.1],dim=-1)
    default = torch.zeros(29)
    cases = []
    for case in ('native_recipe','table_xy_yaw','fixed_poses'):
        cfg = copy.deepcopy(reset_cfg)
        if case == 'table_xy_yaw':
            cfg.table_reset_xy_range_m = [.04,.02]
            cfg.table_reset_yaw_range_deg = 30.
        if case == 'fixed_poses':
            cfg.fixed_start_pose = [.01,-.02,.7,1.,0.,0.,0.]
            cfg.fixed_goal_pose = [.02,.03,.9,1.,0.,0.,0.]
        sampler = SimToolRealReset(cfg,limits,default,'cpu')
        env = NS(cfg=NS(reset=cfg),device='cpu',scene=NS(env_origins=torch.zeros(n,3)),
                 _arm_joint_ids=list(range(7)),_hand_joint_ids=list(range(7,29)),
                 _prev_targets=torch.zeros(n,29),_cur_targets=torch.zeros(n,29),
                 _table_z_per_env=torch.zeros(n),_object_init_z=torch.zeros(n))
        captured = {}
        def write_joint(pos,vel,env_ids):
            captured['joint_pos'],captured['joint_vel'] = pos.clone(),vel.clone()
        def pose_writer(name):
            def write(pose,env_ids):
                captured[name] = pose.clone()
            return write
        def write_velocity(vel,env_ids):
            captured['object_velocity'] = vel.clone()
        env.robot = NS(data=NS(default_joint_pos=default.expand(n,-1),joint_pos_limits=limits.expand(n,-1,-1)),write_joint_state_to_sim=write_joint)
        env.table = NS(write_root_pose_to_sim=pose_writer('table_pose'))
        env.object = NS(write_root_pose_to_sim=pose_writer('object_pose'),write_root_velocity_to_sim=write_velocity)
        env.goal_viz = NS(write_root_pose_to_sim=pose_writer('goal_pose'),data=NS())
        for count in (n,3,1):
            ids = torch.arange(count)
            rng = torch.get_rng_state()
            ref['_randomize_robot_dof_state'](env,ids)
            ref['_reset_table_pose'](env,ids)
            ref['_reset_object_pose'](env,ids)
            ref['_reset_goal_pose'](env,ids,'absolute')
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            actual = sampler.full(count)
            assert torch.equal(torch.get_rng_state(),expected_rng)
            for key in ('joint_pos','joint_vel','table_pose','goal_pose'):
                torch.testing.assert_close(actual[key],captured[key],atol=0,rtol=0)
            torch.testing.assert_close(actual['object_state'],torch.cat([captured['object_pose'],captured['object_velocity']],-1),atol=0,rtol=0)
            torch.testing.assert_close(actual['object_state'][:,2],env._object_init_z[ids],atol=0,rtol=0)
            for step in range(10):
                prev = actual['goal_pose']
                # Native data is indexed globally; this fixture selects the prefix.
                env.goal_viz.data.root_pos_w = prev[:,:3]
                env.goal_viz.data.root_quat_w = prev[:,3:7]
                rng = torch.get_rng_state()
                ref['_reset_goal_pose'](env,ids,cfg.goal_sampling_type)
                expected_rng = torch.get_rng_state()
                torch.set_rng_state(rng)
                actual['goal_pose'] = sampler.goal(count,prev)
                assert torch.equal(torch.get_rng_state(),expected_rng)
                torch.testing.assert_close(actual['goal_pose'],captured['goal_pose'],atol=0,rtol=0)
        cases.append(dict(case=case,passed=True))
    wrench_cases = []
    for case in ('native_recipe','always_fire','zero_decay','ungated'):
        cfg = copy.deepcopy(dr)
        if case == 'zero_decay': cfg.force_decay = cfg.torque_decay = 0.
        if case == 'ungated': cfg.force_only_when_lifted = cfg.torque_only_when_lifted = False
        mass = torch.linspace(.05,1.,n)
        dt = data['sim']['dt'] * data['decimation']
        port = SimToolRealWrench(cfg,mass,dt,'cpu')
        if case == 'always_fire':
            port._random_force_prob[:] = port._random_torque_prob[:] = 1.
        env = NS(**copy.deepcopy(port.__dict__))
        env.cfg = NS(domain_randomization=cfg)
        written = {}
        def apply(forces,torques,is_global):
            assert is_global
            written['force'],written['torque'] = forces.clone(),torques.clone()
        env.object = NS(set_external_force_and_torque=apply)
        for step in range(100):
            if step == 30:
                ids = torch.tensor([1,3])
                rng = torch.get_rng_state()
                env._object_forces[ids] = env._object_torques[ids] = 0
                env._random_force_prob[ids] = ref['sample_log_uniform'](cfg.force_prob_range,len(ids))
                env._random_torque_prob[ids] = ref['sample_log_uniform'](cfg.torque_prob_range,len(ids))
                expected_rng = torch.get_rng_state()
                torch.set_rng_state(rng);port.reset(ids)
                assert torch.equal(torch.get_rng_state(),expected_rng)
            if step == 50:
                saved,rng = port.get_state(),torch.get_rng_state()
                port = SimToolRealWrench(cfg,mass,dt,'cpu');port.set_state(saved)
                torch.set_rng_state(rng)  # Overall continuation restores RNG after construction.
            env._lifted_object = torch.rand(n) > .5
            rng = torch.get_rng_state()
            ref['apply_wrench_dr'](env)
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            force,torque = port.step(env._lifted_object)
            assert torch.equal(torch.get_rng_state(),expected_rng)
            torch.testing.assert_close(force,written['force'],atol=0,rtol=0)
            torch.testing.assert_close(torque,written['torque'],atol=0,rtol=0)
            for field in port._fields:
                torch.testing.assert_close(getattr(port,field),getattr(env,field),atol=0,rtol=0)
        wrench_cases.append(dict(case=case,steps=100,passed=True))
    result = dict(passed=True,reference_commit='8e18764',reset_cases=cases,wrench_cases=wrench_cases,
                  exact_float32_and_rng=True,scope='CPU canonical joint order and zero scene origins; native scene writes replaced by capture, no live contact validation')
    with args.output.open('x') as stream: stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
