"""Finite native-teacher lifecycle rollout, without learner or convergence claims."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-assets', type=Path, required=True)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import warp as wp
    from omegaconf import OmegaConf
    from mjlabenvs.task import SimToolRealTask

    torch.manual_seed(42)
    start = time.monotonic()
    task = SimToolRealTask(OmegaConf.load(args.config).env,args.robot_assets,args.tools,1536,'cuda:0')
    report = dict(num_envs=1536,object_pool=1200,planned_steps=64,completed_steps=0,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        setup_seconds=time.monotonic()-start,max_episode_length=task.max_episode_length,
        max_depth_m=0.,max_joint_limit_violation_rad=0.,terminated=0,truncated=0,
        max_consecutive_substeps_over_5mm=0,all_finite=True)
    for ids,group in task.scene.groups:
        group.deep_streak = torch.zeros(len(ids),dtype=torch.long,device=task.device)
        def audit(original=group.sim.step,group=group):
            original()
            q=group.robot.data.joint_pos[:,group.joint_ids]
            limits=task.scene.joint_limits
            violation=torch.maximum((limits[:,0]-q).clamp_min(0),(q-limits[:,1]).clamp_min(0))
            report['max_joint_limit_violation_rad']=max(report['max_joint_limit_violation_rad'],float(violation.max()))
            data=group.sim.wp_data
            count=int(wp.to_torch(data.nacon)[0])
            assert count<=data.naconmax
            maximum=torch.zeros(len(group.deep_streak),device=task.device)
            if count:
                distances=wp.to_torch(data.contact.dist)[:count]
                assert bool(torch.isfinite(distances).all())
                worlds=wp.to_torch(data.contact.worldid)[:count].long()
                maximum.scatter_reduce_(0,worlds,(-distances).clamp_min(0),reduce='amax',include_self=True)
            group.deep_streak=(group.deep_streak+1)*(maximum>.005)
            report['max_depth_m']=max(report['max_depth_m'],float(maximum.max()))
            report['max_consecutive_substeps_over_5mm']=max(report['max_consecutive_substeps_over_5mm'],int(group.deep_streak.max()))
        group.sim.step=audit
    obs,_=task.reset()
    assert obs['policy'].shape==(1536,140) and obs['critic'].shape==(1536,162)
    torch.save(task.scene.state(),args.output/'initial.pt')
    actions=torch.zeros(1536,29,device=task.device)
    rollout=time.monotonic()
    for step in range(64):
        obs,reward,terminated,truncated,extras=task.step(actions)
        state=task.scene.state()
        assert all(bool(torch.isfinite(v).all()) for v in [*state.values(),*obs.values(),reward])
        q=state['joint_pos'];limits=task.scene.joint_limits
        violation=torch.maximum((limits[:,0]-q).clamp_min(0),(q-limits[:,1]).clamp_min(0))
        report['max_joint_limit_violation_rad']=max(report['max_joint_limit_violation_rad'],float(violation.max()))
        report['completed_steps']=step+1
        report['terminated']+=int(terminated.sum());report['truncated']+=int(truncated.sum())
        report['last_reward_mean']=float(reward.mean())
        if step%8==0:print(json.dumps(report),flush=True)
        # A single impact sample is distinct from sustained contact. End this
        # bounded diagnostic on persistent >5mm depth or >0.05rad violation.
        if report['max_consecutive_substeps_over_5mm']>=12 or report['max_joint_limit_violation_rad']>.05:
            report['stop_reason']='physical_review_required'
            break
    torch.cuda.synchronize()
    report['rollout_seconds']=time.monotonic()-rollout
    report['elapsed_seconds']=time.monotonic()-start
    report['physical_review_required']=report['max_depth_m']>.005 or report['max_joint_limit_violation_rad']>.01
    report['scope']='Full1200tool/1536env native teacher resets, delayed zero action targets, observation/reward/done/autoreset and wrench. No learner/update/checkpoint recovery or contact acceptance claim.'
    torch.save(task.scene.state(),args.output/'final.pt')
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
