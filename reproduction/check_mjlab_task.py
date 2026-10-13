"""Finite native-teacher lifecycle rollout, without learner or convergence claims."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-assets', type=Path, required=True)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-reset-fixture', type=Path,
                        help='Diagnostic only: reuse vetted per-world reset states on every episode.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import warp as wp
    import yaml
    from mjlabenvs.task import SimToolRealTask

    torch.manual_seed(42)
    start = time.monotonic()
    def namespace(value):
        return SimpleNamespace(**{k: namespace(v) for k,v in value.items()}) if isinstance(value,dict) else value
    cfg=namespace(yaml.safe_load(args.config.read_text())['env'])
    task = SimToolRealTask(cfg,args.robot_assets,args.tools,1536,'cuda:0')
    report = dict(num_envs=1536,object_pool=1200,planned_steps=64,completed_steps=0,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        setup_seconds=time.monotonic()-start,max_episode_length=task.max_episode_length,
        max_depth_m=0.,max_joint_limit_violation_rad=0.,terminated=0,truncated=0,
        max_consecutive_substeps_over_5mm=0,all_finite=True)
    fixture = None
    if args.fixed_reset_fixture is not None:
        fixture = torch.load(args.fixed_reset_fixture, map_location='cpu', weights_only=False)
        metadata = fixture['metadata']
        assert metadata['config_sha256'] == report['config_sha256']
        assert metadata['tools_manifest_sha256'] == hashlib.sha256((args.tools/'manifest.json').read_bytes()).hexdigest()
        assert metadata['worlds'] == 1536 and metadata['tool_pool'] == 1200 and not metadata['unresolved_worlds']
        states = {key:value.to(task.device) for key,value in fixture['reset'].items()}
        native_reset = task._reset_idx
        def fixed_reset(ids):
            # Exercise all original reset hooks while supplying this diagnostic's
            # fixed per-world physical states, including partial episode resets.
            original_sampler = task.sampler.full
            task.sampler.full = lambda count: {key:value[ids] for key,value in states.items()}
            try:
                native_reset(ids)
            finally:
                task.sampler.full = original_sampler
        task._reset_idx = fixed_reset
        report['fixed_reset_fixture'] = dict(path=str(args.fixed_reset_fixture),
            sha256=hashlib.sha256(args.fixed_reset_fixture.read_bytes()).hexdigest(), metadata=metadata,
            scope='Fixed vetted reset per world on every episode. Conditional reset variant, no learner.')
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
    if fixture is not None:
        actual = task.scene.state()
        for key in ['joint_pos', 'joint_vel', 'table_pose', 'goal_pose']:
            torch.testing.assert_close(actual[key], states[key], atol=1e-6, rtol=0)
        torch.testing.assert_close(actual['object_pose'], states['object_state'][:, :7], atol=1e-6, rtol=0)
        maximum = 0.
        for _, group in task.scene.groups:
            data = group.sim.wp_data
            count = int(wp.to_torch(data.nacon)[0])
            assert count <= data.naconmax
            if count:
                distances = wp.to_torch(data.contact.dist)[:count]
                assert bool(torch.isfinite(distances).all())
                maximum = max(maximum, float((-distances).clamp_min(0).max()))
        report['fixed_reset_fixture']['native_initial_max_depth_m'] = maximum
        report['fixed_reset_fixture']['native_state_readback_passed'] = True
        assert maximum <= fixture['metadata']['threshold_m'], maximum
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
        if step == 0:
            # Exercise actual native state writes, without introducing extra
            # simulated steps into the contact diagnostic. Separate-process
            # learner recovery remains a later gate.
            checkpoint=task.get_state()
            torch.save(checkpoint,args.output/'task_state.pt')
            task.progress._successes += 1
            task.observations._obs_queue += 1
            for _,group in task.scene.groups:
                group.sim.data.qvel.add_(.01)
            task.set_state(torch.load(args.output/'task_state.pt',map_location='cpu',weights_only=False))
            restored=task.get_state()
            for name in ('control','progress','observations','wrench'):
                for key,value in checkpoint[name].items():
                    actual=restored[name][key]
                    if isinstance(value,torch.Tensor):
                        torch.testing.assert_close(actual,value,atol=0,rtol=0)
                    else:
                        assert actual==value
            for before,after in zip(checkpoint['scene']['groups'],restored['scene']['groups']):
                for key,value in before['integration'].items():
                    torch.testing.assert_close(after['integration'][key],value,atol=0,rtol=0)
            torch.testing.assert_close(restored['rng']['torch'],checkpoint['rng']['torch'],atol=0,rtol=0)
            for a,b in zip(restored['rng']['cuda'],checkpoint['rng']['cuda']):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            report['same_process_task_restore_exact']=True
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
    report['scope']='Full1200tool/1536env native teacher lifecycle and same-process task-state disk roundtrip. No learner/update/separate-process recovery or contact acceptance claim.'
    if fixture is not None:
        report['scope'] += ' Uses fixed vetted per-world resets, so this is a labeled repair diagnostic, not original-reset reproduction.'
    torch.save(task.scene.state(),args.output/'final.pt')
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
