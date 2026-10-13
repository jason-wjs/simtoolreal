"""Native contact queries at saved GPU-policy poses, without trajectory replay.

CPU tensor pipeline + GPU PhysX, original cooked scene, one1us substep per
independently installed pose. Verify native pre-query body poses and report
post-query drift separately. This is not a training or dynamical rollout.
"""
from isaacgym import gymapi, gymtorch
import torch
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from dextoolbench import eval_isaacgym as evaluator
from deployment.isaac.isaac_env import create_env_from_cfg, merge_cfg_with_default_config
from deployment.rl_player_utils import read_cfg_omegaconf
from check_gym_policy_replay import snapshot, scene_properties


def pose_error(actual, expected):
    a, b = torch.tensor(actual, dtype=torch.float64).reshape(-1, 13), torch.tensor(expected, dtype=torch.float64).reshape(-1, 13)
    qa, qb = a[:, 3:7], b[:, 3:7]
    qa, qb = qa/qa.norm(dim=-1, keepdim=True), qb/qb.norm(dim=-1, keepdim=True)
    angle = 4*torch.asin(torch.minimum((qa-qb).norm(dim=-1), (qa+qb).norm(dim=-1)).clamp(max=2)/2)
    return dict(position_m=float((a[:, :3]-b[:, :3]).abs().max()), rotation_rad=float(angle.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    assert source['mode']=='capture' and source['steps']==602
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    report = dict(status='running', source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[],
        scope='Original GPU-policy saved poses, CPU tensor pipeline with GPU PhysX. Each query reinstalls pose and zeros velocities, holds current DOF targets, then one1us substep. Native initialOverlap belongs to this query start; no policy rollout or native GPU-pipeline contact claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, allow_nan=False)+'\n')

    def create_scene(config_path, device, headless, overrides):
        cfg=merge_cfg_with_default_config(read_cfg_omegaconf(config_path, device))
        cfg.task.sim.physx.use_gpu=True
        overrides=dict(overrides)
        overrides.update({'pipeline':'cpu','task.sim.physx.contact_collection':1,
                          'task.sim.dt':1e-6,'task.sim.substeps':1})
        return create_env_from_cfg(cfg, headless=headless, overrides=overrides)

    class Runner:
        def __init__(self, env, **unused):
            self.env=env

        def run_eval(self, num_episodes):
            env=self.env
            assert env.device=='cpu' and env.sim_params.physx.use_gpu
            assert scene_properties(env)==source['properties']
            report.update(native_properties_match=True, dt=env.sim_params.dt, substeps=env.sim_params.substeps)
            try:
                for saved in source['trace']:
                    root=torch.tensor(saved['root'], dtype=torch.float32)
                    dof=torch.tensor(saved['dof'], dtype=torch.float32)
                    root[:,7:]=0.; dof[:,1]=0.
                    target=dof[:,0].contiguous()
                    assert env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(root))
                    assert env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(dof))
                    assert env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(target))
                    before=snapshot(env)
                    error=pose_error(before['bodies'], saved['bodies'])
                    row=dict(step=saved['step'], pre_query_body_error=error)
                    report['cases'].append(row)
                    if error['position_m']>1e-5 or error['rotation_rad']>1e-4:
                        row.update(expected=saved, actual_before=before)
                        raise RuntimeError('Native body poses do not match saved GPU poses before query')
                    env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
                    after=snapshot(env)
                    row['post_query_body_drift']=pose_error(after['bodies'], before['bodies'])
                    contacts=env.gym.get_env_rigid_contacts(env.envs[0])
                    row['contacts']=[{k:int(c[k]) if k in ['body0','body1'] else float(c[k])
                                      for k in ['body0','body1','initialOverlap','minDist']} for c in contacts]
                    if saved['step'] in [0,78,108,240,420,601]:
                        # Clear object/table and object/robot geometry between queries;
                        # subsequent saved poses also test recovery after a large teleport.
                        clear=root.clone(); clear[1,2]+=3.
                        assert env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(clear))
                        assert env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(dof))
                        env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
                        negative=env.gym.get_env_rigid_contacts(env.envs[0])
                        object_rows=[c for c in negative if 30 in [int(c['body0']),int(c['body1'])]]
                        row['clear_object_control_count']=len(object_rows)
                        assert not object_rows, 'Teleported negative control retains object contacts'
                report['status']='completed'
            except BaseException as error:
                report.update(status='stopped', error=repr(error))
                raise
            finally:
                save()
                env.gym.destroy_sim(env.sim)

    evaluator.create_env=create_scene
    evaluator.EvalRunner=Runner
    evaluator.tyro.cli=lambda _: evaluator.EvalArgs(object_category='screwdriver', object_name='short_screwdriver',
        task_name='spin_horizontal', config_path=Path('pretrained_policy/config.yaml'), checkpoint_path=Path('pretrained_policy/model.pth'))
    evaluator.main()
    print('SAVED_POSE_CONTACTS_COMPLETE',len(report['cases']),flush=True)


if __name__=='__main__':
    main()
