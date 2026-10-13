"""Capture one native full-scale training step before any optimizer update."""
from isaacgym import gymapi
import hashlib
import json
from pathlib import Path
import runpy
import torch
import profile_gym_training as profile


class Observer:
    def __init__(self, env, output):
        self.env, self.native, self.output = env, env.gym, output
        self.records = []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def snapshot(self, label):
        env = self.env
        staged_root, staged_dof = env.root_state_tensor.clone(), env.dof_state.clone()
        try:
            self.native.refresh_actor_root_state_tensor(env.sim)
            self.native.refresh_dof_state_tensor(env.sim)
            tensors = dict(staged_root=staged_root.cpu(), staged_dof=staged_dof.cpu(),
                           native_root=env.root_state_tensor.cpu().clone(),
                           native_dof=env.dof_state.cpu().clone(), targets=env.cur_targets.cpu().clone(),
                           lower=env.arm_hand_dof_lower_limits.cpu(), upper=env.arm_hand_dof_upper_limits.cpu(),
                           object_indices=env.object_indices.cpu())
            q = tensors['native_dof'].view(env.num_envs, -1, 2)[:, :env.num_hand_arm_dofs, 0]
            excess = torch.maximum(tensors['lower'] - q, q - tensors['upper']).clamp_min(0)
            worst = int(excess.argmax())
            world, joint = divmod(worst, env.num_hand_arm_dofs)
            assert all(torch.isfinite(t).all() for t in tensors.values())
            torch.save(tensors, self.output / (label + '.pt'))
            self.records.append(dict(label=label, max_joint_excess_rad=float(excess.max()),
                worlds_above_0_05rad=int((excess.amax(dim=1) > .05).sum()), world=world, joint=joint))
        finally:
            env.root_state_tensor.copy_(staged_root)
            env.dof_state.copy_(staged_dof)

    def simulate(self, sim):
        self.snapshot('pre_integrate')
        self.native.simulate(sim)
        self.native.fetch_results(sim, True)
        self.snapshot('post_integrate')


def step(self, *args, **kwargs):
    env = self.vec_env.env
    assert env.num_envs == 24576 and env.control_freq_inv == 1 and not env.randomize
    output = Path(self.config['train_dir'])
    observer = Observer(env, output)
    observer.snapshot('before_env_step')
    torch.save(args[0].detach().cpu(), output / 'policy_actions.pt')
    env.gym = observer
    try:
        profile.step(self, *args, **kwargs)
        observer.snapshot('after_env_step')
    finally:
        env.gym = observer.native
    params = env.gym.get_sim_params(env.sim)
    (output / 'initial_training_trace.json').write_text(json.dumps(dict(
        status='completed_one_step_no_update', records=observer.records, num_envs=env.num_envs,
        object_assets=[str(p) for p in env.object_asset_files],
        dt=params.dt, substeps=params.substeps, position_iterations=params.physx.num_position_iterations,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='One native training env_step, before updates. Native readback and original command staging retained separately; no live GPU-pipeline contact-depth query.'), indent=2) + '\n')
    env.gym.destroy_sim(env.sim)
    raise SystemExit(0)


profile.A2CAgent.env_step = step

if __name__ == '__main__':
    runpy.run_module('isaacgymenvs.train', run_name='__main__')
