"""Capture one native full-scale training step before any optimizer update."""
from isaacgym import gymapi, gymtorch
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import torch
import profile_gym_training as profile
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from check_gym_pipeline_replay import scene_properties


original_after_created = SimToolReal._after_envs_created


def preserve_assets(env):
    original_after_created(env)
    # Hydra has already changed cwd to this run's directory. Preserve the
    # actually loaded files before the task deletes its temporary directory.
    directory = Path.cwd() / 'trace_assets'
    directory.mkdir()
    env._trace_asset_hashes = []
    for index, source in enumerate(env.object_asset_files):
        target = directory / (str(index) + '.urdf')
        shutil.copyfile(source, target)
        env._trace_asset_hashes.append(hashlib.sha256(target.read_bytes()).hexdigest())


SimToolReal._after_envs_created = preserve_assets


class Observer:
    def __init__(self, env, output):
        self.env, self.native, self.output = env, env.gym, output
        self.records = []
        self.wrenches = []
        self.root_writes = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def apply_rigid_body_force_tensors(self, sim, forces, torques, space):
        path = self.output / ('wrench_' + str(len(self.wrenches)) + '.pt')
        torch.save(dict(forces=gymtorch.wrap_tensor(forces).cpu().clone(),
                        torques=gymtorch.wrap_tensor(torques).cpu().clone()), path)
        self.wrenches.append(dict(file=path.name, space=int(space)))
        return self.native.apply_rigid_body_force_tensors(sim, forces, torques, space)

    def snapshot(self, label):
        env = self.env
        staged_root, staged_dof = env.root_state_tensor.clone(), env.dof_state.clone()
        try:
            self.native.refresh_actor_root_state_tensor(env.sim)
            self.native.refresh_dof_state_tensor(env.sim)
            self.native.refresh_rigid_body_state_tensor(env.sim)
            tensors = dict(staged_root=staged_root.cpu(), staged_dof=staged_dof.cpu(),
                           native_root=env.root_state_tensor.cpu().clone(),
                           native_dof=env.dof_state.cpu().clone(), targets=env.cur_targets.cpu().clone(),
                           lower=env.arm_hand_dof_lower_limits.cpu(), upper=env.arm_hand_dof_upper_limits.cpu(),
                           object_indices=env.object_indices.cpu(),
                           selected_bodies=env.rigid_body_states[[12965, 15208]].cpu().clone())
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

    def set_actor_root_state_tensor_indexed(self, sim, states, actor_indices, count):
        self.root_writes += 1
        if os.environ.get('DEXHAND_INITIAL_OBJECT_CLEARANCE') == '1':
            # Intervene inside the task's original batch, preserving its index
            # set and call count. A second setter can replace pending GPU writes.
            assert self.root_writes == 1
            indices = self.env.object_indices[[12965, 15208]].long()
            submitted = gymtorch.wrap_tensor(actor_indices)[:count]
            assert all(bool((submitted == i).any()) for i in indices)
            gymtorch.wrap_tensor(states)[indices, 2] += 3.
        return self.native.set_actor_root_state_tensor_indexed(sim, states, actor_indices, count)

    def simulate(self, sim):
        assert self.root_writes == 1
        self.snapshot('pre_integrate')
        self.native.simulate(sim)
        self.native.fetch_results(sim, True)
        self.snapshot('post_integrate')


def step(self, *args, **kwargs):
    env = self.vec_env.env
    assert env.num_envs == 24576 and env.control_freq_inv == 1 and not env.randomize
    output = Path(self.config['train_dir'])
    assert (output / 'trace_assets/0.urdf').is_file()
    digest = profile.static_scene_digest(env)
    properties = {str(i): scene_properties(env, i) for i in [12965, 15208]}
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
        asset_sha256=env._trace_asset_hashes, native_scene_digest=digest,
        selected_worlds=[12965, 15208], selected_properties=properties,
        wrenches=observer.wrenches,
        root_write_count=observer.root_writes,
        object_clearance_worlds=[12965, 15208] if os.environ.get('DEXHAND_INITIAL_OBJECT_CLEARANCE') == '1' else [],
        dt=params.dt, substeps=params.substeps, position_iterations=params.physx.num_position_iterations,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='One native training env_step, before updates. Native readback and original command staging retained separately; no live GPU-pipeline contact-depth query.'), indent=2) + '\n')
    env.gym.destroy_sim(env.sim)
    raise SystemExit(0)


profile.A2CAgent.env_step = step

if __name__ == '__main__':
    runpy.run_module('isaacgymenvs.train', run_name='__main__')
