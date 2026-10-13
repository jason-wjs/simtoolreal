"""Finite full-scale4-substep/128-iteration candidate with native joint screening."""
from isaacgym import gymapi  # Must precede torch imports.
import hashlib
import json
from pathlib import Path
import runpy
import torch
import profile_gym_training as profile


def step(self, *args, **kwargs):
    env = self.vec_env.env
    output = Path(self.config['train_dir'])
    if not hasattr(self, '_candidate_step'):
        params = env.gym.get_sim_params(env.sim)
        assert env.num_envs == 24576 and self.intr_coef_block_size == 4096
        assert params.substeps == 4 and params.physx.num_position_iterations == 128
        # Native training leaves the explicit PhysX flag at its default; Gym
        # forces GPU PhysX when the GPU tensor pipeline is enabled.
        assert abs(params.dt - 1 / 60) < 1e-8 and params.use_gpu_pipeline
        assert env.root_state_tensor.is_cuda and env.dof_state.is_cuda
        assert not env.randomize
        self._candidate_step = 0
        (output / 'candidate_runtime.json').write_text(json.dumps(dict(
            num_envs=env.num_envs, object_pool_size=len(env.object_asset_files),
            dt=params.dt, substeps=params.substeps, position_iterations=params.physx.num_position_iterations,
            velocity_iterations=params.physx.num_velocity_iterations,
            gpu_pipeline=params.use_gpu_pipeline, explicit_physx_gpu_flag=params.physx.use_gpu,
            root_device=str(env.root_state_tensor.device), dof_device=str(env.dof_state.device),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            profile_sha256=hashlib.sha256(Path(profile.__file__).read_bytes()).hexdigest(),
            scope='Every training-step native joint readback; preserve noisy task staging. No GPU-native contact-depth certification.')) + '\n')
    result = profile.step(self, *args, **kwargs)
    self._candidate_step += 1
    staged = env.dof_state.clone()
    try:
        env.gym.refresh_dof_state_tensor(env.sim)
        q = env.dof_state.view(env.num_envs, -1, 2)[:, :env.num_hand_arm_dofs, 0]
        assert torch.isfinite(q).all()
        excess = torch.maximum(env.arm_hand_dof_lower_limits - q, q - env.arm_hand_dof_upper_limits).clamp_min(0)
        flat = int(excess.argmax())
        world, joint = divmod(flat, env.num_hand_arm_dofs)
        maximum = float(excess[world, joint])
        record = dict(step=self._candidate_step, epoch=self.epoch_num, max_joint_excess_rad=maximum,
                      worlds_above_0_01rad=int((excess.amax(dim=1) > .01).sum()),
                      worlds_above_0_05rad=int((excess.amax(dim=1) > .05).sum()))
        with (output / 'candidate_physics.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        if maximum > .05:
            env.gym.refresh_actor_root_state_tensor(env.sim)
            record.update(world=world, joint=joint, q_rad=float(q[world, joint]),
                          lower_rad=float(env.arm_hand_dof_lower_limits[joint]),
                          upper_rad=float(env.arm_hand_dof_upper_limits[joint]),
                          target_rad=float(env.cur_targets[world, joint]),
                          native_dof=env.dof_state.view(env.num_envs, -1, 2)[world].cpu().tolist(),
                          object_root=env.root_state_tensor[env.object_indices[world]].cpu().tolist(),
                          reason='Native joint excess exceeds0.05rad; stop affected candidate before further updates.')
            (output / 'candidate_physical_stop.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
            if env.conditioned_reset_bank is not None:
                # Retain the whole failed scene; a worst-world scalar cannot
                # distinguish a fresh contact failure from a reset/restore issue.
                torch.save(dict(native_root=env.root_state_tensor.cpu().clone(),
                    native_dof=env.dof_state.cpu().clone(), targets=env.cur_targets.cpu().clone(),
                    lower=env.arm_hand_dof_lower_limits.cpu(), upper=env.arm_hand_dof_upper_limits.cpu(),
                    object_indices=env.object_indices.cpu(), table_indices=env.table_indices.cpu(),
                    bank_rows=env.conditioned_reset_rows.cpu(), rb_forces=env.rb_forces.cpu(),
                    rb_torques=env.rb_torques.cpu()), output / 'candidate_physical_stop_state.pt')
            raise RuntimeError('Candidate native joint physical review required')
    finally:
        env.dof_state.copy_(staged)
    assert all(torch.isfinite(value).all() for value in result[0].values())
    return result


profile.A2CAgent.env_step = step

if __name__ == '__main__':
    runpy.run_module('isaacgymenvs.train', run_name='__main__')
