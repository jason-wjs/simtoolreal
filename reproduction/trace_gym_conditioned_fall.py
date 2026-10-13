"""Retain the last32 actual policy-driven states before the observed ground impact."""
from isaacgym import gymapi
import hashlib
from pathlib import Path
import runpy
import torch
import profile_gym_physical_candidate as candidate


def step(self, *args, **kwargs):
    result = candidate.step(self, *args, **kwargs)
    if self._candidate_step < 97:
        return result
    env = self.vec_env.env
    assert env.num_envs == 24576 and env.conditioned_reset_bank is not None
    root, dof, bodies = env.root_state_tensor.clone(), env.dof_state.clone(), env.rigid_body_states.clone()
    try:
        for kind in ('actor_root', 'dof', 'rigid_body'):
            getattr(env.gym, 'refresh_' + kind + '_state_tensor')(env.sim)
        world = 19396
        row = dict(step=self._candidate_step, world=world,
            root=env.root_state_tensor.view(24576,4,13)[world].cpu().clone(),
            dof=env.dof_state.view(24576,29,2)[world].cpu().clone(),
            bodies=env.rigid_body_states[world].cpu().clone(),
            targets=env.cur_targets[world].cpu().clone(),
            pending_reset=int(env.reset_buf[world]), progress=int(env.progress_buf[world]))
        if not hasattr(self, '_fall_states'):
            self._fall_states = []
        self._fall_states.append(row)
        torch.save(dict(world=world, states=self._fall_states,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Actual GPU policy states after each control step97..128; preserve all staged tensors. No applied state/action changes.'),
            Path(self.config['train_dir']) / 'fall_trajectory.pt')
    finally:
        env.root_state_tensor.copy_(root); env.dof_state.copy_(dof); env.rigid_body_states.copy_(bodies)
    return result


candidate.profile.A2CAgent.env_step = step

if __name__ == '__main__':
    runpy.run_module('isaacgymenvs.train', run_name='__main__')
