"""Observe restore writes and the first resumed integration without changing them."""
from isaacgym import gymapi, gymtorch
import hashlib
import json
from pathlib import Path
import runpy
import torch
import profile_gym_physical_candidate as candidate
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal


class Trace:
    def __init__(self, env):
        self.env, self.native = env, env.gym
        self.output = Path.cwd()
        self.steps, self.events = 0, []
        self.object_index = int(env.object_indices[22808])
        torch.save(dict(root=env.root_state_tensor.cpu().clone(), dof=env.dof_state.cpu().clone()), self.output / 'restore_entry.pt')

    def __getattr__(self, name):
        return getattr(self.native, name)

    def write(self, method, sim, states, indices=None, count=None):
        values = gymtorch.wrap_tensor(states)
        row = dict(method=method, after_steps=self.steps, count=count)
        if indices is not None:
            ids = torch.empty(0, dtype=torch.int32) if count == 0 else gymtorch.wrap_tensor(indices)[:count].cpu().clone()
            row.update(indices_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                includes_worst_object=bool((ids == self.object_index).any()))
        if method.startswith('set_actor_root'):
            row['worst_object_input'] = values[self.object_index].cpu().tolist()
        else:
            row['worst_dof_input'] = values.view(24576, 29, 2)[22808].cpu().tolist()
        self.events.append(row)
        if indices is None:
            return getattr(self.native, method)(sim, states)
        return getattr(self.native, method)(sim, states, indices, count)

    def set_actor_root_state_tensor_indexed(self, sim, states, indices, count):
        return self.write('set_actor_root_state_tensor_indexed', sim, states, indices, count)

    def set_dof_state_tensor(self, sim, states):
        return self.write('set_dof_state_tensor', sim, states)

    def set_dof_state_tensor_indexed(self, sim, states, indices, count):
        return self.write('set_dof_state_tensor_indexed', sim, states, indices, count)

    def simulate(self, sim):
        self.steps += 1
        if self.steps == 1:
            torch.save(dict(root=self.env.root_state_tensor.cpu().clone(), dof=self.env.dof_state.cpu().clone()), self.output / 'before_first_integrate.pt')
        self.events.append(dict(method='simulate', step=self.steps))
        return self.native.simulate(sim)

    def after_step(self):
        if self.steps <= 2 or (self.output / 'candidate_physical_stop.json').exists():
            env = self.env
            root, dof = env.root_state_tensor.clone(), env.dof_state.clone()
            try:
                self.native.refresh_actor_root_state_tensor(env.sim)
                self.native.refresh_dof_state_tensor(env.sim)
                torch.save(dict(root=env.root_state_tensor.cpu().clone(), dof=env.dof_state.cpu().clone()), self.output / ('after_step_' + str(self.steps) + '.pt'))
                self.events.append(dict(method='native_after_step', step=self.steps,
                    worst_object_root=env.root_state_tensor[self.object_index].cpu().tolist()))
            finally:
                env.root_state_tensor.copy_(root); env.dof_state.copy_(dof)
        (self.output / 'restore_write_trace.json').write_text(json.dumps(dict(events=self.events,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')


restore = SimToolReal.set_env_state


def traced_restore(env, state):
    observer = Trace(env)
    env._restore_trace = observer
    env.gym = observer
    return restore(env, state)


def step(self, *args, **kwargs):
    try:
        return candidate.step(self, *args, **kwargs)
    finally:
        self.vec_env.env._restore_trace.after_step()


SimToolReal.set_env_state = traced_restore
candidate.profile.A2CAgent.env_step = step

if __name__ == '__main__':
    runpy.run_module('isaacgymenvs.train', run_name='__main__')
