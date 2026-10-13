"""64 real policy steps, native reset-buffer audit, and no optimizer updates."""
from isaacgym import gymapi
import hashlib
import json
from pathlib import Path
import runpy
import torch
import profile_gym_physical_candidate as candidate
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal


reset = SimToolReal.reset_idx


def audited_reset(env, env_ids, reset_buf_idxs=None, episode_reset=True, tensor_reset=True):
    reset(env, env_ids, reset_buf_idxs, episode_reset, tensor_reset)
    if not tensor_reset or not len(env_ids):
        return
    assert reset_buf_idxs is None and env.conditioned_reset_bank is not None
    bank = env.conditioned_reset_bank
    rows = env.conditioned_reset_rows[env_ids]
    assert torch.equal(rows % 1200, env_ids % 1200)
    assert torch.equal(env.root_state_tensor[env.object_indices[env_ids]], bank.object[rows])
    assert torch.equal(env.root_state_tensor[env.table_indices[env_ids]], bank.table[rows])
    assert torch.equal(env.arm_hand_dof_state[env_ids], bank.dof[rows])
    assert torch.equal(env.prev_targets[env_ids], bank.dof[rows, :, 0])
    assert torch.equal(env.cur_targets[env_ids], bank.dof[rows, :, 0])
    assert torch.equal(env.object_init_state[env_ids, 2], bank.table[rows, 2] + env.cfg['env']['tableObjectZOffset'])
    assert torch.equal(env.root_state_tensor[env.goal_object_indices[env_ids], :7], env.goal_states[env_ids, :7])
    assert bool((env.goal_states[env_ids, 2] >= env.object_init_state[env_ids, 2] - .05 + env.lifting_bonus_threshold).all())
    assert not bool(env.rb_forces[env_ids].any()) and not bool(env.rb_torques[env_ids].any())
    if not hasattr(env, '_conditioned_reset_audit'):
        env._conditioned_reset_audit = []
    env._conditioned_reset_audit.append(dict(count=len(env_ids),
        ids_sha256=hashlib.sha256(env_ids.cpu().numpy().tobytes()).hexdigest(),
        rows_sha256=hashlib.sha256(rows.cpu().numpy().tobytes()).hexdigest(),
        state_goal_height_and_target_checks_passed=True))


class Capture:
    def __init__(self, env):
        self.env, self.native, self.before = env, env.gym, None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def snapshot(self):
        env = self.env
        staged_root, staged_dof = env.root_state_tensor.clone(), env.dof_state.clone()
        try:
            self.native.refresh_actor_root_state_tensor(env.sim)
            self.native.refresh_dof_state_tensor(env.sim)
            return dict(native_root=env.root_state_tensor.clone(), native_dof=env.dof_state.clone(),
                targets=env.cur_targets.clone(), bank_rows=env.conditioned_reset_rows.clone(),
                object_init_state=env.object_init_state.clone(), goal_states=env.goal_states.clone())
        finally:
            env.root_state_tensor.copy_(staged_root); env.dof_state.copy_(staged_dof)

    def simulate(self, sim):
        self.before = self.snapshot()
        return self.native.simulate(sim)


def rollout(self):
    env = self.vec_env.env
    output = Path(self.config['train_dir'])
    assert env.conditioned_reset_bank is not None and self.horizon_length == 16
    observer = Capture(env)
    env.gym = observer
    report = dict(status='running', bank_sha256=env.conditioned_reset_bank.sha256,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='64 native policy steps with no optimizer updates; one explicit partial reset after16steps. Original action/observation histories and target sampling retained. Per-step native joint guard, no in-flight GPU contact-depth certification.')
    before = {k: v.detach().clone() for k, v in self.model.named_parameters()}
    try:
        self.vec_env.set_train_info(self.frame, self)
        self.set_eval()
        with torch.no_grad():
            for block in range(4):
                if block == 1:
                    ids = torch.arange(0, env.num_envs, 97, device=env.device)
                    env.reset_buf[ids] = 1
                    self.dones[ids] = 1
                    report['forced_partial_reset_count'] = len(ids)
                batch, extras = self.play_steps()
                del batch, extras
        assert all(torch.equal(v, before[k]) for k, v in self.model.named_parameters())
        report.update(status='completed_no_update', steps=self._candidate_step, parameters_unchanged=True)
        assert report['steps'] == 64
        assert any(0 < row['count'] < env.num_envs for row in env._conditioned_reset_audit)
    except BaseException as error:
        report.update(status='stopped', error=repr(error), steps=getattr(self, '_candidate_step', 0))
        if observer.before is not None:
            after = observer.snapshot()
            torch.save({label: {k: v.cpu() for k, v in state.items()} for label, state in
                [('pre_integrate', observer.before), ('after_step', after)]}, output / 'conditioned_stop_states.pt')
        raise
    finally:
        report['resets'] = getattr(env, '_conditioned_reset_audit', [])
        (output / 'conditioned_rollout.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        env.gym = observer.native
        env.gym.destroy_sim(env.sim)
    raise SystemExit(0)


SimToolReal.reset_idx = audited_reset
candidate.profile.A2CAgent.train_epoch = rollout

if __name__ == '__main__':
    runpy.run_module('isaacgymenvs.train', run_name='__main__')
