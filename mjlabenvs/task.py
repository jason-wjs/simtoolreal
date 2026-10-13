"""Native SimToolReal Lab teacher lifecycle on the concrete mjlab scene."""
import math
import json
import random

import numpy as np
import torch

from mjlabenvs.control import SimToolRealControl
from mjlabenvs.observations import SimToolRealObservations
from mjlabenvs.progress import SimToolRealProgress
from mjlabenvs.randomization import SimToolRealReset, SimToolRealWrench
from mjlabenvs.scene import SimToolRealScene, PHYSICS_DT, DECIMATION


class SimToolRealTask:
    def __init__(self, cfg, robot_assets, tool_root, num_envs, device):
        self.cfg, self.num_envs, self.device = cfg, num_envs, device
        assert cfg.sim.dt == PHYSICS_DT and cfg.decimation == DECIMATION
        assert cfg.events is None and cfg.action_noise_model is None and cfg.observation_noise_model is None
        self.step_dt = PHYSICS_DT * DECIMATION
        self.max_episode_length = math.ceil(cfg.episode_length_s / self.step_dt)
        self.scene = SimToolRealScene(robot_assets, tool_root, num_envs, device)
        limits, default = self.scene.joint_limits, self.scene.default_joint_pos
        self.control = SimToolRealControl(cfg, limits, default, num_envs, device, self.step_dt)
        self.observations = SimToolRealObservations(cfg, self.scene.object_scales, limits, device)
        lo, hi = cfg.domain_randomization.object_scale_noise_multiplier_range
        self.observations._object_scale_multiplier.uniform_(lo, hi)
        self.progress = SimToolRealProgress(cfg, num_envs, device, self.max_episode_length)
        self.sampler = SimToolRealReset(cfg.reset, limits, default, device)
        self.wrench = SimToolRealWrench(cfg.domain_randomization, self.scene.object_mass, self.step_dt, device)
        self.reward_buf = torch.zeros(num_envs, device=device)
        self.common_step_counter = 0

    def _reset_idx(self, ids):
        reset = self.sampler.full(ids.numel())
        self.scene.reset(ids, **reset)
        self.control.reset(ids, reset['joint_pos'])
        self.progress.reset(ids, reset['object_state'][:, 2])
        self.wrench.reset(ids)
        lo, hi = self.cfg.domain_randomization.object_scale_noise_multiplier_range
        scale = torch.empty(ids.numel(), 3, device=self.device).uniform_(lo, hi)
        self.observations.reset(ids, scale)

    def _observe(self):
        return self.observations.build(self.scene.state(), self.progress,
                                       self.control._prev_targets, self.reward_buf)

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._observe(), {}

    def step(self, actions):
        p = self.progress
        targets = self.control.targets(actions, p.episode_length_buf, p._successes)
        force, torque = self.wrench.step(p._lifted_object)
        self.scene.step(targets, force, torque)
        p.episode_length_buf += 1
        self.common_step_counter += 1
        state = self.scene.state()
        fingertips, keypoints = self.observations.distances(state)
        terminated, truncated, goal_ids = p.dones(state['object_pose'][:, 2], fingertips, keypoints)
        # Native goal sampling happens before rewards, even when the same env
        # is about to end and immediately receive a full reset.
        if goal_ids.numel():
            poses = self.sampler.goal(goal_ids.numel(), state['goal_pose'][goal_ids])
            self.scene.write_goals(goal_ids, poses)
        self.reward_buf = p.rewards(state['object_pose'][:, 2], state['joint_vel'])
        maximum = self.cfg.termination.max_consecutive_successes
        all_goals = p._successes >= maximum if maximum > 0 else torch.zeros_like(p._successes, dtype=torch.bool)
        final = dict(successes=p._successes.float(), all_goals_hit=all_goals.float())
        final.update({f'done_{name}': value.float() for name, value in p._termination_reasons.items()})
        extras = dict(episode_cumulative=p._reward_terms, episode_final=final,
                      successes=p._prev_episode_successes.float(),
                      current_success_tolerance=float(p._current_success_tolerance))
        ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            self._reset_idx(ids)
        return self._observe(), self.reward_buf, terminated, truncated, extras

    def get_state(self):
        return dict(config=json.dumps(self.cfg,default=vars,sort_keys=True),
            scene=self.scene.get_state(), control=self.control.get_state(),
            progress=self.progress.get_state(), observations=self.observations.get_state(),
            wrench=self.wrench.get_state(), reward=self.reward_buf.clone(),
            common_step_counter=self.common_step_counter,
            rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                     torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))

    def set_state(self, state):
        assert state['config'] == json.dumps(self.cfg,default=vars,sort_keys=True)
        for name in ('control','progress','observations','wrench'):
            component=getattr(self,name)
            saved={key:value.to(self.device) if isinstance(value,torch.Tensor) else value
                   for key,value in state[name].items()}
            component.set_state(saved)
        self.reward_buf.copy_(state['reward'].to(self.device))
        self.common_step_counter=state['common_step_counter']
        self.scene.set_state(state['scene'],self.control._cur_targets,
                             self.wrench._object_forces,self.wrench._object_torques)
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state['rng']['cuda']])
