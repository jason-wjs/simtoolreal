"""Concrete SimToolReal adapter for the repository's original SAPG learner."""
import gym
import numpy as np
import torch
from rl_games.common.ivecenv import IVecEnv

from mjlabenvs.observations import compute_obs_dim


class SimToolRealVecEnv(IVecEnv):
    def __init__(self, task, clip_obs, clip_actions):
        self.env = task
        self.clip_obs, self.clip_actions = clip_obs, clip_actions

    def get_env_info(self):
        return dict(observation_space=gym.spaces.Box(-self.clip_obs,self.clip_obs,
                        (compute_obs_dim(self.env.cfg.obs.obs_list),),dtype=np.float32),
                    state_space=gym.spaces.Box(-self.clip_obs,self.clip_obs,
                        (compute_obs_dim(self.env.cfg.obs.state_list),),dtype=np.float32),
                    action_space=gym.spaces.Box(-self.clip_actions,self.clip_actions,(29,),dtype=np.float32),
                    agents=1,value_size=1)

    def _observations(self, obs):
        return dict(obs=obs['policy'].clamp(-self.clip_obs,self.clip_obs),
                    states=obs['critic'].clamp(-self.clip_obs,self.clip_obs))

    def reset(self):
        obs,_=self.env.reset()
        return self._observations(obs)

    def step(self, actions):
        actions=actions.detach().clone().to(self.env.device).clamp(-self.clip_actions,self.clip_actions)
        obs,reward,terminated,truncated,extras=self.env.step(actions)
        if not self.env.cfg.is_finite_horizon:
            extras['time_outs']=truncated
        return self._observations(obs),reward,terminated|truncated,extras

    def get_env_state(self):
        return self.env.get_state()

    def set_env_state(self, state):
        self.env.set_state(state)
