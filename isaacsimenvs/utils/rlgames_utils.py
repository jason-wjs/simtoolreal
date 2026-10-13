"""Glue between Isaac Lab DirectRLEnv and our vendored `./rl_games/`.

Uses `isaaclab_rl.RlGamesVecEnvWrapper` as the env-side adapter (it handles obs/
action clipping, sim↔rl device bridging, obs-group routing, gym↔gymnasium
coercion). The training algorithm/Runner/PPO/SAPG code still comes from
`./rl_games/`.
"""

from __future__ import annotations

from simtoolreal_shared.rlgames_observer import EnvStatsAlgoObserver, MultiObserver


def register_rlgames_env(
    env,
    rl_device: str = "cuda:0",
    clip_obs: float = 1e6,
    clip_actions: float = 1e6,
    name: str = "rlgpu",
):
    """Wrap an Isaac Lab env for rl_games and register under `name` ("rlgpu").

    If the env's ``single_observation_space`` includes a ``"teacher_obs"`` key
    (set by the depth-student PegInHoleEnv), the standard RlGamesVecEnvWrapper
    drops it because its hardcoded ``_obs_groups`` only carries ``"policy"``
    into ``"obs"`` and ``"critic"`` into ``"states"``. The DAggerA2CAgent
    needs that 3rd group to label rollouts with the frozen state-MLP teacher.
    Use a small subclass to pass it through as ``rl_games_obs["teacher"]``.
    """
    from rl_games.common import env_configurations, vecenv
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    wrapper_cls = RlGamesVecEnvWrapper
    if "teacher_obs" in env.unwrapped.single_observation_space.spaces:
        class _DAggerRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
            def _process_obs(self, obs_dict):
                rl_games_obs = super()._process_obs(obs_dict)
                # Pass through the teacher actor obs (already clipped by parent).
                rl_games_obs["teacher"] = obs_dict["teacher_obs"]
                return rl_games_obs

            @property
            def teacher_obs_space(self):
                """Teacher view of the central env's obs — same as the wrapper's
                own ``observation_space``/``state_space`` plumbing, but for the
                ``teacher_obs`` key. Returns a finite-bounded gym Box derived
                from the wrapper's clip_obs (which itself came from the agent
                yaml). Build the teacher's env_info from this — never from
                ``unwrapped.single_observation_space`` (those are unbounded).
                """
                import gymnasium as gym
                spec = self.unwrapped.single_observation_space.spaces["teacher_obs"]
                return gym.spaces.Box(-self._clip_obs, self._clip_obs, spec.shape, dtype=spec.dtype)
        wrapper_cls = _DAggerRlGamesVecEnvWrapper

    wrapped = wrapper_cls(env, rl_device, clip_obs, clip_actions)
    # The native adapter inherits no-op environment checkpoint methods. Only
    # the original SimToolReal teacher has the task-state implementation below.
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    gpu_env_cls = RlGamesGpuEnv
    if type(env.unwrapped) is SimToolRealEnv:
        from isaacsimenvs.tasks.simtoolreal.utils import checkpoint_utils

        class SimToolRealGpuEnv(RlGamesGpuEnv):
            def get_env_state(self):
                return checkpoint_utils.get_env_state(self.env.env.unwrapped)

            def set_env_state(self, state):
                checkpoint_utils.set_env_state(self.env.env.unwrapped, state)

        gpu_env_cls = SimToolRealGpuEnv
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kw: gpu_env_cls(config_name, num_actors, **kw),
    )
    env_configurations.register(
        name,
        {
            "vecenv_type": "IsaacRlgWrapper",
            "env_creator": lambda **kw: wrapped,
        },
    )
    return wrapped


def teacher_env_info(wrapped) -> dict:
    """Build the teacher's rl_games env_info from the central wrapper's own
    clipped spaces.

    Source of truth: the agent yaml drives ``clip_observations`` /
    ``clip_actions`` → those flow into ``RlGamesVecEnvWrapper.{observation_space,
    state_space,action_space}`` and our ``teacher_obs_space`` extension here.
    Reading from the wrapper means there's exactly one place that defines
    bounds + shapes; the teacher loader doesn't reinvent them.

    Crucially, the action_space comes from ``wrapped.action_space`` (already
    ``Box(-clip_actions, +clip_actions, ...)``) rather than from
    ``unwrapped.single_action_space`` (``Box(-inf, +inf, ...)``). The latter
    silently NaNs the teacher's actions via ``rescale_actions`` in
    ``PpoPlayerContinuous.get_action``.
    """
    teacher_view = wrapped.teacher_obs_space if hasattr(wrapped, "teacher_obs_space") else wrapped.observation_space
    state_view = wrapped.state_space if wrapped.state_space is not None else teacher_view
    return {
        "observation_space": teacher_view,
        "state_space": state_view,
        "action_space": wrapped.action_space,
        "agents": 1,
        "value_size": 1,
    }
