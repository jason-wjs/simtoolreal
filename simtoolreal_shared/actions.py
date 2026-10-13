"""Original SimToolReal Torch action math shared by native Lab and mjlab."""
from __future__ import annotations

import math
import torch


def sample_log_uniform(lo_hi: tuple[float, float], n: int) -> torch.Tensor:
    """Log-uniform sample on CPU (caller moves to device)."""
    lo, hi = lo_hi
    return torch.exp(
        torch.empty(n).uniform_(math.log(lo + 1e-12), math.log(hi + 1e-12))
    )


def apply_action_pipeline(env, actions: torch.Tensor) -> None:
    """Apply delay, canonical-to-Lab mapping, smoothing, and target clamps."""
    # Debug replay path accepts an already Lab-order target.
    replay_target = getattr(env, "_replay_target_lab_order", None)
    if replay_target is not None:
        env._cur_targets[:] = replay_target
        env._prev_targets = env._cur_targets.clone()
        return

    dr = env.cfg.domain_randomization
    act_cfg = env.cfg.action
    dt = env.step_dt  # policy step (= decimation * sim.dt = 1/60 s)

    actions = actions.clone().to(env.device)

    # Canonical policy order -> Lab parser order.
    actions = actions[:, env._perm_canon_to_lab]

    # Action delay.
    if dr.use_action_delay and dr.action_delay_max > 0:
        episode_start = (env.episode_length_buf == 0) & (env._successes == 0)
        if episode_start.any():
            env._action_queue[episode_start] = actions[episode_start].unsqueeze(1)
        env._action_queue = torch.roll(env._action_queue, shifts=1, dims=1)
        env._action_queue[:, 0, :] = actions
        delay_idx = torch.randint(
            0, env._action_queue.shape[1], (env.num_envs,), device=env.device
        )
        actions = env._action_queue[
            torch.arange(env.num_envs, device=env.device), delay_idx
        ]

    # Arm: velocity-delta accumulator.
    arm_action = actions[:, :7]
    arm_raw = env._prev_targets[:, :7] + act_cfg.dof_speed_scale * dt * arm_action
    arm_raw = torch.clamp(arm_raw, env._arm_lower, env._arm_upper)
    arm_smoothed = (
        act_cfg.arm_moving_average * arm_raw
        + (1.0 - act_cfg.arm_moving_average) * env._prev_targets[:, :7]
    )
    arm_smoothed = torch.clamp(arm_smoothed, env._arm_lower, env._arm_upper)

    # Hand: absolute [-1, 1] scale.
    hand_action = actions[:, 7:]
    hand_raw = env._hand_lower + 0.5 * (hand_action + 1.0) * (
        env._hand_upper - env._hand_lower
    )
    hand_smoothed = (
        act_cfg.hand_moving_average * hand_raw
        + (1.0 - act_cfg.hand_moving_average) * env._prev_targets[:, 7:]
    )
    hand_smoothed = torch.clamp(hand_smoothed, env._hand_lower, env._hand_upper)

    # Write Lab-order targets and cache them for the next step.
    env._cur_targets[:, env._arm_joint_ids] = arm_smoothed
    env._cur_targets[:, env._hand_joint_ids] = hand_smoothed
    env._prev_targets = env._cur_targets.clone()


