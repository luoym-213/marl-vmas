"""Return equations and mode guards for high-level semi-Markov PPO."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


STRICT_SMDP = "strict_smdp"
LEGACY_EVENT_STEP = "legacy_event_step"
RETURN_MODES = (STRICT_SMDP, LEGACY_EVENT_STEP)


def discounted_option_reward(rewards: Sequence[float] | Tensor, gamma: float) -> Tensor:
    """Return ``sum_l gamma**l * reward[l]`` using environment-step time."""

    values = torch.as_tensor(rewards, dtype=torch.float64)
    powers = torch.arange(values.numel(), dtype=torch.float64, device=values.device)
    return (values.flatten() * torch.pow(torch.as_tensor(gamma, dtype=torch.float64), powers)).sum()


def add_smdp_gae(
    batch: dict[str, Tensor],
    *,
    gamma: float,
    gae_lambda: float,
    return_mode: str,
    time_limit_bootstrap: bool = True,
) -> None:
    """Add advantages and returns to a flat asynchronous transition batch.

    Strict mode uses discounted option rewards already stored in ``reward``,
    explicit end-state values, ``gamma**duration`` bootstrap, and
    ``(gamma * lambda)**duration`` trace decay.  Truncation may bootstrap but
    always cuts the GAE trace.  Legacy mode exactly preserves the historical
    event-step recurrence: undiscounted option reward and
    ``gamma**duration * lambda``.
    """

    if return_mode not in RETURN_MODES:
        raise ValueError(f"unknown high-level return mode: {return_mode!r}")
    n = batch["reward"].shape[0]
    advantages = torch.zeros(n, 1, device=batch["reward"].device)
    returns = torch.zeros_like(advantages)
    env_ids = batch["env_id"]
    agent_ids = batch["agent_id"]
    starts = batch["decision_start_t"]

    for env_id in env_ids.unique():
        for agent_id in agent_ids[env_ids == env_id].unique():
            mask = (env_ids == env_id) & (agent_ids == agent_id)
            indices = torch.nonzero(mask, as_tuple=False).view(-1)
            if indices.numel() == 0:
                continue
            indices = indices[starts[indices].argsort()]
            next_advantage = torch.zeros(1, 1, device=batch["reward"].device)
            legacy_next_value = torch.zeros_like(next_advantage)
            for idx in reversed(indices.tolist()):
                duration = batch["duration"][idx : idx + 1]
                discount = gamma ** duration
                if return_mode == LEGACY_EVENT_STEP:
                    done_mask = 1.0 - batch["done"][idx : idx + 1]
                    delta = (
                        batch["reward"][idx : idx + 1]
                        + discount * legacy_next_value * done_mask
                        - batch["old_value"][idx : idx + 1]
                    )
                    advantage = (
                        delta
                        + discount * gae_lambda * next_advantage * done_mask
                    )
                    legacy_next_value = batch["old_value"][idx : idx + 1]
                else:
                    terminal = batch["terminal"][idx : idx + 1]
                    truncated = batch["time_limit_truncated"][idx : idx + 1]
                    environment_done = batch["environment_done"][idx : idx + 1]
                    valid = batch["train_active_mask"][idx : idx + 1]
                    bootstrap_mask = 1.0 - terminal
                    if not time_limit_bootstrap:
                        bootstrap_mask = bootstrap_mask * (1.0 - truncated)
                    delta = (
                        batch["reward"][idx : idx + 1]
                        + discount
                        * bootstrap_mask
                        * batch["next_value"][idx : idx + 1]
                        - batch["old_value"][idx : idx + 1]
                    )
                    trace_mask = (
                        (1.0 - terminal)
                        * (1.0 - truncated)
                        * (1.0 - environment_done)
                    )
                    trace_discount = (gamma * gae_lambda) ** duration
                    advantage = delta + trace_discount * trace_mask * next_advantage
                    advantage = advantage * valid
                advantages[idx : idx + 1] = advantage
                returns[idx : idx + 1] = (
                    advantage + batch["old_value"][idx : idx + 1]
                )
                next_advantage = advantage

    batch["advantage"] = advantages
    batch["return"] = returns
