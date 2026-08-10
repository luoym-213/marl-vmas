"""End-to-end MAPPO baseline for the frozen Chapter 1 SAR task.

This module deliberately excludes the hierarchical interfaces: no RRT nodes,
options, FCSR release, assigned goals, or low-level policy are consumed.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import yaml

from comm_spread.chapter1_config import (
    PROJECT_ROOT,
    chapter1_sar_kwargs,
    config_sha256,
    validate_task,
)
from comm_spread.env_factory import make_sar_env


BASELINE_SCHEMA_VERSION = 1
BASELINE_NAME = "end_to_end_mappo"
DEFAULT_REWARD_CONFIG = {
    "profile": "sparse_v1",
    "discovery_entropy_scale": 1.0,
    "detected_target_progress_scale": 0.0,
    "collision_coefficient": None,
    "collision_floor": None,
    "boundary_penalty": None,
    "active_time_penalty": None,
}


def load_mappo_baseline_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"baseline config must be a mapping: {source}")
    if document.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError("unsupported MAPPO baseline config schema")
    if document.get("baseline", {}).get("name") != BASELINE_NAME:
        raise ValueError(f"baseline.name must be {BASELINE_NAME!r}")

    task_path = Path(document["task_config"])
    if not task_path.is_absolute():
        task_path = (source.parent / task_path).resolve()
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise ValueError(f"task config must be a mapping: {task_path}")
    validate_task(task)

    resolved = deepcopy(document)
    resolved["_source"] = str(source)
    resolved["_task_path"] = str(task_path)
    reward_override = resolved["baseline"].get("reward", {})
    if not isinstance(reward_override, Mapping):
        raise ValueError("baseline.reward must be a mapping")
    resolved["_reward"] = {
        **DEFAULT_REWARD_CONFIG,
        **deepcopy(dict(reward_override)),
    }
    resolved["_task"] = task
    resolved["_task_sha256"] = config_sha256(task, exclude_volatile=True)
    resolved["_config_sha256"] = config_sha256(
        {key: value for key, value in document.items() if key != "checkpoint"},
        exclude_volatile=True,
    )
    _validate_baseline_config(resolved)
    return resolved


def _validate_baseline_config(config: Mapping[str, Any]) -> None:
    task = config["_task"]
    model = config["baseline"]["model"]
    if int(model["action_dim"]) != int(
        task["physics"]["action_space"]["cardinality"]
    ):
        raise ValueError("model action_dim does not match the frozen task")
    if int(model["message_passing_rounds"]) < 1:
        raise ValueError("message_passing_rounds must be positive")
    if float(model["communication_radius"]) <= 0:
        raise ValueError("communication_radius must be positive")
    if config["baseline"].get("target_observation") != "team_persistent_detected_only":
        raise ValueError("target observation must exclude hidden target truth")
    if config["baseline"].get("critic_information") != "team_observable_state_only":
        raise ValueError("critic must exclude hidden target truth")
    reward = config["_reward"]
    if reward["profile"] not in {"sparse_v1", "observable_dense_v2"}:
        raise ValueError("unsupported baseline reward profile")
    for key in ("discovery_entropy_scale", "detected_target_progress_scale"):
        value = float(reward[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"baseline.reward.{key} must be finite and nonnegative")
    if reward["profile"] == "observable_dense_v2":
        for key in (
            "collision_coefficient",
            "collision_floor",
            "boundary_penalty",
            "active_time_penalty",
        ):
            value = reward[key]
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"baseline.reward.{key} must be finite for dense v2")
        if float(reward["active_time_penalty"]) < 0.0:
            raise ValueError("active_time_penalty must be nonnegative")
        for key in (
            "collision_coefficient", "collision_floor", "boundary_penalty"
        ):
            if float(reward[key]) > 0.0:
                raise ValueError(f"baseline.reward.{key} must be nonpositive")


def make_mappo_baseline_env(
    config: Mapping[str, Any],
    *,
    num_envs: int,
    device: str,
    seed: int,
):
    kwargs = chapter1_sar_kwargs(config["_task"], mode="high")
    reward = config["_reward"]
    kwargs.update(
        {
            "mode": "baseline_mappo",
            "emit_info": False,
            "enable_high_level_state": False,
            "enable_rrt_candidates": False,
            "auto_resample_goals": False,
            "direct_rescue": True,
            "render_assigned_goals": False,
            "render_recent_rrt_candidates": False,
            "direct_reward_profile": reward["profile"],
            "direct_discovery_reward_scale": float(
                reward["discovery_entropy_scale"]
            ),
            "direct_target_progress_scale": float(
                reward["detected_target_progress_scale"]
            ),
        }
    )
    if reward["profile"] == "observable_dense_v2":
        kwargs.update(
            {
                "collision_penalty": float(reward["collision_coefficient"]),
                "max_collision_penalty": float(reward["collision_floor"]),
                "boundary_penalty": float(reward["boundary_penalty"]),
                "time_penalty": float(reward["active_time_penalty"]),
            }
        )
    return make_sar_env(
        num_envs=num_envs,
        device=device,
        seed=seed,
        **kwargs,
    )


def scenario_snapshot(scenario: Any) -> dict[str, Tensor]:
    """Build actor/critic inputs only from team-observable state."""

    positions = torch.stack(
        [agent.state.pos for agent in scenario.world.agents], dim=1
    )
    velocities = torch.stack(
        [agent.state.vel for agent in scenario.world.agents], dim=1
    )
    active = scenario.active_agents.float()
    remaining = (
        1.0 - scenario.world_steps.float() / float(scenario.max_steps)
    ).clamp(0.0, 1.0)
    remaining_agents = remaining[:, None, None].expand(
        -1, scenario.n_agents, 1
    )
    agent_features = torch.cat(
        [positions, velocities, active.unsqueeze(-1), remaining_agents], dim=-1
    )

    detected = scenario.target_detected.any(dim=1)
    visited = scenario.target_visited
    observable = detected & ~visited
    detected_positions = scenario.detected_target_positions
    relative_targets = (
        detected_positions[:, None, :, :] - positions[:, :, None, :]
    )
    target_features = torch.cat(
        [
            relative_targets,
            visited[:, None, :, None]
            .float()
            .expand(-1, scenario.n_agents, -1, -1),
        ],
        dim=-1,
    )
    target_features = torch.where(
        observable[:, None, :, None],
        target_features,
        torch.zeros_like(target_features),
    )
    target_mask = observable[:, None, :].expand(-1, scenario.n_agents, -1)

    pair_distance = torch.cdist(positions, positions)
    critic_targets = torch.cat(
        [
            torch.where(
                detected.unsqueeze(-1),
                detected_positions,
                torch.zeros_like(detected_positions),
            ),
            detected.float().unsqueeze(-1),
            visited.float().unsqueeze(-1),
        ],
        dim=-1,
    )
    critic_state = torch.cat(
        [
            agent_features.flatten(start_dim=1),
            critic_targets.flatten(start_dim=1),
            remaining.unsqueeze(-1),
        ],
        dim=-1,
    )
    return {
        "agent_features": agent_features,
        "target_features": target_features,
        "target_mask": target_mask,
        "pair_distance": pair_distance,
        "active": scenario.active_agents.clone(),
        "critic_state": critic_state,
    }


class EndToEndMAPPO(nn.Module):
    """Legacy-inspired shared actor with a centralized observable-state critic."""

    def __init__(
        self,
        *,
        n_agents: int,
        n_targets: int,
        hidden_dim: int = 64,
        action_dim: int = 5,
        message_passing_rounds: int = 3,
        communication_radius: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.n_targets = int(n_targets)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.message_passing_rounds = int(message_passing_rounds)
        self.communication_radius = float(communication_radius)

        self.ego_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.Tanh()
        )
        self.target_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target_merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh()
        )
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Tanh(),
                )
                for _ in range(message_passing_rounds)
            ]
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        critic_width = n_agents * 6 + n_targets * 4 + 1
        self.critic = nn.Sequential(
            nn.Linear(critic_width, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def actor_logits(self, snapshot: Mapping[str, Tensor]) -> Tensor:
        hidden = self.ego_encoder(snapshot["agent_features"])
        target_hidden = self.target_encoder(snapshot["target_features"])
        scores = (
            self.target_query(hidden).unsqueeze(2) * target_hidden
        ).sum(dim=-1) / (self.hidden_dim**0.5)
        mask = snapshot["target_mask"]
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        target_context = (weights.unsqueeze(-1) * target_hidden).sum(dim=2)
        hidden = self.target_merge(torch.cat([hidden, target_context], dim=-1))

        active = snapshot["active"]
        adjacency = (
            snapshot["pair_distance"] <= self.communication_radius
        ) & active.unsqueeze(1) & active.unsqueeze(2)
        eye = torch.eye(
            self.n_agents, dtype=torch.bool, device=hidden.device
        ).unsqueeze(0)
        adjacency = adjacency & ~eye
        for layer in self.message_layers:
            weights = adjacency.float()
            messages = torch.bmm(weights, hidden)
            messages = messages / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            hidden = layer(torch.cat([hidden, messages], dim=-1))
        return self.actor_head(hidden)

    def value(self, snapshot: Mapping[str, Tensor]) -> Tensor:
        return self.critic(snapshot["critic_state"]).squeeze(-1)

    def act(
        self,
        snapshot: Mapping[str, Tensor],
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.actor_logits(snapshot)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        actions = torch.where(
            snapshot["active"], actions, torch.zeros_like(actions)
        )
        log_prob = distribution.log_prob(actions)
        return actions, log_prob, self.value(snapshot)


def model_from_config(config: Mapping[str, Any]) -> EndToEndMAPPO:
    task = config["_task"]
    model = config["baseline"]["model"]
    return EndToEndMAPPO(
        n_agents=task["scenario"]["num_agents"],
        n_targets=task["scenario"]["num_targets"],
        hidden_dim=model["hidden_dim"],
        action_dim=model["action_dim"],
        message_passing_rounds=model["message_passing_rounds"],
        communication_radius=model["communication_radius"],
    )


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    bootstrap_value: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap_value)
    next_value = bootstrap_value
    for step in reversed(range(rewards.shape[0])):
        nonterminal = (~dones[step]).float()
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        carry = delta + gamma * gae_lambda * nonterminal * carry
        advantages[step] = carry
        next_value = values[step]
    return advantages, advantages + values


def checkpoint_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline": BASELINE_NAME,
        "task_version": config["_task"]["task_version"],
        "task_config_sha256": config["_task_sha256"],
        "reward_profile": config["_reward"]["profile"],
        "baseline_config_sha256": config["_config_sha256"],
        "information_boundary": {
            "target_observation": "team_persistent_detected_only",
            "critic": "team_observable_state_only",
            "hidden_target_truth": False,
            "hierarchy_used": False,
        },
    }


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    metadata = checkpoint.get("metadata", {})
    expected = checkpoint_metadata(config)
    for key in (
        "schema_version",
        "baseline",
        "task_version",
        "task_config_sha256",
        "baseline_config_sha256",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"checkpoint metadata mismatch for {key}: "
                f"expected {expected[key]!r}, got {metadata.get(key)!r}"
            )

    if expected["reward_profile"] == "observable_dense_v2":
        if metadata.get("reward_profile") != expected["reward_profile"]:
            raise ValueError(
                "checkpoint metadata mismatch for reward_profile"
            )
