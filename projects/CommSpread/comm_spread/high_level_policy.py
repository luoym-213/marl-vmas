"""High-level SAR policy adapters used by debugging and future training code."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from comm_spread.models import HeterogeneousGraphActor, HighLevelMapCritic


@dataclass
class HighLevelPolicyStats:
    decisions: int = 0
    explore_actions: int = 0
    target_actions: int = 0
    invalid_actions: int = 0
    nonfinite_logits: int = 0
    target_actions_with_valid_mask: int = 0


class HGSARHighLevelPolicy:
    """Wrap :class:`HeterogeneousGraphActor` as an AsyncSMDP high-level policy."""

    def __init__(
        self,
        *,
        n_agents: int,
        rrt_top_k: int,
        device: str = "cpu",
        deterministic: bool = True,
        actor: HeterogeneousGraphActor | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.rrt_top_k = rrt_top_k
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.actor = actor or HeterogeneousGraphActor()
        self.actor.to(self.device)
        self.actor.eval()
        self.stats = HighLevelPolicyStats()
        self.last_debug: dict[str, Tensor] = {}

    @torch.no_grad()
    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        obs = {
            key: value.to(self.device) if isinstance(value, Tensor) else value
            for key, value in observation.items()
        }
        logits = self.actor(
            obs["ego_node"],
            obs["teammate_nodes"],
            obs["explore_nodes"],
            obs["target_nodes"],
            teammate_mask=obs["teammate_mask"],
            target_mask=obs["target_mask"],
            explore_edges=obs["explore_edges"],
            target_edges=obs["target_edges"],
            action_mask=obs["action_mask"],
        )
        action_mask = obs["action_mask"].bool()
        finite_logits = torch.isfinite(logits)
        self.stats.nonfinite_logits += int((~finite_logits).sum().item())
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        dist = Categorical(logits=masked_logits)
        if self.deterministic:
            actions = masked_logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions).unsqueeze(-1)
        values = torch.zeros(actions.shape[0], 1, dtype=logits.dtype, device=logits.device)

        selected_valid = action_mask.gather(1, actions.view(-1, 1)).squeeze(-1)
        is_target = actions >= self.rrt_top_k
        target_index = (actions - self.rrt_top_k).clamp(min=0)
        valid_target_mask = torch.zeros_like(is_target)
        if obs["target_mask"].numel() > 0:
            valid_target_mask = obs["target_mask"].bool().gather(
                1,
                target_index.clamp(max=obs["target_mask"].shape[1] - 1).view(-1, 1),
            ).squeeze(-1)

        self.stats.decisions += int(actions.numel())
        self.stats.explore_actions += int((~is_target).sum().item())
        self.stats.target_actions += int(is_target.sum().item())
        self.stats.invalid_actions += int((~selected_valid).sum().item())
        self.stats.target_actions_with_valid_mask += int((is_target & valid_target_mask).sum().item())
        self.last_debug = {
            "logits": masked_logits.detach().cpu(),
            "action_mask": action_mask.detach().cpu(),
            "actions": actions.detach().cpu(),
            "selected_node_type": is_target.long().detach().cpu(),
            "selected_node_utility": self._selected_utility(obs, actions).detach().cpu(),
        }
        output_device = observation["ego_node"].device
        return (
            actions.to(output_device),
            log_probs.to(output_device),
            values.to(output_device),
        )

    def metrics(self) -> dict[str, int]:
        return {
            "actor_decisions": self.stats.decisions,
            "actor_explore_actions": self.stats.explore_actions,
            "actor_target_actions": self.stats.target_actions,
            "actor_invalid_actions": self.stats.invalid_actions,
            "actor_nonfinite_logits": self.stats.nonfinite_logits,
            "actor_valid_target_actions": self.stats.target_actions_with_valid_mask,
        }

    def _selected_utility(self, obs: dict[str, Tensor], actions: Tensor) -> Tensor:
        explore_util = obs["explore_nodes"][..., 2]
        target_util = obs["target_nodes"][..., 2]
        explore_index = actions.clamp(max=self.rrt_top_k - 1)
        target_index = (actions - self.rrt_top_k).clamp(min=0)
        selected_explore = explore_util.gather(1, explore_index.view(-1, 1)).squeeze(-1)
        if target_util.shape[1] == 0:
            return selected_explore
        selected_target = target_util.gather(
            1,
            target_index.clamp(max=target_util.shape[1] - 1).view(-1, 1),
        ).squeeze(-1)
        return torch.where(actions >= self.rrt_top_k, selected_target, selected_explore)


class HGSARActorCriticPolicy(torch.nn.Module):
    """Trainable high-level actor-critic policy for async SAR SMDP PPO."""

    def __init__(
        self,
        *,
        n_agents: int,
        rrt_top_k: int,
        device: str = "cpu",
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.rrt_top_k = rrt_top_k
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.actor = HeterogeneousGraphActor()
        self.critic = HighLevelMapCritic(n_agents=n_agents, agent_features=5)
        self.stats = HighLevelPolicyStats()
        self.last_debug: dict[str, Tensor] = {}
        self.to(self.device)

    @torch.no_grad()
    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        was_training = self.training
        self.eval()
        obs = move_observation(observation, self.device)
        logits = self._logits(obs)
        dist = masked_categorical(logits, obs["action_mask"])
        if self.deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions).unsqueeze(-1)
        values = self._values(obs)
        self._update_stats(obs, logits, actions)
        if was_training:
            self.train()
        output_device = observation["ego_node"].device
        return (
            actions.to(output_device),
            log_probs.to(output_device),
            values.to(output_device),
        )

    def evaluate_actions(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        obs = move_observation(batch, self.device)
        actions = obs["action"].long().view(-1)
        logits = self._logits(obs)
        dist = masked_categorical(logits, obs["action_mask"])
        log_probs = dist.log_prob(actions).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        values = self._values(obs)
        return {
            "logits": logits,
            "log_prob": log_probs,
            "entropy": entropy,
            "value": values,
        }

    def metrics(self) -> dict[str, int]:
        return {
            "actor_decisions": self.stats.decisions,
            "actor_explore_actions": self.stats.explore_actions,
            "actor_target_actions": self.stats.target_actions,
            "actor_invalid_actions": self.stats.invalid_actions,
            "actor_nonfinite_logits": self.stats.nonfinite_logits,
            "actor_valid_target_actions": self.stats.target_actions_with_valid_mask,
        }

    def reset_stats(self) -> None:
        self.stats = HighLevelPolicyStats()
        self.last_debug = {}

    def _logits(self, obs: dict[str, Tensor]) -> Tensor:
        return self.actor(
            obs["ego_node"],
            obs["teammate_nodes"],
            obs["explore_nodes"],
            obs["target_nodes"],
            teammate_mask=obs["teammate_mask"],
            target_mask=obs["target_mask"],
            explore_edges=obs["explore_edges"],
            target_edges=obs["target_edges"],
            action_mask=obs["action_mask"],
        )

    def _values(self, obs: dict[str, Tensor]) -> Tensor:
        if "global_map_channels" in obs and "global_agent_nodes" in obs:
            return self.critic(
                obs["global_map_channels"],
                obs["global_agent_nodes"],
                obs.get("agent_id"),
            )
        return self.critic(obs["map_channels"], obs["ego_node"], obs.get("agent_id"))

    def _update_stats(self, obs: dict[str, Tensor], logits: Tensor, actions: Tensor) -> None:
        action_mask = obs["action_mask"].bool()
        selected_valid = action_mask.gather(1, actions.view(-1, 1)).squeeze(-1)
        is_target = actions >= self.rrt_top_k
        target_index = (actions - self.rrt_top_k).clamp(min=0)
        valid_target_mask = torch.zeros_like(is_target)
        if obs["target_mask"].numel() > 0:
            valid_target_mask = obs["target_mask"].bool().gather(
                1,
                target_index.clamp(max=obs["target_mask"].shape[1] - 1).view(-1, 1),
            ).squeeze(-1)

        self.stats.decisions += int(actions.numel())
        self.stats.explore_actions += int((~is_target).sum().item())
        self.stats.target_actions += int(is_target.sum().item())
        self.stats.invalid_actions += int((~selected_valid).sum().item())
        self.stats.nonfinite_logits += int((~torch.isfinite(logits)).sum().item())
        self.stats.target_actions_with_valid_mask += int((is_target & valid_target_mask).sum().item())
        self.last_debug = {
            "logits": logits.detach().cpu(),
            "action_mask": action_mask.detach().cpu(),
            "actions": actions.detach().cpu(),
            "selected_node_type": is_target.long().detach().cpu(),
            "selected_node_utility": self._selected_utility(obs, actions).detach().cpu(),
        }

    def _selected_utility(self, obs: dict[str, Tensor], actions: Tensor) -> Tensor:
        explore_util = obs["explore_nodes"][..., 2]
        target_util = obs["target_nodes"][..., 2]
        explore_index = actions.clamp(max=self.rrt_top_k - 1)
        target_index = (actions - self.rrt_top_k).clamp(min=0)
        selected_explore = explore_util.gather(1, explore_index.view(-1, 1)).squeeze(-1)
        if target_util.shape[1] == 0:
            return selected_explore
        selected_target = target_util.gather(
            1,
            target_index.clamp(max=target_util.shape[1] - 1).view(-1, 1),
        ).squeeze(-1)
        return torch.where(actions >= self.rrt_top_k, selected_target, selected_explore)


def move_observation(batch: dict[str, Tensor], device: torch.device | str) -> dict[str, Tensor]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def masked_categorical(logits: Tensor, action_mask: Tensor) -> Categorical:
    masked_logits = logits.masked_fill(
        ~action_mask.bool(),
        torch.finfo(logits.dtype).min,
    )
    return Categorical(logits=masked_logits)
