"""High-level SAR policy adapters used by debugging and future training code."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from comm_spread.models import (
    HeterogeneousGraphActor,
    HighLevelMapCritic,
    PhaseConditionedActionHead,
    TeammateIntentionCoordinator,
)


@dataclass
class HighLevelPolicyStats:
    decisions: int = 0
    explore_actions: int = 0
    target_actions: int = 0
    invalid_actions: int = 0
    nonfinite_logits: int = 0
    target_actions_with_valid_mask: int = 0
    coordination_pressure_sum: float = 0.0
    coordination_pressure_count: int = 0
    coordination_pressure_max: float = 0.0
    coordination_suppressed_targets: int = 0
    coordination_target_count: int = 0
    coordination_logit_reduction_sum: float = 0.0
    coordination_top1_changes: int = 0
    coordination_gate_near_zero: int = 0
    coordination_gate_saturated: int = 0
    coordination_utility_abs_max: float = 0.0
    rescue_probability_base_sum: float = 0.0
    rescue_probability_final_sum: float = 0.0
    rescue_probability_count: int = 0
    intention_correct: int = 0
    intention_total: int = 0
    intention_none_correct: int = 0
    intention_none_total: int = 0
    intention_rescue_correct: int = 0
    intention_rescue_total: int = 0
    intention_simultaneous_correct: int = 0
    intention_simultaneous_total: int = 0
    intention_predicted_none: int = 0
    intention_kl_sum: float = 0.0
    intention_kl_count: int = 0
    high_probability_conflicts: int = 0
    mutual_yield_events: int = 0
    phase_rescue_probability_sum: float = 0.0
    phase_probability_count: int = 0
    phase_rescue_choices: int = 0


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
        ego_features: int = 5,
        target_features: int = 4,
    ) -> None:
        self.n_agents = n_agents
        self.rrt_top_k = rrt_top_k
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.actor = actor or HeterogeneousGraphActor(
            ego_features=ego_features,
            target_features=target_features,
        )
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
        ego_features: int = 5,
        target_features: int = 4,
        enable_commitment_aware_actor: bool = False,
        enable_teammate_intention_coordination: bool = False,
        enable_phase_policy: bool = False,
        phase_initial_rescue_logit: float = -1.5,
        rescue_distance_logit_scale: float = 0.0,
        coordination_beta: float = 0.25,
        coordination_margin: float = 0.1,
        coordination_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.rrt_top_k = rrt_top_k
        self.device = torch.device(device)
        self.deterministic = deterministic
        if enable_commitment_aware_actor and enable_teammate_intention_coordination:
            raise ValueError(
                "commitment-aware actor and intention coordination are "
                "mutually exclusive experimental paths"
            )
        self.enable_commitment_aware_actor = enable_commitment_aware_actor
        self.enable_teammate_intention_coordination = (
            enable_teammate_intention_coordination
        )
        self.enable_phase_policy = enable_phase_policy
        if rescue_distance_logit_scale < 0:
            raise ValueError("rescue distance logit scale must be non-negative")
        self.rescue_distance_logit_scale = float(rescue_distance_logit_scale)
        self.actor = HeterogeneousGraphActor(
            ego_features=ego_features,
            target_features=target_features,
            teammate_context_features=5 if enable_commitment_aware_actor else 0,
        )
        self.critic = HighLevelMapCritic(n_agents=n_agents, agent_features=ego_features)
        self.coordinator = (
            TeammateIntentionCoordinator(
                target_features=target_features,
                beta=coordination_beta,
                margin=coordination_margin,
                temperature=coordination_temperature,
            )
            if enable_teammate_intention_coordination
            else None
        )
        self.phase_head = (
            PhaseConditionedActionHead(
                ego_features=ego_features,
                initial_rescue_logit=phase_initial_rescue_logit,
            )
            if enable_phase_policy
            else None
        )
        self.stats = HighLevelPolicyStats()
        self.last_debug: dict[str, Tensor] = {}
        self._coord_output: dict[str, Tensor] = {}
        self._phase_probs: Tensor | None = None
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
        output = {
            "logits": logits,
            "log_prob": log_probs,
            "entropy": entropy,
            "value": values,
        }
        output.update(self._intention_loss_metrics(obs))
        return output

    def metrics(self) -> dict[str, int | float]:
        stats = self.stats
        pressure_count = max(stats.coordination_pressure_count, 1)
        target_count = max(stats.coordination_target_count, 1)
        probability_count = max(stats.rescue_probability_count, 1)
        result: dict[str, int | float] = {
            "actor_decisions": self.stats.decisions,
            "actor_explore_actions": self.stats.explore_actions,
            "actor_target_actions": self.stats.target_actions,
            "actor_invalid_actions": self.stats.invalid_actions,
            "actor_nonfinite_logits": self.stats.nonfinite_logits,
            "actor_valid_target_actions": self.stats.target_actions_with_valid_mask,
        }
        if self.enable_phase_policy:
            result.update(
                {
                    "phase_rescue_probability_mean": (
                        stats.phase_rescue_probability_sum
                        / max(stats.phase_probability_count, 1)
                    ),
                    "phase_rescue_choice_ratio": (
                        stats.phase_rescue_choices / max(stats.decisions, 1)
                    ),
                }
            )
        if self.enable_teammate_intention_coordination:
            result.update(
                {
                    "coordination_pressure_mean": stats.coordination_pressure_sum / pressure_count,
                    "coordination_pressure_max": stats.coordination_pressure_max,
                    "coordination_suppressed_target_ratio": stats.coordination_suppressed_targets / target_count,
                    "coordination_logit_reduction_mean": stats.coordination_logit_reduction_sum / target_count,
                    "coordination_target_top1_changes": stats.coordination_top1_changes,
                    "coordination_gate_near_zero_ratio": stats.coordination_gate_near_zero / target_count,
                    "coordination_gate_saturated_ratio": stats.coordination_gate_saturated / target_count,
                    "coordination_utility_abs_max": stats.coordination_utility_abs_max,
                    "rescue_probability_base_mean": stats.rescue_probability_base_sum / probability_count,
                    "rescue_probability_final_mean": stats.rescue_probability_final_sum / probability_count,
                    "intention_top1_accuracy": stats.intention_correct / max(stats.intention_total, 1),
                    "intention_none_accuracy": stats.intention_none_correct / max(stats.intention_none_total, 1),
                    "intention_rescue_accuracy": stats.intention_rescue_correct / max(stats.intention_rescue_total, 1),
                    "intention_simultaneous_accuracy": stats.intention_simultaneous_correct / max(stats.intention_simultaneous_total, 1),
                    "intention_predicted_none_ratio": stats.intention_predicted_none / max(stats.intention_total, 1),
                    "intention_teacher_none_ratio": stats.intention_none_total / max(stats.intention_total, 1),
                    "intention_teacher_samples": stats.intention_total,
                    "intention_rescue_samples": stats.intention_rescue_total,
                    "intention_simultaneous_rescue_samples": stats.intention_simultaneous_total,
                    "intention_rollout_kl_loss": stats.intention_kl_sum / max(stats.intention_kl_count, 1),
                    "high_probability_target_conflicts": stats.high_probability_conflicts,
                    "mutual_yield_events": stats.mutual_yield_events,
                }
            )
        return result

    def reset_stats(self) -> None:
        self.stats = HighLevelPolicyStats()
        self.last_debug = {}
        self._coord_output = {}
        self._phase_probs = None

    def _logits(self, obs: dict[str, Tensor]) -> Tensor:
        teammate_nodes = obs["teammate_nodes"]
        if self.enable_commitment_aware_actor:
            # Execution uses only fresh observable teammate state. No teacher,
            # teammate action/logits, proposal, critic state, or sampling order
            # enters this actor path. PPO is its sole learning signal.
            teammate_nodes = torch.cat(
                [teammate_nodes, obs["teammate_context_nodes"]],
                dim=-1,
            )
        base_logits = self.actor(
            obs["ego_node"],
            teammate_nodes,
            obs["explore_nodes"],
            obs["target_nodes"],
            teammate_mask=obs["teammate_mask"],
            target_mask=obs["target_mask"],
            explore_edges=obs["explore_edges"],
            target_edges=obs["target_edges"],
            action_mask=obs["action_mask"],
        )
        if self.rescue_distance_logit_scale > 0:
            # Each agent uses only its own relative distance to currently
            # visible rescue targets. This breaks homogeneous target ties
            # without agent indices, decision order, proposals, or matching.
            target_distance = torch.linalg.vector_norm(
                obs["target_nodes"][..., 0:2],
                dim=-1,
            )
            base_logits = base_logits.clone()
            base_logits[:, self.rrt_top_k :] = (
                base_logits[:, self.rrt_top_k :]
                - self.rescue_distance_logit_scale * target_distance
            )
        if self.coordinator is None:
            self._coord_output = {}
            final_logits = base_logits
        else:
            self._coord_output = self.coordinator(
                base_logits,
                coord_agent_nodes=obs["coord_agent_nodes"],
                ego_agent_ids=obs["agent_id"],
                target_nodes=obs["target_nodes"],
                target_mask=obs["target_mask"],
                intent_target_mask=obs["intent_target_mask"],
                agent_active_mask=obs["agent_active_mask"],
                rrt_top_k=self.rrt_top_k,
            )
            final_logits = self._coord_output["final_logits"]
        if self.phase_head is None:
            self._phase_probs = None
            return final_logits
        final_logits, self._phase_probs = self.phase_head(
            final_logits,
            obs["ego_node"],
            obs["action_mask"],
            rrt_top_k=self.rrt_top_k,
        )
        return final_logits

    @torch.no_grad()
    def build_intent_teacher(
        self,
        observation: dict[str, Tensor],
    ) -> dict[str, Tensor] | None:
        """Build detached CTDE labels after, and separately from, actor forward."""
        if self.coordinator is None:
            return None
        obs = move_observation(observation, self.device)
        base_logits = self._coord_output["base_logits"].detach()
        base_probs = masked_categorical(
            base_logits, obs["action_mask"]
        ).probs.detach()
        batch_size = base_logits.shape[0]
        n_agents = obs["coord_agent_nodes"].shape[1]
        n_targets = obs["target_ids"].shape[1]
        teacher = torch.zeros(
            batch_size,
            n_agents,
            n_targets + 1,
            device=self.device,
        )
        teacher_mask = torch.zeros(
            batch_size, n_agents, dtype=torch.bool, device=self.device
        )
        simultaneous_mask = torch.zeros_like(teacher_mask)
        env_ids = obs["env_id"].long()
        ego_ids = obs["agent_id"].long()
        target_ids = obs["target_ids"].long()
        target_valid = obs["intent_target_mask"].bool()

        for row in range(batch_size):
            for teammate_id in range(n_agents):
                if teammate_id == int(ego_ids[row]):
                    continue
                if not bool(obs["agent_active_mask"][row, teammate_id]):
                    continue
                teacher_mask[row, teammate_id] = True
                if bool(obs["agent_decision_mask"][row, teammate_id]):
                    source_rows = torch.nonzero(
                        (env_ids == env_ids[row])
                        & (ego_ids == teammate_id),
                        as_tuple=False,
                    ).flatten()
                    if source_rows.numel() == 0:
                        teacher_mask[row, teammate_id] = False
                        continue
                    source = int(source_rows[0])
                    simultaneous_mask[row, teammate_id] = True
                    teacher[row, teammate_id, -1] = base_probs[
                        source, : self.rrt_top_k
                    ].sum()
                    for source_slot in range(n_targets):
                        probability = base_probs[
                            source, self.rrt_top_k + source_slot
                        ]
                        global_id = int(target_ids[source, source_slot])
                        ego_slots = torch.nonzero(
                            (target_ids[row] == global_id)
                            & target_valid[row],
                            as_tuple=False,
                        ).flatten()
                        if ego_slots.numel() == 0:
                            teacher[row, teammate_id, -1] += probability
                        else:
                            teacher[
                                row, teammate_id, int(ego_slots[0])
                            ] += probability
                else:
                    commitment_id = int(
                        obs["agent_commitment_target_id"][row, teammate_id]
                    )
                    ego_slots = torch.nonzero(
                        (target_ids[row] == commitment_id)
                        & target_valid[row],
                        as_tuple=False,
                    ).flatten()
                    if commitment_id >= 0 and ego_slots.numel() > 0:
                        teacher[
                            row, teammate_id, int(ego_slots[0])
                        ] = 1.0
                    else:
                        teacher[row, teammate_id, -1] = 1.0

        normalizer = teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        teacher = (teacher / normalizer).detach()
        labels = {
            "intent_teacher": teacher,
            "intent_teacher_mask": teacher_mask,
            "intent_teacher_simultaneous_mask": simultaneous_mask,
        }
        self._record_intention_metrics(
            self._coord_output["intent_probs"].detach(),
            labels,
        )
        return labels

    def _intention_loss_metrics(
        self,
        obs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        zero = torch.zeros((), device=self.device)
        if self.coordinator is None or "intent_teacher" not in obs:
            return {
                "intent_loss": zero,
                "intent_top1_accuracy": zero,
                "intent_none_accuracy": zero,
                "intent_rescue_accuracy": zero,
                "intent_simultaneous_accuracy": zero,
                "intent_frequency_baseline_accuracy": zero,
            }
        prediction = self._coord_output["intent_probs"].clamp_min(1e-8)
        teacher = obs["intent_teacher"].detach()
        valid = obs["intent_teacher_mask"].bool()
        simultaneous = (
            obs["intent_teacher_simultaneous_mask"].bool()
            & valid
            & (teacher.argmax(dim=-1) != teacher.shape[-1] - 1)
        )
        kl = (
            teacher
            * (
                teacher.clamp_min(1e-8).log()
                - prediction.log()
            )
        ).sum(dim=-1)
        intent_loss = kl[valid].mean() if valid.any() else zero
        predicted_class = prediction.argmax(dim=-1)
        teacher_class = teacher.argmax(dim=-1)
        correct = predicted_class == teacher_class
        none = teacher_class == teacher.shape[-1] - 1
        rescue = ~none

        def accuracy(mask: Tensor) -> Tensor:
            return correct[mask].float().mean() if mask.any() else zero

        return {
            "intent_loss": intent_loss,
            "intent_top1_accuracy": accuracy(valid),
            "intent_none_accuracy": accuracy(valid & none),
            "intent_rescue_accuracy": accuracy(valid & rescue),
            "intent_simultaneous_accuracy": accuracy(simultaneous),
            "intent_frequency_baseline_accuracy": (
                torch.bincount(
                    teacher_class[valid],
                    minlength=teacher.shape[-1],
                ).max().float()
                / valid.sum().clamp_min(1)
                if valid.any()
                else zero
            ),
        }

    def _record_intention_metrics(
        self,
        prediction: Tensor,
        labels: dict[str, Tensor],
    ) -> None:
        teacher = labels["intent_teacher"]
        valid = labels["intent_teacher_mask"].bool()
        predicted_class = prediction.argmax(dim=-1)
        teacher_class = teacher.argmax(dim=-1)
        simultaneous = (
            labels["intent_teacher_simultaneous_mask"].bool()
            & valid
            & (teacher_class != teacher.shape[-1] - 1)
        )
        correct = predicted_class == teacher_class
        none = teacher_class == teacher.shape[-1] - 1
        rescue = ~none
        stats = self.stats
        kl = (
            teacher
            * (
                teacher.clamp_min(1e-8).log()
                - prediction.clamp_min(1e-8).log()
            )
        ).sum(dim=-1)
        stats.intention_kl_sum += float(kl[valid].sum())
        stats.intention_kl_count += int(valid.sum())
        stats.intention_correct += int((correct & valid).sum())
        stats.intention_total += int(valid.sum())
        stats.intention_none_correct += int((correct & valid & none).sum())
        stats.intention_none_total += int((valid & none).sum())
        stats.intention_rescue_correct += int((correct & valid & rescue).sum())
        stats.intention_rescue_total += int((valid & rescue).sum())
        stats.intention_simultaneous_correct += int(
            (correct & simultaneous).sum()
        )
        stats.intention_simultaneous_total += int(simultaneous.sum())
        stats.intention_predicted_none += int(
            ((predicted_class == teacher.shape[-1] - 1) & valid).sum()
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
        if self._phase_probs is not None:
            self.stats.phase_rescue_probability_sum += float(
                self._phase_probs[:, 1].sum().item()
            )
            self.stats.phase_probability_count += int(actions.numel())
            self.stats.phase_rescue_choices += int(is_target.sum().item())
        if self.coordinator is not None:
            self._update_coordination_stats(obs, logits)
        self.last_debug = {
            "logits": logits.detach().cpu(),
            "action_mask": action_mask.detach().cpu(),
            "actions": actions.detach().cpu(),
            "selected_node_type": is_target.long().detach().cpu(),
            "selected_node_utility": self._selected_utility(obs, actions).detach().cpu(),
        }
        if self.coordinator is not None:
            self.last_debug.update(
                {
                    "base_logits": self._coord_output["base_logits"].detach().cpu(),
                    "coordination_pressure": self._coord_output[
                        "competition_pressure"
                    ].detach().cpu(),
                    "intent_probs": self._coord_output["intent_probs"].detach().cpu(),
                    "utilities": self._coord_output["utilities"].detach().cpu(),
                }
            )

    def _update_coordination_stats(
        self,
        obs: dict[str, Tensor],
        final_logits: Tensor,
    ) -> None:
        assert self.coordinator is not None
        stats = self.stats
        pressure = self._coord_output["competition_pressure"]
        valid = obs["target_mask"].bool()
        valid_pressure = pressure[valid]
        beta = self.coordinator.beta
        if valid_pressure.numel() > 0:
            stats.coordination_pressure_sum += float(valid_pressure.sum())
            stats.coordination_pressure_count += int(valid_pressure.numel())
            stats.coordination_pressure_max = max(
                stats.coordination_pressure_max,
                float(valid_pressure.max()),
            )
            stats.coordination_suppressed_targets += int(
                (valid_pressure > 1e-6).sum()
            )
            stats.coordination_target_count += int(valid_pressure.numel())
            stats.coordination_logit_reduction_sum += float(
                (beta * valid_pressure).sum()
            )
            stats.coordination_gate_near_zero += int(
                (valid_pressure < 0.01).sum()
            )
            stats.coordination_gate_saturated += int(
                (valid_pressure > 0.95).sum()
            )
        utilities = self._coord_output["utilities"]
        stats.coordination_utility_abs_max = max(
            stats.coordination_utility_abs_max,
            float(utilities.abs().max()),
        )
        base_logits = self._coord_output["base_logits"]
        base_dist = masked_categorical(base_logits, obs["action_mask"])
        final_dist = masked_categorical(final_logits, obs["action_mask"])
        base_rescue = base_dist.probs[:, self.rrt_top_k :]
        final_rescue = final_dist.probs[:, self.rrt_top_k :]
        stats.rescue_probability_base_sum += float(base_rescue.sum())
        stats.rescue_probability_final_sum += float(final_rescue.sum())
        stats.rescue_probability_count += int(base_rescue.shape[0])

        masked_base_targets = base_logits[:, self.rrt_top_k :].masked_fill(
            ~valid, torch.finfo(base_logits.dtype).min
        )
        masked_final_targets = final_logits[:, self.rrt_top_k :].masked_fill(
            ~valid, torch.finfo(final_logits.dtype).min
        )
        has_target = valid.any(dim=-1)
        stats.coordination_top1_changes += int(
            (
                (masked_base_targets.argmax(dim=-1)
                 != masked_final_targets.argmax(dim=-1))
                & has_target
            ).sum()
        )
        if "env_id" in obs and "target_ids" in obs:
            for env_id in obs["env_id"].unique():
                rows = torch.nonzero(
                    obs["env_id"] == env_id, as_tuple=False
                ).flatten()
                for global_target in obs["target_ids"][rows].unique():
                    base_high = 0
                    final_high = 0
                    for row_tensor in rows:
                        row = int(row_tensor)
                        slots = torch.nonzero(
                            obs["target_ids"][row] == global_target,
                            as_tuple=False,
                        ).flatten()
                        if slots.numel() == 0:
                            continue
                        slot = int(slots[0])
                        base_high += int(base_rescue[row, slot] >= 0.2)
                        final_high += int(final_rescue[row, slot] >= 0.2)
                    stats.high_probability_conflicts += int(final_high >= 2)
                    stats.mutual_yield_events += int(
                        base_high >= 2 and final_high == 0
                    )

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
