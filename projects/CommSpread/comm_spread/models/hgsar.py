"""PyTorch modules extracted from the HGSAR/MPNN design."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class LowLevelGoalConditionedActorCritic(nn.Module):
    """Goal-conditioned continuous controller for VMAS holonomic agents."""

    def __init__(
        self,
        obs_features: int,
        action_features: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_features)
        self.actor_log_std = nn.Parameter(torch.zeros(action_features))
        self.critic = nn.Linear(hidden_dim, 1)
        self.apply(_init_weights)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.encoder(obs)
        mean = torch.tanh(self.actor_mean(features))
        log_std = self.actor_log_std.expand_as(mean)
        value = self.critic(features)
        return mean, log_std, value


class HeterogeneousGraphActor(nn.Module):
    """High-level node selector matching the original HGSAR graph policy path."""

    def __init__(
        self,
        ego_features: int = 5,
        teammate_features: int = 5,
        teammate_context_features: int = 0,
        explore_features: int = 4,
        target_features: int = 4,
        edge_features: int = 3,
        hidden_dim: int = 128,
        node_dim: int = 64,
    ) -> None:
        super().__init__()
        self.ego_encoder = _mlp(ego_features, node_dim)
        self.teammate_encoder = _mlp(teammate_features, node_dim)
        self.teammate_features = teammate_features
        self.teammate_context_features = teammate_context_features
        self.teammate_context_encoder = (
            _mlp(teammate_context_features, node_dim)
            if teammate_context_features > 0
            else None
        )
        self.explore_encoder = _mlp(explore_features, node_dim)
        self.target_encoder = _mlp(target_features, node_dim)
        self.explore_node_linear = nn.Linear(node_dim, node_dim)
        self.target_node_linear = nn.Linear(node_dim, node_dim)
        self.linear_ln = nn.LayerNorm(node_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, 32),
            nn.ReLU(),
        )

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.edge_bias = nn.Linear(32, 1)
        self.attn_dim = node_dim
        self.apply(_init_weights)

    def forward(
        self,
        ego_nodes: Tensor,
        teammate_nodes: Tensor,
        explore_nodes: Tensor,
        target_nodes: Tensor,
        *,
        teammate_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        explore_edges: Tensor | None = None,
        target_edges: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        """Return logits for concatenated [explore_nodes, target_nodes].

        Shapes:
            ego_nodes: [B, 5]
            teammate_nodes: [B, A, 5 + optional context features]
            explore_nodes: [B, K, 4]
            target_nodes: [B, L, 4]
            target_mask: [B, L] or [B, L, 1], true for valid nodes.
            explore_edges: [B, K, 3]
            target_edges: [B, L, 3]
            action_mask: [B, K + L], true for selectable nodes.
        """

        batch_size = ego_nodes.shape[0]
        n_explore = explore_nodes.shape[1]
        n_target = target_nodes.shape[1]

        if explore_edges is None:
            explore_edges = _relative_edges(explore_nodes[..., 0:2])
        if target_edges is None:
            target_edges = _relative_edges(target_nodes[..., 0:2])
        if target_mask is None:
            target_valid = torch.ones(
                batch_size,
                n_target,
                dtype=torch.bool,
                device=ego_nodes.device,
            )
        else:
            target_valid = _squeeze_mask(target_mask).bool()
        if teammate_mask is None:
            teammate_mask = torch.ones(
                batch_size,
                teammate_nodes.shape[1],
                1,
                dtype=teammate_nodes.dtype,
                device=teammate_nodes.device,
            )

        ego_feats = self.ego_encoder(ego_nodes)

        explore_feats = self.explore_encoder(
            explore_nodes.reshape(batch_size * n_explore, -1)
        ).view(batch_size, n_explore, -1)
        explore_feats = self.explore_node_linear(
            explore_feats.reshape(batch_size * n_explore, -1)
        ).view(batch_size, n_explore, -1)

        target_feats = self.target_encoder(
            target_nodes.reshape(batch_size * n_target, -1)
        ).view(batch_size, n_target, -1)
        target_feats = self.target_node_linear(
            target_feats.reshape(batch_size * n_target, -1)
        ).view(batch_size, n_target, -1)

        explore_edge_feats = self.edge_encoder(
            explore_edges.reshape(batch_size * n_explore, -1)
        ).view(batch_size, n_explore, -1)
        target_edge_feats = self.edge_encoder(
            target_edges.reshape(batch_size * n_target, -1)
        ).view(batch_size, n_target, -1)

        explore_valid = torch.ones(
            batch_size,
            n_explore,
            dtype=torch.bool,
            device=ego_nodes.device,
        )
        valid = torch.cat([explore_valid, target_valid], dim=1)
        if action_mask is not None:
            valid = valid & _squeeze_mask(action_mask).bool()
        no_valid = ~valid.any(dim=1)
        if no_valid.any():
            valid[no_valid, :n_explore] = True

        target_task_nodes = target_nodes[..., : explore_nodes.shape[-1]]
        task_nodes = torch.cat([explore_nodes, target_task_nodes], dim=1)
        task_feats = torch.cat([explore_feats, target_feats], dim=1)
        task_feats = self.task_team_hmpnn(
            ego_nodes[:, :2],
            teammate_nodes,
            teammate_mask,
            task_nodes,
            valid.unsqueeze(-1).float(),
            task_feats,
        )
        task_feats = self.linear_ln(task_feats)
        edge_feats = torch.cat([explore_edge_feats, target_edge_feats], dim=1)

        query = self.q_proj(ego_feats)
        keys = self.k_proj(task_feats)
        logits = (keys * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.attn_dim)
        logits = logits + self.edge_bias(edge_feats).squeeze(-1)
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    def task_team_hmpnn(
        self,
        ego_pos: Tensor,
        teammate_nodes: Tensor,
        teammate_mask: Tensor,
        task_nodes: Tensor,
        task_mask: Tensor,
        task_node_feats: Tensor,
    ) -> Tensor:
        batch_size, n_tasks, _ = task_nodes.shape
        n_teammates = teammate_nodes.shape[1]

        task_abs_pos = ego_pos.unsqueeze(1) + task_nodes[..., 0:2]
        teammate_base_nodes = teammate_nodes[..., : self.teammate_features]
        teammate_abs_pos = teammate_base_nodes[..., 0:2]
        teammate_rel_to_task = teammate_abs_pos.unsqueeze(1) - task_abs_pos.unsqueeze(2)
        teammate_other_feats = teammate_base_nodes[..., 2:].unsqueeze(1).expand(
            batch_size,
            n_tasks,
            n_teammates,
            -1,
        )
        teammate_nodes_per_task = torch.cat(
            [teammate_rel_to_task, teammate_other_feats],
            dim=-1,
        )
        teammate_feats = self.teammate_encoder(
            teammate_nodes_per_task.reshape(batch_size * n_tasks * n_teammates, -1)
        ).view(batch_size, n_tasks, n_teammates, -1)
        if self.teammate_context_encoder is not None:
            context = teammate_nodes[..., self.teammate_features :]
            if context.shape[-1] != self.teammate_context_features:
                raise ValueError(
                    "teammate context width does not match actor configuration: "
                    f"expected {self.teammate_context_features}, got {context.shape[-1]}"
                )
            context_feats = self.teammate_context_encoder(
                context.reshape(batch_size * n_teammates, -1)
            ).view(batch_size, 1, n_teammates, -1)
            # The original teammate path stays intact. Fresh commitment state is
            # a shared, permutation-equivariant residual learned only through PPO.
            teammate_feats = teammate_feats + context_feats

        task_query = task_node_feats.unsqueeze(2)
        attn_scores = torch.matmul(
            task_query,
            teammate_feats.transpose(-2, -1),
        ) / math.sqrt(self.attn_dim)
        teammate_mask_expanded = _squeeze_mask(teammate_mask).view(
            batch_size,
            1,
            1,
            n_teammates,
        )
        task_mask_expanded = _squeeze_mask(task_mask).view(
            batch_size,
            n_tasks,
            1,
            1,
        )
        attn_scores = attn_scores.masked_fill(
            teammate_mask_expanded < 0.5,
            torch.finfo(attn_scores.dtype).min,
        )
        has_teammate = (teammate_mask_expanded.sum(dim=-1, keepdim=True) > 0)
        attn_scores = torch.where(has_teammate, attn_scores, torch.zeros_like(attn_scores))
        attn_weights = torch.softmax(attn_scores, dim=-1) * teammate_mask_expanded
        task_aggregated = torch.matmul(attn_weights, teammate_feats).squeeze(2)
        task_aggregated = (
            task_aggregated
            * has_teammate.squeeze(-1).float()
            * task_mask_expanded.squeeze(-1)
        )
        return task_node_feats + task_aggregated


class PhaseConditionedActionHead(nn.Module):
    """Factor a flat node distribution into search/rescue phase and node choice.

    The phase decision reads only the ego observation. The existing graph actor
    remains responsible for ranking nodes within each phase. Invalid phases are
    removed from the phase softmax, so this head never makes an illegal action
    selectable and PPO can reconstruct the exact rollout distribution.
    """

    def __init__(
        self,
        *,
        ego_features: int,
        hidden_dim: int = 64,
        initial_rescue_logit: float = -1.5,
    ) -> None:
        super().__init__()
        self.phase_scorer = nn.Sequential(
            nn.Linear(ego_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.apply(_init_weights)
        final = self.phase_scorer[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(
                torch.tensor([0.0, initial_rescue_logit], dtype=final.bias.dtype)
            )

    def forward(
        self,
        base_logits: Tensor,
        ego_nodes: Tensor,
        action_mask: Tensor,
        *,
        rrt_top_k: int,
    ) -> tuple[Tensor, Tensor]:
        valid = action_mask.bool()
        search_valid = valid[:, :rrt_top_k]
        rescue_valid = valid[:, rrt_top_k:]
        phase_valid = torch.stack(
            [search_valid.any(dim=-1), rescue_valid.any(dim=-1)],
            dim=-1,
        )
        phase_logits = self.phase_scorer(ego_nodes).masked_fill(
            ~phase_valid,
            torch.finfo(base_logits.dtype).min,
        )
        phase_log_probs = torch.log_softmax(phase_logits, dim=-1)

        final_logits = torch.full_like(base_logits, torch.finfo(base_logits.dtype).min)
        if rrt_top_k > 0:
            search_logits = base_logits[:, :rrt_top_k].masked_fill(
                ~search_valid,
                torch.finfo(base_logits.dtype).min,
            )
            search_conditional = search_logits - torch.logsumexp(
                search_logits,
                dim=-1,
                keepdim=True,
            )
            final_logits[:, :rrt_top_k] = (
                search_conditional + phase_log_probs[:, 0:1]
            ).masked_fill(~search_valid, torch.finfo(base_logits.dtype).min)
        if rescue_valid.shape[1] > 0:
            rescue_logits = base_logits[:, rrt_top_k:].masked_fill(
                ~rescue_valid,
                torch.finfo(base_logits.dtype).min,
            )
            rescue_conditional = rescue_logits - torch.logsumexp(
                rescue_logits,
                dim=-1,
                keepdim=True,
            )
            final_logits[:, rrt_top_k:] = (
                rescue_conditional + phase_log_probs[:, 1:2]
            ).masked_fill(~rescue_valid, torch.finfo(base_logits.dtype).min)
        return final_logits, phase_log_probs.exp()


class TeammateIntentionCoordinator(nn.Module):
    """Distributed soft coordination from locally inferred teammate intent.

    The module consumes only the ego actor's current observation. It never
    receives teammate actions, proposals, actor logits, samples, or critic
    features. Training teachers are handled outside forward.
    """

    def __init__(
        self,
        *,
        agent_features: int = 10,
        target_features: int = 4,
        node_dim: int = 64,
        beta: float = 0.25,
        margin: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= beta <= 2.0:
            raise ValueError("coordination beta must be in [0, 2]")
        if temperature <= 0:
            raise ValueError("coordination temperature must be positive")
        self.beta = float(beta)
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.agent_encoder = _mlp(agent_features, node_dim)
        self.target_encoder = _mlp(target_features, node_dim)
        pair_dim = node_dim * 2 + 3
        intent_pair_dim = node_dim * 3 + 3
        self.utility_scorer = nn.Sequential(
            nn.Linear(pair_dim, node_dim),
            nn.Tanh(),
            nn.Linear(node_dim, 1),
        )
        self.intent_scorer = nn.Sequential(
            nn.Linear(intent_pair_dim, node_dim),
            nn.Tanh(),
            nn.Linear(node_dim, 1),
        )
        self.none_scorer = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Tanh(),
            nn.Linear(node_dim, 1),
        )
        self.apply(_init_weights)

    def forward(
        self,
        base_logits: Tensor,
        *,
        coord_agent_nodes: Tensor,
        ego_agent_ids: Tensor,
        target_nodes: Tensor,
        target_mask: Tensor,
        intent_target_mask: Tensor,
        agent_active_mask: Tensor,
        rrt_top_k: int,
    ) -> dict[str, Tensor]:
        batch_size, n_agents, _ = coord_agent_nodes.shape
        n_targets = target_nodes.shape[1]
        agent_feats = self.agent_encoder(
            coord_agent_nodes.reshape(batch_size * n_agents, -1)
        ).view(batch_size, n_agents, -1)
        target_feats = self.target_encoder(
            target_nodes.reshape(batch_size * n_targets, -1)
        ).view(batch_size, n_targets, -1)
        gather_ids = ego_agent_ids.long().view(batch_size, 1, 1).expand(
            -1, 1, agent_feats.shape[-1]
        )
        ego_feats = agent_feats.gather(1, gather_ids).squeeze(1)

        ego_pos = coord_agent_nodes[
            torch.arange(batch_size, device=base_logits.device),
            ego_agent_ids.long(),
            0:2,
        ]
        target_abs_pos = ego_pos.unsqueeze(1) + target_nodes[..., 0:2]
        agent_pos = coord_agent_nodes[..., 0:2]
        agent_to_target = target_abs_pos.unsqueeze(1) - agent_pos.unsqueeze(2)
        distance = torch.linalg.vector_norm(
            agent_to_target, dim=-1, keepdim=True
        )
        pair_geometry = torch.cat([agent_to_target, distance], dim=-1)
        agent_expanded = agent_feats.unsqueeze(2).expand(-1, -1, n_targets, -1)
        target_expanded = target_feats.unsqueeze(1).expand(-1, n_agents, -1, -1)

        utility_input = torch.cat(
            [agent_expanded, target_expanded, pair_geometry], dim=-1
        )
        utilities = self.utility_scorer(utility_input).squeeze(-1)
        ego_utilities = utilities.gather(
            1,
            ego_agent_ids.long().view(batch_size, 1, 1).expand(-1, 1, n_targets),
        ).squeeze(1)

        ego_expanded = ego_feats.view(batch_size, 1, 1, -1).expand(
            -1, n_agents, n_targets, -1
        )
        intent_target_logits = self.intent_scorer(
            torch.cat(
                [ego_expanded, agent_expanded, target_expanded, pair_geometry],
                dim=-1,
            )
        ).squeeze(-1)
        valid_targets = intent_target_mask.bool().unsqueeze(1).expand(
            -1, n_agents, -1
        )
        intent_target_logits = intent_target_logits.masked_fill(
            ~valid_targets, torch.finfo(intent_target_logits.dtype).min
        )
        none_logits = self.none_scorer(
            torch.cat(
                [
                    ego_feats.unsqueeze(1).expand(-1, n_agents, -1),
                    agent_feats,
                ],
                dim=-1,
            )
        )
        intent_logits = torch.cat([intent_target_logits, none_logits], dim=-1)
        intent_probs = torch.softmax(intent_logits, dim=-1)

        final_logits, pressure, comparative_advantage, teammate_mask = (
            self.apply_soft_gate(
                base_logits,
                intent_probs=intent_probs,
                utilities=utilities,
                ego_utilities=ego_utilities,
                ego_agent_ids=ego_agent_ids,
                agent_active_mask=agent_active_mask,
                target_mask=target_mask,
                rrt_top_k=rrt_top_k,
                beta=self.beta,
                margin=self.margin,
                temperature=self.temperature,
            )
        )
        return {
            "final_logits": final_logits,
            "base_logits": base_logits,
            "intent_logits": intent_logits,
            "intent_probs": intent_probs,
            "utilities": utilities,
            "ego_utilities": ego_utilities,
            "competition_pressure": pressure,
            "comparative_advantage": comparative_advantage,
            "teammate_competition_mask": teammate_mask,
        }

    @staticmethod
    def apply_soft_gate(
        base_logits: Tensor,
        *,
        intent_probs: Tensor,
        utilities: Tensor,
        ego_utilities: Tensor,
        ego_agent_ids: Tensor,
        agent_active_mask: Tensor,
        target_mask: Tensor,
        rrt_top_k: int,
        beta: float,
        margin: float,
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply finite residual suppression only to legal rescue logits."""
        n_targets = target_mask.shape[-1]
        teammate_mask = agent_active_mask.bool().clone()
        teammate_mask.scatter_(1, ego_agent_ids.long().view(-1, 1), False)
        comparative_advantage = torch.sigmoid(
            (utilities - ego_utilities.unsqueeze(1) - margin) / temperature
        )
        competition = intent_probs[..., :n_targets] * comparative_advantage
        competition = competition.masked_fill(
            ~teammate_mask.unsqueeze(-1), 0.0
        )
        pressure = competition.max(dim=1).values * target_mask.float()
        final_logits = base_logits.clone()
        final_logits[:, rrt_top_k : rrt_top_k + n_targets] = (
            base_logits[:, rrt_top_k : rrt_top_k + n_targets]
            - beta * pressure
        )
        return final_logits, pressure, comparative_advantage, teammate_mask


class HighLevelMapCritic(nn.Module):
    """Centralized high-level critic over map channels and agent nodes."""

    def __init__(
        self,
        n_agents: int,
        map_channels: int = 3,
        agent_features: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.map_backbone = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
        )
        self.agent_encoder = _mlp(agent_features, 64)
        self.value = nn.Sequential(
            nn.Linear(256 + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init_weights)

    def forward(
        self,
        maps: Tensor,
        agent_nodes: Tensor,
        agent_ids: Tensor | None = None,
    ) -> Tensor:
        if maps.ndim == 4:
            map_features = self.map_backbone(maps)
            agent_features = self.agent_encoder(agent_nodes)
            return self.value(
                torch.cat([map_features, agent_features, agent_features], dim=-1)
            )

        batch_size, n_agents = maps.shape[:2]
        map_features = self.map_backbone(
            maps.reshape(batch_size * n_agents, *maps.shape[2:])
        ).view(batch_size, n_agents, -1)
        global_map_features = map_features.mean(dim=1)

        agent_features = self.agent_encoder(
            agent_nodes.reshape(batch_size * n_agents, -1)
        ).view(batch_size, n_agents, -1)
        global_agent_features = agent_features.mean(dim=1)
        if agent_ids is None:
            selected_agent_features = global_agent_features
        else:
            gather_ids = agent_ids.long().view(batch_size, 1, 1).expand(
                -1,
                1,
                agent_features.shape[-1],
            )
            selected_agent_features = agent_features.gather(1, gather_ids).squeeze(1)
        return self.value(
            torch.cat(
                [global_map_features, global_agent_features, selected_agent_features],
                dim=-1,
            )
        )


def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim),
        nn.ReLU(),
    )


def _squeeze_mask(mask: Tensor) -> Tensor:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask.squeeze(-1).float()
    return mask.float()


def _relative_edges(relative_xy: Tensor) -> Tensor:
    distance = torch.linalg.vector_norm(relative_xy, dim=-1, keepdim=True)
    safe_distance = distance.clamp_min(1e-6)
    cos_theta = relative_xy[..., 0:1] / safe_distance
    sin_theta = relative_xy[..., 1:2] / safe_distance
    return torch.cat([distance, cos_theta, sin_theta], dim=-1)
