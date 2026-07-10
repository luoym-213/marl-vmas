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
        explore_features: int = 4,
        target_features: int = 4,
        edge_features: int = 3,
        hidden_dim: int = 128,
        node_dim: int = 64,
    ) -> None:
        super().__init__()
        self.ego_encoder = _mlp(ego_features, node_dim)
        self.teammate_encoder = _mlp(teammate_features, node_dim)
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
            teammate_nodes: [B, A, 5]
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
        teammate_abs_pos = teammate_nodes[..., 0:2]
        teammate_rel_to_task = teammate_abs_pos.unsqueeze(1) - task_abs_pos.unsqueeze(2)
        teammate_other_feats = teammate_nodes[..., 2:].unsqueeze(1).expand(
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
