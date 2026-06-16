"""PyTorch modules extracted from the HGSAR/MPNN design."""

from __future__ import annotations

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
    """High-level node selector over exploration and detected-target nodes."""

    def __init__(
        self,
        ego_features: int = 5,
        teammate_features: int = 5,
        explore_features: int = 4,
        target_features: int = 4,
        hidden_dim: int = 128,
        node_dim: int = 64,
    ) -> None:
        super().__init__()
        self.ego_encoder = _mlp(ego_features, node_dim)
        self.teammate_encoder = _mlp(teammate_features, node_dim)
        self.explore_encoder = _mlp(explore_features, node_dim)
        self.target_encoder = _mlp(target_features, node_dim)

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim * 2, node_dim)
        self.v_proj = nn.Linear(node_dim * 2, node_dim)
        self.context = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
            nn.ReLU(),
        )
        self.node_selection_head = nn.Linear(node_dim, 1)
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
    ) -> Tensor:
        """Return logits for concatenated [explore_nodes, target_nodes].

        Shapes:
            ego_nodes: [B, 5]
            teammate_nodes: [B, A, 5]
            explore_nodes: [B, K, 4]
            target_nodes: [B, L, 4]
            target_mask: [B, L] or [B, L, 1], true for valid nodes.
        """

        ego = self.ego_encoder(ego_nodes)
        teammates = self.teammate_encoder(teammate_nodes)
        if teammate_mask is not None:
            mask = _squeeze_mask(teammate_mask).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            teammate_context = (teammates * mask).sum(dim=1) / denom
        else:
            teammate_context = teammates.mean(dim=1)

        global_context = self.context(torch.cat([ego, teammate_context], dim=-1))

        explore = self.explore_encoder(explore_nodes)
        target = self.target_encoder(target_nodes)
        nodes = torch.cat([explore, target], dim=1)
        context = global_context.unsqueeze(1).expand(-1, nodes.size(1), -1)

        query = self.q_proj(ego).unsqueeze(1)
        keys = self.k_proj(torch.cat([nodes, context], dim=-1))
        values = self.v_proj(torch.cat([nodes, context], dim=-1))
        attn = torch.softmax((query * keys).sum(dim=-1, keepdim=True) / keys.size(-1) ** 0.5, dim=1)
        fused = values * attn + nodes
        logits = self.node_selection_head(fused).squeeze(-1)

        if target_mask is not None:
            target_valid = _squeeze_mask(target_mask).bool()
            explore_valid = torch.ones(
                logits.shape[0],
                explore_nodes.shape[1],
                dtype=torch.bool,
                device=logits.device,
            )
            valid = torch.cat([explore_valid, target_valid], dim=1)
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        return logits


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
            nn.Linear(256 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init_weights)

    def forward(self, maps: Tensor, agent_nodes: Tensor) -> Tensor:
        map_features = self.map_backbone(maps)
        agent_features = self.agent_encoder(agent_nodes)
        map_features = map_features.unsqueeze(1).expand(-1, self.n_agents, -1)
        return self.value(torch.cat([map_features, agent_features], dim=-1)).squeeze(-1)


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
