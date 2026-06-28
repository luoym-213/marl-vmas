"""Graph residual correction on top of AoI-conditioned temporal estimates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from comm_limited_vmas.estimators.aoi_residual import (
    AoiResidualEstimator,
    make_mlp,
    positional_encoding,
)
from comm_limited_vmas.estimators.base import BaseEstimator
from comm_limited_vmas.estimators.kinematic import KinematicEstimator
from comm_limited_vmas.estimators.stale import StaleEstimator


class GraphAttentionLayer(nn.Module):
    """A small fully-connected directed attention layer with edge bias."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        edge_hidden_layers: Sequence[int],
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.edge_bias = make_mlp(edge_dim, edge_hidden_layers, 1)

    def forward(self, h: Tensor, edge_features: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        logits = logits + self.edge_bias(edge_features).squeeze(-1)

        n_nodes = h.shape[-2]
        eye = torch.eye(n_nodes, dtype=torch.bool, device=h.device)
        logits = logits.masked_fill(eye.view(1, n_nodes, n_nodes), -1.0e9)
        attention = torch.softmax(logits, dim=-1)
        message = torch.matmul(attention, v)
        return h + message, attention


class GnnResidualNetwork(nn.Module):
    """Relation residual network for one receiver-centered communication graph."""

    def __init__(
        self,
        pe_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        node_hidden_layers: Sequence[int] = (128,),
        edge_hidden_layers: Sequence[int] = (128,),
        gate_hidden_layers: Sequence[int] = (128,),
    ):
        super().__init__()
        self.pe_dim = int(pe_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.node_dim = 4 + 4 + self.pe_dim + 1 + 2
        self.edge_dim = 4 + 2 + self.pe_dim + self.pe_dim + 2 + 4

        self.node_encoder = make_mlp(
            input_dim=self.node_dim,
            hidden_layers=node_hidden_layers,
            output_dim=self.hidden_dim,
        )
        self.layers = nn.ModuleList(
            [
                GraphAttentionLayer(
                    hidden_dim=self.hidden_dim,
                    edge_dim=self.edge_dim,
                    edge_hidden_layers=edge_hidden_layers,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.delta_head = make_mlp(self.hidden_dim, gate_hidden_layers, 4)
        self.gate_head = make_mlp(
            self.pe_dim + 1 + 4 + self.hidden_dim + 1,
            gate_hidden_layers,
            1,
        )

    def forward(
        self,
        mu_self: Tensor,
        p_self_diag: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor,
        receiver: int | Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        comm_mask = comm_mask.to(dtype=mu_self.dtype, device=mu_self.device)
        pe = positional_encoding(aoi_steps.to(mu_self.device), self.pe_dim).to(
            dtype=mu_self.dtype,
            device=mu_self.device,
        )
        roles = role_one_hot(mu_self.shape[-2], mu_self.device, mu_self.dtype)
        roles = roles.unsqueeze(0).expand(mu_self.shape[0], -1, -1)

        node_features = torch.cat(
            [mu_self, p_self_diag, pe, comm_mask.unsqueeze(-1), roles],
            dim=-1,
        )
        edge_features = build_edge_features(
            mu_self=mu_self,
            p_self_diag=p_self_diag,
            pe=pe,
            comm_mask=comm_mask,
            roles=roles,
        )

        h = self.node_encoder(node_features)
        attention = None
        for layer in self.layers:
            h, attention = layer(h, edge_features)
        assert attention is not None

        reliability = attention.max(dim=-1, keepdim=True).values
        delta_mu = self.delta_head(h)
        gate_input = torch.cat(
            [pe, comm_mask.unsqueeze(-1), p_self_diag, h, reliability],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_head(gate_input))
        corrected = mu_self + gate * delta_mu

        self_mask = receiver_mask(receiver, mu_self.shape[0], mu_self.shape[1], mu_self.device)
        corrected = torch.where(self_mask.unsqueeze(-1), mu_self, corrected)
        delta_mu = torch.where(self_mask.unsqueeze(-1), torch.zeros_like(delta_mu), delta_mu)
        gate = torch.where(self_mask.unsqueeze(-1), torch.zeros_like(gate), gate)
        return corrected, delta_mu, gate, attention


class GnnResidualEstimator(BaseEstimator):
    """Frozen graph residual estimator used by communication observations."""

    def __init__(
        self,
        state_dim: int,
        device: torch.device,
        dt: float = 0.1,
        base_estimator: dict[str, Any] | None = None,
        pe_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        node_hidden_layers: Sequence[int] = (128,),
        edge_hidden_layers: Sequence[int] = (128,),
        gate_hidden_layers: Sequence[int] = (128,),
        checkpoint_path: str | None = None,
        fallback_to_base: bool = True,
        **_: object,
    ):
        super().__init__(state_dim=state_dim, device=device)
        self.base_estimator = build_base_estimator(
            config=base_estimator,
            state_dim=state_dim,
            device=device,
            dt=dt,
        )
        self.network = GnnResidualNetwork(
            pe_dim=pe_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            node_hidden_layers=node_hidden_layers,
            edge_hidden_layers=edge_hidden_layers,
            gate_hidden_layers=gate_hidden_layers,
        ).to(device)
        self.has_checkpoint = False
        self.fallback_to_base = bool(fallback_to_base)

        if checkpoint_path:
            self.load_checkpoint(Path(checkpoint_path))
        elif not self.fallback_to_base:
            raise FileNotFoundError(
                "gnn_residual requires checkpoint_path when fallback_to_base=false"
            )

        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

    def estimate_state(
        self,
        cached_state: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
    ) -> Tensor:
        return self.base_estimator.estimate_state(cached_state, aoi_steps, comm_mask)

    def estimate_receiver_states(
        self,
        cached_states: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
        receiver: int | None = None,
    ) -> Tensor:
        mean, _ = self.estimate_receiver_distribution(
            cached_states,
            aoi_steps,
            comm_mask,
            receiver,
        )
        return mean

    def estimate_receiver_distribution(
        self,
        cached_states: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
        receiver: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        base_mean, base_cov = self.base_estimator.estimate_receiver_distribution(
            cached_states,
            aoi_steps,
            comm_mask,
            receiver,
        )
        if not self.has_checkpoint:
            return base_mean, base_cov
        if receiver is None:
            raise ValueError("gnn_residual requires receiver index for graph correction")
        if comm_mask is None:
            comm_mask = torch.zeros_like(aoi_steps, dtype=base_mean.dtype)

        with torch.no_grad():
            corrected, _, _, _ = self.network(
                mu_self=base_mean,
                p_self_diag=base_cov,
                aoi_steps=aoi_steps,
                comm_mask=comm_mask,
                receiver=receiver,
            )
        return corrected, base_cov

    def covariance_diag(self, aoi_steps: Tensor) -> Tensor:
        return self.base_estimator.covariance_diag(aoi_steps)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path = checkpoint_path.expanduser()
        if not checkpoint_path.exists():
            if self.fallback_to_base:
                return
            raise FileNotFoundError(f"GNN residual checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.network.load_state_dict(state_dict)
        self.has_checkpoint = True


def build_base_estimator(
    config: dict[str, Any] | None,
    state_dim: int,
    device: torch.device,
    dt: float,
) -> BaseEstimator:
    config = dict(config or {"name": "aoi_residual", "fallback_to_kinematic": True})
    name = str(config.pop("name", "aoi_residual"))
    if name == "stale":
        return StaleEstimator(state_dim=state_dim, device=device, **config)
    if name == "kinematic":
        return KinematicEstimator(state_dim=state_dim, device=device, dt=dt, **config)
    if name == "aoi_residual":
        return AoiResidualEstimator(state_dim=state_dim, device=device, dt=dt, **config)
    raise ValueError(f"Unsupported gnn_residual base estimator: {name}")


def build_edge_features(
    mu_self: Tensor,
    p_self_diag: Tensor,
    pe: Tensor,
    comm_mask: Tensor,
    roles: Tensor,
) -> Tensor:
    n_nodes = mu_self.shape[1]
    target_mu = mu_self.unsqueeze(2).expand(-1, -1, n_nodes, -1)
    source_mu = mu_self.unsqueeze(1).expand(-1, n_nodes, -1, -1)
    rel_mu = source_mu - target_mu

    cov_mean = p_self_diag.mean(dim=-1, keepdim=True)
    target_cov = cov_mean.unsqueeze(2).expand(-1, -1, n_nodes, -1)
    source_cov = cov_mean.unsqueeze(1).expand(-1, n_nodes, -1, -1)

    target_pe = pe.unsqueeze(2).expand(-1, -1, n_nodes, -1)
    source_pe = pe.unsqueeze(1).expand(-1, n_nodes, -1, -1)
    target_mask = comm_mask.unsqueeze(-1).unsqueeze(2).expand(-1, -1, n_nodes, -1)
    source_mask = comm_mask.unsqueeze(-1).unsqueeze(1).expand(-1, n_nodes, -1, -1)
    target_role = roles.unsqueeze(2).expand(-1, -1, n_nodes, -1)
    source_role = roles.unsqueeze(1).expand(-1, n_nodes, -1, -1)
    return torch.cat(
        [
            rel_mu,
            target_cov,
            source_cov,
            target_pe,
            source_pe,
            target_mask,
            source_mask,
            target_role,
            source_role,
        ],
        dim=-1,
    )


def role_one_hot(n_agents: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    roles = torch.zeros(n_agents, 2, dtype=dtype, device=device)
    roles[0, 0] = 1.0
    if n_agents > 1:
        roles[1:, 1] = 1.0
    return roles


def receiver_mask(
    receiver: int | Tensor,
    batch_size: int,
    n_agents: int,
    device: torch.device,
) -> Tensor:
    if isinstance(receiver, Tensor):
        receiver_tensor = receiver.to(device=device, dtype=torch.long)
    else:
        receiver_tensor = torch.full((batch_size,), int(receiver), dtype=torch.long, device=device)
    agent_indices = torch.arange(n_agents, device=device).view(1, n_agents)
    return agent_indices == receiver_tensor.view(-1, 1)
