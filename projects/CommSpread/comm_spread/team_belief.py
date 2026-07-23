"""Torch-native team-shared Bayesian belief utilities for SAR."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def target_occupancy_map(
    cell_world_x: Tensor,
    cell_world_y: Tensor,
    target_positions: Tensor,
    target_radius: float,
) -> Tensor:
    """Return the union of static target footprints as ``[B, H, W]``."""

    dx = cell_world_x.view(1, 1, *cell_world_x.shape) - target_positions[..., 0].view(
        target_positions.shape[0], target_positions.shape[1], 1, 1
    )
    dy = cell_world_y.view(1, 1, *cell_world_y.shape) - target_positions[..., 1].view(
        target_positions.shape[0], target_positions.shape[1], 1, 1
    )
    return (dx.square() + dy.square() <= float(target_radius) ** 2).any(dim=1)


def individual_fov_masks(
    cell_world_x: Tensor,
    cell_world_y: Tensor,
    agent_positions: Tensor,
    sensor_radius: float,
    sensor_enabled: Tensor | None = None,
) -> Tensor:
    """Return one FOV mask per agent as ``[B, A, H, W]``."""

    dx = cell_world_x.view(1, 1, *cell_world_x.shape) - agent_positions[..., 0].view(
        agent_positions.shape[0], agent_positions.shape[1], 1, 1
    )
    dy = cell_world_y.view(1, 1, *cell_world_y.shape) - agent_positions[..., 1].view(
        agent_positions.shape[0], agent_positions.shape[1], 1, 1
    )
    masks = dx.square() + dy.square() <= float(sensor_radius) ** 2
    if sensor_enabled is not None:
        masks &= sensor_enabled.unsqueeze(-1).unsqueeze(-1)
    return masks


def bayesian_update(
    prior: Tensor,
    fov_mask: Tensor,
    occupied_mask: Tensor,
    sensor_fidelity: float,
    eps: float = 1e-10,
) -> Tensor:
    """Apply MPE-equivalent positive and negative Bayesian updates."""

    fidelity = float(sensor_fidelity)
    positive_num = fidelity * prior
    positive_den = positive_num + (1.0 - fidelity) * (1.0 - prior)
    positive = positive_num / positive_den.clamp_min(eps)

    negative_num = (1.0 - fidelity) * prior
    negative_den = negative_num + fidelity * (1.0 - prior)
    negative = negative_num / negative_den.clamp_min(eps)

    observed = torch.where(occupied_mask, positive, negative).clamp_(0.0, 1.0)
    return torch.where(fov_mask, observed, prior)


def label_8_connected(binary_map: Tensor) -> Tensor:
    """Label 8-connected components using Torch max-label propagation.

    Labels are unique positive flattened cell ids. The operation stays on the
    input device and converges in at most ``max(H, W)`` iterations for the
    small target footprints used here.
    """

    if binary_map.ndim != 3:
        raise ValueError("binary_map must have shape [B, H, W]")
    batch, height, width = binary_map.shape
    cell_ids = torch.arange(
        1,
        height * width + 1,
        device=binary_map.device,
        dtype=torch.float32,
    ).view(1, height, width)
    labels = torch.where(binary_map, cell_ids.expand(batch, -1, -1), 0.0)
    for _ in range(max(height, width)):
        propagated = F.max_pool2d(labels.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        updated = torch.where(binary_map, propagated, 0.0)
        if torch.equal(updated, labels):
            break
        labels = updated
    return labels.to(torch.long)


def detected_target_centroids(
    belief_grid: Tensor,
    threshold: float,
    cell_world_x: Tensor,
    cell_world_y: Tensor,
    target_positions: Tensor,
    target_radius: float,
) -> tuple[Tensor, Tensor]:
    """Return threshold detections and 8-connected cluster centroids.

    VMAS spawn separation keeps target footprints disjoint. Each target id is
    matched to the connected component intersecting its true occupancy
    footprint; the position exposed to the policy is the component centroid,
    not the simulator target coordinate.
    """

    binary = belief_grid > float(threshold)
    labels = label_8_connected(binary)
    batch, n_targets, _ = target_positions.shape
    centroids = torch.zeros_like(target_positions)
    detected = torch.zeros(batch, n_targets, dtype=torch.bool, device=belief_grid.device)

    for target_index in range(n_targets):
        target = target_positions[:, target_index]
        dx = cell_world_x.unsqueeze(0) - target[:, 0].view(-1, 1, 1)
        dy = cell_world_y.unsqueeze(0) - target[:, 1].view(-1, 1, 1)
        footprint = dx.square() + dy.square() <= float(target_radius) ** 2
        matching_labels = torch.where(footprint, labels, torch.zeros_like(labels))
        component_ids = matching_labels.flatten(1).max(dim=1).values
        valid = component_ids > 0
        component = labels == component_ids.view(-1, 1, 1)
        component &= valid.view(-1, 1, 1)
        count = component.sum(dim=(-1, -2)).clamp_min(1).to(belief_grid.dtype)
        centroids[:, target_index, 0] = (
            component.to(belief_grid.dtype) * cell_world_x
        ).sum(dim=(-1, -2)) / count
        centroids[:, target_index, 1] = (
            component.to(belief_grid.dtype) * cell_world_y
        ).sum(dim=(-1, -2)) / count
        detected[:, target_index] = valid

    return detected, centroids
