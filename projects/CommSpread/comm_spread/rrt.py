"""RRT-Value candidate generation for SAR high-level exploration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RRTConfig:
    """Configuration for entropy-guided RRT candidate generation."""

    top_k: int = 5
    max_iterations: int = 40
    expand_dis: int = 2
    value_radius: int = 2
    world_semidim: float = 1.0
    uniform_ratio: float = 0.3
    temperature: float = 1.0
    gamma_rrt: float = 0.9
    gamma_voronoi: float = 0.3


def world_to_grid(
    positions: np.ndarray,
    *,
    map_dim: int,
    world_semidim: float,
) -> np.ndarray:
    """Convert world coordinates in [-world_semidim, world_semidim] to grid indices."""

    positions = np.asarray(positions, dtype=np.float32)
    scaled = (positions + world_semidim) / (2.0 * world_semidim)
    grid = np.floor(scaled * map_dim).astype(np.int64)
    return np.clip(grid, 0, map_dim - 1)


def grid_to_world(
    grid_positions: np.ndarray,
    *,
    map_dim: int,
    world_semidim: float,
) -> np.ndarray:
    """Convert grid indices to cell-center world coordinates."""

    grid_positions = np.asarray(grid_positions, dtype=np.float32)
    cell_size = 2.0 * world_semidim / map_dim
    return -world_semidim + (grid_positions + 0.5) * cell_size


def plan_batch(
    agent_positions: np.ndarray,
    assigned_goals: np.ndarray,
    voronoi_masks: np.ndarray,
    entropy_maps: np.ndarray,
    *,
    config: RRTConfig | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return exploration nodes with features [rel_x, rel_y, entropy_value, occupied].

    Args:
        agent_positions: Array [batch, agents, 2] in world coordinates.
        assigned_goals: Array [batch, agents, 2] in world coordinates.
        voronoi_masks: Boolean array [batch, agents, H, W].
        entropy_maps: Float array [batch, H, W].

    Returns:
        Array [batch, agents, top_k, 4].
    """

    cfg = config or RRTConfig()
    rng = np.random.default_rng(seed)

    agent_positions = np.asarray(agent_positions, dtype=np.float32)
    assigned_goals = np.asarray(assigned_goals, dtype=np.float32)
    voronoi_masks = np.asarray(voronoi_masks, dtype=bool)
    entropy_maps = np.asarray(entropy_maps, dtype=np.float32)

    batch_dim, n_agents = agent_positions.shape[:2]
    map_dim = entropy_maps.shape[-1]
    out = np.zeros((batch_dim, n_agents, cfg.top_k, 4), dtype=np.float32)

    for batch_index in range(batch_dim):
        for agent_index in range(n_agents):
            if entropy_maps.ndim == 4:
                entropy_map = entropy_maps[batch_index, agent_index]
            else:
                entropy_map = entropy_maps[batch_index]
            start_grid = world_to_grid(
                agent_positions[batch_index, agent_index],
                map_dim=map_dim,
                world_semidim=cfg.world_semidim,
            )
            mask = voronoi_masks[batch_index, agent_index]
            nodes = _plan_single(start_grid, mask, entropy_map, cfg, rng)
            world_nodes = grid_to_world(
                nodes[:, :2],
                map_dim=map_dim,
                world_semidim=cfg.world_semidim,
            )
            rel = world_nodes - agent_positions[batch_index, agent_index]
            occupied = _occupied_feature(
                world_nodes,
                assigned_goals[batch_index],
                exclude_index=agent_index,
            )
            out[batch_index, agent_index, :, 0:2] = rel
            out[batch_index, agent_index, :, 2] = nodes[:, 2]
            out[batch_index, agent_index, :, 3] = occupied

    return out


def _plan_single(
    start_grid: np.ndarray,
    voronoi_mask: np.ndarray,
    entropy_map: np.ndarray,
    cfg: RRTConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    sample_points = np.argwhere(np.ones_like(entropy_map, dtype=bool))
    if sample_points.size == 0:
        sample_points = _fallback_points(start_grid, entropy_map.shape[0])

    values = entropy_map[sample_points[:, 0], sample_points[:, 1]].astype(np.float64)
    weights = np.exp(values / max(cfg.temperature, 1e-6))
    if weights.sum() <= 0 or not np.isfinite(weights).all():
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    start_grid = np.asarray(start_grid, dtype=np.int64)
    start_value = _soft_voronoi_factor(start_grid, voronoi_mask, cfg) * _local_entropy(
        entropy_map,
        start_grid,
        cfg.value_radius,
    )
    tree = [(start_grid.astype(np.float32), float(start_value))]
    scored: list[tuple[float, int, int]] = []
    for _ in range(max(cfg.max_iterations, cfg.top_k)):
        if rng.random() < cfg.uniform_ratio:
            sample = sample_points[rng.integers(0, len(sample_points))]
        else:
            sample = sample_points[rng.choice(len(sample_points), p=weights)]

        nearest, parent_value = min(
            tree,
            key=lambda item: float(np.linalg.norm(sample - item[0])),
        )
        direction = sample.astype(np.float32) - nearest
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction = direction / norm
        new_point = nearest + direction * cfg.expand_dis
        new_point = np.rint(new_point).astype(np.int64)
        new_point = np.clip(new_point, 0, entropy_map.shape[0] - 1)

        local_value = _local_entropy(entropy_map, new_point, cfg.value_radius)
        value = _soft_voronoi_factor(new_point, voronoi_mask, cfg) * (
            cfg.gamma_rrt * parent_value + local_value
        )
        tree.append((new_point.astype(np.float32), float(value)))
        scored.append((value, int(new_point[0]), int(new_point[1])))

    if len(scored) < cfg.top_k:
        for point in sample_points:
            local_value = _local_entropy(entropy_map, point, cfg.value_radius)
            scored.append(
                (
                    _soft_voronoi_factor(point, voronoi_mask, cfg) * local_value,
                    int(point[0]),
                    int(point[1]),
                )
            )

    scored.sort(key=lambda item: item[0], reverse=True)
    result: list[tuple[float, int, int]] = []
    seen = set()
    for value, x, y in scored:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        result.append((value, x, y))
        if len(result) == cfg.top_k:
            break

    while len(result) < cfg.top_k:
        x = int(np.clip(start_grid[0] + len(result), 0, entropy_map.shape[0] - 1))
        y = int(np.clip(start_grid[1], 0, entropy_map.shape[1] - 1))
        point = np.array([x, y])
        value = _soft_voronoi_factor(point, voronoi_mask, cfg) * _local_entropy(
            entropy_map,
            point,
            cfg.value_radius,
        )
        result.append((value, x, y))

    normalizer = _value_normalizer(cfg)
    return np.asarray([[x, y, value / normalizer] for value, x, y in result], dtype=np.float32)


def _fallback_points(start_grid: np.ndarray, map_dim: int) -> np.ndarray:
    points = []
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            x = int(np.clip(start_grid[0] + dx, 0, map_dim - 1))
            y = int(np.clip(start_grid[1] + dy, 0, map_dim - 1))
            points.append((x, y))
    return np.asarray(sorted(set(points)), dtype=np.int64)


def _local_entropy(entropy_map: np.ndarray, point: np.ndarray, radius: int) -> float:
    x, y = int(point[0]), int(point[1])
    x0, x1 = max(0, x - radius), min(entropy_map.shape[0], x + radius + 1)
    y0, y1 = max(0, y - radius), min(entropy_map.shape[1], y + radius + 1)
    return float(entropy_map[x0:x1, y0:y1].sum())


def _soft_voronoi_factor(
    point: np.ndarray,
    voronoi_mask: np.ndarray,
    cfg: RRTConfig,
) -> float:
    x, y = int(point[0]), int(point[1])
    if voronoi_mask[x, y]:
        return 1.0
    return float(cfg.gamma_voronoi)


def _value_normalizer(cfg: RRTConfig) -> float:
    local_max = max(float((2 * cfg.value_radius + 1) ** 2), 1.0)
    iterations = max(cfg.max_iterations, cfg.top_k)
    if abs(cfg.gamma_rrt - 1.0) < 1e-6:
        cumulative = local_max * max(float(iterations), 1.0)
    else:
        cumulative = local_max * (1.0 - cfg.gamma_rrt ** max(iterations, 1)) / (
            1.0 - cfg.gamma_rrt
        )
    return max(cumulative, local_max, 1.0)


def _occupied_feature(
    candidate_world_nodes: np.ndarray,
    assigned_goals: np.ndarray,
    *,
    exclude_index: int,
    distance_threshold: float = 0.3,
) -> np.ndarray:
    mask = np.ones(len(assigned_goals), dtype=bool)
    if 0 <= exclude_index < len(mask):
        mask[exclude_index] = False
    other_goals = assigned_goals[mask]
    if len(other_goals) == 0:
        return np.zeros(len(candidate_world_nodes), dtype=np.float32)
    dists = np.linalg.norm(candidate_world_nodes[:, None, :] - other_goals[None, :, :], axis=-1)
    valid = dists < distance_threshold
    vals = np.where(valid, ((distance_threshold - dists) / distance_threshold) ** 2, 0.0)
    return vals.sum(axis=-1).astype(np.float32)
