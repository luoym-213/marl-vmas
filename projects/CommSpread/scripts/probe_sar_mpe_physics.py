"""Emit deterministic motion traces from strict MPE-compatible VMAS SAR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from comm_spread.env_factory import make_sar_env


def coverage_cells(positions: list[list[float]], radius: float = 0.3) -> int:
    centers = -1.0 + (np.arange(100) + 0.5) * 0.02
    cell_x, cell_y = np.meshgrid(centers, centers, indexing="ij")
    covered = np.zeros((100, 100), dtype=bool)
    for x, y in positions:
        covered |= (cell_x - x) ** 2 + (cell_y - y) ** 2 <= radius**2
    return int(covered.sum())


def make_env(
    *,
    position: tuple[float, float] = (0.0, 0.0),
    velocity: tuple[float, float] = (0.0, 0.0),
    max_steps: int = 110,
):
    env = make_sar_env(
        num_envs=1,
        device="cpu",
        seed=42,
        max_steps=max_steps,
        physics_profile="mpe_strict",
        enable_rrt_candidates=False,
    )
    env.reset()
    scenario = env.scenario
    for agent, pos in zip(
        scenario.world.agents,
        [position, (-8.0, 8.0), (8.0, 8.0)],
    ):
        agent.set_pos(torch.tensor(pos), batch_index=0)
        agent.set_vel(torch.zeros(2), batch_index=0)
    scenario.world.agents[0].set_vel(torch.tensor(velocity), batch_index=0)
    for target, pos in zip(
        scenario.targets,
        [(-8.0, -8.0), (8.0, -8.0), (0.0, 8.0)],
    ):
        target.set_pos(torch.tensor(pos), batch_index=0)
    return env


def actions(action_id: int) -> list[torch.Tensor]:
    return [torch.tensor([action_id]), torch.tensor([0]), torch.tensor([0])]


def trace(
    action_ids: list[int],
    *,
    position: tuple[float, float] = (0.0, 0.0),
    velocity: tuple[float, float] = (0.0, 0.0),
):
    env = make_env(position=position, velocity=velocity, max_steps=len(action_ids) + 2)
    agent = env.scenario.world.agents[0]
    positions, velocities = [], []
    for action_id in action_ids:
        env.step(actions(action_id))
        positions.append(agent.state.pos[0].tolist())
        velocities.append(agent.state.vel[0].tolist())
    return {"actions": action_ids, "positions": positions, "velocities": velocities}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random_actions = np.random.RandomState(42).randint(0, 5, size=50).tolist()
    sweep_actions = ([2] * 5 + [4] * 5 + [1] * 5 + [3] * 5) * 5
    traces = {
        "constant_right_50": trace([2] * 50),
        "nonzero_alternating_20": trace(
            [2, 1, 4, 3] * 5, velocity=(0.3, -0.2)
        ),
        "seeded_random_50": trace(random_actions),
        "sweep_100": trace(sweep_actions),
    }
    first_steps = {
        str(action_id): trace([action_id])["positions"][0]
        for action_id in range(5)
    }

    collisions = {}
    for distance in (0.05, 0.08, 0.10, 0.101, 0.15):
        env = make_env(max_steps=2)
        scenario = env.scenario
        scenario.world.agents[0].set_pos(
            torch.tensor([-distance / 2.0, 0.0]), batch_index=0
        )
        scenario.world.agents[1].set_pos(
            torch.tensor([distance / 2.0, 0.0]), batch_index=0
        )
        scenario.world.agents[2].set_pos(torch.tensor([8.0, 8.0]), batch_index=0)
        for agent in scenario.world.agents:
            agent.set_vel(torch.zeros(2), batch_index=0)
        env.step([torch.tensor([0]), torch.tensor([0]), torch.tensor([0])])
        collisions[f"{distance:.3f}"] = {
            "positions": [
                scenario.world.agents[0].state.pos[0].tolist(),
                scenario.world.agents[1].state.pos[0].tolist(),
            ],
            "velocities": [
                scenario.world.agents[0].state.vel[0].tolist(),
                scenario.world.agents[1].state.vel[0].tolist(),
            ],
        }

    boundary = trace([2], position=(0.99, 0.0), velocity=(0.5, 0.0))
    boundary["penalty"] = -2.0 if abs(boundary["positions"][0][0]) >= 1.0 else 0.0
    sweep_positions = traces["sweep_100"]["positions"]
    result = {
        "engine": "vmas_sar_mpe_strict",
        "dt": 0.1,
        "substeps": 1,
        "drag": 0.25,
        "force_scale": 5.0,
        "first_steps": first_steps,
        "traces": traces,
        "collisions": collisions,
        "boundary": boundary,
        "coverage_cells": {
            "step_50": coverage_cells(sweep_positions[:50]),
            "step_100": coverage_cells(sweep_positions[:100]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
