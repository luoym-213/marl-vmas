"""Controlled one-episode probe for MPE-aligned VMAS belief semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from comm_spread.env_factory import make_sar_env
from comm_spread.macro_env import SarFixedIntervalMacroEnv


def _set_layout(scenario) -> None:
    agent_positions = [(-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)]
    target_positions = [(-0.70, -0.75), (0.75, 0.75), (-0.75, 0.75)]
    for entity, position in zip(scenario.world.agents, agent_positions):
        entity.set_pos(torch.tensor(position), batch_index=0)
        entity.set_vel(torch.zeros(2), batch_index=0)
    for entity, position in zip(scenario.targets, target_positions):
        entity.set_pos(torch.tensor(position), batch_index=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path)
    args = parser.parse_args()

    env = make_sar_env(
        num_envs=1,
        device="cpu",
        seed=42,
        max_steps=12,
        sensor_radius=0.3,
        enable_rrt_candidates=False,
    )
    env.reset()
    scenario = env.scenario
    macro = SarFixedIntervalMacroEnv(env)
    _set_layout(scenario)
    scenario.belief_maps.fill_(scenario.initial_belief)
    scenario.target_detected.zero_()
    scenario.target_visited.zero_()
    scenario.target_detected_step.fill_(-1)
    scenario.detected_target_positions.zero_()
    scenario.world_steps.zero_()

    timeline: list[dict[str, object]] = []
    snapshots: dict[int, torch.Tensor] = {}
    detected_at: int | None = None
    detection_distance: float | None = None
    for step in range(11):
        scenario._update_beliefs_and_discoveries()
        scenario._refresh_high_level_state()
        high_obs = macro.high_level_observation()
        detected = scenario.target_detected[0, :, 0]
        if bool(detected.all()) and detected_at is None:
            detected_at = step
            detection_distance = float(
                torch.linalg.vector_norm(
                    scenario.world.agents[0].state.pos[0]
                    - scenario.targets[0].state.pos[0]
                )
            )
        if step == 3:
            scenario.world.agents[0].set_pos(torch.tensor([0.75, 0.0]), batch_index=0)
        if step == 8:
            scenario.target_visited[0, 0] = True
            scenario._refresh_high_level_state()
            high_obs = macro.high_level_observation()
        if step in {0, 2, 5, 10}:
            snapshots[step] = scenario.belief_maps[0, 0].detach().cpu().clone()
        target_cell = (
            (scenario.targets[0].state.pos[0] + scenario.world_size / 2.0)
            / scenario.cell_size
        ).long().clamp(0, scenario.map_dim - 1)
        timeline.append(
            {
                "step": step,
                "belief_at_target": float(
                    scenario.belief_maps[0, 0, target_cell[0], target_cell[1]]
                ),
                "team_detected": detected.tolist(),
                "target_node_count_per_agent": high_obs["target_mask"][0].sum(dim=-1).tolist(),
                "visited": bool(scenario.target_visited[0, 0]),
                "joint_fov_cells": int(scenario.last_joint_fov[0].sum()),
                "max_single_fov_cells": int(
                    scenario.last_individual_fov[0].sum(dim=(-1, -2)).max()
                ),
            }
        )
        scenario.world_steps += 1

    result = {
        "seed": 42,
        "sensor_radius": scenario.sensor_radius,
        "sensor_fidelity": scenario.sensor_fidelity,
        "detection_threshold": scenario.belief_detection_threshold,
        "detected_at_probe_step": detected_at,
        "agent_target_distance_at_detection": detection_distance,
        "detected_position": scenario.detected_target_positions[0, 0].tolist(),
        "true_position": scenario.targets[0].state.pos[0].tolist(),
        "timeline": timeline,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(snapshots, args.tensor_output)
    if args.plot_output is not None:
        import matplotlib.pyplot as plt

        args.plot_output.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, len(snapshots), figsize=(12, 3), constrained_layout=True)
        for axis, (step, belief) in zip(axes, snapshots.items()):
            image = axis.imshow(
                belief.numpy(),
                origin="lower",
                vmin=0.0,
                vmax=1.0,
                extent=(-1.0, 1.0, -1.0, 1.0),
                cmap="coolwarm",
            )
            axis.set_title(f"probe step {step}")
            axis.set_xlabel("world x")
            axis.set_ylabel("world y")
        figure.colorbar(image, ax=axes, label="occupancy probability", shrink=0.75)
        figure.savefig(args.plot_output, dpi=160)
        plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
