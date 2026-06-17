"""Inspect SAR entropy-map updates and RRT exploration candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from comm_spread.async_smdp import AsyncSMDPCollector, GreedyExplorePolicy
from comm_spread.env_factory import make_sar_env
from comm_spread.rrt import _local_entropy, world_to_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hierarchical_sar_debug/rrt_entropy_inspect.gif"),
    )
    parser.add_argument("--render-env-index", type=int, default=0)
    parser.add_argument("--render-fps", type=int, default=30)
    return parser.parse_args()


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def save_gif(frames: list, output: Path, fps: int) -> None:
    if not frames:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    from moviepy import ImageSequenceClip

    ImageSequenceClip(frames, fps=fps).write_gif(str(output), fps=fps)
    print(f"saved render: {output}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    env = make_sar_env(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        mode="high",
        emit_info=True,
        enable_high_level_state=True,
        enable_rrt_candidates=True,
        auto_resample_goals=False,
    )
    scenario = env.scenario
    env.reset()

    initial_entropy = scenario._compute_entropy(scenario.belief_maps).detach().clone()
    assert_entropy_range(initial_entropy, "initial_entropy")
    initial_mean = float(initial_entropy.mean().cpu())
    assert_condition(abs(initial_mean - 1.0) < 1e-4, f"initial entropy mean {initial_mean} is not close to 1.0")

    assert_entropy_render_axis_mapping(scenario)
    assert_render_entropy_keeps_retired_agent_history(scenario)
    assert_fixed_seed_rrt(scenario)
    assert_candidate_geometry(scenario)

    before = initial_entropy.clone()
    agent_pos_before = torch.stack([agent.state.pos for agent in env.agents], dim=1).detach().clone()
    zero_actions = [
        torch.zeros(scenario.world.batch_dim, 2, device=scenario.world.device)
        for _ in env.agents
    ]
    env.step(zero_actions)
    after = scenario._compute_entropy(scenario.belief_maps).detach().clone()
    assert_entropy_range(after, "after_one_step_entropy")
    assert_sensor_entropy_update(scenario, before, after, agent_pos_before)
    assert_recent_candidate_freeze(env)

    frames = []
    if args.render:
        collector = AsyncSMDPCollector(env, high_level_policy=GreedyExplorePolicy())
        collector.reset()
        frames.append(env.render(mode="rgb_array", env_index=args.render_env_index))
        for _ in range(args.steps):
            dones = collector.step()
            frames.append(env.render(mode="rgb_array", env_index=args.render_env_index))
            if bool(dones.all()):
                break
        save_gif(frames, args.output, args.render_fps)
    else:
        actions = [
            torch.zeros(scenario.world.batch_dim, 2, device=scenario.world.device)
            for _ in env.agents
        ]
        for _ in range(max(args.steps - 1, 0)):
            env.step(actions)

    final_entropy = scenario._compute_entropy(scenario.belief_maps)
    assert_entropy_range(final_entropy, "final_entropy")

    print("[rrt_entropy_inspect]")
    print(f"num_envs: {args.num_envs}")
    print(f"steps: {args.steps}")
    print(f"entropy_initial_mean: {initial_mean:.6g}")
    print(f"entropy_after_one_step_mean: {float(after.mean().cpu()):.6g}")
    print(f"entropy_final_mean: {float(final_entropy.mean().cpu()):.6g}")
    print(f"rrt_candidates_shape: {tuple(scenario.explore_candidates.shape)}")
    print("fixed_seed_rrt: ok")
    print("candidate_bounds: ok")
    print("candidate_voronoi: ok")
    print("candidate_entropy_value: ok")
    print("sensor_fov_entropy_decrease: ok")
    print("sensor_non_fov_entropy_stable: ok")
    print("entropy_render_axis_mapping: ok")
    print("render_entropy_retired_history: ok")
    print("recent_rrt_candidate_freeze: ok")


def assert_entropy_range(entropy: torch.Tensor, name: str) -> None:
    assert_condition(torch.isfinite(entropy).all().item(), f"{name} contains non-finite values")
    min_value = float(entropy.min().cpu())
    max_value = float(entropy.max().cpu())
    assert_condition(min_value >= -1e-6, f"{name} min {min_value} < 0")
    assert_condition(max_value <= 1.0 + 1e-6, f"{name} max {max_value} > 1")


def assert_fixed_seed_rrt(scenario) -> None:
    scenario._refresh_high_level_state()
    first = scenario.explore_candidates.detach().clone()
    scenario._refresh_high_level_state()
    second = scenario.explore_candidates.detach().clone()
    assert_condition(torch.allclose(first, second), "RRT candidates changed under fixed seed")


def assert_entropy_render_axis_mapping(scenario) -> None:
    point = np.asarray([0.6, -0.4], dtype=np.float32)
    grid = world_to_grid(
        point,
        map_dim=scenario.map_dim,
        world_semidim=scenario.world_size / 2.0,
    )
    entropy = np.ones((scenario.map_dim, scenario.map_dim), dtype=np.float32)
    entropy[grid[0], grid[1]] = 0.0
    render_values = entropy.T
    min_row, min_col = np.unravel_index(render_values.argmin(), render_values.shape)
    assert_condition(
        min_col == grid[0] and min_row == grid[1],
        "entropy render axis mapping should place entropy[x, y] at image[y, x]",
    )
    assert_condition(
        min_col > scenario.map_dim // 2 and min_row < scenario.map_dim // 2,
        "synthetic entropy patch did not land in the expected render quadrant",
    )


def assert_render_entropy_keeps_retired_agent_history(scenario) -> None:
    if scenario.n_agents < 2:
        return
    env_index = 0
    original_belief = scenario.belief_maps[env_index].detach().clone()
    original_active = scenario.active_agents[env_index].detach().clone()
    try:
        scenario.belief_maps[env_index] = scenario.initial_belief
        scenario.belief_maps[env_index, 0, 10:20, 10:20] = 0.9
        scenario.belief_maps[env_index, 1:, 10:20, 10:20] = scenario.initial_belief
        before = scenario._render_entropy_map(env_index).detach().clone()
        scenario.active_agents[env_index, 0] = False
        after = scenario._render_entropy_map(env_index).detach().clone()
        assert_condition(
            torch.allclose(before, after),
            "render entropy should preserve retired agents' historical exploration",
        )
    finally:
        scenario.belief_maps[env_index] = original_belief
        scenario.active_agents[env_index] = original_active


def assert_recent_candidate_freeze(env) -> None:
    scenario = env.scenario
    collector = AsyncSMDPCollector(env, high_level_policy=GreedyExplorePolicy())
    collector.reset()
    active_recent = scenario.recent_decision_ttl > 0
    assert_condition(active_recent.any().item(), "collector did not mark recent high-level decisions")
    frozen = scenario.recent_rrt_candidate_world.detach().clone()
    live_before = scenario.explore_candidates.detach().clone()
    zero_actions = [
        torch.zeros(scenario.world.batch_dim, 2, device=scenario.world.device)
        for _ in env.agents
    ]
    for _ in range(min(3, max(int(scenario.recent_decision_render_steps) - 1, 1))):
        env.step(zero_actions)
        still_recent = active_recent & (scenario.recent_decision_ttl > 0)
        if still_recent.any():
            assert_condition(
                torch.allclose(
                    scenario.recent_rrt_candidate_world[still_recent],
                    frozen[still_recent],
                ),
                "recent RRT candidate render cache changed while TTL was active",
            )
    live_after = scenario.explore_candidates.detach()
    if not torch.allclose(live_before, live_after):
        assert_condition(
            torch.allclose(
                scenario.recent_rrt_candidate_world[active_recent],
                frozen[active_recent],
            ),
            "recent RRT candidate cache followed live explore_candidates refresh",
        )


def assert_candidate_geometry(scenario) -> None:
    candidates = scenario.explore_candidates.detach().cpu().numpy()
    expected_shape = (
        scenario.world.batch_dim,
        scenario.n_agents,
        scenario.rrt_top_k,
        4,
    )
    assert_condition(candidates.shape == expected_shape, f"candidate shape {candidates.shape} != {expected_shape}")

    agent_pos = torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)
    voronoi = scenario._compute_voronoi_masks(agent_pos).detach().cpu().numpy()
    agent_pos_np = agent_pos.detach().cpu().numpy()
    entropy = scenario._compute_entropy(scenario.belief_maps).detach().cpu().numpy()
    world_semidim = scenario.world_size / 2.0
    world_nodes = agent_pos_np[:, :, None, :] + candidates[..., 0:2]
    in_bounds = np.logical_and(world_nodes >= -world_semidim - 1e-6, world_nodes <= world_semidim + 1e-6)
    assert_condition(bool(in_bounds.all()), "at least one candidate is outside world bounds")

    grid_nodes = world_to_grid(
        world_nodes.reshape(-1, 2),
        map_dim=scenario.map_dim,
        world_semidim=world_semidim,
    ).reshape(scenario.world.batch_dim, scenario.n_agents, scenario.rrt_top_k, 2)
    empty_voronoi = []
    for env_index in range(scenario.world.batch_dim):
        for agent_index in range(scenario.n_agents):
            mask = voronoi[env_index, agent_index]
            if not mask.any():
                empty_voronoi.append((env_index, agent_index))
                continue
            for candidate_index in range(scenario.rrt_top_k):
                x, y = grid_nodes[env_index, agent_index, candidate_index]
                assert_condition(
                    bool(mask[x, y]),
                    f"candidate {(env_index, agent_index, candidate_index)} is outside its Voronoi mask",
                )
                expected = _local_entropy(
                    entropy[env_index, agent_index],
                    np.asarray([x, y]),
                    radius=2,
                ) / 25.0
                actual = float(candidates[env_index, agent_index, candidate_index, 2])
                assert_condition(
                    abs(expected - actual) < 1e-5,
                    f"candidate entropy value mismatch: expected {expected}, got {actual}",
                )
    if empty_voronoi:
        print(f"empty_voronoi_fallback_agents: {empty_voronoi}")


def assert_sensor_entropy_update(
    scenario,
    before: torch.Tensor,
    after: torch.Tensor,
    agent_pos_before: torch.Tensor,
) -> None:
    changed_any = False
    for agent_index in range(scenario.n_agents):
        pos = agent_pos_before[:, agent_index]
        dx = scenario.cell_world_x.unsqueeze(0) - pos[:, 0].view(-1, 1, 1)
        dy = scenario.cell_world_y.unsqueeze(0) - pos[:, 1].view(-1, 1, 1)
        in_fov = (dx.square() + dy.square()) <= scenario.sensor_radius**2
        active = scenario.active_agents[:, agent_index].view(-1, 1, 1)
        in_fov = in_fov & active
        fov_delta = before[:, agent_index][in_fov] - after[:, agent_index][in_fov]
        assert_condition(fov_delta.numel() > 0, f"agent {agent_index} has empty FOV")
        assert_condition(float(fov_delta.mean().cpu()) > 1e-3, f"agent {agent_index} FOV entropy did not decrease")
        changed_any = changed_any or bool((fov_delta > 1e-6).any().item())

        non_fov_delta = (before[:, agent_index] - after[:, agent_index])[~in_fov]
        if non_fov_delta.numel() > 0:
            assert_condition(
                float(non_fov_delta.abs().max().cpu()) < 1e-6,
                f"agent {agent_index} non-FOV entropy changed",
            )
    assert_condition(changed_any, "no sensor FOV entropy decrease detected")


if __name__ == "__main__":
    main()
