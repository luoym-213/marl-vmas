"""Generate diagnostics for the hidden-goal leader-follower scenario."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

try:
    from torchrl.envs.libs.vmas import VmasEnv
except ImportError:
    from torchrl.envs import VmasEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

CONFIG_ROOT = PROJECT_ROOT / "configs"

from comm_limited_vmas.scenarios.comm_hidden_goal_navigation import Scenario


OBS_DIM = 25
N_AGENTS = 3
LEADER_INDEX = 0
FIRST_TEAMMATE_BLOCK = 9
TEAMMATE_BLOCK_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose comm_hidden_goal_navigation observations and rendering."
    )
    parser.add_argument("--comm", default="comm_full")
    parser.add_argument("--estimator", default="stale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gain", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/diagnostics/comm_hidden_goal_navigation/{comm}_seed{seed}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_config = load_yaml("task", "comm_hidden_goal_navigation")
    comm_config = load_yaml("comm", args.comm)
    estimator_config = load_yaml("estimator", args.estimator)
    max_steps = int(args.max_steps or task_config["max_steps"])

    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "outputs"
        / "diagnostics"
        / "comm_hidden_goal_navigation"
        / f"{args.comm}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(task_config)
    env_kwargs.pop("scenario", None)
    continuous_actions = bool(env_kwargs.pop("continuous_actions", True))

    env = VmasEnv(
        scenario=Scenario(
            comm_config=comm_config,
            estimator_config=estimator_config,
        ),
        num_envs=1,
        continuous_actions=continuous_actions,
        categorical_actions=not continuous_actions,
        clamp_actions=True,
        seed=args.seed,
        device=args.device,
        **env_kwargs,
    )

    try:
        result = run_diagnostic_episode(
            env=env,
            output_dir=output_dir,
            comm_name=args.comm,
            comm_config=comm_config,
            task_config=task_config,
            max_steps=max_steps,
            gain=args.gain,
            fps=args.fps,
        )
    finally:
        env.close()

    print(f"Wrote diagnostics to: {output_dir}")
    print(f"All checks passed: {result['all_checks_passed']}")
    if not result["all_checks_passed"]:
        failed = [check for check in result["checks"] if not check["passed"]]
        print(json.dumps(failed, indent=2, sort_keys=True))


def run_diagnostic_episode(
    env: VmasEnv,
    output_dir: Path,
    comm_name: str,
    comm_config: dict[str, Any],
    task_config: dict[str, Any],
    max_steps: int,
    gain: float,
    fps: int,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    native_env = getattr(env, "_env", None)
    if native_env is None or not hasattr(native_env, "render"):
        raise RuntimeError("VmasEnv does not expose a renderable native _env")

    td = env.reset()
    initial_td = td.clone()
    frames = [native_env.render(mode="rgb_array", env_index=0)]
    trace_rows = [trace_row(step=0, td=td, native_env=native_env)]

    checks = validate_initial_state(initial_td, native_env, task_config)
    observations = observation_report(initial_td, task_config)

    done = bool(td["done"][0].item())
    success = bool(
        td[("agents", "info", "all_goals_reached")][0, LEADER_INDEX, 0].item()
    )

    for step in range(1, max_steps + 1):
        action = scripted_action(td[("agents", "observation")], gain=gain)
        td.set(("agents", "action"), action)
        step_td = env.step(td)
        td = step_td["next"]

        frames.append(native_env.render(mode="rgb_array", env_index=0))
        trace_rows.append(trace_row(step=step, td=td, native_env=native_env))

        done = bool(td["done"][0].item())
        success = bool(
            td[("agents", "info", "all_goals_reached")][
                0,
                LEADER_INDEX,
                0,
            ].item()
        )
        if done:
            break

    gif_path = output_dir / "episode.gif"
    imageio.mimsave(gif_path, frames, fps=fps)

    trace_path = output_dir / "trace.csv"
    write_trace(trace_path, trace_rows)

    obs_path = output_dir / "observations_step000.json"
    write_json(obs_path, observations)

    summary = {
        "task": "comm_hidden_goal_navigation",
        "comm": comm_name,
        "comm_config": comm_config,
        "max_steps": max_steps,
        "steps_recorded": len(trace_rows) - 1,
        "done": done,
        "success": success,
        "all_checks_passed": all(check["passed"] for check in checks),
        "checks": checks,
        "initial_spawn": spawn_report(initial_td, native_env),
        "final_metrics": final_metrics(td),
        "outputs": {
            "gif": str(gif_path),
            "trace_csv": str(trace_path),
            "observations_step000_json": str(obs_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def scripted_action(observation: Tensor, gain: float) -> Tensor:
    action = torch.zeros(
        observation.shape[0],
        N_AGENTS,
        2,
        dtype=observation.dtype,
        device=observation.device,
    )
    action[:, 0] = observation[:, 0, 4:6]
    action[:, 1] = observation[:, 1, 9:11]
    action[:, 2] = observation[:, 2, 9:11]
    return clamp_with_norm(action * gain, max_norm=1.0)


def clamp_with_norm(value: Tensor, max_norm: float) -> Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return value * scale


def validate_initial_state(
    td,
    native_env: Any,
    task_config: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    obs = td[("agents", "observation")]
    info = td[("agents", "info")]
    world = native_env.world
    agent_positions = stack_agent_positions(world)
    target_pos = world.landmarks[0].state.pos

    add_check(checks, "observation_shape", tuple(obs.shape) == (1, N_AGENTS, OBS_DIM))
    add_check(checks, "leader_role_one_hot", close(obs[0, 0, 7:9], [1.0, 0.0]))
    add_check(checks, "follower_roles_one_hot", close(obs[0, 1:, 7:9], [[0.0, 1.0], [0.0, 1.0]]))
    add_check(checks, "leader_goal_visible", close(obs[0, 0, 6], 1.0))
    add_check(checks, "follower_goal_hidden", close(obs[0, 1:, 4:7], torch.zeros(2, 3)))
    add_check(
        checks,
        "leader_goal_relative_position",
        torch.allclose(obs[0, 0, 4:6], target_pos[0] - agent_positions[0, 0]),
    )

    agent_radius = float(task_config["agent_spawn_radius"])
    target_radius = float(task_config["target_spawn_radius"])
    min_leader_target_dist = float(task_config["min_leader_target_dist"])
    leader_target_dist = torch.linalg.vector_norm(
        target_pos[0] - agent_positions[0, LEADER_INDEX],
    )
    add_check(
        checks,
        "agents_spawn_in_center_box",
        bool((agent_positions.abs() <= agent_radius + 1e-6).all().item()),
    )
    add_check(
        checks,
        "target_spawn_in_target_box",
        bool((target_pos.abs() <= target_radius + 1e-6).all().item()),
    )
    add_check(
        checks,
        "leader_target_min_distance",
        float(leader_target_dist.item()) > min_leader_target_dist,
    )
    add_check(
        checks,
        "agents_non_colliding_entities",
        all(not agent.collide for agent in world.agents),
    )

    cached_state = info["cached_state"]
    aoi = info["aoi"]
    comm_mask = info["comm_mask"]
    aoi_normalizer = float(task_config["aoi_normalizer"])
    for ego_index in range(N_AGENTS):
        for block_index, teammate_index in enumerate(teammate_order(ego_index)):
            block = teammate_block(obs[0, ego_index], block_index)
            cached = cached_state[0, ego_index, teammate_index]
            expected = torch.cat(
                [
                    cached[:2] - obs[0, ego_index, :2],
                    cached[2:] - obs[0, ego_index, 2:4],
                    torch.tensor(
                        [
                            min(float(aoi[0, ego_index, teammate_index].item()) / aoi_normalizer, 1.0),
                            float(comm_mask[0, ego_index, teammate_index].item()),
                        ],
                        dtype=obs.dtype,
                        device=obs.device,
                    ),
                    role_for(teammate_index, obs.device, obs.dtype),
                ],
            )
            add_check(
                checks,
                f"teammate_block_ego{ego_index}_sender{teammate_index}",
                torch.allclose(block, expected, atol=1e-5),
            )
            add_check(
                checks,
                f"normalized_aoi_range_ego{ego_index}_sender{teammate_index}",
                bool((0.0 <= block[4].item()) and (block[4].item() <= 1.0)),
            )
            expected_normalized_aoi = min(
                float(aoi[0, ego_index, teammate_index].item()) / aoi_normalizer,
                1.0,
            )
            add_check(
                checks,
                f"normalized_aoi_value_ego{ego_index}_sender{teammate_index}",
                abs(float(block[4].item()) - expected_normalized_aoi) <= 1e-5,
            )

    return checks


def observation_report(td, task_config: dict[str, Any]) -> dict[str, Any]:
    obs = td[("agents", "observation")][0]
    info = td[("agents", "info")]
    aoi_normalizer = float(task_config["aoi_normalizer"])
    report: dict[str, Any] = {}
    for ego_index in range(N_AGENTS):
        agent_report = {
            "raw": tensor_to_list(obs[ego_index]),
            "self": {
                "pos": tensor_to_list(obs[ego_index, 0:2]),
                "vel": tensor_to_list(obs[ego_index, 2:4]),
            },
            "goal": {
                "relative_pos": tensor_to_list(obs[ego_index, 4:6]),
                "visible": float(obs[ego_index, 6].item()),
            },
            "ego_role": tensor_to_list(obs[ego_index, 7:9]),
            "teammates": [],
            "info": {
                "aoi": tensor_to_list(info["aoi"][0, ego_index]),
                "comm_mask": tensor_to_list(info["comm_mask"][0, ego_index]),
                "cached_state": tensor_to_list(info["cached_state"][0, ego_index]),
            },
        }
        for block_index, teammate_index in enumerate(teammate_order(ego_index)):
            block = teammate_block(obs[ego_index], block_index)
            raw_aoi = float(info["aoi"][0, ego_index, teammate_index].item())
            agent_report["teammates"].append(
                {
                    "agent_index": teammate_index,
                    "relative_cached_pos": tensor_to_list(block[0:2]),
                    "relative_cached_vel": tensor_to_list(block[2:4]),
                    "raw_aoi": raw_aoi,
                    "normalized_aoi": float(block[4].item()),
                    "expected_normalized_aoi": min(raw_aoi / aoi_normalizer, 1.0),
                    "fresh_comm_mask": float(block[5].item()),
                    "role": tensor_to_list(block[6:8]),
                }
            )
        report[f"agent_{ego_index}"] = agent_report
    return report


def trace_row(step: int, td, native_env: Any) -> dict[str, Any]:
    world = native_env.world
    agent_pos = stack_agent_positions(world)[0]
    target_pos = world.landmarks[0].state.pos[0]
    info = td[("agents", "info")]
    reward = td.get(("agents", "reward"), None)
    if reward is None:
        reward_value = 0.0
    else:
        reward_value = float(reward[0, 0, 0].item())

    row = {
        "step": step,
        "done": bool(td["done"][0].item()),
        "success": bool(info["all_goals_reached"][0, 0, 0].item()),
        "reward_agent0": reward_value,
        "target_x": float(target_pos[0].item()),
        "target_y": float(target_pos[1].item()),
        "team_distance": float(info["team_distance"][0, 0, 0].item()),
        "leader_distance": float(info["leader_distance"][0, 0, 0].item()),
        "follow_error": float(info["follow_error"][0, 0, 0].item()),
        "team_progress": float(info["team_progress"][0, 0, 0].item()),
        "leader_progress": float(info["leader_progress"][0, 0, 0].item()),
        "follow_progress": float(info["follow_progress"][0, 0, 0].item()),
        "mean_aoi": float(info["mean_aoi"].mean().item()),
        "mean_comm_mask": float(info["mean_comm_mask"].mean().item()),
    }
    for agent_index in range(N_AGENTS):
        row[f"agent_{agent_index}_x"] = float(agent_pos[agent_index, 0].item())
        row[f"agent_{agent_index}_y"] = float(agent_pos[agent_index, 1].item())
    return row


def spawn_report(td, native_env: Any) -> dict[str, Any]:
    world = native_env.world
    agent_pos = stack_agent_positions(world)[0]
    target_pos = world.landmarks[0].state.pos[0]
    leader_target_dist = torch.linalg.vector_norm(target_pos - agent_pos[0])
    return {
        "agents": {
            f"agent_{i}": {
                "pos": tensor_to_list(agent_pos[i]),
                "collide": bool(world.agents[i].collide),
            }
            for i in range(N_AGENTS)
        },
        "target": {"pos": tensor_to_list(target_pos)},
        "leader_target_distance": float(leader_target_dist.item()),
        "initial_done": bool(td["done"][0].item()),
    }


def final_metrics(td) -> dict[str, float | bool]:
    info = td[("agents", "info")]
    return {
        "done": bool(td["done"][0].item()),
        "success": bool(info["all_goals_reached"][0, 0, 0].item()),
        "team_distance": float(info["team_distance"][0, 0, 0].item()),
        "leader_distance": float(info["leader_distance"][0, 0, 0].item()),
        "follow_error": float(info["follow_error"][0, 0, 0].item()),
        "team_progress": float(info["team_progress"][0, 0, 0].item()),
        "leader_progress": float(info["leader_progress"][0, 0, 0].item()),
        "follow_progress": float(info["follow_progress"][0, 0, 0].item()),
        "mean_aoi": float(info["mean_aoi"].mean().item()),
        "mean_comm_mask": float(info["mean_comm_mask"].mean().item()),
    }


def teammate_order(ego_index: int) -> list[int]:
    return [index for index in range(N_AGENTS) if index != ego_index]


def teammate_block(agent_obs: Tensor, block_index: int) -> Tensor:
    start = FIRST_TEAMMATE_BLOCK + block_index * TEAMMATE_BLOCK_SIZE
    return agent_obs[start : start + TEAMMATE_BLOCK_SIZE]


def role_for(agent_index: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(
        [1.0, 0.0] if agent_index == LEADER_INDEX else [0.0, 1.0],
        dtype=dtype,
        device=device,
    )


def stack_agent_positions(world: Any) -> Tensor:
    return torch.stack([agent.state.pos for agent in world.agents], dim=1)


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool | Tensor,
) -> None:
    if isinstance(passed, Tensor):
        passed = bool(passed.detach().cpu().item())
    checks.append({"name": name, "passed": bool(passed)})


def close(actual: Tensor, expected: Any, atol: float = 1e-5) -> bool:
    if not isinstance(expected, Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype, device=actual.device)
    else:
        expected = expected.to(dtype=actual.dtype, device=actual.device)
    return bool(torch.allclose(actual, expected, atol=atol))


def write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=True)


def tensor_to_list(value: Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        return tensor_to_list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value


def load_yaml(group: str, name: str) -> dict[str, Any]:
    path = CONFIG_ROOT / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


if __name__ == "__main__":
    main()
