"""Inspect asynchronous per-agent high-level SMDP collection."""

from __future__ import annotations

import argparse

from comm_spread.async_smdp import AsyncSMDPCollector
from comm_spread.env_factory import make_sar_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_sar_env(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        mode="high",
        emit_info=False,
        enable_high_level_state=True,
        enable_rrt_candidates=True,
        auto_resample_goals=False,
    )
    collector = AsyncSMDPCollector(env)
    transitions = collector.rollout(args.steps)
    print(f"transitions: {len(transitions)}")
    if transitions:
        durations = [transition.duration for transition in transitions]
        print(f"duration min/max: {min(durations)}/{max(durations)}")
        first = transitions[0]
        print(f"first env_id={first.env_id} agent_id={first.agent_id}")
        print(f"ego_node shape={tuple(first.ego_node.shape)}")
        print(f"teammate_nodes shape={tuple(first.teammate_nodes.shape)}")
        print(f"explore_nodes shape={tuple(first.explore_nodes.shape)}")
        print(f"target_nodes shape={tuple(first.target_nodes.shape)}")
        print(f"target_mask shape={tuple(first.target_mask.shape)}")
        print(f"map_channels shape={tuple(first.map_channels.shape)}")


if __name__ == "__main__":
    main()
