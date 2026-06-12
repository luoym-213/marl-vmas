"""Inspect the VMAS environment interface."""

from __future__ import annotations

import argparse

from custom_vmas_benchmarl.env_factory import make_navigation_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_navigation_env(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
    )
    actions = env.get_random_actions()
    obs, rewards, dones, info = env.step(actions)

    print(f"agents: {[agent.name for agent in env.agents]}")
    print(f"actions: {len(actions)} tensors, first shape={tuple(actions[0].shape)}")
    print(f"obs entries: {len(obs)}, agent 0 keys={list(obs[0].keys())}")
    for key, value in obs[0].items():
        print(f"  obs[0]['{key}']: shape={tuple(value.shape)}")
    print(f"rewards: {len(rewards)} tensors, first shape={tuple(rewards[0].shape)}")
    print(f"dones: shape={tuple(dones.shape)}, true_count={dones.sum().item()}")
    print(f"info entries: {len(info)}, agent 0 keys={list(info[0].keys())}")


if __name__ == "__main__":
    main()
