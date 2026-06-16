"""Inspect the SAR fixed-interval high-level wrapper."""

from __future__ import annotations

import argparse

import torch

from comm_spread.env_factory import make_sar_env
from comm_spread.macro_env import SarFixedIntervalMacroEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_sar_env(num_envs=args.num_envs, device=args.device, seed=args.seed)
    macro_env = SarFixedIntervalMacroEnv(env, high_level_interval=args.interval)
    macro_env.reset()

    high_obs = macro_env.high_level_observation()
    print(f"high obs keys: {list(high_obs.keys())}")
    for key, value in high_obs.items():
        print(f"  {key}: shape={tuple(value.shape)}")

    actions = torch.zeros(
        args.num_envs,
        env.scenario.n_agents,
        dtype=torch.long,
        device=env.scenario.assigned_goals.device,
    )
    _, rewards, dones, info = macro_env.step(actions)
    print(f"macro rewards shape={tuple(rewards.shape)}")
    print(f"dones shape={tuple(dones.shape)}, true_count={dones.sum().item()}")
    print(f"info entries={len(info)}, info[0] keys={list(info[0].keys())}")


if __name__ == "__main__":
    main()
