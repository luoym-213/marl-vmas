"""Roll out CommSpread with random actions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from comm_spread.env_factory import make_comm_spread_env
from comm_spread.scenario import CommSpreadScenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", default="outputs/random_rollout.gif")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_comm_spread_env(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
    )

    frames = []
    start = time.time()
    for step in range(args.steps):
        actions = [env.get_random_action(agent) for agent in env.agents]
        env.step(actions)
        if args.render:
            frames.append(env.render(mode="rgb_array"))
        print(f"step {step + 1}/{args.steps}")

    elapsed = time.time() - start
    scenario_name = CommSpreadScenario.__name__
    print(
        f"{args.steps} steps of {args.num_envs} parallel envs took "
        f"{elapsed:.2f}s on {args.device} for {scenario_name}."
    )

    if args.render:
        from moviepy import ImageSequenceClip

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        ImageSequenceClip(frames, fps=30).write_gif(str(output), fps=30)
        print(f"saved gif: {output}")


if __name__ == "__main__":
    main()
