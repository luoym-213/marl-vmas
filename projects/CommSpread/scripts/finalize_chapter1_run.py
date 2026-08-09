"""Select a checkpoint and derive final evaluation config for a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comm_spread.chapter1_finalizer import finalize_chapter1_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--episodes-per-seed", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision, decision_path, config_path = finalize_chapter1_run(
        args.run_dir,
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
    )
    print(json.dumps(decision["selected_checkpoint"], indent=2, sort_keys=True))
    print(f"health_status: {decision['health_status']}")
    print(f"wrote/reused {decision_path}")
    print(f"wrote/reused {config_path}")


if __name__ == "__main__":
    main()
