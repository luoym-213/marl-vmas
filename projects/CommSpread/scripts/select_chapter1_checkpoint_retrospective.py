"""Select a Chapter 1 checkpoint using health gate v2 retrospective evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comm_spread.chapter1_config import PROJECT_ROOT
from comm_spread.chapter1_retrospective import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    select_retrospective_checkpoint,
    write_retrospective_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--base-eval-config",
        type=Path,
        default=PROJECT_ROOT / "configs/chapter1/eval_v1.yaml",
    )
    parser.add_argument("--episodes-per-seed", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = select_retrospective_checkpoint(
        args.run_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    decision_path, config_path = write_retrospective_outputs(
        args.run_dir,
        decision,
        base_eval_config=args.base_eval_config,
        episodes_per_seed=args.episodes_per_seed,
    )
    print(json.dumps(decision["selected_checkpoint"], indent=2, sort_keys=True))
    print(f"wrote {decision_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
