#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAPPO_DEVICE="${MAPPO_DEVICE:-cuda:0}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python scripts/run_mappo_baseline_radius_sweep.py \
  --experiment-set assignment-prior \
  --radii 030,040,050 \
  --batch-id mappo-assignment-prior-tuned-reward-seed0-v1 \
  --device "${MAPPO_DEVICE}" \
  "$@"
