#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: $0 SEED [OUTPUT_DIR]}"
output_dir="${2:-/tmp/mpe_physics_frozen_high_seed_${seed}}"
project_dir="/workspace/projects/CommSpread"
low_checkpoint="${MPE_PHYSICS_LOW_CHECKPOINT:-${project_dir}/outputs/sar_low_mpe_physics/full_seed_0_20260722/mappo_sar_low_mlp__0ab21273_26_07_22-09_25_34/checkpoints/checkpoint_50040000.pt}"
high_checkpoint="${MPE_PHYSICS_HIGH_CHECKPOINT:-${project_dir}/outputs/sar_high_hgsar_dynamic_staggered_repaired_mappo_sensor_0.60/seed_0/checkpoints/checkpoint_10.pt}"

mkdir -p "${output_dir}"
cd "${project_dir}"

PYTHONPATH=. python scripts/analyze_sar_failure_modes_mpe_physics.py \
  --low-level-controller checkpoint \
  --low-level-checkpoint "${low_checkpoint}" \
  --enable-low-level-goal-fallback \
  --high-level-checkpoint "${high_checkpoint}" \
  --policy hgsar \
  --num-envs 256 \
  --steps 100 \
  --max-steps 100 \
  --sensor-radius 0.6 \
  --seed "${seed}" \
  --device cuda:0 \
  --dynamic-rescue-release \
  --dynamic-rescue-only-after-all-detected \
  --redecide-on-detection-change \
  --redecide-on-assignment-change \
  --dynamic-release-max-new-agents-per-event 1 \
  --enable-finder-first-cascade \
  --finder-cascade-mode finder_only \
  --output "${output_dir}/result.json" \
  --csv-output "${output_dir}/episodes.csv" \
  --markdown "${output_dir}/report.md"
