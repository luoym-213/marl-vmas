# Official VMAS Discovery Baseline

This folder is a standalone BenchMARL baseline for official VMAS `discovery`.
It intentionally does not import `comm_limited_vmas`, custom scenarios, or local task
configs.

## Train

From the repository root mounted in Docker:

```bash
docker run --rm \
  -v /Users/ym/Public/codes/marl-vmas:/workspace \
  -w /workspace/projects \
  marl-vmas-cpu \
  python official_baseline/discovery/run_official_discovery.py \
    --device cuda \
    --seed 0
```

The script loads:

- `ExperimentConfig.get_from_yaml()`
- `VmasTask.DISCOVERY.get_from_yaml()`
- `MappoConfig.get_from_yaml()`
- `MlpConfig.get_from_yaml()` for actor and critic

Only operational fields such as `save_folder`, device, seed, and checkpoint saving
are set by the script.

For a quick smoke test:

```bash
docker run --rm \
  -v /Users/ym/Public/codes/marl-vmas:/workspace \
  -w /workspace/projects \
  marl-vmas-cpu \
  python official_baseline/discovery/run_official_discovery.py \
    --device cpu \
    --seed 0 \
    --max-n-frames 1000
```

## Evaluate

```bash
docker run --rm \
  -v /Users/ym/Public/codes/marl-vmas:/workspace \
  -w /workspace/projects \
  marl-vmas-cpu \
  python official_baseline/discovery/evaluate_official_discovery.py \
    --checkpoint official_baseline/discovery/outputs/<run>/checkpoints/<checkpoint>.pt \
    --episodes 100 \
    --device cpu \
    --seed 100
```

The evaluator reports discovery-specific success proxies for
`targets_respawn=True`, including `success_at_1`, `success_at_5`,
`success_at_10`, `coverage_per_step`, and target coverage statistics.
