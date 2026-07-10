# Entity / Message-Passing Progress Report

Generated: 2026-07-09T12:41:31Z

## Dependency Status

Installed and verified:

- `torch-geometric==2.6.1`
- `torch-cluster==1.6.3+pt26cu124`
- `pyg-lib==0.5.0+pt26cu124`

`torch-geometric==2.8.0` was initially installed, but the existing BenchMARL GNN path failed because its `radius_graph` requires `pyg-lib>=0.6.0`; the available Torch 2.6/cu124 wheel was `0.5.0`. Downgrading `torch-geometric` to `2.6.1` restored compatibility.

## Spread GNN Status

Command smoke-tested successfully:

```bash
PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/train_mappo.py   --variant spread_mpe_physics_spawn_parity   --model gnn   --comms-radius 1.0   --train-device cuda:0   --sampling-device cuda:0   --seed 0   --max-n-frames 1000   --save-folder projects/CommSpread/outputs/gnn_physics_smoke/seed_0   --no-render
```

A first full seed was started for `spread_mpe_physics_spawn_parity --model gnn` and stopped early at about 74/3000 updates. Early mean returns stayed around `-38` to `-43`, which is much worse than the MLP physics-spawn baseline around `-28`. This is not enough to reject GNN fully, but it is enough to avoid spending three full seeds before checking whether the GNN receives the right entity features for MPE parity.

Current config: `configs/spread_mpe_physics_spawn_gnn.yaml`.

## SAR PPO Bug Fix

A real high-level PPO implementation bug was found in `scripts/train_high_ppo_sar.py`:

- `index_batch()` did not consistently apply the minibatch permutation to tensors whose first dimension matched the full transition batch.
- Advantages were permuted, while observations/actions/log-probs could remain unpermuted when the minibatch covered the whole rollout.
- This weakened or corrupted the policy-gradient signal.

The function now indexes all tensors whose first dimension equals the rollout transition count and leaves only shared tensors unindexed.

## SAR Results After Fix

Low-level checkpoint used throughout:

`projects/CommSpread/outputs/sar_low_full/low_safe_0.06/checkpoints/checkpoint_50040000.pt`

### Sensor Radius 0.60

Training command used 16 envs, 400 low-level steps per batch, 30 PPO updates, minibatch size 64.

Before fix:

| metric | first5 | last5 | best |
|---|---:|---:|---:|
| success_rate | 0.025 | 0.025 | 0.0625 |
| visited_targets_mean | 1.15 | 1.0875 | 1.1875 |
| actor_target_actions | 24.6 | 25.6 | 27 |

After fix:

| metric | first5 | last5 | best |
|---|---:|---:|---:|
| success_rate | 0.05 | 0.30 | 0.5625 |
| visited_targets_mean | 1.5375 | 2.125 | 2.5 |
| actor_target_actions | 28.8 | 36.0 | 41 |

Deterministic 32-env evaluation of the final checkpoint:

| success_rate | visited_targets_mean | detected_targets_mean | low_level_goal_hit_per_transition |
|---:|---:|---:|---:|
| 0.28125 | 2.15625 | 2.46875 | 0.92557 |

Files:

- Training: `projects/CommSpread/outputs/sar_high_hgsar_indexfix_sensor_0.60/seed_0`
- Eval JSON: `projects/CommSpread/outputs/hierarchical_sar_diagnostics/hgsar_indexfix_sensor_0.60_eval.json`
- Eval markdown: `projects/CommSpread/hierarchical_sar_hgsar_indexfix_0.60_eval.md`

### Sensor Radius 0.30

After fix, same training setup but original sensor radius.

| metric | first5 | last5 | best |
|---|---:|---:|---:|
| success_rate | 0.0125 | 0.05 | 0.125 |
| visited_targets_mean | 0.9125 | 1.3625 | 1.6875 |
| actor_target_actions | 16.4 | 22.8 | 27 |

Deterministic 32-env evaluation of the final checkpoint:

| success_rate | visited_targets_mean | detected_targets_mean | low_level_goal_hit_per_transition |
|---:|---:|---:|---:|
| 0.03125 | 1.34375 | 1.5625 | 0.88865 |

Files:

- Training: `projects/CommSpread/outputs/sar_high_hgsar_indexfix_sensor_0.30/seed_0`
- Eval JSON: `projects/CommSpread/outputs/hierarchical_sar_diagnostics/hgsar_indexfix_sensor_0.30_eval.json`
- Eval markdown: `projects/CommSpread/hierarchical_sar_hgsar_indexfix_0.30_eval.md`

## Current Diagnosis

1. The low-level policy and RRT candidate generator are not the main SAR bottleneck.
2. The high-level HGSAR/message-passing policy can learn once the PPO minibatch indexing bug is fixed.
3. Relaxed sensing (`sensor_radius=0.60`) is a useful curriculum stage and now exceeds the `target_first` diagnostic policy.
4. Original sensing (`sensor_radius=0.30`) remains sparse/high-variance. It improves after the fix but is not stable enough in 30 updates.
5. Spread GNN is technically unblocked, but the existing BenchMARL GNN starts much worse than the MLP physics baseline; feature/interface parity with the old `entity-mp` policy should be checked before long 3-seed runs.

## Recommended Next Work

1. For SAR, run a curriculum experiment: train at `sensor_radius=0.60`, then resume/fine-tune at `0.30` using the fixed PPO code.
2. Increase high-level PPO rollout size for `0.30` to reduce sparse target-discovery variance, for example `--num-envs 32` or longer low-level batches.
3. For spread, inspect old `entity-mp` input construction and compare it to BenchMARL `GnnConfig(topology="from_pos")`; do not assume the existing GNN is equivalent to the paper model.
4. After confirming GNN/entity features, rerun `spread_mpe_physics_spawn_parity --model gnn` for one full seed before expanding to three seeds.

## Radius 0.6 SAR Audit Update

A focused audit for `sensor_radius=0.60` is available in `sar_radius_0_6_training_audit.md`. `no_retire_on_rescue` is only a counterfactual diagnostic, not a valid final setting. It shows that retirement consumes search capacity, so the valid solution is to improve search-vs-rescue timing while keeping retirement enabled.

## Staged Rescue Follow-Up

Hard staged rescue masks were implemented and tested at `sensor_radius=0.60` while keeping `retire_on_rescue=True`. They did not outperform the fixed baseline. The best valid run remains the fixed PPO baseline at radius `0.60` with best success `0.5625`. The next direction is soft reward/utility shaping for rescue timing, not disabling retirement and not hard masking target actions.
