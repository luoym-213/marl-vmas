# Physics and Objective Diagnosis Report

## Summary

This stage tested the strongest post-information-structure suspects: VMAS/MPE physics mismatch, reset distribution mismatch, and old-PPO objective mismatch.

The main finding is that **physics/action scale mismatch is real and large**. Old MPE applies discrete actions with `sensitivity=5.0`; the previous VMAS parity used force scale `1.0`. A fixed-action probe showed the old VMAS parity moved only `0.247` after 10 repeated right actions, while old MPE dynamics move `1.434`. The new physics parity variant exactly matches the old MPE displacement sequence.

Physics parity improves training reward substantially, but it still does not restore reliable success. Direct old-PPO objective parity with return reward normalization is worse at the tested checkpoint.

## Implemented Variants

| variant | change | yaml |
|---|---|---|
| `spread_mpe_physics_parity` | MPE action force scale and world dynamics parity | `configs/spread_mpe_physics_parity.yaml` |
| `spread_mpe_physics_spawn_parity` | physics parity + old independent uniform reset | `configs/spread_mpe_physics_spawn_parity.yaml` |
| `spread_mpe_physics_spawn_oldppo` | P2 + old PPO-like batch/lr/epochs + return reward normalization | `configs/spread_mpe_physics_spawn_oldppo.yaml` |

Probe script: `scripts/probe_spread_mpe_physics.py`.

## Probe Results

For `spread_mpe_physics_parity` and `spread_mpe_physics_spawn_parity`:

| action id | observed delta | meaning |
|---:|---:|---|
| 0 | `(0.000, 0.000)` | no-op |
| 1 | `(-0.050, 0.000)` | left |
| 2 | `(0.050, 0.000)` | right |
| 3 | `(0.000, -0.050)` | down |
| 4 | `(0.000, 0.050)` | up |

Repeated right action for 10 steps exactly matches old MPE: `0.05, 0.1375, 0.2531, 0.3898, 0.5424, 0.7068, 0.8801, 1.0601, 1.2451, 1.4338`.

## Training Results

All training below uses 3M frames unless noted.

| experiment | seed | success_rate | episode_len | matched_dist | episode_reward | notes |
|---|---:|---:|---:|---:|---:|---|
| baseline parity | 0 | 0 | 50.00 | 0.565 | -30.259 | previous baseline |
| baseline parity | 1 | 0 | 50.00 | 0.616 | -32.088 | previous baseline |
| baseline parity | 2 | 0 | 50.00 | 0.567 | -30.906 | previous baseline |
| P physics parity | 0 | 0.00195 | 49.99 | 0.5659 | -28.828 | reward improves, success rare |
| P physics parity | 1 | 0 | 50.00 | 0.5592 | -28.482 | reward improves |
| P physics parity | 2 | 0 | 50.00 | 0.5700 | -28.951 | reward improves |
| P2 physics+spawn parity | 0 | 0 | 50.00 | 0.5565 | -28.456 | best matched vs baseline but no success |
| P2 physics+spawn parity | 1 | 0.00781 | 49.75 | 0.5505 | -28.193 | strongest run, still weak |
| P2 physics+spawn parity | 2 | 0.00586 | 49.84 | 0.5534 | -28.330 | weak success signal |
| O old-PPO profile | 0 | 0 | 50.00 | 0.5778 | n/a | stopped at 819k; reward normalized, not raw-comparable |

## Diagnosis

1. **Physics/action scale was definitely wrong.** The old VMAS parity dynamics were far slower than MPE. This explains why prior runs plateaued: agents could not traverse the world with the same effective control authority as the paper environment.
2. **Physics parity is necessary but not sufficient.** P/P2 improve episode reward by roughly 2-3 points and P2 gives rare successes, but success_rate is still below 1%.
3. **Independent spawn helps slightly.** P2 improves matched distance to about `0.55`, better than P and baseline, but still far from robust coverage.
4. **Old PPO profile is not the next best direction.** The old-PPO/reward-normalized run was worse at 819k frames and showed no early success signal. It should not be expanded before checking architecture.
5. **The next strongest suspect is network structure.** The old paper code recommends `--entity-mp`; current spread experiments use plain MLP. The current new base should remain `spread_mpe_physics_spawn_parity`, with an entity/message-passing policy as the next spread experiment. The existing BenchMARL `--model gnn` path is blocked until `torch_geometric` is installed.


## SAR Hierarchical Diagnostics

The SAR side was also checked because it separates high-level assignment quality from low-level execution. The low-level checkpoint `projects/CommSpread/outputs/sar_low_full/low_safe_0.06/checkpoints/checkpoint_50040000.pt` is strong: standalone goal evaluation completed `793` goals over 128 vector episodes and reached at least one assigned goal in `98.4%` of agent-episodes.

A new diagnostic script, `scripts/diagnose_hierarchical_sar.py`, evaluates high-level policy choice, RRT candidate quality, low-level goal hits, and relaxed sensor radii. Results are in `hierarchical_sar_diagnostics.md`. The key finding is that low-level tracking and RRT geometry are not the main bottlenecks. `greedy_explore` reaches assigned RRT points but never rescues targets. `target_first` rescues targets and improves with relaxed sensing: success rises from `0.0625` at `sensor_radius=0.30` to `0.25` at `0.60`.

This points to high-level target-switching/assignment as the most useful SAR objective to train. The local HGSAR message-passing actor is available without `torch_geometric`; a 1-update CUDA smoke with `--sensor-radius 0.6` ran successfully.

## Recommended Next Attempts

1. Keep `spread_mpe_physics_spawn_parity` as the new base for further work.
2. Run `spread_mpe_physics_spawn_parity` with an entity/message-passing policy. The existing `--model gnn` path is currently blocked because `torch_geometric` is not installed in this environment.
3. If installing `torch_geometric` is acceptable, smoke-test `--model gnn` first, then run 3 seeds with a 3M-frame plateau check.
4. If installing that dependency is not acceptable or GNN still fails, implement a closer old MPNN low-level actor/critic rather than spending more time on PPO hyperparameters.
5. Preserve the physics probe as a regression test; any future variant claiming MPE parity should pass it.
