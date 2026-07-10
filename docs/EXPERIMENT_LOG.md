# Experiment Log

Generated: 2026-07-10 UTC  
Project: `projects/CommSpread/`  
Git branch: `comm`  
Git HEAD: `3aaf458e00e25049940996e31b906d4964abc811`  
Important caveat: worktree is dirty. For experiments run in this session, record commit as `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`. Older output directories may not record the exact diff state; those entries use `UNKNOWN` where not recoverable from logs.

## E00 - MPE Parity Baseline

- Date: 2026-07-08 based on output names; exact start time UNKNOWN.
- Git commit: UNKNOWN for original run; current report reviewed at `3aaf458e00e25049940996e31b906d4964abc811`.
- Hypothesis: old 10-dimensional MPE-style actor observation and grouped critic should reproduce paper spread behavior after VMAS migration.
- Changed vs prior baseline: MPE-style parity variant `spread_mpe_parity`.
- Config/command: `--variant spread_mpe_parity --model mlp`, outputs in `projects/CommSpread/outputs/baseline_mpe_parity/seed_{0,1,2}`.
- Seeds: 0, 1, 2.
- Checkpoints: BenchMARL output directories under `outputs/baseline_mpe_parity`; exact checkpoint path UNKNOWN in handoff.
- Metrics: success all `0`; episode length `50`; matched distances approx `0.565/0.616/0.567`; episode reward approx `-30.259/-32.088/-30.906`.
- Result: failed baseline.
- Conclusion: simple VMAS MPE parity did not recover paper results.
- Direction excluded: not fully excluded; served as baseline for subsequent information/physics diagnostics.

## E01 - Information Structure A: Global State Critic

- Date: 2026-07-08 based on output names; exact start time UNKNOWN.
- Git commit: UNKNOWN for exact run; current code contains this variant.
- Hypothesis: centralized critic lacks global/relationship information.
- Only changed: critic receives explicit global state; actor remains old 10-dim observation.
- Config: `projects/CommSpread/configs/spread_mpe_global_critic.yaml`; output summary in `information_structure_report.md`.
- Seeds: 0, 1, 2.
- Checkpoint: UNKNOWN.
- Metrics: aggregate success `0`, matched distance mean `0.5653`, episode reward mean `-30.396`.
- Result: marginal reward improvement, no success recovery.
- Conclusion: critic global state alone is not sufficient.
- Direction excluded: critic-only information deficiency mostly excluded.

## E02 - Information Structure B: Actor Other-Agent Relative Positions

- Date: 2026-07-08 based on outputs; exact UNKNOWN.
- Git commit: UNKNOWN for exact run.
- Hypothesis: actor lacks other-agent relative positions.
- Only changed: actor observation includes other-agent relative positions; critic uses richer grouped obs.
- Config: `projects/CommSpread/configs/spread_mpe_actor_rel.yaml`.
- Seeds: 0, 1, 2.
- Metrics: aggregate success `0`, matched distance mean `0.6039`, episode reward mean `-31.393`.
- Result: no success; less stable than baseline in one seed.
- Conclusion: actor relative positions alone are not sufficient.
- Direction excluded: simple actor-observation enrichment mostly excluded.

## E03 - Information Structure C: Richer Actor + Global Critic

- Date: 2026-07-08 based on outputs; exact UNKNOWN.
- Git commit: UNKNOWN for exact run.
- Hypothesis: both actor and critic lack global/relational information.
- Only changed: actor includes other-agent relative positions, critic gets explicit global state.
- Config: `projects/CommSpread/configs/spread_mpe_richer_both.yaml`.
- Seeds: 0, 1, 2.
- Metrics: aggregate success `0.000651`, matched distance mean `0.5615`, reward mean `-29.980`.
- Result: one rare success only; not meaningful recovery.
- Conclusion: information structure helps slightly but does not solve spread.
- Direction excluded: missing simple global/relative info is not the main cause.

## E04 - Action Mapping Probe

- Date: 2026-07-08/09; exact UNKNOWN.
- Git commit: current code includes `scripts/probe_spread_mpe_physics.py`.
- Hypothesis: VMAS action mapping differs from MPE.
- Only changed: deterministic probe, no training.
- Command: `PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/probe_spread_mpe_physics.py --variant spread_mpe_physics_spawn_parity --device cuda:0`.
- Seeds: not applicable.
- Metrics: action ids map as `0 no-op`, `1 left`, `2 right`, `3 down`, `4 up`; repeated right action sequence matches old MPE when physics parity is enabled.
- Result: action direction mapping is correct in physics parity variants.
- Conclusion: action direction mismatch excluded; action scale mismatch remained and was fixed separately.
- Direction excluded: action direction bug excluded.

## E05 - Spread Physics Parity

- Date: 2026-07-09 based on output names.
- Git commit: UNKNOWN exact, current variant present.
- Hypothesis: VMAS physics/action scale mismatch prevents learning.
- Only changed: action force scale and world dynamics parity with old MPE.
- Config: `projects/CommSpread/configs/spread_mpe_physics_parity.yaml`.
- Seeds: 0, 1, 2.
- Checkpoints: outputs in `projects/CommSpread/outputs/physics_parity/seed_*`.
- Metrics: success `0.00195/0/0`, matched distance `0.5659/0.5592/0.5700`, reward approx `-28.828/-28.482/-28.951`.
- Result: reward improved by roughly 2-3 points, success remained rare/zero.
- Conclusion: physics/action scale mismatch was real and necessary to fix, but not sufficient.
- Direction excluded: not excluded; retained as required parity layer.

## E06 - Spread Physics + Independent Spawn Parity

- Date: 2026-07-09 based on output names.
- Git commit: UNKNOWN exact, current variant present.
- Hypothesis: old MPE independent uniform spawn distribution matters.
- Only changed: physics parity plus independent spawn.
- Config: `projects/CommSpread/configs/spread_mpe_physics_spawn_parity.yaml`.
- Seeds: 0, 1, 2.
- Checkpoints: outputs in `projects/CommSpread/outputs/physics_spawn_parity/seed_*`.
- Metrics: success `0/0.00781/0.00586`, matched distance `0.5565/0.5505/0.5534`, reward approx `-28.456/-28.193/-28.330`.
- Result: strongest spread VMAS parity so far, but still very weak.
- Conclusion: retain `spread_mpe_physics_spawn_parity` as current spread baseline.
- Direction excluded: no; this is current baseline.

## E07 - Spread Old-PPO Profile

- Date: 2026-07-09 based on output names.
- Git commit: UNKNOWN exact, current support present.
- Hypothesis: BenchMARL MAPPO objective differs from old PPO enough to cause failure.
- Only changed: old-PPO-like batch/lr/epochs and return reward normalization.
- Config: `projects/CommSpread/configs/spread_mpe_physics_spawn_oldppo.yaml`.
- Seed: 0.
- Checkpoint: output under `projects/CommSpread/outputs/objective_smoke/oldppo/`.
- Metrics: stopped at about 819k frames; success `0`, matched distance `0.5778`, normalized reward not raw-comparable.
- Result: worse early than physics-spawn baseline.
- Conclusion: old-PPO profile should not be expanded before architecture checks.
- Direction excluded: not fully excluded, but deprioritized.

## E08 - Spread GNN Smoke / Early Full Seed

- Date: 2026-07-09.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes` for dependency/code state; exact run diff UNKNOWN.
- Hypothesis: entity/message-passing model is required to recover paper behavior.
- Only changed: `--model gnn` on `spread_mpe_physics_spawn_parity`.
- Config: `projects/CommSpread/configs/spread_mpe_physics_spawn_gnn.yaml`.
- Seed: 0.
- Command smoke-tested:
  `PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/train_mappo.py --variant spread_mpe_physics_spawn_parity --model gnn --comms-radius 1.0 --train-device cuda:0 --sampling-device cuda:0 --seed 0 --max-n-frames 1000 --save-folder projects/CommSpread/outputs/gnn_physics_smoke/seed_0 --no-render`.
- Metrics: smoke passed. A first longer seed was stopped around 74/3000 updates with returns around `-38` to `-43`, worse than MLP physics-spawn around `-28`.
- Result: technical path unblocked, performance initially worse.
- Conclusion: do not spend 3 full seeds until entity features are checked against old `entity-mp`.
- Direction excluded: not excluded; architecture remains high priority.

## E09 - SAR Low-Level Policy Diagnostic

- Date: 2026-07-09 report time; exact run UNKNOWN.
- Git commit: UNKNOWN exact.
- Hypothesis: high-level failures may be caused by low-level policy/RRT failure.
- Only changed: standalone/diagnostic evaluation.
- Checkpoint: `projects/CommSpread/outputs/sar_low_full/low_safe_0.06/checkpoints/checkpoint_50040000.pt`.
- Metrics: standalone goal evaluation completed `793` goals over 128 vector episodes and reached at least one assigned goal in `98.4%` of agent-episodes.
- Result: low-level is strong.
- Conclusion: low-level control is not the main bottleneck.
- Direction excluded: low-level retraining deprioritized.

## E10 - SAR PPO Minibatch Index Fix Baseline, Radius 0.60

- Date: 2026-07-09.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`; exact old broken state UNKNOWN.
- Hypothesis: high-level PPO batch indexing bug corrupts learning signal.
- Only changed: `index_batch()` now indexes all tensors whose first dim equals rollout transition count.
- Command: `train_high_ppo_sar.py --device cuda:0 --seed 0 --num-envs 16 --updates 30 --low-level-steps-per-batch 400 --minibatch-size 64 --sensor-radius 0.6 --save-folder projects/CommSpread/outputs/sar_high_hgsar_indexfix_sensor_0.60/seed_0`.
- Seed: 0.
- Checkpoint: `projects/CommSpread/outputs/sar_high_hgsar_indexfix_sensor_0.60/seed_0/checkpoints/latest.pt`.
- Training metrics from CSV: best success `0.5625`, last5 success `0.300`, final `0.3125`; best visited `2.5`, last5 visited `2.125`; final entropy `1.2514`; final value loss `65.905`.
- Evaluator metrics: deterministic 64-env success `0.34375`, success step `64.27`, visited `2.21875`, detected `2.609375`, detected-unassigned steps `60.859375`.
- Result: major improvement over broken PPO.
- Conclusion: fixed radius-0.6 baseline is the current most reliable SAR learned baseline.
- Direction excluded: no; retained.

## E11 - SAR Radius 0.30 After PPO Fix

- Date: 2026-07-09.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`.
- Hypothesis: original/stricter sensing can work after PPO fix.
- Only changed: `--sensor-radius 0.3`.
- Output: `projects/CommSpread/outputs/sar_high_hgsar_indexfix_sensor_0.30/seed_0`.
- Seed: 0.
- Metrics: best success `0.125`, last5 `0.05`, final `0.125`; final visited `1.5625`, final detected `1.5625`.
- Result: sparse/high variance, much worse than radius 0.6.
- Conclusion: keep radius 0.6 as debug/curriculum regime before returning to 0.3.
- Direction excluded: not excluded; postponed.

## E12 - SAR No-Retire Counterfactual

- Date: 2026-07-09.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`.
- Hypothesis: retirement consumes search capacity.
- Only changed: evaluator/training diagnostic with `retire_on_rescue=False`.
- Output: `projects/CommSpread/outputs/sar_high_hgsar_indexfix_sensor_0.60_no_retire/seed_0` and `failure_hgsar_0.60_no_retire.json`.
- Metrics from report: retire=True success `0.34375`; retire=False eval success `0.671875`, visited `2.59375`.
- Result: success rises strongly without retirement.
- Conclusion: retirement is a bottleneck, but no-retire violates task semantics and must not be used as final solution.
- Direction excluded: no-retire final design excluded; diagnostic retained.

## E13 - SAR Hard Staged Rescue Masks

- Date: 2026-07-09.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`.
- Hypothesis: delaying rescue until enough targets are detected improves search/rescue timing.
- Only changed: `--staged-rescue` target action mask with detected thresholds.
- Outputs: `sar_high_hgsar_staged_t2_sensor_0.60`, `sar_high_hgsar_staged_t3_sensor_0.60`.
- Seeds: seed 0 only.
- Metrics: threshold 2, 21 updates, best success `0.375`, last5 `0.2375`; threshold 3, 7 updates, best `0.25`, last5 `0.15`.
- Result: no improvement over fixed baseline.
- Conclusion: hard target masking is too crude.
- Direction excluded: hard staged masking deprioritized/excluded for now.

## E14 - SAR Soft Stage T2 Shaping

- Date: 2026-07-09/10.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`.
- Hypothesis: soft early-rescue/search-capacity shaping is better than hard mask.
- Only changed: progress features, early rescue penalty, search capacity discovery bonus, all-targets-detected bonus.
- Command included: `--progress-features --early-rescue-penalty 1.0 --early-rescue-detected-threshold 2 --search-capacity-discovery-bonus 0.5 --all-targets-detected-bonus 2.0`.
- Output: `projects/CommSpread/outputs/sar_high_hgsar_soft_stage_t2_sensor_0.60/seed_0`.
- Seed: 0.
- Metrics: best success `0.5625`, last5 `0.2375`, final `0.1875`; deterministic success `0.1875`; detected-unvisited failure worsened.
- Result: matched stochastic peak but worse stability/eval.
- Conclusion: search-pressure-only shaping insufficient.
- Direction excluded: this specific shaping excluded.

## E15 - SAR Soft Rescue-Phase Shaping

- Date: 2026-07-09/10.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`.
- Hypothesis: penalizing exploration when known targets need rescuers improves rescue phase.
- Only changed: `--progress-features --early-rescue-penalty 0.5 --all-targets-detected-bonus 1.0 --rescue-phase-explore-penalty 0.05 --rescue-phase-detected-threshold 2 --rescue-phase-min-search-agents 1`.
- Output: `projects/CommSpread/outputs/sar_high_hgsar_soft_rescue_phase_sensor_0.60/seed_0`.
- Seed: 0.
- Metrics: train best `0.4375`, last5 `0.2625`, final `0.4375`; deterministic eval success `0.25`, detected-unassigned steps `71.078125`, duplicate excess `0.203125`.
- Result: reduced duplicate assignment but increased detected-unassigned time.
- Conclusion: did not solve bottleneck.
- Direction excluded: specific rescue-phase penalty excluded.

## E16 - SAR LR Diagnostic For Soft Rescue Phase

- Date: 2026-07-10.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`.
- Hypothesis: PPO learning rate is too small because KL/clip approach zero.
- Only changed: `--lr 0.001` and 20 updates.
- Output: `projects/CommSpread/outputs/sar_high_hgsar_soft_rescue_phase_lr1e3_sensor_0.60/seed_0`.
- Seed: 0.
- Metrics: last5 success `0.2625`, best `0.375`, final `0.1875`; mean KL `0.00123`, max KL `0.005919`, mean clip `0.015675`.
- Result: larger lr made early updates more active but did not improve success.
- Conclusion: LR alone is not primary bottleneck.
- Direction excluded: pure LR increase excluded.

## E17 - SAR Target Assignment Features

- Date: 2026-07-10.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`.
- Hypothesis: actor needs target load/age/distance features to assign detected targets.
- Only changed: `--progress-features --target-assignment-features`; target nodes add claim count, nearest active-agent distance, detected age.
- Output: `projects/CommSpread/outputs/sar_high_hgsar_target_assign_features_sensor_0.60/seed_0`.
- Seed: 0.
- Metrics: train best `0.375`, last5 `0.2`, final `0.125`; eval success `0.296875`, success step `59`, detected-unassigned `65.921875`, duplicate excess `2.703125`.
- Result: assignment delay improved but duplicate claims increased and success stayed below fixed baseline.
- Conclusion: features alone not enough.
- Direction excluded: not fully excluded; keep optional feature but do not expect it alone to solve.

## E18 - SAR Heuristic Baselines: Greedy Rescue And Search-All-Then-Rescue

- Date: 2026-07-10.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`.
- Hypothesis: heuristic bounds clarify timing tradeoff for shaping.
- Only changed: evaluator heuristic policy; no training.
- Commands: `analyze_sar_failure_modes.py --policy rescue_greedy` and `--policy search_all_then_rescue` with `--sensor-radius 0.6 --num-envs 64 --steps 300`.
- Seeds: 0, 1, 2.
- Outputs: `baseline_rescue_greedy_seed{0,1,2}_0.60.json`; `baseline_search_all_then_rescue_seed{0,1,2}_0.60.json`.
- Metrics:
  - Greedy rescue success `0.224 +/- 0.05156`, success step `66.89 +/- 3.083`, detected `2.542`, visited `2.047`.
  - Search-all-then-rescue success `0.2031 +/- 0.05846`, success step `70.52 +/- 2.934`, detected `2.714`, visited `1.474`, detected-unassigned steps `136`.
- Result: greedy is faster but lower success; search-all detects more but waits too long.
- Conclusion: optimal policy must sit between the heuristics.
- Direction excluded: both pure heuristic strategies are insufficient as final policy.

## E19 - SAR Assignment Bonus/Penalty Shaping

- Date: 2026-07-10.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`.
- Hypothesis: reward unique rescue assignments and penalize duplicate/unassigned targets to reduce assignment latency.
- Only changed: `--progress-features --target-assignment-features --unique-rescue-assignment-bonus 0.2 --duplicate-rescue-assignment-penalty 0.1 --detected-unassigned-target-penalty 0.02`.
- Output: `projects/CommSpread/outputs/sar_high_hgsar_assignment_shaping_sensor_0.60/seed_0`.
- Seed: 0.
- Metrics: train best `0.375`, last5 `0.175`, eval success `0.140625`, success step `56.22`, detected-unassigned `73.375`, duplicate excess `6.5625`, discovery-to-assignment delay `15.15`.
- Result: reduced assignment delay but created duplicate claims and degraded success.
- Conclusion: positive assignment bonus causes duplicate pressure.
- Direction excluded: this bonus-heavy shaping excluded.

## E20 - SAR Latency-Only Duplicate-Strong Shaping

- Date: 2026-07-10.
- Git commit: `3aaf458e00e25049940996e31b906d4964abc811 + uncommitted changes`.
- Hypothesis: avoid positive rescue bonus; use small detected-unassigned penalty and stronger duplicate penalty.
- Only changed: `--progress-features --target-assignment-features --duplicate-rescue-assignment-penalty 0.5 --detected-unassigned-target-penalty 0.005`.
- Output: `projects/CommSpread/outputs/sar_high_hgsar_assignment_latency_only_sensor_0.60/seed_0`.
- Seed: 0.
- Metrics: 20 updates; train best `0.5`, last5 `0.1625`, eval success `0.21875`, success step `71.64`, detected-unassigned `75.8125`, duplicate excess `3.9375`.
- Result: stochastic peak appears, but deterministic eval still poor and duplicate excess remains above fixed baseline.
- Conclusion: reward shaping needs coordinated target selection/masking to prevent duplicate claims.
- Direction excluded: latency-only reward by itself excluded; combined with coordinated target selection remains open.
