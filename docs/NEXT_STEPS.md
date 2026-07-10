# Next Steps

Generated: 2026-07-10 UTC  
Git HEAD: `3aaf458e00e25049940996e31b906d4964abc811`  
Worktree: dirty. Do not start broad multi-seed runs until the current state is committed or archived.

## Priority 0 - Reproducibility Checkpoint

Hypothesis: current results are difficult to reproduce unless code and docs are committed together.

Action:

- Commit code and documentation, or create a patch archive and note untracked output directories.
- Do not change algorithm behavior in this step.

Success criterion:

- `git status --short` after commit shows either clean tree or only intentionally ignored large outputs.

Failure criterion:

- Important reports/outputs remain untracked with no archival policy.

Expected files:

- Git metadata only, no code changes.

Do not change simultaneously:

- Training hyperparameters, rewards, action masks, model architecture.

## Priority 1 - Coordinated Target Selection For SAR

Hypothesis:

- Reward shaping fails because simultaneous agents can still select the same target. Per-env sequential target masking during action sampling will reduce duplicate claims and make latency shaping meaningful.

Experiment:

- Keep flat policy logits.
- During high-level action sampling/evaluation, for each env decision batch, sequentially prevent later agents from selecting target actions already selected by earlier active agents.
- Start without positive unique-assignment bonus.
- Use `--detected-unassigned-target-penalty 0.0025` or `0.005` only after the mask smoke passes.

Success criteria:

- `duplicate_assignment_excess_mean` below fixed baseline `1.797`.
- `detected_unassigned_steps_mean` below fixed baseline `60.86`.
- Deterministic eval success >= fixed baseline `0.34375` or train last5 success > `0.300` without worse deterministic diagnostics.

Failure criteria:

- Duplicate claims remain above baseline.
- Detected-unassigned steps remain above baseline.
- Success falls below greedy heuristic range (`~0.224`) after 30 updates.

Expected files:

- `comm_spread/high_level_policy.py` or `comm_spread/async_smdp.py` depending on where sampling mask is implemented.
- `scripts/analyze_sar_failure_modes.py` if evaluator needs identical coordinated selection behavior.
- Possibly `scripts/train_high_ppo_sar.py` for CLI flags.

Do not change simultaneously:

- Sensor radius.
- Low-level checkpoint.
- Retirement semantics.
- Actor architecture.
- Multiple reward shaping terms beyond one small latency penalty.

## Priority 2 - Two-Stage SAR Actor Prototype

Hypothesis:

- A flat action space over RRT candidates and targets under-selects or misallocates rescue. Separating `search` vs `rescue` mode from candidate choice improves phase allocation.

Experiment:

- Add a mode head and candidate head.
- Compare first without extra shaping to fixed HGSAR baseline.
- Keep target assignment features optional or enabled only in a controlled variant.

Success criteria:

- Deterministic eval success > `0.34375`.
- `detected_unassigned_steps_mean` decreases without duplicate excess increase.
- Entropy does not collapse earlier than fixed baseline.

Failure criteria:

- Mode policy collapses to always search or always rescue.
- Success below heuristic baselines.
- PPO KL/clip become unstable or zero from the start.

Expected files:

- `comm_spread/models/hgsar.py`
- `comm_spread/high_level_policy.py`
- `scripts/train_high_ppo_sar.py`
- `scripts/analyze_sar_failure_modes.py`

Do not change simultaneously:

- Reward shaping.
- Radius.
- Low-level policy.
- Batch size/lr.

## Priority 3 - SAR Radius Curriculum

Hypothesis:

- Training at radius 0.6 learns useful high-level rescue behavior that can transfer to radius 0.3 better than starting at 0.3 from scratch.

Experiment:

- Start from best valid radius 0.6 checkpoint.
- Resume/fine-tune with `--sensor-radius 0.3`.
- Compare to radius 0.3 from-scratch fixed PPO baseline.

Success criteria:

- Radius 0.3 eval success > current from-scratch eval `0.03125` and train best > `0.125`.
- Detected targets and visited targets improve together.

Failure criteria:

- Fine-tuning collapses target actions or reduces detection.
- Improvements are only stochastic training spikes with worse deterministic eval.

Expected files:

- Probably no code changes; only commands and outputs.

Do not change simultaneously:

- Reward shaping.
- Actor features.
- Low-level checkpoint.

## Priority 4 - Spread Entity/Message-Passing Parity Audit

Hypothesis:

- Existing BenchMARL GNN is not equivalent to old paper `entity-mp`; feature construction/topology mismatch explains poor GNN early returns.

Experiment:

- Inspect old `entity-mp` model inputs and compare to current BenchMARL GNN entity features.
- Build a parity checklist before running long GNN training.
- If mismatch is found, implement closer old MPNN feature path or adapter.

Success criteria:

- Documented equivalence or concrete mismatch list.
- One full seed of corrected GNN beats MLP physics-spawn baseline reward/matched distance.

Failure criteria:

- Existing GNN remains around `-38` to `-43` returns while MLP physics-spawn is around `-28`.

Expected files:

- `comm_spread/models/` or BenchMARL model config path, depending on audit.
- `scripts/train_mappo.py` only if a new model option is needed.
- New report under `projects/CommSpread/` or `docs/`.

Do not change simultaneously:

- Spread physics config.
- PPO profile.
- Reward normalization.

## Priority 5 - Spread Full GNN Seed Only After Audit

Hypothesis:

- If entity features are correct, message passing may recover spread performance better than MLP.

Experiment:

```bash
PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/train_mappo.py   --variant spread_mpe_physics_spawn_parity   --model gnn   --comms-radius 1.0   --train-device cuda:0   --sampling-device cuda:0   --seed 0   --max-n-frames 3000000   --save-folder projects/CommSpread/outputs/physics_spawn_gnn/seed_0   --no-render
```

Success criteria:

- Matched distance below `0.55` and success materially above current rare-success baseline.
- Reward not stuck around `-38` to `-43` early.

Failure criteria:

- Early return remains much worse than MLP physics-spawn baseline.

Expected files:

- No code changes if audit passes.

Do not change simultaneously:

- Physics variant.
- PPO profile.
- Global critic variant.
