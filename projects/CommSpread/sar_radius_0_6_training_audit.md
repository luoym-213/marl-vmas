# SAR Radius 0.6 Training Audit

## Scope

This audit keeps the formal SAR task semantics intact: when an agent commits to rescue and reaches the rescue point, it retires. `--no-retire-on-rescue` was used only as a counterfactual diagnostic and is not a candidate final setting.

## Training Flow Issues Found

### 1. PPO minibatch indexing bug

Fixed in `scripts/train_high_ppo_sar.py`.

`index_batch()` previously failed to apply minibatch indices consistently to tensors whose first dimension matched the full transition batch. This could mismatch observations/actions/log-probs with permuted advantages. After fixing it, `sensor_radius=0.60` training improved strongly:

| run | last5 success | best success | best visited |
|---|---:|---:|---:|
| before fix | 0.025 | 0.0625 | 1.1875 |
| after fix | 0.300 | 0.5625 | 2.5 |

Deterministic eval of the fixed final checkpoint over 32 envs: success `0.28125`, visited `2.15625`.

### 2. SAR max_steps propagation bug

Fixed in `comm_spread/env_factory.py` and `comm_spread/sar_scenario.py`.

`make_sar_env(max_steps=150)` updated the scenario config but not the VMAS wrapper horizon, so the environment still truncated at 100 steps. The factory now sends the same horizon to VMAS and to `SarScenario` via `scenario_max_steps`.

## Failure Mode Analysis at Radius 0.6

Using the fixed HGSAR checkpoint with normal `retire_on_rescue=True`, 64-env deterministic analysis produced:

| metric | value |
|---|---:|
| success_rate | 0.34375 |
| visited_targets_mean | 2.21875 |
| detected_targets_mean | 2.609375 |
| retired_agents_mean | 2.21875 |
| detected_unvisited_remaining_rate | 0.328125 |
| undiscovered_remaining_rate | 0.328125 |

Among partial failures, the policy usually rescues about `1.85` targets and retires about `1.85` agents. Failures split almost evenly between targets not discovered and targets discovered but not rescued.

## Time/Speed Limit Test

After fixing max-step propagation, `max_steps=150` gave:

| max_steps | success_rate | visited_targets_mean |
|---:|---:|---:|
| 100 | 0.34375 | 2.21875 |
| 150 | 0.37500 | 2.34375 |

The improvement is small. Agent speed or the 100-step horizon is not the primary limit.

## Retirement Counterfactual

With the same trained checkpoint but `retire_on_rescue=False` during evaluation:

| setting | success_rate | visited_targets_mean |
|---|---:|---:|
| retire=True | 0.34375 | 2.21875 |
| retire=False | 0.671875 | 2.59375 |

This is not a valid final task setting. It diagnoses that retirement consumes search capacity; the valid fix is better search-vs-rescue timing while keeping retirement enabled.

## Delayed Rescue Rule Test

Naive delayed rescue rules failed because the current action interface masks out visited or claimed targets, so a policy using only selectable target count cannot reliably infer global progress.

| rule | success_rate |
|---|---:|
| rescue after 2 currently selectable targets | 0.0 |
| rescue after 3 currently selectable targets | 0.0 |

## Actor Information Issue

The actor does not directly receive global progress. I added optional `--progress-features` with normalized detected count, visited count, active agent count, and global entropy. In a short run it reached best success `0.5`, but remained unstable and did not clearly beat the fixed baseline best `0.5625`.

## Parameter Experiments

A larger batch run (`num_envs=32`, `minibatch_size=128`) reached only about `0.34` by update 11 and did not clearly improve stability. Larger batch alone is not enough.

## Current Conclusion

The best valid direction is still `sensor_radius=0.60`, `retire_on_rescue=True`, fixed PPO indexing. Remaining failures are mainly search-rescue timing under retirement, not low-level control, RRT geometry, or episode length.

The policy needs staged behavior:

1. keep enough agents searching until discovery coverage is high;
2. avoid sending too many agents into retirement too early;
3. rescue detected targets before timeout once enough targets are known.

## Recommended Next Experiments

1. Keep `retire_on_rescue=True`.
2. Add staged rescue action masking based on global detected/visited progress, not just selectable target count.
3. Add reward shaping only if staged action masking is insufficient, e.g. penalty for rescue before enough targets are detected or bonus for detecting all targets before first rescue.
4. Use conservative PPO updates after adding staged masking, likely `lr=1e-4` or fewer epochs if KL/clip spikes.

## Staged Rescue Experiments

I implemented optional staged rescue controls without changing task semantics:

- `--staged-rescue`
- `--rescue-detected-threshold`
- `--rescue-entropy-threshold`
- `--progress-features`

These only change the high-level action mask or actor inputs; `retire_on_rescue=True` remains active.

Short-run comparison at `sensor_radius=0.60`:

| run | updates | best success | best visited | last5 success | last5 visited |
|---|---:|---:|---:|---:|---:|
| fixed baseline | 30 | 0.5625 | 2.5 | 0.3000 | 2.125 |
| progress features | 30 | 0.5000 | 2.3125 | 0.2500 | 1.9625 |
| staged threshold 2 + progress | 21 | 0.3750 | 2.1875 | 0.2375 | 1.975 |
| staged threshold 3 + progress | 7 | 0.2500 | 1.625 | 0.1500 | 1.125 |
| larger batch 32 env | 11 | 0.34375 | 2.3125 | 0.2000 | 2.05625 |

Conclusion: hard staged rescue masking is valid to run but did not improve performance. Threshold 3 delays rescue too much. Threshold 2 still allows early retirement and does not outperform the baseline. Larger batch alone did not solve the issue.

The next better direction is a soft objective rather than hard action masking:

1. Keep target actions available.
2. Add a small penalty for rescue/retirement before enough targets are detected, e.g. before `detected_count >= 2`.
3. Add a bonus for detecting all targets before the first rescue, or for discovering a new target while at least two agents remain active.
4. Keep progress features enabled so the actor and critic can represent the phase.
5. Use conservative PPO settings if KL/clip spikes, e.g. `lr=1e-4` or `ppo_epochs=2`.

## Soft Staged Rescue Follow-up

I tested two soft staged-rescue mechanisms while preserving the formal task semantics: `retire_on_rescue=True` remained enabled, and rescue actions were never disabled.

### Mechanism 1: Early rescue penalty + search capacity discovery bonus

Config summary:

- `--progress-features`
- `--early-rescue-penalty 1.0`
- `--early-rescue-detected-threshold 2`
- `--search-capacity-discovery-bonus 0.5`
- `--all-targets-detected-bonus 2.0`

Seed 0, 30 updates:

| run | last5 success | best success | final success | last5 visited | best visited | deterministic success |
|---|---:|---:|---:|---:|---:|---:|
| fixed baseline | 0.3000 | 0.5625 | 0.3125 | 2.1250 | 2.5000 | 0.34375 |
| soft stage t2 | 0.2375 | 0.5625 | 0.1875 | 2.0625 | 2.1875 | 0.1875 |

Failure analysis of the soft-stage checkpoint showed `detected_unvisited_remaining_rate = 0.484375`, worse than the fixed baseline. The shaping increased search pressure but did not produce a reliable rescue phase, so more episodes ended with known targets still unvisited.

### Mechanism 2: Rescue-phase explore penalty

I then added a second-stage soft penalty for choosing exploration while there are known unvisited targets and too few agents assigned to rescue, while still reserving one active search agent.

Config summary:

- `--progress-features`
- `--early-rescue-penalty 0.5`
- `--early-rescue-detected-threshold 2`
- `--all-targets-detected-bonus 1.0`
- `--rescue-phase-explore-penalty 0.05`
- `--rescue-phase-detected-threshold 2`
- `--rescue-phase-min-search-agents 1`

Seed 0, 30 updates:

| run | last5 success | best success | final success | last5 visited | best visited | deterministic success |
|---|---:|---:|---:|---:|---:|---:|
| fixed baseline | 0.3000 | 0.5625 | 0.3125 | 2.1250 | 2.5000 | 0.34375 |
| soft rescue phase | 0.2625 | 0.4375 | 0.4375 | 2.0875 | 2.3750 | 0.2500 |

Deterministic failure analysis still showed high known-but-unrescued failures:

| metric | value |
|---|---:|
| success_rate | 0.25 |
| visited_targets_mean | 2.125 |
| detected_targets_mean | 2.625 |
| detected_unvisited_remaining_rate | 0.46875 |
| undiscovered_remaining_rate | 0.28125 |

The rescue-phase penalty improved the final training update but did not improve deterministic success or failure composition relative to the fixed baseline.

### Learning-rate diagnostic

Because the staged runs had near-zero PPO KL and clip fraction, I ran a short lr diagnostic with `--lr 0.001` using the second-stage config.

Seed 0, 20 updates:

| run | last5 success | best success | final success | mean KL | max KL | mean clip |
|---|---:|---:|---:|---:|---:|---:|
| soft rescue phase lr=3e-4 | 0.2625 | 0.4375 | 0.4375 | 0.000240 | 0.000767 | 0.000122 |
| soft rescue phase lr=1e-3 | 0.2625 | 0.3750 | 0.1875 | 0.001230 | 0.005919 | 0.015675 |

`lr=1e-3` made early PPO updates more effective but did not improve success. It also decayed back to very low KL later. This suggests the main bottleneck is not simply an overly small learning rate.

## Current Diagnosis After Stage Rescue Tests

The current performance limit is not caused by low-level control, RRT feasibility, max episode length, hard action masking bugs, or catastrophic PPO instability. The remaining issue is the high-level objective and credit assignment around the search-to-rescue transition under retirement.

Important observations:

1. Once agents retire, the team often lacks enough search/rescue capacity to finish.
2. Purely delaying rescue increases known-but-unrescued failures.
3. Soft penalties that encourage rescue after discovery still do not solve assignment timing.
4. PPO updates are numerically stable; KL/clip are often very small after the policy becomes moderately deterministic.
5. Deterministic success is consistently worse than the best stochastic training updates, so the learned policy has high seed/episode variance and weak phase robustness.

## Next Reasonable Attempts

The next change should not be another scalar reward tweak. The evidence points to missing phase/assignment structure in the high-level decision process.

Recommended next experiments:

1. Add explicit high-level action decomposition: a binary mode or head for `search` vs `rescue`, then choose candidate within that mode. This lets the policy learn phase allocation separately from spatial candidate choice.
2. Add a target assignment/load feature: for each detected target, expose how many active agents are already assigned to it and estimated travel distance for each active agent.
3. Add evaluation diagnostics for duplicate rescue assignment and rescue latency: count how often multiple agents chase the same target, how long each detected target waits before first assignment, and how often agents switch away from a detected target before arrival.
4. Train with the fixed baseline reward first, then optionally add small rescue-latency shaping after duplicate/latency metrics identify the failure.
5. If keeping PPO, increase effective update moderately with `lr=5e-4` or larger rollout batches, but do not rely on LR alone; `lr=1e-3` did not solve the bottleneck.

## Assignment Diagnostics

I added explicit high-level assignment diagnostics to `scripts/analyze_sar_failure_modes.py`:

- `detected_unassigned_steps`: how many target-steps had a detected unvisited target but no active agent assigned to rescue it;
- `discovery_to_assignment_delay`: delay from first detection to first rescue assignment;
- `assignment_to_visit_delay`: delay from first assignment to rescue completion;
- `duplicate_assignment_excess`: redundant simultaneous assignments to the same target;
- `switch_away_actions`: assignments changed away from an unfinished rescue target.

For the fixed baseline checkpoint at `sensor_radius=0.60`, deterministic 64-env diagnostics show:

| metric | value |
|---|---:|
| success_rate | 0.34375 |
| detected_unassigned_steps_mean | 60.859375 |
| discovery_to_assignment_delay_mean | 18.726563 |
| assignment_to_visit_delay_mean | 18.216931 |
| duplicate_assignment_excess_mean | 1.796875 |
| switch_away_actions_mean | 0.0 |

For the soft rescue-phase checkpoint:

| metric | value |
|---|---:|
| success_rate | 0.25 |
| detected_unassigned_steps_mean | 71.078125 |
| discovery_to_assignment_delay_mean | 17.580729 |
| assignment_to_visit_delay_mean | 19.376344 |
| duplicate_assignment_excess_mean | 0.203125 |
| switch_away_actions_mean | 0.0 |

This rules out the earlier target-switching hypothesis for the current learned policies: agents are not frequently abandoning unfinished rescue assignments. Duplicate rescue assignment also is not the dominant issue. The dominant issue is that detected targets remain unassigned for many environment steps, and assignment-to-visit latency is also long relative to the 100-step horizon.

The soft rescue-phase reward reduced duplicate assignment but increased detected-unassigned time, so it worsened the actual bottleneck.

## Revised Next Step

The next implementation should target action structure and target urgency directly:

1. Add target urgency/load features to actor and critic: per target claim count, nearest active-agent distance, and normalized detected age.
2. Add a rescue-priority action head or two-stage actor: first choose mode (`search` or `rescue`), then choose candidate within that mode. A flat action over RRT nodes and targets appears to under-select rescue after detection.
3. Add a small rescue-latency reward/penalty tied to detected-unassigned target-steps, not generic early/late rescue shaping.
4. Keep `retire_on_rescue=True`; do not use no-retire except as diagnostic.

## Target Assignment Feature Experiment

I added optional target assignment features without changing default behavior:

- `target_assignment_features` in `SarScenario`;
- `target_detected_step` to track target age after first detection;
- `--target-assignment-features` in train/analyze scripts;
- actor `target_features` dimension parameterization.

When enabled, each target node adds:

1. normalized current claim count;
2. nearest active-agent distance;
3. normalized detected age.

A 30-update seed-0 run with `--progress-features --target-assignment-features` produced:

| run | last5 success | best success | final success | deterministic success |
|---|---:|---:|---:|---:|
| fixed baseline | 0.3000 | 0.5625 | 0.3125 | 0.34375 |
| target assignment features | 0.2000 | 0.3750 | 0.1250 | 0.296875 |

Assignment diagnostics:

| metric | fixed baseline | target assignment features |
|---|---:|---:|
| detected_unassigned_steps_mean | 60.859375 | 65.921875 |
| discovery_to_assignment_delay_mean | 18.726563 | 16.395834 |
| assignment_to_visit_delay_mean | 18.216931 | 18.333333 |
| duplicate_assignment_excess_mean | 1.796875 | 2.703125 |
| switch_away_actions_mean | 0.0 | 0.0 |

Interpretation: target assignment features help the policy assign detected targets slightly sooner, but they do not solve the main unassigned-target bottleneck and they increase duplicate rescue assignments. This reinforces that the flat action head is a poor fit: the policy needs explicit team-level rescue allocation or a mode/assignment decomposition, not only richer per-target features.

## Updated Recommendation

The next high-value implementation is a two-stage high-level actor:

1. mode head: choose `search` vs `rescue`;
2. candidate head: choose an RRT node if searching, or choose a target if rescuing;
3. optional rescue allocation regularizer: penalize detected-unassigned target-steps and duplicate claims directly.

This should be tested without extra scalar shaping first, using the fixed reward and `retire_on_rescue=True`.

