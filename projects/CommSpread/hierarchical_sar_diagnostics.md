# Hierarchical SAR Diagnostics

This report evaluates high-level assignment quality, RRT candidate quality, low-level goal reaching, and relaxed sensor radii without training new code.

| policy | sensor | success | visited | detected | goal_hits/tr | step_hit | goal_dist | entropy_gain | rrt_util | rrt_zero | duration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy_explore | 0.30 | 0.000 | 0.000 | 1.500 | 0.845 | 0.0500 | 0.263 | 3484.4 | 0.804 | 0.000 | 16.9 |
| target_first | 0.30 | 0.062 | 1.250 | 1.375 | 0.878 | 0.0462 | 0.256 | 2691.7 | 0.804 | 0.000 | 19.0 |
| hgsar_untrained | 0.30 | 0.000 | 0.875 | 1.250 | 0.885 | 0.0591 | 0.243 | 2842.9 | 0.804 | 0.000 | 15.0 |
| greedy_explore | 0.45 | 0.000 | 0.000 | 2.250 | 0.822 | 0.0444 | 0.297 | 5152.0 | 0.804 | 0.000 | 18.5 |
| target_first | 0.45 | 0.188 | 2.000 | 2.250 | 0.895 | 0.0389 | 0.285 | 3725.4 | 0.804 | 0.000 | 23.0 |
| hgsar_untrained | 0.45 | 0.000 | 1.375 | 2.000 | 0.882 | 0.0489 | 0.267 | 3973.9 | 0.804 | 0.000 | 18.0 |
| greedy_explore | 0.60 | 0.000 | 0.000 | 2.625 | 0.811 | 0.0429 | 0.293 | 6554.2 | 0.804 | 0.000 | 18.9 |
| target_first | 0.60 | 0.250 | 2.062 | 2.688 | 0.886 | 0.0382 | 0.302 | 5104.3 | 0.804 | 0.000 | 23.2 |
| hgsar_untrained | 0.60 | 0.062 | 1.375 | 2.438 | 0.885 | 0.0492 | 0.266 | 5325.2 | 0.804 | 0.000 | 18.0 |

## Interpretation

- `goal_hits/tr` and `step_hit` measure whether the fixed low-level controller reaches the high-level assigned waypoint.
- `visited` and `success` measure whether high-level assignment plus target handling solves the SAR task.
- Large entropy gain with low visited targets indicates exploration is happening but target-to-rescue assignment is weak.
- If larger `sensor` sharply improves visited/success, the current task is likely too sparse for validating the high-level objective early in training.

## Findings

- The trained low-level policy is effective for assigned waypoint tracking. Standalone evaluation completed `793` goals over `128 * 3` agent-episodes, with `0.984` agent-episode success for at least one goal.
- In hierarchical rollout, low-level goal hits stay high across policies and sensor radii: roughly `0.81-0.90` hits per finalized high-level transition.
- RRT candidates are geometrically valid in these runs: `rrt_in_bounds_fraction=1.0`, `rrt_zero_utility_fraction=0.0`, and stable mean utility around `0.804`.
- `greedy_explore` discovers targets as sensor radius grows but never visits/rescues them, so exploration alone is insufficient.
- `target_first` is the best diagnostic policy: success increases from `0.0625` at radius `0.30` to `0.25` at radius `0.60`, and visited targets rise from `1.25` to `2.06`.
- The untrained HGSAR actor sometimes selects valid target nodes, but target selection is too rare or poorly timed without training. The 1-update smoke with radius `0.6` selected only exploration actions in the first rollout.

## Next Training Direction

Use the existing local `HeterogeneousGraphActor` / `HighLevelMapCritic` path for SAR high-level message passing first. It does not require `torch_geometric`. Run two short validation tracks before long training:

1. `sensor_radius=0.60`: easier target detection to verify the HGSAR actor can learn target switching and rescue sequencing.
2. `sensor_radius=0.30`: original setting after the relaxed run shows a clear learning signal.

Track `actor_target_actions`, `actor_valid_target_actions`, `visited_targets_mean`, `success_rate`, and low-level `goal_hits/tr`. If target actions do not increase, add auxiliary shaping or imitation warm-start from `target_first` rather than tuning the low-level controller.

## PPO Indexing Fix Follow-Up

A minibatch indexing bug in `scripts/train_high_ppo_sar.py` was fixed after these diagnostics. With the fix, HGSAR high-level training at `sensor_radius=0.60` improved from best success `0.0625` to best success `0.5625` in 30 updates, and deterministic final-checkpoint evaluation reached success `0.28125` over 32 envs. At the original `sensor_radius=0.30`, the same fixed trainer improved but remained sparse: best training success `0.125`, deterministic final-checkpoint success `0.03125`.

Detailed results are in `entity_message_passing_progress_report.md`.
