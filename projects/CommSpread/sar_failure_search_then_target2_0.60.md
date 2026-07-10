# SAR Failure Mode Analysis

policy: `search_then_target`
target_threshold: `2`
sensor_radius: `0.6`
max_steps: `None`
retire_on_rescue: `True`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0
- `failure_rate`: 1
- `partial_failure_rate`: 0.890625
- `visited_targets_mean`: 1.32812
- `detected_targets_mean`: 2.70312
- `retired_agents_mean`: 1.32812
- `explore_actions_mean`: 9.65625
- `target_actions_mean`: 1.51562
- `goal_hits_mean`: 9.51562
- `active_step_hit_rate_mean`: 0.0401196
- `entropy_reduction_mean`: 17371.3
- `reason_detected_unvisited_remaining`: 64
- `reason_detected_unvisited_remaining_rate`: 1
- `partial_visited_mean`: 1.49123
- `partial_detected_mean`: 2.75439
- `partial_undiscovered_mean`: 0.245614
- `partial_detected_unvisited_mean`: 1.26316
- `partial_retired_mean`: 1.49123
- `partial_explore_actions_mean`: 9.15789
- `partial_target_actions_mean`: 1.59649
- `partial_last_visit_step_mean`: 59.0877

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
