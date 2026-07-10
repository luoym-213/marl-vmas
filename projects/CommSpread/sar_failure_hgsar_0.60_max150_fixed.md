# SAR Failure Mode Analysis

policy: `hgsar`
sensor_radius: `0.6`
max_steps: `150`
retire_on_rescue: `True`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.375
- `failure_rate`: 0.625
- `partial_failure_rate`: 0.625
- `visited_targets_mean`: 2.34375
- `detected_targets_mean`: 2.65625
- `retired_agents_mean`: 2.34375
- `explore_actions_mean`: 7.26562
- `target_actions_mean`: 2.48438
- `goal_hits_mean`: 9.09375
- `active_step_hit_rate_mean`: 0.0453833
- `entropy_reduction_mean`: 14916.2
- `reason_detected_unvisited_remaining`: 19
- `reason_detected_unvisited_remaining_rate`: 0.296875
- `reason_success`: 24
- `reason_success_rate`: 0.375
- `reason_undiscovered_remaining`: 21
- `reason_undiscovered_remaining_rate`: 0.328125
- `partial_visited_mean`: 1.95
- `partial_detected_mean`: 2.45
- `partial_undiscovered_mean`: 0.55
- `partial_detected_unvisited_mean`: 0.5
- `partial_retired_mean`: 1.95
- `partial_explore_actions_mean`: 9
- `partial_target_actions_mean`: 2.175
- `partial_last_visit_step_mean`: 59.65

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
