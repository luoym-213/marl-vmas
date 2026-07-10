# SAR Failure Mode Analysis

policy: `hgsar`
sensor_radius: `0.6`
max_steps: `150`
retire_on_rescue: `True`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.34375
- `failure_rate`: 0.65625
- `partial_failure_rate`: 0.640625
- `visited_targets_mean`: 2.21875
- `detected_targets_mean`: 2.60938
- `retired_agents_mean`: 2.21875
- `explore_actions_mean`: 6.34375
- `target_actions_mean`: 2.4375
- `goal_hits_mean`: 8.0625
- `active_step_hit_rate_mean`: 0.0475768
- `entropy_reduction_mean`: 14644.2
- `reason_detected_unvisited_remaining`: 21
- `reason_detected_unvisited_remaining_rate`: 0.328125
- `reason_success`: 22
- `reason_success_rate`: 0.34375
- `reason_undiscovered_remaining`: 21
- `reason_undiscovered_remaining_rate`: 0.328125
- `partial_visited_mean`: 1.85366
- `partial_detected_mean`: 2.43902
- `partial_undiscovered_mean`: 0.560976
- `partial_detected_unvisited_mean`: 0.585366
- `partial_retired_mean`: 1.85366
- `partial_explore_actions_mean`: 7.17073
- `partial_target_actions_mean`: 2.17073
- `partial_last_visit_step_mean`: 49.9512

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
