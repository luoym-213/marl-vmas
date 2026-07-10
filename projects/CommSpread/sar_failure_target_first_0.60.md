# SAR Failure Mode Analysis

policy: `target_first`
sensor_radius: `0.6`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.328125
- `failure_rate`: 0.671875
- `partial_failure_rate`: 0.671875
- `visited_targets_mean`: 2.1875
- `detected_targets_mean`: 2.59375
- `retired_agents_mean`: 2.1875
- `explore_actions_mean`: 5.70312
- `target_actions_mean`: 2.29688
- `goal_hits_mean`: 7.1875
- `active_step_hit_rate_mean`: 0.0405536
- `entropy_reduction_mean`: 15089.4
- `reason_detected_unvisited_remaining`: 25
- `reason_detected_unvisited_remaining_rate`: 0.390625
- `reason_success`: 21
- `reason_success_rate`: 0.328125
- `reason_undiscovered_remaining`: 18
- `reason_undiscovered_remaining_rate`: 0.28125
- `partial_visited_mean`: 1.7907
- `partial_detected_mean`: 2.39535
- `partial_undiscovered_mean`: 0.604651
- `partial_detected_unvisited_mean`: 0.604651
- `partial_retired_mean`: 1.7907
- `partial_explore_actions_mean`: 6.76744
- `partial_target_actions_mean`: 1.95349
- `partial_last_visit_step_mean`: 54.2791

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
