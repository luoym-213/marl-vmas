# SAR Failure Mode Analysis

policy: `target_first`
target_threshold: `3`
sensor_radius: `0.6`
max_steps: `None`
retire_on_rescue: `True`
progress_features: `False`
staged_rescue: `True`
rescue_detected_threshold: `2`
rescue_entropy_threshold: `None`
episodes: `8`

## Summary

- `episodes`: 8
- `success_rate`: 0.125
- `failure_rate`: 0.875
- `partial_failure_rate`: 0.875
- `visited_targets_mean`: 2
- `detected_targets_mean`: 2.5
- `retired_agents_mean`: 2
- `explore_actions_mean`: 6.375
- `target_actions_mean`: 2.125
- `goal_hits_mean`: 7.625
- `active_step_hit_rate_mean`: 0.0398486
- `entropy_reduction_mean`: 16166.1
- `reason_detected_unvisited_remaining`: 4
- `reason_detected_unvisited_remaining_rate`: 0.5
- `reason_success`: 1
- `reason_success_rate`: 0.125
- `reason_undiscovered_remaining`: 3
- `reason_undiscovered_remaining_rate`: 0.375
- `partial_visited_mean`: 1.85714
- `partial_detected_mean`: 2.42857
- `partial_undiscovered_mean`: 0.571429
- `partial_detected_unvisited_mean`: 0.571429
- `partial_retired_mean`: 1.85714
- `partial_explore_actions_mean`: 6.85714
- `partial_target_actions_mean`: 2
- `partial_last_visit_step_mean`: 56.8571

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
