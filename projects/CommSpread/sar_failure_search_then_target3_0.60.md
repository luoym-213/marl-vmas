# SAR Failure Mode Analysis

policy: `search_then_target`
target_threshold: `3`
sensor_radius: `0.6`
max_steps: `None`
retire_on_rescue: `True`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0
- `failure_rate`: 1
- `partial_failure_rate`: 0.46875
- `visited_targets_mean`: 0.46875
- `detected_targets_mean`: 2.75
- `retired_agents_mean`: 0.46875
- `explore_actions_mean`: 14.0469
- `target_actions_mean`: 0.5625
- `goal_hits_mean`: 12.2344
- `active_step_hit_rate_mean`: 0.0432435
- `entropy_reduction_mean`: 19002.1
- `reason_detected_unvisited_remaining`: 64
- `reason_detected_unvisited_remaining_rate`: 1
- `partial_visited_mean`: 1
- `partial_detected_mean`: 3
- `partial_undiscovered_mean`: 0
- `partial_detected_unvisited_mean`: 2
- `partial_retired_mean`: 1
- `partial_explore_actions_mean`: 12.0667
- `partial_target_actions_mean`: 1
- `partial_last_visit_step_mean`: 62.4

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
