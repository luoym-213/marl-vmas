# SAR Failure Mode Analysis

policy: `hgsar`
sensor_radius: `0.6`
max_steps: `None`
retire_on_rescue: `False`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.671875
- `failure_rate`: 0.328125
- `partial_failure_rate`: 0.3125
- `visited_targets_mean`: 2.59375
- `detected_targets_mean`: 2.71875
- `retired_agents_mean`: 0
- `explore_actions_mean`: 16.4531
- `target_actions_mean`: 2.75
- `goal_hits_mean`: 16.5
- `active_step_hit_rate_mean`: 0.055
- `entropy_reduction_mean`: 18476.5
- `reason_detected_unvisited_remaining`: 8
- `reason_detected_unvisited_remaining_rate`: 0.125
- `reason_success`: 43
- `reason_success_rate`: 0.671875
- `reason_undiscovered_remaining`: 13
- `reason_undiscovered_remaining_rate`: 0.203125
- `partial_visited_mean`: 1.85
- `partial_detected_mean`: 2.2
- `partial_undiscovered_mean`: 0.8
- `partial_detected_unvisited_mean`: 0.35
- `partial_retired_mean`: 0
- `partial_explore_actions_mean`: 17.3
- `partial_target_actions_mean`: 2.3
- `partial_last_visit_step_mean`: 49.35

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
