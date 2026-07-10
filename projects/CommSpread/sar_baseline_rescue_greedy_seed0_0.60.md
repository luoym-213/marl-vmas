# SAR Failure Mode Analysis

policy: `rescue_greedy`
target_threshold: `3`
sensor_radius: `0.6`
max_steps: `None`
retire_on_rescue: `True`
progress_features: `False`
staged_rescue: `False`
rescue_detected_threshold: `2`
rescue_entropy_threshold: `None`
early_rescue_penalty: `0.0`
early_rescue_detected_threshold: `3`
search_capacity_discovery_bonus: `0.0`
discovery_active_agents_threshold: `2`
all_targets_detected_bonus: `0.0`
all_targets_detected_bonus_requires_no_rescue: `True`
rescue_phase_explore_penalty: `0.0`
rescue_phase_detected_threshold: `2`
rescue_phase_min_search_agents: `1`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.28125
- `success_step_mean`: 62.5556
- `failure_rate`: 0.71875
- `partial_failure_rate`: 0.71875
- `visited_targets_mean`: 2.125
- `detected_targets_mean`: 2.59375
- `retired_agents_mean`: 2.125
- `explore_actions_mean`: 5.875
- `target_actions_mean`: 2.21875
- `goal_hits_mean`: 7.23438
- `active_step_hit_rate_mean`: 0.0407263
- `duplicate_assignment_steps_mean`: 0
- `duplicate_assignment_excess_mean`: 0
- `detected_unassigned_steps_mean`: 71.7188
- `switch_away_actions_mean`: 0
- `discovery_to_assignment_delay_mean`: 18.1484
- `assignment_to_visit_delay_mean`: 21.3151
- `entropy_reduction_mean`: 14944.6
- `reason_detected_unvisited_remaining`: 29
- `reason_detected_unvisited_remaining_rate`: 0.453125
- `reason_success`: 18
- `reason_success_rate`: 0.28125
- `reason_undiscovered_remaining`: 17
- `reason_undiscovered_remaining_rate`: 0.265625
- `partial_visited_mean`: 1.78261
- `partial_detected_mean`: 2.43478
- `partial_undiscovered_mean`: 0.565217
- `partial_detected_unvisited_mean`: 0.652174
- `partial_retired_mean`: 1.78261
- `partial_explore_actions_mean`: 6.65217
- `partial_target_actions_mean`: 1.91304
- `partial_duplicate_assignment_excess_mean`: 0
- `partial_detected_unassigned_steps_mean`: 75.9348
- `partial_switch_away_actions_mean`: 0
- `partial_discovery_to_assignment_delay_mean`: 17.3007
- `partial_assignment_to_visit_delay_mean`: 21.7935
- `partial_last_visit_step_mean`: 50.8696

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
