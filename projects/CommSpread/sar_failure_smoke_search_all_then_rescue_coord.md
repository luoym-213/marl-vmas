# SAR Failure Mode Analysis

policy: `search_all_then_rescue`
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
episodes: `8`

## Summary

- `episodes`: 8
- `success_rate`: 0.125
- `success_step_mean`: 50
- `failure_rate`: 0.875
- `partial_failure_rate`: 0.75
- `visited_targets_mean`: 1.375
- `detected_targets_mean`: 2.875
- `retired_agents_mean`: 1.375
- `explore_actions_mean`: 8.75
- `target_actions_mean`: 2
- `goal_hits_mean`: 9.125
- `active_step_hit_rate_mean`: 0.0407
- `duplicate_assignment_steps_mean`: 0
- `duplicate_assignment_excess_mean`: 0
- `detected_unassigned_steps_mean`: 145
- `switch_away_actions_mean`: 0
- `discovery_to_assignment_delay_mean`: 28.9048
- `assignment_to_visit_delay_mean`: 12.9048
- `entropy_reduction_mean`: 17669.1
- `reason_detected_unvisited_remaining`: 7
- `reason_detected_unvisited_remaining_rate`: 0.875
- `reason_success`: 1
- `reason_success_rate`: 0.125
- `partial_visited_mean`: 1.33333
- `partial_detected_mean`: 3
- `partial_undiscovered_mean`: 0
- `partial_detected_unvisited_mean`: 1.66667
- `partial_retired_mean`: 1.33333
- `partial_explore_actions_mean`: 8.83333
- `partial_target_actions_mean`: 2.16667
- `partial_duplicate_assignment_excess_mean`: 0
- `partial_detected_unassigned_steps_mean`: 152.167
- `partial_switch_away_actions_mean`: 0
- `partial_discovery_to_assignment_delay_mean`: 31.1111
- `partial_assignment_to_visit_delay_mean`: 12.6667
- `partial_last_visit_step_mean`: 66.3333

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
