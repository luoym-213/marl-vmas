# SAR Failure Mode Analysis

policy: `hgsar`
target_threshold: `3`
sensor_radius: `0.6`
max_steps: `None`
retire_on_rescue: `True`
progress_features: `True`
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
unique_rescue_assignment_bonus: `0.2`
duplicate_rescue_assignment_penalty: `0.1`
detected_unassigned_target_penalty: `0.02`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.140625
- `success_step_mean`: 56.2222
- `failure_rate`: 0.859375
- `partial_failure_rate`: 0.859375
- `visited_targets_mean`: 2.03125
- `detected_targets_mean`: 2.59375
- `retired_agents_mean`: 2.03125
- `explore_actions_mean`: 6.39062
- `target_actions_mean`: 2.23438
- `goal_hits_mean`: 7.67188
- `active_step_hit_rate_mean`: 0.0440407
- `duplicate_assignment_steps_mean`: 6.5625
- `duplicate_assignment_excess_mean`: 6.5625
- `detected_unassigned_steps_mean`: 73.375
- `switch_away_actions_mean`: 0
- `discovery_to_assignment_delay_mean`: 15.151
- `assignment_to_visit_delay_mean`: 18.0807
- `entropy_reduction_mean`: 14459.7
- `reason_detected_unvisited_remaining`: 33
- `reason_detected_unvisited_remaining_rate`: 0.515625
- `reason_success`: 9
- `reason_success_rate`: 0.140625
- `reason_undiscovered_remaining`: 22
- `reason_undiscovered_remaining_rate`: 0.34375
- `partial_visited_mean`: 1.87273
- `partial_detected_mean`: 2.52727
- `partial_undiscovered_mean`: 0.472727
- `partial_detected_unvisited_mean`: 0.654545
- `partial_retired_mean`: 1.87273
- `partial_explore_actions_mean`: 6.8
- `partial_target_actions_mean`: 2.10909
- `partial_duplicate_assignment_excess_mean`: 7.63636
- `partial_detected_unassigned_steps_mean`: 77.6
- `partial_switch_away_actions_mean`: 0
- `partial_discovery_to_assignment_delay_mean`: 15.0364
- `partial_assignment_to_visit_delay_mean`: 18.0091
- `partial_last_visit_step_mean`: 47.5818

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
