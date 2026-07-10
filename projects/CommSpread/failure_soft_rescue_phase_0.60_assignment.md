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
early_rescue_penalty: `0.5`
early_rescue_detected_threshold: `2`
search_capacity_discovery_bonus: `0.0`
discovery_active_agents_threshold: `2`
all_targets_detected_bonus: `1.0`
all_targets_detected_bonus_requires_no_rescue: `True`
rescue_phase_explore_penalty: `0.05`
rescue_phase_detected_threshold: `2`
rescue_phase_min_search_agents: `1`
unique_rescue_assignment_bonus: `0.0`
duplicate_rescue_assignment_penalty: `0.0`
detected_unassigned_target_penalty: `0.0`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.25
- `success_step_mean`: 65.8125
- `failure_rate`: 0.75
- `partial_failure_rate`: 0.71875
- `visited_targets_mean`: 2.125
- `detected_targets_mean`: 2.625
- `retired_agents_mean`: 2.125
- `explore_actions_mean`: 6.42188
- `target_actions_mean`: 2.3125
- `goal_hits_mean`: 7.875
- `active_step_hit_rate_mean`: 0.0439082
- `duplicate_assignment_steps_mean`: 0.203125
- `duplicate_assignment_excess_mean`: 0.203125
- `detected_unassigned_steps_mean`: 71.0781
- `switch_away_actions_mean`: 0
- `discovery_to_assignment_delay_mean`: 17.5807
- `assignment_to_visit_delay_mean`: 19.3763
- `entropy_reduction_mean`: 14925.4
- `reason_detected_unvisited_remaining`: 30
- `reason_detected_unvisited_remaining_rate`: 0.46875
- `reason_success`: 16
- `reason_success_rate`: 0.25
- `reason_undiscovered_remaining`: 18
- `reason_undiscovered_remaining_rate`: 0.28125
- `partial_visited_mean`: 1.91304
- `partial_detected_mean`: 2.54348
- `partial_undiscovered_mean`: 0.456522
- `partial_detected_unvisited_mean`: 0.630435
- `partial_retired_mean`: 1.91304
- `partial_explore_actions_mean`: 7.04348
- `partial_target_actions_mean`: 2.13043
- `partial_duplicate_assignment_excess_mean`: 0.282609
- `partial_detected_unassigned_steps_mean`: 74.4348
- `partial_switch_away_actions_mean`: 0
- `partial_discovery_to_assignment_delay_mean`: 16.3659
- `partial_assignment_to_visit_delay_mean`: 19.0652
- `partial_last_visit_step_mean`: 52.3696

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
