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
unique_rescue_assignment_bonus: `0.0`
duplicate_rescue_assignment_penalty: `0.5`
detected_unassigned_target_penalty: `0.005`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.21875
- `success_step_mean`: 71.6429
- `failure_rate`: 0.78125
- `partial_failure_rate`: 0.765625
- `visited_targets_mean`: 2.10938
- `detected_targets_mean`: 2.625
- `retired_agents_mean`: 2.10938
- `explore_actions_mean`: 6.53125
- `target_actions_mean`: 2.25
- `goal_hits_mean`: 7.89062
- `active_step_hit_rate_mean`: 0.0446789
- `duplicate_assignment_steps_mean`: 3.9375
- `duplicate_assignment_excess_mean`: 3.9375
- `detected_unassigned_steps_mean`: 75.8125
- `switch_away_actions_mean`: 0
- `discovery_to_assignment_delay_mean`: 17.7266
- `assignment_to_visit_delay_mean`: 17.6508
- `entropy_reduction_mean`: 14563.4
- `reason_detected_unvisited_remaining`: 31
- `reason_detected_unvisited_remaining_rate`: 0.484375
- `reason_success`: 14
- `reason_success_rate`: 0.21875
- `reason_undiscovered_remaining`: 19
- `reason_undiscovered_remaining_rate`: 0.296875
- `partial_visited_mean`: 1.89796
- `partial_detected_mean`: 2.53061
- `partial_undiscovered_mean`: 0.469388
- `partial_detected_unvisited_mean`: 0.632653
- `partial_retired_mean`: 1.89796
- `partial_explore_actions_mean`: 7.20408
- `partial_target_actions_mean`: 2.06122
- `partial_duplicate_assignment_excess_mean`: 5.14286
- `partial_detected_unassigned_steps_mean`: 77.3469
- `partial_switch_away_actions_mean`: 0
- `partial_discovery_to_assignment_delay_mean`: 15.1395
- `partial_assignment_to_visit_delay_mean`: 17.5306
- `partial_last_visit_step_mean`: 49.9796

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
