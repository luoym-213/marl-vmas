# SAR Failure Mode Analysis

policy: `hgsar`
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
unique_rescue_assignment_bonus: `0.0`
duplicate_rescue_assignment_penalty: `0.0`
detected_unassigned_target_penalty: `0.0`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.34375
- `success_step_mean`: 64.2727
- `failure_rate`: 0.65625
- `partial_failure_rate`: 0.640625
- `visited_targets_mean`: 2.21875
- `detected_targets_mean`: 2.60938
- `retired_agents_mean`: 2.21875
- `explore_actions_mean`: 6.34375
- `target_actions_mean`: 2.4375
- `goal_hits_mean`: 8.0625
- `active_step_hit_rate_mean`: 0.0475768
- `duplicate_assignment_steps_mean`: 1.79688
- `duplicate_assignment_excess_mean`: 1.79688
- `detected_unassigned_steps_mean`: 60.8594
- `switch_away_actions_mean`: 0
- `discovery_to_assignment_delay_mean`: 18.7266
- `assignment_to_visit_delay_mean`: 18.2169
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
- `partial_duplicate_assignment_excess_mean`: 2.80488
- `partial_detected_unassigned_steps_mean`: 64.7561
- `partial_switch_away_actions_mean`: 0
- `partial_discovery_to_assignment_delay_mean`: 18.6301
- `partial_assignment_to_visit_delay_mean`: 17.8537
- `partial_last_visit_step_mean`: 49.9512

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
