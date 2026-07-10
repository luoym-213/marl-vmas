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
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.296875
- `failure_rate`: 0.703125
- `partial_failure_rate`: 0.703125
- `visited_targets_mean`: 2.25
- `detected_targets_mean`: 2.64062
- `retired_agents_mean`: 2.25
- `explore_actions_mean`: 6
- `target_actions_mean`: 2.32812
- `goal_hits_mean`: 7.60938
- `active_step_hit_rate_mean`: 0.0457498
- `duplicate_assignment_steps_mean`: 2.70312
- `duplicate_assignment_excess_mean`: 2.70312
- `detected_unassigned_steps_mean`: 65.9219
- `switch_away_actions_mean`: 0
- `discovery_to_assignment_delay_mean`: 16.3958
- `assignment_to_visit_delay_mean`: 18.3333
- `entropy_reduction_mean`: 14436.4
- `reason_detected_unvisited_remaining`: 24
- `reason_detected_unvisited_remaining_rate`: 0.375
- `reason_success`: 19
- `reason_success_rate`: 0.296875
- `reason_undiscovered_remaining`: 21
- `reason_undiscovered_remaining_rate`: 0.328125
- `partial_visited_mean`: 1.93333
- `partial_detected_mean`: 2.48889
- `partial_undiscovered_mean`: 0.511111
- `partial_detected_unvisited_mean`: 0.555556
- `partial_retired_mean`: 1.93333
- `partial_explore_actions_mean`: 6.95556
- `partial_target_actions_mean`: 2.04444
- `partial_duplicate_assignment_excess_mean`: 3.84444
- `partial_detected_unassigned_steps_mean`: 71.2889
- `partial_switch_away_actions_mean`: 0
- `partial_discovery_to_assignment_delay_mean`: 15.8296
- `partial_assignment_to_visit_delay_mean`: 18.1556
- `partial_last_visit_step_mean`: 50.6444

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
