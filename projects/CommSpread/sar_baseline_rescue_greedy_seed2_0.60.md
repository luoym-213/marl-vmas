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
- `success_rate`: 0.15625
- `success_step_mean`: 69.5
- `failure_rate`: 0.84375
- `partial_failure_rate`: 0.828125
- `visited_targets_mean`: 1.92188
- `detected_targets_mean`: 2.51562
- `retired_agents_mean`: 1.92188
- `explore_actions_mean`: 6.39062
- `target_actions_mean`: 2.04688
- `goal_hits_mean`: 7.42188
- `active_step_hit_rate_mean`: 0.0389547
- `duplicate_assignment_steps_mean`: 0
- `duplicate_assignment_excess_mean`: 0
- `detected_unassigned_steps_mean`: 76.0625
- `switch_away_actions_mean`: 0
- `discovery_to_assignment_delay_mean`: 18.8677
- `assignment_to_visit_delay_mean`: 20.1429
- `entropy_reduction_mean`: 15274.9
- `reason_detected_unvisited_remaining`: 36
- `reason_detected_unvisited_remaining_rate`: 0.5625
- `reason_success`: 10
- `reason_success_rate`: 0.15625
- `reason_undiscovered_remaining`: 18
- `reason_undiscovered_remaining_rate`: 0.28125
- `partial_visited_mean`: 1.75472
- `partial_detected_mean`: 2.45283
- `partial_undiscovered_mean`: 0.54717
- `partial_detected_unvisited_mean`: 0.698113
- `partial_retired_mean`: 1.75472
- `partial_explore_actions_mean`: 6.77358
- `partial_target_actions_mean`: 1.90566
- `partial_duplicate_assignment_excess_mean`: 0
- `partial_detected_unassigned_steps_mean`: 80.2453
- `partial_switch_away_actions_mean`: 0
- `partial_discovery_to_assignment_delay_mean`: 18.6038
- `partial_assignment_to_visit_delay_mean`: 20.2453
- `partial_last_visit_step_mean`: 48.8302

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
