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
early_rescue_penalty: `1.0`
early_rescue_detected_threshold: `2`
search_capacity_discovery_bonus: `0.5`
discovery_active_agents_threshold: `2`
all_targets_detected_bonus: `2.0`
all_targets_detected_bonus_requires_no_rescue: `True`
episodes: `64`

## Summary

- `episodes`: 64
- `success_rate`: 0.1875
- `failure_rate`: 0.8125
- `partial_failure_rate`: 0.796875
- `visited_targets_mean`: 2.04688
- `detected_targets_mean`: 2.5625
- `retired_agents_mean`: 2.04688
- `explore_actions_mean`: 6.01562
- `target_actions_mean`: 2.21875
- `goal_hits_mean`: 7.29688
- `active_step_hit_rate_mean`: 0.041057
- `entropy_reduction_mean`: 14771.9
- `reason_detected_unvisited_remaining`: 31
- `reason_detected_unvisited_remaining_rate`: 0.484375
- `reason_success`: 12
- `reason_success_rate`: 0.1875
- `reason_undiscovered_remaining`: 21
- `reason_undiscovered_remaining_rate`: 0.328125
- `partial_visited_mean`: 1.86275
- `partial_detected_mean`: 2.4902
- `partial_undiscovered_mean`: 0.509804
- `partial_detected_unvisited_mean`: 0.627451
- `partial_retired_mean`: 1.86275
- `partial_explore_actions_mean`: 6.62745
- `partial_target_actions_mean`: 2.05882
- `partial_last_visit_step_mean`: 50.9216

## Interpretation

- `undiscovered_remaining` means the episode timed out without detecting every target.
- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.
- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.
