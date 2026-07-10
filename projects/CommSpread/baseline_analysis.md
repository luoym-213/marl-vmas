# Baseline Analysis: `spread_mpe_parity`
## Executive Summary
- Ran three CUDA MAPPO+MLP baseline seeds on `spread_mpe_parity` with no algorithm-code changes and no tuning. Seed 0 was stopped after the 10.5M-frame checkpoint when the run had clearly plateaued; seeds 1 and 2 were run to the same 10.5M-frame budget for comparable diagnosis.
- All three final checkpoints have `512`-episode evaluator `success_rate = 0.0` and `mean_episode_length = 50.0`, so the learned policies do not solve the MPE parity task.
- Reward improves substantially from initialization but plateaus around `-30` to `-32` episode reward per agent. Matched distance remains around `0.56-0.62`, far above the success threshold of `0.1`.
- Value loss decreases instead of diverging. Entropy decreases but does not collapse to zero. Offline action distributions are broad, so failure is not a simple single-action collapse.
- The current centralized critic is centralized only over the group observations, not an explicit global state. In parity mode each agent observation omits other-agent positions, so the critic likely lacks direct team-assignment/global-coverage information.
- Current evidence points most strongly to algorithm/model/input-information mismatch or hyperparameter/objective mismatch under MAPPO, not a basic environment implementation failure.
## Runs
| seed | frames | iterations | checkpoint | final collection reward | best collection reward | final collection success | best collection success | final matched dist | final entropy | final critic loss |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10620000 | 177 | `projects/CommSpread/outputs/baseline_mpe_parity/seed_0/mappo_spread_mlp__b62bbdd7_26_07_08-06_58_42/checkpoints/checkpoint_10500000.pt` | -30.137 | -29.942 | 0.000000 | 0.000050 | 0.603 | 0.773 | 3.619 |
| 1 | 10500000 | 175 | `projects/CommSpread/outputs/baseline_mpe_parity/seed_1/mappo_spread_mlp__68e19ce8_26_07_08-07_37_41/checkpoints/checkpoint_10500000.pt` | -31.859 | -31.497 | 0.000000 | 0.000033 | 0.637 | 0.687 | 4.551 |
| 2 | 10500000 | 175 | `projects/CommSpread/outputs/baseline_mpe_parity/seed_2/mappo_spread_mlp__839a72c4_26_07_08-08_11_49/checkpoints/checkpoint_10500000.pt` | -30.887 | -30.310 | 0.000000 | 0.000033 | 0.619 | 0.736 | 4.114 |

## Final Evaluator Results
| seed | success_rate | mean_episode_length | mean_final_matched_distance | mean_step_reward | mean_episode_reward_per_agent |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 50.0 | 0.565 | -0.605 | -30.259 |
| 1 | 0.000 | 50.0 | 0.616 | -0.642 | -32.088 |
| 2 | 0.000 | 50.0 | 0.567 | -0.618 | -30.906 |

## Action Distribution
Action entropy is computed over actions `0..4`; the maximum possible entropy is `ln(5)=1.609`.

| seed | deterministic probs [0,1,2,3,4] | det entropy | det max share | stochastic entropy | stochastic max share |
|---:|---|---:|---:|---:|---:|
| 0 | `[0.100, 0.213, 0.201, 0.241, 0.245]` | 1.570 | 0.245 | 1.555 | 0.240 |
| 1 | `[0.253, 0.261, 0.181, 0.205, 0.100]` | 1.563 | 0.261 | 1.576 | 0.275 |
| 2 | `[0.306, 0.132, 0.181, 0.158, 0.223]` | 1.565 | 0.306 | 1.598 | 0.251 |

## Diagnostic Questions
1. **Reward 是否稳定上升**: yes initially, but not to task success. Seed 0 improves from `-40.50` to about `-30.76`; seed 1 from `-39.94` to `-31.86`; seed 2 from `-40.69` to `-30.89`. All plateau far from zero reward.
2. **success_rate 是否上升**: no meaningful rise. Collection success remains effectively zero, with best scalar values around one or a few `1e-5` events. Final 512-episode evaluator success is `0.0` for all seeds.
3. **value loss 是否发散**: no. Critic loss drops sharply: seed 0 `55.37 -> 3.92`, seed 1 `52.82 -> 4.91`, seed 2 `53.61 -> 3.92` approximately. This argues against obvious critic numerical divergence.
4. **entropy 是否过早下降**: entropy drops from about `1.59` toward `0.77-0.90` by 10.5M frames, below the uniform-action maximum `1.609` but not near zero. It narrows exploration while success is still zero, so entropy decay may be premature relative to task difficulty, but it is not complete collapse.
5. **action distribution 是否塌缩**: no single-action collapse. Deterministic rollout max action share is `0.245`, `0.261`, and `0.306` for seeds 0/1/2. This is broad enough that failure is more subtle than one constant action.
6. **episode length 是否始终卡在 50**: yes. Evaluator mean and median episode length are exactly `50` for all seeds; eval scalar logs also stay at or extremely near `50`. Policies rarely, if ever, trigger all-landmark coverage termination.
7. **centralized critic 是否真的使用了足够信息**: probably not. `CommSpreadTask.state_spec()` returns `None`; BenchMARL MAPPO then builds a centralized critic from grouped agent observations. In `spread_mpe_parity`, each agent observation contains self velocity, self position, and landmark-relative positions, but no other-agent relative positions. The critic sees all agents as a grouped observation tensor, but there is no explicit global state with all absolute agent/landmark positions or assignment structure. This may be insufficient for learning coordinated Hungarian-style coverage.
8. **失败更像环境、算法实现、还是超参数问题**: environment parity basics look correct from prior deterministic probes and evaluator reward scale. The failure is most consistent with algorithm/model/input-information mismatch plus possible hyperparameter issues: MAPPO+plain MLP improves dense matched-distance reward but does not discover successful assignment/coverage. The central critic input limitation is a concrete suspect. Hyperparameters may also matter because entropy decays while success remains zero, but value loss stability and broad action use make pure optimizer instability less likely.

## Logs And TensorBoard Notes
- BenchMARL was configured with CSV logging only (`loggers=["csv"]`), so the available scalar logs are CSV files under each run's `scalars/` directory rather than TensorBoard event files. No TensorBoard event files were produced by these runs.
- Checkpoints are retained at 300k-frame intervals, with final analysis using `checkpoint_10500000.pt` for each seed.
- Seed 0 was manually stopped after the 10.5M checkpoint due to plateau; seed 1/2 used `--max-n-frames 10500000` to match that diagnostic budget.

## Recommendation
- Do not tune rewards first; the parity environment already gives a smooth dense signal and the policies reduce matched distance.
- Next experiment should isolate critic/input structure: add an explicit global state or restore the old relational/message-passing inductive bias, then compare against this baseline.
- For SAR, train high level with the fixed `low_safe_0.06` checkpoint rather than jointly training high and low; the current `train_high_ppo_sar.py` path is already structured that way.
