# CommSpread MARL 实验日志

更新日期：2026-07-22 UTC；当前 HEAD：`54c95bd0fbb30312a744bc37b0c08a8c351c2974`。
证据标记：**事实**来自保存产物；**观察**不等于因果；无法恢复的运行 commit 标为 `UNKNOWN`。

## 历史实验（E00–E20，压缩交接）

| 编号 | 只改变量/假设 | 结果（保存报告或旧日志） | 结论 | 是否排除 |
|---|---|---|---|---|
| E00–E06 | Spread/MPE parity、global critic、actor 特征、physics/spawn | success 0 至 .00781 | 迁移线未形成可用策略 | 不作为 SAR 主线 |
| E07–E08 | old-PPO profile、generic GNN | success 0 或早期 return −38 到 −43 | 未达 parity | 暂缓 |
| E09 | 低层 goal reaching | auto-resampled 指标约 98.4% | 非部署一致；不得作正式成功率 | 低层重训降级 |
| E10 | PPO minibatch indexing 修复 | radius .6 deterministic 64 `.34375` | 修复保留 | 保留 |
| E11 | sensor radius `.3` | eval `.03125` | `.6` 更可靠 | `.3` 排除 |
| E12 | `retire_on_rescue=False` | `.671875` | 改变任务语义，仅 ceiling | 正式排除 |
| E13–E15 | hard stage / soft shaping / rescue phase shaping | 最佳 `.375`、`.1875`、`.25` | 不稳定或出现 duplicate | 降级 |
| E16 | 仅 LR `1e-3` | best `.375`、final `.1875` | learning-rate-only 不足 | 排除 |
| E17 | progress/target assignment features | `.296875`、duplicate `2.703125` | 仅特征不足 | optional |
| E18 | greedy/search-all heuristic | 约 `.224/.203` | 非最终策略 | 排除 |
| E19–E20 | assignment/latency/duplicate shaping | 低 success 或 duplicate 增大 | reward shaping 不作为当前路线 | 暂缓 |

这些历史实验多数运行于 dirty worktree，精确运行 commit 为 `UNKNOWN`；不得把当前 HEAD 倒填为其唯一 commit。

### E00–E20 逐项最小索引

| 编号 | 日期/Git | 假设与唯一变化 | 配置、seed、checkpoint、曲线 | 评估结果与结论 |
|---|---|---|---|---|
| E00 | 日期 UNKNOWN / Git UNKNOWN | Spread MPE parity baseline | seeds 0/1/2；路径 UNKNOWN | success 0；parity 不充分 |
| E01 | UNKNOWN / UNKNOWN | 仅 global critic | seeds 0/1/2 | success 0；排除单独 critic |
| E02 | UNKNOWN / UNKNOWN | 仅 actor 队友相对位置 | seeds 0/1/2 | success 0；排除单独 actor 特征 |
| E03 | UNKNOWN / UNKNOWN | richer actor + global critic | seeds 0/1/2 | success .000651；非主因 |
| E04 | UNKNOWN / UNKNOWN | deterministic action mapping probe | 无训练 | 方向映射正确；排除 mapping bug |
| E05 | UNKNOWN / UNKNOWN | physics/action scale parity | seeds 0/1/2 | `.00195/0/0`；物理 mismatch 不充分 |
| E06 | UNKNOWN / UNKNOWN | physics + independent spawn parity | seeds 0/1/2 | `0/.00781/.00586`；MPE 线仍弱 |
| E07 | UNKNOWN / UNKNOWN | old-PPO profile | seed0，约819k frames | success 0、distance .5778；不扩展 |
| E08 | UNKNOWN / UNKNOWN | generic GNN | smoke/早停长跑 | return −38 至 −43；待 feature parity 审计 |
| E09 | UNKNOWN / UNKNOWN | low-level goal-reaching diagnosis | fixed checkpoint；793 goals | auto-resampled 到达约98.4%，非部署指标 |
| E10 | UNKNOWN / UNKNOWN | PPO minibatch indexing fix | seed0/latest、radius .6 | deterministic64 `.34375`；修复保留 |
| E11 | UNKNOWN / UNKNOWN | sensor radius `.3` | seed0 | train best `.125`、eval `.03125`；排除 `.3` |
| E12 | UNKNOWN / UNKNOWN | `retire_on_rescue=False` | seed0 | `.671875` vs retire `.34375`；仅 ceiling |
| E13 | UNKNOWN / UNKNOWN | hard staged threshold 2/3 | seed0 | best `.375/.25`，不稳；降级 |
| E14 | UNKNOWN / UNKNOWN | soft stage shaping | seed0 | `.1875`；排除该 shaping |
| E15 | UNKNOWN / UNKNOWN | rescue-phase shaping | seed0 | `.25`、duplicate `.203125`；不足 |
| E16 | UNKNOWN / UNKNOWN | only LR `1e-3` | seed0、20 updates | best `.375`、final `.1875`；排除 |
| E17 | UNKNOWN / UNKNOWN | progress + assignment features | seed0 | `.296875`、duplicate `2.703125`；optional |
| E18 | UNKNOWN / UNKNOWN | greedy/search-all heuristics | 3 seeds | 约`.224/.203`；非最终策略 |
| E19 | UNKNOWN / UNKNOWN | assignment bonus/penalty shaping | seed0 | `.140625`、duplicate `6.5625`；排除 |
| E20 | UNKNOWN / UNKNOWN | latency/duplicate-strong shaping | 旧产物 | 低于 fixed baseline；暂缓 |

## E21 — coverage / timing oracle

- 日期：2026-07-16；Git：`e584e296` + dirty diff（精确 diff `UNKNOWN`）。
- 假设：理想搜索/低层可分离物理上限与高层调度上限。
- 只改变量：oracle timing（immediate、threshold2、dynamic、all-detected）；同一 256 布局。
- 结果：123/256、138/256、135/256、185/256；full-information repaired-MAPPO `254/256`。
- 结论：物理/传感器不是唯一瓶颈，高层搜索/调度仍有大量空间。**观察**，不是可部署 policy。
- 产物：`outputs/hierarchical_sar_diagnostics/dynamic_release_research/`。

## E22 — event-driven dynamic release + staggered commitment

- 日期：2026-07-16；Git：`e584e296` + dirty diff。
- 假设：检测/assignment 时 stale option 和对称同步决策导致 rescue delay/duplicate。
- 相对基线只改：event redecision、observable dynamic release、每事件最多一个 commitment、latch、terminal rescue-only mask；不改 reward、PPO、低层、retire 语义。
- checkpoint/seed：repaired update10；seed0/seed1 paired 256。
- 结果：66 vs 53、62 vs 52；pooled `128/512` vs `105/512`，`+4.492pp`，CI `[+0.391,+8.594]pp`。
- 结论：**事实**，保留为 finder-only 之前的正式 baseline；checkpoint 10 不能被 latest 自动替代。

## E23 — assignment diagnostics parity

- 日期：2026-07-17；Git：`e584e296` + dirty diff。
- 假设：只加入诊断不会改变 rollout。
- 只改：评估输出 cost/regret/crossing，不改 actor/env。
- 配置：旧 staggered checkpoint10，64 env、seed0、100 steps、radius .6、repaired fallback。
- 结果：逐环境字段与旧结果一致；20/64、148 commitment events、5 crossing pairs、crossing 3.378%、sum/makespan regret `.01309/.00980`。
- 结论：**事实**，诊断 path parity 通过，保留。

## E24–E25 — finder cascade smoke 与旧 checkpoint 64 筛选

- 日期：2026-07-17；Git：`e584e296` + dirty diff。
- 假设：finder-first 能在不读取隐藏目标、无固定 agent index、at-most-one commitment 下改善执行。
- 只改：默认关闭的 cascade state / mode；低层、reward、网络和高层 checkpoint 不变。
- 64 结果：baseline 20/64；finder-only 30/64（paired 12/2）；immediate 28/64；one-event 29/64。finder-only terminal `reason_undiscovered_remaining` 为 **16/64**；轨迹 failure category 1 为 22/64，二者不可互换。
- 结论：**观察**，finder-only 候选最高，但 crossing/regret 增大；进入训练与正式验证。
- 产物：`outputs/hierarchical_sar_diagnostics/finder_cascade_research/{finder_only_64,immediate_64,one_event_64}.*`。

## E26 — finder-first seed-0 训练

- 日期：2026-07-17；训练元数据 Git `e584e296`、diff SHA256 `53b2b87083eea46dcf38c0a0d4f1a3c908a278074707fdc9f8b9d36e82e5bab3`。
- 假设：finder-only 筛选收益可由 HGSAR + repaired MAPPO 学到。
- 相对 E22 只改：`enable_finder_first_cascade=True`、`finder_cascade_mode=finder_only`；其余 reward、网络、PPO、低层、sensor/horizon/retire 不变。
- 配置/命令：`outputs/sar_high_hgsar_finder_first_finder_only_repaired_mappo_sensor_0.60/seed_0/texts/config.json`。
- seed/checkpoints：seed0；16 env、30 updates；保存 10/20/30/latest。
- 曲线：update 1/2/3 success `5/16, 7/16, 10/16`；update 30 `5/16`；最终 KL `.002639`、clip `.02409`、invalid/nonfinite 0。
- 结论：训练稳定性是**事实**；batch success 不构成泛化结论。

## E27 — finder-only checkpoint 选择与正式配对验证

- 日期：2026-07-18；评估元数据 Git `54c95bd`，运行时 dirty diff SHA256 见各 JSON metadata。
- 假设：finder-only 的旧 checkpoint 执行机制收益可在两套独立布局复现；finder-first 重训是否有额外收益。
- 控制：3 UAV/3 targets/100 steps/radius .6/retire、repaired MAPPO + fallback、同一 reset 顺序；无训练、无代码改动。
- 64 选择：trained update10/20/30 都 26/64；按预先规则选 update30（baseline-only 最少 2、detect-all 50、category1 最少 26），不使用 latest 或 batch success。
- 正式机制效果：staggered `128/512` -> old checkpoint + finder-only `220/512`，`+17.969pp`，107/15 flips，CI `[14.063,21.875]pp`；seed0 66->114，seed1 62->106。
- 重训额外效果：old+finder `220/512` -> trained update30 `223/512`，`+0.586pp`，30/27，CI `[-2.344,3.516]pp`。
- 诊断：old+finder pooled accept/reject/unavailable `963/320/102`；297/320 rejected targets later claimed，平均/中位/P95/max wait `34.67/36/72/94`；duplicate/invalid/nonfinite 皆 0。
- 结论：**事实**，可靠提升来自执行机制；重训练额外收益未证实。**不排除 finder-only 机制；排除“已证明 retraining 有额外收益”的说法。**
- 产物：`outputs/hierarchical_sar_diagnostics/finder_cascade_research/p0_finder_only_checkpoint_and_formal_report.md`。

## E28 — 高层模块替换 2×2×2 消融

- 日期：2026-07-18；评估元数据 Git `54c95bd`，A 的运行 diff SHA256 `90e25697e93f49ade3024157687b5a8491baae0e30ad40306b55c4bb235ae1d1`。
- 假设：42.97% -> 约 65% 的差距可拆分为搜索、救援时机、任务分配；若单项和与总效应不符则存在交互。
- 控制：A–H 都使用旧 checkpoint + finder-only、同一独立 seed0/seed1 256 布局、repaired MAPPO + fallback、3/3/100/.6/retire；无训练；不改 reward/网络/物理/成功判定。D 只看已发现 target；核心表不含 proportional controller。
- 单模块：A `220/512`；B（三通道）`241/512`，`+4.10pp` CI `[-.59,+8.98]`；C（all-detected）`230/512`，`+1.95pp` CI `[-1.76,+5.66]`；D（exact matching）`216/512`，`-.78pp` CI `[-2.15,+.59]`。
- 交互触发：单项和 `+5.27pp`，而 E（全替换）`328/512 = 64.06%`，`+21.09pp` CI `[+16.02,+26.17]`；残差 `+15.82pp` CI `[+10.16,+21.68]`，故执行 F/G/H 完整 2×2×2。
- F/G/H：274/512、238/512、254/512；factorial interaction search×timing `+4.49pp`、timing×assignment `+5.47pp`、三阶 `+5.66pp`。
- 关键诊断：B 相对 A team coverage `+.0213`、overlap `-.0665`、detect-all `+13.09pp`；A premature retirement 357/512，all-detected gate 为 0。A 未发现 225、发现后未完成 67；E 为 24、160。
- 结论：**事实**，差距强非可加；单独分配不是当前瓶颈。**观察**：有效方向是搜索与可靠终局 handoff 的联合设计。
- 审计：最初 `D_seed0.json` 有 stale target action/duplicate，保留为无效产物；正式使用 `D_v2_seed0/1`，exact cells duplicate/invalid/nonfinite 均 0。
- 产物：`outputs/hierarchical_sar_diagnostics/module_ablation_20260718/module_ablation_report.md`、逐环境 JSON、coverage CSV、`paired_comparisons.csv`、`experiment_config_and_commands.json`。

## 当前结论索引

- 当前正式策略：E27 的 old checkpoint + finder-only，220/512。
- 当前最佳诊断参考：E28 的 E，328/512；它是模块上限/归因实验，不是已训练策略。
- 下一步：见 `docs/NEXT_STEPS.md`。所有旧实验完整运行 commit 若无 JSON metadata 支持则为 `UNKNOWN`。
