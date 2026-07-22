# CommSpread MARL 项目状态与会话交接

生成日期：2026-07-22 UTC
Git 根目录：`/workspace`；项目目录：`projects/CommSpread/`
分支：`research/dynamic-search-rescue-release-20260716`；当前 HEAD：`54c95bd0fbb30312a744bc37b0c08a8c351c2974`

## 证据等级

- **已验证事实**：当前代码、配置、训练 CSV、评估 JSON/CSV 或配对报告直接支持。
- **观察结果**：已测量描述，不能自动解释因果。
- **推测**：尚未充分验证的机制解释。
- **待验证假设**：后续单变量实验要检验的命题。
- **UNKNOWN**：当前仓库与保存产物不能确认。

在 `/workspace` 及 `projects/CommSpread/` 下未找到可读取的 `AGENTS.md`；项目级附加指引为 **UNKNOWN**。

## 1. 研究目标与问题定义

### 已验证事实

- 当前主线是 VMAS SAR 分层多智能体任务：3 UAV、3 targets、100 steps、`sensor_radius=0.6`。
- 正式任务语义为 `retire_on_rescue=True`；`no_retire` 只能作上限诊断，不能混入正式比较。
- 高层 HGSAR 在 RRT 搜索节点和已发现 target 救援动作之间选择；固定 repaired MAPPO 低层执行目标，并启用 goal fallback。
- 当前部署候选是**旧 high-level checkpoint** `outputs/sar_high_hgsar_dynamic_staggered_repaired_mappo_sensor_0.60/seed_0/checkpoints/checkpoint_10.pt` 加 finder-only 执行机制，不是 finder-first 重训练 checkpoint。
- 部署匹配的低层证据是固定目标三智能体成功 `249/256 = 97.27%`，full-information SAR `254/256 = 99.22%`。auto-resampled goals 的约 98.4% 不可作部署一致成功率。

当前问题是：旧 checkpoint + finder-only 在两套独立 256 布局上为 `220/512 = 42.97%`，而三通道搜索 + 全发现 gate + 精确匹配为 `328/512 = 64.06%`；需定位学习式搜索、切换时机、任务分配及其交互造成的差距。

## 2. 当前算法和训练/评估流程

1. `sar_scenario.py` 维护 belief map、目标检测/访问/认领、active/retired UAV、RRT candidates 与奖励。
2. `AsyncSMDPCollector` 在 option 完成、检测变化、assignment 变化或 cascade 事件时请求高层新决策。
3. `HGSARActorCriticPolicy` 包装 HGSAR actor/map critic；动作是 RRT search nodes 或 target nodes。
4. event-driven staggered release 每事件最多新增一个 rescue commitment，已承诺 UAV 使用 latch；检测/assignment change 都触发重决策。
5. finder-only 在新目标发现时先向 finder 开放，拒绝/不可用/窗口耗尽才扩散；只使用可观测状态，默认关闭。

历史 finder-first seed-0 训练配置已保存：16 env、30 updates、400 low-level steps/batch；PPO `lr=3e-4`、`gamma=.99`、`gae_lambda=.95`、`clip_eps=.2`、4 epochs、minibatch 64，且固定 repaired MAPPO + fallback。2026-07-17 完成，update-30 batch success `5/16`，该训练曲线不用于 checkpoint 或泛化选择。

固定 64 环境仅用于预先确定的 checkpoint 筛选；正式验证使用 seed-0 256、seed-1 256、pooled 512 的逐环境配对布局。两套正式布局和 64 筛选集的 per-environment layout hash 均无交集。

## 3. 已实现模块

### 已验证并保留

- repaired MAPPO goal fallback、SAR PPO minibatch indexing 修复。
- 动态 release、检测/assignment event 重决策、staggered commitment、commitment latch、finder cascade。
- 逐环境失败诊断、assignment quality/crossing/finder event 日志。
- default-off 高层模块消融：`three_lane` 搜索、`all_detected` 时机 gate、仅对已发现且未访问/未认领目标的 exact visible matching。
- 覆盖诊断：每 UAV 探索面积、团队覆盖、重叠/重复访问、搜索路程、coverage efficiency、option switches、轨迹连续性。

### 尚未提交的当前实现

- `comm_spread/sar_module_ablation.py`：评估 wrapper、确定性三通道与 exact visible matching。
- `scripts/analyze_sar_failure_modes.py`：默认关闭的 module-ablation/coverage CLI、invalid/nonfinite、rejected-target wait。
- `scripts/report_sar_module_ablation.py`：逐环境 JSON 的配对 2×2×2 汇总。

这些尚未单独审阅/提交；除消融评估外，不应宣称已进入训练或生产执行路径。

## 4. 代码入口和关键文件

- `comm_spread/sar_scenario.py`：环境语义。
- `comm_spread/async_smdp.py`：异步决策、release、cascade、transition。
- `comm_spread/high_level_policy.py` 与 `comm_spread/models/hgsar.py`：高层 actor-critic。
- `comm_spread/low_level_policy.py`：固定低层 checkpoint/fallback。
- `scripts/train_high_ppo_sar.py`：训练 CLI；`scripts/analyze_sar_failure_modes.py`：评估与诊断。
- `scripts/compare_sar_paired_results.py`：paired success/CI；`scripts/report_sar_module_ablation.py`：消融报告。

## 5. 最可靠的运行命令

### 当前正式策略：旧 checkpoint + finder-only

```bash
PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/analyze_sar_failure_modes.py \
  --high-level-checkpoint outputs/sar_high_hgsar_dynamic_staggered_repaired_mappo_sensor_0.60/seed_0/checkpoints/checkpoint_10.pt \
  --low-level-controller checkpoint --enable-low-level-goal-fallback \
  --num-envs 256 --steps 100 --max-steps 100 --sensor-radius 0.6 --seed 0 --device cuda:0 \
  --dynamic-rescue-release --dynamic-rescue-only-after-all-detected \
  --redecide-on-detection-change --redecide-on-assignment-change \
  --dynamic-release-max-new-agents-per-event 1 \
  --enable-finder-first-cascade --finder-cascade-mode finder_only \
  --output <result.json> --csv-output <result.csv> --markdown <result.md>
```

seed 1 时只改 `--seed` 和输出路径；不得改 checkpoint、horizon、传感器、低层或 reset 顺序。完整 2×2×2 消融命令/配置见 `outputs/hierarchical_sar_diagnostics/module_ablation_20260718/experiment_config_and_commands.json`。

## 6. 当前可靠结果

### Finder-only 正式验证（已验证事实）

| 集合 | staggered -> old checkpoint + finder-only | 提升 | candidate-only / baseline-only | 95% CI |
|---|---:|---:|---:|---:|
| seed-0 256 | 66 -> 114 | +18.750pp | 57 / 9 | [12.891, 24.609]pp |
| seed-1 256 | 62 -> 106 | +17.188pp | 50 / 6 | [11.719, 22.656]pp |
| pooled 512 | 128 -> 220 | +17.969pp | 107 / 15 | [14.063, 21.875]pp |

finder-first 重训练 update 30 相对 old+finder 为 `+3/512 = +0.586pp`，95% CI `[-2.344, 3.516]pp`；额外训练收益尚未验证。

### 2×2×2 高层模块消融（已验证事实）

基准 A（旧 checkpoint + finder-only）`220/512 = 42.97%`；完整脚本化 E `328/512 = 64.06%`，总提升 `+21.09pp`，95% CI `[+16.02,+26.17]pp`。

| 单模块 | Success 差异 | 95% CI | 观察 |
|---|---:|---:|---|
| B−A：三通道搜索 | +4.10pp | [−0.59,+8.98]pp | detect-all +13.09pp，覆盖 +.0213，重叠 −.0665 |
| C−A：all-detected 时机 | +1.95pp | [−1.76,+5.66]pp | premature retirement 69.73% -> 0 |
| D−A：精确匹配 | −0.78pp | [−2.15,+0.59]pp | 几何 regret/crossing 降低，但独立 success 无收益 |

单模块之和 `+5.27pp`，总效应 `+21.09pp`，交互残差 `+15.82pp`（95% CI `[+10.16,+21.68]pp`）；已因此补完 F/G/H 双模块 cell。

## 7. 已知问题与未完成任务

### 已验证事实

- A 的主失败是未发现全部目标：225/512；detect-all 后未完成为 67/512。E 将未发现降为 24/512，但 detect-all 后未完成增至 160/512，瓶颈转移到终局救援。
- finder-only A 有 357/512（69.73%）发生 premature retirement；all-detected gate 为 0。
- A pooled accept/reject/unavailable `1236/320/102`；297 个拒绝目标后来被认领，平均等待 34.67 步、P95 72。
- 正式 exact matching cells 的 duplicate、invalid、nonfinite 均为 0。`D_seed0.json` 是 stale target action 的无效审计产物；统计只使用 `D_v2_seed0/1`。

### 观察结果

- 三通道提高 detect-all 且降低重叠；已 detect-all 的条件样本中 all-detected step 更晚，可能是额外纳入困难布局的组成效应。
- exact matching 在 all-detected 稳定终局阶段有条件收益（H−C `+4.69pp`，E−F `+10.55pp`），finder timing 下无独立收益。

### 推测与待验证假设

- 下一瓶颈更可能是学习式搜索和可靠终局 handoff 的耦合，而非单独 assignment 学习。
- 把三通道/gate 的有效交互转化为可训练高层策略仍待验证。

### UNKNOWN

- finder-only 的独立训练 seed-1 复现。
- 多数历史实验运行时的完整、唯一 dirty diff。
- 当前混合工作树的理想 commit 拆分。
