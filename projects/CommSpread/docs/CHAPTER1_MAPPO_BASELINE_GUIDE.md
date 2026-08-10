# Chapter 1 端到端 MAPPO 基线迁移与实验指南

## 1. 基线定义与旧实现审计

旧代码位于 `old_algo/my_marl_transfer-base-MAPPO/my_marl_transfer-base-MAPPO`。其评估入口
`eval_baseline.py` 每个 MPE 环境步调用一次共享策略，不经过高层 option 或低层导航器。

旧实现的有效算法结构为：

- 3 架 UAV 共享一个 actor 参数集，直接输出 `noop/left/right/down/up` 五类物理动作；
- `mpnn.py` 先编码 ego 状态，并在 `--entity-mp` 下对目标实体进行注意力聚合；
- UAV 之间执行 3 轮消息传递，`--mask-dist 2` 屏蔽通信距离外的 UAV；
- `--mask-obs-dist 0.5` 屏蔽观测半径外的目标实体；
- `learner.py` 使用 PPO 更新共享策略。虽然工程名称为 MAPPO，其中的 direct-policy trainer
  类名为 `IPPO`；迁移版采用共享 actor 和 centralized observable-state critic，明确实现 MAPPO；
- 旧评估指定的 `../marlsave/save_new/0807onlyLM32/ep8400.pt` 不在仓库中。仓库内其他
  `.pt` 文件与该模型、当前 Strict-MPE 状态和任务状态机均不兼容，因此不能作为迁移权重使用。

旧仓库经过多轮实验性修改，当前版本甚至存在环境 step reward 被置零的代码路径。迁移只保留可审计的
算法结构，不复刻这些不一致状态。

## 2. 当前 VMAS 迁移版的公平性边界

迁移版为真正的端到端对照：

- actor 每个环境步直接选择五类离散物理动作；
- 不使用 RRT、option、SMDP return、FCSR、finder-first、commitment、assigned goal 或低层 checkpoint；
- actor 使用 ego 位置/速度/active/time、通信半径内的 UAV 消息，以及团队已经持久化检测且尚未救援的目标质心；
- actor 与 centralized critic 均不能读取未检测目标的真实位置；
- 目标被团队检测后，活动 UAV 物理进入 rescue radius 即可救援，随后沿用 frozen task 的 UAV retirement；
- direct 模式取消随机 assigned-goal 距离奖励；公平强基线使用与 HIG-SAR 完全相同的基础环境奖励，只额外加入明确声明的一对一 assignment-progress shaping；
- 物理、出生分布、Bayesian belief、检测阈值、100 步 horizon、退休规则和五指标均来自相同 Chapter 1 task YAML。

这意味着旧命令中的 `mask-obs-dist` 不再是第二套几何真值过滤器。观测半径由 frozen task 的
`perception.sensor_radius` 决定，目标只有通过当前 Bayesian detection 状态机后才会进入策略输入，
并按任务定义持续可见至救援。通信半径仍为旧基线的 `2.0`。

## 3. 配置文件

旧 sparse-v1 模板分别为：

- `configs/chapter1/baseline_mappo_r030_seed0_v1.yaml`
- `configs/chapter1/baseline_mappo_r040_seed0_v1.yaml`
- `configs/chapter1/baseline_mappo_r050_seed0_v1.yaml`

历史 dense-v2 实验使用以下隔离模板（保留用于复现，不再作为公平主对比）：

- `configs/chapter1/baseline_mappo_reward_v2_r030_seed0_v1.yaml`
- `configs/chapter1/baseline_mappo_reward_v2_r040_seed0_v1.yaml`
- `configs/chapter1/baseline_mappo_reward_v2_r050_seed0_v1.yaml`

三个历史 dense-v2 模板的奖励、网络和 PPO 参数完全相同，只改变 frozen task 的 sensor radius。
它们分别引用现有的 `task_immediate_open_r030/r040/r050_v1.yaml`。这里任务文件中的
`immediate_open` 只用于复用完全相同的环境参数；端到端 MAPPO 不调用该 FCSR 机制。

常用修改项：

- 训练 seed：`training.seed`
- GPU/CPU：`training.device` 和 `evaluation.device`
- 训练量：`training.num_envs`、`rollout_steps`、`updates`
- PPO 参数：`learning_rate`、`gamma`、`gae_lambda`、`clip_epsilon`
- 正式评估量：`evaluation.episodes`，默认 500
- actor 通信半径：`baseline.model.communication_radius`，基线固定值为 2.0

不要通过修改 baseline YAML 的数字来改变 sensor radius；应改用对应半径的 frozen task 引用。

## 4. 正式训练

以下命令均从 `projects/CommSpread` 目录运行。半径 0.3：

```bash
PYTHONPATH=. python scripts/train_mappo_baseline_sar.py \
  --config configs/chapter1/baseline_mappo_r030_seed0_v1.yaml
```

0.4 和 0.5 只需分别换为 `r040`、`r050` 配置。输出默认位于：

```text
outputs/chapter1_baselines/mappo/r030/seed0/
  checkpoint_latest.pt
  checkpoint_000010.pt
  ...
  training_history.json
  resolved_baseline_config.json
```

训练 checkpoint 带 task hash、信息边界和 baseline 类型元数据。评估会拒绝 task hash 不匹配的
checkpoint。若需要多个训练 seed，应复制模板并同时修改文件名、`training.seed` 和
`training.output_dir`，不要让不同 seed 覆盖同一目录。

## 5. Checkpoint 选择与评估

当前基线不套用 Chapter 1 高层策略的 checkpoint finalizer。先根据
`training_history.json` 和独立 validation 结果选定 checkpoint，再把准确路径显式传给 evaluator。
不能用最终测试 seed 反向选择 checkpoint。

以半径 0.3、评估 seed 0 为例：

```bash
PYTHONPATH=. python scripts/evaluate_mappo_baseline_sar.py \
  --config configs/chapter1/baseline_mappo_r030_seed0_v1.yaml \
  --checkpoint outputs/chapter1_baselines/mappo/r030/seed0/checkpoint_000200.pt \
  --seed 0 \
  --output-dir outputs/chapter1_baselines/mappo/r030/eval_seed0
```

对同一 checkpoint 运行 seed 1：

```bash
PYTHONPATH=. python scripts/evaluate_mappo_baseline_sar.py \
  --config configs/chapter1/baseline_mappo_r030_seed0_v1.yaml \
  --checkpoint outputs/chapter1_baselines/mappo/r030/seed0/checkpoint_000200.pt \
  --seed 1 \
  --output-dir outputs/chapter1_baselines/mappo/r030/eval_seed1
```

配置默认每个 seed 评估 500 回合。为复现主方法的 paired layout stream，评估强制
`batch_size == episodes`，正式模板两者均为 500。可用 `--episodes`、`--batch-size`、`--device`
作运行时覆盖，但前两者必须一起改为相同值。正式论文实验应保持与主方法相同的 seed、回合数和半径；
`--episodes 4 --batch-size 4` 仅用于烟雾测试。

每次评估产生：

- `evaluation_metrics.json`：固定五指标汇总；
- `evaluation_episodes.json`：每回合检测/救援时刻与 101 点 entropy trace；
- `policy_diagnostics.json`：动作占比、noop、策略熵、logit margin 及发现/救援覆盖率；
- 终端同步打印五指标。

固定五指标为 success rate、成功回合的 `T_all`、成功回合的平均
`mean_delta_t_res`、0 至 100 步的 `eta_search` 曲线，以及检测全部目标回合的
`T_det_last`。标量按现有协议输出 mean、sample standard deviation 和样本数。

## 6. 已完成的非正式验证

迁移实现已通过以下范围的检查：

- 信息边界测试：移动未检测目标真值不会改变 actor target 输入；
- direct rescue 与 retirement 状态机测试；
- actor/action/value shape 和 inactive UAV noop 测试；
- terminal-aware GAE 测试；
- 0.3/0.4/0.5 配置与 frozen task 半径一致性测试；
- 2 个 CPU 环境、8 步、1 次 PPO update 的 smoke training；
- smoke checkpoint 的 4 回合、完整 100 步评估和五指标 JSON 导出。

这些结果只证明代码路径可运行，不是可报告的基线性能，不应与正式训练结果混用。

## 7. 三半径连续训练、选模与评估

建议先运行不会启动训练的预检：

```bash
PYTHONPATH=. python scripts/run_mappo_baseline_radius_sweep.py \
  --dry-run \
  --device cuda:0
```

正式执行：

```bash
PYTHONPATH=. python scripts/run_mappo_baseline_radius_sweep.py \
  --batch-id mappo-radius-sweep-seed0-v1 \
  --device cuda:0
```

脚本固定按 0.3、0.4、0.5 顺序处理。每个半径包括：

1. 使用对应 baseline YAML 完成 200 update 训练；
2. 对 update 100 至 200、间隔 10 的 11 个 checkpoint，使用独立 seed 2 评估 100 回合；
3. 按 success rate 降序、`T_all` 升序、救援响应时间升序、update 降序选模；
4. 对选中 checkpoint 运行 seed 0 和 seed 1 各 500 回合正式评估；
5. 合并两个 `evaluation_episodes.json` 的 1000 条原始回合，重新计算 pooled 五指标。

validation 不使用最终 seed 0/1。时间指标为 null 时按正无穷参与排序；如果所有排序指标相同，
选择较晚 update。任一 validation 候选失败都会使该半径选模失败，不会使用不完整候选集合。

训练输出位于：

```text
outputs/chapter1_baselines/mappo/r<radius>/seed0/runs/<run-id>/
  checkpoint_*.pt
  training_history.json
  checkpoint_selection.json
  validation/seed_2/checkpoint_*/
  evaluation_final_1000/seed_0/
  evaluation_final_1000/seed_1/
  evaluation_final_1000/evaluation_metrics_pooled.json
```

流水线状态和日志位于：

```text
outputs/chapter1_baselines/mappo/radius_sweep/<batch-id>/
  pipeline.lock
  pipeline_state.json
  pipeline_summary.json
  logs/
```

以相同 `--batch-id` 重新运行时，脚本会重新验证已成功阶段的产物并跳过仍有效的部分。
训练阶段中断或失败后会建立新的 attempt，避免把不完整 checkpoint 混入后续选模。
某个半径失败不会阻止后续半径；最终退出码 0 表示全部成功，1 表示完成但存在失败，
2 表示配置、锁或其他流程级错误。

每个半径结束发送 `0.3success`、`0.3fail` 等 Bark 通知；整个流程最终始终请求
`train_done`。可通过 `BARK_BASE_URL` 环境变量或 `--bark-base-url` 修改通知地址。
通知发送失败只记录 warning，不改变训练、评估状态。

## 8. 历史 dense-v2 半径对比

旧命令默认仍使用 sparse-v1，以保证历史批次可以恢复。复现 dense-v2 三半径实验必须显式指定实验集：

```bash
PYTHONPATH=. python scripts/run_mappo_baseline_radius_sweep.py \
  --experiment-set dense-v2 \
  --radii 030,040,050 \
  --batch-id mappo-reward-v2-r030-r050-seed0-v1 \
  --device cuda:0
```

dense-v2 固定 discovery scale 1.0、detected-target progress scale 5.0、collision/floor −0.2、
boundary −0.05、active time penalty 0.01 和 PPO entropy coefficient 0.001。不得针对单个半径
改变这些参数，否则会把观测半径效应与奖励调参混合。

三个半径全部成功后，批次目录额外生成 `radius_comparison.json`，其中包含每个半径的配置哈希、
选中 checkpoint、1000 回合固定五指标完整数据和 seed 0/1 合并后的策略诊断。最终测试 seed 不参与选模。

## 9. 公平强 MAPPO：HIG-SAR 对齐基础奖励 + assignment prior

论文主对比使用以下配置，而不是 dense-v2：

- `configs/chapter1/baseline_mappo_assignment_prior_r030_seed0_v1.yaml`
- `configs/chapter1/baseline_mappo_assignment_prior_r040_seed0_v1.yaml`
- `configs/chapter1/baseline_mappo_assignment_prior_r050_seed0_v1.yaml`

profile `higsar_aligned_assignment_v1` 使用按救援顺序递增的 20/40/60 奖励、
collision coefficient/floor −0.2、boundary −0.05 和 active time penalty 0.01。
它不启用 low-level goal reward，也不启用 dense-v2 的 nearest-detected-target progress。
唯一额外奖励为 `r_assignment(i,t) = 5 * (d_previous(i,t) - d_current(i,t))`。

每一步仅对转换开始前已被团队检测且尚未救援的目标和 active UAV 做确定性最小总距离
一对一匹配。未匹配 UAV 奖励为 0；相同总代价按 UAV/目标编号的字典序稳定打破平局。
新发现目标到下一环境步才可参与匹配，未检测目标真值不参与匹配或奖励。

这是公开声明的人工任务分配先验。策略仍逐步直接输出五类物理动作，自行学习搜索与趋近；
不使用 RRT、option、assigned goal、低层导航器、FCSR、finder-first、commitment 或动态释放。
救援采用 automatic proximity rescue：任一 active UAV 进入已检测目标的 rescue radius 即
完成救援并按 frozen task 退休。checkpoint 与 evaluation JSON 均记录 shaping 和 rescue 语义。

预检命令：

```bash
PYTHONPATH=. python scripts/run_mappo_baseline_radius_sweep.py \\
  --experiment-set assignment-prior \\
  --radii 030,040,050 \\
  --batch-id mappo-assignment-prior-tuned-reward-seed0-v1 \\
  --device cuda:0 --dry-run
```

移除 `--dry-run` 即执行连续训练、validation 选模、seed 0/1 各 500 回合评估和 pooled 五指标。
结果写入 `outputs/chapter1_baselines/mappo_assignment_prior/r<radius>/seed0/runs/`。
三个半径必须共用相同 assignment 系数与 PPO 参数，不得按半径调奖励。
