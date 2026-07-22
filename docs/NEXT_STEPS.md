# CommSpread 下一步实验计划

更新日期：2026-07-22 UTC。所有 P0–P3 固定：3 UAV、3 targets、100 steps、radius .6、`retire_on_rescue=True`、repaired MAPPO + fallback、旧 checkpoint10、同一 reset 顺序、相同 seed-0/seed-1 独立 256 布局；不改 reward、网络、物理或成功判定。每项只验证一个主要假设。

## P0 — 对 E28 结果做纯离线逐环境归因（不改策略代码）

- **单一假设**：搜索改进的 detect-all 收益不是仅由条件样本构成变化造成，且可用 coverage/trajectory 指标解释。
- **动作**：用 A/B/E/F/G/H 的逐环境 JSON、coverage CSV、paired CSV，按 success、detect-all、failure category 分层；计算 coverage/overlap/efficiency、option switching、detection step 与 completion 的 paired 差异/相关。
- **成功判据**：能说明三通道的增益主要在哪类布局/失败转移发生，并报告条件化与无条件化结果；不把相关写成因果。
- **失败判据**：现有字段无法支持分层或结论完全由少数环境驱动；则明确记录不可判定。
- **预计文件**：新增离线分析脚本和 `outputs/.../module_ablation_20260718/` 报告附件；不改环境/策略代码。
- **不要同时改变**：任何 policy、checkpoint、布局、评估器语义。

## P1 — 可训练的搜索—终局 handoff 最小单变量原型

- **单一假设**：若保持 finder-only 执行、只向高层提供一个可观测的终局阶段信号/训练约束，可在不替换搜索和分配的前提下降低 premature retirement 与 post-detection over-exploration。
- **前置条件**：先完成 P0，并在代码审阅中确定一个唯一、可观察、默认关闭的实现方案；当前具体实现为 **UNKNOWN**，不得同时加入三通道、精确 matching 或 reward shaping。
- **成功判据**：固定 64 checkpoint/策略 smoke 无 invalid/nonfinite；随后 paired 256 的 success/detect-all 不低于 A，且 premature retirement 或 post-detection over-exploration 明确下降。
- **失败判据**：只降低 retirement 而 success 无提升、引入 search starvation、或 CI 不能排除有害方向。
- **预计文件**：可能涉及 `async_smdp.py`、`high_level_policy.py`、`train_high_ppo_sar.py`、`analyze_sar_failure_modes.py`；开始前必须单独设计并获确认。
- **不要同时改变**：reward、低层、RRT 参数、assignment matching、finder mode、layout/horizon。

## P2 — 训练 seed-1 复现 finder-only（仅在需要复核训练结论时）

- **单一假设**：finder-only 重训练没有可靠额外收益这一结论可在独立训练 seed 复现。
- **动作**：完整复制 E26，仅训练 seed 改为 1；固定 64 选 checkpoint，再在独立 256 配对评估 old+finder vs trained+finder。
- **成功判据**：若额外训练效应在 seed1 和 pooled 上有正 CI，才重开“retraining”路线；若仍 CI 跨零/为负，正式排除其额外收益。
- **失败判据**：训练不稳定、配置与 E26 不可比、或未按预注册 checkpoint 规则。
- **预计文件**：无算法文件；新增 `outputs/sar_high_hgsar_finder_first_finder_only_repaired_mappo_sensor_0.60/seed_1/` 及评估结果。
- **不要同时改变**：cascade mode、网络、PPO、reward、动态 gate、低层、评估布局。

## P3 — 精确 matching 的终局阶段诊断/约束（仅在 P1 显示稳定终局后）

- **单一假设**：exact matching 的条件收益来自 all-detected 后的分配质量，而不是 hidden-information 泄漏或评价器副作用。
- **动作**：固定搜索和 timing；只比较 commitment 与 exact visible matching，使用全部已发现目标、active/current-decision UAV，保留 assignment/crossing/regret/visit delay。
- **成功判据**：paired 256 中 assignment 的正效应 CI 不跨零，duplicate/invalid/nonfinite 保持 0，且不增加未发现失败。
- **失败判据**：收益只在脚本化 oracle 中存在，或依赖改变搜索/时机；则不进入学习式策略。
- **预计文件**：优先评估/离线脚本；若 P1 后才需修改，可能为 `sar_module_ablation.py` 和 evaluator，默认关闭。
- **不要同时改变**：搜索模块、all-detected gate、reward、低层、任务数量。

## P4 — 将三通道视为参考，而非立即替换部署策略

- **单一假设**：三通道的低 overlap/高 detect-all 可被可训练搜索表征或目标选择机制部分复现。
- **动作**：仅在 P0/P1 完成后，定义一个单一训练变量（例如搜索 option continuity 或可观察 coverage feature）；先 64 smoke，再 paired 256。
- **成功判据**：detect-all/coverage 改善并有 success 正向证据，且不同时偷偷改变 timing/assignment。
- **失败判据**：仅复现覆盖而不改善 success，或造成低层 residual failure 增加。
- **预计文件**：目前具体文件为 **UNKNOWN**；需设计确认后再写。
- **不要同时改变**：finder-only、all-detected gate、exact matching、reward shaping。

## P5 — 归档与提交

- **单一假设**：结果可以从独立 commit、配置和产物复核。
- **动作**：先提交文档交接；再单独审阅/提交消融实现与报告脚本；不混入旧报告、`old_algo/`、外部项目目录或 `outputs/` 大量产物。
- **成功判据**：每个 commit 的主题单一、`git diff --check` 通过，docs 记录准确 HEAD/dirty 状态/产物路径。
- **失败判据**：把算法、旧实验、输出和文档混在一个不可审计 commit。
- **预计文件**：本轮仅 `docs/PROJECT_STATUS.md`、`docs/EXPERIMENT_LOG.md`、`docs/DECISIONS.md`、`docs/NEXT_STEPS.md`。
- **不要同时改变**：任何算法、训练、评估设置。
