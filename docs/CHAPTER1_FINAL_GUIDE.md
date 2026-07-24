# Chapter 1 Strict-SMDP 最终复现与归档指南

冻结日期：2026-07-24
冻结分支：`research/align-mpe-physics`

本指南记录 Chapter 1 的最终 Strict-SMDP 工程入口、冻结配置和已验收
结果。它不替代严格 SMDP 的实现审计；数学定义、状态/掩码语义及历史模式
边界见 [CHAPTER1_SMDP_AUDIT.md](CHAPTER1_SMDP_AUDIT.md)。冻结的 v1 配置
不得就地修改；新研究应复制为新的版本化配置。

## 1. 工程结构

项目根目录为 `projects/CommSpread/`。

| 位置 | 作用 |
| --- | --- |
| `comm_spread/` | CommSpread 的核心 Python 包：配置解析、SAR 场景、异步 SMDP 采集、回报计算、策略及诊断。 |
| `comm_spread/sar_scenario.py`、`env_factory.py`、`mpe_physics.py` | VMAS/MPE 严格物理环境、场景构建和物理 profile 的实现。 |
| `comm_spread/high_level_policy.py`、`async_smdp.py`、`smdp_returns.py` | 图高层策略、事件触发的异步 option 采集器及严格 SMDP GAE/return。 |
| `comm_spread/low_level_policy.py` | 冻结低层导航策略的网络与接口兼容性支持。 |
| `scripts/train_high_ppo_sar.py` | Chapter 1 高层 Strict-SMDP PPO 训练入口。 |
| `scripts/train_mappo.py` | 低层 MAPPO 训练入口。 |
| `scripts/analyze_sar_failure_modes_mpe_physics.py` | 冻结评估协议的确定性评估和失败诊断入口。 |
| `scripts/select_chapter1_checkpoint_retrospective.py`、`comm_spread/chapter1_retrospective.py` | 仅基于既有 validation records 的 v2 回顾性 checkpoint 选择；不训练也不改 checkpoint。 |
| `configs/chapter1/` | 版本化任务、低层训练、高层训练与评估 YAML。 |
| `tests/` | 单元及配置一致性测试，包括 SMDP return、验证器、health gate 和回顾选择器。 |
| `outputs/` | 本地运行产物：run manifest、日志、checkpoint、validation 与评估报告。该目录被 Git 忽略，不能作为源码提交。 |

## 2. 最终冻结实验配置

| 用途 | 文件 | 冻结要点 |
| --- | --- | --- |
| 任务 | `configs/chapter1/task_final_v1.yaml` | 3 UAV、3 target、sensor radius `0.6`、horizon `100`、`mpe_strict` 物理和五动作离散空间。 |
| 低层训练 | `configs/chapter1/low_train_v1.yaml` | MAPPO/MLP，seed `0`，50M frames，`gamma=0.99`、`gae_lambda=0.95`、学习率 `5e-5`。 |
| 高层训练 | `configs/chapter1/high_train_v1.yaml` | 异步 Strict-SMDP PPO，seed `0`，16 env，200 updates，4 PPO epochs，minibatch 512，`gamma=0.99`、`lambda=0.95`、学习率 `3e-4`。 |
| 基线评估 | `configs/chapter1/eval_v1.yaml` | 冻结的 deterministic 高/低层评估协议模板。 |
| 最终评估 | `<strict-run>/derived_eval_v2.yaml` | 对该 run 的派生、hash 绑定配置：checkpoint_140，seeds `0,1` 各 500 episodes。它是运行产物，不是通用可编辑模板。 |

任务配置 SHA256：
`1ac17a9cb5c0bc7631b0c615c5bc52d90b30573acc983f4742df508bdd0e267c`。

高层训练使用事件触发的 `strict_smdp`：任务成功是训练 terminal；环境可继续
到固定 horizon；成功后的步从训练 storage、return、GAE 和 PPO loss 中排除；
timeout 使用 bootstrap。高层 PPO 的 clip 为 `0.2`、entropy coefficient 为
`0.01`、value coefficient 为 `0.5`、最大梯度范数为 `0.5`。

冻结低层 checkpoint 为：

```
outputs/sar_low_mpe_physics/full_seed_0_20260722/
mappo_sar_low_mlp__0ab21273_26_07_22-09_25_34/checkpoints/checkpoint_50040000.pt
```

其 SHA256 为
`4cc80182f328238eb59c2562ad0483710499061674dbcff9dedeca9319594029`，并要求
`sar_low_mpe_physics` 与 `mpe_strict` runtime profile。高层配置会在启动前
校验此依赖。

## 3. 如何重新训练 Chapter 1

以下命令用于开始一个**新的**严格 run；不要在已经存在的 run 目录上执行。容器
路径可能因部署不同而变化，示例使用项目在容器中的通常挂载路径。

```bash
docker exec -it marl-vmas-dev bash
cd /workspace/projects/CommSpread
export PYTHONPATH=/workspace/projects/CommSpread
RUN_ID="chapter1-strict-smdp-seed0-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C /workspace rev-parse --short HEAD)"
test ! -e "outputs/chapter1/high_train_v1/seed_0/runs/$RUN_ID"
python scripts/train_high_ppo_sar.py \
  --chapter1-config configs/chapter1/high_train_v1.yaml \
  --run-id "$RUN_ID"
```

`--run-id` 必须唯一。严格 fresh run 要求输出目录不存在，随机初始化高层，且禁止
加载旧高层训练状态；不要通过复用 run-id 创建“新”训练。高层训练依赖上一节列出的
冻结低层 checkpoint。checkpoint、manifest 和日志均写入：

```
outputs/chapter1/high_train_v1/seed_0/runs/<run-id>/
```

可通过 `scalars/train.csv`、`texts/run_summary.txt`、`validation/` 和
`health_decisions.jsonl` 观察训练状态。训练过程会在 update 0 和每 10 update
进行固定验证，并记录健康门决策；不要手工替换其中的 checkpoint 或 manifest。

## 4. Checkpoint 与输出结果

一个高层 run 下的主要内容如下。

| 位置/文件 | 含义 |
| --- | --- |
| `checkpoints/checkpoint_<update>.pt` | 完整 update 边界保存的高层模型与训练状态。 |
| `checkpoints/latest.pt` | 最新保存点；它不是最终评估 checkpoint 的默认选择依据。 |
| `validation/update_<update>/summary.json` | 固定 seed/layout 的确定性 validation 汇总；逐 episode 记录在相邻的 `per_episode.jsonl`。 |
| `run_manifest.json`、`resolved_config.json` | 任务/配置 hash、低层依赖、初始化来源和解析后的配置；run-id 由其 run 目录名标识。 |
| `health_decisions.jsonl` | 训练时 health gate 的决策记录。 |
| `health_gate_v2_retrospective.json` | v2 回顾选择的证据；仅从既有 validation 与训练诊断派生。 |
| `derived_eval_v2.yaml` | 绑定已选 checkpoint 和 SHA256 的最终 1000-episode 评估配置。 |
| `evaluation_v2_1000/` | evaluation manifest、每个 seed 的逐 episode 结果、汇总及最终 Markdown 报告。 |

应从 `health_gate_v2_retrospective.json` 的 `selected_checkpoint` 找到最终高层
checkpoint，并对文件重新计算 SHA256。最终报告位于
`evaluation_v2_1000/chapter1_evaluation_report.md`，其 `evaluation_manifest.json`
和 completion manifest 保留评估配置、输入 hash 和产物 hash。

## 5. 参数修改指南

冻结 v1 不是试验草稿。请复制 YAML 并建立新的任务/实验版本，而不是直接修改
`*_v1.yaml`。

- **仅改变 evaluation 参数**：复制并修改 eval 配置即可；仍须显式绑定 task、
  高层 checkpoint 和低层 checkpoint 的 SHA256，不能把新评估写回冻结结果目录。
- **改变任务定义**（target 数量、sensor radius、horizon、physics 或任务语义）：
  复制 `task_final_v1.yaml`，更新 task version，并创建与之对应的低/高层训练和
  eval 配置；任务 hash 改变后必须重新训练高层。
- **改变 UAV 数量**：除上述任务重训外，必须检查低层 observation 输入宽度。当前
  3-UAV 接口宽度为 8，其中其他 UAV 相对位置占 4 维；数量变化通常使该维度和
  checkpoint 兼容性失效，因此需要重新训练低层，随后再训练高层。

## 6. Chapter 1 最终冻结结果

最终严格 run 使用回顾性 health gate v2 选择的 `checkpoint_140.pt`，而非
`latest.pt`。高层 checkpoint SHA256：

`ccefb97872e904141f61f04494a31c316fe93041e6a0d8537703a5dd72c10460`。

使用该 run 的 `derived_eval_v2.yaml`，在严格冻结 task、冻结低层 checkpoint、
确定性高/低层策略下，对 seeds 0、1 各运行 500 episode（共 1000）得到：

| 指标 | 最终结果 |
| --- | ---: |
| Success rate | 90.4%（904/1000） |
| Detect-all rate | 93.3%（933/1000） |
| 平均成功步数 | 40.135 |
| 未发现全部目标 | 67（6.7%） |
| 发现过晚 | 20（2.0%） |
| 发现后过度探索 | 3（0.3%） |
| 残余低层执行失败 | 6（0.6%） |
| assignment conflict / invalid / nonfinite | 0 / 0 / 0 |

该结果是 Chapter 1 的最终冻结结果。完整 failure、assignment、crossing 和 regret
诊断以 `evaluation_v2_1000/` 中的 manifest、pooled summary 与最终报告为准；这些
实验产物刻意不进入 Git。
