# 第一章最终 SAR 任务规范

版本：`chapter1_sar_final_v1`
唯一任务源：`projects/CommSpread/configs/chapter1/task_final_v1.yaml`
冻结 task SHA256：`1ac17a9cb5c0bc7631b0c615c5bc52d90b30573acc983f4742df508bdd0e267c`

## 1. 冻结范围与证据层级

本规范冻结第一章后续低层训练、高层训练、验证和正式评估共同使用的任务定义。任务参数只在 `task_final_v1.yaml` 中出现；`low_train_v1.yaml`、`high_train_v1.yaml` 和 `eval_v1.yaml` 通过 `task_ref` 引用它。代码默认值不是规范源，入口必须先解析 YAML，再保存 `resolved_config.json`。

当前代码证据包括：场景构造与状态张量 `sar_scenario.py:37-373`，reset/动作/观测/done `sar_scenario.py:375-475`，reward 与 rescue 状态机 `sar_scenario.py:704-913`，共享 Bayesian belief `sar_scenario.py:914-1058`、`team_belief.py:27-138`，异步事件执行 `async_smdp.py:207-269,680-1033`，以及 strict profile guard `env_factory.py:102-117,147-182`。

仓库状态文档仍把旧 legacy pipeline 的 `220/512 = 42.97%` 视为正式主结果；更新的跨平台报告记录当前联合 MPE-aligned pipeline 为 `420/512 = 82.03%`。本规范采用后者实际执行语义，但不把联合提升解释成纯物理收益。

## 2. 环境、物理与动作

| 项目 | 冻结值 |
|---|---|
| Scenario | `comm_spread_sar` / `SarScenario` |
| UAV / target | 3 / 3 |
| Horizon | 100 world steps |
| World | x、y semidim 均为 1.0；belief world size 2.0 |
| Physics profile | `mpe_strict` |
| dt / substeps | 0.1 / 1 |
| Damping | drag 0.25；线性/角摩擦 0 |
| Max speed | 未配置，不限速 |
| Action | 离散 5 动作：停、左、右、下、上 |
| Force sensitivity | 5.0 |
| Radius | UAV 0.05；target 0.05 |
| Agent collision | 物理碰撞开启；soft-contact force 100，margin 0.001 |
| Collision reward | safe distance 0.15，系数 -20，单 pair floor -20 |
| Target collision | 关闭，target 不可移动 |
| Boundary | 无墙、无 clamp；UAV 中心 `abs(pos)>=1` 时 -2 |

`mpe_strict` 使用 `MPEParityWorld` 和 MPE soft-contact 方程（`mpe_physics.py:9-54`）。第一步与 MPE 的离散作用力、积分和阻尼已由跨引擎探针对齐。`mpe_strict + continuous_actions=True` 是非法组合，环境工厂和配置解析均立即报错。

## 3. 初始分布与 seed

六个实体按 `[agent_0..2,target_0..2]` 顺序做联合均匀拒绝采样，x/y 均为 `[-1,1]`。所有 entity pair 的中心最小距离为 `2*0.05+0.05=0.15`，因此 agent-agent、target-target、agent-target 的最小距离相同。0.15 小于传感半径 0.6，允许 target 初始处于一个或多个 UAV 的 FOV 内。

`make_env(seed=...)` 调用 VMAS 1.5.2 的环境局部 RNG；同一 seed、batch 大小和 reset 序列确定布局。训练、验证和测试使用同一分布，只用显式不同 seed 区分样本；不引入更窄的训练出生区或更容易的评估出生区。代码入口为 `sar_scenario.py:375-386`。

## 4. 观测、通信、belief 与检测

传感半径固定为 0.6 个世界欧氏距离单位。100×100 grid 覆盖 `[-1,1]^2`，cell size 0.02，初值 0.5，sensor fidelity 0.8，检测阈值为严格 `p>0.95`。联合 FOV 是所有 UAV 圆形 FOV 的并集；默认包含 inactive/retired UAV。

对 FOV 内 occupied cell：

`p' = 0.8p / (0.8p + 0.2(1-p))`

对 FOV 内 empty cell：

`p' = 0.2p / (0.2p + 0.8(1-p))`

reset 在第一次高层决策前先更新一次。持续正观测从 0.5 依次成为约 0.8、0.941176、0.984615，因此需要三次 Bayesian 更新（包括 reset update）才越过阈值。阈值 cell 做 8 邻域聚类，cluster centroid 转回世界坐标。检测后位置和 target identity 全队共享并持久化；离开 FOV 不清除，rescue 后才从有效高层 target nodes 移除（`sar_scenario.py:914-1058,1061-1084`）。

这是集中式、无限带宽的 team-shared belief/通信假设。环境内部用真值 occupancy 生成传感 likelihood，并用真实 target footprint 将 cluster 关联到 target identity；这是真实传感器模拟和 bookkeeping，不直接给 actor 未发现 target 坐标。

## 5. Actor 与 critic 信息边界

高层 actor 输入：

- ego `[absolute position(2), velocity(2), remaining-horizon fraction(1)]`；
- active teammates 的 `[absolute position(2), velocity(2), assigned-goal distance(1)]`，排除 self；
- RRT search nodes `[relative xy, entropy utility, occupied]`；
- 未 rescue 的、已检测 target nodes `[relative centroid xy, utility, claimed]`；
- 相对 edge features 和合法动作 mask。

当前冻结 actor 不启用 progress、assignment、commitment-aware、phase-policy 或 intention-coordination 扩展。actor 不消费 map tensor；belief 通过 RRT candidates 和 detected target nodes 影响 actor（`async_smdp.py:831-1033`、`high_level_policy.py:88-139`）。

高层 critic 是 centralized map critic，输入全队 `[belief entropy, belief probability, detected-target heatmap]` 和所有 UAV nodes；inactive UAV 仍在 global agent nodes 中（`high_level_policy.py:591-598`）。VMAS critic 的 target heatmap来自持久检测坐标，不是隐藏 target 真值。因此当前 VMAS 不复现旧 MPE critic 的 privileged/stale landmark heatmap 泄漏。

低层 actor 输入固定 8 维：ego velocity 2、`goal-ego position` 2、另外两架 UAV 的相对位置 4；无 target/landmark slot（`sar_scenario.py:444-467`）。MAPPO critic 可集中处理各 agent 的这些低层观测，但同样没有隐藏 target 坐标。

## 6. Discover / rescue / retire 状态机

`undiscovered -> detected -> visited(rescued)` 是三个持久状态。Discover 由 Bayesian threshold 触发；rescue 不是 detect 的同义词。

rescue 要同时满足：UAV 当前是 active、option 是 target task、UAV 到 assigned detected centroid 的距离不超过 0.05、assigned centroid 能在 0.05 内匹配一个真实 target、该 target 对团队已检测且尚未 visited。成功后 `target_visited` 永久置位；若 `retire_on_rescue=True`，执行 rescue 的 UAV 立即 inactive（`sar_scenario.py:881-913`）。

退休 UAV：动作强制为零；不再获得后续 active reward、不参加 collision reward pair、不参加高层决策，也不作为 actor active teammate；但物理 collision body 仍存在、仍可被碰撞推动，且固定为 `belief_include_inactive_agents=True` 因而继续感知；critic global nodes 仍保留它。

Success 精确定义为三个 `target_visited` 全为真。`done_when_all_targets_visited=False`，所以 success 不 early done，成功和失败 episode 均执行到 horizon 100。正式 failure label 为 `undiscovered_remaining`、`detected_unvisited_remaining`、`no_active_agents_for_search`、`no_active_agents_for_detected_target` 或 `other_timeout`（`sar_scenario.py:469-473,704-747`；`analyze_sar_failure_modes.py:1035-1045`）。

## 7. 高层 option 与事件触发

每次动作在 5 个 RRT search nodes 和 3 个 detected target nodes 中选择。RRT 固定 top-k 5、最多 40 iterations、seed 0；基于 team belief entropy、active-agent Voronoi 和既有 public goals，未读取隐藏 target 坐标。完全相同 xy 候选只保留最高 score。

`AsyncSMDPCollector` 不按固定 5-step interval 决策。option 在以下事件结束/重决策：low-level goal 达到阈值、agent inactive、episode done；此外 detection set change 和 rescue commitment/assignment change 触发未 commitment active agents 重决策（`async_smdp.py:207-269`）。

Rescue release 为 event-driven staggered gate：只有全部 targets detected 后才开放新的 rescue commitment；每个事件最多新释放 1 架 UAV。动态 gate 的其余参数仍显式冻结（min searchers 2、entropy ratio 0.58、entropy rate 0.0015、stagnation step 25、22 search steps/target、assumed speed 0.035、margin 8），但 `only_after_all_targets_detected=True` 使未全检测阶段不释放 rescue。

Finder-first 开启且 mode 为 `finder_only`：新 target 的最近 active sensor contributor 首先获得处理该 target 的决策权；finder 拒绝时不立即扩散，直到 terminal all-detected 规则或 finder unavailable。已有 rescue assignment 构成 commitment latch；commitment agent 不因别处 detection/assignment change 被重新分配，只在 option completion、retirement 或 episode done 释放。全部检测且该 agent 有 rescue action 时，search actions 被 mask。合法集设计应始终非空；网络防御性 fallback 是重新开放第一个 RRT explore node。

## 8. 低层接口和 checkpoint 契约

低层 variant 固定 `sar_low_mpe_physics`，离散 5 动作，goal 为世界绝对 xy（进入 observation 时变为相对坐标），goal threshold 0.05。推理为 deterministic。

Goal-stagnation fallback 开启：距离 ≤0.25 或连续 5 步未改善至少 0.005 后 latch；新 goal 清 latch。strict profile 下 fallback 不是比例连续控制，而是沿绝对误差最大轴选择 cardinal action（`low_level_policy.py:127-192`）。运行时 profile guard 已在 `low_level_policy.py:76-83` 强制执行。

新低层训练完成后，每个 checkpoint sidecar 必须记录：task version、task SHA256、physics profile、low variant、observation width、action kind/cardinality、goal threshold 和 checkpoint SHA256。当前 82.03% 使用的 seed-0 checkpoint 早于本契约，没有 sidecar；它仅以固定文件 SHA256加 runtime profile guard 作为显式 legacy 例外，不应推广给新 checkpoint。

## 9. 奖励

低层 agent reward 包含 distance progress、goal reward 3、rescue reward、collision/boundary penalties；无 time penalty。高层 SMDP reward 包含 entropy/discovery、rescue 和显式为零的 shaping 项。Rescue base 为 10，并按 rescue order 乘 1、2、3。所有为零的实验 shaping 参数仍在 task YAML 显式出现，以防默认值漂移。

## 10. 与历史任务的区别

### 历史 MPE `simple_spread`

历史 MPE 不是本 SAR 任务的同义实现。MPE raw obs 虽含 landmark slots，但高/低层裁剪它们；高层通过 team-shared Bayesian belief 获得已检测 landmark centroid。历史评估用 sensor radius 0.5、训练用 0.3，而本任务统一为 0.6。更重要的是，本任务有显式 discover/rescue/retire、target visited、异步 option、staggered release 和 100-step SAR success；不能将 MPE 的 87.4% 与本任务成功率直接解释为同任务算法差异。

### Legacy VMAS SAR

Legacy VMAS 使用 continuous `[-1,1]^2`、5 physics substeps、默认 force multiplier 1、hard boundary clamp、较短第一步位移（0.006）和不同 collision/boundary reward。本任务使用 strict MPE recurrence：离散动作、1 substep、sensitivity 5、无 clamp，第一步 cardinal displacement 0.05。Legacy physics 在 final config 下被硬拒绝。

## 11. 冻结核心与训练超参数

冻结核心是 `task_final_v1.yaml` 的全部内容：实体数、horizon、物理/动作、spawn、belief/detection、reward、状态机、信息边界、RRT/option/event、release/finder/commitment 和低层接口。任何核心字段变化都必须创建新的 task version，不能覆盖 v1。

训练超参数只存在于 `low_train_v1.yaml` 或 `high_train_v1.yaml`：frame/update 数、PPO batch/epoch、学习率、设备、seed、checkpoint interval、模型训练扩展等。评估样本数、seed 集、checkpoint 和输出协议只存在于 `eval_v1.yaml`。改变训练超参数不改变 task hash；改变 task 核心必须改变 task hash。

## 12. 解析、指纹与非法组合

`comm_spread.chapter1_config` 使用 key-sorted compact JSON 作为 canonical serialization；float 以 `.17g` tagged decimal 固定表示。`task_config_sha256` 只哈希完整 task mapping。完整 role `config_sha256` 还包含训练/评估协议，但递归排除 output/save folder、timestamp、command 等 volatile fields。

入口保存 `<output_dir>/resolved_config.json`，其中并列保存原 YAML、程序默认值、CLI overrides、最终 resolved values 和两个 hash，并打印 task version、physics profile、agent/target 数、horizon、sensor radius、retire、event replanning、finder-first 和 hash。

以下情况立即失败：strict + continuous；final task + legacy；role/task version 不同；eval CLI 修改核心任务参数；formal eval seed/episode 数不在协议内；checkpoint SHA 不符；low runtime profile 与 checkpoint variant profile 不符。

## 13. 与 82.03% 配置的一致性及限定

`eval_v1.yaml` 的行为配置与 420/512 评估一致：相同旧 high checkpoint、strict seed-0 low checkpoint + fallback、HGSAR deterministic、seed 0/1 各 256、100 steps、radius 0.6、dynamic release、all-detected rescue-only、detection/assignment redecision、每事件 1 架、finder-only。新增的 YAML 解析、hash、checkpoint guard 和 `resolved_config.json` 不改变策略行为。

有两项 provenance 限定不是任务行为差异：

1. 旧 high checkpoint 当年训练为 30 updates、minibatch 64，且 finder-first 是后来加入的执行机制；`high_train_v1.yaml` 冻结的是后续新训练协议（200 updates、minibatch 512），不是声称旧 checkpoint 由该 YAML 训练。
2. 当前 low checkpoint 的训练 target 是 50,000,000 frames，因 batch 粒度最终产物为 50,040,000 frames；它缺少新 metadata sidecar。低层 actor observation/reward 不使用 sensor/target belief，因此历史 low variant 中 sensor 0.3 与 final task 0.6 不构成低层策略输入差异，但 provenance 必须保留。

## 14. 尚未消除的不确定性

- 当前正式 82.03% 是 belief/persistence、strict physics 和新 low policy 的联合 pipeline 结果；没有 factorial 证据拆分各模块因果贡献。
- 新 `high_train_v1` 尚未训练，不能预先声称会复现或超过旧 checkpoint 的 82.03%。
- 环境内部 cluster-to-target identity matching 使用真值 footprint。actor 无直接旁路，但若论文要求完全去中心化的 data association，需要另立任务版本实现并评估。
- inactive UAV 继续感知且保留物理 collision body 是当前对齐选择；它是任务定义，不代表所有实际 SAR 系统的合理硬件语义。
