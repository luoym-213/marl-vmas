# CustormVmasBenchmarl

这个目录把 Colab 教程
`Simulation_and_training_in_VMAS_and_BenchMARL.ipynb` 拆成了一个小工程，便于逐步学习和修改。

## 目录结构

```text
CustormVmasBenchmarl/
  custom_vmas_benchmarl/
    scenario.py          # VMAS 自定义场景：智能体、障碍物、观测、奖励、done、render
    env_factory.py       # VMAS 环境构造函数
    benchmarl_task.py    # BenchMARL TaskClass 适配层
    training_config.py   # MAPPO、MLP、GNN 和实验配置
  scripts/
    inspect_env.py       # 查看环境接口和张量形状
    random_rollout.py    # 随机动作 rollout，可选渲染 gif
    train_mappo.py       # BenchMARL 训练入口
  configs/               # 教程中的几组 task config
  docs/                  # 学习笔记
```

## 安装

建议在新虚拟环境中安装：

```bash
cd /Users/ym/Public/codes/vmas-test/CustormVmasBenchmarl
python3 -m pip install -e .
```

如果要训练 GNN 版本，`torch-cluster` 需要按你的 PyTorch/CUDA 版本安装对应 wheel：

```bash
python3 - <<'PY'
import torch
torch_version = torch.__version__.split("+")[0]
cuda = "cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", "")
print(f"https://data.pyg.org/whl/torch-{torch_version}+{cuda}.html")
PY
```

然后用输出的 URL 安装，例如：

```bash
python3 -m pip install torch-cluster -f <上一步输出的URL>
python3 -m pip install torch-geometric
```

## 学习顺序

1. 阅读 `custom_vmas_benchmarl/scenario.py`，理解 `make_world`、`reset_world_at`、`observation`、`reward`。
2. 运行环境接口检查：

```bash
python3 scripts/inspect_env.py --num-envs 8 --device cpu
```

3. 随机动作 rollout：

```bash
python3 scripts/random_rollout.py --steps 100 --render --output outputs/random_rollout.gif
```

4. 运行快速 BenchMARL smoke training：

```bash
python3 scripts/train_mappo.py --variant heterogeneous --model mlp
```

5. 对比教程中的实验：

```bash
python3 scripts/train_mappo.py --variant homogeneous --model mlp
python3 scripts/train_mappo.py --variant no_lidar --model mlp
python3 scripts/train_mappo.py --variant no_lidar_gnn --model gnn --comms-radius 1
```

默认训练是短 smoke run。完整训练加 `--full`，会把帧数切换到教程里的长期训练配置。

## 和原 Colab 的区别

Colab 版把安装、场景定义、环境演示、训练都放在一个 notebook 文件里。本工程做了这些调整：

- 移除了 `!pip install`、`apt-get`、`IPython.display` 等 notebook 专用代码。
- 场景类改名为 `HeterogeneousNavigationScenario`，放在独立模块中。
- BenchMARL 不再 monkey patch `VmasTask.NAVIGATION`，而是提供 `CustomVmasNavigationTask`。
- 训练实验通过 `--variant` 和 `--model` 选择，便于重复运行和比较。

