# BenchMARL Training Notes

The notebook used a small trick: it replaced `VmasTask.NAVIGATION.get_env_fun`
so BenchMARL would run the custom scenario instead of the built-in VMAS
navigation task.

This project uses a cleaner local adapter:

- `CustomVmasNavigationTask` is the enum-like task entry.
- `CustomVmasNavigationClass` builds a TorchRL `VmasEnv` with
  `HeterogeneousNavigationScenario`.
- `scripts/train_mappo.py` passes the resulting task object directly to
  `benchmarl.experiment.Experiment`.

## Tutorial variants

The variants are in `custom_vmas_benchmarl/training_config.py` and mirrored in
`configs/`:

- `heterogeneous`: 2 holonomic agents, 1 differential-drive agent, 1 car
- `homogeneous`: 4 holonomic agents with LIDAR
- `no_lidar`: 4 holonomic agents without LIDAR
- `no_lidar_gnn`: 4 holonomic agents without LIDAR, with GNN communication

## Commands

Short smoke runs:

```bash
python3 -m scripts.train_mappo --variant heterogeneous --model mlp
python3 -m scripts.train_mappo --variant homogeneous --model mlp
python3 -m scripts.train_mappo --variant no_lidar --model mlp
python3 -m scripts.train_mappo --variant no_lidar_gnn --model gnn --comms-radius 1
```

Long training:

```bash
python3 scripts/train_mappo.py --variant heterogeneous --model mlp --full
```

## Why GNN requires homogeneous agents here

BenchMARL groups VMAS agents by name prefix. `holonomic_0` and `holonomic_1`
belong to the same group, while `diff_drive_0` and `car_0` are separate groups.
The GNN policy communicates within a group, so the tutorial switches to four
holonomic agents before using the GNN model.
