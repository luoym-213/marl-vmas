# CommSpread

VMAS implementation of a cooperative `simple_spread`-style task with BenchMARL
MAPPO training helpers.

## Task

Multiple holonomic agents must collectively cover a set of landmarks. Unlike
the previous color-matched navigation task, landmarks are not assigned to
specific agents. The team is rewarded when every landmark has at least one
nearby agent, while agent-agent collisions are penalized.

The default reward follows the MPE `simple_spread` structure:

```text
reward_i = -sum_l min_a distance(agent_a, landmark_l) + collision_penalty_i
```

By default the episode relies on `max_steps`, like MPE. Set
`done_when_all_covered: true` to terminate early once every landmark is within
`coverage_radius` of at least one agent.

## Layout

```text
CommSpread/
  comm_spread/
    scenario.py          # VMAS simple_spread scenario
    env_factory.py       # VMAS make_env helper
    benchmarl_task.py    # BenchMARL Task adapter
    training_config.py   # MAPPO, MLP, GNN and experiment configs
  configs/
    spread.yaml
    spread_gnn.yaml
  scripts/
    inspect_env.py
    random_rollout.py
    train_mappo.py
```

## Install

```bash
cd /Users/ym/Public/codes/marl-vmas/projects/CommSpread
python3 -m pip install -e .
```

## Quick checks

```bash
python3 scripts/inspect_env.py --num-envs 8 --device cpu
python3 scripts/random_rollout.py --steps 100 --render --output outputs/random_rollout.gif
python3 scripts/train_mappo.py --variant spread --model mlp
python3 scripts/train_mappo.py --variant spread_early_done --model mlp 
```

For graph policies:

```bash
python3 scripts/train_mappo.py --variant spread_gnn --model gnn --comms-radius 1.0
```

## Full training

```bash
python3 -m scripts.train_mappo --variant spread_early_done --model mlp --full
python3 scripts/train_mappo.py --variant spread_gnn --model gnn --comms-radius 1.0 --full
```