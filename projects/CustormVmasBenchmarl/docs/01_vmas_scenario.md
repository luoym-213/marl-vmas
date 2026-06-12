# VMAS Scenario Learning Notes

The scenario lives in `custom_vmas_benchmarl/scenario.py`.

## Core lifecycle

`make_world` creates every physical object in the vectorized world:

- holonomic agents
- differential-drive agents
- kinematic-bicycle car agents
- one color-matched goal per agent
- fixed circular obstacles
- one LIDAR sensor per agent

`reset_world_at` randomizes positions. VMAS can reset all environments at once
with `env_index=None`, or reset only one vectorized environment when a single
batch element is done.

`observation` returns a dictionary:

- `obs`: relative goal position, relative obstacle positions, and LIDAR readings
- `pos`: absolute agent position, used by GNN topology construction
- `vel`: agent velocity, used by GNN edge features
- `rot` and `ang_vel`: only for non-holonomic agents

`reward` has three parts:

- position progress reward: previous goal distance minus current goal distance
- collision penalty: local penalty for agent-agent and agent-obstacle collisions
- final reward: small global reward when all goals are reached

`done` returns true when every agent in a vectorized environment reaches its
goal. `extra_render` adds non-holonomic orientation markers and optional
communication lines.

## Things to modify first

Start with these kwargs in `make_world`:

- `n_agents_holonomic`
- `n_agents_diff_drive`
- `n_agents_car`
- `n_obstacles`
- `lidar_range`
- `n_lidar_rays`
- `shared_rew`
- `agent_collision_penalty`

The quickest feedback loop is:

```bash
python3 -m scripts.inspect_env
python3 -m scripts.random_rollout --steps 100 --render
```
