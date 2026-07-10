"""Probe VMAS spread parity physics against old MPE simple_spread dynamics."""

from __future__ import annotations

import argparse
from math import isclose

import numpy as np
import torch
from torchrl.envs.libs.vmas import VmasEnv

from comm_spread.scenario import CommSpreadScenario
from comm_spread.training_config import TASK_VARIANTS


def make_env(variant: str, device: str):
    cfg = dict(TASK_VARIANTS[variant])
    continuous = cfg.pop("continuous_actions", False)
    return VmasEnv(
        scenario=CommSpreadScenario(),
        num_envs=1,
        continuous_actions=continuous,
        categorical_actions=True,
        clamp_actions=True,
        device=device,
        **cfg,
    )


def set_layout(env):
    scenario = env._env.scenario
    world = scenario.world
    td = env.reset()
    agent_positions = [(-0.8, 0.0), (0.4, 0.4), (0.4, -0.4)]
    landmark_positions = [(-0.8, 0.8), (0.8, 0.8), (0.0, -0.8)]
    for agent, pos in zip(world.agents, agent_positions):
        agent.state.pos[:] = torch.tensor([pos], device=world.device, dtype=torch.float32)
        agent.state.vel.zero_()
    for landmark, pos in zip(scenario.landmarks, landmark_positions):
        landmark.state.pos[:] = torch.tensor([pos], device=world.device, dtype=torch.float32)
    return td


def mpe_expected_right(n_steps: int):
    v = 0.0
    x = 0.0
    xs = []
    for _ in range(n_steps):
        v = v * 0.75 + 5.0 * 0.1
        x += v * 0.1
        xs.append(x)
    return xs


def probe_actions(env):
    scenario = env._env.scenario
    world = scenario.world
    out = []
    for action_id in range(5):
        td = set_layout(env)
        action = torch.zeros((1, 3), device=world.device, dtype=torch.long)
        action[0, 0] = action_id
        td.set(("agents", "action"), action)
        before = world.agents[0].state.pos.detach().clone()[0]
        env.step(td)
        after = world.agents[0].state.pos.detach().clone()[0]
        delta = (after - before).detach().cpu().numpy()
        out.append((action_id, delta))
    return out


def probe_repeated_right(env, n_steps: int = 10):
    world = env._env.scenario.world
    td = set_layout(env)
    start = world.agents[0].state.pos.detach().clone()[0]
    xs = []
    for _ in range(n_steps):
        action = torch.zeros((1, 3), device=world.device, dtype=torch.long)
        action[0, 0] = 2
        td.set(("agents", "action"), action)
        td = env.step(td)
        pos = world.agents[0].state.pos.detach().clone()[0]
        xs.append(float((pos - start)[0].cpu()))
    return xs


def probe_reward_done(env):
    scenario = env._env.scenario
    world = scenario.world
    td = env.reset()
    close_positions = [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.0)]
    for agent, pos in zip(world.agents, close_positions):
        agent.state.pos[:] = torch.tensor([pos], device=world.device, dtype=torch.float32)
        agent.state.vel.zero_()
    for landmark, pos in zip(scenario.landmarks, close_positions):
        landmark.state.pos[:] = torch.tensor([pos], device=world.device, dtype=torch.float32)
    action = torch.zeros((1, 3), device=world.device, dtype=torch.long)
    td.set(("agents", "action"), action)
    td = env.step(td)
    reward = float(td.get(("next", "agents", "reward"))[0, 0, 0].cpu())
    done = bool(td.get(("next", "done"))[0, 0].cpu())
    success = bool(scenario.all_landmarks_covered[0].cpu())
    return reward, done, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="spread_mpe_physics_parity", choices=sorted(TASK_VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    env = make_env(args.variant, args.device)
    action_deltas = probe_actions(env)
    repeated = probe_repeated_right(env)
    expected = mpe_expected_right(len(repeated))
    reward, done, success = probe_reward_done(env)

    print(f"variant: {args.variant}")
    print("action deltas:")
    for action_id, delta in action_deltas:
        print(f"  {action_id}: ({delta[0]:.6f}, {delta[1]:.6f})")
    print("repeated right x deltas:")
    print("  observed:", [round(x, 4) for x in repeated])
    print("  old_mpe_expected:", [round(x, 4) for x in expected])
    print("  max_abs_error:", round(max(abs(a-b) for a, b in zip(repeated, expected)), 6))
    print(f"fixed success reward={reward:.6f} done={done} success={success}")

    if args.variant == "spread_mpe_physics_parity":
        assert max(abs(a-b) for a, b in zip(repeated, expected)) < 1e-5
        assert reward > -1e-4 and done and success


if __name__ == "__main__":
    main()
