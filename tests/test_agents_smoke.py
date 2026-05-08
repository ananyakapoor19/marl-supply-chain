"""Smoke tests: 1000-step training runs for each agent type."""
import pytest
import numpy as np
import math

from src.envs.supply_chain import SupplyChainEnv
from src.agents.idqn import IDQNAgent
from src.agents.cdqn import CDQNAgent
from src.agents.vdn import VDNAgent
from src.common.seeding import set_global_seeds


ENV_CONFIG = {
    "num_agents": 3,
    "lead_times": [2, 2, 2],
    "max_order": 20,
    "holding_cost": 1.0,
    "stockout_cost": 2.0,
    "initial_inventory": 10,
    "episode_length": 100,
    "demand_mode": "stationary",
    "demand_lambda": 5.0,
}

AGENT_CONFIG = {
    "total_timesteps": 1000,
    "learning_starts": 200,
    "batch_size": 32,
    "buffer_size": 5000,
    "gamma": 0.99,
    "lr": 5e-4,
    "tau": 0.005,
    "grad_clip": 10.0,
    "hidden_sizes": [64, 64],
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 500,
    "device": "cpu",
}


def run_smoke_test(agent, env, n_steps=1000):
    """Run n_steps, return list of losses."""
    obs, _ = env.reset(seed=0)
    losses = []
    for _ in range(n_steps):
        actions = agent.act(obs, explore=True)
        next_obs, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        agent.store_transition(obs, actions, reward, next_obs, done)
        metrics = agent.update()
        if metrics and "loss" in metrics:
            losses.append(metrics["loss"])
        if done:
            obs, _ = env.reset(seed=42)
        else:
            obs = next_obs
    return losses


def test_idqn_no_nan():
    """IDQN 1000-step run: no NaN losses, losses are finite."""
    set_global_seeds(0)
    env = SupplyChainEnv(ENV_CONFIG)
    agent = IDQNAgent(AGENT_CONFIG, env.obs_sizes, env.action_size, env.num_agents)
    losses = run_smoke_test(agent, env)
    assert len(losses) > 0, "No updates occurred"
    assert all(not math.isnan(l) for l in losses), "NaN loss detected"
    assert all(not math.isinf(l) for l in losses), "Inf loss detected"


def test_cdqn_no_nan():
    """CDQN 1000-step run: no NaN losses."""
    set_global_seeds(0)
    env = SupplyChainEnv(ENV_CONFIG)
    agent = CDQNAgent(AGENT_CONFIG, env.global_obs_size, env.action_size, env.num_agents)
    losses = run_smoke_test(agent, env)
    assert len(losses) > 0
    assert all(not math.isnan(l) for l in losses)


def test_vdn_no_nan():
    """VDN 1000-step run: no NaN losses."""
    set_global_seeds(0)
    env = SupplyChainEnv(ENV_CONFIG)
    agent = VDNAgent(AGENT_CONFIG, env.obs_sizes, env.action_size, env.num_agents)
    losses = run_smoke_test(agent, env)
    assert len(losses) > 0
    assert all(not math.isnan(l) for l in losses)


def test_idqn_actions_valid():
    """IDQN actions are in valid range."""
    env = SupplyChainEnv(ENV_CONFIG)
    agent = IDQNAgent(AGENT_CONFIG, env.obs_sizes, env.action_size, env.num_agents)
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        actions = agent.act(obs, explore=True)
        assert len(actions) == 3
        assert all(0 <= a <= 20 for a in actions)
        obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            obs, _ = env.reset(seed=1)


def test_save_load_idqn(tmp_path):
    """IDQN save/load preserves weights."""
    env = SupplyChainEnv(ENV_CONFIG)
    agent = IDQNAgent(AGENT_CONFIG, env.obs_sizes, env.action_size, env.num_agents)
    path = str(tmp_path / "idqn_test.pt")
    agent.save(path)

    agent2 = IDQNAgent(AGENT_CONFIG, env.obs_sizes, env.action_size, env.num_agents)
    agent2.load(path)

    obs, _ = env.reset(seed=5)
    import torch
    with torch.no_grad():
        for i in range(3):
            import torch as th
            o = th.tensor(obs["agents"][i], dtype=th.float32).unsqueeze(0)
            q1 = agent.q_nets[i](o)
            q2 = agent2.q_nets[i](o)
            assert th.allclose(q1, q2), f"Weights differ for agent {i} after load"
