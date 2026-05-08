"""Tests for classical baseline agents."""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.supply_chain import SupplyChainEnv
from src.agents.base_stock import BaseStockAgent
from src.agents.ss_policy import SSPolicyAgent

BASE_CFG = {
    "num_agents": 3,
    "lead_times": [2, 2, 2],
    "max_order": 20,
    "holding_cost": 1.0,
    "stockout_cost": 2.0,
    "initial_inventory": 10.0,
    "episode_length": 100,
    "demand_mode": "stationary",
    "demand_lambda": 5.0,
    "seed": 42,
}


def make_env(overrides: dict | None = None) -> SupplyChainEnv:
    """Create env with optional config overrides."""
    cfg = dict(BASE_CFG)
    if overrides:
        cfg.update(overrides)
    return SupplyChainEnv(cfg)


def make_obs(inventory: float, incoming_demand: float, pipeline: list[float]) -> dict:
    """Construct a minimal obs_dict for a single agent."""
    obs_arr = np.array(
        [inventory, incoming_demand] + pipeline,
        dtype=np.float32,
    )
    return {"agents": [obs_arr]}


# ---------------------------------------------------------------------------
# Test 1: BaseStockAgent acts correctly
# ---------------------------------------------------------------------------

def test_base_stock_acts_correctly():
    """High S -> always orders; verify order = max(0, S - inv_pos)."""
    S = 30.0
    inventory = 5.0
    pipeline = [2.0, 3.0]  # lead_time = 2
    # inventory_position = 5 + 2 + 3 = 10
    # order = max(0, 30 - 10) = 20, clipped to max_order=20
    obs_dict = make_obs(inventory, 0.0, pipeline)
    agent = BaseStockAgent(base_stock_levels=[S], max_order=20)
    action = agent.act(obs_dict, 0)
    expected = min(int(max(0.0, S - (inventory + sum(pipeline)))), 20)
    assert action == expected, f"Expected {expected}, got {action}"

    # With inventory_position > S, order should be 0
    obs_dict_high = make_obs(40.0, 0.0, [5.0, 5.0])
    action_zero = agent.act(obs_dict_high, 0)
    assert action_zero == 0, f"Expected 0, got {action_zero}"


# ---------------------------------------------------------------------------
# Test 2: SSPolicyAgent triggers when below s
# ---------------------------------------------------------------------------

def test_ss_policy_below_s():
    """When inv_pos < s -> orders up to S."""
    s = 15.0
    S = 25.0
    inventory = 3.0
    pipeline = [2.0, 2.0]
    # inv_pos = 3 + 2 + 2 = 7  < s=15 → order = 25 - 7 = 18
    obs_dict = make_obs(inventory, 0.0, pipeline)
    agent = SSPolicyAgent(s_levels=[s], S_levels=[S], max_order=20)
    action = agent.act(obs_dict, 0)
    expected = min(int(max(0.0, S - (inventory + sum(pipeline)))), 20)
    assert action == expected, f"Expected {expected}, got {action}"


# ---------------------------------------------------------------------------
# Test 3: SSPolicyAgent returns 0 when above or at s
# ---------------------------------------------------------------------------

def test_ss_policy_above_s():
    """When inv_pos >= s -> orders 0."""
    s = 10.0
    S = 20.0
    inventory = 8.0
    pipeline = [3.0, 2.0]
    # inv_pos = 8 + 3 + 2 = 13 >= s=10 → order = 0
    obs_dict = make_obs(inventory, 0.0, pipeline)
    agent = SSPolicyAgent(s_levels=[s], S_levels=[S], max_order=20)
    action = agent.act(obs_dict, 0)
    assert action == 0, f"Expected 0, got {action}"

    # Exactly at s: inv_pos == s → should NOT trigger (not strictly below)
    obs_exact = make_obs(5.0, 0.0, [3.0, 2.0])
    # inv_pos = 5 + 3 + 2 = 10 == s=10 → order = 0
    action_at_s = agent.act(obs_exact, 0)
    assert action_at_s == 0, f"Expected 0 at s boundary, got {action_at_s}"


# ---------------------------------------------------------------------------
# Test 4: both baselines complete a full episode
# ---------------------------------------------------------------------------

def test_baselines_run_full_episode():
    """Both baselines complete a 100-step episode without error."""
    env = make_env()

    # BaseStockAgent
    bs_agent = BaseStockAgent(base_stock_levels=[20.0, 20.0, 20.0], max_order=20)
    obs, _ = env.reset(seed=0)
    terminated = truncated = False
    step_count = 0
    while not (terminated or truncated):
        actions = bs_agent.act_all(obs)
        assert len(actions) == 3
        obs, reward, terminated, truncated, info = env.step(actions)
        step_count += 1
    assert step_count == 100, f"BaseStock ran {step_count} steps, expected 100"

    # SSPolicyAgent
    ss_agent = SSPolicyAgent(
        s_levels=[10.0, 10.0, 10.0],
        S_levels=[20.0, 20.0, 20.0],
        max_order=20,
    )
    obs, _ = env.reset(seed=1)
    terminated = truncated = False
    step_count = 0
    while not (terminated or truncated):
        actions = ss_agent.act_all(obs)
        assert len(actions) == 3
        obs, reward, terminated, truncated, info = env.step(actions)
        step_count += 1
    assert step_count == 100, f"SS ran {step_count} steps, expected 100"
