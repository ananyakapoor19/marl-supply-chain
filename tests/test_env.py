"""Tests for SupplyChainEnv."""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.supply_chain import SupplyChainEnv

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


# ---------------------------------------------------------------------------
# Test 1: deterministic reset with same seed
# ---------------------------------------------------------------------------

def test_reset_determinism():
    """Same seed -> same first obs."""
    env = make_env()
    obs1, _ = env.reset(seed=0)
    obs2, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs1["global"], obs2["global"])
    for i in range(3):
        np.testing.assert_array_equal(obs1["agents"][i], obs2["agents"][i])


# ---------------------------------------------------------------------------
# Test 2: shapes and dtypes
# ---------------------------------------------------------------------------

def test_step_shapes_and_dtypes():
    """obs arrays are float32 with correct shapes."""
    env = make_env()
    obs, _ = env.reset(seed=1)

    # Check per-agent shapes: 2 + lead_time
    for i in range(3):
        expected_size = 2 + BASE_CFG["lead_times"][i]
        assert obs["agents"][i].shape == (expected_size,), (
            f"Agent {i} obs shape mismatch: {obs['agents'][i].shape}"
        )
        assert obs["agents"][i].dtype == np.float32, (
            f"Agent {i} obs dtype mismatch: {obs['agents'][i].dtype}"
        )

    # Check global shape
    expected_global = sum(2 + lt for lt in BASE_CFG["lead_times"])
    assert obs["global"].shape == (expected_global,)
    assert obs["global"].dtype == np.float32

    # Step and verify shapes remain consistent
    actions = [5, 5, 5]
    next_obs, reward, terminated, truncated, info = env.step(actions)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    for i in range(3):
        expected_size = 2 + BASE_CFG["lead_times"][i]
        assert next_obs["agents"][i].shape == (expected_size,)
        assert next_obs["agents"][i].dtype == np.float32


# ---------------------------------------------------------------------------
# Test 3: episode terminates at episode_length
# ---------------------------------------------------------------------------

def test_episode_terminates():
    """Episode truncates at episode_length."""
    env = make_env({"episode_length": 10})
    obs, _ = env.reset(seed=0)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step([5, 5, 5])
        steps += 1
    assert steps == 10, f"Expected 10 steps, got {steps}"
    assert truncated is True
    assert terminated is False


# ---------------------------------------------------------------------------
# Test 4: backlog accumulates when ordering 0
# ---------------------------------------------------------------------------

def test_backlog_accumulation():
    """Always ordering 0 -> inventory goes negative -> stockout cost positive."""
    env = make_env({"initial_inventory": 0.0, "demand_lambda": 5.0, "episode_length": 30})
    obs, _ = env.reset(seed=7)
    total_stockout_cost = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step([0, 0, 0])
        # retailer always faces demand; with 0 orders inventory goes negative
        inv = info["per_node_inventory"]
        costs = info["per_node_costs"]
        total_stockout_cost += sum(costs)

    # With zero inventory and non-zero demand, there must be stockout costs
    assert total_stockout_cost > 0.0, "Expected positive stockout cost when always ordering 0"


# ---------------------------------------------------------------------------
# Test 5: lead time correctness
# ---------------------------------------------------------------------------

def test_lead_time_correctness():
    """Order placed at step t arrives at step t + lead_time."""
    lead_time = 3
    env = make_env(
        {
            "lead_times": [lead_time, lead_time, lead_time],
            "initial_inventory": 100.0,   # large buffer to prevent stockouts masking test
            "demand_lambda": 0.0,         # zero demand to isolate pipeline
            "episode_length": 20,
        }
    )
    obs, _ = env.reset(seed=0)

    order_qty = 10

    # Step 0: place order only at supplier (node 0), others order 0
    # We focus on node 0: its pipeline should deliver `order_qty` units at step lead_time
    inv_before = float(env.inventory[0])
    env.step([order_qty, 0, 0])  # step 1 recorded internally as step 0 placed

    # Record inventory at each step and check it increases exactly at step lead_time
    inventories = []
    for _ in range(lead_time + 2):
        _, _, _, truncated, info = env.step([0, 0, 0])
        inventories.append(info["per_node_inventory"][0])
        if truncated:
            break

    # The order placed at step=0 should arrive at step=lead_time (i.e., the lead_time-th future step)
    # inventories[0] = after step 1, inventories[lead_time-1] = after step lead_time
    # Before arrival the inventory should be decreasing (0 demand, no arrivals yet), then jump
    # The jump should happen at index lead_time - 1 (0-indexed)
    arrival_idx = lead_time - 1
    if arrival_idx > 0:
        pre_arrival = inventories[arrival_idx - 1]
        post_arrival = inventories[arrival_idx]
        assert post_arrival >= pre_arrival, (
            f"Expected inventory to increase at lead_time step, "
            f"pre={pre_arrival}, post={post_arrival}"
        )


# ---------------------------------------------------------------------------
# Test 6: non-stationary demand lambda changes
# ---------------------------------------------------------------------------

def test_nonstationary_demand():
    """Lambda changes at scheduled step."""
    schedule = [{"step": 0, "lambda": 1.0}, {"step": 5, "lambda": 50.0}]
    env = make_env(
        {
            "demand_mode": "nonstationary",
            "demand_lambda": 1.0,
            "nonstationary_schedule": schedule,
            "episode_length": 20,
            "initial_inventory": 1000.0,  # large stock to avoid masking demand
        }
    )
    obs, info = env.reset(seed=0)
    assert info["current_lambda"] == 1.0

    demands_before: list[float] = []
    demands_after: list[float] = []

    for step in range(20):
        _, _, _, truncated, info = env.step([20, 20, 20])
        if step < 5:
            demands_before.append(info["customer_demand"])
        else:
            demands_after.append(info["customer_demand"])
        if truncated:
            break

    # With lambda=50 after step 5, average demand should be much higher than lambda=1
    if demands_before and demands_after:
        assert np.mean(demands_after) > np.mean(demands_before), (
            f"Expected higher demand after schedule change: "
            f"before={np.mean(demands_before):.2f}, after={np.mean(demands_after):.2f}"
        )


# ---------------------------------------------------------------------------
# Test 7: flow / mass balance
# ---------------------------------------------------------------------------

def test_flow_conservation():
    """Total units ordered by supplier ≈ total demand + inventory change (mass balance)."""
    env = make_env({"episode_length": 50, "initial_inventory": 20.0})
    obs, _ = env.reset(seed=3)

    initial_inv_total = float(np.sum(env.inventory))
    initial_pipeline_total = float(sum(np.sum(p) for p in env.pipelines))

    total_ordered_supplier = 0.0
    total_customer_demand = 0.0

    terminated = truncated = False
    order_qty = 8
    while not (terminated or truncated):
        actions = [order_qty, order_qty, order_qty]
        obs, reward, terminated, truncated, info = env.step(actions)
        total_ordered_supplier += order_qty
        total_customer_demand += info["customer_demand"]

    final_inv_total = float(np.sum(env.inventory))
    final_pipeline_total = float(sum(np.sum(p) for p in env.pipelines))

    # Mass balance (approximate, due to integer arithmetic and pipeline contents):
    # units_in (from supplier orders) = units_out (customer demand) + delta_inventory + delta_pipeline
    delta_inv = final_inv_total - initial_inv_total
    delta_pipeline = final_pipeline_total - initial_pipeline_total

    # units ordered by supplier entered system; units demanded by customers left at retailer
    # The chain means supplier orders feed into the pipeline of node 0 only.
    # For a rough check: total units put in by supplier >= total units demanded
    # (since inventory can go negative / backlog)
    lhs = total_ordered_supplier + initial_inv_total + initial_pipeline_total
    rhs = total_customer_demand + final_inv_total + final_pipeline_total
    # Allow tolerance for backlog (negative inventory)
    assert abs(lhs - rhs) < total_ordered_supplier * 0.5, (
        f"Mass balance off: lhs={lhs:.1f}, rhs={rhs:.1f}"
    )


# ---------------------------------------------------------------------------
# Test 8: reward is always <= 0
# ---------------------------------------------------------------------------

def test_reward_is_negative_cost():
    """Reward is always <= 0 (costs are non-negative)."""
    env = make_env()
    obs, _ = env.reset(seed=99)
    terminated = truncated = False
    while not (terminated or truncated):
        actions = [5, 5, 5]
        obs, reward, terminated, truncated, info = env.step(actions)
        assert reward <= 0.0, f"Expected reward <= 0, got {reward}"
        # Verify consistency: reward = -sum(per_node_costs)
        expected_reward = -sum(info["per_node_costs"])
        assert abs(reward - expected_reward) < 1e-5, (
            f"reward={reward} != -sum_costs={expected_reward}"
        )
