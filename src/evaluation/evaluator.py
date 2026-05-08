from __future__ import annotations

"""Agent evaluation via greedy rollouts."""
import csv
import inspect
from pathlib import Path

import numpy as np

from src.envs.supply_chain import SupplyChainEnv
from src.evaluation.metrics import compute_all_metrics


def evaluate(agent, env_config: dict, n_episodes: int = 100, seed_offset: int = 77777) -> dict:
    """
    Run n_episodes greedy (explore=False) rollouts.

    Returns dict with:
    - "metrics": result of compute_all_metrics(trajectories)
    - "trajectories": list of trajectory dicts (raw data for plotting)

    Each trajectory dict:
    {
        "episode_cost": float,
        "per_step_costs": list[float],
        "per_node_orders": list[list[float]],   # shape: (T, 3)
        "per_node_inventory": list[list[float]], # shape: (T, 3)
        "customer_demands": list[float],         # shape: (T,)
        "actions": list[list[int]],              # shape: (T, 3)
    }

    For baselines (which have no 'explore' kwarg), detect and call agent.act_all(obs) instead.
    """
    # Detect if this is a baseline agent (has act_all but act() doesn't accept explore kwarg)
    is_baseline = hasattr(agent, "act_all") and not _agent_accepts_explore(agent)

    env = SupplyChainEnv(env_config)
    trajectories = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        terminated = False
        truncated = False

        per_step_costs = []
        per_node_orders = []
        per_node_inventory = []
        customer_demands = []
        actions_list = []

        while not (terminated or truncated):
            if is_baseline:
                actions = agent.act_all(obs)
            else:
                actions = agent.act(obs, explore=False)

            next_obs, reward, terminated, truncated, info = env.step(actions)

            step_cost = float(-reward)
            per_step_costs.append(step_cost)
            per_node_orders.append([float(a) for a in actions])
            per_node_inventory.append([float(v) for v in info["per_node_inventory"]])
            customer_demands.append(float(info["customer_demand"]))
            actions_list.append([int(a) for a in actions])

            obs = next_obs

        traj = {
            "episode_cost": sum(per_step_costs),
            "per_step_costs": per_step_costs,
            "per_node_orders": per_node_orders,
            "per_node_inventory": per_node_inventory,
            "customer_demands": customer_demands,
            "actions": actions_list,
        }
        trajectories.append(traj)

    metrics = compute_all_metrics(trajectories)
    return {"metrics": metrics, "trajectories": trajectories}


def _agent_accepts_explore(agent) -> bool:
    """Return True if agent.act() accepts an 'explore' keyword argument."""
    try:
        sig = inspect.signature(agent.act)
        return "explore" in sig.parameters
    except (TypeError, ValueError):
        return False


def save_evaluation_results(metrics: dict, path: str | Path) -> None:
    """Save metrics dict to CSV (one row per metric)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def load_evaluation_results(path: str | Path) -> dict:
    """Load metrics dict from CSV."""
    path = Path(path)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    row = rows[0]
    # Convert all values to float
    return {k: float(v) for k, v in row.items()}
