from __future__ import annotations

"""Supply chain evaluation metrics."""
import numpy as np
from typing import Any


def total_cost(trajectories: list[dict]) -> float:
    """
    Mean episode cost averaged over trajectories.

    Each trajectory dict has key 'per_step_costs' (list of per-step total costs, positive floats).
    Returns mean total episode cost (positive).
    """
    episode_costs = [sum(traj["per_step_costs"]) for traj in trajectories]
    return float(np.mean(episode_costs))


def bullwhip_ratio(trajectories: list[dict], node_idx: int) -> float:
    """
    Bullwhip ratio for node node_idx.

    Each trajectory dict has 'per_node_orders': list of [o_0, o_1, o_2] per step,
    and 'customer_demands': list of scalar customer demands per step.

    bullwhip_ratio_i = Var(orders_i across all timesteps across all episodes) /
                       Var(customer_demands across all episodes)

    Returns float. Returns 1.0 if variance of customer demand is zero.
    """
    all_orders = []
    all_demands = []
    for traj in trajectories:
        for step_orders in traj["per_node_orders"]:
            all_orders.append(step_orders[node_idx])
        for d in traj["customer_demands"]:
            all_demands.append(d)

    all_orders = np.array(all_orders, dtype=float)
    all_demands = np.array(all_demands, dtype=float)

    var_demand = float(np.var(all_demands))
    if var_demand == 0.0:
        return 1.0
    return float(np.var(all_orders) / var_demand)


def stockout_frequency(trajectories: list[dict]) -> float:
    """
    Fraction of (timestep, node) pairs across all episodes where inventory < 0.

    Each trajectory dict has 'per_node_inventory': list of [inv_0, inv_1, inv_2] per step.
    """
    total_pairs = 0
    stockout_pairs = 0
    for traj in trajectories:
        for step_inv in traj["per_node_inventory"]:
            for inv in step_inv:
                total_pairs += 1
                if inv < 0:
                    stockout_pairs += 1
    if total_pairs == 0:
        return 0.0
    return float(stockout_pairs / total_pairs)


def compute_all_metrics(trajectories: list[dict]) -> dict:
    """
    Compute all metrics and return as dict:
    {
        "total_cost_mean": float,
        "total_cost_std": float,   # std across episodes
        "bullwhip_ratio_node0": float,
        "bullwhip_ratio_node1": float,
        "bullwhip_ratio_node2": float,
        "bullwhip_ratio_avg": float,
        "stockout_frequency": float,
    }

    For total_cost_std: compute per-episode cost, then take std across episodes.
    """
    episode_costs = np.array([sum(traj["per_step_costs"]) for traj in trajectories], dtype=float)
    bw0 = bullwhip_ratio(trajectories, 0)
    bw1 = bullwhip_ratio(trajectories, 1)
    bw2 = bullwhip_ratio(trajectories, 2)
    return {
        "total_cost_mean": float(np.mean(episode_costs)),
        "total_cost_std": float(np.std(episode_costs)),
        "bullwhip_ratio_node0": bw0,
        "bullwhip_ratio_node1": bw1,
        "bullwhip_ratio_node2": bw2,
        "bullwhip_ratio_avg": float(np.mean([bw0, bw1, bw2])),
        "stockout_frequency": stockout_frequency(trajectories),
    }
