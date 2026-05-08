"""Grid search tuning for classical supply chain baselines."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.supply_chain import SupplyChainEnv
from src.agents.base_stock import BaseStockAgent
from src.agents.ss_policy import SSPolicyAgent


def run_episode(agent: Any, env: SupplyChainEnv) -> float:
    """Run one episode and return total cost (positive value)."""
    obs_dict, _ = env.reset()
    total_cost = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        actions = agent.act_all(obs_dict)
        obs_dict, reward, terminated, truncated, _ = env.step(actions)
        total_cost += -reward  # reward is negative cost
    return total_cost


def tune_base_stock(
    env_config: dict,
    n_seeds: int = 3,
    n_episodes: int = 50,
) -> dict:
    """
    Grid search S_i ∈ {5,10,15,20,25,30} per node.

    Returns {"S_levels": [best_S_0, best_S_1, best_S_2], "mean_cost": float}.
    """
    candidates = [5, 10, 15, 20, 25, 30]
    best_cost = float("inf")
    best_S: list[float] = [10.0, 10.0, 10.0]

    for combo in itertools.product(candidates, repeat=env_config.get("num_agents", 3)):
        costs: list[float] = []
        for seed in range(n_seeds):
            cfg = dict(env_config)
            cfg["seed"] = seed
            env = SupplyChainEnv(cfg)
            agent = BaseStockAgent(
                base_stock_levels=list(combo),
                max_order=env.max_order,
            )
            for _ in range(n_episodes):
                obs_dict, _ = env.reset(seed=seed)
                terminated = truncated = False
                ep_cost = 0.0
                while not (terminated or truncated):
                    actions = agent.act_all(obs_dict)
                    obs_dict, reward, terminated, truncated, _ = env.step(actions)
                    ep_cost += -reward
                costs.append(ep_cost)
        mean_cost = float(np.mean(costs))
        if mean_cost < best_cost:
            best_cost = mean_cost
            best_S = list(combo)

    return {"S_levels": best_S, "mean_cost": best_cost}


def tune_ss_policy(
    env_config: dict,
    n_seeds: int = 3,
    n_episodes: int = 50,
) -> dict:
    """
    Grid search (s_i, S_i) with s_i ∈ {5,10,15}, S_i ∈ {10,15,20,25,30}, s_i < S_i.

    Returns {"s_levels": [...], "S_levels": [...], "mean_cost": float}.
    """
    s_candidates = [5, 10, 15]
    S_candidates = [10, 15, 20, 25, 30]
    num_agents = env_config.get("num_agents", 3)

    # Build valid (s, S) pairs per node
    valid_pairs = [(s, S) for s in s_candidates for S in S_candidates if s < S]

    best_cost = float("inf")
    best_s: list[float] = [5.0] * num_agents
    best_S: list[float] = [20.0] * num_agents

    for combo in itertools.product(valid_pairs, repeat=num_agents):
        s_vec = [pair[0] for pair in combo]
        S_vec = [pair[1] for pair in combo]
        costs: list[float] = []
        for seed in range(n_seeds):
            cfg = dict(env_config)
            cfg["seed"] = seed
            env = SupplyChainEnv(cfg)
            agent = SSPolicyAgent(
                s_levels=s_vec,
                S_levels=S_vec,
                max_order=env.max_order,
            )
            for _ in range(n_episodes):
                obs_dict, _ = env.reset(seed=seed)
                terminated = truncated = False
                ep_cost = 0.0
                while not (terminated or truncated):
                    actions = agent.act_all(obs_dict)
                    obs_dict, reward, terminated, truncated, _ = env.step(actions)
                    ep_cost += -reward
                costs.append(ep_cost)
        mean_cost = float(np.mean(costs))
        if mean_cost < best_cost:
            best_cost = mean_cost
            best_s = list(s_vec)
            best_S = list(S_vec)

    return {"s_levels": best_s, "S_levels": best_S, "mean_cost": best_cost}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune classical baselines via grid search.")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_episodes", type=int, default=20)
    args = parser.parse_args()

    stationary_cfg = {
        "num_agents": 3,
        "lead_times": [2, 2, 2],
        "max_order": 20,
        "holding_cost": 1.0,
        "stockout_cost": 2.0,
        "initial_inventory": 10.0,
        "episode_length": 100,
        "demand_mode": "stationary",
        "demand_lambda": 5.0,
    }
    nonstationary_cfg = dict(stationary_cfg)
    nonstationary_cfg["demand_mode"] = "nonstationary"
    nonstationary_cfg["nonstationary_schedule"] = [
        {"step": 0, "lambda": 5.0},
        {"step": 50, "lambda": 12.0},
    ]

    results: dict[str, Any] = {}
    for name, cfg in [("stationary", stationary_cfg), ("nonstationary", nonstationary_cfg)]:
        print(f"Tuning base-stock [{name}]...")
        bs = tune_base_stock(cfg, n_seeds=args.n_seeds, n_episodes=args.n_episodes)
        print(f"  Best S_levels={bs['S_levels']}, mean_cost={bs['mean_cost']:.2f}")
        print(f"Tuning (s,S) policy [{name}]...")
        ss = tune_ss_policy(cfg, n_seeds=args.n_seeds, n_episodes=args.n_episodes)
        print(f"  Best s={ss['s_levels']} S={ss['S_levels']}, mean_cost={ss['mean_cost']:.2f}")
        results[name] = {"base_stock": bs, "ss_policy": ss}

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_params.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
