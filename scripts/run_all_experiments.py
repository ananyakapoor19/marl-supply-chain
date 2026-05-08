"""
Run all experiments. Produces every result needed for the report.

Usage:
    python scripts/run_all_experiments.py          # full sweep (~8h)
    python scripts/run_all_experiments.py --quick  # ~20 min sanity check

Matrix:
- Methods: [base_stock, ss_policy, idqn, cdqn, vdn]
- Settings: [stationary, nonstationary]
- Seeds: [0, 1, 2, 3, 4]
-> 50 total runs for RL methods; baselines tune once then eval
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.supply_chain import SupplyChainEnv
from src.agents.base_stock import BaseStockAgent
from src.agents.ss_policy import SSPolicyAgent
from src.common.seeding import set_global_seeds
from src.evaluation.evaluator import evaluate, save_evaluation_results
from src.training.tune_baselines import tune_base_stock, tune_ss_policy
from src.utils.config import load_config


SEEDS = [0, 1, 2, 3, 4]
DEMAND_MODES = ["stationary", "nonstationary"]
RL_METHODS = ["idqn", "cdqn", "vdn"]

PROJECT_ROOT = Path(__file__).parent.parent


def results_exist(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def train_rl_method(method: str, demand_mode: str, seed: int, quick: bool) -> Path:
    """Train one RL method. Skip if checkpoint already exists."""
    ckpt = PROJECT_ROOT / f"results/checkpoints/{method}_{demand_mode}_seed{seed}_final.pt"
    if ckpt.exists():
        print(f"  [skip] {method} {demand_mode} seed={seed} — checkpoint exists")
        return ckpt

    config_path = PROJECT_ROOT / f"configs/{method}.yaml"

    import yaml
    config = load_config(config_path)
    config["env"]["demand_mode"] = demand_mode
    if demand_mode == "nonstationary":
        config["env"]["nonstationary_schedule"] = [
            {"step": 0, "lambda": 5.0},
            {"step": 50, "lambda": 12.0},
        ]

    tmp_config = PROJECT_ROOT / f"configs/_tmp_{method}_{demand_mode}_{seed}.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(config, f)

    cmd = [
        sys.executable, "-m", "src.training.train_dqn",
        "--config", str(tmp_config),
        "--seed", str(seed),
    ]
    if quick:
        cmd.append("--quick")

    print(f"  Training {method.upper()} | {demand_mode} | seed={seed}...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    tmp_config.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  ERROR: training failed for {method} {demand_mode} seed={seed}")

    return ckpt


def eval_rl_method(method: str, demand_mode: str, seed: int, n_episodes: int) -> None:
    """Evaluate a trained RL agent and save metrics."""
    out_path = PROJECT_ROOT / f"results/logs/{method}_{demand_mode}_seed{seed}_eval.csv"
    if results_exist(out_path):
        print(f"  [skip] eval {method} {demand_mode} seed={seed}")
        return

    ckpt = PROJECT_ROOT / f"results/checkpoints/{method}_{demand_mode}_seed{seed}_final.pt"
    if not ckpt.exists():
        print(f"  [warn] No checkpoint for {method} {demand_mode} seed={seed}, skipping eval")
        return

    import torch
    from src.agents.idqn import IDQNAgent
    from src.agents.cdqn import CDQNAgent
    from src.agents.vdn import VDNAgent

    config = load_config(PROJECT_ROOT / f"configs/{method}.yaml")
    config["env"]["demand_mode"] = demand_mode
    env_config = config["env"]

    env = SupplyChainEnv(env_config)
    if method == "cdqn":
        agent = CDQNAgent(config, env.global_obs_size, env.action_size, env.num_agents)
    else:
        AGENT_CLASSES = {"idqn": IDQNAgent, "vdn": VDNAgent}
        agent = AGENT_CLASSES[method](config, env.obs_sizes, env.action_size, env.num_agents)
    agent.load(str(ckpt))

    set_global_seeds(seed + 50000)
    result = evaluate(agent, env_config, n_episodes=n_episodes, seed_offset=seed * 100000 + 50000)
    save_evaluation_results(result["metrics"], out_path)
    print(f"  Eval {method} {demand_mode} seed={seed}: cost={result['metrics']['total_cost_mean']:.1f}")


def run_baselines(demand_mode: str, n_episodes: int, quick: bool) -> None:
    """Tune baselines (once) then evaluate for each seed."""
    params_path = PROJECT_ROOT / "results/tables/baseline_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)

    if demand_mode == "stationary":
        env_config_base = load_config(PROJECT_ROOT / "configs/env_default.yaml")
    else:
        env_config_base = load_config(PROJECT_ROOT / "configs/env_nonstationary.yaml")

    # Load or compute tuned params
    all_params = {}
    if params_path.exists():
        with open(params_path) as f:
            all_params = json.load(f)

    for method in ["base_stock", "ss_policy"]:
        key = f"{method}_{demand_mode}"
        if key not in all_params:
            print(f"  Tuning {method} for {demand_mode}...")
            n_tune_seeds = 2 if quick else 3
            n_tune_eps = 20 if quick else 50
            if method == "base_stock":
                params = tune_base_stock(dict(env_config_base), n_seeds=n_tune_seeds, n_episodes=n_tune_eps)
            else:
                params = tune_ss_policy(dict(env_config_base), n_seeds=n_tune_seeds, n_episodes=n_tune_eps)
            all_params[key] = params
            with open(params_path, "w") as f:
                json.dump(all_params, f, indent=2)

        params = all_params[key]

        for seed in SEEDS:
            out_path = PROJECT_ROOT / f"results/logs/{method}_{demand_mode}_seed{seed}_eval.csv"
            if results_exist(out_path):
                print(f"  [skip] {method} {demand_mode} seed={seed}")
                continue

            set_global_seeds(seed + 99999)
            if method == "base_stock":
                agent = BaseStockAgent(
                    params["S_levels"],
                    max_order=env_config_base.get("max_order", 20),
                )
            else:
                agent = SSPolicyAgent(
                    params["s_levels"],
                    params["S_levels"],
                    max_order=env_config_base.get("max_order", 20),
                )

            result = evaluate(agent, dict(env_config_base), n_episodes=n_episodes, seed_offset=seed * 200000)
            save_evaluation_results(result["metrics"], out_path)
            print(f"  Eval {method} {demand_mode} seed={seed}: cost={result['metrics']['total_cost_mean']:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast smoke run (~20 min)")
    args = parser.parse_args()

    n_episodes = 20 if args.quick else 100

    print("=" * 60)
    print("MARL Supply Chain Experiment Suite")
    print("=" * 60)

    for demand_mode in DEMAND_MODES:
        print(f"\n--- Baselines: {demand_mode} ---")
        run_baselines(demand_mode, n_episodes, args.quick)

    for demand_mode in DEMAND_MODES:
        for method in RL_METHODS:
            print(f"\n--- {method.upper()} | {demand_mode} ---")
            for seed in SEEDS:
                train_rl_method(method, demand_mode, seed, args.quick)
                eval_rl_method(method, demand_mode, seed, n_episodes)

    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("Run: python scripts/make_figures.py && python scripts/make_tables.py")


if __name__ == "__main__":
    main()
