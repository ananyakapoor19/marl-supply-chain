"""
Run all experiments. Produces every result needed for the report.

Usage:
    python scripts/run_all_experiments.py              # full sweep, seeds in parallel
    python scripts/run_all_experiments.py --quick      # ~10 min sanity check
    python scripts/run_all_experiments.py --workers 3  # limit parallelism

Matrix:
- Methods: [base_stock, ss_policy, idqn, cdqn, vdn]
- Settings: [stationary, nonstationary]
- Seeds: [0, 1, 2, 3, 4]

All 5 seeds per (method, setting) pair train simultaneously. Wall time
drops from ~8h to ~1.5–2h on a quad-core machine.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base_stock import BaseStockAgent
from src.agents.ss_policy import SSPolicyAgent
from src.common.seeding import set_global_seeds
from src.envs.supply_chain import SupplyChainEnv
from src.evaluation.evaluator import evaluate, save_evaluation_results
from src.training.tune_baselines import tune_base_stock, tune_ss_policy
from src.utils.config import load_config


SEEDS = [0, 1, 2, 3, 4]
DEMAND_MODES = ["stationary", "nonstationary"]
RL_METHODS = ["idqn", "cdqn", "vdn"]


def results_exist(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _train_one(method: str, demand_mode: str, seed: int, quick: bool) -> tuple[str, bool]:
    """Train a single (method, demand_mode, seed) run. Returns (label, success)."""
    label = f"{method.upper()} | {demand_mode} | seed={seed}"
    ckpt = PROJECT_ROOT / f"results/checkpoints/{method}_{demand_mode}_seed{seed}_final.pt"

    if ckpt.exists():
        print(f"  [skip] {label} — checkpoint exists")
        return label, True

    config = load_config(PROJECT_ROOT / f"configs/{method}.yaml")
    config["env"]["demand_mode"] = demand_mode
    if demand_mode == "nonstationary":
        config["env"]["nonstationary_schedule"] = [
            {"step": 0, "lambda": 5.0},
            {"step": 50, "lambda": 12.0},
        ]

    tmp_config = PROJECT_ROOT / f"configs/_tmp_{method}_{demand_mode}_{seed}.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(config, f)

    cmd = [sys.executable, "-m", "src.training.train_dqn",
           "--config", str(tmp_config), "--seed", str(seed)]
    if quick:
        cmd.append("--quick")

    print(f"  Starting {label}...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                            capture_output=False)
    tmp_config.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  ERROR: {label} failed (exit {result.returncode})")
        return label, False

    print(f"  Done: {label}")
    return label, True


def _eval_one_rl(method: str, demand_mode: str, seed: int, n_episodes: int) -> None:
    """Evaluate one trained RL agent in-process."""
    out_path = PROJECT_ROOT / f"results/logs/{method}_{demand_mode}_seed{seed}_eval.csv"
    if results_exist(out_path):
        print(f"  [skip] eval {method} {demand_mode} seed={seed}")
        return

    ckpt = PROJECT_ROOT / f"results/checkpoints/{method}_{demand_mode}_seed{seed}_final.pt"
    if not ckpt.exists():
        print(f"  [warn] No checkpoint for {method} {demand_mode} seed={seed}, skipping eval")
        return

    from src.agents.cdqn import CDQNAgent
    from src.agents.idqn import IDQNAgent
    from src.agents.vdn import VDNAgent

    config = load_config(PROJECT_ROOT / f"configs/{method}.yaml")
    config["env"]["demand_mode"] = demand_mode
    env_config = config["env"]

    env = SupplyChainEnv(env_config)
    if method == "cdqn":
        agent = CDQNAgent(config, env.global_obs_size, env.action_size, env.num_agents)
    elif method == "idqn":
        agent = IDQNAgent(config, env.obs_sizes, env.action_size, env.num_agents)
    else:
        agent = VDNAgent(config, env.obs_sizes, env.action_size, env.num_agents)
    agent.load(str(ckpt))

    set_global_seeds(seed + 50000)
    result = evaluate(agent, env_config, n_episodes=n_episodes,
                      seed_offset=seed * 100000 + 50000)
    save_evaluation_results(result["metrics"], out_path)
    print(f"  Eval {method} {demand_mode} seed={seed}: "
          f"cost={result['metrics']['total_cost_mean']:.1f}")


def run_baselines(demand_mode: str, n_episodes: int, quick: bool) -> None:
    """Tune baselines once, then evaluate across all seeds."""
    params_path = PROJECT_ROOT / "results/tables/baseline_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_file = "env_default.yaml" if demand_mode == "stationary" else "env_nonstationary.yaml"
    env_cfg = load_config(PROJECT_ROOT / "configs" / cfg_file)

    all_params: dict = {}
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
                params = tune_base_stock(dict(env_cfg), n_seeds=n_tune_seeds,
                                         n_episodes=n_tune_eps)
            else:
                params = tune_ss_policy(dict(env_cfg), n_seeds=n_tune_seeds,
                                        n_episodes=n_tune_eps)
            all_params[key] = params
            with open(params_path, "w") as f:
                json.dump(all_params, f, indent=2)

        params = all_params[key]
        max_order = env_cfg.get("max_order", 20)

        for seed in SEEDS:
            out_path = PROJECT_ROOT / f"results/logs/{method}_{demand_mode}_seed{seed}_eval.csv"
            if results_exist(out_path):
                print(f"  [skip] {method} {demand_mode} seed={seed}")
                continue

            set_global_seeds(seed + 99999)
            if method == "base_stock":
                agent = BaseStockAgent(params["S_levels"], max_order=max_order)
            else:
                agent = SSPolicyAgent(params["s_levels"], params["S_levels"],
                                      max_order=max_order)

            result = evaluate(agent, dict(env_cfg), n_episodes=n_episodes,
                              seed_offset=seed * 200000)
            save_evaluation_results(result["metrics"], out_path)
            print(f"  Eval {method} {demand_mode} seed={seed}: "
                  f"cost={result['metrics']['total_cost_mean']:.1f}")


def train_all_parallel(demand_mode: str, quick: bool, workers: int) -> None:
    """Train all (method, seed) pairs for one demand_mode in parallel."""
    jobs = [(method, seed) for method in RL_METHODS for seed in SEEDS]
    total = len(jobs)
    done = 0

    print(f"\n  Launching {total} training jobs with {workers} workers "
          f"[{demand_mode}]...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_train_one, method, demand_mode, seed, quick): (method, seed)
            for method, seed in jobs
        }
        for fut in as_completed(futures):
            label, ok = fut.result()
            done += 1
            status = "OK" if ok else "FAILED"
            print(f"  [{done}/{total}] {status}: {label}")


def eval_all(demand_mode: str, n_episodes: int) -> None:
    """Evaluate all trained RL agents for one demand_mode (sequential, fast)."""
    for method in RL_METHODS:
        for seed in SEEDS:
            _eval_one_rl(method, demand_mode, seed, n_episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Short smoke run (~10 min)")
    parser.add_argument("--workers", type=int,
                        default=min(5, os.cpu_count() or 4),
                        help="Parallel training workers (default: min(5, cpu_count))")
    args = parser.parse_args()

    n_episodes = 20 if args.quick else 100

    print("=" * 60)
    print("MARL Supply Chain Experiment Suite")
    print(f"Workers: {args.workers} | Quick: {args.quick}")
    print("=" * 60)

    # Phase 1: classical baselines (tuning is fast, run sequentially)
    for demand_mode in DEMAND_MODES:
        print(f"\n--- Baselines: {demand_mode} ---")
        run_baselines(demand_mode, n_episodes, args.quick)

    # Phase 2: parallel RL training, then eval — one demand_mode at a time
    # (running both demand modes fully in parallel would use 2x workers)
    for demand_mode in DEMAND_MODES:
        print(f"\n{'='*60}")
        print(f"Training phase: {demand_mode}")
        print(f"{'='*60}")
        train_all_parallel(demand_mode, args.quick, args.workers)

        print(f"\n--- Evaluating RL agents: {demand_mode} ---")
        eval_all(demand_mode, n_episodes)

    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("Run: python scripts/make_figures.py && python scripts/make_tables.py")


if __name__ == "__main__":
    main()
