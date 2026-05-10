from __future__ import annotations
"""Generate all report figures from experiment logs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import json

METHODS = ["base_stock", "ss_policy", "idqn", "cdqn", "vdn"]
RL_METHODS = ["idqn", "cdqn", "vdn"]
DEMAND_MODES = ["stationary", "nonstationary"]
SEEDS = [0, 1, 2, 3, 4]

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "results/figures"
LOGS_DIR = PROJECT_ROOT / "results/logs"

METHOD_COLORS = {
    "base_stock": "#7f7f7f",
    "ss_policy": "#bcbd22",
    "idqn": "#1f77b4",
    "cdqn": "#d62728",
    "vdn": "#2ca02c",
}
METHOD_LABELS = {
    "base_stock": "Base-Stock",
    "ss_policy": "(s,S) Policy",
    "idqn": "IDQN",
    "cdqn": "CDQN",
    "vdn": "VDN",
}
NODE_LABELS = ["Supplier", "Warehouse", "Retailer"]


def load_eval_metrics(method: str, demand_mode: str) -> list[dict]:
    """Load eval metrics for all seeds. Returns list of metric dicts."""
    results = []
    for seed in SEEDS:
        path = LOGS_DIR / f"{method}_{demand_mode}_seed{seed}_eval.csv"
        if path.exists():
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                row = {k: float(v) for k, v in rows[0].items()}
                results.append(row)
    return results


def load_training_curves(method: str, demand_mode: str) -> dict:
    """Load training curve CSVs for all seeds. Returns {seed: list of dicts}."""
    curves = {}
    for seed in SEEDS:
        path = LOGS_DIR / f"{method}_{demand_mode}_seed{seed}.csv"
        if path.exists():
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                curves[seed] = rows
    return curves


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def save_fig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"  Saved {name}.png + .pdf")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 1: Training curves
# -----------------------------------------------------------------------

def figure1_training_curves():
    """Training curves (2x1 panel: stationary | non-stationary)."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax_idx, demand_mode in enumerate(DEMAND_MODES):
        ax = axes[ax_idx]

        # Horizontal baselines
        for method in ["base_stock", "ss_policy"]:
            metrics_list = load_eval_metrics(method, demand_mode)
            if not metrics_list:
                continue
            mean_cost = np.mean([m["total_cost_mean"] for m in metrics_list])
            ax.axhline(
                mean_cost,
                color=METHOD_COLORS[method],
                linestyle="--",
                linewidth=1.5,
                label=METHOD_LABELS[method],
                alpha=0.8,
            )

        # RL training curves
        for method in RL_METHODS:
            curves = load_training_curves(method, demand_mode)
            if not curves:
                print(f"  [warn] No training curves for {method} {demand_mode}")
                continue

            all_steps = None
            all_costs = []
            for seed, rows in curves.items():
                steps = [int(r["step"]) for r in rows]
                costs = [float(r["mean_eval_cost"]) for r in rows]
                if all_steps is None:
                    all_steps = steps
                all_costs.append(costs)

            if all_steps is None or not all_costs:
                continue

            # Align lengths
            min_len = min(len(c) for c in all_costs)
            all_costs = np.array([c[:min_len] for c in all_costs])
            steps_arr = np.array(all_steps[:min_len])

            mean_curve = all_costs.mean(axis=0)
            std_curve = all_costs.std(axis=0)

            ax.plot(
                steps_arr,
                mean_curve,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=2,
            )
            ax.fill_between(
                steps_arr,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=METHOD_COLORS[method],
                alpha=0.2,
            )

        ax.set_xlabel("Training steps")
        ax.set_ylabel("Mean eval cost")
        ax.set_title(demand_mode.capitalize())
        ax.legend(fontsize=8)

    fig.suptitle("Figure 1: Training Curves", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig1_training_curves")


# -----------------------------------------------------------------------
# Figure 2: Final cost comparison bar chart
# -----------------------------------------------------------------------

def figure2_cost_comparison():
    """Final cost comparison bar chart: 5 methods x 2 settings."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    n_methods = len(METHODS)
    n_modes = len(DEMAND_MODES)
    bar_width = 0.35
    group_gap = 0.1
    group_width = n_modes * bar_width + group_gap

    x = np.arange(n_methods) * (group_width + 0.2)

    mode_styles = [
        {"hatch": "", "alpha": 0.85, "label_suffix": " (stationary)"},
        {"hatch": "//", "alpha": 0.85, "label_suffix": " (nonstationary)"},
    ]

    any_data = False
    for mode_idx, demand_mode in enumerate(DEMAND_MODES):
        means = []
        stds = []
        for method in METHODS:
            metrics_list = load_eval_metrics(method, demand_mode)
            if metrics_list:
                costs = [m["total_cost_mean"] for m in metrics_list]
                means.append(np.mean(costs))
                stds.append(np.std(costs))
            else:
                means.append(0.0)
                stds.append(0.0)

        # Clip lower error bar at 0 — episode costs cannot be negative
        err_lower = [min(s, m) for m, s in zip(means, stds)]
        err_upper = stds

        offset = mode_idx * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=[err_lower, err_upper],
            capsize=4,
            color=[METHOD_COLORS[m] for m in METHODS],
            hatch=mode_styles[mode_idx]["hatch"],
            alpha=mode_styles[mode_idx]["alpha"],
            label=demand_mode.capitalize(),
            error_kw={"elinewidth": 1.2},
        )
        if any(m > 0 for m in means):
            any_data = True

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=15, ha="right")
    ax.set_ylabel("Mean episode cost")
    ax.set_title("Figure 2: Final Cost Comparison")
    ax.legend()

    if not any_data:
        ax.text(0.5, 0.5, "No data available yet\nRun experiments first",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")

    fig.tight_layout()
    save_fig(fig, "fig2_cost_comparison")


# -----------------------------------------------------------------------
# Figure 3: Bullwhip ratio per node
# -----------------------------------------------------------------------

def figure3_bullwhip():
    """Bullwhip ratio per node, grouped by method, 2 subplots."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, demand_mode in enumerate(DEMAND_MODES):
        ax = axes[ax_idx]

        n_methods = len(METHODS)
        n_nodes = 3
        bar_width = 0.15
        x = np.arange(n_nodes)

        any_data = False
        for m_idx, method in enumerate(METHODS):
            metrics_list = load_eval_metrics(method, demand_mode)
            if not metrics_list:
                continue

            bw_vals = []
            for node_idx in range(3):
                key = f"bullwhip_ratio_node{node_idx}"
                vals = [m[key] for m in metrics_list if key in m]
                bw_vals.append(np.mean(vals) if vals else 0.0)

            offset = (m_idx - n_methods / 2) * bar_width + bar_width / 2
            ax.bar(
                x + offset,
                bw_vals,
                bar_width,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                alpha=0.85,
            )
            any_data = True

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Ratio=1")
        ax.set_xticks(x)
        ax.set_xticklabels(NODE_LABELS)
        ax.set_ylabel("Bullwhip Ratio")
        ax.set_title(demand_mode.capitalize())
        ax.legend(fontsize=8)

        if not any_data:
            ax.text(0.5, 0.5, "No data available yet",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")

    fig.suptitle("Figure 3: Bullwhip Ratio per Node", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig3_bullwhip_ratio")


# -----------------------------------------------------------------------
# Figure 4: Stockout frequency bar chart
# -----------------------------------------------------------------------

def figure4_stockout():
    """Stockout frequency bar chart (same structure as Figure 2)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    n_methods = len(METHODS)
    bar_width = 0.35
    group_gap = 0.1
    group_width = len(DEMAND_MODES) * bar_width + group_gap

    x = np.arange(n_methods) * (group_width + 0.2)

    mode_styles = [
        {"hatch": "", "alpha": 0.85},
        {"hatch": "//", "alpha": 0.85},
    ]

    any_data = False
    for mode_idx, demand_mode in enumerate(DEMAND_MODES):
        means = []
        stds = []
        for method in METHODS:
            metrics_list = load_eval_metrics(method, demand_mode)
            if metrics_list:
                vals = [m["stockout_frequency"] for m in metrics_list]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0.0)
                stds.append(0.0)

        offset = mode_idx * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=4,
            color=[METHOD_COLORS[m] for m in METHODS],
            hatch=mode_styles[mode_idx]["hatch"],
            alpha=mode_styles[mode_idx]["alpha"],
            label=demand_mode.capitalize(),
            error_kw={"elinewidth": 1.2},
        )
        if any(m > 0 for m in means):
            any_data = True

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=15, ha="right")
    ax.set_ylabel("Stockout Frequency")
    ax.set_title("Figure 4: Stockout Frequency")
    ax.legend()

    if not any_data:
        ax.text(0.5, 0.5, "No data available yet\nRun experiments first",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")

    fig.tight_layout()
    save_fig(fig, "fig4_stockout_frequency")


# -----------------------------------------------------------------------
# Figure 5: Inventory trajectories (one sample episode per method)
# -----------------------------------------------------------------------

def figure5_inventory_trajectories():
    """3 rows x 1 col: per-node inventory trajectories for each method."""
    from src.envs.supply_chain import SupplyChainEnv
    from src.utils.config import load_config

    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for method in METHODS:
        traj = _get_one_trajectory(method, demand_mode="stationary", seed=0)
        if traj is None:
            print(f"  [warn] No trajectory for {method}, skipping in Fig 5")
            continue

        inv_arr = np.array(traj["per_node_inventory"])  # (T, 3)
        for node_idx in range(3):
            axes[node_idx].plot(
                inv_arr[:, node_idx],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=1.5,
                alpha=0.85,
            )

    for node_idx, ax in enumerate(axes):
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"Inventory\n{NODE_LABELS[node_idx]}")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Figure 5: Inventory Trajectories (sample episode, stationary)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig5_inventory_trajectories")


# -----------------------------------------------------------------------
# Figure 6: Non-stationary adaptation (per-step costs)
# -----------------------------------------------------------------------

def figure6_nonstationary_adaptation():
    """Per-step costs on a non-stationary episode; dashed line at regime shift step 50."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    any_data = False
    for method in METHODS:
        traj = _get_one_trajectory(method, demand_mode="nonstationary", seed=0)
        if traj is None:
            print(f"  [warn] No trajectory for {method}, skipping in Fig 6")
            continue

        costs = traj["per_step_costs"]
        ax.plot(
            range(len(costs)),
            costs,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=1.5,
            alpha=0.85,
        )
        any_data = True

    ax.axvline(50, color="red", linestyle="--", linewidth=1.5, label="Regime shift (step 50)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Per-step total cost")
    ax.set_title("Figure 6: Non-Stationary Adaptation")
    ax.legend(fontsize=8)

    if not any_data:
        ax.text(0.5, 0.5, "No data available yet\nRun experiments first",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")

    fig.tight_layout()
    save_fig(fig, "fig6_nonstationary_adaptation")


# -----------------------------------------------------------------------
# Helper: get one trajectory for a method/setting
# -----------------------------------------------------------------------

def _get_one_trajectory(method: str, demand_mode: str, seed: int) -> dict | None:
    """Run one greedy episode and return the trajectory dict. Returns None on failure."""
    try:
        from src.evaluation.evaluator import evaluate
        from src.utils.config import load_config
        from src.envs.supply_chain import SupplyChainEnv

        if method in ("base_stock", "ss_policy"):
            params_path = PROJECT_ROOT / "results/tables/baseline_params.json"
            if not params_path.exists():
                return None
            with open(params_path) as f:
                all_params = json.load(f)

            key = f"{method}_{demand_mode}"
            if key not in all_params:
                return None
            params = all_params[key]

            if demand_mode == "stationary":
                env_config = load_config(PROJECT_ROOT / "configs/env_default.yaml")
            else:
                env_config = load_config(PROJECT_ROOT / "configs/env_nonstationary.yaml")

            if method == "base_stock":
                from src.agents.base_stock import BaseStockAgent
                agent = BaseStockAgent(
                    params["S_levels"],
                    max_order=env_config.get("max_order", 20),
                )
            else:
                from src.agents.ss_policy import SSPolicyAgent
                agent = SSPolicyAgent(
                    params["s_levels"],
                    params["S_levels"],
                    max_order=env_config.get("max_order", 20),
                )
            # For baselines, run multiple episodes and return the median-cost one
            result = evaluate(agent, dict(env_config), n_episodes=11, seed_offset=seed * 999999 + 1)
            costs = [sum(t["per_step_costs"]) for t in result["trajectories"]]
            median_idx = int(np.argsort(costs)[len(costs) // 2])
            return result["trajectories"][median_idx]
        else:
            import torch
            from src.agents.idqn import IDQNAgent
            from src.agents.cdqn import CDQNAgent
            from src.agents.vdn import VDNAgent

            ckpt = PROJECT_ROOT / f"results/checkpoints/{method}_{demand_mode}_seed{seed}_final.pt"
            if not ckpt.exists():
                return None

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
            env_config = config["env"]

            # Run multiple episodes across seeds and return the median-cost one
            all_trajs = []
            for s in SEEDS:
                ckpt_s = PROJECT_ROOT / f"results/checkpoints/{method}_{demand_mode}_seed{s}_final.pt"
                if not ckpt_s.exists():
                    continue
                agent.load(str(ckpt_s))
                r = evaluate(agent, dict(env_config), n_episodes=3, seed_offset=s * 999999 + 1)
                all_trajs.extend(r["trajectories"])
            if not all_trajs:
                return None
            costs = [sum(t["per_step_costs"]) for t in all_trajs]
            median_idx = int(np.argsort(costs)[len(costs) // 2])
            return all_trajs[median_idx]

    except Exception as e:
        print(f"  [warn] Could not get trajectory for {method} {demand_mode}: {e}")
        return None


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("Generating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("  Figure 1: Training curves")
    figure1_training_curves()

    print("  Figure 2: Cost comparison")
    figure2_cost_comparison()

    print("  Figure 3: Bullwhip ratio")
    figure3_bullwhip()

    print("  Figure 4: Stockout frequency")
    figure4_stockout()

    print("  Figure 5: Inventory trajectories")
    figure5_inventory_trajectories()

    print("  Figure 6: Non-stationary adaptation")
    figure6_nonstationary_adaptation()

    print("Done. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
