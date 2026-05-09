from __future__ import annotations
"""Generate LaTeX and CSV tables from experiment results."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "results/logs"
TABLES_DIR = PROJECT_ROOT / "results/tables"

METHODS = ["base_stock", "ss_policy", "idqn", "cdqn", "vdn"]
DEMAND_MODES = ["stationary", "nonstationary"]
SEEDS = [0, 1, 2, 3, 4]

METHOD_LABELS = {
    "base_stock": "Base-Stock",
    "ss_policy": "(s,S) Policy",
    "idqn": "IDQN",
    "cdqn": "CDQN",
    "vdn": "VDN",
}


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


def fmt_cost(mean: float, std: float) -> str:
    return f"{mean:.1f} +/- {std:.1f}"


def fmt_ratio(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def fmt_freq(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def fmt_cost_tex(mean: float, std: float) -> str:
    return f"${mean:.1f} \\pm {std:.1f}$"


def fmt_ratio_tex(mean: float, std: float) -> str:
    return f"${mean:.3f} \\pm {std:.3f}$"


def fmt_freq_tex(mean: float, std: float) -> str:
    return f"${mean:.3f} \\pm {std:.3f}$"


# -----------------------------------------------------------------------
# Table 1: Main results
# -----------------------------------------------------------------------

def build_table1():
    """
    Table 1: methods x (stationary cost, nonstationary cost, bullwhip avg, stockout freq)
    Columns: Method | Cost_stat | Cost_nonstat | Bullwhip_stat | Bullwhip_nonstat | Stockout_stat | Stockout_nonstat
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    columns_csv = [
        "method",
        "cost_stationary", "cost_nonstationary",
        "bullwhip_stationary", "bullwhip_nonstationary",
        "stockout_stationary", "stockout_nonstationary",
    ]
    rows_csv = []

    columns_tex = [
        "Method",
        "Cost (stat.)", "Cost (nonstat.)",
        "Bullwhip (stat.)", "Bullwhip (nonstat.)",
        "Stockout (stat.)", "Stockout (nonstat.)",
    ]
    rows_tex = []

    for method in METHODS:
        row_csv = {"method": METHOD_LABELS[method]}
        row_tex = [METHOD_LABELS[method]]

        for demand_mode in DEMAND_MODES:
            metrics_list = load_eval_metrics(method, demand_mode)
            if metrics_list:
                costs = [m["total_cost_mean"] for m in metrics_list]
                bws = [m["bullwhip_ratio_avg"] for m in metrics_list]
                sos = [m["stockout_frequency"] for m in metrics_list]

                cost_m, cost_s = np.mean(costs), np.std(costs)
                bw_m, bw_s = np.mean(bws), np.std(bws)
                so_m, so_s = np.mean(sos), np.std(sos)

                row_csv[f"cost_{demand_mode}"] = fmt_cost(cost_m, cost_s)
                row_csv[f"bullwhip_{demand_mode}"] = fmt_ratio(bw_m, bw_s)
                row_csv[f"stockout_{demand_mode}"] = fmt_freq(so_m, so_s)

                row_tex.append(fmt_cost_tex(cost_m, cost_s))
                row_tex.append(fmt_ratio_tex(bw_m, bw_s))
                row_tex.append(fmt_freq_tex(so_m, so_s))
            else:
                row_csv[f"cost_{demand_mode}"] = "N/A"
                row_csv[f"bullwhip_{demand_mode}"] = "N/A"
                row_csv[f"stockout_{demand_mode}"] = "N/A"
                row_tex.extend(["N/A", "N/A", "N/A"])

        rows_csv.append(row_csv)
        rows_tex.append(row_tex)

    # Reorder tex columns to match (cost stat, cost nonstat, bullwhip stat, bullwhip nonstat, ...)
    # The current tex rows have: cost_stat, bullwhip_stat, stockout_stat, cost_nonstat, bullwhip_nonstat, stockout_nonstat
    # Reorder to: cost_stat, cost_nonstat, bullwhip_stat, bullwhip_nonstat, stockout_stat, stockout_nonstat
    def reorder_tex_row(r):
        return [r[0], r[1], r[4], r[2], r[5], r[3], r[6]]

    rows_tex = [reorder_tex_row(r) for r in rows_tex]

    # Save CSV
    csv_path = TABLES_DIR / "table1_main_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns_csv)
        writer.writeheader()
        writer.writerows(rows_csv)
    print(f"  Saved {csv_path}")

    # Save LaTeX
    tex_path = TABLES_DIR / "table1_main_results.tex"
    with open(tex_path, "w") as f:
        ncols = len(columns_tex)
        col_spec = "l" + "c" * (ncols - 1)
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Main Results: Mean $\\pm$ Std across 5 seeds.}\n")
        f.write("\\label{tab:main_results}\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(columns_tex) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows_tex:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  Saved {tex_path}")


# -----------------------------------------------------------------------
# Table 2: Baseline parameters
# -----------------------------------------------------------------------

def build_table2():
    """
    Table 2: Baseline params
    Rows: base_stock stationary, base_stock nonstationary, ss_policy stationary, ss_policy nonstationary
    Columns: Method, Setting, Parameters, Mean Cost
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    params_path = TABLES_DIR / "baseline_params.json"
    if not params_path.exists():
        print(f"  [warn] {params_path} not found, skipping Table 2")
        return

    with open(params_path) as f:
        all_params = json.load(f)

    columns_csv = ["method", "demand_mode", "parameters", "mean_cost"]
    rows_csv = []
    rows_tex = []

    entries = [
        ("base_stock", "stationary"),
        ("base_stock", "nonstationary"),
        ("ss_policy", "stationary"),
        ("ss_policy", "nonstationary"),
    ]

    for method, demand_mode in entries:
        key = f"{method}_{demand_mode}"
        if key not in all_params:
            rows_csv.append({
                "method": METHOD_LABELS[method],
                "demand_mode": demand_mode,
                "parameters": "N/A",
                "mean_cost": "N/A",
            })
            rows_tex.append([METHOD_LABELS[method], demand_mode.capitalize(), "N/A", "N/A"])
            continue

        params = all_params[key]
        mean_cost = params.get("mean_cost", float("nan"))

        if method == "base_stock":
            param_str = f"S={params['S_levels']}"
            param_tex = f"$S={params['S_levels']}$"
        else:
            param_str = f"s={params['s_levels']}, S={params['S_levels']}"
            param_tex = f"$s={params['s_levels']},\\ S={params['S_levels']}$"

        rows_csv.append({
            "method": METHOD_LABELS[method],
            "demand_mode": demand_mode,
            "parameters": param_str,
            "mean_cost": f"{mean_cost:.1f}",
        })
        rows_tex.append([
            METHOD_LABELS[method],
            demand_mode.capitalize(),
            param_tex,
            f"{mean_cost:.1f}",
        ])

    # Save CSV
    csv_path = TABLES_DIR / "table2_baseline_params.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns_csv)
        writer.writeheader()
        writer.writerows(rows_csv)
    print(f"  Saved {csv_path}")

    # Save LaTeX
    tex_path = TABLES_DIR / "table2_baseline_params.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Tuned Baseline Parameters.}\n")
        f.write("\\label{tab:baseline_params}\n")
        f.write("\\begin{tabular}{llll}\n")
        f.write("\\toprule\n")
        f.write("Method & Setting & Parameters & Mean Cost \\\\\n")
        f.write("\\midrule\n")
        for row in rows_tex:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  Saved {tex_path}")


def main():
    print("Generating tables...")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("  Table 1: Main results")
    build_table1()

    print("  Table 2: Baseline parameters")
    build_table2()

    print("Done. Tables saved to results/tables/")


if __name__ == "__main__":
    main()
