"""Sweeps the compressor-coefficient damping alpha vs horizon_days, plus an 8-day solve without DPP.
Writes fig_s10."""

import csv
import glob
import json
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Different intermediate folder than shared paper intermediates
os.environ["AERATION_INTERMEDIATE_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output_data", "intermediate", "solver_comparison"
)

from helpers.config_keys import cb_palette, get_base_key, get_config_name
from helpers.parameters import (
    CONFIG_DIR,
    OUTPUT_DATA_DIR,
    SI_PLOTS_DIR,
    SOLVER,
    get_intermediate_dir,
)
from helpers.plotting import save_fig
from step3_run_all import run_all_configs

# use a design where gas storage actually charges
CE_CONFIG_PATH = os.path.join(CONFIG_DIR, "test_air.json")
OUTPUT_DIR = os.path.join(OUTPUT_DATA_DIR, "solver_comparison")
CE_CSV_PATH = os.path.join(OUTPUT_DIR, "solver_comparison_test_air_results.csv")


def get_run_name(horizon_days):
    return f"temp_config_horizon_{horizon_days}"


def run_and_record(solve_mode, horizon_days, solver_settings):
    """Solve test_air for one sweep cell from an empty cache and append its metrics to the results CSV."""
    print(f"  h={horizon_days} {solve_mode} {solver_settings}")
    with open(CE_CONFIG_PATH) as f:
        config = json.load(f)
    config["design_space"]["horizon_days"] = horizon_days
    config["design_space"]["solver_settings"] = solver_settings
    config["script_settings"]["run_plotting"] = False
    config["script_settings"]["run_solve_optimization"] = True
    config["base_run_name"] = get_run_name(horizon_days)
    result_dir = os.path.join(OUTPUT_DATA_DIR, config["base_run_name"])
    for path in glob.glob(os.path.join(result_dir, "*___*.json")):
        os.remove(path)
    # Empty the solve cache for this config and horizon so every sweep cell re-solves
    facility = config["facility"]
    for base_wwtp_key, upgrade_key_list in config["upgrade_keys"].items():
        for upgrade_key in upgrade_key_list:
            for tariff_key in config["tariff_keys_to_run"]:
                for sc in config["summer_config_template"]:
                    combined = f"{get_base_key(base_wwtp_key)}__{facility['scale_mgd']}__{sc['multiplier']}__{sc['smoothing']}"
                    h_dir = get_intermediate_dir(
                        facility["data_name"],
                        get_config_name(combined, upgrade_key, tariff_key),
                        horizon_days,
                    )
                    if os.path.isdir(h_dir):
                        shutil.rmtree(h_dir)

    t0 = time.time()
    run_all_configs([config])
    elapsed = time.time() - t0

    # Report the optimal (min lifetime cost) design, so the sweep compares a consistent, meaningful design rather
    # than an arbitrary one. tank_type guards interpretation: if liquid wins the medium competition, there is no
    # bilinear charging term and RMSE is expected empty.
    designs = []
    for path in glob.glob(os.path.join(result_dir, "*___*.json")):
        with open(path) as f:
            designs += json.load(f)["all_results"].values()
    designs = [d for d in designs if d["lifetime_cost"].get("total") is not None]
    best = min(designs, key=lambda d: d["lifetime_cost"]["total"], default={})
    month_time = best.get("mean_month_solve_time")
    approx_error = best.get("approx_error") or {}
    entry = {
        "solve_mode": solve_mode,
        "solver": SOLVER,
        "alpha": solver_settings.get("alpha"),
        "horizon_days": horizon_days,
        "success": True,
        "avg_month_time": int(round(month_time)) if month_time else None,
        "lifetime_cost": best.get("lifetime_cost", {}).get("total"),
        "total_elapsed": int(round(elapsed)),
        "approx_error_rmse_mean": approx_error.get("rmse_mean"),
        "approx_error_rmse_min": approx_error.get("rmse_min"),
        "approx_error_rmse_max": approx_error.get("rmse_max"),
        "tank_type": best.get("tank_metrics", {}).get("tank_type"),
    }
    write_header = not os.path.exists(CE_CSV_PATH) or os.path.getsize(CE_CSV_PATH) == 0
    with open(CE_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(entry))
        if write_header:
            writer.writeheader()
        writer.writerow(entry)


def plot_ce_results():
    """fig_s10: top, lifetime cost, solve time and approximation error over alpha x horizon_days; bottom,
    8-day rolling vs full-month lifetime cost, and solve time with and without DPP on the rolling horizon.
    """
    all_rows = pd.read_csv(CE_CSV_PATH)
    df = all_rows[all_rows["solve_mode"] == "iterative_coeff"].copy()
    df["horizon_days"] = df["horizon_days"].astype(int)
    df["lc_col"] = df["lifetime_cost"] / 1000000
    specs = [
        ("lc_col", "Lifetime cost ($M)", "{:.2f}", "viridis"),
        ("avg_month_time", "Solve time per month (s)", "{:.0f}", "magma"),
        ("approx_error_rmse_mean", "Net power RMSE (kW)", "{:.2f}", "plasma"),
    ]
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.6)
    for k, (col, label, fmt, cmap) in enumerate(specs):
        ax = fig.add_subplot(grid[0, 2 * k : 2 * k + 2])
        piv = df.pivot_table(index="alpha", columns="horizon_days", values=col, aggfunc="mean")
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel("Horizon (days)", fontsize=16)
        if k == 0:  # alpha axis shared across the row
            ax.set_ylabel("Damping α", fontsize=16)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=15)
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                val = piv.values[i, j]
                if not np.isnan(val):
                    # Dark text on the bright end of each colormap
                    color = "white" if im.norm(val) < 0.6 else "black"
                    ax.text(
                        j, i, fmt.format(val), ha="center", va="center", color=color, fontsize=15
                    )
        colorbar = fig.colorbar(im, ax=ax)
        colorbar.set_label(label, fontsize=16)
        colorbar.set_ticks([])  # every cell is labeled with its value

    # Bottom row at the default alpha 0.5. DPP is on by default below 28-day horizons, so the 8-day sweep cell
    # is the DPP run and the full month solves without it
    def month_row(mode, horizon_days):
        rows = all_rows[
            (all_rows["solve_mode"] == mode) & (all_rows["horizon_days"] == horizon_days)
        ]
        if mode == "iterative_coeff":
            rows = rows[rows["alpha"] == 0.5]
        return rows.iloc[0] if len(rows) else None

    rolling, full = month_row("iterative_coeff", 8), month_row("iterative_coeff", 31)
    rolling_no_dpp = month_row("no_dpp", 8)
    ax_cost = fig.add_subplot(grid[1, 0:2])
    costs = [rolling["lifetime_cost"] / 1e6, full["lifetime_cost"] / 1e6]
    ax_cost.bar(
        ["8-day rolling", "Full month"], costs, color=[cb_palette[0], cb_palette[2]], width=0.6
    )
    gap_pct = 100 * (costs[0] / costs[1] - 1)
    ax_cost.text(0, costs[0] * 1.02, f"+{gap_pct:.1f}%", ha="center", va="bottom", fontsize=16)
    ax_cost.set_ylim(0, max(costs) * 1.15)
    ax_cost.set_ylabel("Lifetime cost ($M)", fontsize=16)

    ax_time = fig.add_subplot(grid[1, 3:6])
    bars = [("8-day rolling\n(DPP)", rolling, cb_palette[0])]
    if rolling_no_dpp is not None:
        bars.append(("8-day rolling\n(no DPP)", rolling_no_dpp, cb_palette[6]))
    bars.append(("Full month", full, cb_palette[2]))
    ax_time.bar(
        [label for label, _, _ in bars],
        [row["avg_month_time"] for _, row, _ in bars],
        color=[color for _, _, color in bars],
        width=0.6,
    )
    ax_time.set_ylabel("Solve time per month (s)", fontsize=16)
    for ax in (ax_cost, ax_time):
        ax.tick_params(labelsize=15)
        ax.spines[["top", "right"]].set_visible(False)
    os.makedirs(SI_PLOTS_DIR, exist_ok=True)
    save_fig(fig, os.path.join(SI_PLOTS_DIR, "fig_s10_hyperparameters.png"))


if __name__ == "__main__":
    # Multiple horizons populate the horizon-vs-alpha heatmap.
    horizon_days_values = [3, 8, 31]  # [8] for fig s10
    alpha_values = [0.3, 0.5, 0.7]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Sweep cells already in the results CSV are skipped, so re-running only redraws the figure
    done = set()
    if os.path.exists(CE_CSV_PATH):
        prev = pd.read_csv(CE_CSV_PATH)
        done = set(zip(prev["solve_mode"], prev["horizon_days"], prev["alpha"].fillna(0)))
    try:
        for hd in horizon_days_values:
            run_dir = os.path.join(OUTPUT_DATA_DIR, get_run_name(hd))
            os.makedirs(run_dir, exist_ok=True)  # folder for each horizon
            shutil.copy2(os.path.join(OUTPUT_DATA_DIR, "test_air", "day_profiles.json"), run_dir)
        print(f"CE sweep: {len(horizon_days_values) * len(alpha_values)} combinations")
        for hd in horizon_days_values:
            for alpha in alpha_values:
                if ("iterative_coeff", hd, alpha) not in done:
                    run_and_record("iterative_coeff", hd, {"alpha": alpha})
        # 8-day rolling horizon without DPP at the default alpha; the alpha 0.5 sweep cell above is the DPP run
        if ("no_dpp", 8, 0) not in done:
            run_and_record("no_dpp", 8, {"ignore_dpp": True})
        plot_ce_results()
    finally:  # including ctrl+c cancel
        # Remove the temporary dirs
        for hd in horizon_days_values:
            shutil.rmtree(os.path.join(OUTPUT_DATA_DIR, get_run_name(hd)), ignore_errors=True)
    print(f"CE results saved to {CE_CSV_PATH}")
