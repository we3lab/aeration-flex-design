"""Step 1: fit tank and blower cost curves from data/capex/. Writes output_data/smoothed_tank_coefficients.json and
SI fig_s5-s7."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from helpers.tank_cost import (
    get_tank_cost,
    get_tank_cost_composite,
    get_tank_cost_literature_by_vol,
    get_liquid_tank_cost,
    quadratic_cost_function,
    SMOOTHED_COEFFS_PATH,
)
from helpers.parameters import DATA_DIR, SI_PLOTS_DIR
from helpers.tariffs import to_usd_2025
from helpers.capex_lifetime_cost_energy_metrics import cost_compressor, cost_blower
from helpers.plotting import save_fig


STEEL_MAX_PRESSURE = 20.0  # MPa: composite always wins above ~14 MPa, and steel here needs
# hundreds of tiny tanks (very slow to solve), so skip steel above this in the envelope fit


def load_cost_csv(filename, label_col="label"):
    """Load cost CSV, convert cost_usd column to 2025 USD."""
    csv_path = os.path.join(DATA_DIR, "capex", filename)
    df = pd.read_csv(csv_path).dropna(subset=["cost_usd"])
    df["cost_2025"] = df.apply(lambda r: to_usd_2025(r["cost_usd"], int(r["year"])), axis=1)
    if label_col in df.columns:
        df["_label"] = df[label_col]
    return df


def scatter_annotate(ax, x, y, labels=None, fontsize=9, **scatter_kwargs):
    """Scatter plot with text annotations from a labels series."""
    ax.scatter(x, y, **scatter_kwargs)
    if labels is not None:
        for xi, yi, lbl in zip(x, y, labels):
            ax.annotate(
                lbl,
                xy=(xi, yi),
                xytext=(4, 0),
                textcoords="offset points",
                color="black",
                fontsize=fontsize,
            )


def create_smoothed_cost_function():
    """Fit quadratic cost curves to the min(steel, composite) cost envelope per pressure.

    At each (pressure, volume) we cost both a steel ASME vessel and a carbon-fiber
    composite vessel and keep the cheaper feasible one, so the steel->composite transition
    emerges where the two are equal instead of at a hard-coded pressure. Cost is ~linear
    in volume, so a handful of log-spaced volumes is enough for the quadratic fit. Steel
    needs impractically many tanks above STEEL_MAX_PRESSURE and never wins there, so we
    skip its (slow) solve above that pressure.
    """
    pressures = np.linspace(1, 70, 20)
    volumes = np.logspace(1, 4, 8)

    collected = {
        P: {k: [] for k in ["V", "cost", "n_tanks", "thickness", "length", "aspect_ratio"]}
        for P in pressures
    }

    for P in pressures:
        for V in volumes:
            candidates = [get_tank_cost_composite(V, P)]
            if P <= STEEL_MAX_PRESSURE:
                candidates.append(get_tank_cost(V, P))
            candidates = [c for c in candidates if not np.isnan(c[0]) and c[0] > 0]
            if not candidates:
                continue
            cost, n, t, L, AR = min(candidates, key=lambda c: c[0])
            for key, val in zip(
                ["V", "cost", "n_tanks", "thickness", "length", "aspect_ratio"],
                [V, cost, n, t, L, AR],
            ):
                collected[P][key].append(val)

    coefficients = {}
    for P, d in collected.items():
        V = np.array(d["V"])
        if len(V) < 3:
            continue
        try:
            pc = {}
            for key in ["cost", "n_tanks", "thickness", "length", "aspect_ratio"]:
                pc[key], _ = curve_fit(quadratic_cost_function, V, np.array(d[key]))
            a, b, _ = pc["cost"]
            if 2 * a * V.min() + b < 0 or 2 * a * V.max() + b < 0:
                print(f"WARNING: non-monotonic cost fit at P={P:.1f} MPa")
            coefficients[P] = pc
            print(f"P={P:.1f} MPa: cost = {a:.2e}·V² + {b:.2e}·V + {pc['cost'][2]:.2e}")
        except Exception as e:
            print(f"Fit failed at P={P:.1f} MPa: {e}")

    print(f"Fitted {len(coefficients)} pressure points")
    return coefficients


def _smoothed_cost(V, P, coefficients):
    """Evaluate smoothed quadratic cost (min-cost envelope) at the nearest sampled pressure."""
    if not coefficients:
        return np.nan
    closest = min(coefficients, key=lambda p: abs(p - P))
    return quadratic_cost_function(V, *coefficients[closest]["cost"])


def plot_tank_costs():
    """fig_s5: gas tank cost vs volume at P = 1, 10, 70 MPa (left column) and vs pressure at V = 10, 100,
    1000 m³ (right column)."""
    coefficients = create_smoothed_cost_function()
    # Save fitted coefficients to JSON for use by the optimizer
    serializable = {str(P): {k: v.tolist() for k, v in c.items()} for P, c in coefficients.items()}
    with open(SMOOTHED_COEFFS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)

    csv_data = load_cost_csv("tank_cost_data.csv")
    volumes = np.logspace(0, 3, 50)

    # Shin and Ha 2023: composite H2 tank costs at 350–700 bar
    # 400–700 USD/kg H2, density 17–33 kg H2/m3: 6,800–23,100 USD/m3 (2023)
    shin_lo = to_usd_2025(6800, 2023)
    shin_hi = to_usd_2025(23100, 2023)

    pressures_plot = [1, 10, 70]
    volumes_plot = [10, 100, 1000]
    pressure_range = np.logspace(0, np.log10(70), 50)

    fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))

    # Left column: cost vs volume at fixed pressures
    for ax, P in zip(axes[:, 0], pressures_plot):
        composite_costs = np.array([get_tank_cost_composite(V, P)[0] for V in volumes])
        lit_by_vol = np.array([get_tank_cost_literature_by_vol(V) for V in volumes])
        lit_pres = np.array([get_tank_cost(V, P, method="literature")[0] for V in volumes])
        smooth_costs = np.array([_smoothed_cost(V, P, coefficients) for V in volumes])

        if P <= STEEL_MAX_PRESSURE:
            steel_costs = np.array([get_tank_cost(V, P)[0] for V in volumes])
            steel_valid = ~np.isnan(steel_costs)
            if steel_valid.any():
                ax.loglog(
                    volumes[steel_valid],
                    steel_costs[steel_valid] / 1000,
                    "b-",
                    lw=2,
                    label="Steel tank model",
                )
        ax.loglog(volumes, composite_costs / 1000, "c-", lw=2, label="Composite tank model")
        ax.loglog(volumes, lit_by_vol / 1000, "g:", lw=2, label="Zheng 2022 (vol-based)")

        lit_pres_valid = ~np.isnan(lit_pres)
        if lit_pres_valid.any():
            ax.loglog(
                volumes[lit_pres_valid],
                lit_pres[lit_pres_valid] / 1000,
                color="brown",
                ls="-.",
                lw=2,
                label="Houssainy (pressure-aware)",
            )

        smooth_valid = ~np.isnan(smooth_costs)
        if smooth_valid.any():
            ax.loglog(
                volumes[smooth_valid],
                smooth_costs[smooth_valid] / 1000,
                "m--",
                lw=2,
                label="Smoothed fit (min-cost envelope)",
            )

        if P == 70:
            ax.fill_between(
                volumes,
                shin_lo * volumes / 1000,
                shin_hi * volumes / 1000,
                alpha=0.3,
                color="orange",
                label="Shin 2023 (350–700 bar composite)",
            )

        mask = np.abs(csv_data["Max pressure (MPa)"].values - P) < 0.1
        if mask.any():
            scatter_annotate(
                ax,
                csv_data["Tank volume (m3)"].values[mask],
                csv_data["cost_2025"].values[mask] / 1000,
                labels=csv_data["_label"].values[mask],
                color="gray",
                s=64,
                alpha=0.7,
                label="Data (2025 USD)",
                fontsize=12,
            )

        ax.set_xlabel("Total Tank Volume (m³)", fontsize=14)
        ax.set_ylabel("Cost (kUSD)", fontsize=14)
        ax.set_title(f"P = {P} MPa", fontsize=14)
        ax.set_ylim(1e0, 1e4)
        ax.tick_params(labelsize=13)

    # Right column: cost vs pressure at fixed volumes
    for ax, V in zip(axes[:, 1], volumes_plot):
        composite_costs = np.array([get_tank_cost_composite(V, P)[0] for P in pressure_range])
        lit_costs = np.array([get_tank_cost_literature_by_vol(V) for _ in pressure_range])
        lit_pres = np.array([get_tank_cost(V, P, method="literature")[0] for P in pressure_range])
        smooth_costs = np.array([_smoothed_cost(V, P, coefficients) for P in pressure_range])

        steel_pr = pressure_range[pressure_range <= STEEL_MAX_PRESSURE]
        steel_costs = np.array([get_tank_cost(V, P)[0] for P in steel_pr])
        steel_valid = ~np.isnan(steel_costs)
        if steel_valid.any():
            ax.loglog(
                steel_pr[steel_valid],
                steel_costs[steel_valid] / 1000,
                "b-",
                lw=2,
                label="Steel tank model",
            )
        ax.loglog(pressure_range, composite_costs / 1000, "c-", lw=2, label="Composite tank model")
        ax.loglog(pressure_range, lit_costs / 1000, "g:", lw=2, label="Zheng 2022 (vol-based)")

        lit_pres_valid = ~np.isnan(lit_pres)
        if lit_pres_valid.any():
            ax.loglog(
                pressure_range[lit_pres_valid],
                lit_pres[lit_pres_valid] / 1000,
                color="brown",
                ls="-.",
                lw=2,
                label="Houssainy (pressure-aware)",
            )

        smooth_valid = ~np.isnan(smooth_costs)
        if smooth_valid.any():
            ax.loglog(
                pressure_range[smooth_valid],
                smooth_costs[smooth_valid] / 1000,
                "m--",
                lw=2,
                label="Smoothed fit (min-cost envelope)",
            )

        # Shin 2023: horizontal band across the composite-relevant pressure range (350-700 bar)
        shin_pressures = pressure_range[pressure_range >= 35.0]
        if shin_pressures.size > 0:
            ax.fill_between(
                shin_pressures,
                shin_lo * V / 1000,
                shin_hi * V / 1000,
                alpha=0.3,
                color="orange",
                label="Shin 2023 (350–700 bar composite)",
            )

        mask = np.abs(csv_data["Tank volume (m3)"].values - V) < V * 0.1
        if mask.any():
            scatter_annotate(
                ax,
                csv_data["Max pressure (MPa)"].values[mask],
                csv_data["cost_2025"].values[mask] / 1000,
                labels=csv_data["_label"].values[mask],
                color="gray",
                s=64,
                alpha=0.7,
                label="Data (2025 USD)",
                fontsize=12,
            )

        ax.set_xlabel("Pressure (MPa)", fontsize=14)
        ax.set_ylabel("Cost (kUSD)", fontsize=14)
        ax.set_title(f"V = {V} m³ (water volume)", fontsize=14)
        ax.set_ylim(1e0, 1e4)
        ax.tick_params(labelsize=13)

    # One legend for all panels; each panel only draws the series that apply to it
    handles = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    plt.tight_layout()
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        fontsize=13,
        frameon=False,
    )
    os.makedirs(SI_PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(SI_PLOTS_DIR, "fig_s5_gas_tank_cost.png")
    save_fig(plt.gcf(), output_path)
    print(f"Tank cost plot saved to: {output_path}")


def plot_liquid_tank_costs():
    """Liquid O2 tank cost vs storage capacity."""
    csv_data = load_cost_csv("liquid_tank_cost_data.csv", label_col="source")
    Nmax_range = np.logspace(4, 7, 100)

    costs_mat = np.array([get_liquid_tank_cost(N, method="materials")[0] for N in Nmax_range])
    costs_emp = np.array([get_liquid_tank_cost(N, method="empirical")[0] for N in Nmax_range])
    valid = ~np.isnan(costs_mat)
    avg = np.nanmean([costs_mat, costs_emp], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.loglog(Nmax_range[valid], costs_mat[valid] / 1000, "b-", lw=2, label="Materials-based model")
    ax.loglog(Nmax_range[valid], costs_emp[valid] / 1000, "r-", lw=2, label="Empirical model")
    ax.loglog(Nmax_range[valid], avg[valid] / 1000, "m--", lw=2, label="Average of methods")
    scatter_annotate(
        ax,
        csv_data["Nmax (mol)"],
        csv_data["cost_2025"] / 1000,
        labels=csv_data["_label"],
        color="k",
        s=64,
        label="Literature data (2025 USD)",
    )
    ax.set_xlabel("Maximum Storage Capacity (mol)", fontsize=12)
    ax.set_ylabel("Cost (kUSD)", fontsize=12)
    ax.set_xlim([1e4, 1e7])
    ax.set_ylim([1e0, 1e4])
    ax.legend(fontsize=11)
    plt.tight_layout()
    os.makedirs(SI_PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(SI_PLOTS_DIR, "fig_s6_liquid_tank_cost.png")
    save_fig(plt.gcf(), output_path)


def plot_blower_costs():
    """Fit Cost = a * power_kw^b (2025 USD) to the blower data (the coefficients in cost_blower), and compare
    compressor_cost (Matche), blower_cost (fitted) and the literature data."""
    csv_data = load_cost_csv("blower_cost_data.csv")
    data_powers = csv_data["max_power"].values.astype(float)
    data_costs = csv_data["cost_2025"].values
    b_fit, log_a = np.polyfit(np.log(data_powers), np.log(data_costs), 1)
    a_fit = np.exp(log_a)
    print(
        f"Blower Cost: {a_fit:.1f} * power^{b_fit:.3f}  "
        f"(R²={np.corrcoef(np.log(data_powers), np.log(data_costs))[0, 1]**2:.4f})"
    )
    for p, c in zip(data_powers, data_costs):
        pred = a_fit * p**b_fit
        print(
            f"  {p:>6.0f} kW: actual=${c:>10,.0f}  predicted=${pred:>10,.0f}  error={100*(pred-c)/c:+.1f}%"
        )

    powers = np.linspace(10, 2000, 200)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(
        powers,
        [cost_compressor(p) / 1000 for p in powers],
        "b--",
        lw=2,
        label="Compressor Cost (Matche)",
    )
    ax.plot(
        powers, [cost_blower(p) / 1000 for p in powers], "r-", lw=2, label="Blower Cost (fitted)"
    )
    scatter_annotate(
        ax,
        csv_data["max_power"],
        csv_data["cost_2025"] / 1000,
        labels=csv_data["_label"],
        color="black",
        s=80,
        zorder=5,
        label="Literature data (2025 USD)",
    )
    ax.set_xlabel("Power (kW)", fontsize=12)
    ax.set_ylabel("Cost (kUSD)", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    os.makedirs(SI_PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(SI_PLOTS_DIR, "fig_s7_blower_cost.png")
    save_fig(plt.gcf(), output_path)
    print(f"Blower cost plot saved to: {output_path}")


if __name__ == "__main__":
    plot_tank_costs()
    plot_liquid_tank_costs()
    plot_blower_costs()
