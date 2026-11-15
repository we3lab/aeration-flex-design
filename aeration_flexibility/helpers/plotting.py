"""Paper and SI figures from the result JSONs (generate_plots).
fig2 has cost curves and day profiles, fig3 cost bars, fig4 sensitivity heatmaps."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
import glob
import re
import shutil
from datetime import datetime
from collections import defaultdict

from PIL import Image
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.transforms import blended_transform_factory
from sklearn.isotonic import IsotonicRegression

from helpers.parameters import *
from helpers.config_keys import *
from helpers.tariffs import get_tariff_data
from helpers.tank_cost import HYDROGEN_TANK_SPECS, get_tank_cost_smoothed

from eeco.costs import detect_charge_periods, PEAK, HALF_PEAK, ENERGY


# fig2 column-ii: don't paint the increase/decrease fill for per-timestep changes below this
# fraction of baseline power — sub-shift round-trip noise (e.g. the ~1-2% air halo) reads as flat.
POWER_FILL_REL_TOL = 0.01
PROFILE_LINEWIDTH = 0.6  # for baseline/optimized
# fig2 column-ii: profile values smaller than this in magnitude are zeroed before plotting
PROFILE_ZERO_THRESHOLD = 50
# fig2 column-ii: 5% two-way swing so the shared day shows a visible peak discharge (green)
CYCLE_SWING_THRESHOLD = 0.05
HALF_PEAK_COLOR = "#e8e8e8"
PEAK_COLOR = "#c0c0c0"
BAR_WIDTH = 0.9
CONVERGENCE_FONTSIZE = 17

BASELINE_TARIFF = "0.0__cas_d1.0_e1.0_w0"
GRID_ONLY_TARIFF_PREFIX = "0.0__"  # tariff keys with solar fraction 0.0
BASELINE_MULTIPLIER = 1.5
BASELINE_SMOOTHING = 1.0
COUNTERFACTUAL_GROUP = "Counterfactual (expansion-only)"
# Case-id suffix order for bars and dots: -N, -T, -E, -B
CASE_CODE_ORDER = {"N": 0, "T": 1, "E": 2, "B": 3}

# Bands that fall as storage grows: more storage shrinks supplemental-aeration capex and the
# energy/demand charges it displaces. Fit to that direction so sampling noise can't make them rise.
ENVELOPE_DECREASING_COMPONENTS = ("capex_addtl_capacity", "energy", "demand")
# A bin whose net objective tops BOTH neighbors by more than this is an under-sampled stray: drop it
# before fitting (isotonic pools spikes). Keyed on neighbors so a rising curve keeps its points.
ENVELOPE_SPIKE_MULT = 1.25

# fig4: cool (blue/teal) = higher NPV (good), warm (yellow/orange) = lower NPV (bad). Saturated
# color plateaus close to the white center so small NPV changes (~3-5%) read as color, not white.
FIG4_NEAR_WARM = tuple(1.0 - 0.75 * (1.0 - c) for c in to_rgb(cb_palette[3]))
FIG4_NEAR_COOL = tuple(1.0 - 0.75 * (1.0 - c) for c in to_rgb(cb_palette[2]))
FIG4_CMAP = LinearSegmentedColormap.from_list(
    "fig3_diverging",
    [
        (0.00, to_rgb(cb_palette[7])),  # most negative: deep warm
        (0.25, to_rgb(cb_palette[3])),  # medium warm
        (0.40, to_rgb(cb_palette[3])),  # plateau — holds color close to zero
        (0.47, FIG4_NEAR_WARM),  # 75%-saturated warm right next to white
        (0.50, (1.0, 1.0, 1.0)),  # white at zero
        (0.53, FIG4_NEAR_COOL),  # 75%-saturated cool right next to white
        (0.60, to_rgb(cb_palette[2])),  # plateau
        (0.75, to_rgb(cb_palette[2])),  # medium cool
        (1.00, to_rgb(cb_palette[0])),  # most positive: deep cool
    ],
    N=256,
).with_extremes(bad="white")

LIQUID_TANK_TYPES = ("liquid", "liquid_from_gas", "liquid_air")

facility_labels = {
    "air__compressor": "AIR-BLOWER",
    "o2__psa": "HPOAS-PSA",
    "o2__cryo": "HPOAS-CRYO",
}

# Maps facility base_key to color (used for technology-specific plots)
facility_colors = {
    "air__compressor": cb_palette[1],
    "o2__psa": cb_palette[4],
    "o2__cryo": cb_palette[7],
}

# Maps facility base_key to marker shape (used when color encodes upgrade type)
facility_markers = {
    "air__compressor": "o",
    "o2__psa": "s",
    "o2__cryo": "^",
}

case_ids = {
    "air__compressor": {
        "compressor__gas_tank": "AB-T",
        "none__battery": "AB-B",
        "counterfactual": "AB-N",
    },
    "o2__psa": {
        "none__gas_tank": "OP-T",
        "none__battery": "OP-B",
        "elec__gas_tank": "OP-E",
        "counterfactual": "OP-N",
    },
    "o2__cryo": {
        "none__liquid_tank": "OC-T",
        "none__battery": "OC-B",
        "elec__gas_tank": "OC-E",
        "counterfactual": "OC-N",
    },
}

sweep_axis_labels = {
    "window": "Peak Charge\nDuration",
    "energy": "Energy\nCharge",
    "demand": "Demand\nCharge",
    "multiplier": "O$_2$ Requirement\nMagnitude",
    "smoothing": "O$_2$ Requirement\nDiurnal Variation",
    "storage_capex": "Storage\nCapEx",
    "addtl_capacity_capex": "Supplemental\nAeration CapEx",
}

# Baseline x-value of each fig4 sweep axis (its 0% deviation column)
SWEEP_AXIS_BASELINES = {
    "window": 0,
    "energy": 1.0,
    "demand": 1.0,
    "multiplier": BASELINE_MULTIPLIER,
    "smoothing": BASELINE_SMOOTHING,
    "storage_capex": 1.0,
    "addtl_capacity_capex": 1.0,
}

dim_dict = {
    "Hours": {"color": cb_palette[0], "marker": "o", "label": "Storage hours"},
    "Compression Ratio": {"color": cb_palette[7], "marker": "s", "label": "Compression ratio"},
    "Charge Power": {"color": cb_palette[2], "marker": "^", "label": "Charge power ratio"},
    "Addtl Capacity Frac": {
        "color": cb_palette[5],
        "marker": "D",
        "label": "Supplemental capacity fraction",
    },
}

# Facility base_key -> the tank upgrade shown in that facility's fig_2 row
stacked_lifetime_cost_line_upgrade_by_facility = {
    "air__compressor": "compressor__gas_tank",
    "o2__psa": "none__gas_tank",
    "o2__cryo": "none__liquid_tank",
}


def save_fig(fig, path, dpi=150):
    # Pasted at full page width (6.5 in: letter with 1 in margins), text should print at >= ~10 pt
    width_in = fig.get_tightbbox(fig.canvas.get_renderer()).width
    font_sizes = [
        t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
    ]
    if font_sizes and min(font_sizes) * 6.5 / width_in < 9.5:
        print(
            f"SMALL TEXT {os.path.basename(path)}: {min(font_sizes):g} pt in a {width_in:.1f} in "
            f"figure prints at {min(font_sizes) * 6.5 / width_in:.1f} pt at 6.5 in wide"
        )
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    tif_path = os.path.splitext(path)[0] + ".tif"
    fig.savefig(tif_path, bbox_inches="tight", dpi=dpi, format="tiff")
    plt.close(fig)
    Image.open(tif_path).convert("CMYK").save(tif_path, format="TIFF", dpi=(dpi, dpi))
    print(f"Saved to {path} and {tif_path}")


def setup_plot_style(ax, ylabel, fontsize, xlabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)


def add_subplot_label(ax, idx, x, label=None, y=1.0, fontsize=8):
    ax.text(
        x,
        y,
        label if label is not None else chr(65 + idx) + ".",
        transform=ax.transAxes,
        fontsize=fontsize,
        va="top",
        ha="left",
        zorder=100,
    )


def legend_rect(facecolor, label):
    """Create a plt.Rectangle for use as a legend handle."""
    return plt.Rectangle(
        (0, 0), 1, 1, facecolor=facecolor, edgecolor="black", linewidth=0.5, label=label
    )


def legend_patch(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return plt.Rectangle(
        (xdescent, ydescent),
        width,
        height,
        facecolor=orig_handle.get_facecolor(),
        edgecolor="black",
        hatch=orig_handle.get_hatch(),
        linewidth=1.0,
    )


def line_handle(label, **kwargs):
    defaults = dict(
        marker="o",
        linestyle="None",
        color="black",
        linewidth=1.5,
        markersize=12,
        markerfacecolor="none",
        markeredgecolor="none",
        markeredgewidth=1.5,
    )
    defaults.update(kwargs)
    return Line2D([0], [0], label=label, **defaults)


def cost_component_handles(h2_nonzero=True, export_nonzero=True):
    return [
        legend_rect(style["color"], label=style["label"])
        for key, style in lifetime_cost_components.items()
        if (key != "h2" or h2_nonzero) and (key != "export" or export_nonzero)
    ]


def round_up_axis_limit(value):
    """Round a positive axis extent up to a step of 5 x 10^(order of magnitude - 1)."""
    step = 5 * 10 ** (int(np.floor(np.log10(value))) - 1)
    return float(np.ceil(value / step) * step)


def net_cost_components(lifetime_cost):
    """Cost components net of the no-upgrade baseline (by_component holds the opex deltas)."""
    merged = {**lifetime_cost, **lifetime_cost["by_component"]}
    return {k: merged.get(k, 0) for k in lifetime_cost_components}


def get_design_data(config_data):
    all_results = config_data["all_results"]
    return all_results, get_best_design_key(all_results)


def iter_best_design_rows(results, tariff_keys=None):
    for config_key, config_data in results.items():
        base_wwtp_key, upgrade_key, tariff_key = parse_config_key(config_key)
        if tariff_keys and tariff_key not in tariff_keys:
            continue
        all_results, best_design_key = get_design_data(config_data)
        # If the upgrade doesn't beat the counterfactual, the cheapest design still supplies data
        data_key = cheapest_design_key(all_results)
        if data_key is None:
            continue
        yield {
            "base_wwtp_key": base_wwtp_key,
            "upgrade_key": upgrade_key,
            "tariff_key": tariff_key,
            "facility": facility_labels[get_base_key(base_wwtp_key)],
            "hours": 0.0 if best_design_key is None else float(parse_design_key(data_key)[0]),
            "design_data": all_results[data_key],
        }


def days_with_charge_and_discharge(profile_dirs_and_keys):
    """Days where every (profile_dir, design_key) optimized design shows BOTH visible charging
    (net above baseline) AND discharging (net below baseline) by more than CYCLE_SWING_THRESHOLD —
    so the row-ii panels show the storage actually cycling, not a day that is flat in one config
    (e.g. 2022-10-26 is dead for the PSA reference design but fine for air)."""
    qualifying_day_sets = []
    for profile_dir, design_key in profile_dirs_and_keys:
        qualifying = set()
        for file_name in sorted(os.listdir(profile_dir)):
            # Exact match: a prefix test lets design "7.1__44.0__0.11__0.3" pick up "..._0.37" files
            match = re.fullmatch(
                re.escape(design_key) + r"_(\d{4}-\d{2}-\d{2})_profile\.csv", file_name
            )
            if not match:
                continue
            df = pd.read_csv(os.path.join(profile_dir, file_name))
            baseline = df["Edot_t_baseline"].values
            net = df["Edot_t_net_recalc"].values
            nonzero = baseline != 0
            relative_change = (net[nonzero] - baseline[nonzero]) / baseline[nonzero]
            if (
                np.max(relative_change) > CYCLE_SWING_THRESHOLD
                and np.min(relative_change) < -CYCLE_SWING_THRESHOLD
            ):
                qualifying.add(match.group(1))
        qualifying_day_sets.append(qualifying)
    return set.intersection(*qualifying_day_sets) if qualifying_day_sets else set()


def monotone_cost_envelope(by_hour):
    """Per-hour best design: storage capex rises with storage size, its offsetting opex falls, total is a valley.

    A bigger store can't cost less to build, so storage upgrade capex is monotone non-decreasing in
    hours; the opex it buys down - supplemental-aeration capex, energy, and demand charges - falls as
    storage grows. Their sum is the cost valley whose floor marks the optimum. The per-hour min-total
    design is a noisy sample of each band (it trades tank size against charge-gear size at nearly equal
    total), so fit each band to its physical direction and let the total follow. The marked optimum
    keeps its real component split.
    """
    hours = sorted(by_hour)
    best = {h: dict(min(by_hour[h], key=lambda d: d["obj_total"])) for h in hours}
    while True:
        for i in range(1, len(hours) - 1):
            neighbor_max = max(best[hours[i - 1]]["obj_total"], best[hours[i + 1]]["obj_total"])
            if best[hours[i]]["obj_total"] > ENVELOPE_SPIKE_MULT * neighbor_max:
                hours.pop(i)
                break
        else:
            break

    selected = {h: best[h] for h in hours}
    x = np.array(hours, dtype=float)

    # Snapshot the optimum's real component split before fitting overwrites it: the vertical line marks
    # this exact design, so its stack must show what actually gets built (e.g. zero supplemental).
    opt = min(hours, key=lambda h: sum(selected[h][k] for k in lifetime_cost_components))
    raw_opt = {c: float(selected[opt][c]) for c in lifetime_cost_components}

    # Supplemental capex only falls past the optimum, so cap it there at the optimum's real value,
    # before the fit: else sparse high-hour bins pool into pre-optimum hours and inflate the band.
    for h in hours:
        if h >= opt:
            selected[h]["capex_addtl_capacity"] = min(
                selected[h]["capex_addtl_capacity"], raw_opt["capex_addtl_capacity"]
            )

    # Fit each band to its physical direction so per-hour sampling noise (gear vs tank sizing) can't
    # move it the wrong way: opex-side bands non-increasing, storage capex non-decreasing.
    for comp in (*ENVELOPE_DECREASING_COMPONENTS, "storage_capex"):
        y = np.array([selected[h][comp] for h in hours])
        fit = IsotonicRegression(increasing=comp == "storage_capex").fit_transform(x, y)
        for h, value in zip(hours, fit):
            selected[h][comp] = float(value)

    # Restore the marked design's real split, then re-enforce storage monotonicity with a running max
    # so the restore can't reintroduce a dip at the optimum.
    for c in lifetime_cost_components:
        selected[opt][c] = raw_opt[c]
    running_max = 0.0
    for h in hours:
        running_max = max(running_max, selected[h]["storage_capex"])
        selected[h]["storage_capex"] = running_max
    return selected


def build_fig3_bar_data(results, relative):
    if relative:
        # First row per (facility, tariff) sets the expansion-only capex its upgrades net against
        expansion_only_counterfactual = {}
        for row in iter_best_design_rows(results):
            expansion_only_counterfactual.setdefault(
                (row["base_wwtp_key"], row["tariff_key"]),
                float(row["design_data"]["lifetime_cost"]["expansion_only"]),
            )

    rows = []
    counterfactual_by_facility = {}
    for row in iter_best_design_rows(results):
        lifetime_cost = row["design_data"]["lifetime_cost"]

        if row["hours"] == 0:  # upgrade doesn't beat counterfactual
            components = (
                {k: 0.0 for k in lifetime_cost_components}
                if relative
                else counterfactual_cost_components(lifetime_cost)
            )
        elif relative:
            components = net_cost_components(lifetime_cost)
            components["capex_addtl_capacity"] = (
                lifetime_cost["capex_addtl_capacity"]
                - expansion_only_counterfactual[(row["base_wwtp_key"], row["tariff_key"])]
            )
            # Negate all components so positive NPV means improvement (lower costs)
            components = {k: -v for k, v in components.items()}
        else:
            components = flat_cost_components(lifetime_cost, "new")

        rows.append(
            {
                "Facility": row["facility"],
                "upgrade_key": row["upgrade_key"],
                "group": f"{row['upgrade_key']} - With-flexibility",
                "lifetime_cost": sum(components.values()),
                **{f"{k}_lifetime_cost": v for k, v in components.items()},
            }
        )

        if not relative and row["facility"] not in counterfactual_by_facility:
            counterfactual = counterfactual_cost_components(lifetime_cost)
            counterfactual_by_facility[row["facility"]] = {
                "Facility": row["facility"],
                "group": COUNTERFACTUAL_GROUP,
                "lifetime_cost": sum(counterfactual.values()),
                **{f"{k}_lifetime_cost": v for k, v in counterfactual.items()},
            }

    rows.extend(counterfactual_by_facility.values())
    return pd.DataFrame(rows)


def build_fig2_cost_curves(results):
    curves = {facility: {"data": []} for facility in facility_labels}
    for config_key, config_data in results.items():
        base_wwtp_key, upgrade_key, _ = parse_config_key(config_key)
        facility = get_base_key(base_wwtp_key)
        if (
            parse_base_wwtp_key(base_wwtp_key).multiplier != BASELINE_MULTIPLIER
            or get_base_key(upgrade_key) != stacked_lifetime_cost_line_upgrade_by_facility[facility]
        ):
            continue
        curve = curves[facility]
        for design_key, design_data in config_data["all_results"].items():
            lifetime_cost = design_data["lifetime_cost"]
            if not is_valid_lifetime_cost(lifetime_cost["total"]):
                continue
            components = flat_cost_components(lifetime_cost, "new")
            components["total"] = sum(components.values())
            # obj_total (net cost vs counterfactual) is sharply valley-shaped, unlike the opex-heavy
            # absolute "total", so it picks each hour's best design and flags stray bins.
            curve["data"].append(
                {
                    "hours": float(parse_design_key(design_key)[0]),
                    "tank_metrics": design_data["tank_metrics"],
                    "medium_costs": design_data.get("medium_costs", {}),
                    "obj_total": lifetime_cost["total"],
                    **components,
                }
            )
            if "counterfactual" not in curve:
                counterfactual = counterfactual_cost_components(lifetime_cost)
                counterfactual["total"] = sum(counterfactual.values())
                curve["counterfactual"] = {"hours": 0.0, **counterfactual}

    fig2_curves = []
    for curve in curves.values():
        data = curve["data"]
        best_design = min(data, key=lambda d: d["total"])
        if best_design["total"] >= curve["counterfactual"]["total"]:
            best_actual_hours = 0.0
        else:
            best_actual_hours = round(best_design["hours"], 1)

        # Gas->liquid handoff hour (None: all-gas AB-T / native-liquid OC-T) from raw per-design
        # winners, like the solver's refinement; BY COST per hour so a sparse outlier can't flip it.
        best_gas, best_liquid = {}, {}
        for d in data:
            hour_bin = round(d["hours"])
            gas_cost, liquid_cost = d["medium_costs"].get("gas"), d["medium_costs"].get("liq")
            best_gas[hour_bin] = min(
                best_gas.get(hour_bin, np.inf), np.inf if gas_cost is None else gas_cost
            )
            best_liquid[hour_bin] = min(
                best_liquid.get(hour_bin, np.inf), np.inf if liquid_cost is None else liquid_cost
            )
        liquid_crossover_hours = None
        # Liquid never cheaper anywhere -> all-gas (rejects sparse expensive liquid bins at
        # un-sampled-gas hours)
        if min(best_liquid.values()) < min(best_gas.values()):
            # Liquid dominant: first hour liquid beats gas; None if liquid from the start
            first_liquid_win = next(h for h in sorted(best_gas) if best_liquid[h] < best_gas[h])
            if any(best_gas[h] < np.inf for h in best_gas if h < first_liquid_win):
                liquid_crossover_hours = first_liquid_win

        best_is_liquid = best_design["tank_metrics"].get("tank_type") in LIQUID_TANK_TYPES
        # Gas below the crossover, liquid at/above: the SAME medium the panel note claims. Per-hour
        # cheapest let a degenerate tiny-liquid design (huge storage capex) spike the gas region.
        gas_x_hi = (
            liquid_crossover_hours
            if liquid_crossover_hours is not None
            else (0 if best_is_liquid else float("inf"))
        )
        by_hour = defaultdict(list)
        for d in data:
            hour_bin = round(d["hours"])
            is_liquid = d["tank_metrics"].get("tank_type") in LIQUID_TANK_TYPES
            if is_liquid == (hour_bin >= gas_x_hi):
                by_hour[hour_bin].append(d)
        envelope = monotone_cost_envelope(by_hour)

        # Plot each point at its design's ACTUAL hours, not the rounded bin, so a sub-1h design
        # can't land on x=0 and hide the counterfactual. The optimum sits on the fitted envelope.
        points = [curve["counterfactual"]] + [envelope[h] for h in sorted(envelope)]
        fig2_curves.append(
            {
                "hours_arr": np.array([p["hours"] for p in points]),
                "component_arrays": {
                    k: np.array([p[k] for p in points]) / 1e6 for k in lifetime_cost_components
                },
                "best_actual_hours": best_actual_hours,
                "best_is_liquid": best_is_liquid,
                "liquid_crossover_hours": liquid_crossover_hours,
            }
        )
    return fig2_curves


def plot_fig2_cost_curve(ax, curve):
    hours_arr = curve["hours_arr"]

    # Separate positive and negative components
    bottom_pos = np.zeros_like(hours_arr)
    bottom_neg = np.zeros_like(hours_arr)
    has_negative = False
    for key, style in lifetime_cost_components.items():
        values = curve["component_arrays"][key]
        if not np.any(values):
            continue
        is_negative = np.any(values < 0)
        has_negative = has_negative or is_negative
        bottom = bottom_neg if is_negative else bottom_pos
        ax.fill_between(
            hours_arr,
            bottom,
            bottom + values,
            color=style["color"],
            label=style["label"],
            edgecolor="black",
            linewidth=0.5,
        )
        if is_negative:
            bottom_neg = bottom + values
        else:
            bottom_pos = bottom + values

    # Plot total line if there are negative components
    if has_negative:
        ax.plot(
            hours_arr,
            bottom_pos + bottom_neg,
            color="black",
            linewidth=3.0,
            zorder=10,
            label="Total",
        )

    opt_hours = curve["best_actual_hours"]
    ax.axvline(opt_hours, color="black", linewidth=1.5, zorder=11)
    # Flip the label to the left of the line once the line is past mid-axis, else it runs off the edge.
    past_midaxis = opt_hours > 24 * 0.5
    ax.text(
        opt_hours - 0.3 if past_midaxis else opt_hours + 0.3,
        0.97,
        f"{opt_hours:.1f}h",
        transform=blended_transform_factory(ax.transData, ax.transAxes),
        va="top",
        ha="right" if past_midaxis else "left",
        fontsize=10.5,
    )

    # Which medium the plotted optimum uses, stated above the panel
    # Right-aligned so it clears the panel's roman-numeral label.
    crossover = curve["liquid_crossover_hours"]
    if crossover is None:
        medium_note = "Liquid-phase storage" if curve["best_is_liquid"] else "Gas-phase storage"
    else:
        medium_note = f"Liquid-phase above {crossover:.0f}h"
    ax.text(1.0, 1.02, medium_note, transform=ax.transAxes, va="bottom", ha="right", fontsize=10.5)
    # Only draw the handoff where it is visible as its own boundary
    if crossover is not None and 2 < crossover < 22:
        ax.axvline(crossover, color="0.45", linestyle=":", linewidth=1.0, zorder=9)

    ax.axhline(0, color="black", linewidth=0.8)

    # Adjust y-axis limits to accommodate negative values
    all_values = np.concatenate([bottom_pos, bottom_neg])
    y_min = np.min(all_values)
    y_max = np.max(all_values)
    if y_min < 0:
        y_floor = int(np.floor(y_min / 5) * 5)
        y_ceil = int(np.ceil(y_max / 10) * 10)
    else:
        y_floor = 0
        y_ceil = min(int(np.ceil(y_max / 10) * 10), 60)  # 60 for air

    ax.set_ylim(y_floor, y_ceil)
    ax.set_yticks(np.arange(y_floor, y_ceil + 1, 5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlabel("Hours of O₂ Storage", fontsize=11)
    ax.set_ylabel("Lifetime Cost ($M)", fontsize=11)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])


def bar_group_sort_key(group):
    """Sort counterfactual first, then by case-code suffix order: -N, -T, -E, -B"""
    if group == COUNTERFACTUAL_GROUP:
        return (0, 0, group)
    upgrade_base = get_base_key(group.split(" - With-flexibility")[0])
    for upgrades in case_ids.values():
        if upgrade_base in upgrades:
            return (1, CASE_CODE_ORDER[upgrades[upgrade_base][-1]], group)
    return (1, 999, group)  # Unknown upgrade at the end


def plot_lifetime_cost_bars(lifetime_cost_data, ax, group_by, total_or_change="total"):
    x_categories = [
        cat for cat in facility_labels.values() if cat in lifetime_cost_data["Facility"].unique()
    ]
    x_positions = np.arange(len(x_categories))
    width = min(BAR_WIDTH, BAR_WIDTH * 8 / len(x_categories))
    group_types = sorted(lifetime_cost_data[group_by].unique(), key=bar_group_sort_key)
    available_groups_by_category = {
        x_cat: [
            g
            for g in group_types
            if (
                (lifetime_cost_data[group_by] == g) & (lifetime_cost_data["Facility"] == x_cat)
            ).any()
        ]
        for x_cat in x_categories
    }

    ax.axhline(0, color="black", lw=0.5)
    highest_point, lowest_point, h2_nonzero = 0, 0, False
    group_handles = []
    bar_labels = {}
    for group in group_types:
        group_data = lifetime_cost_data[lifetime_cost_data[group_by] == group]
        totals = []
        for j, x_cat in enumerate(x_categories):
            x_cat_data = group_data[group_data["Facility"] == x_cat]
            if x_cat_data.empty:
                continue
            present_groups = available_groups_by_category[x_cat]
            bar_width = width / len(present_groups)
            x = x_positions[j] - width / 2 + present_groups.index(group) * bar_width + bar_width / 2
            pos_bottom, neg_bottom = 0, 0
            for key, style in lifetime_cost_components.items():
                value = x_cat_data[f"{key}_lifetime_cost"].iloc[0] / 1e6
                if key == "h2" and value != 0.0:
                    h2_nonzero = True
                ax.bar(
                    x,
                    value,
                    bar_width,
                    color=style["color"],
                    bottom=neg_bottom if value < 0 else pos_bottom,
                    edgecolor="black",
                    linewidth=0.5,
                )
                if value < 0:
                    neg_bottom += value
                else:
                    pos_bottom += value
            # Headroom above the stack for the bar label
            highest_point = max(highest_point, pos_bottom + 0.02)
            lowest_point = min(lowest_point, neg_bottom)
            totals.append((x, x_cat_data["lifetime_cost"].iloc[0] / 1e6))

            facility = next(k for k, v in facility_labels.items() if v == x_cat)
            upgrade_key = "counterfactual" if group == COUNTERFACTUAL_GROUP else get_base_key(group)
            bar_labels[x] = case_ids[facility].get(upgrade_key, upgrade_key)

        if total_or_change == "change":
            for x, total in totals:
                if total != 0:
                    ax.plot(x, total, "o", color="black", markersize=4, zorder=3)
        group_handles.append(legend_rect("white", label=group))

    ylabel = "Lifetime Cost ($M)" if total_or_change == "total" else "Net Present Value ($M)"
    setup_plot_style(ax, ylabel, 7)
    y_bottom = -round_up_axis_limit(-lowest_point) if lowest_point < 0 else 0
    ax.set_ylim(y_bottom, round_up_axis_limit(highest_point))
    y_span = max(highest_point, -lowest_point)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:.1f}" if y_span < 3 else f"{x:g}")
    )
    ax.tick_params(axis="y", labelsize=6)
    sorted_labels = sorted(bar_labels.items())
    ax.set_xticks([x for x, _ in sorted_labels])
    ax.set_xticklabels([label for _, label in sorted_labels], rotation=0, fontsize=6)

    cost_handles = cost_component_handles(h2_nonzero=h2_nonzero)
    if total_or_change == "change":
        cost_handles.append(
            line_handle(
                label="Total Lifetime Cost ($M)",
                color="w",
                markerfacecolor="black",
                markeredgecolor="black",
            )
        )
    ax.legend(
        handles=group_handles + cost_handles,
        bbox_to_anchor=(0.5, 1.08),
        loc="upper center",
        ncol=max(len(group_handles), 3),
        frameon=False,
        fontsize=24,
        bbox_transform=plt.gcf().transFigure,
        handler_map={plt.Rectangle: HandlerPatch(patch_func=legend_patch)},
    )


def plot_convergence(ax, config_data, upgrade_key, design_space_ranges):
    upgrade_info = get_upgrade_info(upgrade_key)
    if upgrade_info["is_gas_tank"]:
        dims = ["Hours", "Compression Ratio", "Charge Power", "Addtl Capacity Frac"]
    elif upgrade_info["is_liquid_tank"]:
        dims = ["Hours", "Charge Power", "Addtl Capacity Frac"]
    else:
        dims = ["Hours", "Charge Power"]
    range_map = {
        "Hours": design_space_ranges["o2_range"],
        "Compression Ratio": design_space_ranges["comp_ratio_range"],
        "Charge Power": design_space_ranges["charge_power_ratio_range"],
        "Addtl Capacity Frac": [0, 1],
    }

    history = config_data["mu_sigma_history"][1:]
    iterations = list(range(len(history)))
    mus = [np.ravel(np.array(m["mu"], dtype=float)) for m in history]
    stds = [np.sqrt(np.diag(np.array(m["sigma"], dtype=float))) for m in history]
    active_dims = [(label, info) for label, info in dim_dict.items() if label in dims]
    # All dimensions share one axis: mean and std as a fraction of each dimension's search range
    for dim_idx, (label, info) in enumerate(active_dims):
        lo, hi = range_map[label]
        ax.errorbar(
            iterations,
            [(float(mu[dim_idx]) - lo) / (hi - lo) for mu in mus],
            yerr=[float(std[dim_idx]) / (hi - lo) for std in stds],
            fmt=info["marker"],
            color=info["color"],
            markersize=8,
            elinewidth=1.5,
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Iteration", fontsize=CONVERGENCE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=CONVERGENCE_FONTSIZE - 1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def classify_tariff_sweep(tariff_key):
    """Classify a tariff key into sweep axes and extract x-values.

    Returns list of (axis_name, x_value) tuples. Baseline belongs to multiple axes.
    """
    # Charge-factor sweeps scale one charge ("alldemand0.8"); the rest set d/e/w ("d1.0_e1.0_wneg1")
    all_match = re.search(r"all(demand|energy)([0-9.]+)", tariff_key)
    if all_match:
        value = float(all_match.group(2))
        demand, energy = (value, 1.0) if all_match.group(1) == "demand" else (1.0, value)
        window = 0
    else:
        demand, energy, window_neg, window = re.search(
            r"d([0-9.]+)_e([0-9.]+)_w(neg)?([0-9.]+)", tariff_key
        ).groups()
        demand, energy = float(demand), float(energy)
        window = -float(window) if window_neg else float(window)
    axes = []
    if demand == 1.0 and energy == 1.0:
        axes.append(("window", window))
    if demand == 1.0 and window == 0:
        axes.append(("energy", energy))
    if energy == 1.0 and window == 0:
        axes.append(("demand", demand))
    return axes


def build_fig4_sweeps(sensitivity_config, results_by_config):
    scale_mgd = sensitivity_config["scale_mgd"]
    capex_sensitivity = sensitivity_config["capex_sensitivity"] or []

    summer_sweep_tariff = BASELINE_TARIFF
    for config_key in results_by_config:
        result_base_wwtp, _, result_tariff = parse_config_key(config_key)
        if parse_base_wwtp_key(result_base_wwtp).multiplier != BASELINE_MULTIPLIER:
            summer_sweep_tariff = result_tariff
            break

    facility_sweep = {}
    facility_counterfactual = {}
    for wwtp_type, upgrade_key_list in sensitivity_config["upgrade_keys"].items():
        base_wwtp_key = f"{wwtp_type}__{scale_mgd}__{BASELINE_MULTIPLIER}__{BASELINE_SMOOTHING}"
        upgrade_key = upgrade_key_list[0]
        tech_prefix = "__".join(parse_base_wwtp_key(base_wwtp_key)[:3])
        sweep_data = {axis: {} for axis in sweep_axis_labels}
        counterfactual_data = {axis: {} for axis in sweep_axis_labels}

        for config_key, value in results_by_config.items():
            result_base_wwtp, result_upgrade, result_tariff = parse_config_key(config_key)
            parsed = parse_base_wwtp_key(result_base_wwtp)
            if result_upgrade != upgrade_key or "__".join(parsed[:3]) != tech_prefix:
                continue

            lifetime_cost = value["lifetime_cost"]
            lifetime_cost_value = (
                lifetime_cost["total"] if is_valid_lifetime_cost(lifetime_cost["total"]) else np.nan
            )
            counterfactual_value = float(lifetime_cost.get("expansion_only", 0))
            sweep_points = []
            if parsed.multiplier == BASELINE_MULTIPLIER and parsed.smoothing == BASELINE_SMOOTHING:
                sweep_points += classify_tariff_sweep(result_tariff)
            if result_tariff == summer_sweep_tariff:
                # Baseline (1.5, 1.0) belongs to both summer axes
                if parsed.smoothing == BASELINE_SMOOTHING:
                    sweep_points.append(("multiplier", parsed.multiplier))
                if parsed.multiplier == BASELINE_MULTIPLIER:
                    sweep_points.append(("smoothing", parsed.smoothing))
            for axis, x_val in sweep_points:
                sweep_data[axis][x_val] = lifetime_cost_value
                counterfactual_data[axis][x_val] = counterfactual_value

        baseline_value = results_by_config.get(
            get_config_name(base_wwtp_key, upgrade_key, BASELINE_TARIFF)
        )
        if baseline_value and is_valid_lifetime_cost(baseline_value["lifetime_cost"]["total"]):
            # Air keeps its panel even though storage doesn't pay for it (baseline NPV < 0): the
            # non-o2 figure shows all three facilities. The o2-only figure drops air by prefix.
            baseline_cost = baseline_value["lifetime_cost"]
            baseline_counterfactual = float(baseline_cost["expansion_only"])
            for mult in capex_sensitivity:
                for axis, capex_key in [
                    ("storage_capex", "storage_capex"),
                    ("addtl_capacity_capex", "capex_addtl_capacity"),
                ]:
                    sweep_data[axis][mult] = baseline_cost["total"] + baseline_cost[capex_key] * (
                        mult - 1
                    )
                    counterfactual_data[axis][mult] = baseline_counterfactual

        facility_sweep[wwtp_type] = sweep_data
        facility_counterfactual[wwtp_type] = counterfactual_data

    active_axes = [
        axis
        for axis in sweep_axis_labels
        if any(sweeps[axis] for sweeps in facility_sweep.values())
    ]
    axis_x_vals = {
        axis: sorted(set().union(*(sweeps[axis] for sweeps in facility_sweep.values())))
        for axis in active_axes
    }

    max_dev_other = max(
        (
            abs(100 * (v - SWEEP_AXIS_BASELINES[a]) / SWEEP_AXIS_BASELINES[a])
            for a in active_axes
            if SWEEP_AXIS_BASELINES[a] != 0
            for v in axis_x_vals[a]
        ),
        default=30,
    )
    axis_devs = {}
    for axis in active_axes:
        base = SWEEP_AXIS_BASELINES[axis]
        x_vals = axis_x_vals[axis]
        if base == 0:
            # Zero-baseline axes (window shift) are scaled onto the other axes' deviation range
            max_abs = max(abs(v) for v in x_vals)
            devs = [round(x / max_abs * max_dev_other, 1) if max_abs > 0 else 0 for x in x_vals]
        else:
            devs = [round(100 * (x - base) / abs(base), 1) for x in x_vals]
        axis_devs[axis] = list(zip(devs, x_vals))
    all_dev_pcts = sorted(set(d for devs in axis_devs.values() for d, _ in devs))
    dev_to_col = {d: i for i, d in enumerate(all_dev_pcts)}

    return facility_sweep, facility_counterfactual, active_axes, axis_devs, all_dev_pcts, dev_to_col


def build_fig4_matrix(sweep_data, active_axes, axis_devs, dev_to_col):
    matrix = np.full((len(active_axes), len(dev_to_col)), np.nan)
    infeasible = np.zeros(matrix.shape, dtype=bool)
    for row_idx, axis in enumerate(active_axes):
        for dev_pct, x_val in axis_devs[axis]:
            col = dev_to_col[dev_pct]
            value = sweep_data[axis].get(x_val)
            if value is None:
                continue
            elif np.isnan(value):
                infeasible[row_idx, col] = True
            else:
                matrix[row_idx, col] = value
    return matrix, infeasible


def pct_change_matrix(matrix, reference):
    reference = np.broadcast_to(reference, matrix.shape)
    result = np.full_like(matrix, np.nan)
    valid = np.isfinite(matrix) & np.isfinite(reference) & (np.abs(reference) > 1e-12)
    result[valid] = 100.0 * (matrix[valid] - reference[valid]) / np.abs(reference[valid])
    return result


def si_figure_5(results, output_dir):
    # Capacity factor is ~1.0 for every config, so plot power (peak shaving) vs energy (load
    # shifting) capacity. Color = case-id type (tanks all -T), marker = O2 tech; together = case id.
    type_colors = {
        "compressor__gas_tank": ("-T", cb_palette[0]),
        "none__gas_tank": ("-T", cb_palette[0]),
        "none__liquid_tank": ("-T", cb_palette[0]),
        "elec__gas_tank": ("-E", cb_palette[3]),
        "none__battery": ("-B", cb_palette[4]),
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    setup_plot_style(
        ax,
        "Normalized Energy Capacity\n(load shifting)",
        24,
        xlabel="Normalized Power Capacity\n(peak shaving)",
    )
    ax.set_xlim(0.0, 0.08)
    ax.set_ylim(0.0, 0.08)
    ax.set_xticks(np.arange(0.0, 0.10, 0.02))
    ax.set_yticks(np.arange(0.0, 0.10, 0.02))

    present_facilities, present_upgrades = set(), set()
    for config_key, config_data in results.items():
        base_wwtp_key, upgrade_key, tariff_key = parse_config_key(config_key)
        if not tariff_key.startswith(GRID_ONLY_TARIFF_PREFIX):
            continue
        all_results = config_data["all_results"]
        best_design_key = cheapest_design_key(all_results)
        if best_design_key is None:
            continue
        design_data = all_results[best_design_key]
        facility = get_base_key(base_wwtp_key)
        upgrade = get_base_key(upgrade_key)
        ax.scatter(
            design_data["mean_power_capacity"],
            design_data["mean_energy_capacity"],
            marker=facility_markers[facility],
            color=type_colors[upgrade][1],
            s=250,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        present_facilities.add(facility)
        present_upgrades.add(upgrade)

    facility_handles = [
        line_handle(
            label=facility_labels[facility],
            marker=marker,
            color="black",
            markerfacecolor="none",
            markeredgecolor="black",
        )
        for facility, marker in facility_markers.items()
        if facility in present_facilities
    ]
    spacer = line_handle(label="", marker="None", linestyle="None", color="none", alpha=0)
    upgrade_handles = {}
    for upgrade, (label, color) in type_colors.items():
        if upgrade in present_upgrades and label not in upgrade_handles:
            upgrade_handles[label] = legend_rect(color, label=label)
    ax.legend(
        handles=facility_handles + [spacer] + list(upgrade_handles.values()),
        loc="upper left",
        fontsize=15,
        frameon=True,
        ncol=1,
    )

    plt.tight_layout()
    save_fig(fig, os.path.join(output_dir, "si_figure_5.png"))


def si_sweep_scatter(results, output_dir, sweep_from, xlabel, filename, tariff_key=None):
    """Best design's NPV vs a swept value, with the no-storage counterfactual at zero.
    sweep_from: 'upgrade': float(split_key(upgrade_key)[2])
    'tariff' : float(this_tariff_key.split('__')[0])"""
    points_by_facility = defaultdict(list)
    for config_key, config_data in results.items():
        base_wwtp_key, upgrade_key, this_tariff_key = parse_config_key(config_key)
        if tariff_key and this_tariff_key != tariff_key:
            continue
        all_results = config_data["all_results"]
        best_design_key = cheapest_design_key(all_results)
        if best_design_key is None:
            continue
        if sweep_from == "upgrade":
            sweep_value = float(split_key(upgrade_key)[2])
        else:
            sweep_value = float(this_tariff_key.split("__")[0])
        lifetime_cost = all_results[best_design_key]["lifetime_cost"]
        points_by_facility[get_base_key(base_wwtp_key)].append(
            (sweep_value, (lifetime_cost["expansion_only"] - lifetime_cost["total"]) / 1e6)
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    # Deterministic color/marker/label per facility (shared with the other technology figures)
    for facility, facility_label in facility_labels.items():
        if facility not in points_by_facility:
            continue
        sweep_values, lifetime_costs = zip(*points_by_facility[facility])
        ax.scatter(
            sweep_values,
            lifetime_costs,
            s=100,
            marker=facility_markers[facility],
            color=facility_colors[facility],
            label=facility_label,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="No storage")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Best Net Present Value (US$M)", fontsize=16)
    ax.legend(fontsize=12, frameon=True)
    plt.tight_layout()
    save_fig(fig, os.path.join(output_dir, filename))


# Chart VS-series bulk LOX tanks: (gross water volume m³, outside diameter m).
# Chart Bulk Storage Catalog (2026), https://files.chartindustries.com/13608592_BulkCatalog.pdf
CHART_VS_TANKS = ((6.21, 1.676), (11.92, 2.184), (22.75, 2.184), (35.43, 2.896))


def plot_storage_vs_bioreactor_volume(results, output_dir, config_values):
    """SI footprint figure: one panel per HPO storage upgrade at its optimal design, with each tank
    footprint drawn to scale inside the aeration basins' footprint (area proportional to footprint).

    Args:
        config_values: needs the facility's aeration_basin_volume_mg (total, MG) and
            aeration_basin_depth_ft (side water depth, ft)
    """
    basin_volume = config_values["aeration_basin_volume_mg"] * MGD_TO_M3_PER_DAY  # m³
    basin_area = basin_volume / (config_values["aeration_basin_depth_ft"] * 0.3048)  # m²
    # Largest single ANL Type IV H2 vessel (by size; the high-DGE rows are multi-tank systems),
    # laid flat in one layer. Internal volume is its geometry less the wall.
    h2_specs = HYDROGEN_TANK_SPECS
    h2_tank = h2_specs.loc[
        (h2_specs["Length (inches)"] * h2_specs["Diameter (inches)"] ** 2).idxmax()
    ]
    h2_length = h2_tank["Length (inches)"] * 0.0254
    h2_diameter = h2_tank["Diameter (inches)"] * 0.0254
    r_in = h2_diameter / 2 - (h2_tank["Liner THK (mm)"] + h2_tank["ANL Comp THK (mm)"]) / 1000
    h2_tank_volume = np.pi * r_in**2 * (h2_length - 2 * r_in) + 4 / 3 * np.pi * r_in**3

    tech_label = {"psa": "HPO-PSA", "cryo": "HPO-CRYO"}
    panels = []  # (sort key, label, [(footprint m², volume m³, color, text color)])
    for config_key, config_data in results.items():
        base_wwtp_key, upgrade_key, tariff_key = parse_config_key(config_key)
        technology = split_key(base_wwtp_key)[1]
        if technology not in tech_label or not tariff_key.startswith(GRID_ONLY_TARIFF_PREFIX):
            continue
        if "gas_tank" not in upgrade_key and "liquid_tank" not in upgrade_key:
            continue
        design_key = cheapest_design_key(config_data["all_results"])
        if design_key is None:
            continue
        p = config_data["all_results"][design_key]["new_param_vals"]
        if p["is_liquid_storage"]:
            # Tank volume (at least the smallest bulk LOX tank), split into equal Chart VS tanks
            v_o2 = max(moles_to_mass(p["N_max"], M_O2) / rho_hpo_liquid, min_liquid_tank_volume_m3)
            n_o2 = int(np.ceil(v_o2 / max_liquid_tank_volume_m3))
            o2_diameter = next(d for v, d in CHART_VS_TANKS if v >= v_o2 / n_o2)
            o2_area = n_o2 * np.pi * o2_diameter**2 / 4
        else:
            v_o2 = p["V_tank"]
            _, n_o2, _, length, aspect = get_tank_cost_smoothed(v_o2, p["P_max"] - P_ATM)
            o2_area = n_o2 * length**2 / aspect
        tanks = [(o2_area, v_o2, "#1f77b4", "#1f77b4")]
        if p["y_h2"]:
            v_h2 = p["N_max"] * p["y_h2"] * R * T_room / (P_MAX_H2 * Pa_per_MPa)
            h2_area = np.ceil(v_h2 / h2_tank_volume) * h2_length * h2_diameter
            tanks.append((h2_area, v_h2, "#ff7f0e", "#d95f0e"))
        store = "electrolyzer" if "elec" in upgrade_key else "excess O₂"
        sort_key = (list(tech_label).index(technology), store == "electrolyzer")
        panels.append((sort_key, f"{tech_label[technology]}\n({store} storage)", tanks))
    panels.sort(key=lambda panel: panel[0])

    def fmt(x):
        return f"{x:,.1f}" if x < 10 else f"{x:,.0f}"

    # Basin square has side 1, so a footprint A is drawn as a square of side sqrt(A / basin_area)
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3, 2.75 * nrows), squeeze=False)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    for ax, (_, label, tanks) in zip(axes.flat, panels):
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#e0e0e0", zorder=0))
        ax.text(
            0.5,
            0.6,
            f"Aeration basins\n{basin_area:,.0f} m² ({basin_volume:,.0f} m³)",
            ha="center",
            va="center",
            fontsize=9,
        )
        # O2 tanks in the bottom-left corner, H2 tanks in the bottom-right (label on a second line)
        for (area, volume, color, text_color), at_right in zip(tanks, (False, True)):
            side = max(np.sqrt(area / basin_area), 0.015)
            x = 1 - side - 0.03 if at_right else 0.03
            ax.add_patch(plt.Rectangle((x, 0.03), side, side, facecolor=color, zorder=2))
            ax.text(
                1 if at_right else 0,
                -0.11 if at_right else -0.02,
                f"{fmt(area)} m² ({fmt(volume)} m³)",
                ha="right" if at_right else "left",
                va="top",
                fontsize=9,
                color=text_color,
            )
        ax.text(0.5, 1.02, label, ha="center", va="bottom", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.22, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.08, hspace=0.08)
    save_fig(fig, os.path.join(output_dir, "storage_vs_bioreactor_area.png"))


def si_convergence_combined_figure(results, output_dir, design_space_ranges):
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.5), sharey=True)
    for row_idx, (facility, facility_case_ids) in enumerate(case_ids.items()):
        upgrades = [u for u in facility_case_ids if u != "counterfactual"]
        for col_idx, ax in enumerate(axes[row_idx]):
            if col_idx >= len(upgrades):
                ax.axis("off")
                continue
            upgrade_base = upgrades[col_idx]
            for config_key, config_data in results.items():
                base_wwtp_key, upgrade_key, _ = parse_config_key(config_key)
                if (
                    get_base_key(base_wwtp_key) == facility
                    and get_base_key(upgrade_key) == upgrade_base
                ):
                    plot_convergence(ax, config_data, upgrade_key, design_space_ranges)
                    ax.set_title(facility_case_ids[upgrade_base], fontsize=CONVERGENCE_FONTSIZE)
                    break

    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of\nsearch range", fontsize=CONVERGENCE_FONTSIZE)
    fig.tight_layout()
    handles = [
        Line2D(
            [],
            [],
            color=info["color"],
            marker=info["marker"],
            linestyle="None",
            label=info["label"],
        )
        for info in dim_dict.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=CONVERGENCE_FONTSIZE,
        frameon=False,
    )
    save_fig(fig, os.path.join(output_dir, "convergence_combined.png"))


def si_solar_figure(results, output_dir, fontsize=8):
    """SI solar figure: NPV contributions vs solar fraction, one panel per upgrade, styled like
    Fig 3 (savings positive, costs negative, black dot = NPV). Upgrades that don't beat the
    counterfactual show as zero, as in Fig 3, and upgrades that never beat it get no panel.
    Storage CapEx includes the solar array.
    """
    by_upgrade = defaultdict(dict)
    for config_key, config_data in results.items():
        base_wwtp_key, upgrade_key, tariff_key = parse_config_key(config_key)
        all_results, best_design_key = get_design_data(config_data)
        components = {k: 0.0 for k in lifetime_cost_components}
        if best_design_key is not None:
            lifetime_cost = all_results[best_design_key]["lifetime_cost"]
            components = net_cost_components(lifetime_cost)
            components["capex_addtl_capacity"] = (
                lifetime_cost["capex_addtl_capacity"] - lifetime_cost["expansion_only"]
            )
            components = {k: -v / 1e6 for k, v in components.items()}
        code = case_ids[get_base_key(base_wwtp_key)][get_base_key(upgrade_key)]
        by_upgrade[code][100 * float(tariff_key.split("__", 1)[0])] = components

    codes = [
        c
        for c in ("AB-B", "OP-B", "OC-B", "OP-E", "OC-E")
        if any(v != 0 for comps in by_upgrade.get(c, {}).values() for v in comps.values())
    ]
    stack_extents = [
        (sum(min(v, 0) for v in comps.values()), sum(max(v, 0) for v in comps.values()))
        for code in codes
        for comps in by_upgrade[code].values()
    ]
    y_low = min(lo for lo, _ in stack_extents)
    y_high = max(hi for _, hi in stack_extents)
    keys_shown = [
        k
        for k in lifetime_cost_components
        if any(comps[k] != 0 for code in codes for comps in by_upgrade[code].values())
    ]

    # Panels plus one legend cell, at most 4 across
    ncols = min(len(codes) + 1, 4)
    nrows = int(np.ceil((len(codes) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.3, 2.0 * nrows), sharey=True, squeeze=False)
    for ax, code in zip(axes.flat, codes):
        fractions = sorted(by_upgrade[code])
        for x, fraction in enumerate(fractions):
            components = by_upgrade[code][fraction]
            pos_bottom, neg_bottom = 0.0, 0.0
            for key in keys_shown:
                value = components[key]
                ax.bar(
                    x,
                    value,
                    0.6,
                    bottom=neg_bottom if value < 0 else pos_bottom,
                    color=lifetime_cost_components[key]["color"],
                    edgecolor="black",
                    linewidth=0.5,
                )
                if value < 0:
                    neg_bottom += value
                else:
                    pos_bottom += value
            npv = sum(components.values())
            if npv != 0:
                ax.plot(x, npv, "o", color="black", markersize=4, zorder=3)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(range(len(fractions)))
        ax.set_xticklabels([f"{fraction:g}" for fraction in fractions])
        ax.text(0.03, 0.97, code, transform=ax.transAxes, va="top", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)
    axes.flat[0].set_ylim(
        -round_up_axis_limit(-y_low) if y_low < 0 else 0, round_up_axis_limit(y_high)
    )

    for ax in axes.flat[len(codes) :]:
        ax.axis("off")
    handles = [
        legend_rect(
            lifetime_cost_components[k]["color"], label=lifetime_cost_components[k]["label"]
        )
        for k in keys_shown
    ] + [
        line_handle(
            label="Net Present Value",
            color="w",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=5,
        )
    ]
    axes.flat[-1].legend(handles=handles, loc="center", frameon=False, fontsize=fontsize)
    fig.supxlabel("Solar fraction (%)", fontsize=fontsize)
    fig.supylabel("Net Present Value ($M)", fontsize=fontsize)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "solar_comparison.png"))


def stacked_bar_figure(results, output_dir, filename):
    """Combined stacked bar figure for upgrade comparison with left (complete costs -N only)
    and right (relative NPV all configs) subplots, with connector line."""
    data_complete = build_fig3_bar_data(results, relative=False)
    data_relative = build_fig3_bar_data(results, relative=True)

    # One row of paired subplots per WWTP (left + right). Only keep facilities that have data (some
    # configs run a single technology)
    facilities_with_data = set(data_complete["Facility"])
    facilities = [
        facility
        for facility, label in facility_labels.items()
        if label in facilities_with_data and facility.startswith("o2__")
    ]
    n_wwtp = len(facilities)

    # Prepare figure: n_wwtp rows of paired subplots, plus a legend row below.
    fig = plt.figure(figsize=(3.33, 2.2 * n_wwtp))
    gs = fig.add_gridspec(
        n_wwtp + 1,
        2,
        width_ratios=[0.4, 1],
        hspace=0.5,
        wspace=0.5,
        height_ratios=[1] * n_wwtp + [0.45],
    )

    # Legend spans both columns in the bottom row.
    ax_legend = fig.add_subplot(gs[-1, :])

    panel_rows = []  # (ax_left, ax_right, facility label)
    for idx, facility in enumerate(facilities):
        facility_label = facility_labels[facility]
        ax_left = fig.add_subplot(gs[idx, 0])
        ax_right = fig.add_subplot(gs[idx, 1])
        add_subplot_label(ax_left, idx, x=-0.8)
        panel_rows.append((ax_left, ax_right, facility_label))

        # Left subplot: counterfactual complete costs for this facility - single bar
        plot_lifetime_cost_bars(
            data_complete[
                (data_complete["Facility"] == facility_label)
                & (data_complete["group"] == COUNTERFACTUAL_GROUP)
            ],
            ax_left,
            "group",
        )
        ax_left.get_legend().remove()

        # Overlay total-cost dots for each upgrade config on the left subplot
        upgrade_rows = data_complete[
            (data_complete["Facility"] == facility_label)
            & (data_complete["group"] != COUNTERFACTUAL_GROUP)
        ]
        codes_and_costs = sorted(
            (
                (case_ids[facility][get_base_key(upgrade_key)], lifetime_cost / 1e6)
                for upgrade_key, lifetime_cost in zip(
                    upgrade_rows["upgrade_key"], upgrade_rows["lifetime_cost"]
                )
            ),
            key=lambda code_and_cost: CASE_CODE_ORDER[code_and_cost[0][-1]],
        )
        if len(codes_and_costs) == 1:
            xs = [0.0]
        else:
            xs = np.linspace(-BAR_WIDTH / 2 * 0.7, BAR_WIDTH / 2 * 0.7, len(codes_and_costs))
        for x_off, (code, y_val) in zip(xs, codes_and_costs):
            ax_left.plot(x_off, y_val, "o", color="black", markersize=4, zorder=4)
            ax_left.text(x_off, y_val - 1.7, code, ha="center", va="top", fontsize=5)

        # Right subplot: relative NPV for this facility (all configs)
        plot_lifetime_cost_bars(
            data_relative[data_relative["Facility"] == facility_label],
            ax_right,
            "group",
            total_or_change="change",
        )
        ax_right.get_legend().remove()

    legend_elements = cost_component_handles(
        export_nonzero=data_relative["export_lifetime_cost"].abs().sum() > 0
    ) + [
        line_handle(
            label="Total Net Present Value",
            color="w",
            markerfacecolor="black",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=1,
        )
    ]
    ax_legend.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(-0.2, 1.5),
        fontsize=7,
        frameon=False,
        ncol=2,
    )
    ax_legend.axis("off")

    plt.subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.02)

    # Draw titles and connectors in figure coords after layout is final so transforms are accurate.
    # Connectors end at ax_right's tight-bbox left edge (incl. tick labels) so they miss tick text.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_from_display = fig.transFigure.inverted()
    for ax_left, ax_right, facility_label in panel_rows:
        # Center the row title over the combined width incl. y-labels (tight bboxes), in ax_right
        # axes fraction so it stays anchored at save time.
        x_mid_display = (
            ax_left.get_tightbbox(renderer).x0 + ax_right.get_tightbbox(renderer).x1
        ) / 2
        x_in_ax_right = ax_right.transAxes.inverted().transform((x_mid_display, 0))[0]
        ax_right.set_title(facility_label, fontsize=7, pad=8, x=x_in_ax_right)

        first_patch = ax_left.patches[0]
        bar_x = first_patch.get_x() + first_patch.get_width()
        x_start_display = ax_left.transData.transform((bar_x, 0))[0]
        x_end_display = ax_right.get_tightbbox(renderer).x0 + 20
        y_zero_display = ax_right.transData.transform((0, 0))[1]
        p_right = figure_from_display.transform((x_end_display, y_zero_display))
        for patch in ax_left.patches:
            height = patch.get_height()
            if height <= 0:
                continue
            y_bottom_display = ax_left.transData.transform((0, patch.get_y()))[1]
            y_top_display = ax_left.transData.transform((0, patch.get_y() + height))[1]
            fig.add_artist(
                MplPolygon(
                    [
                        figure_from_display.transform((x_start_display, y_bottom_display)),
                        figure_from_display.transform((x_start_display, y_top_display)),
                        p_right,
                    ],
                    transform=fig.transFigure,
                    facecolor=patch.get_facecolor(),
                    edgecolor="none",
                    alpha=0.2,
                    zorder=-1,
                    clip_on=False,
                )
            )

    save_fig(fig, os.path.join(output_dir, filename), dpi=300)


def set_hour_xaxis(ax, day_label, fontsize):
    ax.set_xticks([0, 8, 16, 24])
    ax.set_xticklabels([f"0:00\n{day_label}", "8:00", "16:00", "24:00"], fontsize=fontsize)
    ax.set_xlim(0, 24)


def fig_2_function(results, output_dir, centroid_ranked_days, data_name, horizon_days):
    # Rows are the three cases (A/B/C); columns: i. schematic, ii. day profile, iii. lifetime cost;
    # legends in a 4th row. Lengths in inches so ii/iii are square; xlabel_h fits the last x label.
    fig_w, margin, img_w, label_w = 7.0, 0.03, 1.5, 0.45
    col_gap, xlabel_h, note_h, legend_h = 0.85, 0.7, 0.25, 1.8
    # Each row gap carries the x label of the row above plus the storage-medium note over the cost panel
    row_gap = xlabel_h + note_h
    # The longest cost-legend entry runs past its column, so hold that width back on the right
    legend_overhang = 0.35
    panel = (fig_w - 2 * margin - label_w - legend_overhang - img_w - 2 * col_gap) / 2
    fig_h = 2 * margin + note_h + 3 * panel + 2 * row_gap + xlabel_h + legend_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    data_gs = fig.add_gridspec(
        3,
        2,
        left=(margin + label_w + img_w + col_gap) / fig_w,
        right=1 - (margin + legend_overhang) / fig_w,
        top=1 - (margin + note_h) / fig_h,
        bottom=(margin + legend_h + xlabel_h) / fig_h,
        hspace=row_gap / panel,
        wspace=col_gap / panel,
    )
    ax_profile = [fig.add_subplot(data_gs[row, 0]) for row in range(3)]
    ax_cost = [fig.add_subplot(data_gs[row, 1]) for row in range(3)]
    # Each legend sits directly below the column it describes
    ax_profile_legend = fig.add_axes(
        [ax_profile[2].get_position().x0, margin / fig_h, panel / fig_w, legend_h / fig_h]
    )
    ax_cost_legend = fig.add_axes(
        [ax_cost[2].get_position().x0, margin / fig_h, panel / fig_w, legend_h / fig_h]
    )

    curves = build_fig2_cost_curves(results)

    # Column i: schematic images A-C aligned to each row's band. An image's top band (label ink ends at 9%,
    # title starts at 16%) holds only its A./B./C. label: crop it and redraw the label in the strip.
    figures_dir = os.path.join(DATA_DIR, "figures")
    ax_images = []
    for row in range(3):
        band = ax_profile[row].get_position()
        ax_image = fig.add_axes([(margin + label_w) / fig_w, band.y0, img_w / fig_w, band.height])
        img = Image.open(os.path.join(figures_dir, f"{chr(65 + row)}.png"))
        ax_image.imshow(np.array(img)[int(0.12 * img.height) :])
        ax_image.axis("off")
        # imshow shrinks the axes box to the image aspect; anchoring it north keeps the box top on the
        # row top, so "i." lands at the same height as "ii." and "iii."
        ax_image.set_anchor("N")
        # A./B./C. sit 0.15 in left of the label strip; the tight bbox grows to fit
        add_subplot_label(ax_image, row, x=-(label_w + 0.15) / img_w, y=1.03, fontsize=13)
        ax_images.append(ax_image)

    _, _, tariff_key = parse_config_key(next(iter(results)))
    cases = [
        (
            facility,
            upgrade,
            get_config_name(
                f"{facility}__1.0__{BASELINE_MULTIPLIER}__{BASELINE_SMOOTHING}",
                f"{upgrade}__0",
                tariff_key,
            ),
        )
        for facility, upgrade in stacked_lifetime_cost_line_upgrade_by_facility.items()
    ]

    day = centroid_ranked_days[0]
    rate_data = get_tariff_data()[tariff_key]
    day_dt = datetime.strptime(day, "%Y-%m-%d")
    period_cats, _, _ = detect_charge_periods(rate_data, ENERGY, day_dt.month, day_dt.weekday())
    tariff_spans = {
        period: [
            (rate_data.loc[idx, "hour_start"], rate_data.loc[idx, "hour_end"])
            for idx, p in period_cats.items()
            if p == period
        ]
        for period in (PEAK, HALF_PEAK)
    }

    # Column ii shows each config's OPTIMIZED design (matching column iii) on the day closest to the
    # centroid where every optimum visibly charges AND discharges; else the plain centroid day.
    finder_items = []
    for _, _, config_key in cases:
        _, best_design_key = get_design_data(results[config_key])
        # Skip ~0h-optimum configs (e.g. AB-T): they never cycle, so requiring a two-way response
        # from them would distort the shared-day pick for the configs that do.
        if best_design_key is None or float(parse_design_key(best_design_key)[0]) < 0.5:
            continue
        finder_items.append(
            (get_intermediate_dir(data_name, config_key, horizon_days), best_design_key)
        )
    qualifying = days_with_charge_and_discharge(finder_items)
    shared_day = next((d for d in centroid_ranked_days if d in qualifying), day)
    if shared_day == day and day not in qualifying:
        print(
            f"WARNING fig2 column-ii: no day had visible charge+discharge for all 3 configs; using centroid day {day}"
        )
    day_label = datetime.strptime(shared_day, "%Y-%m-%d").strftime("%m/%d")

    for row, (facility, _, config_key) in enumerate(cases):
        # Cheapest design even if it doesn't beat the counterfactual (AB-T's minimum tank still plots)
        best_design_key = cheapest_design_key(results[config_key]["all_results"])
        # EXACT design_key only: an alphabetical fallback once plotted the wrong design/medium
        profile = pd.read_csv(
            os.path.join(
                get_intermediate_dir(data_name, config_key, horizon_days),
                f"{best_design_key}_{shared_day}_profile.csv",
            )
        )
        day_data = {
            col: np.where(np.abs(values) >= PROFILE_ZERO_THRESHOLD, values, 0)
            for col, values in profile.items()
        }
        time = np.linspace(0, 24, len(profile))
        ax = ax_profile[row]
        for period, color in [(HALF_PEAK, HALF_PEAK_COLOR), (PEAK, PEAK_COLOR)]:
            for start, end in tariff_spans[period]:
                ax.axvspan(start, end, color=color, zorder=0)

        baseline_power = day_data["Edot_t_baseline"] / 1000
        # If no viable upgrade (best_actual_hours == 0), show baseline only — no optimized diff
        no_upgrade = curves[row]["best_actual_hours"] == 0.0
        net = baseline_power if no_upgrade else day_data["Edot_t_net_recalc"] / 1000
        ax.plot(time, net, color="k", linewidth=PROFILE_LINEWIDTH, zorder=2, label="Optimized")
        ax.plot(
            time,
            baseline_power,
            color="grey",
            linewidth=PROFILE_LINEWIDTH,
            linestyle="-",
            zorder=3,
            label="Baseline",
        )
        if not no_upgrade:
            power_diff = net - baseline_power
            tol = POWER_FILL_REL_TOL * np.abs(baseline_power)
            for mask, color in [
                (power_diff > tol, cb_palette[7]),
                (power_diff < -tol, cb_palette[2]),
            ]:
                ax.fill_between(
                    time,
                    baseline_power,
                    net,
                    where=mask,
                    interpolate=True,
                    color=color,
                )
        set_hour_xaxis(ax, day_label, fontsize=10.5)
        ax.set_ylabel("Power (MW)", fontsize=11)
        ax.tick_params(axis="y", labelsize=10.5)

        tick_step = 0.1 if facility.startswith("air__") else 0.5
        all_power = (
            baseline_power
            if no_upgrade
            else np.concatenate([baseline_power, day_data["Edot_t_net"] / 1000])
        )
        y_floor = np.floor(all_power.min() / tick_step) * tick_step
        y_ceil = np.ceil(all_power.max() / tick_step) * tick_step
        ax.set_ylim(y_floor, y_ceil)
        ax.set_yticks(np.arange(y_floor, y_ceil + tick_step / 2, tick_step))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    profile_handles = [
        line_handle(
            label=label, color=color, linestyle="-", linewidth=PROFILE_LINEWIDTH, marker="None"
        )
        for label, color in [("Baseline", "grey"), ("Optimized", "k")]
    ] + [
        legend_rect(color, label=label)
        for color, label in [
            (cb_palette[7], "Power Increase"),
            (cb_palette[2], "Power Decrease"),
            (HALF_PEAK_COLOR, "Half-Peak Window"),
            (PEAK_COLOR, "Peak Window"),
        ]
    ]
    # Trimmed handles keep the longest entries inside their column's width
    ax_profile_legend.legend(
        handles=profile_handles,
        loc="upper left",
        bbox_to_anchor=(0, 1.0),
        fontsize=10.5,
        frameon=False,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.6,
    )
    ax_profile_legend.axis("off")

    for ax, curve in zip(ax_cost, curves):
        plot_fig2_cost_curve(ax, curve)

    # Only legend the components that actually appear — H2 and export are zero unless the case
    # recovers gas or sells back
    nonzero_components = {
        key
        for curve in curves
        for key, values in curve["component_arrays"].items()
        if np.any(values)
    }
    legend_elements = [
        *cost_component_handles(
            h2_nonzero="h2" in nonzero_components,
            export_nonzero="export" in nonzero_components,
        ),
        line_handle(
            label="Optimized Design",
            color="black",
            linestyle="None",
            marker="|",
            markersize=10,
            markeredgewidth=0.8,
            markeredgecolor="black",
        ),
    ]
    ax_cost_legend.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0, 1.0),
        fontsize=10.5,
        frameon=False,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.6,
    )
    ax_cost_legend.axis("off")

    fig.align_ylabels(ax_profile + ax_cost)
    # Roman numerals sit just left of each panel's y-axis label (the schematic for i.), measured after layout
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 3 / 72 * fig.dpi
    for row in range(3):
        for ax, label in ((ax_images[row], "i."), (ax_profile[row], "ii."), (ax_cost[row], "iii.")):
            left = ax.yaxis.get_tightbbox(renderer) if ax.axison else ax.get_window_extent(renderer)
            x = ax.transAxes.inverted().transform((left.x0 - pad, 0))[0]
            ax.text(x, 1.0, label, transform=ax.transAxes, fontsize=11, va="top", ha="right", zorder=100)
    save_fig(
        fig,
        os.path.join(output_dir, "fig_2stacked_lifetime_cost"),
        dpi=300,
    )
    return shared_day


def fig4_function(sensitivity_config, results_by_config, output_dir):
    facility_sweep, facility_counterfactual, active_axes, axis_devs, all_dev_pcts, dev_to_col = (
        build_fig4_sweeps(sensitivity_config, results_by_config)
    )
    # Fig 4 shows only the HPOAS facilities
    facility_sweep = {
        wwtp_type: sweeps
        for wwtp_type, sweeps in facility_sweep.items()
        if wwtp_type.startswith("o2__")
    }
    n_fac = len(facility_sweep)
    baseline_col = dev_to_col.get(0)
    if baseline_col is None:
        print("[FIGURE 3] No baseline column (0% deviation) — skipping relative heatmap")
        return
    non_baseline_cols = [c for c in range(len(all_dev_pcts)) if c != baseline_col]

    # Per facility: NPV % change vs baseline column, infeasible.
    # NPV = expansion_only - lifetime_cost.total; pct change is relative to baseline NPV.
    full_matrices = {}
    for wwtp_type, sweeps in facility_sweep.items():
        lifetime_cost_matrix, infeasible = build_fig4_matrix(
            sweeps, active_axes, axis_devs, dev_to_col
        )
        counterfactual_matrix, _ = build_fig4_matrix(
            facility_counterfactual[wwtp_type], active_axes, axis_devs, dev_to_col
        )
        npv_matrix = counterfactual_matrix - lifetime_cost_matrix
        full_matrices[wwtp_type] = (
            pct_change_matrix(npv_matrix, npv_matrix[:, [baseline_col]]),
            infeasible,
        )

    # Drop rows with no off-baseline NPV change in any facility; show without the baseline column
    keep_row_idx = [
        i
        for i in range(len(active_axes))
        if any(
            np.any(np.nan_to_num(relative[i, non_baseline_cols]) != 0)
            for relative, _ in full_matrices.values()
        )
    ]
    display_matrices = {
        wwtp_type: [m[np.ix_(keep_row_idx, non_baseline_cols)] for m in matrices]
        for wwtp_type, matrices in full_matrices.items()
    }
    display_axes = [active_axes[i] for i in keep_row_idx]
    display_dev_pcts = [d for d in all_dev_pcts if d != 0]
    n_display_cols = len(display_dev_pcts)
    n_display_rows = len(display_axes)
    divider_x = sum(1 for d in display_dev_pcts if d < 0) - 0.5  # between last neg and first pos
    # 1-column journal width; height scaled to number of rows and facilities
    figsize = (3.33, (0.25 * n_display_rows + 1.8) * n_fac)

    # Shared colour scale: computed on the display slice (no baseline col, filtered rows)
    shared_vals = np.concatenate([relative.ravel() for relative, _ in display_matrices.values()])
    shared_vals = shared_vals[np.isfinite(shared_vals)]
    raw_max = max(np.abs(shared_vals).max(initial=0.0), 1e-6)
    abs_max = float(min(np.ceil(raw_max / 20) * 20, np.ceil(raw_max / 25) * 25))

    fig, axes = plt.subplots(n_fac, 1, figsize=figsize, squeeze=False)
    for fac_idx, (wwtp_type, (relative, infeasible)) in enumerate(display_matrices.items()):
        ax = axes[fac_idx, 0]
        upgrade_key = sensitivity_config["upgrade_keys"][wwtp_type][0]
        # Exact-zero cells are analytic true zeros (e.g. unused supplemental capex): render them
        # set_bad white, not a faint near-white tint. Epsilon so real sub-1% changes keep color.
        matrix = np.where(np.abs(relative) < 1e-9, np.nan, relative)
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap=FIG4_CMAP,
            interpolation="nearest",
            norm=TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max),
        )

        for r, c in np.argwhere(infeasible):
            ax.add_patch(
                plt.Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor="black",
                    edgecolor="white",
                    hatch="////",
                    linewidth=0.5,
                )
            )

        ax.set_xticks(np.arange(n_display_cols) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_display_rows) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", size=0)
        ax.set_yticks(range(n_display_rows))
        ax.set_yticklabels([sweep_axis_labels[a] for a in display_axes], fontsize=7)
        ax.set_xticks(range(n_display_cols))
        ax.set_xticklabels([f"{d:.0f}%" for d in display_dev_pcts], fontsize=6)
        ax.set_xlabel("Deviation from Baseline", fontsize=7)
        ax.axvline(x=divider_x, color="black", linewidth=1.0, zorder=6)
        ax.set_title(case_ids[wwtp_type][get_base_key(upgrade_key)], fontsize=7)
        add_subplot_label(ax, fac_idx, x=-0.45)
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.18, fraction=0.05)
        cbar.set_label("NPV Change (%)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        tick_step = abs_max / 3
        cbar.set_ticks(np.arange(-abs_max, abs_max + tick_step / 2, tick_step))

    fig.subplots_adjust(left=0.30, right=0.97, top=0.97, bottom=0.03, hspace=0.25)
    save_fig(fig, os.path.join(output_dir, "fig4.png"), dpi=300)

    scatter_axes = [a for a in ("window", "energy", "demand") if a in active_axes]
    if scatter_axes:
        scatter_colors = {"window": cb_palette[0], "energy": cb_palette[1], "demand": cb_palette[2]}
        scatter_markers = {"window": "o", "energy": "s", "demand": "^"}
        fig, axes_sc = plt.subplots(1, n_fac, figsize=(2.6 * n_fac, 3.2), squeeze=False)
        for fac_idx, (wwtp_type, sweep_data) in enumerate(facility_sweep.items()):
            ax = axes_sc[0, fac_idx]
            for axis in scatter_axes:
                valid = [
                    (dev, sweep_data[axis][x])
                    for dev, x in axis_devs[axis]
                    if sweep_data[axis].get(x) is not None and np.isfinite(sweep_data[axis][x])
                ]
                if valid:
                    vx, vy = zip(*valid)
                    vy = [v / 1e6 for v in vy]  # $M
                    ax.scatter(
                        vx,
                        vy,
                        color=scatter_colors[axis],
                        marker=scatter_markers[axis],
                        label=sweep_axis_labels[axis],
                        s=40,
                        zorder=3,
                    )
                    ax.plot(vx, vy, color=scatter_colors[axis], linewidth=1, zorder=2)
            if fac_idx == 0:
                ax.set_ylabel("Lifetime cost ($M)", fontsize=12)
            ax.set_title(facility_labels[wwtp_type], fontsize=12)
            ax.tick_params(labelsize=12)
        fig.supxlabel("Deviation from baseline (%)", fontsize=12)
        plt.tight_layout()
        # Every panel draws the same sweep axes, so one legend covers them
        fig.legend(
            *axes_sc[0, 0].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(scatter_axes),
            fontsize=12,
            frameon=False,
        )
        save_fig(fig, os.path.join(output_dir, "billing_sensitivity_scatter.png"))


def find_centroid_day(scaled_data_path, solved_days, save_plot_dir=None, fig2_day=None):
    """Rank days by how close their airflow profile is to the mean (centroid) profile.

    Returns a list of date strings (YYYY-MM-DD) ordered closest-to-centroid first, or []
    if no complete days. Only considers solved_days (set of date strings).
    If save_plot_dir is given, saves a comparison plot there, adding fig2_day's profile if given.
    """
    datetime_col = "DateTime"
    df = pd.read_csv(scaled_data_path)
    if "Blower_AerationBasin_Air_Flow" in df.columns:
        flow_col = "Blower_AerationBasin_Air_Flow"
    else:
        flow_col = "O2Plant_AerationBasin_Oxygen_Flow"
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["date"] = df[datetime_col].dt.date

    # Keep only complete solved days (96 timesteps = 24 hours at 15-min intervals)
    daily_counts = df.groupby("date").size()
    df = df[df["date"].isin(daily_counts[daily_counts == 96].index)]
    df = df[df["date"].isin({pd.Timestamp(d).date() for d in solved_days})]

    profiles = {
        date: df[df["date"] == date].sort_values(datetime_col)[flow_col].values
        for date in df["date"].unique()
    }
    profile_array = np.array(list(profiles.values()))
    dates = np.array(list(profiles.keys()))
    centroid = np.mean(profile_array, axis=0)
    distances = np.sqrt(np.sum((profile_array - centroid) ** 2, axis=1))

    ranked_dates = [str(d) for d in dates[np.argsort(distances)]]
    if save_plot_dir is None:
        return ranked_dates
    closest_idx = np.argmin(distances)
    furthest_idx = np.argmax(distances)

    hours = np.arange(96) / 4
    # Same air-flow color as fig_s3; line style tells the three profiles apart
    plt.figure(figsize=(8, 5.5))
    plt.plot(
        hours, centroid, "-", color=cb_palette[2], linewidth=3.5, label="Centroid (Mean Profile)"
    )
    plt.plot(
        hours,
        profile_array[closest_idx],
        "--",
        color=cb_palette[2],
        linewidth=2.5,
        label=f"Closest Day ({dates[closest_idx]})",
    )
    plt.plot(
        hours,
        profile_array[furthest_idx],
        ":",
        color=cb_palette[2],
        linewidth=2.5,
        label=f"Furthest Day ({dates[furthest_idx]})",
    )
    if fig2_day is not None:
        plt.plot(
            hours,
            profiles[pd.Timestamp(fig2_day).date()],
            "-.",
            color="black",
            linewidth=1.5,
            label=f"Fig. 2 Day ({fig2_day})",
        )
    plt.xlabel("Hour of Day", fontsize=18)
    plt.ylabel("Blower Airflow (m³/day)", fontsize=18)
    plt.legend(fontsize=14, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    save_fig(plt.gcf(), os.path.join(save_plot_dir, "centroid_day.png"))

    return ranked_dates


def generate_plots(run_configs, config_values):
    run_name = config_values["base_run_name"]
    output_dir = os.path.join(CONFIG_PLOTS_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for file_path in glob.glob(os.path.join(OUTPUT_DATA_DIR, run_name, "*___*.json")):
        with open(file_path) as f:
            results[os.path.splitext(os.path.basename(file_path))[0]] = json.load(f)

    if not config_values["fig4"]:
        filtered = {}
        for config_key, config_data in results.items():
            base_wwtp_key, upgrade_key, tariff_key = parse_config_key(config_key)
            matching = next(
                (k for k in run_configs if base_wwtp_key.startswith(get_base_key(k))), None
            )
            if matching is None:
                continue
            run_config = run_configs[matching]
            if (
                upgrade_key in run_config["upgrade_keys"]
                and tariff_key in run_config["tariff_configs"]
            ):
                filtered[config_key] = config_data
        results = filtered

    if config_values["si_solar"]:
        si_solar_figure(results, output_dir)

    if config_values.get("si_nu_rec", False):
        si_sweep_scatter(
            results,
            output_dir,
            "upgrade",
            "Recovery Efficiency",
            "recovery_efficiency_scatter.png",
            tariff_key=BASELINE_TARIFF,
        )

    if config_values["fig_2"]:
        si_figure_5(results, output_dir)
        # The footprint figure needs the case-study plant's basin geometry, which only the paper config has
        if "aeration_basin_volume_mg" in config_values:
            plot_storage_vs_bioreactor_volume(results, output_dir, config_values)
        si_convergence_combined_figure(results, output_dir, config_values)
        with open(os.path.join(OUTPUT_DATA_DIR, run_name, "day_profiles.json")) as f:
            solved_days = json.load(f)["day_profiles"]
        # Row ii should show a dry-season day — matches the summer_multiplier design condition.
        dry_season_days = {d for d in solved_days if is_dry_season(d.split("-")[1])}
        scaled_data_path = os.path.join(OUTPUT_DATA_DIR, run_name, "scaled_data.csv")
        centroid_ranked_days = find_centroid_day(scaled_data_path, dry_season_days)
        fig2_day = fig_2_function(
            results,
            output_dir,
            centroid_ranked_days,
            config_values["data_name"],
            config_values["horizon_days"],
        )
        # Drawn after fig 2 so it can show the day fig 2 plots
        find_centroid_day(scaled_data_path, dry_season_days, output_dir, fig2_day)

    if config_values["fig3"]:
        stacked_bar_figure(results, output_dir, "fig3.png")

    if config_values["fig4"]:
        design_points_map = config_values.get("design_points", {})
        single_design_point = config_values.get("design_point")
        results_by_config = {}
        for config_key, config_data in results.items():
            base_wwtp_key, upgrade_key, _ = parse_config_key(config_key)
            all_results = config_data["all_results"]
            if design_points_map:
                design_point = design_points_map.get(
                    f"{get_base_key(base_wwtp_key)}___{upgrade_key}"
                )
                if design_point is None:
                    continue
                design_key = build_design_key(design_point)
            elif single_design_point is not None:
                design_key = build_design_key(single_design_point)
            else:
                design_key = cheapest_design_key(all_results)
                if design_key is None:
                    continue
            if design_key not in all_results:
                print(f"Warning: {design_key} not in {config_key}, skipping")
                continue
            results_by_config[config_key] = all_results[design_key]

        # Only pass upgrade_keys for wwtp types that have at least one valid result
        result_keys = [parse_config_key(k)[:2] for k in results_by_config]
        valid_upgrade_keys = {
            wwtp_type: upgrades
            for wwtp_type, upgrades in config_values["upgrade_keys"].items()
            if any(
                base_wwtp_key.startswith(wwtp_type + "__") and upgrade_key == upgrades[0]
                for base_wwtp_key, upgrade_key in result_keys
            )
        }

        if valid_upgrade_keys:
            sensitivity_config = {
                "capex_sensitivity": config_values.get("capex_sensitivity"),
                "upgrade_keys": valid_upgrade_keys,
                "scale_mgd": config_values["scale_mgd"],
            }
            fig4_function(sensitivity_config, results_by_config, output_dir)

    # Paper figures (config "paper_figures": run-folder file -> path under output_plots/) leave diagnostics
    for diagnostic_name, paper_name in config_values.get("paper_figures", {}).items():
        paper_path = os.path.join(OUTPUT_PLOTS_DIR, paper_name)
        os.makedirs(os.path.dirname(paper_path), exist_ok=True)
        for ext in (".png", ".tif"):
            diagnostic_path = os.path.join(output_dir, os.path.splitext(diagnostic_name)[0] + ext)
            # Copy then delete, not os.replace, so a symlinked paper path is written through
            shutil.copyfile(diagnostic_path, os.path.splitext(paper_path)[0] + ext)
            os.remove(diagnostic_path)
        print(f"Moved {diagnostic_name} to {paper_path} (+ .tif)")
