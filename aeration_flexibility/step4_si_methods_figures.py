"""SI figures that need no facility data, written to output_plots/si/: fig_s1 (US HPOAS map), fig_s2 (energy and
O2 intensity by treatment technology) and fig_s9 (iterative compressor-coefficient linearization).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from helpers.parameters import *
from helpers.compressor_power import *
from helpers.config_keys import cb_palette
from helpers.plotting import save_fig


EXTRA_HPOAS_CWNS_IDS = [
    "29001007001",  # Hannibal WWTP, MO. 2022 CWNS UNIT_PROCESS_NOTES
    "25000004001",  # South Essex Sewer District WWTP, MA. 2022 CWNS UNIT_PROCESS_NOTES
    "25000128001",  # MWRA Deer Island, MA. 2022 CWNS UNIT_PROCESS_NOTES
    "06002032005",  # SF Oceanside WPCP, CA. https://www.waterboards.ca.gov/sanfranciscobay/board_info/agendas/2016/August/sfpuc/Signed_Proposed_Settlement_Agreement.pdf
    "06002032003",  # SF Southeast WPCP, CA. https://www.enr.com/articles/64098-jacobs-pcl-wins-193m-san-francisco-wastewater-contract
    "06004010001",  # LA Hyperion, CA. https://www.waterboards.ca.gov/water_issues/programs/tmdl/records/region_4/2011/ref3839.pdf
    "22009071001",  # New Orleans, LA. https://swbno.org/Wastewater/Overview
    "42000094002",  # Philadelphia Southwest WPCP, PA. https://lp.constantcontactpages.com/ev/reg/jauxkh2
    "12000017028",  # MDWASD South District, FL. https://www.miamidade.gov/water/library/reports/coastal-wetlands-conceptual-appendix-e.pdf
]
# HPOAS plants in the Chapin et al. (2024) WWTP tariff dataset
TARIFF_DATASET_CWNS_IDS = [
    "12000017004",
    "34006012001",
    "12000053001",
    "47000245002",
    "18000061001",
    "53000776002",
    "21000025001",
    "53001280001",
    "24000001002",
    "06002036001",
    "26000596001",
    "06004009001",
    "34001082001",
    "34002065001",
]

PSA_MAX_FLOW_MGD = 30  # larger HPOAS plants likely use CRYO. Shammas & Wang 2009 Fig 7.3


def figure_hpo_map():
    """fig_s1: US HPOAS facilities sized by design flow, outlined if in the Chapin et al. (2024) tariff
    dataset. Also prints the HPOAS shares of US activated sludge plants quoted in the Introduction and S1.1.

    HPOAS plants list "Activated Sludge, Pure Oxygen" in the EPA 2012 CWNS (the latest survey with unit
    processes), plus EXTRA_HPOAS_CWNS_IDS.
    """
    # EPA 2012 Clean Watersheds Needs Survey files, copied unchanged from npdes-analysis data/cwns/2012
    cwns_dir = os.path.join(DATA_DIR, "maps", "cwns_2012")
    units = pd.read_csv(
        os.path.join(cwns_dir, "Unit_Process_Details.csv"), encoding="latin1", dtype=str
    )
    design_flow = (
        pd.read_csv(
            os.path.join(cwns_dir, "Facility_Details.csv"),
            encoding="latin1",
            dtype={"CWNS Number": str},
            low_memory=False,
        )
        .drop_duplicates("CWNS Number")
        .set_index("CWNS Number")["Present Design Total Flow (Mgal/d)"]
    )  # MGD

    # Activated sludge = any suspended-growth process (conventional and variants, oxidation ditch, SBR, MBR)
    hpoas_ids = set(
        units.loc[units["Unit Process"] == "Activated Sludge, Pure Oxygen", "CWNS Number"]
    )
    hpoas_ids = sorted(hpoas_ids | set(EXTRA_HPOAS_CWNS_IDS))
    is_activated_sludge = units["Unit Process"].str.contains(
        "Activated Sludge|Oxidation Ditch|Sequencing Batch|Membrane Bio", na=False
    )
    activated_sludge_ids = set(units.loc[is_activated_sludge, "CWNS Number"]) | set(hpoas_ids)
    hpoas_flow = design_flow.reindex(hpoas_ids).fillna(0)
    activated_sludge_flow = design_flow.reindex(sorted(activated_sludge_ids)).fillna(0).sum()
    print(
        f"HPOAS: {len(hpoas_ids)} of {len(activated_sludge_ids)} US activated sludge plants "
        f"({100 * len(hpoas_ids) / len(activated_sludge_ids):.1f}%), "
        f"{100 * hpoas_flow.sum() / activated_sludge_flow:.1f}% of their design flow"
    )
    is_cryo = hpoas_flow >= PSA_MAX_FLOW_MGD
    print(
        f"HPOAS >= {PSA_MAX_FLOW_MGD} MGD (likely CRYO): {is_cryo.sum()} of {len(hpoas_ids)} plants, "
        f"{100 * hpoas_flow[is_cryo].sum() / hpoas_flow.sum():.1f}% of HPOAS design flow"
    )

    # FacLat/FacLong from EPA ECHO by NPDES permit (2026-11-12), https://echodata.epa.gov/echo/cwa_rest_services.get_facilities?p_pid=<permit>&qcolumns=1,2,26,27
    # FLA041378, FL0024805, FLA042137 and IN0031950 were matched by facility name (no usable CWNS permit)
    coordinates = pd.read_csv(
        os.path.join(DATA_DIR, "maps", "hpoas_coordinates.csv"), dtype={"cwns_id": str}
    ).set_index("cwns_id")
    facilities = coordinates.loc[hpoas_ids].assign(
        design_flow_mgd=hpoas_flow,
        in_tariff_dataset=lambda df: df.index.isin(TARIFF_DATASET_CWNS_IDS),
    )
    # US Census Bureau cb_2018_us_state_20m state boundaries, coordinates rounded to 4 decimals
    with open(os.path.join(DATA_DIR, "maps", "us_states_20m.geojson")) as f:
        states = json.load(f)["features"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for state in states:
        geometry = state["geometry"]
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        for polygon in polygons:
            ax.add_patch(Polygon(polygon[0], facecolor="#e8e8e8", edgecolor="white", linewidth=0.5))

    size_scale = 10
    for in_tariff, edgecolor, linewidth in ((False, cb_palette[6], 1), (True, "black", 3)):
        subset = facilities[facilities["in_tariff_dataset"] == in_tariff]
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            s=subset["design_flow_mgd"] * size_scale,
            color=cb_palette[0],
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=0.7,
            zorder=4 if in_tariff else 3,
        )

    legend_elements = [
        Patch(facecolor="none", edgecolor="black", linewidth=3, label="In tariff dataset"),
    ] + [
        Line2D(
            [],
            [],
            color="white",
            marker="o",
            linestyle="None",
            markerfacecolor=cb_palette[0],
            markeredgecolor=cb_palette[6],
            alpha=0.7,
            markersize=np.sqrt(flow * size_scale),
            label=f"{flow} MGD",
        )
        for flow in (1, 10, 100)
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        fontsize=15,
        frameon=False,
    )
    ax.set_xlim(-130, -65)
    ax.set_ylim(23, 50)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_fig(fig, os.path.join(SI_PLOTS_DIR, "fig_s1_hpo_map.png"))


def figure_energy_o2_intensity():
    """fig_s2: literature ranges of secondary-process energy (aeration + mixing) and O2 demand by
    treatment technology, from data/capex/energy_intensities.csv (sources in its Source column)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "capex", "energy_intensities.csv"))
    y = np.arange(len(df))
    fig, (ax_e, ax_o) = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True, dpi=200)
    for ax, lo_col, hi_col, color, xlabel in (
        (
            ax_e,
            "Lower limit secondary energy (kwh/m3)",
            "Upper limit secondary energy (kwh/m3)",
            cb_palette[2],
            "Secondary aeration + mixing energy (kWh/m$^3$ treated)",
        ),
        (
            ax_o,
            "Oxygen demand low (kg O2 / m3)",
            "Oxygen demand high (kg O2 / m3)",
            cb_palette[0],
            "O$_2$ demand (kg O$_2$/m$^3$ treated)",
        ),
    ):
        lo, hi = df[lo_col].to_numpy(), df[hi_col].to_numpy()
        # A single sourced value (low == high) would be a zero-width bar, so mark it instead
        single = lo == hi
        ax.barh(y[~single], (hi - lo)[~single], left=lo[~single], height=0.5, color=color)
        ax.plot(lo[single], y[single], "D", color=color, markersize=7)
        ax.set_xlim(0, np.nanmax(hi) * 1.1)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    # CSV row order (non-nitrifying first) reads top to bottom
    ax_e.set_yticks(y, df["Technology"])
    ax_e.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, os.path.join(SI_PLOTS_DIR, "fig_s2_energy_o2_intensity.png"))


def figure_iterative_coeff_linearization():
    """fig_s9: the compression power coefficient c(P) is concave in tank pressure, so the LP fixes it per timestep
    and re-sets it from the solved pressure after each solve (damped), shown for a PSA O2 tank."""
    P_low = P_init_map["psa"]
    P_min, P_max = (P_AER + P_ATM) * 1.05, 4.0  # delivery-pressure floor, a typical OP-T rating
    # Stage count is fixed from the design maximum, as in add_gas_tank_constraints
    n_stages = n_stages_for_ratio(P_max / P_low)
    pressures = np.linspace(P_min, P_max, 200)
    # W per mol/hr reads more easily than kW per mol/hr
    coeff = lambda P: 1000 * comp_lp_coeff(P_low, P, gas="o2", n_stages=n_stages)
    P_start, P_solved = (P_min + P_max) / 2, P_min + 0.75 * (P_max - P_min)
    c_start, c_solved = coeff(P_start), coeff(P_solved)

    # Damped update with alpha = 0.5 (solve_with_iterative_coeff), holding the solved pressure fixed
    c_iter = [c_start]
    for _ in range(10):
        c_iter.append(0.5 * c_solved + 0.5 * c_iter[-1])
    gap_pct = [100 * abs(c - c_solved) / c_solved for c in c_iter]

    fig, (ax_c, ax_gap) = plt.subplots(1, 2, figsize=(7, 3))
    ax_c.plot(pressures, [coeff(P) for P in pressures], color="black", linewidth=1.5)
    ax_c.plot(P_start, c_start, "o", markerfacecolor="white", markeredgecolor=cb_palette[0])
    ax_c.plot(P_solved, c_solved, "o", color=cb_palette[1])
    ax_c.annotate(
        "Initial guess",
        (P_start, c_start),
        (8, -22),
        textcoords="offset points",
        fontsize=11,
        color=cb_palette[0],
    )
    ax_c.annotate(
        "Solved\npressure",
        (P_solved, c_solved),
        (8, -26),
        textcoords="offset points",
        fontsize=11,
        color=cb_palette[1],
    )
    ax_c.set_xlabel("Tank pressure (MPa)", fontsize=11)
    ax_c.set_ylabel("Compression power\n(W per mol/hr O$_2$)", fontsize=11)

    ax_gap.plot(range(len(gap_pct)), gap_pct, "o-", color="black", markersize=4)
    ax_gap.set_ylim(bottom=0)
    ax_gap.set_xlabel("Iteration", fontsize=11)
    ax_gap.set_ylabel("Coefficient error (%)", fontsize=11)
    for ax, label in ((ax_c, "A"), (ax_gap, "B")):
        ax.text(-0.3, 1.02, label, transform=ax.transAxes, fontsize=12)
        ax.tick_params(labelsize=11)
    fig.tight_layout()
    save_fig(fig, os.path.join(SI_PLOTS_DIR, "fig_s9_coeff_linearization.png"), dpi=300)


if __name__ == "__main__":
    os.makedirs(SI_PLOTS_DIR, exist_ok=True)
    figure_hpo_map()
    figure_energy_o2_intensity()
    figure_iterative_coeff_linearization()
