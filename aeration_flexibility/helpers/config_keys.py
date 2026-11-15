"""Build and parse the string keys naming configs, plants, upgrades, tariffs and designs.
Also holds the shared cost-component lists and plot colors."""

from collections import namedtuple

import numpy as np


BaseWwtpKey = namedtuple("BaseWwtpKey", ["gas", "tech", "scale_mgd", "multiplier", "smoothing"])


def split_key(key):
    return key.split("__")


def get_base_key(key):
    """Returns first 2 parts of either a base wwtp key or the upgrade key."""
    return "__".join(split_key(key)[:2])


def get_config_name(base_wwtp_key, upgrade_key, tariff_key):
    return f"{base_wwtp_key}___{upgrade_key}___{tariff_key}"


def parse_base_wwtp_key(key):
    """Parse base_wwtp_key into its 5 named fields."""
    parts = split_key(key)
    return BaseWwtpKey(
        gas=parts[0],
        tech=parts[1],
        scale_mgd=parts[2],
        multiplier=float(parts[3]),
        smoothing=float(parts[4]),
    )


def parse_config_key(config_key):
    """Parse config key into base_wwtp_key, upgrade_key, tariff_key.
    The base_wwtp_key contains summer config info (e.g., air__compressor__1.0__1.5__1.0)
    """
    parts = config_key.split("___")
    return parts[0], parts[1], parts[2]


def parse_design_key(design_key):
    """Parse design key into hours, compression_ratio, charge_power_ratio, addtl_capacity_frac.

    Ignores a trailing "~gas"/"~liq" storage-medium marker — the job queue runs one gas_tank
    design point under both media and keeps the cheaper (see build_job_queue_for_design_keys).
    """
    parts = design_key.split("~")[0].split("__")
    return parts[0], parts[1], parts[2], parts[3]


def build_design_key(design_point):
    """Build design key string from a 4-tuple (hr_of_O2, compression_ratio, charge_power_ratio, addtl_capacity_frac)."""
    return f"{design_point[0]}__{design_point[1]}__{design_point[2]}__{design_point[3]}"


def is_valid_lifetime_cost(lifetime_cost):
    """Check if Lifetime Cost is a valid finite number."""
    if lifetime_cost is None:
        return False
    return np.isfinite(lifetime_cost)


def flat_cost_components(lifetime_cost, source="new"):
    """lifetime_cost_components values, with the by_component_{source} breakdown layered over the totals."""
    by_component = lifetime_cost.get(
        f"by_component_{source}", lifetime_cost.get("by_component", {})
    )
    merged = {**lifetime_cost, **by_component}
    return {k: merged.get(k, 0) for k in lifetime_cost_components}


def counterfactual_cost_components(lifetime_cost):
    """Baseline cost components with zero storage capex and expansion_only aeration capex."""
    components = flat_cost_components(lifetime_cost, "baseline")
    components["storage_capex"] = 0.0
    components["o_and_m"] = 0.0  # counterfactual has no storage equipment to maintain
    components["capex_addtl_capacity"] = lifetime_cost.get("expansion_only", 0)
    return components


def cheapest_design_key(all_results):
    """Lowest-cost valid design, whether or not it beats the counterfactual (None if none valid)."""
    return min(
        (k for k, r in all_results.items() if is_valid_lifetime_cost(r["lifetime_cost"]["total"])),
        key=lambda k: all_results[k]["lifetime_cost"]["total"],
        default=None,
    )


def get_best_design_key(all_results):
    """Return the lowest-cost valid design if it beats the counterfactual (positive NPV), else None."""
    best_key = cheapest_design_key(all_results)
    if best_key is None:
        return None
    lifetime_cost = all_results[best_key]["lifetime_cost"]
    # total is net of baseline opex, so the counterfactual's matching cost is its expansion-only capex
    if lifetime_cost["total"] >= lifetime_cost["expansion_only"]:
        return None
    return best_key


def get_upgrade_info(upgrade_key):
    """Return classification dict for an upgrade key."""
    parts = split_key(upgrade_key)
    o2_tech = parts[0]
    storage_type = parts[1]
    is_liquid_tank = storage_type == "liquid_tank"
    is_gas_tank = storage_type == "gas_tank"
    nu_rec = float(parts[2])
    has_elec = o2_tech == "elec"
    return {
        "storage_type": storage_type,
        "nu_rec": nu_rec,
        "o2_tech_for_storage": o2_tech if o2_tech.lower() != "none" else None,
        "is_liquid_tank": is_liquid_tank,
        "is_battery": storage_type == "battery",
        "is_gas_tank": is_gas_tank,
        "is_tank": is_liquid_tank or is_gas_tank,
        "has_elec": has_elec,
        "has_surplus": o2_tech == "none",
        # Derived feature flags
        "has_recovery": nu_rec > 0,
        "has_h2": has_elec and is_gas_tank,
    }


cb_palette = [
    "#0173B2",
    "#DA8E20",
    "#4aa68dff",
    "#cecc3dff",
    "#537170",
    "#b295ceff",
    "#7d7e7aff",
    "#9C4A3F",
]

lifetime_cost_components = {
    "storage_capex": {
        "label": "Storage Upgrade CapEx",
        "color": cb_palette[6],
    },
    "capex_addtl_capacity": {
        "label": "Supplemental Aeration CapEx",
        "color": cb_palette[5],
    },
    "energy": {
        "label": "Energy Charges",
        "color": cb_palette[0],
    },
    "demand": {
        "label": "Demand Charges",
        "color": cb_palette[3],
    },
    "o_and_m": {
        "label": "O&M",
        # Orange, not the teal at index 2 — that is the power-decrease fill in the fig2 day profiles
        "color": cb_palette[1],
    },
    "export": {
        "label": "Export",
        "color": cb_palette[4],
    },
    "h2": {"label": "H$_2$ Value", "color": cb_palette[7]},
}
