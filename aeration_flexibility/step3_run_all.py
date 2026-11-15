import os

# Limit linear algebra libraries to 1 thread per process for parallel uses of solver (set before numpy loads them)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import glob
import json

import numpy as np

from helpers.config_keys import (
    get_base_key,
    parse_config_key,
    parse_design_key,
    cheapest_design_key,
)
from helpers.plotting import generate_plots
from helpers.design_optimization_algorithm import run_configuration
from helpers.parameters import CONFIG_DIR, OUTPUT_DATA_DIR
from helpers.tariffs import register_tariff_variants

# The paper's configurations, run in order (fig4 reads fig2_fig3's optimal designs)
PAPER_CONFIGS = [
    "fig2_fig3_cas_data_cas_tariff",
    "fig4_sensitivity_cas_data_cas_tariff",
    "si_hpoas_data_hpoas_tariff",
    "si_hpoas_data_cas_tariff",
    "si_energy_recovery",
    "si_solar_cas_elec",
]


def frange(lo, hi, n):
    """Inclusive float range with n evenly-spaced points from lo to hi."""
    return [round(v, 4) for v in np.linspace(lo, hi, n)]


def expand_sweep_ranges(sweep_ranges):
    """Generate tariff keys, summer configs, and capex sensitivity from sweep range specs.

    Each range is [lo, hi, step]. Generates one-at-a-time parameter sweeps:
    - peak_window: tariff keys with d1.0, e1.0, varying w
    - charge_factor: alldemand and allenergy tariff keys
    - o2_multiplier: summer configs with smoothing=1.0
    - smoothing: summer configs with peaking-factor multiplier=1.5 on max/mean ratio
    - capex: capex sensitivity multipliers

    Also registers required tariff variants so they are generated/loaded automatically.
    """

    tariff_keys = []
    tariff_variants = []

    for w in frange(*sweep_ranges["peak_window"]):
        w_str = f"neg{abs(w):g}" if w < 0 else f"{w:g}"
        tariff_keys.append(f"0.0__cas_d1.0_e1.0_w{w_str}")
        if w != 0:
            tariff_variants.append(
                {
                    "shift_peak_hours_before": -w / 2,
                    "shift_peak_hours_after": w / 2,
                    "variant_name": f"d1.0_e1.0_w{w_str}",
                }
            )

    for f in frange(*sweep_ranges["charge_factor"]):
        if f != 1.0:
            for charge_type in ("demand", "energy"):
                tariff_keys.append(f"0.0__cas_all{charge_type}{f:g}")
                tariff_variants.append(
                    {
                        "percent_change_dict": {charge_type: f},
                        "variant_name": f"all{charge_type}{f:g}",
                    }
                )

    summer_configs = [
        {"multiplier": m, "smoothing": 1.0} for m in frange(*sweep_ranges["o2_multiplier"])
    ]
    summer_configs += [
        {"multiplier": 1.5, "smoothing": s} for s in frange(*sweep_ranges["smoothing"]) if s != 1.0
    ]

    capex_sensitivity = frange(*sweep_ranges["capex"]) if "capex" in sweep_ranges else None

    if tariff_variants:
        register_tariff_variants(tariff_variants)

    return tariff_keys, summer_configs, capex_sensitivity


def get_unique_configurations(config_values):
    """Generate unique configurations from run_configs by expanding all combinations."""

    run_configs = {}
    summer_config_template = config_values["summer_config_template"]
    upgrade_keys = config_values["upgrade_keys"]
    tariff_keys_to_run = config_values["tariff_keys_to_run"]

    # Handle sensitivity configurations (fixed design_point or per-config design_points from source_run)
    if "design_point" in config_values or "design_points" in config_values:
        # Baseline tariff is this config's w0 tariff
        baseline_tariff = next(
            (t for t in tariff_keys_to_run if t.endswith("_w0")), tariff_keys_to_run[0]
        )
        config_values["baseline_tariff"] = baseline_tariff

        # Loop over each base_wwtp_key and its upgrade_keys
        for base_wwtp_key, upgrade_key_list in upgrade_keys.items():
            # Create separate run configs for each summer config
            for summer_config in summer_config_template:
                combined_base_wwtp_key = f"{base_wwtp_key}__{config_values['scale_mgd']}__{summer_config['multiplier']}__{summer_config['smoothing']}"
                run_configs[combined_base_wwtp_key] = {
                    "upgrade_keys": upgrade_key_list,
                    "summer_configs": [summer_config],
                    "tariff_configs": [baseline_tariff],
                }

            # Add one more config for all tariff variations with the baseline summer config
            baseline_summer = next(
                (
                    sc
                    for sc in summer_config_template
                    if sc["multiplier"] == 1.5 and sc["smoothing"] == 1.0
                ),
                summer_config_template[0],
            )

            combined_base_wwtp_key = f"{base_wwtp_key}__{config_values['scale_mgd']}__{baseline_summer['multiplier']}__{baseline_summer['smoothing']}_tariff_variations"
            run_configs[combined_base_wwtp_key] = {
                "upgrade_keys": upgrade_key_list,
                "summer_configs": [baseline_summer],
                "tariff_configs": tariff_keys_to_run,
            }
    else:
        # Handle WWTP COMPARISON configurations
        for wwtp_type in ["air__compressor", "o2__psa", "o2__cryo"]:
            for config_item in summer_config_template:
                upgrades = upgrade_keys[wwtp_type]
                if upgrades:  # Only add if there are upgrades
                    combined_base_wwtp_key = f"{wwtp_type}__{config_values['scale_mgd']}__{config_item['multiplier']}__{config_item['smoothing']}"
                    run_configs[combined_base_wwtp_key] = {
                        "upgrade_keys": upgrades,
                        "summer_configs": [config_item],
                        "tariff_configs": tariff_keys_to_run,
                    }
                else:
                    print(f"Skipping {wwtp_type} because no upgrades defined")

    unique_configs = []

    for base_wwtp_key, run_config in run_configs.items():

        # Extract base WWTP type (first two parts of key)
        wwtp_type = get_base_key(base_wwtp_key)

        # Generate all combinations
        for upgrade_key in run_config["upgrade_keys"]:
            for tariff_key in run_config["tariff_configs"]:
                for summer_config in run_config["summer_configs"]:
                    # Build combined base key with summer parameters
                    combined_base_key = f"{wwtp_type}__{config_values['scale_mgd']}__{summer_config['multiplier']}__{summer_config['smoothing']}"
                    unique_configs.append(
                        {
                            "base_wwtp_key": combined_base_key,
                            "upgrade_key": upgrade_key,
                            "tariff_key": tariff_key,
                            "summer_config": summer_config,
                        }
                    )

    # Deduplicate identical (base, upgrade, tariff, summer) configs
    deduped = {}
    for c in unique_configs:
        # base_wwtp_key already carries the summer multiplier and smoothing
        deduped.setdefault((c["base_wwtp_key"], c["upgrade_key"], c["tariff_key"]), c)

    return run_configs, list(deduped.values())


def run_all_configs(configs=PAPER_CONFIGS):
    """Solve and plot each config: a name in configs/, or a config dict carrying its own base_run_name."""
    for config_item in configs:
        if isinstance(config_item, dict):
            config = config_item
            base_run_name = config["base_run_name"]
        else:
            config_path = os.path.join(CONFIG_DIR, f"{config_item}.json")
            with open(config_path, "r") as f:
                print(f"Loading config from {config_path}")
                config = json.load(f)
            base_run_name = config_item

        # Expand sweep_ranges into tariff_keys, summer_configs, capex_sensitivity if present
        if "sweep_ranges" in config:
            print("adding sweep ranges")
            tariff_keys, summer_configs, capex_sensitivity = expand_sweep_ranges(
                config["sweep_ranges"]
            )
        else:
            tariff_keys = config["tariff_keys_to_run"]
            summer_configs = config["summer_config_template"]
            capex_sensitivity = config.get("capex_sensitivity")

        config_values = {
            # Derived from filename
            "base_run_name": base_run_name,
            # Direct mappings from JSON
            "summer_config_template": summer_configs,
            "upgrade_keys": config["upgrade_keys"],
            "tariff_keys_to_run": tariff_keys,
            # Hardcoded
            "lc_convergence_threshold": 1e-3,
            # Flattened nested sections
            **config["facility"],
            **config["script_settings"],
            **config.get("design_space", {}),
        }

        # Defaults for design_space keys
        config_values.setdefault("include_upgrade_capex", True)
        # Opt-in: write only the profile columns / metadata fields read back downstream
        config_values.setdefault("save_necessary_only", False)

        # Optional sensitivity analysis keys: the optimal design point per (facility, upgrade) from a
        # previous run's results, keeping the best across tariff variations
        if "source_run" in config:
            print(f"Loading optimal design points from source run: {config['source_run']}")
            target_upgrades = {
                uk for upgrade_list in config["upgrade_keys"].values() for uk in upgrade_list
            }
            best_per_key = {}  # "facility_base_key___upgrade_key" -> (design_point, lifetime cost)
            source_dir = os.path.join(OUTPUT_DATA_DIR, config["source_run"])
            for path in glob.glob(os.path.join(source_dir, "*___*___*.json")):
                base_wwtp_key, upgrade_key, _ = parse_config_key(
                    os.path.splitext(os.path.basename(path))[0]
                )
                if upgrade_key not in target_upgrades:
                    continue
                with open(path, "r") as f:
                    all_results = json.load(f)["all_results"]
                best_design_key = cheapest_design_key(all_results)
                if best_design_key is None:
                    continue
                best_cost = all_results[best_design_key]["lifetime_cost"]["total"]
                lookup_key = f"{get_base_key(base_wwtp_key)}___{upgrade_key}"
                if lookup_key not in best_per_key or best_cost < best_per_key[lookup_key][1]:
                    best_per_key[lookup_key] = (
                        tuple(float(p) for p in parse_design_key(best_design_key)),
                        best_cost,
                    )
                    print(
                        f"  Source design for {lookup_key}: {best_design_key} (lifetime_cost=${best_cost:,.0f})"
                    )
            config_values["design_points"] = {k: v[0] for k, v in best_per_key.items()}
            config_values["capex_sensitivity"] = capex_sensitivity

        run_configs, unique_configs = get_unique_configurations(config_values)

        # Optimization
        if config_values["run_solve_optimization"]:
            print(f"Running {len(unique_configs)} configurations")
            run_configuration(unique_configs, config_values)
            print("Optimization complete")

        # Plotting
        if config_values["run_plotting"]:
            generate_plots(run_configs, config_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", default=PAPER_CONFIGS)
    run_all_configs(parser.parse_args().configs)
