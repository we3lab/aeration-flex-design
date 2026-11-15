"""Design search for each config (run_configuration): sample storage designs, solve their operation month by month
with a per-day cache, and aggregate lifetime cost against the expansion-only counterfactual."""

import numpy as np
import pandas as pd
import os
import gc
import glob
import re
import time
import signal
import traceback
import psutil
import json
from multiprocessing import Pool

from eeco import costs, metrics

from helpers.operational_problem import OperationalProblem, build_new_param_vals
from helpers.compressor_power import comp_lp_coeff
from helpers.capex_lifetime_cost_energy_metrics import calculate_capex
from helpers.config_keys import *
from helpers.tariffs import (
    get_tariff_data,
    get_charge_dict_for_month,
    get_month_dates,
    CALENDAR_MONTHS,
    to_usd_2025,
)
from helpers.parameters import *
from helpers.design_space import (
    get_effective_ranges,
    get_initial_design_points,
    round_point,
    sample_from_multivariate_normal,
    infeasibility_boundary_frac,
    collect_infeasibility_data,
)

# Profile keys present for all problems (var_dict base + recalc)
PROFILE_KEYS_BASE = ["Edot_t_baseline", "Edot_t_net_recalc"]
# Superset of all profile keys needed downstream (for slimming loaded/solved profiles)
PROFILE_KEYS_ALL = frozenset(PROFILE_KEYS_BASE + ["E", "N", "Ndot_c"])
ZERO_MAX_VALUES = {
    "Ndot_c": 0,
    "Ndot_b_excess": 0,
    "Edot_c": 0,
    "addtl_capacity": 0,
    "Edot_addtl_capacity": 0,
    "unmet_o2": 0,
}
SCALAR_BASELINE_KEYS = {"P_init_base", "ei_o2_base", "ei_evap", "P_b", "Edot_b_comp_coeff"}
GRID_REFINEMENT_UPGRADE_KEYS = ("none__liquid_tank", "none__gas_tank", "compressor__gas_tank")
OPEX_KEYS = ("energy", "demand", "export", "h2")
# Any day with more unmet O2 than this marks the design point infeasible at its frac
UNMET_O2_THRESHOLD = 0.5
# Grid points per solve+aggregate batch: each batch's full-year results are freed once merged into
# state["all_results"], so peak memory stays bounded regardless of how many points the grid has.
GRID_BATCH_SIZE = 50
PAYBACK_DISCOUNT_RATE = 0.03
PAYBACK_MAX_YEARS = 100

# Pool-worker globals (set once via initializer, avoids per-task pickling)
worker_tariff_data = None
worker_config_values = None

# Memoizes baseline-tariff SoC seeds (see the sensitivity branch in process_single_month).
BASELINE_SEED_CACHE = {}


def round_cost_values(d, decimals=2):
    """Recursively round all float values in a dict to the given number of decimals.

    Used to stabilize cost outputs across runs so that floating-point jitter
    doesn't prevent hypercube sampling from recognising the same design points.
    """
    if isinstance(d, dict):
        return {k: round_cost_values(v, decimals) for k, v in d.items()}
    if isinstance(d, float) and np.isfinite(d):
        return round(d, decimals)
    return d


def to_json_safe(obj):
    """Recursively convert numpy values, arrays and timestamps to JSON types, with inf/nan as None.

    json.dump's default= hook only fires for unserializable types; Python float is natively
    serializable and json encodes inf as the non-standard token 'Infinity'. This pre-pass catches
    those values before they reach the encoder.
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if isinstance(obj, (float, np.integer, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "isoformat"):  # pd.Timestamp, datetime.date, datetime.datetime
        return obj.isoformat()
    return obj


class SingleLineListEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        s = "".join(super().iterencode(to_json_safe(o), _one_shot))
        # Compress flat numeric lists to single line
        s = re.sub(
            r"\[\s*([\d.eE+\-,\s]+)\s*\]",
            lambda match: "["
            + ", ".join(item.strip() for item in match.group(1).split(",") if item.strip())
            + "]",
            s,
        )
        return [s]


def write_json_file(path, obj):
    """Atomically write obj as JSON (temp file + rename) so a crash mid-write can't corrupt a
    file that resume/skip logic later depends on."""
    temp_path = path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(obj, f, indent=2, cls=SingleLineListEncoder)
        f.flush()
        os.fsync(f.fileno())
    os.rename(temp_path, path)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def strip_month_dates(charge_key):
    # Charge keys are date-stamped (e.g., ..._20220801_20220831_...) so they differ per month;
    # stripping the dates matches semantically equivalent keys across months.
    return "_".join(charge_key.split("_")[:-3]) + "_" + charge_key.split("_")[-1]


def pool_initializer(tariff_data, config_values):
    """Called once per worker process to set shared read-only data."""
    # Workers ignore SIGINT so Ctrl+C is handled ONLY by the main process, which then terminates the
    # pool (otherwise each worker raises KeyboardInterrupt mid-solve and the pool doesn't shut down cleanly).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global worker_tariff_data, worker_config_values
    worker_tariff_data = tariff_data
    worker_config_values = config_values


def process_design_point_all_months(design_point_data, horizon_days, shared_state):
    """Process a single design point across ALL months sequentially.

    Months are solved in order so storage state carries forward between months.
    The last window of each month overflows into the next month's first days.
    Returns the list of per-month results.
    """
    design_key, ordered_month_keys, new_param_vals = design_point_data
    month_to_days = shared_state["month_to_days"]

    month_results = []
    carry_forward_storage_state = None
    for month_index, month_key in enumerate(ordered_month_keys):
        # Determine overflow days for last window of this month
        if month_index + 1 < len(ordered_month_keys):
            next_month_days = month_to_days[ordered_month_keys[month_index + 1]][:horizon_days]
        else:
            next_month_days = sorted(shared_state["overflow_dates"])[:horizon_days]

        result = process_single_month(
            design_key=design_key,
            month_key=month_key,
            month_days=month_to_days[month_key],
            next_month_days=next_month_days,
            new_param_vals=new_param_vals,
            carry_forward_storage_state=carry_forward_storage_state,
            horizon_days=horizon_days,
            shared_state=shared_state,
        )
        carry_forward_storage_state = result["end_of_month_storage_state"]
        month_results.append(result)
        # Free the month's CVXPY problem (reference cycles) before building the next one
        gc.collect()
    return month_results


def process_single_month(
    design_key,
    month_key,
    month_days,
    next_month_days,
    new_param_vals,
    carry_forward_storage_state,
    horizon_days,
    shared_state,
):
    """Process a single month for one design point, with overflow into next month."""
    tariff_data = worker_tariff_data
    config_values = worker_config_values

    base_wwtp_key = shared_state["base_wwtp_key"]
    upgrade_key = shared_state["upgrade_key"]
    tariff_key = shared_state["tariff_key"]
    baseline_vals = shared_state["baseline_vals"]

    decomposition_type = config_values.get("decomposition_type")
    solver_settings = config_values.get("solver_settings", {})
    save_necessary_only = config_values["save_necessary_only"]
    # Intermediate CSV columns (None = all): fig2-plotted configs keep everything for fig2 row-ii (non-best
    # designs are slimmed after grid refinement); every other config keeps only the columns read back.
    profile_keys_to_save = None
    if save_necessary_only and not any(k in upgrade_key for k in GRID_REFINEMENT_UPGRADE_KEYS):
        profile_keys_to_save = PROFILE_KEYS_ALL

    # Process the month using N-day optimization with 1-day overlap
    month_start_time = time.time()
    initial_memory = get_memory_usage()
    year, month = month_key.split("-")

    config_name = get_config_name(base_wwtp_key, upgrade_key, tariff_key)
    print(f"processing {design_key} in {month}, {year} for {config_name} ({initial_memory:.1f} MB)")

    charge_dict_month = get_charge_dict_for_month(tariff_data[tariff_key], int(year), int(month))
    # Charge dict for next month (needed for overflow windows), keyed back to this month's charge keys
    next_year, next_month, _ = next_month_days[0].split("-")
    charge_dict_next_month = get_charge_dict_for_month(
        tariff_data[tariff_key], int(next_year), int(next_month)
    )
    next_month_keys = {strip_month_dates(k): k for k in charge_dict_next_month}
    charge_key_mapping = {
        k: next_month_keys[strip_month_dates(k)]
        for k in charge_dict_month
        if strip_month_dates(k) in next_month_keys
    }

    # Local demand dict to avoid UnboundLocalError and preserve per-horizon state
    local_prev_demand = {
        k: {"demand": 0.0, "cost": 0.0} for k in charge_dict_month if "demand" in str(k)
    }

    # Intermediates go into h-specific subfolder
    intermediate_dir = get_intermediate_dir(config_values["data_name"], config_name, horizon_days)
    os.makedirs(intermediate_dir, exist_ok=True)
    month_results_local = {}
    window_rmse_vals = []
    i_local = 0
    cached_cvxpy_problem = None
    end_of_month_storage_state = None

    # All available days: current month + overflow from next month
    all_available_days = list(month_days) + list(next_month_days)
    month_days_set = set(month_days)

    while i_local < len(month_days):
        # Window from all available days (can extend past month boundary)
        group_days = all_available_days[i_local : i_local + horizon_days]
        # Save every month day except overflow days and the overlap day, which the next window re-solves
        is_last_window = i_local + horizon_days >= len(month_days)
        days_to_save = [
            day
            for day_index, day in enumerate(group_days)
            if day in month_days_set and (is_last_window or day_index != horizon_days - 1)
        ]

        all_intermediate_available = all(
            os.path.exists(os.path.join(intermediate_dir, f"{design_key}_{day}_metadata.json"))
            for day in days_to_save
        )
        if all_intermediate_available:  # already solved all days, just need to aggregate them
            for day in days_to_save:
                # Load from CSV + JSON files in h-specific subfolder
                with open(
                    os.path.join(intermediate_dir, f"{design_key}_{day}_metadata.json"), "r"
                ) as f:
                    max_values = json.load(f)["max_values"]

                # Check if profile CSV exists (successful solve) or not (dummy file)
                profile_path = os.path.join(intermediate_dir, f"{design_key}_{day}_profile.csv")
                if os.path.exists(profile_path):
                    # Successful solve - load profile, slim to needed keys only
                    profile_df = pd.read_csv(profile_path)
                    profile = {
                        col: profile_df[col].values.tolist()
                        for col in profile_df.columns
                        if col in PROFILE_KEYS_ALL
                    }
                    month_results_local[day] = (profile, max_values)

                    # Accumulate this day's demand into the running monthly max ({demand, cost} per
                    # charge, from the day's slice of the month charge arrays).
                    day_of_month = int(day.split("-")[2]) - 1
                    day_charges = {
                        k: charge_dict_month[k][day_of_month * 96 : (day_of_month + 1) * 96]
                        for k in charge_dict_month
                        if "demand" in str(k)
                    }
                    local_prev_demand = costs.get_prev_demand_dict(
                        day_charges,
                        np.asarray(profile["Edot_t_net_recalc"], dtype=float),
                        np.datetime64(day),
                        [],
                        local_prev_demand,
                    )
                else:  # Dummy file - failed solve, no profile CSV
                    month_results_local[day] = (None, max_values)
                    print(f"  {design_key} Skipping failed {day} (dummy file found)")

            # Advance to next group after loading cached results
            i_local += horizon_days - 1
            continue

        # Not all intermediates available: solve the N-day group
        is_sensitivity = "design_point" in config_values or "design_points" in config_values
        if is_sensitivity and tariff_key != config_values["baseline_tariff"]:
            # Tariff-perturbed sensitivity cell: seed from the baseline tariff's SoC, not carry-forward, since
            # carryover is tariff-dependent (a low tank entering a peak day can be spuriously unmet under one tariff).
            seed_day = group_days[0]
            # Strip the ~gas/~liq medium suffix: the baseline profile is keyed by the bare design (its winning medium).
            seed_design_key = design_key.split("~")[0]
            baseline_config_name = get_config_name(
                base_wwtp_key, upgrade_key, config_values["baseline_tariff"]
            )
            seed_cache_key = (baseline_config_name, seed_design_key, seed_day)
            if seed_cache_key not in BASELINE_SEED_CACHE:
                seed_path = os.path.join(
                    get_intermediate_dir(
                        config_values["data_name"], baseline_config_name, horizon_days
                    ),
                    f"{seed_design_key}_{seed_day}_profile.csv",
                )
                seed = None
                if os.path.exists(seed_path):
                    seed_df = pd.read_csv(seed_path)
                    seed = {
                        k: float(seed_df[k].iloc[0]) for k in ("N", "E") if k in seed_df.columns
                    }
                BASELINE_SEED_CACHE[seed_cache_key] = seed or None
            initial_storage_state = BASELINE_SEED_CACHE[seed_cache_key]
            if initial_storage_state is None:  # haven't solved baseline yet
                raise RuntimeError(
                    f"Missing baseline SoC seed for sensitivity cell {config_name} "
                    f"(design {design_key}, day {seed_day}): baseline solve "
                    f"{baseline_config_name} must complete first."
                )
        elif i_local == 0:
            # First window: use carry-forward from previous month (None for first month)
            initial_storage_state = carry_forward_storage_state
        else:
            # Later windows start from the final storage state of the previous window's last saved day
            prev_profile = month_results_local[month_days[i_local - 1]][0]
            initial_storage_state = (
                {k: np.asarray(prev_profile[k])[-1] for k in ("N", "E") if k in prev_profile}
                if prev_profile
                else None
            )

        # Process the N-day group
        first_day_vals = baseline_vals[group_days[0]]
        concatenated_baseline_vals = {
            k: (
                pd.Series(np.concatenate([baseline_vals[d][k] for d in group_days]))
                if isinstance(v, np.ndarray)
                else v
            )  # For non-indexed parameters, use the first day's value
            for k, v in first_day_vals.items()
        }

        # Concatenate charge dictionary (may span two months for overflow windows)
        concatenated_charge_dict = {}
        for charge_key, month_charges in charge_dict_month.items():
            if len(month_charges) <= 1:
                # Scalar charge (e.g., customer) — not per-timestep, pass through
                concatenated_charge_dict[charge_key] = month_charges
                continue
            day_slices = []
            for day in group_days:
                day_of_month = int(day.split("-")[2])
                if day in month_days_set:
                    day_slices.append(month_charges[(day_of_month - 1) * 96 : day_of_month * 96])
                else:
                    next_charges = charge_dict_next_month[
                        charge_key_mapping.get(charge_key, charge_key)
                    ]
                    day_slices.append(next_charges[(day_of_month - 1) * 96 : day_of_month * 96])
            concatenated_charge_dict[charge_key] = np.concatenate(day_slices)

        # Create problem instance for N days
        var_length = len(concatenated_baseline_vals["Ndot_target"])
        first_day = pd.to_datetime(group_days[0])
        dates = f"{group_days[0]}_to_{group_days[-1]}"
        if cached_cvxpy_problem is not None and var_length == cached_cvxpy_problem.var_length:
            problem = cached_cvxpy_problem
            problem.date = dates
        else:
            problem = OperationalProblem(
                prob_name=config_name,
                design_key=design_key,
                horizon_days=horizon_days,
                m=first_day.month,
                y=first_day.year,
                date=dates,
                var_length=var_length,
                decomposition_type=decomposition_type,
                solver_settings=solver_settings,
            )
            # Only cache full-sized windows; don't replace with a short end-of-run window
            if cached_cvxpy_problem is None:
                cached_cvxpy_problem = problem

        # Construct and solve optimization problem with N-day data
        problem.construct_problem(
            charge_dict=concatenated_charge_dict,
            baseline_vals=concatenated_baseline_vals,
            new_param_vals=new_param_vals,
            prev_demand_dict=local_prev_demand,
            initial_storage_state=initial_storage_state,
        )

        solve_start = time.time()
        profile, solve_info = problem.solve_O2Problem()
        rmse, nrmse = 0.0, 0.0
        if profile:
            diff = profile["Edot_t_net_recalc"] - profile["Edot_t_net"]
            rmse = round(float(np.sqrt(np.mean(diff**2))), 3)
            nrmse = round(100.0 * rmse / float(np.mean(np.abs(profile["Edot_t_net_recalc"]))), 3)
            if rmse > 0:
                window_rmse_vals.append(rmse)
        solve_time = time.time() - solve_start
        objective = solve_info.get("objective")
        # infeasible/unsolved returns objective None; don't crash the whole job on the log line
        objective_str = f"{objective:10.0f}" if objective is not None else f"{'infeas':>10s}"
        print(
            f" {upgrade_key:<20s} {design_key:<20s} {dates:<20s} h{horizon_days:>2d} {solve_time:>7.0f}s {objective_str:>10s}"
            f" rmse={rmse:>7.3f} ({nrmse:>6.3f}%)"
        )

        profiles_dict = {}
        max_values_dict = {}
        if profile:
            # In the month's last window, keep the storage state at the last timestep that belongs to this month
            if is_last_window:
                t = sum(1 for d in group_days if d in month_days_set) * 96 - 1
                end_of_month_storage_state = {
                    k: float(profile[k][t]) for k in ("N", "E") if k in profile
                }

            for day_index, day in enumerate(group_days):
                # Extract day-specific profile
                start_index = day_index * 96
                day_profile = {
                    k: pd.Series(v[start_index : start_index + 96]) for k, v in profile.items()
                }
                profiles_dict[day] = day_profile

                # Calculate max values for this day (addtl_capacity, Edot_addtl_capacity from O2 problem)
                max_values_dict[day] = {
                    k: np.max(day_profile[k]) if k in day_profile else 0 for k in ZERO_MAX_VALUES
                }

                # Accumulate this day's demand into the running monthly max (start_index is this
                # day's offset into the concatenated horizon charge arrays).
                day_charges = {
                    k: concatenated_charge_dict[k][start_index : start_index + 96]
                    for k in concatenated_charge_dict
                    if "demand" in str(k)
                }
                local_prev_demand = costs.get_prev_demand_dict(
                    day_charges,
                    np.asarray(day_profile["Edot_t_net_recalc"], dtype=float),
                    np.datetime64(day),
                    [],
                    local_prev_demand,
                )
        else:
            print(f"Solver failed - no profile returned for {design_key} {upgrade_key}")
            print("Saving dummy files to prevent re-solving")
            for day in group_days:
                profiles_dict[day] = None
                max_values_dict[day] = dict(ZERO_MAX_VALUES)

        # Save results / dummy files for each saved day; a failed window has profiles_dict[day] None,
        # so the same loop writes only the metadata "day solved" marker.
        for day in days_to_save:
            # max_values is the only metadata field ever read back (feasibility unmet_o2 + capex on
            # resume); when slimming, drop the write-only fields but keep the file as the marker.
            if save_necessary_only:
                metadata = {"max_values": max_values_dict[day]}
            else:
                metadata = {
                    "new_param_vals": new_param_vals,
                    "max_values": max_values_dict[day],
                    "prev_demand_dict": local_prev_demand,
                    "horizon_days": horizon_days,
                    "solve_info": solve_info,
                }
            day_profile = profiles_dict[day]
            if day_profile is not None:
                if profile_keys_to_save is not None:
                    day_profile = {
                        k: v for k, v in day_profile.items() if k in profile_keys_to_save
                    }
                profile_path = os.path.join(intermediate_dir, f"{design_key}_{day}_profile.csv")
                pd.DataFrame(day_profile).to_csv(profile_path + ".tmp", index=False)
                os.rename(profile_path + ".tmp", profile_path)
            # Marker last, so an interrupted write re-solves the day instead of caching it as failed
            write_json_file(
                os.path.join(intermediate_dir, f"{design_key}_{day}_metadata.json"), metadata
            )
            month_results_local[day] = (
                (
                    {k: v for k, v in profiles_dict[day].items() if k in PROFILE_KEYS_ALL}
                    if profiles_dict[day]
                    else None
                ),
                max_values_dict[day],
            )
            profiles_dict[day] = None  # free full profile

        i_local += horizon_days - 1  # include 1 day overlap for next group

    month_approx_error = (
        {
            "rmse_min": round(min(window_rmse_vals), 3),
            "rmse_max": round(max(window_rmse_vals), 3),
            "rmse_mean": round(sum(window_rmse_vals) / len(window_rmse_vals), 3),
        }
        if window_rmse_vals
        else {}
    )
    total_time = time.time() - month_start_time
    memory_used = get_memory_usage() - initial_memory

    # Check if any day had unmet_o2 > threshold (design point infeasible at this frac)
    max_unmet = max(max_values["unmet_o2"] for _, max_values in month_results_local.values())
    has_unmet_o2 = max_unmet > UNMET_O2_THRESHOLD
    if has_unmet_o2:
        print(f"{max_unmet} unmet_o2 detected for {design_key} in {month_key} for {base_wwtp_key}")
    print(f"{month_key} for {design_key} ({memory_used:.1f}MB)")

    return {
        "design_key": design_key,
        "month_key": month_key,
        "month_results": month_results_local,
        "new_param_vals": new_param_vals,
        "month_solve_time": total_time,
        "month_approx_error": month_approx_error,
        "has_unmet_o2": has_unmet_o2,
        "end_of_month_storage_state": end_of_month_storage_state,
    }


def get_annual_metrics(
    annual_profiles,
    tariff_key,
    upgrade_key,
    new_param_vals,
    tariff_data,
    num_months,
    start_date,
    end_date,
    decomposition_type=None,
):
    """Calculate annual metrics from concatenated profiles."""

    Edot_t_net_recalc = np.array(annual_profiles["Edot_t_net_recalc"])
    Edot_t_baseline = np.array(annual_profiles["Edot_t_baseline"])

    # Forecast error (Bolorinos et al. 2023, eqs S16-S18): the BILLED net and baseline demand both carry the same
    # N(0, forecast_error_std * |net demand|) noise; physics metrics stay clean. Off by default (forecast_error_std = 0).
    net_billed, base_billed = Edot_t_net_recalc, Edot_t_baseline
    if forecast_error_std > 0:
        eps = np.random.RandomState(RANDOM_SEED).normal(
            0.0, forecast_error_std * np.abs(Edot_t_baseline)
        )
        net_billed = Edot_t_net_recalc + eps
        base_billed = Edot_t_baseline + eps

    # Calculate annual costs
    charge_dict = costs.get_charge_dict(
        start_date, end_date, tariff_data[tariff_key], resolution="15m"
    )
    kwargs = {
        "desired_utility": "electric",
        "resolution": "15m",
        "decomposition_type": decomposition_type,
    }
    new_itemized_costs, _ = costs.calculate_itemized_cost(
        charge_dict, {"electric": net_billed}, **kwargs
    )
    baseline_itemized_costs, _ = costs.calculate_itemized_cost(
        charge_dict, {"electric": base_billed}, **kwargs
    )

    # Demand charges: recompute month-by-month
    timestamps = start_date + np.arange(len(Edot_t_net_recalc)) * np.timedelta64(15, "m")
    new_demand_total = baseline_demand_total = 0.0
    for month_start in pd.date_range(pd.Timestamp(start_date), periods=num_months, freq="MS"):
        month_dates = get_month_dates(month_start.year, month_start.month)
        mask = (timestamps >= month_dates["start"]) & (timestamps < month_dates["end"])
        month_charge_dict = get_charge_dict_for_month(
            tariff_data[tariff_key], month_start.year, month_start.month
        )
        new_month_costs, _ = costs.calculate_itemized_cost(
            month_charge_dict, {"electric": net_billed[mask]}, **kwargs
        )
        baseline_month_costs, _ = costs.calculate_itemized_cost(
            month_charge_dict, {"electric": base_billed[mask]}, **kwargs
        )
        new_demand_total += new_month_costs["electric"]["demand"]
        baseline_demand_total += baseline_month_costs["electric"]["demand"]
    new_itemized_costs["electric"]["demand"] = new_demand_total
    baseline_itemized_costs["electric"]["demand"] = baseline_demand_total

    # Extract individual opex components, add H2 if applicable, and scale to 1-year
    annualize = 12 / num_months
    new_electric = new_itemized_costs["electric"]
    baseline_electric = baseline_itemized_costs["electric"]
    cost_keys = ("energy", "demand", "export", "customer")
    h2_value = (
        moles_to_mass(
            np.sum(annual_profiles.get("Ndot_c", 0)) / timestep_factor * new_param_vals["y_h2"],
            M_H2,
        )
        * price_h2_kg
    )
    opex = {
        "new": {**{k: new_electric[k] * annualize for k in cost_keys}, "h2": -h2_value * annualize},
        "baseline": {**{k: baseline_electric[k] * annualize for k in cost_keys}, "h2": 0},
        "savings": {
            **{k: (baseline_electric[k] - new_electric[k]) * annualize for k in cost_keys},
            "h2": h2_value * annualize,
        },
    }

    # Capacity factor: use storage state if present; otherwise 0 for expansion-only.
    storage_key = "E" if "battery" in upgrade_key else "N"
    capacity_factor = (
        float(np.max(annual_profiles[storage_key]) / float(new_param_vals[f"{storage_key}_max"]))
        if storage_key in annual_profiles
        else 0.0
    )

    return {
        "capacity_factor": round(capacity_factor, 3),
        "rte": round(metrics.roundtrip_efficiency(Edot_t_baseline, Edot_t_net_recalc), 3),
        "mean_energy_capacity": round(
            metrics.energy_capacity(
                Edot_t_baseline,
                Edot_t_net_recalc,
                timestep=0.25,
                ec_type="discharging",
                relative=True,
            ),
            3,
        ),
        "mean_power_capacity": round(
            metrics.power_capacity(
                Edot_t_baseline,
                Edot_t_net_recalc,
                timestep=0.25,
                pc_type="average",
                relative=True,
            ),
            3,
        ),
        "opex": opex,
    }


def present_value(annual_amount, years=10):
    """Present value of a per-year saving over `years`, sign-flipped to a cost delta (-NPV)."""
    # eeco discounts range(1, upgrade_lifetime) = years 1..upgrade_lifetime-1, so pass years+1 to
    # discount all `years` cash-flow years (1..years).
    return -metrics.net_present_value(
        capital_cost=0,
        electricity_savings=annual_amount,
        maintenance_diff=0,
        timestep=0.25,
        upgrade_lifetime=years + 1,
    )


def get_itemized_lifetime_cost(
    annual_metrics,
    capex,
    capex_components,
    expansion_only_capex=0,
    years=10,
    battery_kw=None,
    battery_kwh=None,
    is_battery=False,
):
    om_frac = battery_om_frac_per_year if is_battery else storage_om_frac_per_year
    opex = annual_metrics["opex"]
    savings = opex["savings"]

    by_component = {k: present_value(savings[k], years) for k in OPEX_KEYS}
    capex_addtl = capex_components["capex_addtl_capacity"]
    storage_capex = capex - capex_addtl

    # Recurring storage O&M: a per-year cost (not a saving), incremental to storage only — the
    # aeration-expansion O&M cancels vs the counterfactual.
    by_component["o_and_m"] = present_value(-om_frac * storage_capex, years)
    total = capex + sum(by_component.values())
    annual_savings = sum(savings[k] for k in OPEX_KEYS)
    opex_pv = sum(by_component[k] for k in OPEX_KEYS)  # negative (a saving)

    # ROI/payback for our storage capex and, battery only, Bolorinos et al. 2023's ($267k + $425/kW + $217/kWh =
    # ATB 2021 Moderate, year-2021 column, 2019 USD) — isolates how much of our ROI comes from the battery cost basis.
    # Both ATB editions price usable kWh, so battery_kwh (nameplate E_max) is scaled the same way as cost_battery.
    storage_capex_cases = {"ours": storage_capex}
    if battery_kw is not None and battery_kwh is not None:
        storage_capex_cases["bolorinos"] = to_usd_2025(
            267000.0 + 425.0 * battery_kw + 217.0 * battery_usable_frac * battery_kwh, 2019
        )
    returns = {}
    for case, case_storage_capex in storage_capex_cases.items():
        case_capex = case_storage_capex + capex_addtl
        case_total = case_capex + opex_pv + present_value(-om_frac * case_storage_capex, years)
        lifetime_roi = (
            100.0 * (expansion_only_capex - case_total) / case_storage_capex
            if case_storage_capex > 0
            else None
        )
        # Discounted payback of the incremental capex vs the must-expand counterfactual, from net bill
        # savings: 0.0 if no costlier than the counterfactual, None if it never pays back.
        payback_capex = case_capex - expansion_only_capex
        payback_cash_flow = annual_savings - om_frac * case_storage_capex
        payback_years = None
        if payback_capex <= 0:
            payback_years = 0.0
        elif payback_cash_flow > 0:
            cumulative = 0.0
            for year in range(1, PAYBACK_MAX_YEARS + 1):
                cumulative += payback_cash_flow / ((1 + PAYBACK_DISCOUNT_RATE) ** year)
                if cumulative >= payback_capex:
                    payback_years = float(year)
                    break
        returns[case] = {
            "total": case_total,
            "lifetime_roi_pct": lifetime_roi,
            "annual_roi_pct": lifetime_roi / years if lifetime_roi is not None else None,
            "payback_years": payback_years,
        }

    return {
        "from capex": capex,
        "total": total,
        "by_component": by_component,
        "capex_addtl_capacity": capex_addtl,
        "storage_capex": storage_capex,
        # top-level too so the flat/"new"-dispatch stacked views pick it up (it lives in by_component
        # above for the primary view); the counterfactual zeroes it explicitly.
        "o_and_m": by_component["o_and_m"],
        "annual_savings": annual_savings,
        "by_component_new": {k: present_value(-opex["new"][k], years) for k in OPEX_KEYS},
        "by_component_baseline": {k: present_value(-opex["baseline"][k], years) for k in OPEX_KEYS},
        "expansion_only": expansion_only_capex,
        "annual_roi_pct": returns["ours"]["annual_roi_pct"],
        "lifetime_roi_pct": returns["ours"]["lifetime_roi_pct"],
        "payback_years": returns["ours"]["payback_years"],
        "bolorinos_comparison": (
            {"storage_capex": storage_capex_cases["bolorinos"], **returns["bolorinos"]}
            if "bolorinos" in returns
            else None
        ),
    }


def aggregate_and_save_config_results(
    config_name,
    state,
    all_designs_month_results,
    unsolved_pts,
    tariff_data,
    config_values,
):
    """Aggregate results for a single config and save immediately."""
    upgrade_key = state["upgrade_key"]
    tariff_key = state["tariff_key"]
    storage_type = split_key(upgrade_key)[1]

    first_day_baseline = next(iter(state["baseline_vals"].values()))
    baseline_param_vals = {k: v for k, v in first_day_baseline.items() if k in SCALAR_BASELINE_KEYS}

    # Pre-group results by design_key to avoid O(n²) scan
    results_by_design = {}
    for r in all_designs_month_results:
        results_by_design.setdefault(r["design_key"], []).append(r)

    aggregation_start = time.time()
    n_unsolved = len(unsolved_pts)
    for design_index, design_key in enumerate(unsolved_pts):
        if (design_index + 1) % 20 == 0 or design_index == 0:
            print(
                f"  Aggregating {config_name}: {design_index + 1}/{n_unsolved} ({time.time() - aggregation_start:.1f}s)"
            )

        # Medium variants fanned out by build_job_queue_for_design_keys compete here, each with its own physics/capex;
        # the cheaper lifetime cost is stored under the bare design key (non-fanned configs have one bare-key variant).
        variant_keys = [
            k
            for k in (f"{design_key}~gas", f"{design_key}~liq", design_key)
            if k in results_by_design
        ]
        if not variant_keys:
            print(f" {design_key} INFEASIBLE (all months skipped) — lifetime cost set to inf")
            state["all_results"][design_key] = {"lifetime_cost": {"total": np.inf}}
            continue

        variant_metrics = {}  # job_key -> annual_metrics, or None if infeasible
        any_incomplete = False
        for job_key in variant_keys:
            design_month_results = results_by_design[job_key]

            # Aggregate per-month results into annual profiles and per-key maxima
            annual_profiles = {key: [] for key in state["keys_to_append"]}
            max_values = dict(ZERO_MAX_VALUES)
            for result in design_month_results:
                for profile, day_max_values in result["month_results"].values():
                    if profile:
                        for key in state["keys_to_append"]:
                            annual_profiles[key].extend(profile[key])
                        for key in max_values:
                            max_values[key] = max(max_values[key], day_max_values[key])
            month_approx_errors = [
                r["month_approx_error"] for r in design_month_results if r["month_approx_error"]
            ]
            processed_months = sorted(
                r["month_key"] for r in design_month_results if r["month_results"]
            )
            expected_months = len(state["month_to_days"])

            # Mark infeasible only if confirmed unmet_o2. For incomplete results (partial months), do NOT
            # save to all_results so the design point gets re-queued on next iteration/run.
            if any(r["has_unmet_o2"] for r in design_month_results):
                print(f" {job_key} INFEASIBLE (unmet_o2 > threshold) — lifetime cost set to inf")
                variant_metrics[job_key] = None
                continue
            if len(processed_months) < expected_months or not annual_profiles["Edot_t_net_recalc"]:
                reason = (
                    f"only {len(processed_months)}/{expected_months} months solved"
                    if processed_months
                    else "no valid month data"
                )
                print(f" {job_key} INCOMPLETE ({reason}) — skipping, will retry")
                any_incomplete = True
                break

            start_dt = pd.to_datetime(CALENDAR_MONTHS[min(processed_months)]["start"])
            end_dt = pd.to_datetime(CALENDAR_MONTHS[max(processed_months)]["end"])
            # Check profile length matches expected calendar days
            expected_timesteps = (end_dt - start_dt).days * 96
            actual_timesteps = len(annual_profiles["Edot_t_net_recalc"])
            if actual_timesteps != expected_timesteps:
                print(
                    f" {job_key} INFEASIBLE (profile length {actual_timesteps} != expected {expected_timesteps}) — lifetime cost set to inf"
                )
                variant_metrics[job_key] = None
                continue

            # Get the correct new_param_vals for this specific design point
            design_new_param_vals = design_month_results[0]["new_param_vals"]
            annual_metrics = get_annual_metrics(
                annual_profiles,
                tariff_key,
                upgrade_key,
                design_new_param_vals,
                tariff_data,
                len(design_month_results),
                start_dt,
                end_dt,
                config_values.get("decomposition_type"),
            )

            capex, tank_metrics, capex_components, expansion_only_capex = calculate_capex(
                {**design_new_param_vals, **baseline_param_vals},
                tariff_key,
                {
                    k: max_values[k]
                    for k in ["Ndot_c", "Ndot_b_excess", "addtl_capacity", "Edot_addtl_capacity"]
                },
                upgrade_key,
                state["base_wwtp_key"],
            )

            # Calculate Lifetime Cost
            effective_capex = (
                capex
                if (config_values["include_upgrade_capex"] or storage_type == "battery")
                else capex_components["capex_addtl_capacity"]
            )
            lifetime_cost = get_itemized_lifetime_cost(
                annual_metrics,
                effective_capex,
                capex_components,
                expansion_only_capex,
                battery_kw=tank_metrics.get("battery_power_kw"),
                battery_kwh=tank_metrics.get("battery_energy_kwh"),
                is_battery=storage_type == "battery",
            )

            # Round all cost components to nearest cent to avoid run-to-run jitter
            capex = round(capex, 2)
            lifetime_cost = round_cost_values(lifetime_cost)
            capex_components = round_cost_values(capex_components)
            annual_metrics["opex"] = round_cost_values(annual_metrics["opex"])

            components = lifetime_cost["by_component"]
            print(
                f" {job_key:<30s} LC ${lifetime_cost['total']:>14,.2f}  energy {components['energy']:>12,.2f}"
                f"  demand {components['demand']:>12,.2f}  exp {components['export']:>10,.2f}  H2 {components['h2']:>10,.2f}  capex {capex:>12,.2f}"
            )
            # Tank sizing for a realism check (skipped for battery / no-tank upgrades)
            tank_type = tank_metrics.get("tank_type")
            if tank_type:
                print(
                    f"   {'tank:':<26s} {tank_type}  n={tank_metrics.get('n_tanks', 0):.0f}"
                    f"  total {tank_metrics.get('total_volume_m3', 0):,.0f} m3"
                    f"  ({tank_metrics.get('volume_per_tank_m3', 0):,.1f} m3/tank)"
                )
            month_solve_times = [r["month_solve_time"] for r in design_month_results]
            annual_metrics.update(
                {
                    "max_values": max_values,
                    "lifetime_cost": lifetime_cost,
                    "new_param_vals": design_new_param_vals,
                    "capex_components": capex_components,
                    "capex": capex,
                    "tank_metrics": tank_metrics,
                    "mean_month_solve_time": round(
                        sum(month_solve_times) / len(month_solve_times), 2
                    ),
                    "approx_error": (
                        {
                            "rmse_min": round(min(e["rmse_min"] for e in month_approx_errors), 3),
                            "rmse_max": round(max(e["rmse_max"] for e in month_approx_errors), 3),
                            "rmse_mean": round(
                                sum(e["rmse_mean"] for e in month_approx_errors)
                                / len(month_approx_errors),
                                3,
                            ),
                        }
                        if month_approx_errors
                        else {}
                    ),
                }
            )
            variant_metrics[job_key] = annual_metrics

        # An incomplete variant means the medium comparison can't be made fairly yet;
        # leave the design point out of all_results so it gets re-queued
        if any_incomplete:
            continue

        feasible_variants = {k: m for k, m in variant_metrics.items() if m is not None}
        if not feasible_variants:
            state["all_results"][design_key] = {"lifetime_cost": {"total": np.inf}}
            continue

        best_job_key = min(
            feasible_variants, key=lambda k: feasible_variants[k]["lifetime_cost"]["total"]
        )
        best_metrics = feasible_variants[best_job_key]
        if "~" in best_job_key:
            # Record both media's costs for transparency, then keep only the winner's
            # intermediate files (renamed to the bare key so timeseries plotting finds them)
            best_metrics["medium_costs"] = {
                k.rsplit("~", 1)[-1]: (
                    variant_metrics[k]["lifetime_cost"]["total"] if variant_metrics[k] else np.inf
                )
                for k in variant_keys
            }
            print(
                f"   medium winner: {best_job_key.rsplit('~', 1)[-1]}  {best_metrics['medium_costs']}"
            )
            h_dirs = get_intermediate_dir(config_values["data_name"], config_name, "*")
            for path in glob.glob(os.path.join(h_dirs, f"{design_key}~*_*")):
                filename = os.path.basename(path)
                if filename.startswith(f"{best_job_key}_"):
                    os.replace(
                        path,
                        os.path.join(
                            os.path.dirname(path),
                            filename.replace(f"{best_job_key}_", f"{design_key}_", 1),
                        ),
                    )
                else:
                    os.remove(path)
        state["all_results"][design_key] = best_metrics

    print(f"  Writing JSON for {config_name} ({len(state['all_results'])} design points)")
    best_design_key = cheapest_design_key(state["all_results"])
    best_design = state["all_results"].get(best_design_key, {}) if best_design_key else {}
    best_lifetime_cost = best_design.get("lifetime_cost", {})
    baseline_opex_total = sum(best_design.get("opex", {}).get("baseline", {}).values())
    results_data = {
        "all_results": dict(state["all_results"]),
        "baseline_param_vals": baseline_param_vals,
        "mu_sigma_history": state["mu_sigma_history"],
        "sampled_points_history": state["sampled_points_history"],
        "converged": state["converged"],
        "best_lifetime_cost": state["best_lifetime_cost"],
        # annual_roi_pct is now per-year; scale back to the 10-year cumulative figure this field names
        "best_10_year_roi": (
            best_lifetime_cost.get("annual_roi_pct") * 10
            if best_lifetime_cost.get("annual_roi_pct") is not None
            else None
        ),
        "best_annual_opex_savings_pct": (
            100.0 * best_lifetime_cost.get("annual_savings", 0) / baseline_opex_total
            if baseline_opex_total
            else None
        ),
        "no_improvement_count": state["no_improvement_count"],
    }
    output_file = os.path.join(
        OUTPUT_DATA_DIR, config_values["base_run_name"], f"{config_name}.json"
    )
    write_json_file(output_file, results_data)
    print(f"  Saved results for {config_name} to {output_file}")
    return results_data


def build_job_queue_for_design_keys(design_keys, state, config_name, horizon_days):
    """Build (design_key, ordered_month_keys, new_param_vals) tuples for unsolved design keys.

    Each job covers ALL months for one design_key (months solved sequentially).
    Returns (job_queue, feasible_keys) — feasible_keys is the subset of design_keys
    with at least one feasible medium variant (see fan-out below).
    """
    ordered_month_keys = sorted(state["month_to_days"].keys())
    # gas_tank designs fan out into "~gas" and "~liq" jobs, each solved with its own medium physics and capex;
    # aggregation keeps the cheaper lifetime cost under the bare key, so cost competition decides the crossover.
    fan_out_media = get_upgrade_info(state["upgrade_key"])["is_gas_tank"]
    job_queue = []
    feasible_keys = []
    for design_key in design_keys:
        job_keys = [f"{design_key}~gas", f"{design_key}~liq"] if fan_out_media else [design_key]
        any_feasible = False
        for job_key in job_keys:
            new_param_vals = build_new_param_vals(
                config_name, job_key, state["annual_param_statistics"], horizon_days
            )
            if new_param_vals.get("INFEASIBLE"):
                print(f"Skipping infeasible design {job_key} (N_min > N_max)")
                continue
            any_feasible = True
            job_queue.append((job_key, ordered_month_keys, new_param_vals))
        if any_feasible:
            feasible_keys.append(design_key)
    return job_queue, feasible_keys


def run_async_job_queue(pool, unified_job_queue, horizon_days, n_jobs):
    """Dispatch jobs via pool and collect results.

    unified_job_queue is a list of (config_name, job_data, shared_state) tuples.
    tariff_data and config_values are available in workers via pool initializer.
    Returns dict of {config_name: [per-month results]}.
    """
    results_by_config = {}
    pending_jobs = {}
    job_index = 0
    last_progress_report = 0

    while pending_jobs or job_index < len(unified_job_queue):
        while len(pending_jobs) < n_jobs and job_index < len(unified_job_queue):
            config_name, job_data, shared_state = unified_job_queue[job_index]
            async_result = pool.apply_async(
                process_design_point_all_months,
                args=(job_data, horizon_days, shared_state),
            )
            pending_jobs[async_result] = config_name
            job_index += 1

        for async_result in list(pending_jobs.keys()):
            if async_result.ready():
                config_name = pending_jobs.pop(async_result)
                try:
                    results_by_config.setdefault(config_name, []).extend(async_result.get())
                except Exception as e:
                    print(f"  Job failed for {config_name}: {e}")
                    traceback.print_exc()

        completed_count = sum(len(r) for r in results_by_config.values())
        if completed_count >= last_progress_report + 10:
            last_progress_report = completed_count
            gc.collect()

        time.sleep(0.1)
    return results_by_config


def make_job_shared_state(state):
    return {
        k: state[k]
        for k in (
            "base_wwtp_key",
            "upgrade_key",
            "tariff_key",
            "baseline_vals",
            "month_to_days",
            "overflow_dates",
        )
    }


def init_config_state(config_item, day_profiles, overflow_dates, config_values):
    """Build state dict for one config, or return None if it should be skipped."""
    run_name = config_values["base_run_name"]
    output_dir = os.path.join(OUTPUT_DATA_DIR, run_name)
    is_sensitivity = "design_point" in config_values or "design_points" in config_values
    max_iterations = config_values["max_iterations"]

    base_wwtp_key = config_item["base_wwtp_key"]
    upgrade_key = config_item["upgrade_key"]
    tariff_key = config_item["tariff_key"]
    config_name = get_config_name(base_wwtp_key, upgrade_key, tariff_key)
    gas, o2_tech_base, _, summer_multiplier, summer_smoothing = split_key(base_wwtp_key)
    upgrade_info = get_upgrade_info(upgrade_key)

    # Baseline: Edot_b_baseline = Ndot_target (O2 demand from airflow, mol/hr) × (comp_factor + ei_o2_base), plus Edot_rem
    # (rest of facility) = Edot_t_baseline. comp_factor uses P_AER (14.5 psig), not the facility's higher pressure (gap: Edot_rem).

    # Loop-invariants: derived once per CONFIG
    peaking_multiplier = float(summer_smoothing)
    P_b = P_AER + P_ATM
    P_init_base = P_init_map[o2_tech_base]
    ei_o2_base = ei_o2_map[o2_tech_base]
    ei_evap_base = ei_evap_map[o2_tech_base]
    Edot_b_comp_coeff = comp_lp_coeff(P_init_base, P_b, rec=False)
    o2_factor = o2_multiplier_map[gas] * frac_o2_map["air"] / 24
    comp_factor = Edot_b_comp_coeff / frac_o2_map[gas]

    month_to_days = {}
    baseline_vals = {}
    for date, day_profile in sorted(day_profiles.items()):
        year, month, _ = date.split("-")
        month_key = f"{year}-{int(month):02d}"
        # Overflow days get baseline_vals but are NOT part of month_to_days
        if date not in overflow_dates:
            month_to_days.setdefault(month_key, []).append(date)

        # Apply power transform for diurnal curve parametrization
        # summer_smoothing: 1.0 = original, 0.9 = 10% less variation, etc
        air_flow_raw = day_profile["Blower_AerationBasin_Air_Flow"]
        Ndot_target_raw = volume_to_moles_stp(np.asarray(air_flow_raw)) * o2_factor
        gamma = find_gamma_for_peaking_target(air_flow_raw, peaking_multiplier)
        scale = float(summer_multiplier) if is_dry_season(month) else 1.0
        Ndot_target_day = (
            volume_to_moles_stp(apply_temporal_transform(air_flow_raw, gamma=gamma, scale=scale))
            * o2_factor
        )
        Ndot_target_unscaled = (
            volume_to_moles_stp(apply_temporal_transform(air_flow_raw, gamma=gamma, scale=1.0))
            * o2_factor
        )

        Edot_b_baseline = Ndot_target_day * (comp_factor + ei_o2_base)
        Edot_rem = np.array(day_profile["VirtualDemand_RestOfFacilityPower"])

        baseline_vals[date] = {
            "Ndot_target": Ndot_target_day,
            "Ndot_target_unscaled": Ndot_target_unscaled,  # for Ndot_target_min_unscaled
            "Edot_rem": Edot_rem,
            "Edot_t_baseline": Edot_b_baseline + Edot_rem,
            "Edot_b_baseline": Edot_b_baseline,
            "P_init_base": P_init_base,
            "ei_o2_base": ei_o2_base,
            "ei_evap": ei_evap_base,
            "P_b": P_b,
            "Edot_b_comp_coeff": Edot_b_comp_coeff,
            "Ndot_b_max": max(Ndot_target_raw),  # from the UNtransformed profile
        }

    # Annual statistics excluding overflow buffer days. Ndot_b_max is the record-wide max of the raw profile (no
    # summer_multiplier or diurnal smoothing), so existing blower capacity stays fixed across both summer sweep axes.
    selected_baseline = [v for d, v in baseline_vals.items() if d not in overflow_dates]
    target_all = np.concatenate([v["Ndot_target"] for v in selected_baseline])
    annual_param_statistics = {
        "Ndot_target_min": float(np.min(target_all)),
        "Ndot_target_max": float(np.max(target_all)),
        "Ndot_target_mean": float(np.mean([np.mean(v["Ndot_target"]) for v in selected_baseline])),
        "Edot_b_mean": float(np.mean([np.mean(v["Edot_b_baseline"]) for v in selected_baseline])),
        "Edot_t_baseline_max": float(
            np.max(np.concatenate([v["Edot_t_baseline"] for v in selected_baseline]))
        ),
        "Ndot_b_max": float(max(v["Ndot_b_max"] for v in selected_baseline)),
        "Ndot_target_min_unscaled": float(
            np.min(np.concatenate([v["Ndot_target_unscaled"] for v in selected_baseline]))
        ),
    }

    # Compute effective design variable ranges (all clamps in one place)
    storage_type = upgrade_info["storage_type"]
    if not is_sensitivity:
        type_overrides = config_values.get("storage_type_ranges", {}).get(storage_type, {})
        ranges = get_effective_ranges(
            storage_type,
            gas,
            type_overrides.get("o2_range", config_values["o2_range"]),
            config_values["comp_ratio_range"],
            type_overrides.get(
                "charge_power_ratio_range", config_values["charge_power_ratio_range"]
            ),
            annual_param_statistics,
            summer_multiplier,
        )
        if ranges["skipped"]:
            print(f"  WARNING: {ranges['skip_reason']} ({config_name}), skipping")
            return None

    if "design_point" in config_values:
        initial_design_points = [config_values["design_point"]]
    elif "design_points" in config_values:
        lookup_key = f"{get_base_key(base_wwtp_key)}___{upgrade_key}"
        source_design_point = config_values["design_points"].get(lookup_key)
        if source_design_point is None:
            print(f"WARNING: No source design point for {lookup_key}, skipping")
            return None
        initial_design_points = [source_design_point]
    else:
        initial_design_points = get_initial_design_points(ranges, config_values["designs_per_run"])

    # Skip if all iterations completed AND grid refinement done; resume otherwise
    output_file = f"{output_dir}/{config_name}.json"
    existing = {}
    already_converged = False
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing = json.load(f)
        completed_iterations = len(existing.get("mu_sigma_history", []))
        grid_done = existing.get("grid_refinement_done", False)  # not plotted vs storage hours
        grid_applicable = not run_name.startswith("si_") and any(
            k in upgrade_key for k in GRID_REFINEMENT_UPGRADE_KEYS
        )
        grid_satisfied = grid_done or not grid_applicable
        saved_converged = existing.get("converged", False) or grid_done
        if (saved_converged or completed_iterations >= max_iterations) and grid_satisfied:
            print(f"Skipping {config_name} (converged + grid refinement complete)")
            return None
        if is_sensitivity and (saved_converged or completed_iterations >= max_iterations):
            existing_results = existing.get("all_results", {})
            if existing_results:
                best_existing_cost = min(
                    v.get("lifetime_cost", {}).get("total", np.inf)
                    for v in existing_results.values()
                )
                if is_valid_lifetime_cost(best_existing_cost):
                    any_result = next(iter(existing_results.values()))
                    expansion_only = float(
                        any_result.get("lifetime_cost", {}).get("expansion_only", 0) or 0
                    )
                    if best_existing_cost >= expansion_only:
                        print(f"Skipping {config_name} (sensitivity complete, best NPV negative)")
                        return None
        if saved_converged or completed_iterations >= max_iterations:
            print(f"{config_name}: converged, grid refinement pending")
            already_converged = True
        else:
            print(
                f"Resuming {config_name} (completed {completed_iterations}/{max_iterations} iterations)"
            )

    # Profile keys to aggregate, consistent with var_dict: battery has E (energy_storage), tanks have
    # N/Ndot_c (o2_storage), expansion-only (none) has only base keys.
    keys_to_append = list(PROFILE_KEYS_BASE)
    if upgrade_info["is_battery"]:
        keys_to_append.append("E")
    elif upgrade_info["is_liquid_tank"] or upgrade_info["is_gas_tank"]:
        keys_to_append.extend(["N", "Ndot_c"])

    saved_best = existing.get("best_lifetime_cost")
    return {
        "base_wwtp_key": base_wwtp_key,
        "upgrade_key": upgrade_key,
        "tariff_key": tariff_key,
        "month_to_days": month_to_days,
        "baseline_vals": baseline_vals,
        "annual_param_statistics": annual_param_statistics,
        "overflow_dates": sorted(overflow_dates),
        "effective_o2_range": None if is_sensitivity else ranges["o2_range"],
        "effective_comp_ratio_range": None if is_sensitivity else ranges["comp_ratio_range"],
        "effective_cp_range": None if is_sensitivity else ranges["cp_range"],
        "all_results": existing.get("all_results", {}),
        "initial_design_points": initial_design_points,
        "keys_to_append": keys_to_append,
        "iteration": len(existing.get("mu_sigma_history", [])),
        "best_lifetime_cost": saved_best if saved_best is not None else np.inf,
        "no_improvement_count": existing.get("no_improvement_count", 0),
        "converged": already_converged,
        "mu_sigma_history": existing.get("mu_sigma_history", [None]),
        "sampled_points_history": existing.get(
            "sampled_points_history", [list(initial_design_points)]
        ),
    }


def solve_design_points(points, state, config_name, config_values, pool, tariff_data):
    """Solve and aggregate the not-yet-solved design points, GRID_BATCH_SIZE at a time."""
    horizon_days = config_values["horizon_days"]
    unsolved_keys = [
        build_design_key(p) for p in points if build_design_key(p) not in state["all_results"]
    ]
    for batch_start in range(0, len(unsolved_keys), GRID_BATCH_SIZE):
        batch_keys = unsolved_keys[batch_start : batch_start + GRID_BATCH_SIZE]
        job_queue, feasible_keys = build_job_queue_for_design_keys(
            batch_keys, state, config_name, horizon_days
        )
        if not job_queue:
            print(f"{config_name}: batch of {len(batch_keys)} grid points all infeasible")
            continue

        shared_state = make_job_shared_state(state)
        print(f"{config_name}: {len(feasible_keys)} grid points, {len(job_queue)} jobs (batch)")
        grid_job_queue = [(config_name, job_data, shared_state) for job_data in job_queue]
        grid_results = run_async_job_queue(
            pool, grid_job_queue, horizon_days, config_values["n_jobs"]
        ).get(config_name, [])
        if grid_results:
            print(f"  Aggregating grid results for {config_name}")
            aggregate_and_save_config_results(
                config_name=config_name,
                state=state,
                all_designs_month_results=grid_results,
                unsolved_pts=feasible_keys,
                tariff_data=tariff_data,
                config_values=config_values,
            )


def run_grid_refinement(config_states, pool, config_values, tariff_data):
    """Run post-convergence grid refinement for all tank-based configs."""
    print("Grid refinement phase (post-convergence)")
    run_name = config_values["base_run_name"]
    horizon_days = config_values["horizon_days"]

    for config_name, state in config_states.items():
        # Only the fig2-plotted configs (air compressor / o2 psa / o2 cryo) need the per-hour cost envelope; others
        # (elec/battery) skip WITHOUT marking done, so their grid_refinement_done stays False (not a misleading "N/A done").
        if not any(k in state["upgrade_key"] for k in GRID_REFINEMENT_UPGRADE_KEYS):
            continue

        mu_sigma = next(ms for ms in reversed(state["mu_sigma_history"]) if ms is not None)
        mu = mu_sigma["mu"]
        upgrade_info = get_upgrade_info(state["upgrade_key"])
        h_min, h_max = state["effective_o2_range"]
        comp_ratio_lo, comp_ratio_hi = state["effective_comp_ratio_range"]
        charge_power_lo, charge_power_hi = state["effective_cp_range"]
        # Design-key indices of the free (non-hour) dims, in mu[1:] order: gas [comp_ratio, charge_power, frac],
        # liquid [charge_power, frac]
        free_dims = [1, 2, 3] if upgrade_info["is_gas_tank"] else [2, 3]

        # Same coarse frac grid at every hour, plus the converged optimum's frac (gas mu[3], liquid mu[2]): the coarse
        # grid alone missed it (e.g. ~0.18), so off-optimum hours never got a near-optimal design.
        frac_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        if upgrade_info["is_liquid_tank"]:
            frac_values = sorted(set(frac_values + [round(float(mu[2]), 2)]))
        elif upgrade_info["is_gas_tank"]:
            frac_values = sorted(set(frac_values + [round(float(mu[3]), 2)]))

        # Pass 1: coarse feasibility-aware cr x frac grid every 4 hours over the full range, so the fig2 per-hour cost
        # envelope has real points to draw (each hour fans out to gas + liquid downstream).
        infeasible_hour_frac_points, _, valid_designs_by_hour = collect_infeasibility_data(
            state["all_results"]
        )
        points = set()
        for hour in np.arange(h_min, h_max + 4, 4):
            min_frac = infeasibility_boundary_frac(hour, infeasible_hour_frac_points)
            feasible_fracs = [f for f in frac_values if f > min_frac]
            if upgrade_info["is_liquid_tank"]:
                # mu dims: [hours, charge_power_ratio, frac]
                for frac in feasible_fracs:
                    points.add(round_point((hour, 100.0, mu[1], frac)))
            elif upgrade_info["is_gas_tank"]:
                # mu dims: [hours, comp_ratio, charge_power_ratio, frac]. Narrow comp_ratio to solved results within
                # ±8 hours (independent of the hours step so comp_ratio exploration isn't starved by sparse bins)
                nearby_comp_ratios = [
                    comp_ratio
                    for hour_bin, designs in valid_designs_by_hour.items()
                    if abs(hour_bin - hour) <= 8
                    for comp_ratio, _ in designs
                ]
                comp_ratio_values = [mu[1]]
                if nearby_comp_ratios:
                    comp_ratio_values = [
                        max(comp_ratio_lo, min(nearby_comp_ratios) * 0.8),
                        mu[1],
                        min(comp_ratio_hi, max(nearby_comp_ratios) * 1.2),
                    ]
                for comp_ratio in comp_ratio_values:
                    for frac in feasible_fracs:
                        points.add(round_point((hour, comp_ratio, mu[2], frac)))
        solve_design_points(points, state, config_name, config_values, pool, tariff_data)

        # Pass 2: short per-hour conditional CEM over [h_min, opt+3], where fig2 row-iii draws and the gas/liquid boundary
        # sits; free dims seeded from the converged MVN, frac floored at each hour's feasibility boundary.
        band_rng = np.random.RandomState(RANDOM_SEED)
        opt_h = float(parse_design_key(cheapest_design_key(state["all_results"]))[0])
        band = np.arange(h_min, min(h_max, opt_h + 3) + 1, 1.0)
        print(f"{config_name}: mini-CEM refining {len(band)} band hours")

        # Per-hour (mean, cov) CEM, but each iteration samples all live hours into ONE solve call
        # so it fills the pool (per-hour barriers idle the cores). 3 iters, <1%-improvement stop/hour.
        mean_free = np.array(mu, float)[1:]  # free dims = drop the hours dim
        cov_free = (
            np.atleast_2d(np.array(mu_sigma["sigma"], float)[1:, 1:] * 4.0)
            + np.eye(len(mean_free)) * 1e-3
        )  # inflate to explore
        cem_by_hour = {
            hour: {"mean": mean_free.copy(), "cov": cov_free.copy(), "best": np.inf, "done": False}
            for hour in band
        }
        for _ in range(3):  # max_iters
            live_hours = [hour for hour in band if not cem_by_hour[hour]["done"]]
            if not live_hours:
                break
            infeasible_hour_frac_points, _, _ = collect_infeasibility_data(state["all_results"])
            points = set()
            for hour in live_hours:
                hour_state = cem_by_hour[hour]
                # Clip sampled free dims to the config's ranges and frac to the hour's feasibility-boundary floor
                bounds = {
                    1: (comp_ratio_lo, comp_ratio_hi),
                    2: (charge_power_lo, charge_power_hi),
                    3: (infeasibility_boundary_frac(hour, infeasible_hour_frac_points), 1.0),
                }
                for _ in range(5):  # points_per_iter
                    point = [hour, 100.0, None, None]
                    for dim, value in zip(
                        free_dims,
                        band_rng.multivariate_normal(hour_state["mean"], hour_state["cov"]),
                    ):
                        point[dim] = float(np.clip(value, *bounds[dim]))
                    points.add(round_point(tuple(point)))
            solve_design_points(points, state, config_name, config_values, pool, tariff_data)

            for hour in live_hours:
                hour_state = cem_by_hour[hour]
                # (total cost, free-dim vector) for every valid solved design in this rounded-hour bin, cheapest first
                elite = sorted(
                    (
                        (
                            result["lifetime_cost"]["total"],
                            [float(parse_design_key(key)[dim]) for dim in free_dims],
                        )
                        for key, result in state["all_results"].items()
                        if round(float(parse_design_key(key)[0])) == round(hour)
                        and is_valid_lifetime_cost(result["lifetime_cost"]["total"])
                    ),
                    key=lambda row: row[0],
                )
                if not elite:
                    hour_state["cov"] *= 1.5  # all infeasible this hour — widen and retry
                    continue
                top = np.array([free_values for _, free_values in elite[: max(2, len(elite) // 2)]])
                hour_state["mean"] = top.mean(axis=0)
                hour_state["cov"] = (
                    np.atleast_2d(np.cov(top.T) if len(top) > 1 else hour_state["cov"])
                    + np.eye(len(hour_state["mean"])) * 1e-3
                )
                if (
                    elite[0][0] > hour_state["best"] * 0.99
                ):  # <1% improvement — stop refining this hour
                    hour_state["done"] = True
                hour_state["best"] = elite[0][0]

        # Gas-region anchor: the mini-CEM seeds from the liquid-favorable mode, so probe the high-frac/high-cr geometry
        # where compressed gas is cheapest at the lowest band hours (gas-tank configs only; liquid has no medium boundary).
        if upgrade_info["is_gas_tank"]:
            probe = {
                round_point((hour, comp_ratio_hi, float(mu[2]), frac))
                for hour in band[:2]
                for frac in (0.75, 1.0)
            }
            solve_design_points(probe, state, config_name, config_values, pool, tariff_data)

        # Mark grid refinement done
        output_file = os.path.join(OUTPUT_DATA_DIR, run_name, f"{config_name}.json")
        if not os.path.exists(output_file):
            continue
        with open(output_file, "r") as f:
            saved = json.load(f)
        saved["grid_refinement_done"] = True
        write_json_file(output_file, saved)
        print(f"  Marked grid refinement done for {config_name}")

        # Config is fully solved: keep the full profile only for the best design (fig2 row-ii) and slim the rest to the
        # recompile columns, in place (never deleted, so lifetime cost and the sensitivity SoC seed stay recomputable).
        if config_values["save_necessary_only"]:
            best_design_key = cheapest_design_key(state["all_results"])
            intermediate_dir = get_intermediate_dir(
                config_values["data_name"], config_name, horizon_days
            )
            for path in glob.glob(os.path.join(intermediate_dir, "*_profile.csv")):
                design_key = re.sub(r"_\d{4}-\d{2}-\d{2}_profile\.csv$", "", os.path.basename(path))
                if design_key == best_design_key:
                    continue
                profile_df = pd.read_csv(path)
                keep_cols = [c for c in profile_df.columns if c in PROFILE_KEYS_ALL]
                if len(keep_cols) < len(profile_df.columns):
                    profile_df[keep_cols].to_csv(path, index=False)

    print("Grid refinement complete.")


def run_configuration(configs, config_values):
    """Run optimization for a list of configurations in parallel using a unified job queue.

    Args:
        configs: List of configuration dictionaries
        config_values: Dictionary containing optimization parameters
    """
    run_name = config_values["base_run_name"]
    n_jobs = config_values["n_jobs"]
    max_iterations = config_values["max_iterations"]
    horizon_days = config_values["horizon_days"]

    tariff_data = get_tariff_data()
    with open(os.path.join(OUTPUT_DATA_DIR, run_name, "day_profiles.json"), "r") as f:
        day_profiles_raw = json.load(f)
    overflow_dates = set(day_profiles_raw["overflow_dates"])
    # Convert lists back to numpy arrays for compatibility
    day_profiles = {
        date_str: {key: np.array(value) for key, value in day_data["profile"].items()}
        for date_str, day_data in day_profiles_raw["day_profiles"].items()
    }

    config_states = {}
    for config_item in configs:
        config_name = get_config_name(
            config_item["base_wwtp_key"], config_item["upgrade_key"], config_item["tariff_key"]
        )
        if config_name in config_states:
            continue
        state = init_config_state(config_item, day_profiles, overflow_dates, config_values)
        if state is not None:
            config_states[config_name] = state

    del day_profiles
    gc.collect()

    if not config_states:
        print("No configurations to run")
        return

    # Recycle workers after 4 tasks to cap cumulative memory growth.
    # tariff_data and config_values are passed once via initializer (not pickled per task).
    # Leaving the with block terminates the workers, also on Ctrl+C, which then stops the whole run.
    config_start_time = time.time()
    with Pool(
        processes=n_jobs,
        maxtasksperchild=4,
        initializer=pool_initializer,
        initargs=(tariff_data, config_values),
    ) as shared_pool:
        while any(
            not s["converged"] and s["iteration"] < max_iterations for s in config_states.values()
        ):
            unified_job_queue = []
            unsolved_pts_by_config = {}

            for config_name, state in config_states.items():
                if state["converged"] or state["iteration"] >= max_iterations:
                    continue

                iteration = state["iteration"]
                all_results = state["all_results"]
                print(f"{config_name}: iteration {iteration}")

                if iteration == 0:
                    design_points = state["initial_design_points"]
                else:
                    design_points, mu_sigma = sample_from_multivariate_normal(
                        all_results,
                        config_values["designs_per_run"],
                        state["effective_o2_range"],
                        state["effective_comp_ratio_range"],
                        state["effective_cp_range"],
                        state["upgrade_key"],
                        iteration,
                    )
                    state["mu_sigma_history"].append(mu_sigma)
                    state["sampled_points_history"].append(list(design_points))

                unsolved_pts = [
                    build_design_key(pt)
                    for pt in design_points
                    if build_design_key(pt) not in all_results
                ]
                if not unsolved_pts:
                    # Collapsed sampler: distribution tightened onto already-solved designs, so no
                    # new work — treat as converged. Skip iteration 0 (initial grid, not collapse).
                    if iteration >= 1:
                        state["converged"] = True
                        print(
                            f"{config_name}: sampler exhausted (sampled points already solved) — converged"
                        )
                    state["iteration"] += 1
                    continue

                # Check convergence
                current_best_lifetime_cost = min(
                    [
                        r["lifetime_cost"]["total"]
                        for r in all_results.values()
                        if is_valid_lifetime_cost(r["lifetime_cost"]["total"])
                    ],
                    default=np.inf,
                )

                # at least 3 iterations before checking convergence
                if iteration >= 3:
                    # Cost convergence
                    if current_best_lifetime_cost < state["best_lifetime_cost"]:
                        improvement = (
                            (state["best_lifetime_cost"] - current_best_lifetime_cost)
                            / abs(state["best_lifetime_cost"])
                            if state["best_lifetime_cost"] != np.inf
                            else np.inf
                        )
                        if improvement < config_values["lc_convergence_threshold"]:
                            state["no_improvement_count"] += 1
                        else:
                            state["no_improvement_count"] = 0
                        state["best_lifetime_cost"] = current_best_lifetime_cost
                    else:
                        # No improvement at all
                        state["no_improvement_count"] += 1

                    # Sigma (distribution) convergence: check if mu and sigma diagonal have stabilized
                    # between the last two MVN fits (relative to the mean)
                    sigma_converged = False
                    mu_sigma_history = state["mu_sigma_history"]
                    if (
                        len(mu_sigma_history) >= 2
                        and mu_sigma_history[-1] is not None
                        and mu_sigma_history[-2] is not None
                    ):
                        mu_prev = np.array(mu_sigma_history[-2]["mu"])
                        mu_curr = np.array(mu_sigma_history[-1]["mu"])
                        sigma_prev = np.sqrt(np.abs(np.diag(mu_sigma_history[-2]["sigma"])))
                        sigma_curr = np.sqrt(np.abs(np.diag(mu_sigma_history[-1]["sigma"])))

                        # Relative change in mu and in std dev (both normalized by current mu)
                        mu_scale = np.maximum(np.abs(mu_curr), 1e-6)
                        mu_rel_change = np.abs(mu_curr - mu_prev) / mu_scale
                        sigma_rel_change = np.abs(sigma_curr - sigma_prev) / mu_scale

                        sigma_threshold = config_values.get("sigma_convergence_threshold", 0.1)
                        if np.all(mu_rel_change < sigma_threshold) and np.all(
                            sigma_rel_change < sigma_threshold
                        ):
                            sigma_converged = True

                    cost_converged = state["no_improvement_count"] >= 3
                    if cost_converged and sigma_converged:
                        print(
                            f"{config_name}: Converged after {iteration + 1} iterations"
                            f" (cost + distribution)"
                        )
                        state["converged"] = True
                        continue
                    elif cost_converged:
                        print(
                            f"{config_name}: Cost converged but distribution still shifting"
                            f" — continuing"
                        )
                        # Decrement to allow more iterations for distribution to settle
                        state["no_improvement_count"] = max(state["no_improvement_count"] - 1, 0)

                # Build job queue; shared state includes all baseline_vals (all months + overflow)
                job_queue, unsolved_pts = build_job_queue_for_design_keys(
                    unsolved_pts,
                    state,
                    config_name,
                    horizon_days,
                )
                job_shared_state = make_job_shared_state(state)
                unified_job_queue.extend(
                    (config_name, job_data, job_shared_state) for job_data in job_queue
                )
                print(f"Added {len(job_queue)} jobs to the queue")
                unsolved_pts_by_config[config_name] = unsolved_pts

            if not unified_job_queue:
                print("NO more jobs")
                break

            np.random.RandomState(RANDOM_SEED).shuffle(unified_job_queue)  # intermix configs

            print(
                f"Processing {len(unified_job_queue)} jobs across {len(unsolved_pts_by_config)} configs"
            )
            all_results_by_config = run_async_job_queue(
                shared_pool, unified_job_queue, horizon_days, n_jobs
            )

            # Aggregate and save any configs (even on interruption — save what we have)
            # Pop results per-config to free memory incrementally
            for config_name, unsolved_pts in unsolved_pts_by_config.items():
                results = all_results_by_config.pop(config_name, [])
                state = config_states[config_name]

                if results:
                    print(
                        f"\n  Aggregating results for {config_name}: {len(results)} month results from {len(unsolved_pts)} design points"
                    )
                    aggregate_and_save_config_results(
                        config_name=config_name,
                        state=state,
                        all_designs_month_results=results,
                        unsolved_pts=unsolved_pts,
                        tariff_data=tariff_data,
                        config_values=config_values,
                    )
                else:
                    print(
                        f"\n  WARNING: No results for {config_name} iteration {state['iteration']}"
                    )
                    print(f"           Expected to aggregate {len(unsolved_pts)} design points")
                    # Still increment iteration counter so we don't get stuck
                del results

                print(
                    f"  {config_name}: after iteration {state['iteration']}, all_results has {len(state['all_results'])} designs"
                )
                state["iteration"] += 1
                gc.collect()

        # Convergence is detected after the last save, so persist it (else a re-run resumes the search)
        for config_name, state in config_states.items():
            output_file = os.path.join(OUTPUT_DATA_DIR, run_name, f"{config_name}.json")
            if state["converged"] and os.path.exists(output_file):
                with open(output_file) as f:
                    results_data = json.load(f)
                results_data["converged"] = True
                write_json_file(output_file, results_data)

        # Skip grid refinement for single-design-point configs (sensitivity_optimal, speed_test), and for SI runs,
        # which only need fig 3 (the per-config optimum), not the expensive fig 2 per-hour cost envelope.
        if max_iterations > 1 and not run_name.startswith("si_"):
            run_grid_refinement(config_states, shared_pool, config_values, tariff_data)

    print(f"Pool shut down. Total time: {time.time() - config_start_time:.1f}s")
