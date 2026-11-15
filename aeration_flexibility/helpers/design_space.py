"""Design-space sampling for the search: corner and Latin hypercube starting points, rounding, resampling."""

from itertools import product
import numpy as np
from scipy.stats import qmc
from helpers.config_keys import (
    parse_design_key,
    build_design_key,
    is_valid_lifetime_cost,
    get_upgrade_info,
)
from helpers.parameters import (
    M_O2,
    max_comp_ratio_map,
    rho_hpo_liquid,
    min_liquid_tank_volume_m3,
    RANDOM_SEED,
)

# Comp ratio placeholder in design keys for storage without compression (liquid tanks, batteries)
DUMMY_COMP_RATIO = 100.0
# All design-space limits in one place for easy editing
DESIGN_LIMITS = {
    "gas_tank": {
        "fixed_comp_ratio": None,  # active
        "frac_range": (0.0, 1.0),  # active
        "o2_max_override": None,  # use config range
        "corner_fracs": (0.2, 1.0),
    },
    "liquid_tank": {
        "fixed_comp_ratio": DUMMY_COMP_RATIO,
        "frac_range": (0.0, 1.0),  # active
        "o2_max_override": None,
        "min_volume_m3": min_liquid_tank_volume_m3,  # min liquid tank: clamps hr_of_O2 min
        "corner_fracs": (0.1, 1.0),
    },
    "battery": {
        "fixed_comp_ratio": DUMMY_COMP_RATIO,
        "frac_range": (1.0, 1.0),  # batteries don't expand O2 capacity
        "o2_max_override": 10.0,  # batteries need fewer hours
        "min_charge_power_kW": 100.0,  # NREL cost data validity threshold
        "corner_fracs": (1.0,),  # fixed
    },
}
# Intermediate-hour sweep at low charge power: corners only probe hour extremes, and the cheap slow-cycling designs
# live at cp_min, so seed that style across sizes (at frac 0.25)
SWEEP_HOURS = (2.0, 4.0, 8.0, 16.0, 24.0, 32.0)
SWEEP_FRAC = 0.25
# CEM fit: 16 (vs 8) elites keep multiple basins in the cumulative-elite fit so sigma doesn't collapse onto one basin
# before co-dependent dims converge
ELITE_SAMPLES = 16
EXP_WEIGHT = 5.0  # softmax temperature; concentrates mu on cheapest elites (2.0 was too flat, mu drifted off the min)
MIN_VARIANCE = 1e-3  # for numerical stability
MAX_SAMPLE_ATTEMPTS_PER_DESIGN = 100


def round_point(point):
    """Round a design point as design keys store it: whole compression ratio, other dims to 0.01."""
    return tuple(round(float(x), 0) if i == 1 else round(float(x), 2) for i, x in enumerate(point))


def get_effective_ranges(
    storage_type,
    gas,
    o2_range,
    comp_ratio_range,
    charge_power_ratio_range,
    annual_param_statistics,
    summer_multiplier,
):
    """Compute effective design variable ranges after applying all clamps.

    Returns dict with keys: storage_type, o2_range, comp_ratio_range, cp_range, frac_range,
    fixed_comp_ratio, corner_fracs, skipped, skip_reason.
    """
    limits = DESIGN_LIMITS[storage_type]

    # Start with config ranges
    effective_o2_range = list(o2_range)
    effective_comp_ratio_range = list(comp_ratio_range)
    effective_cp_range = list(charge_power_ratio_range)

    # Gas type max compression ratio (applies to all)
    effective_comp_ratio_range[1] = min(effective_comp_ratio_range[1], max_comp_ratio_map[gas])

    # O2 max override (battery: 10h cap)
    if limits["o2_max_override"] is not None:
        effective_o2_range[1] = min(effective_o2_range[1], limits["o2_max_override"])

    # Battery: min charge power: clamp cp_min
    skip = False
    skip_reason = ""
    if "min_charge_power_kW" in limits:
        min_charge_power_ratio = (
            limits["min_charge_power_kW"]
            * float(summer_multiplier)
            / annual_param_statistics["Edot_t_baseline_max"]
        )
        if min_charge_power_ratio > effective_cp_range[0]:
            print(
                f"  Battery min clamp: cp>={min_charge_power_ratio:.3f} "
                f"(Edot_c>={limits['min_charge_power_kW']}kW)"
            )
        effective_cp_range[0] = max(effective_cp_range[0], min_charge_power_ratio)
        if effective_cp_range[0] > effective_cp_range[1]:
            skip, skip_reason = True, "Facility too small for battery"

    # Liquid tank: min volume: clamp hr_of_O2 min
    if "min_volume_m3" in limits:
        min_hours = (
            limits["min_volume_m3"]
            * rho_hpo_liquid
            * 1000
            / (M_O2 * annual_param_statistics["Ndot_target_mean"])
        )
        if min_hours > effective_o2_range[0]:
            print(
                f"  Liquid tank min clamp: hr_o2>={min_hours:.2f} "
                f"(liquid vol>={limits['min_volume_m3']}m³)"
            )
        effective_o2_range[0] = max(effective_o2_range[0], min_hours)
        if effective_o2_range[0] > effective_o2_range[1]:
            skip, skip_reason = True, "Facility too small for liquid tank"

    return {
        "storage_type": storage_type,
        "o2_range": tuple(effective_o2_range),
        "comp_ratio_range": tuple(effective_comp_ratio_range),
        "cp_range": tuple(effective_cp_range),
        "frac_range": limits["frac_range"],
        "fixed_comp_ratio": limits["fixed_comp_ratio"],
        "corner_fracs": limits["corner_fracs"],
        "skipped": skip,
        "skip_reason": skip_reason,
    }


def get_initial_design_points(ranges, designs_per_run):
    """Generate initial design points: corners for boundary coverage + center points and LHS for interior.

    Args:
        ranges: dict from get_effective_ranges()
        designs_per_run: number of LHS interior points
    """
    (h_min, h_max) = ranges["o2_range"]
    (r_min, r_max) = ranges["comp_ratio_range"]
    (cp_min, cp_max) = ranges["cp_range"]
    fixed_comp_ratio = ranges["fixed_comp_ratio"]
    # Per design-key dimension (hours, comp ratio, charge power, frac); comp ratio is only free for gas tanks
    dim_ranges = [
        (h_min, h_max),
        (r_min, r_max) if fixed_comp_ratio is None else (fixed_comp_ratio, fixed_comp_ratio),
        (cp_min, cp_max),
        ranges["frac_range"],
    ]

    # True corner points — all active dimensions at their extremes
    corners = list(
        product(
            (h_min, h_max),
            dict.fromkeys(dim_ranges[1]),
            (cp_min, cp_max),
            ranges["corner_fracs"],
        )
    )
    if ranges["storage_type"] != "battery":
        mid_comp_ratio = (
            fixed_comp_ratio if fixed_comp_ratio is not None else (r_min * r_max) ** 0.5
        )
        corners += [
            (h, mid_comp_ratio, cp_min, SWEEP_FRAC) for h in SWEEP_HOURS if h_min < h < h_max
        ]

    # Only sample truly free dimensions (lo < hi); fixed dims get their single value directly.
    free_dims = [i for i, (lo, hi) in enumerate(dim_ranges) if lo < hi]

    # Center points: the center plus a 2^k factorial at 25%/75% of each free dim, so the search doesn't depend on the LHS
    # landing mid-space. Hours, comp ratio and charge power span decades, so they're log-spaced; frac is linear.
    for quantiles in [(0.5,) * len(free_dims)] + list(product((0.25, 0.75), repeat=len(free_dims))):
        point = [lo for lo, _ in dim_ranges]
        for i, q in zip(free_dims, quantiles):
            lo, hi = dim_ranges[i]
            point[i] = lo + q * (hi - lo) if i == 3 else lo * (hi / lo) ** q
        corners.append(point)
    corners = list(dict.fromkeys(round_point(p) for p in corners))

    # LHS interior: designs_per_run points on top of corners and center points.
    sampler = qmc.LatinHypercube(d=len(free_dims), seed=int(RANDOM_SEED))
    lhs_samples = qmc.scale(
        sampler.random(n=int(designs_per_run)),
        [dim_ranges[i][0] for i in free_dims],
        [dim_ranges[i][1] for i in free_dims],
    )
    interior = []
    for row in lhs_samples:
        point = [lo for lo, _ in dim_ranges]
        for i, value in zip(free_dims, row):
            point[i] = value
        if round_point(point) not in corners:
            interior.append(round_point(point))

    return corners + interior


def infeasibility_boundary_frac(hours, infeasible_points):
    """Return the interpolated frac on the infeasibility boundary at the given hours.

    infeasible_points is a list of (hours, frac) tuples that were infeasible.
    The boundary is the linear interpolation through these points (sorted by hours).
    Points at or below this frac for a given hours value are presumed infeasible.

    Returns 0.0 if no boundary can be determined (candidate is not constrained).
    """
    if not infeasible_points:
        return 0.0
    # Sort by hours ascending for interpolation
    sorted_points = sorted(infeasible_points, key=lambda p: p[0])
    # Extrapolate left: if hours <= smallest infeasible hours, use the max frac seen
    if hours <= sorted_points[0][0]:
        return max(f for _, f in sorted_points)
    # Extrapolate right: beyond largest infeasible hours, no constraint
    if hours >= sorted_points[-1][0]:
        return 0.0
    # Linear interpolation between bracketing points (the first match always has h0 < hours, so h1 > h0)
    for i in range(len(sorted_points) - 1):
        h0, f0 = sorted_points[i]
        h1, f1 = sorted_points[i + 1]
        if h0 <= hours <= h1:
            return f0 + (hours - h0) / (h1 - h0) * (f1 - f0)


def collect_infeasibility_data(all_results, is_battery=False):
    """Classify design results into infeasibility boundary data and a valid-by-hour lookup.

    Returns:
        infeasible_hour_frac_points: list of (hours, frac) for infeasible non-battery designs
        infeasible_hours_max: max infeasible hours for battery (0.0 otherwise)
        valid_by_hour: dict of rounded_hour -> [(comp_ratio, lifetime_cost)]
    """
    infeasible_hour_frac_points, valid_by_hour = [], {}
    infeasible_hours_max = 0.0
    for design_key, design_data in all_results.items():
        lifetime_cost = design_data["lifetime_cost"]["total"]
        parts = parse_design_key(design_key)
        hours, comp_ratio, frac = float(parts[0]), float(parts[1]), float(parts[3])
        if not is_valid_lifetime_cost(lifetime_cost):
            if is_battery:
                infeasible_hours_max = max(infeasible_hours_max, hours)
            else:
                infeasible_hour_frac_points.append((hours, frac))
        else:
            valid_by_hour.setdefault(round(hours), []).append((comp_ratio, lifetime_cost))
    return infeasible_hour_frac_points, infeasible_hours_max, valid_by_hour


def sample_from_multivariate_normal(
    all_designs,
    designs_per_run,
    o2_range,
    comp_ratio_range,
    charge_power_ratio_range,
    upgrade_key,
    iteration,
):
    """
    Sample new points using the cross-entropy method (hyperparameters at the top of this module).
    """
    # Local random state for consistent sampling from seed, advancing by iteration
    rng = np.random.RandomState(RANDOM_SEED + iteration - 1)

    # Collect infeasible points for boundary-aware clipping
    upgrade_info = get_upgrade_info(upgrade_key)
    infeasible_hour_frac_points, infeasible_hours_max, _ = collect_infeasibility_data(
        all_designs, is_battery=upgrade_info["is_battery"]
    )
    # Design-key dims the distribution is fit over: hours, [comp_ratio (gas only)], charge power, [frac (tanks)]
    if upgrade_info["is_liquid_tank"]:
        free_dims = [0, 2, 3]
    elif upgrade_info["is_battery"]:
        free_dims = [0, 2]
    else:
        free_dims = [0, 1, 2, 3]

    # Extract valid solved_points and Lifetime Costs
    solved_points, lifetime_costs = [], []
    for design_key, design_data in all_designs.items():
        lifetime_cost = design_data["lifetime_cost"]["total"]
        if is_valid_lifetime_cost(lifetime_cost):
            coords = round_point(parse_design_key(design_key))
            solved_points.append([coords[i] for i in free_dims])
            lifetime_costs.append(lifetime_cost)

    solved_points, lifetime_costs = np.array(solved_points), np.array(lifetime_costs)

    # Select elite points
    elite_count = min(ELITE_SAMPLES, len(solved_points))
    elite_indices = np.argsort(lifetime_costs)[:elite_count]
    elite_points = solved_points[elite_indices]
    elite_lifetime_costs = lifetime_costs[elite_indices]

    # Calculate weights
    if len(elite_lifetime_costs) > 1 and np.std(elite_lifetime_costs) > 1e-10:
        normalized_lifetime_costs = (elite_lifetime_costs - np.min(elite_lifetime_costs)) / (
            np.max(elite_lifetime_costs) - np.min(elite_lifetime_costs)
        )
        weights = np.exp(-EXP_WEIGHT * normalized_lifetime_costs)
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(elite_lifetime_costs)) / len(elite_lifetime_costs)

    # Fit distribution (multivariate normal for all types)
    mu = np.average(elite_points, axis=0, weights=weights)
    centered = elite_points - mu
    n_dims = elite_points.shape[1]
    cov = np.zeros((n_dims, n_dims))
    for i in range(len(elite_points)):
        cov += weights[i] * np.outer(centered[i], centered[i])

    # Regularize covariance
    cov += np.eye(n_dims) * MIN_VARIANCE
    mu_sigma = {"mu": mu.tolist(), "sigma": cov.tolist()}

    # Generate new points
    new_points = set()
    for _ in range(designs_per_run * MAX_SAMPLE_ATTEMPTS_PER_DESIGN):
        if len(new_points) >= designs_per_run:
            break

        # Sample from fitted distribution and clip each dim; battery hours also skip the max infeasible hour
        sample = rng.multivariate_normal(mu, cov)
        min_hours = (
            max(o2_range[0], infeasible_hours_max) if upgrade_info["is_battery"] else o2_range[0]
        )
        hours = np.clip(sample[0], min_hours, o2_range[1])
        bounds = {
            1: comp_ratio_range,
            2: charge_power_ratio_range,
            3: (infeasibility_boundary_frac(hours, infeasible_hour_frac_points), 1.0),
        }
        point = [hours, DUMMY_COMP_RATIO, None, 1.0]
        for i, value in zip(free_dims[1:], sample[1:]):
            point[i] = np.clip(value, bounds[i][0], bounds[i][1])
        candidate = round_point(point)

        # Add if not already in all_designs or new_points
        if build_design_key(candidate) not in all_designs and candidate not in new_points:
            new_points.add(candidate)

    return list(new_points), mu_sigma
