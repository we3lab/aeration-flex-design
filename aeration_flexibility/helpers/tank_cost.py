"""Gas and liquid storage tank sizing and cost models, including the smoothed cost envelope fitted in step1."""

import numpy as np
import math
import json
import pandas as pd
import os
from helpers.parameters import *
from helpers.tariffs import to_usd_2025

# Composite (carbon-fiber) vessel installed cost per m3 = floor + slope * pressure (MPa): the slope tracks the lower edge of the
# Shin and Ha 2023 350-700 bar band (https://doi.org/10.3390/en16135233); the pressure-independent floor makes composite
# cross steel near the 10-15 MPa boundary.
COMPOSITE_FLOOR_USD_PER_M3 = 2000.0
COMPOSITE_SLOPE_USD_PER_M3_PER_MPA = 192.0
SMOOTHED_COEFFS_PATH = os.path.join(OUTPUT_DATA_DIR, "smoothed_tank_coefficients.json")
HYDROGEN_TANK_SPECS = pd.read_csv(
    os.path.join(DATA_DIR, "capex", "hydrogen_tank_specifications.csv")
)
# Steel vessel limits: 30 m³ water volume per tank (typical industrial maximum - really could be 10). With L/D <= 5 below,
# a 30 m³ vessel is at most ~10 m long, so no separate length cap is needed.
MAX_TANK_VOLUME_M3 = 30.0  # 2.6 m wide at L/D 2.5, the US truck width limit (23 CFR 658.15), https://www.ecfr.gov/current/title-23/chapter-I/subchapter-G/part-658/section-658.15
MIN_WALL_THICKNESS_M = 0.0015  # ASME BPVC VIII-1 (2023) UG-16(b), 1/16 in
MAX_WALL_THICKNESS_M = 0.032  # ASME BPVC VIII-1 (2023) Table UCS-56-1, no preheat or PWHT
# Length/diameter grid and tank-count tiers, searched until a configuration fits. L/D 2.5-5.0: Couper et al. 2012 Sec 18.1
# (drums); standard horizontal storage tanks there run L/D 1.1-4.5 (Table 18.3)
TANK_SEARCH = {
    "steel": (np.linspace(2.5, 5, 26), ((1, 101), (101, 1001), (1001, 10000))),
    "literature": (np.linspace(2.5, 5, 26), ((1, 101), (101, 1000), (1001, 10000))),
}
smoothed_coefficients_cache = None


def find_tank_configurations(V_tank, P_max, n_tanks, aspect_ratios):
    """Helper function to calculate tank configurations for given parameters.

    Args:
        V_tank: Total tank volume in m³
        P_max: Maximum pressure in MPa
        n_tanks: Number of tanks
        aspect_ratios: Array of aspect ratios to try

    Returns:
        List of (thickness, length, n_tanks, aspect_ratio, radius) tuples for valid configurations
    """
    volume_per_tank = V_tank / n_tanks
    if volume_per_tank > MAX_TANK_VOLUME_M3:
        return []

    configurations = []
    for aspect_ratio in aspect_ratios:
        # Calculate tank radius for cylinder with spherical end caps
        radius = (3 * volume_per_tank / (2 * math.pi * (3 * aspect_ratio - 1))) ** (1 / 3)
        length = aspect_ratio * (2 * radius)

        # Required thickness for circumferential stress: ASME BPVC VIII-1 UG-27(c); thick shells Mandatory Appendix 1-2
        if P_max <= 0.385 * Smax * E:
            t_circumferential = (P_max * radius) / (Smax * E - 0.6 * P_max)
        else:
            t_circumferential = radius * (math.exp(P_max / (Smax * E)) - 1)

        # Calculate required thickness for longitudinal stress
        if P_max <= 1.25 * Smax * E:
            t_longitudinal = (P_max * radius) / (2 * Smax * E + 0.4 * P_max)
        else:
            Z = P_max / (Smax * E) + 1
            t_longitudinal = radius * (math.sqrt(Z) - 1)

        # Required thickness is maximum of the two, within the feasible wall range
        thickness = max(t_circumferential, t_longitudinal)
        if thickness > MAX_WALL_THICKNESS_M:
            continue
        if thickness < MIN_WALL_THICKNESS_M:
            thickness = MIN_WALL_THICKNESS_M

        # Skip configurations where walls are thicker than tank radius (physically impossible)
        if thickness > radius:
            continue

        # Verify the maximum allowable working pressure (MAWP) is sufficient
        if thickness <= radius / 2:
            mawp_circumferential = (Smax * E * thickness) / (radius + 0.6 * thickness)
            mawp_longitudinal = (2 * Smax * E * thickness) / (radius - 0.4 * thickness)
        else:
            mawp_circumferential = Smax * E * math.log((radius + thickness) / radius)
            Z = ((radius + thickness) / radius) ** 2
            mawp_longitudinal = Smax * E * (Z - 1)
        if min(mawp_circumferential, mawp_longitudinal) < P_max:
            continue

        configurations.append((thickness, length, n_tanks, aspect_ratio, radius))

    return configurations


def get_tank_cost(V_tank, P_max, method="steel"):
    """Calculate cost of gas storage tank system using ASME pressure vessel code.
    http://docs.codecalculation.com/mechanical/pressure-vessel/thickness-calculation.html

    method: "steel" (ASME steel materials + labor) or "literature" (Houssainy et al. vendor-data
    pressure-factor model). Loops through tank counts and aspect ratios to find minimum cost.

    Args:
        V_tank: Maximum volume in m3
        P_max: Maximum pressure in MPa

    Returns:
        Tuple of (cost, number of tanks, wall thickness, length, aspect ratio)
    """
    aspect_ratios, tank_count_tiers = TANK_SEARCH[method]
    configurations = []
    for low, high in tank_count_tiers:
        for n_tanks in np.arange(low, high):
            configurations.extend(find_tank_configurations(V_tank, P_max, n_tanks, aspect_ratios))
        if configurations:
            break
        print(f"No valid configuration found with {low}-{high - 1} tanks, trying more tanks...")
    else:
        print(
            f"No valid tank configuration found for V_tank={V_tank:.2f} m3, P_max={P_max:.2f} MPa"
        )
        return np.nan, 0, 0, 0, 0

    costed_configurations = []
    for thickness, length, n_tanks, aspect_ratio, radius in configurations:
        if method == "steel":
            surface_area = 4 * aspect_ratio * math.pi * radius**2
            steel_cost_per_tank = surface_area * thickness * cost_per_m3
            manufacturing_cost_per_tank = surface_area * labor_per_m2 + labor_per_tank
            cost_per_tank = (manufacturing_cost_per_tank + steel_cost_per_tank) * (1 + markup)
            cost = cost_per_tank * n_tanks
        else:
            pressure_factor = max(
                (P_max * (2 * radius) / (2 * (Smax * E - 0.6 * P_max)) + 0.00315) / 0.063,
                1.0,
            )
            cost = 1000.0 * n_tanks * (V_tank / n_tanks) * pressure_factor
        costed_configurations.append((cost, n_tanks, thickness, length, aspect_ratio))

    # Find the configuration with minimum cost
    return min(costed_configurations, key=lambda x: x[0])


def get_tank_cost_literature_by_vol(V_tank):
    # https://www.sciencedirect.com/science/article/pii/S2352152X23030438?via%3Dihub%27=#bb0240 (Table 4), from
    # https://www.sciencedirect.com/science/article/pii/S2352152X22004273?via%3Dihub (Table 3)
    return 4042 * V_tank**0.506


def get_tank_cost_composite(V_tank, P_max):
    """Installed cost of a carbon-fiber composite storage vessel for pressures above 10 MPa,
    where all-steel vessels become impractically thick and need many small tanks.

    Args:
        Vtank: Tank volume in m3
        Pmax: Maximum pressure in MPa

    Returns:
        Tuple of (cost, number of tanks, wall thickness, length, aspect ratio)

    Cost per m3 is affine in pressure: an increasing term from the composite material
    required (wall thickness ~pressure) plus a pressure-independent manufacturing floor
    (liner, layup, boss, labor). The pressure slope tracks the lower edge of the Shin
    2023 350-700 bar Type IV band (~$192/m3 per MPa); at 70 MPa this gives ~15,400 USD/m3,
    inside that band. The floor makes composite realistically more expensive than steel at
    low pressure, so a min(steel, composite) envelope crosses where the two are actually
    equal (~10-15 MPa) rather than at a hard-coded pressure.
      - Shin and Ha 2023: ~6,800-23,100 USD/m3 (2023) for 350-700 bar Type IV vessels
      - DOE Type IV onboard status 2024: ~15-19 USD/kWh
        https://www.hydrogen.energy.gov/docs/hydrogenprogramlibraries/pdfs/24006-onboard-storage-cost-performance-status.pdf
    Per-tank geometry (count, thickness, length, aspect ratio) is read from the ANL
    hydrogen-tank specifications so the returned dimensions stay physical.
    """
    # Installed composite-vessel cost per m3 (2025 USD): manufacturing floor + pressure-
    # driven composite material. Cost is independent of the tank count. See docstring.
    total_cost = (COMPOSITE_FLOOR_USD_PER_M3 + COMPOSITE_SLOPE_USD_PER_M3_PER_MPA * P_max) * V_tank

    # Reported geometry (cosmetic): tanks of at most 75 m3, dimensions from the largest ANL tank (inches: m), wall thickness
    # scaled with pressure. This does not affect cost, so it never fails for large volumes.
    n_tanks = max(1, math.ceil(V_tank / 75.0))
    largest_tank = HYDROGEN_TANK_SPECS.loc[HYDROGEN_TANK_SPECS["Internal Volume (DGE)"].idxmax()]
    length = largest_tank["Length (inches)"] * 0.0254
    aspect_ratio = length / (largest_tank["Diameter (inches)"] * 0.0254)
    thickness = (largest_tank["ANL Comp THK (mm)"] * P_max / 70.0) / 1000.0
    return total_cost, n_tanks, thickness, length, aspect_ratio


def quadratic_cost_function(V, a, b, c):
    """Quadratic cost function: Cost = a*V^2 + b*V + c"""
    return a * V**2 + b * V + c


def get_tank_cost_smoothed(V_tank, P_max):
    """Get tank cost and metrics using smoothed function (for use in calculate_capex).

    Args:
        V_tank: Tank volume in m³
        P_max: Maximum pressure in MPa

    Returns:
        Tuple of (cost, n_tanks, thickness, length, aspect_ratio)
    """
    global smoothed_coefficients_cache
    if smoothed_coefficients_cache is None:
        with open(SMOOTHED_COEFFS_PATH, "r") as f:
            smoothed_coefficients_cache = {
                float(pressure): {k: np.array(v) for k, v in coefficients.items()}
                for pressure, coefficients in json.load(f).items()
            }

    # Coefficients are fit to the min(steel, composite) envelope, so the steel->composite
    # transition is already baked in; just pick the nearest sampled pressure.
    coeffs = smoothed_coefficients_cache[
        min(smoothed_coefficients_cache.keys(), key=lambda p: abs(p - P_max))
    ]

    cost, n_tanks, thickness, length, aspect_ratio = (
        quadratic_cost_function(V_tank, *coeffs[k])
        for k in ("cost", "n_tanks", "thickness", "length", "aspect_ratio")
    )
    return cost, max(1, round(n_tanks)), thickness, length, aspect_ratio


def get_liquid_tank_cost(Nmax, method="average", gas="o2"):
    """Liquid tank cost from a materials-based and a Wei et al. 2012 empirical model.

    method: "average" (default, used in optimization), "materials", or "empirical"
    (the latter two let step1 plot the two methods separately in figS5).
    gas: "o2" (LOX) or "air" (liquid air). Nmax is in mol O2-equivalent; storing air means
    banking the whole air stream for its 20.9% O2, so the liquid volume per O2-equivalent
    mol is ~5.7x LOX (M_air/frac_o2_air at 870 kg/m3 vs M_O2 at 1141 kg/m3).

    Returns:
        Tuple of (cost, n_tanks, volume_per_tank_m3)
    """
    if gas == "air":
        liquid_volume_m3 = moles_to_mass(Nmax / frac_o2_air, M_air) / rho_liquid_air
    else:
        # divide mass by density of liquid oxygen
        liquid_volume_m3 = moles_to_mass(Nmax, M_O2) / rho_hpo_liquid
    liquid_volume_m3 = max(liquid_volume_m3, min_liquid_tank_volume_m3)  # enforce minimum tank size

    # Split into equal tanks of at most max_liquid_tank_volume_m3 each
    n_tanks = math.ceil(liquid_volume_m3 / max_liquid_tank_volume_m3)
    volume_per_tank = liquid_volume_m3 / n_tanks
    radius = (3 * volume_per_tank / (4 * math.pi)) ** (1 / 3)

    surface_area = 4 * math.pi * radius**2

    # thickness 10 mm, 4.0x markup since cylinder is conservative, double-walled insulation
    V_steel = (surface_area * 0.01 * 4.0) * 2.0
    labor = 4 * (labor_per_m2 * surface_area + labor_per_tank)  # labor 4x the gas-tank rate

    cost_materials = (V_steel * cost_per_m3 + labor) * (1 + markup) * n_tanks

    # Wei et al. 2012 (Energy 42:424-433) empirical vessel cost model, fM=1 (stainless steel), P=0.5 MPa for cryogenic storage:
    # C = 1780 * L^0.87 * d^1.23 * (2.86 + 1.694*fM*(10.01 - 7.408*ln(P) + 1.395*(ln P)^2))
    P_liq = 0.5
    diameter = 2 * radius
    length = volume_per_tank / (math.pi * radius**2)
    wei_vessel_price = (
        1780
        * length**0.87
        * diameter**1.23
        * (2.86 + 1.694 * 1.0 * (10.01 - 7.408 * math.log(P_liq) + 1.395 * math.log(P_liq) ** 2))
    )
    cost_empirical = to_usd_2025(wei_vessel_price, 2011) * n_tanks

    if method == "materials":
        cost = cost_materials
    elif method == "empirical":
        cost = cost_empirical
    else:
        cost = (cost_materials + cost_empirical) / 2
    return (cost, n_tanks, volume_per_tank)
