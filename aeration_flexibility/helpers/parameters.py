"""Shared constants, cost assumptions and repo paths. Machine-specific settings load from .env."""

import os

import numpy as np
from dotenv import load_dotenv

# Paths resolve from this file, so scripts run from any working directory
AERATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(AERATION_DIR)
CONFIG_DIR = os.path.join(AERATION_DIR, "configs")
DATA_DIR = os.path.join(AERATION_DIR, "data")
OUTPUT_DATA_DIR = os.path.join(AERATION_DIR, "output_data")
OUTPUT_PLOTS_DIR = os.path.join(AERATION_DIR, "output_plots")
SI_PLOTS_DIR = os.path.join(OUTPUT_PLOTS_DIR, "si")
CONFIG_PLOTS_DIR = os.path.join(OUTPUT_PLOTS_DIR, "diagnostics")

# Machine-specific settings from <repo>/.env (see .env.example); variables already set in the shell win
load_dotenv(os.path.join(REPO_DIR, ".env"))
# Shared per-day solve cache. The paper runs write millions of small files, so it can live on another disk.
INTERMEDIATE_DIR = os.environ.get(
    "AERATION_INTERMEDIATE_DIR", os.path.join(OUTPUT_DATA_DIR, "intermediate")
)
SOLVER = os.environ.get("AERATION_SOLVER", "gurobi").lower()  # "gurobi" or "highs"

# Run settings
RANDOM_SEED = 4

# Dry season: May 1 – September 30 (months 5–9); dry-season days get the summer O2 multiplier
DRY_SEASON_MONTHS = frozenset(range(5, 10))

# Timesteps
timestep_factor = 60 / 15  # 15-minute timesteps per hour

# Constants
R = 8.3144621  # Gas constant, (J/K*mol)
M_O2 = 32  # molecular weight of O2 (g O2 / mol O2)
M_H2 = 2  # molecular weight of H2 (g H2 / mol H2)
n_air = 1.4  # ratio of specific heats for air
n_o2 = 1.4  # ratio of specific heats for oxygen
P_ATM = 0.101325  # Atmospheric pressure (MPa)

MGD_TO_M3_PER_DAY = 3785.41  # 1 MGD = 3785.41 m^3/day
HRT_DAYS = 0.25  # 6 hours

# Conversions
KW_TO_HP = 1.34102209
Pa_per_MPa = 1e6
mol_per_hr_to_ton_per_annum = (24 * 365) * (M_O2 / 1000000)

# BOD and nitrogen load and removal: paper defaults for moles_o2_required() (SI eq. S1-S2)
f_BOD_primary = 0.35  # fraction of BOD remover in inlet works + primary clarifiers
BOD_in = 300  # mg/L BOD5
BOD_out = 20  # Effluent limits (mg/L BOD5)
epsilon_BOD = 1.1  # kg O2 required to oxidize 1 kg BOD
f_N_primary = 0.35  # fraction of N removed in inlet works + primary clarifiers
N_in = 50  # N load in influent (mg/L)
N_out = 40  # effluent TN target (mg/L)
epsilon_N = 4.6  # kg O2 required to oxidize 1 kg NH3 to NO3 (Metcalf & Eddy 2012)

# Water properties
rho_hpo_liquid = 1141  # density of liquid oxygen (kg/m3)

# Air and aeration system properties
frac_o2_air = 0.21  # Fraction of O2 in air
frac_o2_hpo = 0.95  # Fraction of O2 in high purity oxygen
alpha = 0.8  # correction factor to convert from clean water SOTE to wastewater
beta = 0.9  # assumption for any additional losses or fouling of diffusers
SOTE = {"Disc": 0.7, "Invent": 0.8}  # standard oxygen transfer efficiency (%)

o2_multiplier_map = {"air": 1, "o2": 0.7 / 0.6 / SOTE["Invent"]}

# Blower/compressor efficiency parameters — used in comp_lp_coeff()
eta = 0.75  # mechanical (blower + motor) efficiency ηm; ×ηp = 0.60 overall, inside Sanitaire design guide 60-80% (70% typical)
polytropic_efficiency = 0.8  # isentropic path efficiency (ηp) in polytropic compression formula

# Storage compressors are multi-stage with interstage cooling (single-stage is impractical above ~5x: discharge
# temperature/material limits); stages are sized from this conservative maximum per-stage pressure ratio.
max_stage_pressure_ratio = 5.0

# Aeration pressure assuming disc diffusers at 18 ft submergence plus diffuser losses
P_AER = 0.1  # MPa gauge discharge pressure for disc diffuser aeration
T_room = 20 + 273.15  # temperature in kelvin
P_MAX_H2 = 35.0  # MPa

# Annual O&M as a fraction of storage-equipment capex
battery_om_frac_per_year = 0.04 # NLR ATB 2025
storage_om_frac_per_year = 0.015  # Bolorinos 2023

# Minimum battery array power
battery_min_power_kw = 100.0  # NLR ATB 2025 commercial range minimum
battery_max_power_kw = 2000.0  # NLR ATB 2025 commercial range maximum
battery_max_usable_hours = 8.0  # NLR ATB 2025 commercial range maximum

# Battery standing self-discharge (fraction of stored energy lost per day). 0.17%/day for Li-ion,
# Kebede et al. 2021 (Renew. Sustain. Energy Rev. 159:112213) — same value used by Bolorinos et al. 2023.
battery_self_discharge_per_day = 0.0017

# Usable fraction of nameplate battery energy (80% depth of discharge). NREL prices usable kWh (83586 Table 9:
# "rated (usable)", 10-90% SOC), so capex prices only this fraction of E_max.
battery_usable_frac = 0.8

# Controller forecast error: per-timestep std of the net-demand prediction error as a fraction of net demand (Bolorinos et al.
# 2023 eqs S16-S18 use 0.02), applied equally to baseline and optimized billed demand. 0.0 = perfect foresight.
forecast_error_std = 0.02

ei_psa_kg = 0.44  # kWh/kg O2. OSI Oxygen VPSA white paper (2022) https://osioxygen.com/wp-content/uploads/2022/03/VPSA-White-Paper.pdf
# cryo OTR (2.5 lb O2/hp-h) minus the dissolution-only LOX OTR (6.5)
ei_cryo_kg = (1 / 2.5 - 1 / 6.5) * 0.7457 / 0.4536  # kWh/kg O2. EPA 430/9-78-009 Fact Sheets 2.1.4-2.1.5. Shammas & Wang 2009 Figs 7.2/7.6
ei_elec_kg = 6.25  # kWh/kg O2. IRENA 2020 Table 6 PEM stack range 47-66 kWh/kg H2 / 8
ei_elec_mol = ei_elec_kg / 1000 * M_O2  # kwh/mol O2
electrolyzer_max_power_kw = 20000.0 # EPRI's PEM capex
# Feedwater purification energy, Winter et al 2022 (SI eq S29); energy-only (purification capex negligible).
f_pretreat = 0.02
ei_psa_mol = ei_psa_kg / 1000 * M_O2  # kWh/mol O2
ei_cryo_mol = ei_cryo_kg / 1000 * M_O2  # kWh/mol O2

cryo_turndown = 0.1  # internal engineering estimate (no external source)
compressor_turndown = 0.5  # O2 GENERATION (blower/PSA) turndown floor
storage_compressor_min_turndown = 0.8  # when running

# 14.83 KW/MGD from Howard Curren plant https://www.tampa.gov/sites/default/files/content/files/migrated/hfc_awtp_-_master_plan_-_volume_1_0.pdf.
# VPSA turns down to ~50% and uses 30-70% less electricity than cryo: Linde https://www.linde-engineering.com/products-and-services/process-plants/adsorption-and-membrane-plants/powering-industry-with-on-site-o2-generation, OSI https://osioxygen.com/wp-content/uploads/2022/03/VPSA-White-Paper.pdf

price_h2_kg = 5.0  # $/kg

# Capex
cost_per_m3 = 10000  # material cost in $/m^3. US standard plate Mar-Dec 2025 avg $1,239/t x 7,850 kg/m3, https://steelbenchmarker.com/history.pdf
labor_per_m2 = 100  # labor cost per m2 constructed
labor_per_tank = 2000  # labor cost per finished tank
markup = 0.5  # markup and overhead. Zhang et al. 2012 (ORNL) fabricator contingency + profit ~52% of direct cost, https://info.ornl.gov/sites/publications/files/pub41882.pdf
Smax = 20000 * 0.00689476  # Maximum Allowable Stress (MPa)
E = 1.0  # Joint efficiency: seamless or fully radiographed shells, ASME VIII-1 (2023) Table UW-12 Type 1

# Mappings
frac_o2_map = {"air": frac_o2_air, "o2": frac_o2_hpo}

n_map = {
    "air": n_air,
    "o2": n_o2,
}

ei_o2_map = {
    "psa": ei_psa_mol,
    "cryo": ei_cryo_mol,
    "elec": ei_elec_mol * (1 + f_pretreat),  # +2% feedwater pretreatment (Winter et al, SI S29)
    "compressor": 0,
}

# O2 latent heat of vaporization, per-molecule (diatomic). Element tables list 3.4099 kJ/mol per
# ATOM of O, so O2 is 2x that (~213 kJ/kg). NOAA CAMEO https://cameochemicals.noaa.gov/chris/OXY.pdf
ei_evap_kj = 6.8198  # kJ/mol
ei_evap_kwh = ei_evap_kj / 3600  # kWh/mol (1 kWh = 3600 kJ); ~0.059 kWh/kg

# Liquefying gaseous surplus O2 (PSA/electrolyzer) for bulk liquid storage: ~0.2-0.4 kWh/kg in the liquid-air energy storage
# literature; 0.30 (below full cryo separation at 0.405, no separation step), applied to the flow charged into liquid storage.
ei_liquefy_kg = 0.30  # kWh/kg O2
ei_liquefy_mol = ei_liquefy_kg / 1000 * M_O2  # kWh/mol O2

# Liquid AIR storage (air-blower family): ~870 kg/m3 at boiling point (https://en.wikipedia.org/wiki/Liquid_air); air liquefaction spans 0.22-0.7
# kWh/kg (~0.2-0.3 typical LAES, Vecchi et al. 2021 https://www.sciencedirect.com/science/article/pii/S2666792421000391), so 0.30 like the LOX liquefier.
rho_liquid_air = 870  # kg/m3
M_air = 28.96  # g/mol
ei_liquefy_air_kg = 0.30  # kWh/kg air

# Liquefaction energy per mol of O2-EQUIVALENT flow charged into bulk liquid storage: storing air liquefies the whole air
# stream to bank its 20.9% O2, so the per-O2 burden is ~4.8x the mol-air value.
ei_liquefy_mol_map = {
    "o2": ei_liquefy_mol,
    "air": ei_liquefy_air_kg / 1000 * M_air / frac_o2_air,
}

# Regasifying stored liquid: ambient-air vaporizers supply the latent heat at ~no electric cost (standard LOX practice), so all 0.
# For ELECTRIC regas set entries to ei_evap_kwh, keyed by storage medium (it applies to any liquid storage, not the base o2_tech).
ei_evap_map = {
    "psa": 0,
    "cryo": 0,
    "elec": 0,
    "compressor": 0,
}

P_init_map = {
    "psa": P_ATM * 1.1,
    "cryo": P_ATM,
    "elec": P_ATM * 1.5,
    "compressor": P_ATM,
}

# Maximum compression ratio by gas type
max_comp_ratio_map = {
    "air": 680,  # 68.9 MPa. ASME VIII-2 Sec. 1.2.2 has no pressure limit but points to Div. 3 above 68.95 MPa (10,000 psi)
    "o2": 80,  # P_max 8.9 MPa PSA, 8.1 cryo, 12.2 elec (elec clipped by the cap below)
}

max_o2_gas_pressure_mpa = 10.0 # Assuming steel-only. Composite O2 vessels are small (https://www.luxfercylinders.com/product/l6x-composite-cylinder-medical/)
max_o2_gas_storage_mol = 342 / 0.022414  #OSHA 1910.104 bulk oxygen system threshold, 13,000 ft³ at NTP = 342 Nm³

min_liquid_tank_volume_m3 = 5.7  # Smallest bulk size (m3) per Chart Bulk Storage Catalog (2026) VS/HS-1500SC, https://files.chartindustries.com/13608592_BulkCatalog.pdf
max_liquid_tank_volume_m3 = 35.4  # largest bulk size (m3) tank water volume. Chart Bulk Storage Catalog VS/HS-9000SC, 15,000 gal

boiloff_frac_per_day = 0.0035  # mid-to-upper range. https://www.dswgascylinder.com/evaporation-rate-of-a-cryogenic-liquid/

# Duck curve for solar: daylight hours (6am to 6pm), bell curve peaking at noon
solar_hours = np.arange(96) / 4
hourly_solar_multiplier_default = np.where(
    (6 <= solar_hours) & (solar_hours < 18), np.sin((solar_hours - 6) * np.pi / 12), 0.0
)


def get_intermediate_dir(data_name, config_name, horizon_days):
    """Cached day solves for one config and horizon. Keyed by input dataset as well as config, since configs on
    different data can share a name; runs on the same data (e.g. fig4 seeding from fig2) share solves.
    """
    # Only the horizon this run was configured with — other h-dirs under the same config
    # belong to other runs (e.g. the 31-day variant) and must not leak into its profiles
    return os.path.join(INTERMEDIATE_DIR, data_name, config_name, f"h{horizon_days}")


def is_dry_season(month):
    """Return True if month (int 1–12) falls in the dry season (May–September)."""
    return int(month) in DRY_SEASON_MONTHS


def volume_to_moles_stp(volume):
    """Calculate moles of air from volume (m3) based on ideal gas law"""
    return P_ATM * Pa_per_MPa * volume / (R * T_room)


def moles_to_mass(moles, M):
    """Calculate mass (kg) of gas from moles based on ideal gas law"""
    return moles * M / 1000


def apply_temporal_transform(x, gamma, scale=1.0):
    x = np.asarray(x)

    if gamma == 1.0 and scale == 1.0:
        return x  # No transformation needed

    # Mean-normalize to get dimensionless diurnal shape r(t)
    mu = np.mean(x)
    r = x / mu  # mean of r(t) is 1.0

    # Apply power transform, handling values that are very small
    r_gamma = np.power(np.maximum(r, 0), gamma)

    # Renormalize to preserve original mean
    r_tilde = r_gamma / np.mean(r_gamma)

    # Reconstruct with original mean and optional scale
    x_gamma = mu * r_tilde * scale

    return x_gamma


def find_gamma_for_peaking_target(x, peaking_multiplier, tol=1e-6, max_iter=50):
    # Find gamma so the transformed peak excess (max/mean - 1) is peaking_multiplier x the original's (on the excess, since
    # max/mean >= 1 makes ratio scaling unattainable on flat days): < 1 flattens peaks AND troughs, > 1 sharpens, 1.0 = unchanged.
    if peaking_multiplier == 1.0:
        return 1.0
    x = np.asarray(x)
    peaking_orig = np.max(x) / np.mean(x)
    target_peaking = 1.0 + peaking_multiplier * (peaking_orig - 1.0)

    lo, hi = 0.01, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        transformed = apply_temporal_transform(x, gamma=mid)
        if np.max(transformed) / np.mean(transformed) < target_peaking:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    gamma = (lo + hi) / 2

    # Bisection returns a bracket bound when the target is outside [0.01, 5.0]'s reach (silently weakening the perturbation);
    # the tolerance is 1% of the requested swing change, so the check scales with how much the day actually swings.
    transformed = apply_temporal_transform(x, gamma=gamma)
    achieved = np.max(transformed) / np.mean(transformed)
    swing_change = abs(target_peaking - peaking_orig)
    if swing_change > 0 and abs(achieved - target_peaking) > 0.01 * swing_change:
        print(
            f"[PEAKING] target max/mean {target_peaking:.4f} unreachable at multiplier "
            f"{peaking_multiplier:g}: gamma={gamma:.4f} gives {achieved:.4f} "
            f"(original {peaking_orig:.4f})"
        )
    return gamma
