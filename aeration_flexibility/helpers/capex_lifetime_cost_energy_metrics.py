"""Capital costs of storage tanks, O2 generation, compressors, batteries and supplemental aeration."""

from helpers.parameters import *
from helpers.config_keys import split_key, get_upgrade_info
from helpers.tariffs import to_usd_2025
from helpers.tank_cost import get_tank_cost_smoothed, get_liquid_tank_cost

# O2 generation capex per ton-per-annum of capacity: (USD, source year)
O2_GENERATION_COST_PER_TPA = {
    "psa": (208, 2019),  # Tampa Bay Water Water Master Plan (Table 7.5-1)
    # https://thundersaidenergy.com/downloads/cryogenic-air-separation-the-economics/
    "cryo": (200, 2020),  # Thunder Said Energy cryogenic air separation economics
}


def cost_o2_generation(flowrate, o2_tech="cryo"):
    """O2 generation capex (PSA or cryo) for a capacity in mol/hr, in 2025 USD.

    Cryo cross referenced with CLINTON HEALTH ACCESS INITIATIVE guide 2021
    US$25 million for a 200-metric ton per day plant ($342/tpa)
    US$125 million for a 3,000 ton per day plant ($114/tpa)
    Flat $200/tpa is a fixed-plant-scale simplification (the cited $342->$114/tpa scaling not modeled).
    """
    price, year = O2_GENERATION_COST_PER_TPA[o2_tech]
    return mol_per_hr_to_ton_per_annum * flowrate * to_usd_2025(price, year)


def cost_blower(power):
    """Low-pressure blower cost for aeration (~8-10 psig).
    Coefficients are a fit of blower_cost_data.csv (step1_fit_capex_costs.py::plot_blower_costs),
    already converted to 2025 USD via to_usd_2025. Cost in 2025 USD, power in kW.
    """
    return 9278.5 * power**0.642


def cost_compressor(power_kw):
    """Multi-stage reciprocating compressor for high-pressure gas storage.

    Base from Matche equipment cost data: Cost [USD] = 1880 * (power [hp] ^ 0.671).
    The 1.3x factor is an assumed oil-free premium (applied to air storage too): O2 compressors
    must keep crankcase oil out of the gas stream (AIGA 048/18 Sec. 4.3.2), and lubricant-free
    reciprocating compressors "usually have higher first costs" (Compressed Air Challenge Fact
    Sheet #1, 2026).
    https://asiaiga.org/uploaded_docs/en_AIGA_048_18_Reciprocating_Compressors_for_Oxygen_Service.pdf
    https://www.compressedairchallenge.org/data/sites/1/media/library/factsheets2026/fact-sheet-1-assessing-compressed-air-needs_updated2026.pdf
    """
    power_hp = power_kw * KW_TO_HP
    cost = to_usd_2025(1880 * power_hp**0.671, 2014)
    return 1.3 * cost


def cost_vfd(param_vals):
    """Bare variable-frequency-drive cost for the storage compressor, scaled by drive power.

    Returns bare equipment cost only; calculate_capex applies the construction markup to
    this along with the other equipment. Bare ~60 USD/kW x the 4.18x markup lands at
    ~250 USD/kW installed, matching Thunder Said Energy's 15-project VFD average and WWTP
    VFD-retrofit benchmarks (~240-320 USD/kW installed).
    https://thundersaidenergy.com/downloads/variable-frequency-drives-the-economics/
    """
    power_kw = param_vals["Edot_c_max"] or param_vals["Edot_r_max"]
    return to_usd_2025(60.0 * power_kw, 2024)


def liquid_mass_scale(param_vals):
    """Mass-throughput scale for liquid-air machinery, relative to the pure-O2 basis.

    cost_o2_generation prices on O2 tonnage (mol x M_O2). Flows are tracked as O2-equivalent
    moles, so liquefying/evaporating AIR handles M_air/frac_o2_air of mass per
    O2-equivalent mol (~4.3x the M_O2 basis). frac_o2 identifies the gas (air=0.21, o2=1).
    """
    frac_o2 = param_vals["frac_o2"]
    return M_air / (frac_o2 * M_O2) if frac_o2 < 0.9 else 1.0


def cost_evaporator(param_vals):
    """
    7% of heat exchanger costs, which are 54% of cryo system cost
    (Young et al. 2021, Ind. Eng. Chem. Res. 60:14830, Fig. 5)
    """
    return (
        cost_o2_generation(param_vals["Ndot_r_max"] * liquid_mass_scale(param_vals)) * 0.54 * 0.07
    )


def cost_battery(param_vals):
    """
    NREL ATB 2025 installed cost for commercial-scale (100 - 2000 kW) battery storage, 2024 cost year in
    2023 USD: 277.47 $/kWh x usable kWh + 504.07 $/kW x kW (the ATB 2025 constant term is 0).
    https://atb.nlr.gov/electricity/2025/commercial_battery_storage
    Data: https://data.openei.org/submissions/8759 (2025 ATB workbook, 'Commercial Battery - R&D' sheet)
    """
    # NREL $/kWh is per usable kWh, so price only the usable part of nameplate E_max
    usable_kwh = battery_usable_frac * param_vals["E_max"]
    cost = to_usd_2025(277.47 * usable_kwh + 504.07 * param_vals["Edot_c_max"], 2023)
    # Power/energy passed through so the lifetime-cost step can also price the battery under Bolorinos'
    # capex curve for comparison (see get_itemized_lifetime_cost).
    return cost, {
        "battery_power_kw": param_vals["Edot_c_max"],
        "battery_energy_kwh": param_vals["E_max"],
    }


def cost_gas_or_liquid_tank(param_vals, max_values):
    """Cost of the bulk storage medium chosen pre-solve (param_vals['is_liquid_storage']).

    A gas_tank upgrade stores its surplus gas either as compressed gas (gas_tank +
    compressor + vfd) or as bulk liquid (liquefier + liquid_tank + evaporator + vfd).
    Each design point is solved under BOTH media with matching operational physics and the
    cheaper lifetime cost wins (see build_job_queue_for_design_keys), so this capex just
    prices whichever medium param_vals carries.
    """
    vfd = cost_vfd(param_vals)

    if param_vals["is_liquid_storage"]:
        gas = "air" if param_vals["frac_o2"] < 0.9 else "o2"
        liquid_tank, n_tanks, volume_per_tank = get_liquid_tank_cost(param_vals["N_max"], gas=gas)
        # Liquefier: ~50% of cryo ASU cost, about its heat-exchanger share (54% of equipment cost,
        # Young et al. 2021 Ind. Eng. Chem. Res. 60:14830 Fig. 5); liquefaction skips distillation
        liquefied_flow = max_values["Ndot_b_excess"] + max_values["Ndot_c"]
        liquefier = cost_o2_generation(liquefied_flow * liquid_mass_scale(param_vals)) * 0.5
        evaporator = cost_evaporator(param_vals)
        cost = liquid_tank + liquefier + evaporator + vfd
        return cost, {
            "n_tanks": n_tanks,
            "volume_per_tank_m3": volume_per_tank,
            "total_volume_m3": n_tanks * volume_per_tank,
            "tank_type": "liquid_air" if gas == "air" else "liquid_from_gas",
        }

    gas_tank, n_tanks, _, length, aspect_ratio = get_tank_cost_smoothed(
        param_vals["V_tank"], param_vals["P_max"] - P_ATM
    )
    cost = gas_tank + cost_compressor(param_vals["Edot_c_max"]) + vfd
    return cost, {
        "n_tanks": n_tanks,
        "volume_per_tank_m3": param_vals["V_tank"] / n_tanks,
        "total_volume_m3": param_vals["V_tank"],
        "tank_type": "gas",
        "length": length,
        "aspect_ratio": aspect_ratio,
    }


def calculate_addtl_o2_capacity_capex(
    additional_o2_capacity, o2_supply_tech, Edot_b_comp_coeff, frac_o2
):
    if additional_o2_capacity <= 0:
        return 0

    if o2_supply_tech == "compressor":
        equipment_cost = cost_blower(additional_o2_capacity * Edot_b_comp_coeff / frac_o2)
    else:
        equipment_cost = cost_o2_generation(additional_o2_capacity, o2_supply_tech)

    return construction_engineering_markup(equipment_cost) + equipment_cost


def construction_engineering_markup(equipment_cost):
    """Apply construction + engineering markup based on sampled WWTP project data (SI Table S2).

    Percentages apply to nested bases, matching the Table S2 column definitions:
      - Additional materials/installation: 125% of primary equipment cost
      - Construction management:            53% of total material cost (equipment + materials)
      - Engineering + permitting:           29% of construction cost
      - Contingency + overhead + profit:    33% of construction cost
    Cascade gives a total installed cost of 4.18x equipment; this returns the added
    markup only (3.18x), since callers add it on top of the base equipment cost.
    """
    additional_material = 1.25 * equipment_cost
    total_material = equipment_cost + additional_material
    construction = 0.53 * total_material
    engineering = 0.29 * construction
    contingency = 0.33 * construction
    return additional_material + construction + engineering + contingency


def calculate_capex(param_vals, tariff_key, max_values, upgrade_key, base_wwtp_key):
    """Calculate CapEx for a given configuration.
    Two categories:
      1. Storage upgrade CAPEX (storage equipment + engineering markup):
         Equipment for the energy flexibility-related system components
      2. Capacity Expansion CapEx (capex_addtl_capacity):
         Additional capacity to meet O2 demand not covered by storage release.

    Counterfactual: equivalent of capex_addtl_capacity for NO storage: expansion_only_capex
    """
    upgrade_info = get_upgrade_info(upgrade_key)

    # Storage hardware. compressor__gas_tank routes through cost_gas_or_liquid_tank too: its gas branch
    # prices gas_tank + compressor + vfd and its liquid branch prices liquid AIR, so the media compete.
    tank_metrics = {}
    if upgrade_info["is_battery"]:
        equipment_cost, tank_metrics = cost_battery(param_vals)
    elif upgrade_info["is_liquid_tank"]:
        liquid_tank, n_tanks, volume_per_tank = get_liquid_tank_cost(param_vals["N_max"])
        equipment_cost = liquid_tank + cost_evaporator(param_vals) + cost_vfd(param_vals)
        tank_metrics = {
            "n_tanks": n_tanks,
            "volume_per_tank_m3": volume_per_tank,
            "total_volume_m3": n_tanks * volume_per_tank,
            "tank_type": "liquid",
        }
    else:
        equipment_cost, tank_metrics = cost_gas_or_liquid_tank(param_vals, max_values)
        if upgrade_info["has_elec"]:
            # Electrolyzer: EPRI TEA electrolysis CAPEX, PEM higher case <20 MW, 2830 USD/kW (2022) plant capex for stack + balance of plant
            # + compressor: https://apps.epri.com/lcri-electrolysis-tea/en/capex-rates.html. PEM over alkaline for its 5-120% load range
            # (alkaline 15-100%, IRENA 2020 Table 6), matching the LP's free turndown.
            equipment_cost += max_values["Ndot_c"] * ei_elec_mol * to_usd_2025(2830, 2022)
            # H2 storage, 450 USD/kg (2023): https://www.hydrogen.energy.gov/docs/hydrogenprogramlibraries/pdfs/review23/st235_houchins_2023_p-pdf.pdf?Status=Master
            h2_mass_kg = moles_to_mass(param_vals["N_max"] * param_vals["y_h2"], M_H2)
            equipment_cost += to_usd_2025(450 * h2_mass_kg, 2023)

    # Solar scales linearly with the tariff prefix's solar fraction (the operational model offsets solar_frac x
    # charging power). NREL installed cost per kW DC: 1761 community minus 50 commercial, x 1.21 inverter loading.
    # https://docs.nrel.gov/docs/fy23osti/87303.pdf
    solar_frac = float(tariff_key.split("__")[0])
    solar_cost = solar_frac * param_vals["Edot_c_max"] * to_usd_2025(1761 - 50, 2023) * 1.21

    # Engineering markup — applied to sourced equipment only. Battery (NREL ATB installed cost) and
    # solar (NREL Q1-2023 modeled-market benchmark) are already fully-installed prices, so both bypass it.
    if upgrade_info["is_battery"]:
        construction_and_management = 0
    else:
        construction_and_management = construction_engineering_markup(equipment_cost)

    base_equipment_cost = equipment_cost + solar_cost

    # Capacity Expansion CapEx: additional installed O2 capacity from the BASE WWTP's tech (blower/PSA/cryo)
    base_o2_supply_tech = split_key(base_wwtp_key)[1]
    capex_addtl_capacity = calculate_addtl_o2_capacity_capex(
        param_vals["addtl_capacity_max"],
        base_o2_supply_tech,
        param_vals["Edot_b_comp_coeff"],
        param_vals["frac_o2"],
    )

    total_capex = base_equipment_cost + construction_and_management + capex_addtl_capacity

    # Counterfactual expansion-only capex
    expansion_only_capex = calculate_addtl_o2_capacity_capex(
        float(param_vals["Ndot_target_max"]) - float(param_vals["Ndot_b_max"]),
        base_o2_supply_tech,
        param_vals["Edot_b_comp_coeff"],
        param_vals["frac_o2"],
    )

    capex_components = {
        "base_equipment_cost": base_equipment_cost,
        "capex_addtl_capacity": capex_addtl_capacity,
        "construction_and_management_cost": construction_and_management,
    }
    return total_capex, tank_metrics, capex_components, expansion_only_capex
