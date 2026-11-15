"""CVXPY linear program for one rolling-horizon window of storage operation (OperationalProblem), with the
iterative compressor-coefficient and turndown passes."""

import calendar
import os
import time
import cvxpy as cp
import numpy as np
import json
from eeco import costs
from helpers.parameters import *
from helpers.compressor_power import comp_lp_coeff, pressure_from_moles, n_stages_for_ratio
from helpers.config_keys import get_upgrade_info, parse_design_key, split_key

HELPERS_DIR = os.path.dirname(os.path.abspath(__file__))
# Fix-and-relax turndown: timesteps below this fraction of rated flow are forced off
TURNDOWN_OFF_FRACTION = 0.4
TURNDOWN_MAX_PASSES = 6


def build_new_param_vals(config_name, design_key, annual_stats, horizon_days):
    """Build new_param_vals (system-level expansion limits + design-dependent storage, O2, solar, etc.)
    for one design key from the config's annual statistics.

    Returns {"INFEASIBLE": True} for invalid tank designs. A "~gas"/"~liq" design-key suffix forces that
    storage medium for a gas_tank upgrade — the job queue evaluates each design point under both and
    keeps the cheaper, so neither medium is chosen by rule.
    """
    base_key, upgrade_key, tariff_key = config_name.split("___")
    gas, o2_tech_base, _, summer_multiplier, _ = split_key(base_key)
    upgrade_info = get_upgrade_info(upgrade_key)
    o2_tech_for_storage = upgrade_info["o2_tech_for_storage"] or o2_tech_base
    is_cryo = o2_tech_base == "cryo"
    solar_multiplier = float(tariff_key.split("__", 1)[0])
    hr_of_O2, compression_ratio, charge_power_ratio, addtl_capacity_frac = map(
        float, parse_design_key(design_key)
    )
    medium = design_key.rsplit("~", 1)[1] if "~" in design_key else None

    # System-level: addtl_capacity_frac is the fraction of (Ndot_target_max - Ndot_b_max) allowed as expansion
    Ndot_b_max = annual_stats["Ndot_b_max"]
    Ndot_target_max = annual_stats["Ndot_target_max"]
    turndown = cryo_turndown if is_cryo else compressor_turndown
    addtl_capacity_max = addtl_capacity_frac * max(Ndot_target_max - Ndot_b_max, 0)
    if addtl_capacity_frac >= 1.0:
        addtl_capacity_max *= 1.001

    P_min = (P_AER + P_ATM) * 1.05
    P_init = P_init_map[o2_tech_for_storage]
    Ndot_target_mean = annual_stats["Ndot_target_mean"]

    frac_o2 = frac_o2_map[gas]
    N_max = hr_of_O2 * Ndot_target_mean

    # Pure O2 gas storage is capped at the 10 MPa steel-vessel envelope (parameters.max_o2_gas_pressure_mpa):
    # bulk composite pricing is H2-calibrated and does not apply to O2 service.
    P_max_gas = compression_ratio * P_init
    if gas == "o2":
        P_max_gas = min(P_max_gas, max_o2_gas_pressure_mpa)

    # Medium comes from the job's fan-out, so the gas/liquid crossover emerges from cost competition
    # (the min_liquid_tank_volume_m3 cost floor is what makes small liquid installs lose).
    is_liquid = bool(upgrade_info["is_liquid_tank"] or medium == "liq")

    P_max = 1e12 if is_liquid else P_max_gas
    V_tank = hr_of_O2 * Ndot_target_mean * R * T_room / (P_max * Pa_per_MPa * frac_o2)
    N_min = 0 if is_liquid else V_tank * (P_min * Pa_per_MPa) * frac_o2 / (R * T_room)
    E_max = annual_stats["Edot_b_mean"] * hr_of_O2 if upgrade_info["is_battery"] else 0

    if not is_liquid and not upgrade_info["is_battery"] and N_min > N_max:
        return {"INFEASIBLE": True}
    # OSHA bulk oxygen threshold (parameters.max_o2_gas_storage_mol): larger O2 stores go liquid
    if (
        gas == "o2"
        and not is_liquid
        and not upgrade_info["is_battery"]
        and N_max > max_o2_gas_storage_mol
    ):
        return {"INFEASIBLE": True}

    Ndot_c_max_val = 0 if (upgrade_info["is_battery"] or upgrade_info["has_surplus"]) else N_max
    if upgrade_info["has_elec"]:
        # Ndot_c is the electrolyzer's O2 output, so this caps its power at the EPRI <20 MW size class
        Ndot_c_max_val = min(Ndot_c_max_val, electrolyzer_max_power_kw / ei_elec_mol)

    Edot_r_max_val = 0
    if is_liquid:
        Edot_r_max_val = (
            annual_stats["Edot_t_baseline_max"] * charge_power_ratio / float(summer_multiplier)
        )
    elif upgrade_info["has_recovery"]:
        Edot_r_max_val = (
            comp_lp_coeff(P_min, P_max, efficiency=upgrade_info["nu_rec"], gas=gas, rec=True)
            * Ndot_target_max
            / frac_o2
        )

    Edot_c_max_val = (
        annual_stats["Edot_t_baseline_max"] * charge_power_ratio / float(summer_multiplier)
    )
    if upgrade_info["is_battery"]:
        Edot_c_max_val = min(max(Edot_c_max_val, battery_min_power_kw), battery_max_power_kw)
        E_max = min(E_max, battery_max_usable_hours * Edot_c_max_val / battery_usable_frac)
        # One inverter serves both directions: discharge is capped at the same (costed) power rating, so
        # charge_power_ratio sizes the whole inverter (E_max/hr let it shave peaks it wasn't costed for).
        Edot_r_max_val = Edot_c_max_val

    return {
        "Ndot_b_max": Ndot_b_max,
        "Ndot_b_excess_max": Ndot_b_max if upgrade_info["has_surplus"] else 0,
        "Ndot_b_min": min(annual_stats["Ndot_target_min_unscaled"], (1 - turndown) * Ndot_b_max),
        "Ndot_r_max": (
            0 if upgrade_info["is_battery"] else max(Ndot_target_max - addtl_capacity_max, 0)
        ),
        "Ndot_target_max": Ndot_target_max,
        "addtl_capacity_max": addtl_capacity_max,
        "Edot_t_max": annual_stats["Edot_t_baseline_max"] * 10,
        "Ndot_c_max": Ndot_c_max_val,
        "Edot_r_max": Edot_r_max_val,
        "Edot_c_max": Edot_c_max_val,
        "E_max": E_max,
        "P_min": P_min,
        "P_max": P_max,
        "P_init": P_init,
        "V_tank": V_tank,
        "N_min": N_min,
        "N_max": N_max,
        "is_liquid_storage": is_liquid,
        "eta_comp": eta,
        "eta_bat": 0.85,
        "nu_rec": upgrade_info["nu_rec"],
        "l_o2": 0.05 if is_cryo else 0.02,
        "losses": boiloff_frac_per_day / 96 if is_liquid else 0.0,
        "battery_self_discharge_per_step": (
            battery_self_discharge_per_day / 96 if upgrade_info["is_battery"] else 0.0
        ),
        "y_h2": 2 if upgrade_info["has_elec"] else 0,
        "ei_o2_new": ei_o2_map[o2_tech_for_storage],
        "frac_o2": frac_o2,
        "solar_multiplier": solar_multiplier,
        "hourly_solar_multiplier": (
            solar_multiplier * hourly_solar_multiplier_default
            if solar_multiplier > 0
            else np.ones(horizon_days * 96)
        ),
        "Edot_c_comp_h2_coeff": (
            comp_lp_coeff(P_init, 35, rec=False) if upgrade_info["has_elec"] else 0
        ),
    }


class OperationalProblem:
    """CVXPY/Gurobi LP for wastewater O2 delivery and storage optimization."""

    def __init__(
        self,
        prob_name,
        design_key,
        horizon_days,
        m,
        y,
        date,
        var_length,
        decomposition_type,
        solver_settings,
    ):
        base_key, self.upgrade_key, _ = prob_name.split("___")
        self.gas = split_key(base_key)[0]
        self.upgrade_info = get_upgrade_info(self.upgrade_key)

        self.design_key = design_key
        self.date = date
        self.horizon_days = horizon_days
        self.var_length = var_length
        self.demand_scale_factor = var_length / (calendar.monthrange(int(y), int(m))[1] * 96)
        self.decomposition_type = decomposition_type
        self.var_dict = json.load(open(os.path.join(HELPERS_DIR, "var_dict.json")))
        self.profile_columns = [
            name
            for category in self.var_dict["variables"].values()
            for name, props in category.items()
            if props.get("indexed", True)
        ] + list(self.var_dict["parameters"]["indexed"])
        self.solver_settings = {"solver": SOLVER, **solver_settings}

        # CVXPY objects — populated by construct_problem
        self.variables = {}
        self.parameters = {}
        self.derived_expressions = {}
        self.constraints = []
        self.problem = None
        self.is_built = False
        self.charge_params = None
        self.prev_demand_params = None

        # Iterative coefficient for the bilinear compressor charging power (see add_gas_tank_constraints)
        self.coeff_param = None
        self.rec_coeff_param = None
        self.compressor_stages = None

    @staticmethod
    def sanitize_symbol(name):
        return "".join(ch if ch.isalnum() else "_" for ch in str(name))

    def update_base_parameters(self, baseline_vals, new_vals):
        vals = {**baseline_vals, **new_vals}

        for param_name in self.var_dict["parameters"]["indexed"]:
            raw = vals[param_name]
            if param_name == "hourly_solar_multiplier":
                raw = np.tile(raw, self.horizon_days)[: self.var_length]
            self.parameters[param_name].value = np.array(raw, dtype=float)

        for category in self.var_dict["parameters"]["non_indexed"].values():
            for param_name in category:
                self.parameters[param_name].value = float(vals[param_name])

        # Helper parameters: products of parameters, precomputed so the problem stays DPP
        inv_frac_o2 = 1.0 / float(self.parameters["frac_o2"].value)
        edot_b_comp_coeff = float(self.parameters["Edot_b_comp_coeff"].value)
        ei_o2_base = float(self.parameters["ei_o2_base"].value)
        edot_t_max = float(self.parameters["Edot_t_max"].value)
        hourly_solar_multiplier = np.array(
            self.parameters["hourly_solar_multiplier"].value, dtype=float
        )

        self.parameters["inv_frac_o2"].value = inv_frac_o2
        self.parameters["Edot_b_comp_per_mol"].value = edot_b_comp_coeff * inv_frac_o2
        self.parameters["Edot_b_excess_per_mol"].value = (
            edot_b_comp_coeff * inv_frac_o2 + ei_o2_base
        )
        self.parameters["solar_cap_profile"].value = edot_t_max * hourly_solar_multiplier
        self.parameters["Edot_c_comp_h2_per_mol"].value = float(
            self.parameters["y_h2"].value
        ) * float(self.parameters["Edot_c_comp_h2_coeff"].value)

    def set_window_context_values(self, initial_storage_state):
        # Default to a half-full store when no carried-forward state is given
        initial_storage_state = initial_storage_state or {}
        mid_N = (
            float(self.parameters["N_min"].value) + float(self.parameters["N_max"].value)
        ) / 2.0
        self.parameters["window_initial_N"].value = float(initial_storage_state.get("N", mid_N))
        mid_E = float(self.parameters["E_max"].value) / 2.0
        self.parameters["window_initial_E"].value = float(initial_storage_state.get("E", mid_E))

    def update_cost_parameter_values(self, charge_dict, prev_demand_dict):
        for charge_key, charge_values in charge_dict.items():
            self.charge_params[charge_key].value = np.array(charge_values, dtype=float)

        for demand_key, demand_params in self.prev_demand_params.items():
            demand_params["demand"].value = float(prev_demand_dict[demand_key]["demand"])
            demand_params["cost"].value = float(prev_demand_dict[demand_key]["cost"])

    def create_vars_and_params(self, baseline_vals, new_vals):
        """Create CVXPY variables and parameters."""
        # Bulk storage medium decided pre-solve in build_new_param_vals: a gas_tank
        # upgrade may store surplus O2 as bulk liquid instead of compressed gas.
        self.is_liquid_storage = bool(new_vals["is_liquid_storage"])

        # Fix-and-relax compressor turndown, always enforced for gas-compression storage: these bounds are
        # updated between solves to force the storage compressor to >= compressor_min_turndown or off.
        self.use_turndown = self.upgrade_info["is_gas_tank"] and not self.is_liquid_storage
        if self.use_turndown:
            self.parameters["compressor_flow_lb"] = cp.Parameter(
                self.var_length, nonneg=True, value=np.zeros(self.var_length)
            )
            self.parameters["compressor_flow_ub"] = cp.Parameter(
                self.var_length, nonneg=True, value=np.full(self.var_length, 1e9)
            )

        # Indexed parameters
        for param_name, param_props in self.var_dict["parameters"]["indexed"].items():
            self.parameters[param_name] = cp.Parameter(
                self.var_length,
                nonneg=(param_props["domain"] == "NonNegativeReals"),
                name=param_name,
            )

        # Non-indexed parameters
        for category in self.var_dict["parameters"]["non_indexed"].values():
            for param_name, param_props in category.items():
                self.parameters[param_name] = cp.Parameter(
                    nonneg=(param_props["domain"] == "NonNegativeReals"),
                    name=param_name,
                )

        # Helper parameters (initialized to zero, updated in update_base_parameters)
        helper_spec = self.var_dict["helper_parameters"]
        for param_name, param_props in helper_spec["non_indexed"].items():
            self.parameters[param_name] = cp.Parameter(
                nonneg=(param_props["domain"] == "NonNegativeReals"),
                value=0.0,
                name=param_name,
            )
        for param_name, param_props in helper_spec["indexed"].items():
            self.parameters[param_name] = cp.Parameter(
                self.var_length,
                nonneg=(param_props["domain"] == "NonNegativeReals"),
                value=np.zeros(self.var_length),
                name=param_name,
            )

        self.update_base_parameters(baseline_vals, new_vals)

        # Variables, with their bounds as constraints
        var_categories = ["base"]
        if self.upgrade_info["is_battery"]:
            var_categories.append("energy_storage")
        elif self.upgrade_info["is_tank"]:
            var_categories.append("o2_storage")
            if self.upgrade_info["has_h2"]:
                var_categories.append("gas_tank")

        for cat in var_categories:
            for name, props in self.var_dict["variables"][cat].items():
                is_nonneg = props["domain"] == "NonNegativeReals"
                if props.get("indexed", True):
                    var = cp.Variable(self.var_length, nonneg=is_nonneg, name=name)
                else:
                    var = cp.Variable(nonneg=is_nonneg, name=name)
                self.variables[name] = var

                lb, ub = props["bounds"]
                self.constraints.append(
                    var >= (self.parameters[lb].value if isinstance(lb, str) else lb)
                )
                self.constraints.append(
                    var <= (self.parameters[ub].value if isinstance(ub, str) else ub)
                )

    def construct_problem(
        self,
        charge_dict,
        baseline_vals,
        new_param_vals,
        prev_demand_dict,
        initial_storage_state=None,
    ):
        """Build CVXPY problem. On subsequent calls, only update parameter values."""
        if self.is_built:
            self.update_base_parameters(baseline_vals, new_param_vals)
            self.set_window_context_values(initial_storage_state)
            self.update_cost_parameter_values(charge_dict, prev_demand_dict)
            return

        self.create_vars_and_params(baseline_vals, new_param_vals)
        self.set_window_context_values(initial_storage_state)
        self.build_derived_expressions()

        self.add_base_constraints()
        if self.upgrade_info["is_tank"]:
            self.add_o2_storage_constraints()
        elif self.upgrade_info["is_battery"]:
            self.add_battery_constraints()

        # Cyclic window: storage starts and ends at the carried-forward state
        for storage_key in ("N", "E"):
            if storage_key in self.variables:
                initial = self.parameters[f"window_initial_{storage_key}"]
                self.constraints.append(self.variables[storage_key][0] == initial)
                self.constraints.append(self.variables[storage_key][-1] == initial)

        self.add_cost_constraints(charge_dict, prev_demand_dict)
        self.construct_objective()
        self.is_built = True

    def build_derived_expressions(self):
        """Create derived symbolic expressions that are not decision variables."""
        edot_b_comp_per_mol = self.parameters["Edot_b_comp_per_mol"]
        edot_b_excess_per_mol = self.parameters["Edot_b_excess_per_mol"]
        ei_o2_base = self.parameters["ei_o2_base"]

        self.derived_expressions["Edot_b_comp"] = cp.multiply(
            edot_b_comp_per_mol, self.variables["Ndot_b_aer"]
        )
        self.derived_expressions["Edot_b_gen"] = ei_o2_base * self.variables["Ndot_b_aer"]
        self.derived_expressions["Edot_b_excess"] = cp.multiply(
            edot_b_excess_per_mol, self.variables["Ndot_b_excess"]
        )
        self.derived_expressions["Edot_addtl_capacity_comp"] = cp.multiply(
            edot_b_comp_per_mol, self.variables["addtl_capacity"]
        )
        self.derived_expressions["Edot_addtl_capacity_gen"] = (
            ei_o2_base * self.variables["addtl_capacity"]
        )

        if self.upgrade_info["is_tank"]:
            self.derived_expressions["Edot_c_gen"] = (
                self.parameters["ei_o2_new"] * self.variables["Ndot_c"] + 1e-4
            )
            if self.upgrade_info["is_gas_tank"]:
                self.derived_expressions["Edot_c_comp_h2"] = cp.multiply(
                    self.parameters["Edot_c_comp_h2_per_mol"], self.variables["Ndot_c"]
                )
            else:
                self.derived_expressions["Edot_c_comp_h2"] = np.zeros(self.var_length)

    def add_base_constraints(self):
        """Add operational constraints common to all upgrade types."""
        Edot_b_comp = self.derived_expressions["Edot_b_comp"]
        Edot_b_gen = self.derived_expressions["Edot_b_gen"]
        Edot_addtl_capacity_comp = self.derived_expressions["Edot_addtl_capacity_comp"]
        Edot_addtl_capacity_gen = self.derived_expressions["Edot_addtl_capacity_gen"]

        self.constraints.append(self.variables["Edot_b"] == Edot_b_comp + Edot_b_gen)
        self.constraints.append(
            self.variables["Ndot_b"]
            == self.variables["Ndot_b_aer"] + self.variables["Ndot_b_excess"]
        )
        self.constraints.append(
            self.variables["Edot_addtl_capacity"]
            == Edot_addtl_capacity_comp + Edot_addtl_capacity_gen
        )
        self.constraints.append(
            self.variables["Edot_t_net"]
            == self.variables["Edot_t"]
            - cp.multiply(self.variables["Edot_c"], self.parameters["solar_multiplier"])
        )
        self.constraints.append(
            self.variables["Edot_c"] <= self.parameters["solar_cap_profile"] + 1e-3
        )

        # Compressor ramp-up limit
        self.constraints.append(
            self.variables["Edot_c"][1:] - self.variables["Edot_c"][:-1]
            <= self.parameters["Edot_t_max"] * 0.4
        )

        if self.upgrade_info["is_tank"]:
            self.constraints.append(
                self.variables["Edot_c"]
                == self.derived_expressions["Edot_c_gen"]
                + self.variables["Edot_c_comp"]
                + self.derived_expressions["Edot_c_comp_h2"]
            )

    def add_o2_storage_constraints(self):
        """Constraints for O2 storage (gas tank or liquid tank)."""
        N = self.variables["N"]
        Ndot_c = self.variables["Ndot_c"]
        Ndot_b_excess = self.variables["Ndot_b_excess"]
        Ndot_r = self.variables["Ndot_r"]
        Ndot_b_aer = self.variables["Ndot_b_aer"]
        l_o2 = self.parameters["l_o2"]
        losses = self.parameters["losses"]
        Ndot_target = self.parameters["Ndot_target"]

        # O2 demand balance: Ndot_target = Ndot_b_aer + (1 - l_o2)*Ndot_r + addtl_capacity + unmet_o2
        self.constraints.append(
            Ndot_target
            == Ndot_b_aer
            + (1 - l_o2) * Ndot_r
            + self.variables["addtl_capacity"]
            + self.variables["unmet_o2"]
        )

        # Storage mass balance: N[t+1] = N[t]*(1 - losses) + (Ndot_c + Ndot_b_excess - Ndot_r) / 4, where /4 turns
        # mol/hr rates into mol per 15-min step and losses = standing liquid-O2 boil-off (0 for gas/battery).
        self.constraints.append(
            N[1:] == N[:-1] * (1 - losses) + (Ndot_c[:-1] + Ndot_b_excess[:-1] - Ndot_r[:-1]) / 4
        )

        # Total energy balance: Edot_t = Edot_rem + Edot_c + Edot_b + Edot_b_excess + Edot_addtl + Edot_r_o2 - Edot_r
        self.constraints.append(
            self.variables["Edot_t"]
            == self.parameters["Edot_rem"]
            + self.variables["Edot_c"]
            + self.variables["Edot_b"]
            + self.derived_expressions["Edot_b_excess"]
            + self.variables["Edot_addtl_capacity"]
            + self.variables["Edot_r_o2"]
            - self.variables["Edot_r"]
        )

        if self.is_liquid_storage:
            self.add_liquid_tank_constraints()
        else:
            self.add_gas_tank_constraints()

        if self.upgrade_info["has_h2"]:
            self.constraints.append(
                self.variables["total_h2"]
                == cp.sum(self.variables["Ndot_c"] * self.parameters["y_h2"]) / 4
            )

    def add_gas_tank_constraints(self):
        """Pressure bound and iterative-coefficient charging constraint for gas tanks."""
        frac_o2 = float(self.parameters["frac_o2"].value)
        P_low = float(self.parameters["P_init"].value)
        V_tank = float(self.parameters["V_tank"].value)
        efficiency = float(self.parameters["eta_comp"].value)
        N_at_P_low = P_low * V_tank * Pa_per_MPa * frac_o2 / (R * T_room)  # ideal gas
        self.constraints.append(self.variables["N"] >= N_at_P_low)

        # Power = coeff(N) * flow is bilinear, so coeff is a per-timestep Parameter initialized at N_mid and
        # updated after each solve with exact coeff(N_solved); concavity of coeff(N) guarantees convergence.
        flow_vec = (
            cp.sum([self.variables["Ndot_c"], self.variables["Ndot_b_excess"]], axis=0) / frac_o2
        )

        # Size the compression stage count once from the design's maximum pressure, so it
        # stays fixed while the tank pressure varies during operation.
        N_max = float(self.parameters["N_max"].value)
        P_max_design = max(float(pressure_from_moles(N_max, V_tank, frac_o2)), P_low)
        self.compressor_stages = n_stages_for_ratio(P_max_design / P_low + 1e-6)

        N_mid = (self.parameters["N_min"].value + self.parameters["N_max"].value) / 2
        P_mid = max(float(pressure_from_moles(N_mid, V_tank, frac_o2)), P_low)
        coeff_init = comp_lp_coeff(
            P_low,
            P_mid,
            efficiency=efficiency,
            rec=False,
            gas=self.gas,
            n_stages=self.compressor_stages,
        )
        self.coeff_param = cp.Parameter(
            self.var_length,
            nonneg=True,
            name="coeff_comp_iter_compressor_charging",
            value=np.full(self.var_length, coeff_init),
        )
        self.constraints.append(
            self.variables["Edot_c_comp"] >= cp.multiply(self.coeff_param, flow_vec)
        )

        # Recovered power needs released gas: coeff_rec(N) is iterated like charging, starting at 0
        if self.upgrade_info["has_recovery"]:
            self.rec_coeff_param = cp.Parameter(
                self.var_length, nonneg=True, name="coeff_rec_iter", value=np.zeros(self.var_length)
            )
            self.constraints.append(
                self.variables["Edot_r"]
                <= cp.multiply(self.rec_coeff_param, self.variables["Ndot_r"]) / frac_o2
            )

        compressor_flow = self.variables["Ndot_c"] + self.variables["Ndot_b_excess"]
        self.constraints.append(compressor_flow >= self.parameters["compressor_flow_lb"])
        self.constraints.append(compressor_flow <= self.parameters["compressor_flow_ub"])

    def add_liquid_tank_constraints(self):
        """Evaporation energy and zero compressor recovery power for liquid tanks.

        When a gas_tank upgrade stores its gaseous surplus O2 (from PSA/electrolyzer) as
        bulk liquid, that surplus must first be liquefied, so the charging power carries a
        liquefaction term instead of gas-compression power. Native liquid_tank upgrades
        (cryo) already produce liquid O2, so they carry no liquefaction charging power.
        """
        self.constraints.append(
            self.variables["Edot_r_o2"] == self.parameters["ei_evap"] * self.variables["Ndot_r"]
        )
        self.constraints.append(self.variables["Edot_r"] == 0)
        if self.upgrade_info["is_gas_tank"]:
            # Per-gas: storing air means liquefying the whole air stream, so the energy per
            # mol of O2-equivalent flow is ~4.8x the pure-O2 value (see ei_liquefy_mol_map)
            self.constraints.append(
                self.variables["Edot_c_comp"]
                == ei_liquefy_mol_map[self.gas]
                * (self.variables["Ndot_c"] + self.variables["Ndot_b_excess"])
            )

    def add_battery_constraints(self):
        """Constraints for battery storage."""
        Edot_c = self.variables["Edot_c"]
        Edot_r = self.variables["Edot_r"]
        E = self.variables["E"]

        # Total energy balance (no Edot_r_o2 for batteries)
        self.constraints.append(
            self.variables["Edot_t"]
            == Edot_c
            + self.variables["Edot_b"]
            + self.derived_expressions["Edot_b_excess"]
            + self.variables["Edot_addtl_capacity"]
            + self.parameters["Edot_rem"]
            - Edot_r
        )

        # O2 demand balance (no storage recovery for battery)
        self.constraints.append(
            self.parameters["Ndot_target"]
            == self.variables["Ndot_b_aer"]
            + self.variables["addtl_capacity"]
            + self.variables["unmet_o2"]
        )

        # Battery energy dynamics: E[t+1] = E[t]*(1 - sd) + (Edot_c*sqrt(eta) - Edot_r/sqrt(eta)) / 4, where /4 turns kW
        # into kWh per 15-min step and sd is the per-step self-discharge (see battery_self_discharge_per_day).
        eta_bat = float(self.parameters["eta_bat"].value)
        sqrt_eta = np.sqrt(eta_bat)
        self_discharge = float(self.parameters["battery_self_discharge_per_step"].value)
        self.constraints.append(
            E[1:]
            == E[:-1] * (1 - self_discharge) + (Edot_c[:-1] * sqrt_eta - Edot_r[:-1] / sqrt_eta) / 4
        )

        # 80% depth-of-discharge floor; capex prices only the usable 80% (see battery_usable_frac)
        dod_floor = (1 - battery_usable_frac) * float(self.parameters["E_max"].value)
        self.constraints.append(E >= dod_floor)

    def add_cost_constraints(self, charge_dict, prev_demand_dict):
        """eeco itemized cost + tariff epigraph + H2 value constraint."""
        self.charge_params = {
            charge_key: cp.Parameter(
                len(charge_values),
                nonneg=True,
                value=np.zeros(len(charge_values)),
                name=f"charge_{self.sanitize_symbol(charge_key)}",
            )
            for charge_key, charge_values in charge_dict.items()
        }
        self.prev_demand_params = {
            demand_key: {
                "demand": cp.Parameter(
                    nonneg=True, value=0.0, name=f"prev_demand_{self.sanitize_symbol(demand_key)}"
                ),
                "cost": cp.Parameter(
                    nonneg=True, value=0.0, name=f"prev_cost_{self.sanitize_symbol(demand_key)}"
                ),
            }
            for demand_key in charge_dict
            if "demand" in str(demand_key)
        }
        self.update_cost_parameter_values(charge_dict, prev_demand_dict)

        itemized_costs, cvxpy_constraints = costs.calculate_itemized_cost(
            charge_dict=self.charge_params,
            consumption_data_dict={"electric": self.variables["Edot_t_net"]},
            prev_demand_dict=self.prev_demand_params or None,
            desired_utility="electric",
            demand_scale_factor=self.demand_scale_factor,
            resolution="15m",
            decomposition_type=self.decomposition_type,
            big_m=float(self.parameters["Edot_t_max"].value),
        )
        # cvxpy_constraints is None when no decomposition is needed (e.g. no exports)
        self.constraints.extend(cvxpy_constraints or [])
        self.constraints.append(self.variables["tariff_cost"] >= itemized_costs["total"])
        if self.upgrade_info["has_h2"]:
            self.constraints.append(
                self.variables["h2_value"]
                == moles_to_mass(self.variables["total_h2"], M_H2) * price_h2_kg
            )

    def construct_objective(self):
        obj = self.variables["tariff_cost"]
        if self.upgrade_info["has_h2"]:
            obj = obj - self.variables["h2_value"]
        if not self.upgrade_info["is_battery"]:
            # Tiebreaker: prefer Ndot_b over addtl_capacity (identical energy intensity)
            obj = obj + 1e-6 * cp.sum(self.variables["addtl_capacity"])
        # Tiebreaker: prefer aeration over tank charging (identical per-mol cost)
        obj = obj + 1e-6 * cp.sum(self.variables["Ndot_b_excess"])
        # Large penalty drives solver to treat unmet_o2 as infeasibility signal
        obj = obj + 1e6 * cp.sum(self.variables["unmet_o2"])
        self.objective = cp.Minimize(obj)

    def solve_O2Problem(self):
        if self.problem is None:
            self.problem = cp.Problem(self.objective, self.constraints)

        if self.use_turndown:
            # Start each solve from relaxed compressor bounds so the coefficient iteration
            # finds the unconstrained optimum before turndown is imposed.
            self.parameters["compressor_flow_lb"].value = np.zeros(self.var_length)
            self.parameters["compressor_flow_ub"].value = np.full(self.var_length, 1e9)

        if self.coeff_param is not None:
            profile, solve_info = self.solve_with_iterative_coeff()
        else:
            profile, solve_info = self.solve_once()

        # True discard (initial relaxed solve infeasible, not a recovered turndown pass). Print once
        # here so transient recovered sub-solve failures don't spam the log.
        if profile is None:
            print(
                f"CVXPY solve failed (discard): {self.upgrade_key} {self.design_key} {self.date} status={solve_info['status']}"
            )
        elif self.use_turndown:
            profile, solve_info = self.run_turndown_passes(profile, solve_info)
        return profile, solve_info

    def solve_once(self):
        """Solve the CVXPY problem once and return (profile, solve_info)."""
        # DPP canonicalization for short rolling-horizon windows (7-8 days).
        solver_kwargs = {
            "ignore_dpp": self.solver_settings.get("ignore_dpp", self.horizon_days >= 28)
        }
        if self.solver_settings["solver"] == "gurobi":
            solver_kwargs.update(
                solver=cp.GUROBI,
                BarConvTol=1e-6,
                FeasibilityTol=1e-6,
                NumericFocus=1,
                Threads=1,
                Seed=RANDOM_SEED,
                TimeLimit=180,
            )
        else:
            # Open-source fallback: HiGHS bundled with SciPy (no license needed, slower than Gurobi)
            solver_kwargs.update(
                solver=cp.SCIPY, scipy_options={"method": "highs", "time_limit": 180}
            )
        # The WLS license needs the network: retry license/network errors (else cached as a discard), raise after 12 h
        retry_wait, waited = 60, 0
        while True:
            try:
                self.problem.solve(**solver_kwargs)
                break
            except Exception as e:
                if getattr(e, "errno", None) not in (10009, 10010, 10022):
                    print(f"CVXPY solve exception: {e}")
                    return None, {"status": "exception"}
                if waited >= 12 * 3600:
                    raise
                print(f"Gurobi license/network error ({e}), retrying in {retry_wait}s", flush=True)
                time.sleep(retry_wait)
                waited += retry_wait
                retry_wait = min(2 * retry_wait, 600)
                import gurobipy  # optional dependency

                gurobipy.disposeDefaultEnv()

        solve_info = {"status": self.problem.status}
        if self.problem.status not in ["optimal", "optimal_inaccurate", "user_limit"]:
            # No print here — solve_O2Problem prints once if the FINAL result is a true discard
            # (turndown/coeff sub-solves fail transiently then recover).
            return None, solve_info
        solve_info.update(
            {"objective": self.problem.value, "solve_time": self.problem.solver_stats.solve_time}
        )
        return self.get_profile(), solve_info

    def run_turndown_passes(self, profile, solve_info):
        """Fix-and-relax compressor turndown (no binary): iteratively force the storage
        compressor to run at >= compressor_min_turndown of rated flow or off, deciding the
        on/off pattern from each solved flow profile. Each pass is a pure LP re-solve.

        Timesteps below turndown_off_fraction of rated flow are forced off; timesteps
        already at/above the turndown are held on (>= turndown x rated); the band between
        is left relaxed to avoid forcing charge the tank cannot absorb. Heuristic: yields
        a turndown-respecting schedule, not a guaranteed global optimum.
        """
        flow_cap = float(self.parameters["Ndot_c_max"].value) + float(
            self.parameters["Ndot_b_excess_max"].value
        )
        flow_lb = self.parameters["compressor_flow_lb"]
        flow_ub = self.parameters["compressor_flow_ub"]
        prev_state = None
        for passes in range(1, TURNDOWN_MAX_PASSES + 1):
            flow = np.asarray(self.variables["Ndot_c"].value, dtype=float) + np.asarray(
                self.variables["Ndot_b_excess"].value, dtype=float
            )
            state = np.zeros(self.var_length)  # -1 off, +1 on, 0 undecided
            state[flow < TURNDOWN_OFF_FRACTION * flow_cap] = -1
            state[flow >= storage_compressor_min_turndown * flow_cap] = 1
            if prev_state is not None and np.array_equal(state, prev_state):
                break
            flow_lb.value = np.where(state > 0, storage_compressor_min_turndown * flow_cap, 0.0)
            flow_ub.value = np.where(state < 0, 0.0, flow_cap)
            prev_state = state

            new_profile, new_info = self.solve_once()
            if new_profile is None:
                break  # keep the last feasible profile
            profile, solve_info = new_profile, new_info

        # Re-converge the compressor coefficients for the final turndown bounds so the
        # linearization matches the constrained schedule (otherwise the recalc error inflates).
        final_profile, final_info = self.solve_with_iterative_coeff()
        if final_profile is not None:
            profile, solve_info = final_profile, final_info

        solve_info["turndown_passes"] = passes
        solve_info["turndown_on_fraction"] = float(np.mean(prev_state > 0))
        return profile, solve_info

    def solve_with_iterative_coeff(self):
        """Iteratively solve, updating compressor coeff from solved N via DPP.

        At each iteration: solve LP: extract N[t]: compute exact coeff(N[t])
       : damped update of coeff_param: re-solve. Damping prevents oscillation.
        """
        max_iters = int(self.solver_settings.get("coeff_max_iters", 15))
        tol = float(self.solver_settings.get("coeff_tol", 1e-2))
        # Constant damping: undamped (1.0) limit-cycles
        alpha = float(self.solver_settings.get("alpha", 0.5))
        frac_o2 = float(self.parameters["frac_o2"].value)
        P_low = float(self.parameters["P_init"].value)
        V_tank = float(self.parameters["V_tank"].value)
        efficiency = float(self.parameters["eta_comp"].value)

        for iteration in range(max_iters):
            profile, solve_info = self.solve_once()
            if profile is None:
                break

            N_vals = np.array(self.variables["N"].value, dtype=float)
            exact_coeffs = np.zeros(self.var_length)
            for t in range(self.var_length):
                P_high = max(pressure_from_moles(N_vals[t], V_tank, frac_o2), P_low)
                exact_coeffs[t] = comp_lp_coeff(
                    P_low,
                    P_high,
                    efficiency=efficiency,
                    rec=False,
                    gas=self.gas,
                    n_stages=self.compressor_stages,
                )

            # The coefficient warm-starts the next window, since construct_problem reuses this Parameter
            old_coeffs = self.coeff_param.value
            new_coeffs = alpha * exact_coeffs + (1 - alpha) * old_coeffs
            rel_change = np.max(np.abs(new_coeffs - old_coeffs) / np.maximum(old_coeffs, 1e-10))
            self.coeff_param.value = new_coeffs

            if self.rec_coeff_param is not None:
                # Conservative: price each step's recovery at the lower of its start/end tank pressure
                N_low = np.minimum(N_vals, np.append(N_vals[1:], N_vals[-1]))
                P_min = float(self.parameters["P_min"].value)
                exact_rec = np.array(
                    [
                        comp_lp_coeff(
                            P_min,
                            max(pressure_from_moles(n, V_tank, frac_o2), P_min),
                            efficiency=float(self.parameters["nu_rec"].value),
                            rec=True,
                            gas=self.gas,
                            n_stages=self.compressor_stages,
                        )
                        for n in N_low
                    ]
                )
                old_rec = self.rec_coeff_param.value
                new_rec = alpha * exact_rec + (1 - alpha) * old_rec
                rec_change = np.max(np.abs(new_rec - old_rec)) / max(np.max(new_rec), 1e-6)
                rel_change = max(rel_change, rec_change)
                self.rec_coeff_param.value = new_rec

            if rel_change < tol:
                break

        solve_info["coeff_iterations"] = iteration + 1
        solve_info["coeff_converged"] = profile is not None and rel_change < tol
        return profile, solve_info

    def get_profile(self):
        """Extract solution values into profile dictionary."""
        profile = {}
        for column in self.profile_columns:
            if column in self.variables:
                values = self.variables[column].value
            elif column in self.parameters:
                values = self.parameters[column].value
            else:
                continue
            profile[column] = np.asarray(
                values if values is not None else np.zeros(self.var_length), dtype=float
            )

        # Gas-compression power recalc only applies to designs that store compressed gas; liquid-storage
        # designs (native or gas_tank storing surplus as liquid) copy the already-exact charging power through.
        if self.upgrade_info["is_gas_tank"] and not self.is_liquid_storage:
            frac_o2 = float(self.parameters["frac_o2"].value)
            P_init = float(self.parameters["P_init"].value)
            efficiency = float(self.parameters["eta_comp"].value)
            pressures = pressure_from_moles(
                profile["N"], float(self.parameters["V_tank"].value), frac_o2
            )
            # Same stage count as the charging constraint, so the recalculated power matches what the optimizer solved
            exact_power = np.array(
                [
                    (
                        comp_lp_coeff(
                            P_init,
                            pressure,
                            efficiency=efficiency,
                            gas=self.gas,
                            n_stages=self.compressor_stages,
                        )
                        * ((ndot_c + ndot_b_excess) / frac_o2)
                        if pressure >= P_init
                        else 1e-4
                    )
                    for ndot_c, ndot_b_excess, pressure in zip(
                        profile["Ndot_c"], profile["Ndot_b_excess"], pressures
                    )
                ],
                dtype=float,
            )
            profile["Edot_c_comp_recalc"] = exact_power
            profile["Edot_t_net_recalc"] = profile["Edot_t_net"] + (
                exact_power - profile["Edot_c_comp"]
            )
        else:
            profile["Edot_t_net_recalc"] = profile["Edot_t_net"]
            if self.is_liquid_storage:
                profile["Edot_c_comp_recalc"] = profile["Edot_c_comp"]
        return profile
