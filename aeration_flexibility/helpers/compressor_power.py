"""Polytropic compression power per unit molar flow, tank pressure from moles, and first-principles O2 demand
from BOD and nitrogen removal (SI eq. S1-S2)."""

from helpers.parameters import *
import numpy as np


def n_stages_for_ratio(P_ratio):
    """Number of compressor stages for a pressure ratio, capped at max_stage_pressure_ratio
    per stage (a real multi-stage machine keeps each stage below ~5x for temperature limits)."""
    return max(1, int(np.ceil(np.log(P_ratio) / np.log(max_stage_pressure_ratio))))


def comp_lp_coeff(P_low, P_high, efficiency=eta, rec=False, gas="o2", n_stages=None):
    """Compression/expansion power coefficient (kW per mol/hr of gas).

    Compression (rec=False) is modeled as a multi-stage machine with interstage cooling:
    n_stages equal stages, each with pressure ratio P_ratio**(1/n_stages) and gas cooled
    back to inlet temperature between stages, which draws far less power than a single
    stage at high ratios. n_stages defaults to n_stages_for_ratio(P_ratio); pass it
    explicitly (e.g. sized from the design's maximum pressure) to keep it fixed while the
    tank pressure varies. A low-ratio aeration blower resolves to n_stages=1 (single stage).
    Expansion/recovery (rec=True) uses the same stages, reheated to inlet temperature between
    them as compressed-air expanders are (one stage from 167:1 would chill air to ~70 K), with
    textbook expansion work per stage. Generous recovery is the conservative choice for the
    SI finding that recovery doesn't pay.

    The coefficient C satisfies: Power(kW) = C × Ndot(mol/hr), where Ndot is the molar
    flow of the gas being compressed. To get power per mol O2 for air-blower systems,
    divide by frac_o2_air (since only 21% of the compressed air is O2).

    Args:
        P_low: Inlet pressure (MPa absolute)
        P_high: Outlet pressure (MPa absolute)
        efficiency: Motor efficiency (default eta = 0.75); for rec, the whole expander's
            efficiency (nu_rec), with no polytropic factor
        rec: True for expansion/recovery, False for compression
        gas: "air" or "o2" — selects the polytropic index n from n_map
        n_stages: number of compression stages; None auto-sizes from the pressure ratio

    Returns:
        kW per mol/hr of the compressed gas
    """
    # https://web.archive.org/web/20250425060100/https://www.webpages.uidaho.edu/ce431/Handouts/Sanitaire%20Diffused%20Air%20Design%20Guide.pdf
    n = n_map[gas]
    P_ratio = P_high / P_low + 1e-6

    if n_stages is None:
        n_stages = n_stages_for_ratio(P_ratio)
    stage_ratio = P_ratio ** (1.0 / n_stages)

    if rec:
        # Per stage: n/(n-1) RT [1 - r_s^-(n-1)/n]
        return (
            n_stages
            * efficiency
            * (n / (n - 1))
            / 3600
            * R
            * T_room
            * (1 - stage_ratio ** (-(n - 1) / n))
            / 1000
        )

    eff_multiplier = (n / (n - 1)) * (1 / polytropic_efficiency) * (1 / efficiency)

    coeff = (
        n_stages
        * eff_multiplier
        / 3600  # mol/hr: mol/s
        * R
        * T_room
        * (stage_ratio ** ((n - 1) / n) - 1)
        / 1000  # W: kW
    )

    return coeff


def moles_o2_required(
    Q,
    BOD_in=BOD_in,
    BOD_out=BOD_out,
    N_in=N_in,
    N_out=N_out,
    f_BOD_primary=f_BOD_primary,
    f_N_primary=f_N_primary,
):
    """First-principles O2 demand from BOD and nitrogen removal (eq. S1-S2 from SI).

    Computes the moles of O2 per hour needed to meet effluent BOD and nitrogen targets,
    accounting for primary removal fractions, oxygen stoichiometry (epsilon_BOD, epsilon_N),
    and oxygen transfer efficiency (SOTE × alpha × beta). Defaults are the paper's assumptions;
    all inputs may be arrays (e.g. a 15-min influent timeseries).

    Args:
        Q: Average wastewater flowrate (m³/day)
        BOD_in, BOD_out: Influent and effluent BOD5 (mg/L)
        N_in, N_out: Influent and effluent nitrogen (mg/L)
        f_BOD_primary, f_N_primary: fractions removed upstream of aeration (primary treatment)

    Returns:
        Moles O2 per hour required from the aeration system
    """
    g_o2_per_L = (
        1
        / 1000
        * (
            (BOD_in - BOD_out) * (1 - f_BOD_primary) * epsilon_BOD
            + (N_in - N_out) * (1 - f_N_primary) * epsilon_N
        )
    )  # g O2/L
    moles_o2_per_L = g_o2_per_L / M_O2  # g/L / (g/mol) = mol / L
    moles_per_hour = (
        moles_o2_per_L * (Q * 1000 / 24) / (SOTE["Disc"] * alpha * beta)
    )  # mol / L * m3/day * 1000 L/m3 / 24 hr / day / SOTE => mol/hr
    return moles_per_hour


def pressure_from_moles(N, V_tank, frac_o2):
    """
    Calculate pressure from moles.
    P = N / frac_o2 * R * T / (V * Pa_per_MPa)

    Args:
        N: Moles (can be numeric or CVXPY Variable)
        V_tank: Tank volume (numeric or parameter)
        frac_o2: O2 fraction (numeric or parameter)

    Returns:
        Pressure expression (same type as N)
    """
    return N / frac_o2 * R * T_room / (V_tank * Pa_per_MPa)
