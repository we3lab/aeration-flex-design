"""Step 0: build synthetic demo inputs from the IWA BSM1 dry-weather influent (no facility data needed).

Reads data/demo/bsm1_dry_weather_influent.csv (source and citations in data/demo/README.md) and writes
output_data/demo_iwa_bsm1/day_profiles.json and scaled_data.csv.
"""

import json
import os
import numpy as np
import pandas as pd

from helpers.compressor_power import moles_o2_required
from helpers.parameters import (
    DATA_DIR,
    HRT_DAYS,
    MGD_TO_M3_PER_DAY,
    OUTPUT_DATA_DIR,
    P_ATM,
    Pa_per_MPa,
    R,
    T_room,
    frac_o2_air,
)

RUN_NAME = "demo_iwa_bsm1"
START_DATE = "2023-07-01"
N_DAYS = 31 + 8  # July plus the 8 overflow days
STEPS_PER_DAY = 96

# # BSM1, Alex et al. 2008, scaled to 11 MGD
MEAN_FLOW_MGD = 11.0
BOD_OUT = 10 
N_OUT = 18
BOD5_PER_COD = 0.65
F_P = 0.08
REST_OF_FACILITY_KW_PER_MGD = 30.0  # demo value

# BSM1 gives 14 days at 15-min steps (the last row repeats t=0); tile it to N_DAYS
influent = pd.read_csv(os.path.join(DATA_DIR, "demo", "bsm1_dry_weather_influent.csv"))
influent = influent.iloc[: 14 * STEPS_PER_DAY]
influent = influent.iloc[np.arange(N_DAYS * STEPS_PER_DAY) % len(influent)].reset_index(drop=True)

flow_m3_day = influent["Q"] * MEAN_FLOW_MGD * MGD_TO_M3_PER_DAY / influent["Q"].mean()
bod_in = BOD5_PER_COD * (influent["S_S"] + influent["X_S"] + (1 - F_P) * influent["X_BH"])
n_in = influent["S_NH"] + influent["S_ND"] + influent["X_ND"]  # biodegradable N (ammonia + organic)

# BSM1 has no primary clarifier, so nothing is removed upstream of aeration
mol_o2_hr = moles_o2_required(
    flow_m3_day,
    BOD_in=bod_in,
    BOD_out=BOD_OUT,
    N_in=n_in,
    N_out=N_OUT,
    f_BOD_primary=0,
    f_N_primary=0,
)
# The aeration basin buffers influent load, so O2 demand follows its mean over one hydraulic residence time
mol_o2_hr = mol_o2_hr.rolling(int(HRT_DAYS * STEPS_PER_DAY), center=True, min_periods=1).mean()
# Supplied air (m³/day at room temperature and 1 atm), the unit the solver reads for air-blower plants
air_m3_day = mol_o2_hr / frac_o2_air * 24 * R * T_room / (P_ATM * Pa_per_MPa)

data = pd.DataFrame(
    {
        "DateTime": pd.date_range(START_DATE, periods=len(influent), freq="15min"),
        "Influent_Flow_MGD": flow_m3_day / MGD_TO_M3_PER_DAY,
        "Influent_BOD5": bod_in,
        "Influent_N": n_in,
        "Blower_AerationBasin_Air_Flow": air_m3_day,
        "VirtualDemand_RestOfFacilityPower": REST_OF_FACILITY_KW_PER_MGD
        * flow_m3_day
        / MGD_TO_M3_PER_DAY,
    }
).round(3)

day_profiles = {}
for day, day_data in data.groupby(data["DateTime"].dt.date):
    day_profiles[str(day)] = {
        "profile": {
            column: day_data[column].tolist()
            for column in ("Blower_AerationBasin_Air_Flow", "VirtualDemand_RestOfFacilityPower")
        },
        "month": day.month,
        "year": day.year,
    }

output_dir = os.path.join(OUTPUT_DATA_DIR, RUN_NAME)
os.makedirs(output_dir, exist_ok=True)
data.to_csv(os.path.join(output_dir, "scaled_data.csv"), index=False)
with open(os.path.join(output_dir, "day_profiles.json"), "w") as f:
    json.dump(
        {"day_profiles": day_profiles, "overflow_dates": sorted(day_profiles)[-8:]}, f, indent=2
    )
