"""Electricity tariffs from data/billing (base rates plus registered variants) and monthly eeco charge dicts.
Also converts costs to 2025 USD."""

import calendar
from datetime import date, timedelta

import numpy as np
import os
import pandas as pd
from eeco import costs
from eeco.costs import parametrize_rate_data

from helpers.parameters import DATA_DIR

BILLING_DIR = os.path.join(DATA_DIR, "billing")
UTILITIES = ("cas", "hpoas")
# Billing rates are identical across solar fractions; the fraction scales solar generation in the optimization model
SOLAR_FRACTIONS = ("0.0", "0.25", "0.5", "0.75", "1.0")

# Calendar months and their date ranges (Jan 2022 – Dec 2024)
CALENDAR_MONTHS = {
    f"{year}-{month:02d}": {
        "start": str(date(year, month, 1)),
        "end": str(date(year, month, 1) + timedelta(days=calendar.monthrange(year, month)[1])),
    }
    for year in range(2022, 2025)
    for month in range(1, 13)
}

YEARLY_INFLATION = (
    pd.read_csv(os.path.join(DATA_DIR, "inflation.csv"))
    .set_index("Year")["Average"]
    .str.rstrip("%")
    .astype(float)
    / 100
)

# Tariff variants beyond the base rates: the unchanged baseline plus any registered ones (e.g. step3 sweep ranges).
# A peak-window variant "w{x}" widens the peak window by x hours (x/2 each side); "wneg{x}" narrows it.
registered_variants = [
    {"percent_change_dict": {"demand": 1.0, "energy": 1.0}, "variant_name": "d1.0_e1.0_w0"}
]
tariff_data_cache = None


def register_tariff_variants(variants):
    """Register additional tariff variants to be generated/loaded.

    Each variant is a dict with at least 'variant_name' and the parametrization
    keys accepted by parametrize_rate_data (percent_change_dict,
    shift_peak_hours_before, shift_peak_hours_after).

    Invalidates the tariff_data cache on a new variant, so an earlier config's
    get_tariff_data() call in the same process can't freeze it stale.
    """
    global tariff_data_cache
    existing_names = {v["variant_name"] for v in registered_variants}
    for variant in variants:
        if variant["variant_name"] not in existing_names:
            registered_variants.append(variant)
            existing_names.add(variant["variant_name"])
            tariff_data_cache = None


def get_month_dates(year, month):
    """Get start and end dates for a specific month."""
    month_key = f"{year}-{month:02d}"
    return {
        "start": np.datetime64(CALENDAR_MONTHS[month_key]["start"]),
        "end": np.datetime64(CALENDAR_MONTHS[month_key]["end"]),
    }


def get_tariff_data():
    """Load all tariff CSV files from the billing directory, once per process.
    Returns a dictionary mapping tariff keys ("{solar fraction}__{utility}[_{variant}]") to DataFrames.
    Missing variant CSVs are generated first, so every registered variant is available."""
    global tariff_data_cache
    if tariff_data_cache is None:
        tariff_data_cache = {}
        for utility in UTILITIES:
            base_rates = pd.read_csv(os.path.join(BILLING_DIR, f"{utility}.csv"))
            rates_by_name = {utility: base_rates}
            for variant in registered_variants:
                name = f"{utility}_{variant['variant_name']}"
                path = os.path.join(BILLING_DIR, f"{name}.csv")
                if not os.path.exists(path):
                    parametrize_rate_data(base_rates.copy(), **variant).to_csv(path, index=False)
                    print(f"Generated {name}.csv")
                rates_by_name[name] = pd.read_csv(path)
            for name, rates in rates_by_name.items():
                for solar_fraction in SOLAR_FRACTIONS:
                    tariff_data_cache[f"{solar_fraction}__{name}"] = rates
    return tariff_data_cache


def get_charge_dict_for_month(tariff_data, year, month):
    """
    Get charge dict for a specific month.
    Args:
        tariff_data: DataFrame containing the tariff rate data
        year: Year (int)
        month: Month (int)
    Returns:
        Dictionary of charge arrays for the month
    """
    month_dates = get_month_dates(year, month)
    return costs.get_charge_dict(
        month_dates["start"], month_dates["end"], tariff_data, resolution="15m"
    )


def to_usd_2025(cost, from_year):
    """Convert a cost from a given year to 2025 dollars using inflation data."""
    # Cumulative forward inflation from the source year to 2025. Each year's rate is its change over the prior
    # year, so a from_year cost picks up from_year+1 through 2025.
    cumulative_inflation = 1.0
    for year in range(from_year + 1, 2026):
        cumulative_inflation *= 1 + YEARLY_INFLATION[year]
    return cost * cumulative_inflation
