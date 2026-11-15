# Aeration Flexibility

Sizes and operates storage for flexible wastewater aeration: compressed air or O2 tanks, liquid O2, electrolyzers, and
batteries. Storage shifts aeration electricity away from peak tariff periods. For each storage design, a linear
program schedules a month of 15-minute operation. A design search then minimizes lifetime cost (capex plus the NPV of
electricity bills) against a counterfactual that only expands aeration capacity. The methods are in the paper's SI.

## Pipeline

- `step0_make_demo_data.py`: synthetic demo inputs from the IWA BSM1 influent ([data/demo](aeration_flexibility/data/demo/README.md))
- `step1_fit_capex_costs.py`: tank cost curves (`output_data/smoothed_tank_coefficients.json`) and SI figures
- `step2_solver_compare.py`: Solver selection with facility data
- `step3_run_all.py`: design search, operational solves and figures for each config
- `step4_si_methods_figures.py`: SI figures

Step 3 reads `output_data/<config>/day_profiles.json` (15-min air demand and rest-of-facility power per day). It writes
result JSONs to `output_data/<config>/`.

## Setup

Python 3.10–3.12.

```bash
poetry install -E gurobi        # or: poetry install   (HiGHS only)
cp .env.example .env            # optional: cache location, solver
```

Gurobi needs a license (free for academics). Without one, set `AERATION_SOLVER=highs` in `.env`. HiGHS is installed with SciPy.

## Demo

```bash
python aeration_flexibility/step0_make_demo_data.py
python aeration_flexibility/step3_run_all.py demo_iwa_bsm1
```

The demo sizes tank and battery storage for each base technology over one month. It writes
`fig_2stacked_lifetime_cost.png`, `fig3.png` and diagnostics to `output_plots/diagnostics/demo_iwa_bsm1/`. It takes about 7 minutes on 4 cores with either solver.

## Reproducing the paper

`python aeration_flexibility/step3_run_all.py` runs every paper config. The paper's result JSONs and figures are in
`output_data/` and `output_plots/`. Re-solving needs the facility SCADA data, which is not public: `day_profiles.json`
is gitignored for the paper configs.

Solves are cached per day in `AERATION_INTERMEDIATE_DIR/<data_name>/`, so configs on the same data reuse each other's solves.

## Configs

`aeration_flexibility/configs/<name>.json`:

- `upgrade_keys`: for each base technology (`air__compressor`, `o2__psa`, `o2__cryo`), the storage upgrades to size,
  e.g. `compressor__gas_tank__0`, `none__liquid_tank__0`, `elec__gas_tank__0`, `none__battery__0`
- `tariff_keys_to_run`: `{solar fraction}__{utility}_{variant}`, with rates in `data/billing/{utility}.csv`
- `summer_config_template`: dry-season O2 demand `multiplier` and diurnal peaking `smoothing`
- `facility`: `data_name` (input dataset, which names the solve cache), `scale_mgd` (flow scaling label) and
  `plant_capacity_mgd` (bioreactor size)
- `design_space`: search ranges and budget (`o2_range` in storage hours, `comp_ratio_range`,
  `charge_power_ratio_range`, `designs_per_run`, `max_iterations`, `n_jobs`, `horizon_days`)
- `script_settings`: `run_solve_optimization`, `run_plotting`, and `paper_figures`
- `sweep_ranges`, `source_run`, `design_point`: sensitivity runs that sweep tariffs and demand around designs from
  another run, or around a fixed design

## License

MIT. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
