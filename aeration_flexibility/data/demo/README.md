# Demo influent: IWA BSM1 dry-weather profile

`bsm1_dry_weather_influent.csv` is the dry-weather influent of the IWA Benchmark Simulation Model No. 1 (BSM1):
14 days at 15-minute steps of flow `Q` (m³/day) and ASM1 influent states (g COD/m³ or g N/m³). It was copied from
QSDsan (`qsdsan/data/sanunit_data/_inf_dry_2006.tsv`), with tabs converted to commas. The values are unchanged.

The data is synthetic. It does not describe any real facility. It lets someone clone this repository and run the
pipeline end-to-end without the private facility data the paper used.

## Use

`aeration_flexibility/step0_make_demo_data.py` scales this profile to an 11 MGD plant over July 2023 and converts its
BOD and nitrogen loads to an O2 demand and air flow (SI eq. S1–S2). It writes `output_data/demo_iwa_bsm1/`. The
assumptions are commented at the top of the script.

## Citations

- Alex, J., Benedetti, L., Copp, J., Gernaey, K.V., Jeppsson, U., Nopens, I., Pons, M.-N., Rieger, L., Rosen, C.,
  Steyer, J.-P., Vanrolleghem, P., Winkler, S. (2008). *Benchmark Simulation Model no. 1 (BSM1)*. Technical report,
  Department of Industrial Electrical Engineering and Automation, Lund University.
- Gernaey, K.V., Jeppsson, U., Vanrolleghem, P.A., Copp, J.B. (2014). *Benchmarking of Control Strategies for
  Wastewater Treatment Plants*. IWA Scientific and Technical Report No. 23. IWA Publishing.
- IWA Task Group on Benchmarking of Control Strategies for WWTPs: https://iwa-mia.org/benchmarking/
- Li, Y., Zhang, X., Morgan, V.L., Lohman, H.A.C., Rowles, L.S., Mittal, S., Kogler, A., Cusick, R.D., Tarpeh, W.A.,
  Guest, J.S. (2022). QSDsan: an integrated platform for quantitative sustainable design of sanitation and resource
  recovery systems. *Environmental Science: Water Research & Technology* 8(10), 2289–2303.
  https://doi.org/10.1039/d2ew00455k (source of the data file; QSDsan is released under the University of
  Illinois/NCSA Open Source License)
