# Validation notes

This page summarizes the current **literature-backed validation scope** for the
simulation-diagnostic workflows that already have dedicated regression or
integration coverage.

## NPDE

OpenPKPD's NPDE implementation is currently validated against the core behavior
described in the canonical NPDE references:

- Brendel K, Comets E, Laffont C, Laveille C, Mentré F (2006), *Metrics for
  external model evaluation with an application to the population
  pharmacokinetics of gliclazide*
- Comets E, Brendel K, Mentré F (2008), *Computing normalised prediction
  distribution errors to evaluate nonlinear mixed-effect models: the npde
  add-on package for R*

Current OpenPKPD checks cover:

- calibration under a correctly specified simulation model
- stability of NPDE mean/variance across small scenario grids
- strong separation under clear scale misspecification across multiple seeds
- regression drift detection via `tests/regression/reference_runs/diagnostic_npde.json`

These checks live primarily in:

- `tests/unit/simulation/test_npde.py`
- `tests/regression/test_diagnostics_regression.py`

## VPC / pcVPC principles

OpenPKPD's VPC validation currently tracks the core expectations from the
standard VPC and prediction-corrected VPC literature:

- Karlsson MO, Holford N (2008), *A tutorial on visual predictive checks*
- Bergstrand M, Hooker AC, Wallin JE, Karlsson MO (2011), *Prediction-corrected
  visual predictive checks for diagnosing nonlinear mixed-effects models*

Current OpenPKPD checks cover:

- regression stability of observed vs simulated percentile summaries
- coverage-style checks for observed median percentiles against simulated bands
- sensitivity to clear clearance misspecification across multiple seeds

These checks live primarily in:

- `tests/integration/test_vpc_pipeline.py`
- `tests/regression/test_diagnostics_regression.py`

## NCA

OpenPKPD's dense-profile NCA checks currently align with common industry NCA
parameter definitions, public PKNCA summaries, and analytic one-compartment
reference behavior.

Reference anchors currently used for D2 are:

- Certara Phoenix WinNonlin NCA parameter formulas (AUClast, Lambda_z,
  AUCINF, CL/F, Vz/F, MRT definitions)
- PKNCA usage and defaults documentation for standard interval selection and
  lin-up/log-down calculation conventions
- Han S (2018), *Validation of Noncompartmental Analysis Performed by
  NonCompart R package*, for published WinNonlin-backed Indometh reference
  tables
- Gabrielsson & Weiner's standard NCA textbook conventions for derived PK
  endpoints

Current OpenPKPD checks cover:

- exact or near-exact agreement with closed-form IV bolus monoexponential
  reference values for AUC, half-life, clearance, volume, and MRT
- oral theophylline benchmark checks against public PKNCA summaries for
  AUClast(0-24), Cmax, Tmax, half-life, and AUCinf.obs using the
  `linear-up-log-down` method family
- published Indometh zero-start core benchmark checks for `R²`, Lambda_z,
  `t½`, `Cmax`, `Tmax`, `AUClast`, `AUCinf`, `CL`, and `Vz` against
  WinNonlin-backed tables
- published Indometh IV bolus benchmark checks for back-extrapolated `C0`,
  `AUClast`, `AUMClast`, `AUCinf`, `AUMCinf`, `CL`, `Vz`, and `MRT` against
  WinNonlin-backed tables
- published Indometh IV infusion benchmark checks for `R²`, Lambda_z, `t½`,
  `Cmax`, `Tmax`, `AUClast`, `AUMClast`, `AUCinf`, `AUMCinf`, `CL`, `Vz`, and
  infusion-adjusted `MRT` against WinNonlin-backed tables
- published Indometh extravascular benchmark checks for the zero-start,
  `AUMC`, and `MRT` endpoints against WinNonlin-backed tables
- oral one-compartment theophylline-like benchmark checks for AUCinf, Tmax,
  half-life, Lambda_z, CL/F, and Vz/F
- deterministic regression drift detection via
  `tests/regression/reference_runs/diagnostic_nca.json`

These checks live primarily in:

- `tests/unit/nca/test_nca.py`
- `tests/external_validation/test_vs_pknca.py`
- `tests/external_validation/test_vs_winnonlin_indometh.py`
- `tests/regression/test_diagnostics_regression.py`

## SAEM / Monolix parity

OpenPKPD now includes a public-theophylline SAEM parity check against Monolix
project parameters exposed through the `monolix2rx` conversion examples.

Reference anchors currently used are:

- public Monolix theophylline dataset/project documentation
- `monolix2rx` conversion outputs that expose the final Monolix fixed effects
  for the bundled theophylline project

Current OpenPKPD checks cover:

- SAEM recovery of the public Monolix theophylline `ka`, `Cl`, and `V`
  population parameters after matching Monolix's mg/kg dose convention
- continued stochastic-averaging stability checks on the same theophylline fit

These checks live primarily in:

- `tests/external_validation/test_vs_monolix.py`
- `tests/external_validation/test_saem_reference.py`

## What these validation milestones do and do not claim

The current D1/D2 milestone work should be read as **method-level external-reference
validation**, not full external parity certification.

What is covered now:

- literature-aligned expectations for NPDE calibration and misspecification
  sensitivity
- literature-aligned expectations for VPC percentile-band behavior and
  misspecification sensitivity
- analytic and reference-workflow-aligned expectations for core dense-profile
  NCA endpoints
- public cross-tool parity checks for Monolix SAEM, PKNCA/Phoenix-style NCA,
  and WinNonlin-backed Indometh NCA tables
- explicit provenance links in the diagnostic regression baselines

What is **not** yet covered:

- cross-software parity against proprietary WinNonlin executable outputs or a
  full vendor validation suite distributed with the software
- broad cross-software parity against NONMEM / PsN / `vpc` / `npde` outputs on
  the same external dataset
- formal acceptance envelopes derived from regulatory or consortium reference
  suites

Those broader comparisons remain future work for later validation milestones.

For the concrete benchmark cases and current findings, see
`docs/user_guide/external_validation_benchmarks.md`.