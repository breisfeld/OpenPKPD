# Changelog

All notable changes to OpenPKPD are documented here. This project follows
[Semantic Versioning](https://semver.org).



## 0.3.0 — 2026-06-26

### Added

- **Bundled dataset loaders**: `openpkpd.data.load_theophylline()` and
  `openpkpd.data.load_warfarin()` return the example datasets as
  `NONMEMDataset` objects, so the tutorial and quickstart workflows run without
  a separate data download.
- **Plot convenience exports**: `openpkpd.plots.gof_panel` (an alias for the
  seven-panel `diagnostic_panel` goodness-of-fit figure) and
  `openpkpd.plots.vpc_plot` are now exported from the top-level `openpkpd.plots`
  namespace.

### Fixed

- **SAEM dosing scale**: aligned the SAEM control-stream path with absolute-mg
  dosing so SAEM recovers the FOCEI optimum on the theophylline benchmark. Added
  `tests/integration/test_theophylline.py::test_control_stream_saem_recovers_focei_optimum`
  to guard the dosing-scale fix against regression.
- **GUI packaging**: the maturin build configuration now sets
  `python-packages = ["openpkpd", "openpkpd_gui"]` so the desktop GUI package is
  bundled into the wheel; the `openpkpd-gui` entry point now resolves on a clean
  PyPI install.
- **manylinux release build**: set a CMake policy floor
  (`CMAKE_POLICY_VERSION_MINIMUM=3.5`) so the vendored SUNDIALS sub-projects
  configure under CMake 4.x, and restore host-user ownership of `dist/` after
  the root-owned Docker build step.
- **GOF panel comment**: corrected the `gof_panel` alias comment to describe the
  actual seven-panel GOF figure (was "four-panel").

### Documentation

- **API reference**: documented the bundled-dataset loaders (`data`) and the
  simulation-based VPC plots (`vpc_plot`, `pcvpc_plot`, `stratified_vpc_plot`)
  plus the `gof_panel` alias.


## 0.2.9 — 2026-05-09

### Fixed

- **PyPI project metadata**: updated project URLs from GitLab to GitHub in the
  package metadata so the PyPI project page points to the current canonical
  repository and issue tracker.

### Documentation

- **README**: switched the logo image URL and development clone example to the
  GitHub repository.


## 0.2.8 — 2026-05-09

### Fixed

- **Release automation**: added GitHub Actions workflows for manual multi-platform
  wheel builds and tagged PyPI releases covering manylinux_2_28 x86_64, macOS
  x86_64, macOS arm64, Windows x64, and source distributions.
- **Wheel builds**: passed `--manifest-path rust/Cargo.toml` to all `maturin`
  commands in local build recipes, CI workflows, and the manylinux helper
  script so Rust-extension packaging resolves the correct manifest reliably.
- **manylinux CVODES packaging**: bumped `sundials-sys` from `0.2.3` to
  `0.2.5` and adjusted the Linux build pipeline to keep vendored SUNDIALS
  builds working under current CMake behavior.

### Documentation

- **Capabilities diagram**: switched the generated SVG text to a standard
  sans-serif stack for more consistent rendering across platforms.


## 0.2.7 — 2026-04-02

### Added2

**Steady-state (SS=1, II) dosing — ADVAN1/2/3/4/6**
- ADVAN1: scale bolus by `1/(1-exp(-K·τ))` at periodic SS.
- ADVAN2: per-pole accumulation factors `ss_k`, `ss_ka`; limit form for KA≈K.
- ADVAN3: `_biexp_central_ss()` with per-eigenvalue factors; degenerate
  fallback.
- ADVAN4: `_triexp_oral_ss()` + `_decay_difference_ss()` for per-pole SS oral
  dosing.
- ADVAN6: iterative periodic-orbit solver `_find_ss_state()` — integrates up to
  500 dosing cycles until the pre-dose state converges (rtol 1e-5); verified
  against the ADVAN2 analytical SS result for a 1-cmt oral model.
- All subroutines warn once on SS=1 with infusion dosing (not yet implemented).

**Numerical accuracy tests**
- `TestADVAN4VsODE::test_degenerate_eigenvalue_infusion_matches_ode`: validates
  the `abs(dl) < 1e-10` branch of `_infusion_triexp` against ADVAN6 ODE (rtol
  1e-3).

### Fixed

- **NCA**: `compute_dataset` now filters strictly to EVID==0 rows before
  trapezoidal integration; EVID=2/3/4 dose/reset records no longer corrupt AUC
  estimates.
- **SAEM convergence**: phase-2 phi vector includes full omega lower-triangle
  and sigma diagonal, preventing premature convergence when off-diagonal omegas
  or residual variances drift.
- **SAEM OFV monitoring**: OFV now uses the Rao-Blackwell chain-mean η̂ instead
  of a single chain (chain 0), giving a lower-variance monitoring signal.
- **SAEM n_eta=0 fast-path**: MH E-step loop is skipped entirely for models with
  no random effects, avoiding redundant per-subject RNG calls.
- **IMP ESS feedback**: warn when ESS/isample remains low after reaching
  `MAX_ISAMPLE_DOUBLINGS` so users know to increase `isample` manually.
- **FO Cholesky solve**: replaced explicit `C_i^{-1}·I` construction with a
  Cholesky factorisation that provides both `log|C_i|` and the quadratic form in
  one pass.
- **Laplacian**: removed duplicate `LOG2PI_LOCAL`; now uses shared `LOG2PI`
  constant.
- **FO `hasattr` dead code**:
  `result.message if hasattr(result, "message") else ""` simplified to
  `getattr(result, "message", "")`.
- **pcVPC**: warn when simulated data lacks a PRED column and skip correction
  rather than silently producing uncorrected output.
- **Rust `neg2ll_obs_loop`**: replaced silent min-truncation with an explicit
  length check (returns 1e30 on mismatch); Python-side ValueError guard added.
- **Rust transit PD RHS**: `emax` clamped to `[0, 1]` to prevent negative PD
  production and ODE destabilisation.

### Documentation

- `docs/user_guide/analysis_validation_gaps.md`: added *Recently resolved gaps*
  table.
- `docs/changelog.md`: this entry.



## 0.2.6 — 2026-04-02

### Added

**P1.4 — Native acceleration for user-defined `$DES` ODE models**
- `CompiledDESCallable.as_multidose_probe()` — compiles the user's NM-TRAN
  `$DES` block to Numba `@njit` and wraps it in a piecewise multi-dose
  integration engine. Returns four probe callables (bolus + infusion variants,
  state + sensitivity variants) that satisfy the `_NativeOdeTemplate` contract
  already used by the built-in Rust probes.
- `IndividualModel._try_build_user_ode_template()` — detects ADVAN6 +
  `CompiledDESCallable`, auto-derives the volume parameter name (`V`, `V1`,
  `V2`, or `V3`) and output compartment index, and builds a lazily-cached
  `_NativeOdeTemplate` with `eligible_advans={6}`.
- `IndividualModel._iter_templates()` — prepends the user template to the static
  `_NATIVE_ODE_TEMPLATES` list so all four dispatch loops
  (`_try_native_ode_probe`, `native_advan6_prediction_eta_jacobian`,
  `_native_gauss_newton_hessian`, `_native_eta_objective_value_grad`) pick it up
  automatically.
- `_build_native_ode_contract()` gate updated to admit ADVAN6 +
  `CompiledDESCallable` even when no compiled Rust template is present.
- User ODE template excluded from `__getstate__` (closures cannot be pickled);
  rebuilt lazily in parallel worker processes on first probe call.

The acceleration is **transparent** — existing models using `.des(…)` benefit
automatically when `openpkpd[jit]` is installed. No API changes for users.

Downstream benefits activated for user `$DES` models:
- Single-probe IPRED prediction (replaces full Python `evaluate()` loop)
- Native G_i = ∂IPRED/∂η for FOCE/FOCEI inner loop
- Native Gauss-Newton Hessian for Laplacian/BAYES
- Native eta gradient for IMPMAP MAP optimization

Nine new unit tests in `tests/unit/test_native_cvodes.py` (Section 18): gate
activation, template caching, state-probe accuracy vs. analytical 1-cmt IV (rtol
1e-4), `_try_native_pk_backend` dispatch, sensitivity-probe shape and FD
agreement (rtol 1e-3), and G_i vs. FD reference (rtol 1e-2).

**Documentation**
- Added *"Native acceleration for user-defined `$DES` models"* section to
  `docs/user_guide/pk_subroutines.md` covering activation, eligibility
  conditions, and limitations.
- Updated Model workflow tooltip in the GUI to mention that `openpkpd[jit]`
  enables automatic Numba acceleration for `$DES` models.



## 0.2.5 — 2026-04-01

### Added

**GUI — BLQ/M3 support (P2-C)**
- Added a scalar **LOQ** spinner to the Data workflow options row. When set, the
  value is injected as a constant `LLOQ` column at fit time if the dataset does
  not already contain one.
- Added a **BLQ method** combo to the Model workflow estimation settings row,
  exposing M1 (ignore BLQ, default) and M3 (censored likelihood). The selection
  is persisted in `ModelSpec.estimation.options` and applied automatically at
  fit time by setting `population_model.blq_method`.

**GUI — Interactive GOF subject highlighting (P2-B)**
- Added a **Subject** filter combo to the Diagnostics workflow filter row.
  Selecting a subject ID re-renders the active GOF plot with that subject's
  observations overlaid in red, without leaving the Diagnostics page. Supported
  plot types: DV vs IPRED, DV vs PRED, CWRES vs TIME, CWRES vs PRED, and |IWRES|
  vs IPRED.

**GUI — VPC stratification and pcVPC (P3-F)**
- Added a **Stratify by** combo to the Advanced workflow VPC tab, populated from
  the active dataset columns (mandatory NONMEM columns excluded). The selected
  column is passed to `VPCEngine.compute(stratify_by=...)`. Repopulates and
  restores selection on every refresh.
- Renamed the prediction-corrected checkbox to **pcVPC** and added a descriptive
  tooltip.
- Added `stratify_by: str | None` to `VPCConfig`; run summary text includes
  `stratify=<column>` when stratification is active.

**Documentation**
- Rewrote `docs/user_guide/gui.md` to fully reflect all current workflow pages,
  controls, and behaviors including: LOQ spinner, BLQ method combo, subject
  highlighting, VPC stratification/pcVPC, named model presets, advanced
  FOCE/FOCEI optimizer controls, NCA options, Results comparison and delta
  panels, Diagnostics NPDE controls, Advanced Design tab controls, Artifacts tab
  scope filtering, keyboard shortcuts, and BLQ troubleshooting guidance.



## 0.2.4 — 2026-03-28

### Changed

**Notebooks and documentation**
- Refreshed the full marimo notebook suite for the current APIs, added solver
  and FOCEI advanced-option examples, and strengthened notebook integration
  tests so they check both successful execution and expected outputs.
- Updated the README and user-facing docs to reflect the current example suite,
  notebook extra, GUI review flow, validation coverage, and the new method-level
  validation matrix.

**GUI and examples**
- Added results-page comparison navigation so the GUI can jump directly to a
  strong sibling scenario for side-by-side review.
- Added a PFIM-backed optimal-design example and refreshed the advanced example
  inventory, including the renumbered four-compartment ADVAN5 workflow.

**Validation**
- Expanded external validation with additional advanced-estimator checks,
  PFIM/design reference tests, Monolix benchmark safeguards, and a warfarin
  FOCEI diagnostic harness to document the current validated basin behavior.

## 0.2.3 — 2026-03-28

### Changed

**FOCE / FOCEI estimation**
- Corrected FOCE/FOCEI objective handling for the interaction path and expanded
  analytic and cross-tool regression coverage against `nlmixr2`, `NONMEM`,
  `Monolix`, `PKNCA`, `WinNonlin`, and Pharmpy-backed workflows.
- Added configurable FOCEI outer-optimizer controls, fallback/polish settings,
  best-iterate retention, and structured retry options across the Python API,
  control-stream runtime, and parser.

**GUI and examples**
- Expanded the GUI advanced estimation surface to expose the new FOCEI controls,
  improved error visibility in post-fit workflows, and added regression tests
  for the new behavior.
- Added new runnable examples covering FOCEI optimizer controls, persisted
  control-stream optimizer extensions, phenobarbital population PK, and indometh
  NCA.

**Validation and developer tooling**
- Added stronger example integration checks, live R-backed validation coverage,
  and a local R dependency installer with `just install-r-test-deps` and
  `just check-r-test-deps`.
- Restored the Sphinx docs theme to Read the Docs style and improved release
  tooling so the version bump script now updates Rust and changelog release
  metadata as well.

## 0.2.2 — 2026-03-24

### Changed

**IMP estimation — corrected marginal likelihood normalisation**
- Fixed a systematic bias in `IMPMethod._importance_sample()` where the
  log-prior contribution was missing the `−n_eta/2 · log(2π)` normalisation
  term.
- The bias was proportional to the number of random effects (~11 OFV units per
  ETA per subject for a 12-subject run), causing IMP to converge to a false mode
  or fail to converge at all on models where FOCE/SAEM converge cleanly.
- Updated the Theophylline IMP regression reference from the placeholder value
  (OFV = 5736, theta at initials) to the true minimum (OFV = 3381).

**ETA de-shrinkage (Combes 2013)**
- `EstimationResult.compute_deshrinkage_etas()` returns a subject-keyed dict of
  de-shrunken EBEs using the Combes (2013) rescaling correction:
  `eta_adj_ik = eta_ik / (1 − shrinkage_k)`.
- This adjusts the EBE dispersion to match `sqrt(omega_kk)` exactly, making
  covariate plots and ETA histograms valid even when FOCE shrinkage exceeds 30%.
- A warning note ("Consider de-shrinkage") is now shown in the HTML report for
  any ETA row whose shrinkage exceeds 30%.
- See `docs/user_guide/estimation_methods.md` for full usage documentation and
  the Combes (2013) reference.

**Fast-path error model evaluation (observation-model loop bypass)**
- `IndividualModel._fast_obs_model()` detects standard `$ERROR` patterns
  (proportional, additive, proportional_theta, additive_theta, combined_theta,
  combined_eps) and evaluates them with vectorized NumPy instead of the
  per-observation Python loop.
- The fast path is used automatically when `eps=0` (estimation path); the full
  per-observation loop is retained for simulation.
- Measured on the Theophylline benchmark (12 subjects, 7 obs, proportional
  error): `evaluate_observation_model` reduced from ~50 µs/call to ~18 µs/call
  (~2.8×).



## 0.2.1 — 2026-03-22

### Changed

**Control-stream prior support**
- `$PRIOR` runtime support now includes a documented, tested Gaussian-prior
  subset: `$THETAP`/`$THETAPV` for THETA priors and `$OMEGAP`/`$OMEGAPD` for
  OMEGA lower-triangle priors during control-stream execution.
- OMEGA prior blocks are now wired through `Problem.from_control_stream(...)`
  into `PriorSpec` / `PriorAugmentedModel` rather than remaining parse-only.
- `$SIGMAP*` prior blocks remain parse-only at this stage.

**Control-stream simulation support**
- `$SIMULATION` now has a documented runtime subset in the runner: first-seed
  handling, `ONLYSIMULATION`, `SUBPROBLEMS=n`, and a default `<run>.sim.csv`
  output artifact.
- Simulation-only control streams now execute without an estimation step by
  simulating from the active control-stream parameter state.
- `TRUE=FINAL` remains parseable but is not yet given separate runtime behavior.

**Control-stream mixture support**
- `$MIXTURE` now has a documented runtime subset in the runner for `NSPOP=n`
  finite mixtures using dedicated `<run>.mix.json` and
  `<run>.mix_assignments.csv` artifacts.
- The current subset supports `FO`, `FOCE`/`FOCEI`, and `LAPLACIAN` as the inner
  estimation method.
- `PMIX=THETA(n)` remains parseable but is not yet used to drive runtime mixing.

**GUI VPC workflow**
- The GUI **Advanced** page now provides a real post-fit VPC workflow rather
  than a pure placeholder.
- Users can generate VPC artifacts from the latest successful fit, configure
  replicate/bin/seed settings, request prediction-corrected VPC, and preview the
  latest plot/summary artifacts directly in the GUI.

**GUI bootstrap workflow**
- The GUI **Advanced** page now also provides a post-fit bootstrap workflow.
- Users can generate bootstrap summary, CI-table, and raw-sample artifacts from
  the latest successful fit and configure replicate count, worker count, seed,
  and CI level directly in the GUI.

**GUI design workflow**
- The GUI **Advanced** page now also provides a post-fit optimal-design
  workflow.
- Users can generate design summary, metrics, schedule, FIM, and expected-SE
  artifacts from the latest successful fit and configure sample count, subject
  count, time window, criterion, and optimization method directly in the GUI.

**GUI Advanced hub cleanup**
- The GUI **Advanced** page is now organized as a tabbed hub with dedicated VPC,
  bootstrap, design, and artifact-browser tabs.
- The shared artifact browser now supports workflow-specific filtering while
  preserving direct preview/open/export actions.

**NPDE / VPC validation notes**
- Added a user-guide validation note that links the current NPDE and VPC checks
  to the canonical literature and explains the current validation scope.
- NPDE tests now include a multi-seed misspecification-separation check, and the
  NPDE/VPC regression baselines now carry literature-aligned validation
  metadata.

**NCA validation notes**
- Extended the validation note with the current NCA validation scope, including
  analytic one-compartment checks and reference-workflow formula alignment.
- Added stronger oral-reference NCA assertions for Lambda_z, CL/F, and Vz/F, and
  enriched the NCA regression baseline with validation metadata/provenance.

### Added

**Data handling**
- `CovariateImputer` (`openpkpd.data.impute`): fills missing covariate values
  with `mean`, `median`, `locf` (last-observation-carried-forward), `nocb`
  (next-observation-carried-backward), or `knn` (k-nearest-neighbours via
  scikit-learn).
- `NONMEMDataset.impute_covariates(columns, method='locf')`: convenience wrapper
  that returns a new dataset with imputed values.

**NCA**
- `to_cdisc_pp()` (`openpkpd.nca.cdisc_pp`): converts NCA results to CDISC PP
  domain format (long-format DataFrame with STUDYID, USUBJID, DOMAIN, PARAMCD,
  PARAM, AVAL, DTYPE).
- `SparseNCAEngine` (`openpkpd.nca.sparse`): model-based sparse-sampling NCA.
  Optimises post-hoc ETAs over sparse observations, reconstructs a dense
  predicted profile, then delegates to the standard `NCAEngine`.

**Output**
- `write_cdisc_adppk()` (`openpkpd.output.cdisc_writer`): writes a CDISC
  ADPPK-style CSV with observation rows, population parameter rows
  (THETA/OMEGA/SIGMA), and post-hoc ETA rows.
- `openpkpd.output` package now exports `write_cdisc_adppk`.

**Tests**
- 895 unit tests passing (up from 860).
- New test modules: `tests/unit/data/test_impute.py`,
  `tests/unit/nca/test_cdisc_pp.py`, `tests/unit/nca/test_sparse_nca.py`,
  `tests/unit/output/test_cdisc_writer.py`,
  `tests/unit/prior/test_control_stream_prior.py`.



## 0.1.0 — 2026-03-03

### Added

**Core estimation engine**
- FO, FOCE, FOCEI, Laplacian, SAEM, and IMP estimation methods
- ADVAN1–4 (1-compartment IV/oral, 2-compartment IV/oral) with TRANS1–6
- NM-TRAN code compiler: `$PK` / `$ERROR` blocks → Python callables
- `ModelBuilder` fluent Python API (no `.ctl` file required)
- NONMEM control stream parser (`$PROBLEM`, `$DATA`, `$INPUT`, `$SUBROUTINES`,
  `$PK`, `$ERROR`, `$THETA`, `$OMEGA`, `$SIGMA`, `$ESTIMATION`, `$COVARIANCE`,
  `$TABLE`)
- `NONMEMDataset`: CSV loading with EVID/MDV auto-generation, ADDL/II expansion
- R/S sandwich covariance estimator
- NONMEM 7.x-compatible output files: `.lst`, `.ext`, `.phi`, `.cov`, `.cor`
- `$TABLE` output writer
- CLI: `openpkpd run model.ctl`, `openpkpd parse model.ctl`

**Diagnostic plots** (`OpenPKPD[plots]`)
- `compute_diagnostics()`: PRED/IPRED/CWRES/WRES/IWRES/ETA DataFrame
- GOF: `diagnostic_panel`, `dv_vs_ipred`, `dv_vs_pred`, `cwres_vs_time`,
  `cwres_vs_pred`, `cwres_qq`, `abs_iwres_vs_ipred`
- PK: `spaghetti_plot`, `concentration_time`, `mean_profile`
- PD: `effect_time`, `emax_curve`, `hysteresis_loop`, `pd_individual`
- ETA: `eta_histograms`, `eta_pairs`, `eta_vs_covariate`
- Model performance: `ofv_history`, `vpc`

**Testing**
- 162 tests passing (unit + integration); 1 skipped (regression baseline)
- Unit coverage: parser, data, PK subroutines, estimation, plots
- Integration: theophylline, warfarin, 2-compartment, Emax PD

**Documentation**
- Full Sphinx documentation with ReadTheDocs theme
- Getting started, user guide, 7 annotated examples, API reference
