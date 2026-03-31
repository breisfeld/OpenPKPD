# Changelog

All notable changes to OpenPKPD are documented here.
This project follows [Semantic Versioning](https://semver.org).

---

## 0.2.4 — 2026-03-28

### Changed

**Notebooks and documentation**
- Refreshed the full marimo notebook suite for the current APIs, added solver
  and FOCEI advanced-option examples, and strengthened notebook integration
  tests so they check both successful execution and expected outputs.
- Updated the README and user-facing docs to reflect the current example suite,
  notebook extra, GUI review flow, validation coverage, and the new
  method-level validation matrix.

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
  improved error visibility in post-fit workflows, and added regression tests for
  the new behavior.
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
- Fixed a systematic bias in `IMPMethod._importance_sample()` where the log-prior
  contribution was missing the `−n_eta/2 · log(2π)` normalisation term.
- The bias was proportional to the number of random effects (~11 OFV units per ETA per
  subject for a 12-subject run), causing IMP to converge to a false mode or fail to
  converge at all on models where FOCE/SAEM converge cleanly.
- Updated the Theophylline IMP regression reference from the placeholder value
  (OFV = 5736, theta at initials) to the true minimum (OFV = 3381).

**ETA de-shrinkage (Combes 2013)**
- `EstimationResult.compute_deshrinkage_etas()` returns a subject-keyed dict of
  de-shrunken EBEs using the Combes (2013) rescaling correction:
  `eta_adj_ik = eta_ik / (1 − shrinkage_k)`.
- This adjusts the EBE dispersion to match `sqrt(omega_kk)` exactly, making covariate
  plots and ETA histograms valid even when FOCE shrinkage exceeds 30%.
- A warning note ("Consider de-shrinkage") is now shown in the HTML report for any
  ETA row whose shrinkage exceeds 30%.
- See `docs/user_guide/estimation_methods.md` for full usage documentation and
  the Combes (2013) reference.

**Fast-path error model evaluation (observation-model loop bypass)**
- `IndividualModel._fast_obs_model()` detects standard `$ERROR` patterns
  (proportional, additive, proportional_theta, additive_theta, combined_theta,
  combined_eps) and evaluates them with vectorized NumPy instead of the
  per-observation Python loop.
- The fast path is used automatically when `eps=0` (estimation path); the
  full per-observation loop is retained for simulation.
- Measured on the Theophylline benchmark (12 subjects, 7 obs, proportional error):
  `evaluate_observation_model` reduced from ~50 µs/call to ~18 µs/call (~2.8×).

---

## 0.2.1 — 2026-03-22

### Changed

**Control-stream prior support**
- `$PRIOR` runtime support now includes a documented, tested Gaussian-prior subset:
  `$THETAP`/`$THETAPV` for THETA priors and `$OMEGAP`/`$OMEGAPD` for OMEGA
  lower-triangle priors during control-stream execution.
- OMEGA prior blocks are now wired through `Problem.from_control_stream(...)`
  into `PriorSpec` / `PriorAugmentedModel` rather than remaining parse-only.
- `$SIGMAP*` prior blocks remain parse-only at this stage.

**Control-stream simulation support**
- `$SIMULATION` now has a documented runtime subset in the runner: first-seed handling,
  `ONLYSIMULATION`, `SUBPROBLEMS=n`, and a default `<run>.sim.csv` output artifact.
- Simulation-only control streams now execute without an estimation step by simulating
  from the active control-stream parameter state.
- `TRUE=FINAL` remains parseable but is not yet given separate runtime behavior.

**Control-stream mixture support**
- `$MIXTURE` now has a documented runtime subset in the runner for `NSPOP=n`
  finite mixtures using dedicated `<run>.mix.json` and `<run>.mix_assignments.csv`
  artifacts.
- The current subset supports `FO`, `FOCE`/`FOCEI`, and `LAPLACIAN` as the inner
  estimation method.
- `PMIX=THETA(n)` remains parseable but is not yet used to drive runtime mixing.

**GUI VPC workflow**
- The GUI **Advanced** page now provides a real post-fit VPC workflow rather than
  a pure placeholder.
- Users can generate VPC artifacts from the latest successful fit, configure
  replicate/bin/seed settings, request prediction-corrected VPC, and preview the
  latest plot/summary artifacts directly in the GUI.

**GUI bootstrap workflow**
- The GUI **Advanced** page now also provides a post-fit bootstrap workflow.
- Users can generate bootstrap summary, CI-table, and raw-sample artifacts from
  the latest successful fit and configure replicate count, worker count, seed,
  and CI level directly in the GUI.

**GUI design workflow**
- The GUI **Advanced** page now also provides a post-fit optimal-design workflow.
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
  NPDE/VPC regression baselines now carry literature-aligned validation metadata.

**NCA validation notes**
- Extended the validation note with the current NCA validation scope, including
  analytic one-compartment checks and reference-workflow formula alignment.
- Added stronger oral-reference NCA assertions for Lambda_z, CL/F, and Vz/F,
  and enriched the NCA regression baseline with validation metadata/provenance.

### Added

**Data handling**
- `CovariateImputer` (`openpkpd.data.impute`): fills missing covariate values with
  `mean`, `median`, `locf` (last-observation-carried-forward), `nocb`
  (next-observation-carried-backward), or `knn` (k-nearest-neighbours via scikit-learn).
- `NONMEMDataset.impute_covariates(columns, method='locf')`: convenience wrapper that
  returns a new dataset with imputed values.

**NCA**
- `to_cdisc_pp()` (`openpkpd.nca.cdisc_pp`): converts NCA results to CDISC PP domain
  format (long-format DataFrame with STUDYID, USUBJID, DOMAIN, PARAMCD, PARAM, AVAL, DTYPE).
- `SparseNCAEngine` (`openpkpd.nca.sparse`): model-based sparse-sampling NCA.
  Optimises post-hoc ETAs over sparse observations, reconstructs a dense predicted
  profile, then delegates to the standard `NCAEngine`.

**Output**
- `write_cdisc_adppk()` (`openpkpd.output.cdisc_writer`): writes a CDISC ADPPK-style
  CSV with observation rows, population parameter rows (THETA/OMEGA/SIGMA), and
  post-hoc ETA rows.
- `openpkpd.output` package now exports `write_cdisc_adppk`.

**Tests**
- 895 unit tests passing (up from 860).
- New test modules: `tests/unit/data/test_impute.py`,
  `tests/unit/nca/test_cdisc_pp.py`, `tests/unit/nca/test_sparse_nca.py`,
  `tests/unit/output/test_cdisc_writer.py`, `tests/unit/prior/test_control_stream_prior.py`.

---

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

---

## Planned — 0.3.0

- ADVAN5/7 (general linear, matrix exponential)
- NUTS/BAYES full posterior hardening — currently at experimental maturity
- Trust-region optimizer for improved FOCEI convergence on non-convex surfaces
- CDISC ADPPK domain export
