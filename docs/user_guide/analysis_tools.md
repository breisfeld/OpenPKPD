# Analysis tools and test map

This page inventories the main **analysis-oriented tooling** in OpenPKPD and
links each area to the tests that currently exercise it. The goal is to make
it easy to see which areas already have strong numerical/reference validation
and which areas are covered mostly by unit-level behavioral checks.

## How to read this page

- **Unit** = focused deterministic checks of formulas, invariants, dataclasses,
  and local fitting behavior
- **Integration** = multi-component workflow tests
- **Regression** = fixed-baseline drift detection using stored reference values
- **Validation character** describes whether the tests are mainly:
  - **analytical/reference-heavy**: closed forms, known formulas, SciPy,
    literature-backed expectations, or saved baselines
  - **behavioral**: shape/type/monotonicity/convergence/smoke-style checks

## Extended analysis model families (`src/openpkpd/models/`)

| Tool family | Implementation | Main tests | Validation character |
| --- | --- | --- | --- |
| Time-to-event / survival | `tte.py` (`ConstantHazardModel`, `WeibullHazardModel`, `GompertzHazardModel`, `LogLogisticHazardModel`, `RepeatedTTEModel`) | `tests/unit/models/test_tte.py`, `tests/external_validation/test_extended_models_reference.py` | Strong: unit closed forms + **external validation** against `scipy.stats.expon.sf` / `weibull_min.sf`; S(t) = exp(−H(t)) identity; Weibull(shape=1) = exponential equivalence; fit recovery |
| Count models | `count.py` (`PoissonModel`, `NegativeBinomialModel`, `ZeroInflatedPoissonModel`) | `tests/unit/models/test_count.py`, `tests/external_validation/test_extended_models_reference.py` | Strong: unit coverage + **external validation** against `scipy.stats.poisson` / `nbinom`; NB→Poisson limit; ZIP k=0 identity; fit recovery |
| Ordered categorical / proportional odds | `categorical.py` (`ProportionalOddsModel`) | `tests/unit/models/test_categorical.py`, `tests/external_validation/test_extended_models_reference.py` | Strong: probability normalization, threshold ordering + **external validation** of cumulative logit formula, probability sum to one, directional covariate effect |
| CTMC / Markov / HMM | `categorical.py` (`DiscreteTimeMarkovModel`, `ContinuousTimeMarkovModel`), `markov.py` (`ContinuousTimeHMM`) | `tests/unit/models/test_categorical.py`, `tests/unit/models/test_markov.py`, `tests/external_validation/test_extended_models_reference.py` | Strong: forward algorithm vs direct enumeration, Viterbi vs brute force, DTMM row-stochasticity + **external validation** of CTMC transition matrix against `scipy.linalg.expm`, Chapman-Kolmogorov identity |
| Direct and mechanistic PD / PK-PD | `pkpd.py` (`LinearPDModel`, `EmaxModel`, `HillModel`, `InhibEmaxModel`, `IndirectResponseModel`, `EffectCompartmentModel`, `TurnoverModel`, `PlaceboResponseModel`, `TumorGrowthInhibitionModel`, `SequentialPKPDWorkflow`) | `tests/unit/models/test_pkpd.py`, `tests/unit/models/test_sequential.py`, `tests/integration/test_emax_pd.py` | Broad coverage: direct-model parameter recovery on simulated data, closed-form checks for effect-compartment/turnover/placebo, qualitative TGI behavior, sequential equivalence, plus FO integration checks for 1-cmt IV + Emax PD |
| Population PD | `population_pd.py` (`PopulationPDModel`) | `tests/unit/models/test_population_pd.py` | Strong numerical unit coverage: pack/unpack round-trip, multi-seed parameter recovery for population Emax/inhibitory Emax, sigma²/omega recovery, fixed-variance behavior |
| TMDD | `tmdd.py` (`FullTMDD`, `QSSATMDDModel`, `MichaelisMentenTMDD`) | `tests/unit/models/test_tmdd.py`, `tests/external_validation/test_extended_models_reference.py` | Strong: unit checks + **external validation** of FullTMDD→ADVAN1 when kon=0, QSSA→ADVAN1 when kint=0, Michaelis-Menten linear limit at low C, nonlinear saturation at high C, mass conservation |
| Static DDI analysis | `ddi.py` (`competitive_inhibition_r`, `time_dependent_inhibition_r`, `induction_r`, `DDIStudyAnalysis`) | `tests/unit/models/test_ddi.py` | Strong formula-level unit checks: known values, asymptotic limits, monotonicity, and round-trip back-calculation of perpetrator parameters |

## Diagnostics and simulation-based analysis

| Tool family | Implementation | Main tests | Validation character |
| --- | --- | --- | --- |
| Diagnostics tables | `plots/diagnostics.py` (`compute_diagnostics`, `_finite_diff_jacobian`, `_cwres_subject`, `compute_npde`) | `tests/unit/plots/test_diagnostics.py`, `tests/unit/plots/test_plots.py` | Mostly deterministic unit coverage: Jacobian correctness, CWRES fallback, duplicate-time alignment, covariate preservation, required diagnostics columns |
| VPC / pcVPC | `simulation/vpc.py` (`VPCEngine`, prediction correction helpers) | `tests/unit/simulation/test_pcvpc.py`, `tests/integration/test_vpc_pipeline.py`, `tests/regression/test_diagnostics_regression.py`, `tests/external_validation/test_diagnostics_reference.py` | Strongest validation in this area: prediction-correction arithmetic, end-to-end percentile-band behavior, misspecification sensitivity, regression baselines, and **external validation** of 90% PI coverage under correct model |
| NPDE | `simulation/npde.py` (`NPDEEngine`) | `tests/unit/simulation/test_npde.py`, `tests/regression/test_diagnostics_regression.py`, `tests/external_validation/test_diagnostics_reference.py` | Strong: helper math, decorrelation, calibration, misspecification, regression baselines + **external validation** of Φ⁻¹(pd) vs `scipy.stats.norm.ppf`, Cholesky whitening identity, boundary clipping, NPDE symmetry |
| NPC | `simulation/npc.py` (`NPCEngine`) | `tests/unit/simulation/test_npc.py`, `tests/external_validation/test_diagnostics_reference.py` | Unit coverage + **external validation** of two-sided p-value formula = 2·min(p_below, 1−p_below), and KS-test uniformity under correctly specified model |
| SSE | `simulation/sse.py` (`SSEEngine`, `_empirical_coverage`) | `tests/unit/simulation/test_sse.py` | Deterministic unit coverage for bias/RMSE/coverage math and re-estimation routing; currently less externally anchored than VPC/NPDE/NCA |
| Plotting helpers | `plots/gof.py`, `plots/pk.py`, `plots/pd.py`, `plots/eta.py`, `plots/model_perf.py`, `plots/simulation.py` | `tests/unit/plots/test_plots.py`, `tests/unit/plots/test_simulation_plots.py` | Mixed: many figure-return/smoke tests, with a smaller set of explicit numerical checks for plotted values, labels, identity lines, and custom percentile handling |

### Profiling helpers for analysis workloads

- `scripts/profile_analysis.py` profiles representative analysis routines without
  needing a full external benchmark harness.
- The new `diagnostics_covariate` workload builds a static-covariate
  `ADVAN2/TRANS2` model with SCM-style power/linear/exponential effects and runs
  diagnostics twice: once with symbolic ETA Jacobians enabled and once with the
  finite-difference fallback forced.
- Use it to measure both wall time and path selection (`supports_prediction_eta_jacobian`,
  kernel name, Jacobian stage totals).
- Example:
  `uv run python scripts/profile_analysis.py --workloads diagnostics_covariate --covariate-subjects 140 --json-out artifacts/profiling/diagnostics_covariate.json`

## Non-compartmental and related analysis utilities

| Tool family | Implementation | Main tests | Validation character |
| --- | --- | --- | --- |
| Core dense-profile NCA | `nca/nca.py` (`NCAEngine`) | `tests/unit/nca/test_nca.py`, `tests/regression/test_diagnostics_regression.py` | Very strong numerical/reference coverage: Cmax/Tmax, exact monoexponential AUC/half-life/CL/Vz/MRT checks, partial AUC closed forms, and regression baselines |
| Multidose and BLQ handling | `nca/nca.py` multidose helpers | `tests/unit/nca/test_multidose_nca.py` | Unit-level numerical checks for partial AUC identities, multidose summary parameters, dose normalization, and predose BLQ rules |
| Sparse NCA | `nca/sparse.py` (`SparseNCAEngine`) | `tests/unit/nca/test_sparse_nca.py` | Analytical-reference checks against 1-cmt IV bolus truth, including AUC recovery and exact dose scaling under fallback prediction |
| Urine NCA | `nca/urine.py` (`UrineNCAEngine`) | `tests/unit/nca/test_urine_nca.py` | Good numerical unit coverage: interval-rate formulas, exact Ae∞/fraction excreted/renal-clearance equations, and fallback behavior for nonpositive `lambda_z` |
| Crossover BE / power / sample size | `nca/crossover.py` | `tests/unit/nca/test_crossover.py` | Behavioral and quantitative checks: GMR/CI structure, power monotonicity, and sample-size target attainment |
| CDISC PP export | `nca/cdisc_pp.py` | `tests/unit/nca/test_cdisc_pp.py` | Structural mapping checks for PP-domain export |

## Cross-method and regression validation outside the model-family unit tests

| Validation layer | Main tests | What it contributes |
| --- | --- | --- |
| PK/PD workflow integration | `tests/integration/test_emax_pd.py` | Simulates 1-cmt IV PK + direct Emax PD data; checks finite OFV, positivity, dose-response ordering, and multi-seed/multi-scenario PK/PD recovery |
| PK pipeline integration — ADVAN2/4 | `tests/integration/test_pk_integration.py` | FOCE end-to-end on synthetic theophylline-like data: convergence, OFV finiteness, CL/V/KA in physiological range, OMEGA positive, ETAs finite; 1-cmt oral (ADVAN2) and 2-cmt oral (ADVAN4) |
| VPC workflow integration | `tests/integration/test_vpc_pipeline.py` | Exercises `PopulationModel` + `SimulationEngine` + `VPCEngine`; checks percentile structures and coverage degradation under misspecified clearance |
| Diagnostic regression baselines | `tests/regression/test_diagnostics_regression.py` | Locks VPC, NPDE, and NCA summaries to JSON reference runs in `tests/regression/reference_runs/` |
| Cross-estimation consistency | `tests/regression/test_cross_method_validation.py` | Compares FOCE, Laplacian, and Nonparametric estimation on a shared dataset for OFV relationships, THETA agreement, PSD covariance behavior, and NP weight normalization |
| Estimation OFV formula validation | `tests/external_validation/test_estimation_reference.py` | Closed-form verification of the FOCE OFV formula (including prior_const = n_eta·log(2π) + log\|Ω\|), Laplacian correction (OFV_base + log\|H\|), and IMP convergence to analytic marginal in linear-Gaussian case |
| PK subroutine formula validation | `tests/external_validation/test_pk_subroutines_reference.py` | ADVAN1/2/3 vs closed-form equations (Bateman, biexponential); post-infusion ODE cross-validation via `scipy.integrate.odeint`; half-life, Tmax, and superposition identities |
| Covariance estimator validation | `tests/external_validation/test_covariance_reference.py` | Sandwich covariance structural properties (symmetry, PSD, SE=√diag); HC0 reference checks for R matrix, S matrix, Ĉ=R⁻¹SR⁻¹, and scale-invariance of the SR estimator |
| Extended model external validation | `tests/external_validation/test_extended_models_reference.py` | TTE, count, proportional-odds, CTMC, and TMDD against closed-form or scipy references (see extended model rows above) |
| SAEM behavior validation | `tests/external_validation/test_saem_reference.py` | SAEM returns finite OFV, populates OFV history, and works correctly with multiple chains on a linear-Gaussian mock |

## Current strengths

- **Strongest numerical/reference coverage today** is in:
  - NCA (`test_nca.py` + diagnostics regression)
  - VPC / NPDE (`test_vpc_pipeline.py`, `test_npde.py`, diagnostics regression + external validation)
  - NPDE and NPC formula validation (`test_diagnostics_reference.py`)
  - DDI static formulas (`test_ddi.py`)
  - TMDD limit-case equivalence — unit checks + external validation against ADVAN1 (`test_extended_models_reference.py`)
  - HMM/CTMC small-example exact checks + `scipy.linalg.expm` cross-validation (`test_extended_models_reference.py`)
  - TTE/count/categorical models — unit coverage + scipy closed-form external validation
  - PK subroutines ADVAN1/2/3 — closed-form and ODE cross-validation (`test_pk_subroutines_reference.py`)
  - Estimation formulas — FOCE, Laplacian, IMP OFV formulas verified against linear-Gaussian reference (`test_estimation_reference.py`)
  - Sandwich covariance — structural properties + HC0 reference checks (`test_covariance_reference.py`)
  - Population PD recovery tests across seeds (`test_population_pd.py`)

- **More behavior-oriented areas** remain:
  - plotting APIs, where many tests still focus on figure creation and layout
  - SSE, which has solid helper tests but less reference-run anchoring than VPC/NPDE/NCA

## Related documentation

- `docs/user_guide/validation.md` summarizes the current literature-backed
  validation claims for NCA, VPC, and NPDE.
- `docs/user_guide/analysis_validation_gaps.md` prioritizes the biggest
  remaining numerical-validation gaps and the best next tests to add.
- The test files listed above are the best source for the exact numerical
  acceptance criteria currently enforced in CI.