# Analysis validation gaps and next tests

This page is a follow-up to `analysis_tools.md`. It focuses on two questions:

1. **Where are the biggest remaining numerical-validation gaps?**
2. **What are the highest-value tests to add next?**

Here, a *strong* validation test means one of the following:

- exact comparison to a closed-form result
- comparison to an independent implementation or established solver
- comparison to a literature- or guidance-backed worked example
- regression against a stored reference run with stable quantitative tolerances

## Biggest remaining gaps

| Priority | Area | Current state | Main gap |
| --- | --- | --- | --- |
| P1 | `simulation/engine.py` | Tests cover structure, seed reproducibility, REP semantics, and plausible values | Validation is still mostly structural; there are few exact checks that simulated trajectories/noise match analytical expectations under controlled conditions |
| P1 | `plots/` and `plots/simulation.py` | Many tests assert “returns `Figure`” and basic labels/layout | The plotting layer is still the most smoke-test-heavy part of analysis coverage; few tests extract plotted data and verify exact lines/bands/points |
| P2 | SSE | Helper math and high-level plausibility are tested | Lacks stored regression baselines and calibration-style external checks already present for VPC, NPDE, and NCA |
| P2 | Bioequivalence / RSABE / crossover | Logic and edge-case behavior are covered well | There are few direct worked-example comparisons against regulatory or textbook examples with fixed expected GMR/CI/pass-fail outcomes |
| P2 | Advanced estimator external benchmarking | SAEM/IMP/BAYES/nonparametric now have unit tests, regression baselines, four empirical cross-tool datasets (theophylline via nlmixr2, PK-only `warfarin` via nlmixr2, reduced mixed-endpoint joint `warfarin` PK/PD via nlmixr2 with both 4-subject release-gated and 6-subject second-tier variants, and phenobarbital `pheno` via Pharmpy), and exact Gaussian / EM external-reference tests for BAYES, NUTS, and NPML components | External benchmark breadth is still thinner than for the core FO/FOCE path; the new mixed-endpoint empirical coverage is still limited to reduced runtime-practical subsets rather than a full mixed-effects benchmark |
| P3 | Analysis-model fit workflows in `src/openpkpd/models/` | TTE, count, categorical, and TMDD families now have **external validation** against scipy closed forms and limit-case reductions | Fixed-dataset fit baselines for PD model families (Emax, indirect response) are still missing |

## Why these are the biggest gaps

### Simulation is foundational to downstream validation

VPC, NPDE, NPC, SSE, and many diagnostics depend on `SimulationEngine` and the
plotting stack. If those layers are only checked structurally, higher-level
analysis tests can still pass while subtle numerical drift slips through.

### External validation has grown significantly but gaps remain

The most recent test additions expanded external-validation coverage to:

- **PK subroutines**: ADVAN1/2/3 closed-form formula checks and ODE cross-validation
- **Estimation OFV formulas**: FOCE, Laplacian, IMP verified against linear-Gaussian
  closed-form references
- **Sandwich covariance**: structural properties and HC0 reference
- **NPDE/NPC formulas**: Φ⁻¹(pd) vs scipy, Cholesky whitening, p-value identity
- **Extended model families**: TTE vs scipy.stats, count models vs scipy.stats,
  CTMC vs scipy.linalg.expm, TMDD limit reductions

Remaining external-validation gaps are mainly in the PD model fit workflows and
in broader multi-endpoint / more-complex empirical coverage beyond the current
reduced `warfarin` PK/PD benchmarks.

### Mixed-endpoint benchmark roadmap

The current benchmark ladder is:

- **4-subject reduced FO benchmark** — release-gated practical path
- **6-subject reduced FO benchmark** — broader second-tier validation path
- **full 32-subject mixed-effects benchmark** — long-term empirical target once
  runtime and stability are good enough for routine validation

## Best next tests to add

### 1. Exact analytical checks for `SimulationEngine`

**Targets:** `tests/unit/simulation/test_engine.py`

Add controlled no-noise tests where simulated `IPRED`/`DV` should match exact
ADVAN2/ADVAN3 solutions under:

- single IV bolus
- oral dosing with known Bateman curve
- repeated doses / multiple event rows
- `simulate_new_design()` with fixed observation grids

This is the single highest-leverage addition because VPC/NPDE/NPC/SSE all build
on top of these simulations.

### 2. Add stored regression baselines for SSE

**Targets:** `tests/regression/test_diagnostics_regression.py`,
`tests/regression/reference_runs/`

Mirror the existing VPC/NPDE/NCA/NPC pattern for:

- SSE bias / RMSE / empirical coverage summaries

These should run on a fixed seed and fixed design so drift is caught early.

### 3. Convert plot smoke tests into data-bearing assertions

**Targets:** `tests/unit/plots/test_plots.py`, `tests/unit/plots/test_simulation_plots.py`

For the highest-value plots, assert the plotted numerical content, not just that
the function returns a figure:

- VPC bands and observed percentile traces
- QQ/reference lines
- mean-profile lines and error bars
- Emax overlays and hysteresis point ordering
- prediction-interval ribbons in simulation plots

Prefer extracting artist data (`Line2D`, collections, patches) over image-based
golden tests.

### 4. Fixed-dataset regression fits for PD / PK-PD model families

**Targets:** new regression tests under `tests/regression/`

Add one or two saved reference datasets and result baselines for:

- `EmaxModel`
- `HillModel`
- one indirect-response model
- `PopulationPDModel`

Acceptance criteria should include final parameter vectors, OFV, and key derived
predictions within explicit tolerances.

### 5. TMDD regime-validation grid *(partially addressed)*

**Targets:** `tests/unit/models/test_tmdd.py`, `tests/external_validation/test_extended_models_reference.py`

The external validation tests already cover Full→ADVAN1 and QSSA→ADVAN1 limit
reductions and the Michaelis-Menten linear/nonlinear regime. The remaining gap is
a systematic grid of parameter regimes with explicit error thresholds on
concentration trajectories for:

- Full TMDD ≈ QSSA TMDD
- Full TMDD ≈ Michaelis-Menten approximation

### 6. Regulatory/worked-example tests for ABE, RSABE, and crossover tools

**Targets:** `tests/unit/nca/test_nca.py`, `tests/unit/nca/test_rsabe.py`,
`tests/unit/nca/test_crossover.py`

Add fixed examples with precomputed expected results for:

- GMR and 90% CI
- exact pass/fail near 80–125% bounds
- RSABE scaled criterion and upper confidence bound
- crossover power / sample-size outputs at selected design points

This would turn currently good logic tests into externally anchored validation.

### 7. Second-dataset regression matrix for SAEM / IMP / BAYES / nonparametric

**Targets:** `tests/regression/test_regression.py`, `tests/regression/reference_runs/`

Keep the current theophylline baseline, but add a second benchmark dataset so
advanced methods are not validated on only one synthetic reference problem.

### 8. Misspecification-detection thresholds for SSE

**Targets:** `tests/unit/simulation/test_sse.py`

VPC, NPDE, and NPC already test sensitivity to wrong models. Add the same style
of quantitative misspecification checks for SSE so it is judged by its ability
to detect parameter bias/coverage degradation under the wrong model.

## Previously addressed gaps *(completed)*

| Gap | Implementation | Tests added |
| --- | --- | --- |
| TTE likelihood validation against scipy | Closed-form survival and hazard vs `scipy.stats.expon.sf` / `weibull_min.sf` | `tests/external_validation/test_extended_models_reference.py` |
| Count model likelihood validation against scipy | Poisson/NegBin/ZIP PMF vs `scipy.stats.poisson` / `nbinom` | `tests/external_validation/test_extended_models_reference.py` |
| Categorical/CTMC exact checks | Cumulative logit formula; CTMC transition matrix vs `scipy.linalg.expm`; Chapman-Kolmogorov | `tests/external_validation/test_extended_models_reference.py` |
| TMDD limit reductions | Full→ADVAN1 (kon=0), QSSA→ADVAN1 (kint=0), MM linear/nonlinear regimes, mass conservation | `tests/external_validation/test_extended_models_reference.py` |
| FOCE/Laplacian/IMP OFV formula verification | Closed-form linear-Gaussian reference for all three methods | `tests/external_validation/test_estimation_reference.py` |
| PK subroutine closed-form validation | ADVAN1/2/3 vs analytic equations; scipy.odeint cross-validation | `tests/external_validation/test_pk_subroutines_reference.py` |
| Sandwich covariance structural properties | Symmetry, PSD, SE=√diag, cor diagonal, R/S/Ĉ reference formulas | `tests/external_validation/test_covariance_reference.py` |
| NPDE/NPC formula validation | Φ⁻¹(pd) vs scipy, whitening identity, two-sided p-value formula, KS uniformity | `tests/external_validation/test_diagnostics_reference.py` |

## Suggested implementation order

If adding tests incrementally, the best return-on-effort order is:

1. `SimulationEngine` exact analytical checks
2. SSE regression baselines
3. data-bearing plot assertions
4. PD / PK-PD fixed-dataset regression fits
5. TMDD regime grid (systematic parameter sweep)
6. BE / RSABE / crossover worked examples
7. second-dataset advanced-estimator regression matrix

## Related pages

- `docs/user_guide/analysis_tools.md`
- `docs/user_guide/validation.md`