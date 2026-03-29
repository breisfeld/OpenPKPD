# OpenPKPD — Remaining Priorities
*Last updated: 2026-03-29 (full codebase reanalysis)*

This document is generated from the current source tree, tests, examples,
docs, and benchmark evidence. It supersedes earlier drafts. For detailed
execution tickets with timestamped progress notes see `todo/remaining_priorities.md`.

---

## What Is Now Fully Shipped

The following should NOT be treated as open items. They are real, tested,
and documented:

| Area | State |
|------|-------|
| FO / FOCE / FOCEI / Laplacian estimation | ✅ Production-ready with external validation |
| SAEM | ✅ Validated vs Monolix on Theophylline |
| IMP / IMPMAP | ✅ IMPMAP warm-starts from FOCEI; empirical Warfarin benchmark |
| Bayesian — Laplace backend | ✅ Validated on NONMEM 402 two-compartment IV |
| Bayesian — pure-NumPy NUTS | ✅ FOCE marginal likelihood; analytic symbolic gradients for ADVAN1–4; warm-start eta cache; benchmark shows 8.5 s/12-draw on 6-subject oral-PK |
| Nonparametric NPML | ✅ Pharmpy-backed pheno dataset benchmark |
| Analytical PK (ADVAN1–4, 11, 12) | ✅ |
| ODE PK (ADVAN6/8/13 with forward sensitivity) | ✅ JAX dependency removed; sensitivity via FD on des_callable |
| Symbolic derivative kernels (ADVAN1–4) | ✅ SymPy-compiled kernels with disk caching; activated for NUTS |
| NONMEM control-stream parse + runtime subset | ✅ $PRIOR, $SIMULATION, $MIXTURE runtime supported and tested |
| Desktop GUI: data, model, fit, diagnostics, NCA, covariate, advanced | ✅ |
| GUI: Bayesian artifacts (MCMC traces, R-hat, ESS, posterior density) | ✅ |
| MCMC diagnostics (split-R-hat, ESS, autocorr plots) | ✅ |
| Benchmarking entry points | ✅ `just benchmark-estimation`, `just profile-analysis` |
| Monolix migration guide | ✅ `docs/user_guide/monolix_migration.md` |
| SBML import, HTML reports, NONMEM-like output | ✅ |

---

## Confirmed Bugs Fixed This Session

Found and fixed during codebase reanalysis (`cb250ad3`):

1. **`_fast_obs_model` combined-EPS variance** — `individual.py` fast path for
   `Y = F + EPS(1) + F·EPS(2)` was computing `var = σ₀₀ + F²·σ₁₁`, ignoring
   the `2·F·σ₀₁` off-diagonal term. Symbolic kernel and slow path were correct;
   fast estimation path was not. Fixed. (4 failing tests now pass.)

2. **`DerivativeKernelCapabilities` over-strict equality check** — Test helper
   `_assert_symbolic_derivatives_match_fd` required exact capability equality.
   ADVAN1/ADVAN2 kernels gained `theta_data_objective_gradient=True` when symbolic
   derivatives were activated, causing 22 failures. Fixed to check minimum required
   capabilities. (22 failures resolved.)

3. **Control-stream library test not updated** — `38_prior_gaussian_subset.ctl`
   and `39_onlysimulation_subproblems.ctl` were added to `examples/control_streams/`
   but `EXPECTED_EXAMPLE_FILES` was not updated and the simulation-only file has no
   `$ESTIMATION` block. Fixed both. (2 failures resolved.)

---

## P0 — Highest-Value Remaining Work

These are the items most limiting competitive credibility:

### 1. Advanced-estimator empirical validation breadth (RP-001, open)

FOCE/FOCEI external validation is strong. The thinner areas:

- `SAEM` — validated only on Theophylline; no second published dataset
- `IMP / IMPMAP` — Warfarin benchmark exists but no external reference (NONMEM/Monolix)
- `BAYES(NUTS)` — theophylline second-tier benchmark exists; R-hat still 1.19 at
  `n_samples=24`; not yet positioned as a primary release path
- Nonparametric — pheno dataset only; no second dataset or method comparison

Next action: add one empirical dataset per method family with an external anchor.

### 2. Bayesian/NUTS maturity (RP-004, partially complete)

Current supported NUTS surface (documented):

- Samples theta only; Omega and Sigma are held fixed
- Uses FOCE marginal likelihood (correct); FD gradient (expensive but working)
- Analytic symbolic gradient active for ADVAN1–4 models via `derivative_kernels.py`
- Exact-theta LRU cache cuts repeated FOCE evaluations within leapfrog tree
- Warm-start eta cache reduces cold inner-loop starts from ~200 to 1–5 iters
- `nuts_estimate()` single-chain wrapper: R-hat diagnostics are now real (not ones)
- Positioned as dependency-free second-tier path; `BAYES(Laplace)` recommended for speed

Remaining gaps:
- Omega/Sigma not sampled → posterior is marginal theta only, not full joint NLME posterior
- FD gradient still costs `2·n_theta` FOCE evals per leapfrog step for models without symbolic kernels
- ADVAN5–13 and ODE models fall back to pure FD (symbolic kernels only cover ADVAN1–4)

### 3. Parser/runtime/export support contract (RP-002, mostly complete)

Significant progress this sprint: `$PRIOR`, `$SIMULATION`, `$MIXTURE` all have
runtime integration tests. Remaining:

- `to_nmtran()` / round-trip coverage is incomplete for some record families
  (ABBREVIATED, NONPARAMETRIC, SIZES, DESIGN, CONTR are parse-only)
- No tested round-trip for all `$TABLE` format variants
- Migration docs and README should be checked against actual supported subset

### 4. Benchmark suite (RP-003, mostly complete)

`just benchmark-estimation` covers FO/FOCE/FOCEI/SAEM/IMP/BAYES/NUTS.
Gaps:
- No stored baseline artifacts checked into repo (current outputs are ephemeral)
- Simulation/VPC benchmarks missing
- No profiling report for FOCE outer loop or GUI artifact rendering

### 5. GUI result review and comparison (RP-005, partially complete)

Bayesian artifacts registered (MCMC trace, R-hat, ESS, posterior density, forest).
Gaps:
- Run-to-run comparison artifact diff workflow not fully polished
- Multi-run project navigation still has hidden-state traps
- Recovery paths (session persistence, error recovery) not fully regression-tested

---

## P1 — Important Next Items

| Area | Status | Gap |
|------|--------|-----|
| **NUTS: Omega/Sigma sampling** | Not started | Full joint posterior requires Wishart/InvWishart priors and expanded state vector |
| **`nutpie` Rust NUTS backend** | Not evaluated | Could eliminate FD gradient bottleneck for all model types |
| **ADVAN7 (matrix-exponential)** | In-progress | `scipy.linalg.expm` path partially done; not merged |
| **ADVAN13 sensitivity wired into FOCE gradient** | Partial | `solve_with_sensitivity()` exists but `_compute_G_i` still uses FD |
| **Symbolic derivatives for ADVAN5–13** | Not started | Only ADVAN1–4 have SymPy kernels; ODE models pay full FD cost |
| **FREM external validation** | Not started | EBE-based approximation exists; not compared to Pharmpy FREM |
| **AMD pipeline end-to-end** | Scaffolded | Structural + covariate search path exists; not validated on real datasets |
| **CDISC ADPPK domain** | Helper-grade | Not submission-grade; PP domain variables incomplete |
| **Comprehensive test expansions** | Ongoing | See `todo/remaining_priorities.md` P1 testing section |

---

## Suggested Execution Order (next 4–6 weeks)

1. **Advanced-estimator empirical benchmark expansion** (RP-001)
   One more dataset per stochastic method family with an external anchor.
   Highest credibility return per engineering hour.

2. **NUTS Omega/Sigma sampling**
   Most significant correctness gap in shipped Bayesian functionality.

3. **`nutpie` evaluation**
   Low-effort evaluation; high potential speedup for all model types.

4. **ADVAN7 close-out** (quick win)
   In-progress; complete and merge.

5. **External validation suite** (RP-001 / P0)
   FOCE/SAEM vs NONMEM/Monolix reference on shared published datasets.

6. **FREM + AMD scope clarification**
   Document honest current scope; add one benchmark-backed example each.

---

## What Not To Do Now

- Adding raw feature checkboxes without validation
- Broad GUI redesign before comparison/review workflow is stable
- Regulatory-pack claims beyond current research-grade scope
- Performance claims without reproducible stored benchmark evidence

---

## Reference Files

- `todo/remaining_priorities.md` — detailed RP-001–RP-005 tickets with status notes
- `todo/gap_analysis_auggie.md` — original competitive gap analysis
- `docs/user_guide/external_validation_benchmarks.md` — validated method surface
