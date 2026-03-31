# Validation matrix

This page is the current source-of-truth matrix for **what OpenPKPD claims is
validated**, **how that claim is supported**, and **where the support boundary
still stops**.

Use it together with:

- [`external_validation_benchmarks.md`](external_validation_benchmarks.md) for
  the concrete external datasets and tools
- [`analysis_tools.md`](analysis_tools.md) for model-family validation outside
  the core estimation stack
- [`testing.md`](testing.md) for the test-tier structure

## How to read this page

Validation level:

- **Primary**: mathematically standard implementation, strong unit/regression
  coverage, and multiple external anchors or a strong analytic reference basis
- **Secondary**: real and useful, but external coverage is still narrower or
  operational limits matter materially
- **Experimental**: implemented and tested, but not yet broad enough to market
  as a high-confidence general workflow

Evidence types:

- **Analytic**: checked against closed forms or exact numerical identities
- **Method**: unit or regression tests for algorithm behavior and diagnostics
- **External**: compared against external software, published references, or
  bundled third-party fixtures

## Estimation methods

| Surface | Algorithm basis | Validation level | Main evidence | Current external anchors | Current gaps / notes |
|---|---|---|---|---|---|
| `FO` | first-order linearization around `eta=0` | **Primary** | analytic OFV checks; unit tests for fallback Jacobians; regression and cross-method checks | `nlmixr2` reduced warfarin PK/PD FO; Bauer-style and PK reference workflows through broader FO/FOCE tests | ODE-heavy FO still pays finite-difference cost on some paths |
| `FOCE` | first-order conditional estimation with per-subject `eta_hat` optimization | **Primary** | analytic OFV checks; unit tests for optimizer/ETA handling; regression and diagnostic tests | `nlmixr2`, `NONMEM`, Bauer tutorial workflows, covariate models | mixed-endpoint ODE-heavy runtime still needs more breadth |
| `FOCEI` | FOCE with interaction term | **Primary** | method tests, regression checks, diagnostics around warfarin parity gaps | `nlmixr2` theophylline/warfarin fixtures plus maintained `NONMEM` Run 402 and Run 504 empirical references in `tests/external_validation/` | still thinner than FOCE on full model-family breadth, but now includes a live covariate-rich external anchor |
| `Laplacian` | FOCE plus Hessian correction term | **Primary** | analytic OFV correction checks; regression and cross-method validation | analytic linear-Gaussian reference; included in cross-method regression surface | thinner direct external breadth than FOCE/FOCEI |
| `SAEM` | stochastic approximation EM with MH-based latent updates | **Secondary** | stochastic-behavior tests, convergence-history checks, multi-chain checks | public Monolix theophylline; `nlmixr2` warfarin PK; reduced warfarin PK/PD mixed-endpoint `nlmixr2` basin | credible on current slices, but still needs another empirical family and broader model-family coverage beyond the current reduced mixed-endpoint slice |
| `IMP` | importance sampling around conditional modes | **Secondary** | analytic Gaussian-reference tests; deterministic objective/draw diagnostics; empirical basin checks | theophylline and warfarin PK vs `nlmixr2` FOCEI basins | benchmark budget matters materially; still narrower than FO/FOCEI |
| `IMPMAP` | IMP with FOCEI warm start for MAP-style path | **Secondary** | unit tests for warm-start behavior and diagnostics; empirical recommendation tests | warfarin PK empirical validation vs `nlmixr2` FOCEI; measured reduced warfarin PK/PD mixed-endpoint probe | validated on current PK slice; reduced mixed-endpoint probe lands in the right basin but is still too heavy/noisy to promote as a stable external benchmark |
| `BAYES(Laplace)` | FOCE MAP + local Gaussian approximation | **Secondary** | posterior-summary tests, covariance/Hessian checks, inverse-Hessian fallback tests | theophylline and warfarin PK vs `nlmixr2`; reduced warfarin PK/PD mixed-endpoint `nlmixr2` basin; NONMEM 402 empirical basin | useful weak-prior Bayesian path, but not full MCMC parity; still needs broader model-family breadth beyond the current reduced mixed-endpoint slice |
| `BAYES(NUTS)` | native pure-NumPy NUTS on FOCE-backed theta posterior | **Experimental** | sampler diagnostics, bounded benchmark artifacts, cache/warm-start tests, analytic-theta-gradient routing on supported symbolic subset | second-tier empirical theophylline benchmark; synthetic bounded benchmark artifact | real implementation, but still theta-only, runtime-sensitive, and not broad enough for primary Bayesian support claims |
| `NONPARAMETRIC` | NPML-style support-point distribution over EBEs | **Secondary** | exact/reference tests for weight normalization and empirical moments; cross-method regression checks | Pharmpy `pheno`; theophylline oral PK empirical anchor vs `nlmixr2` FOCEI basin | now past the original "second anchor" threshold, but still needs broader dataset and model-family coverage |

## Other validated analytical / model surfaces

| Surface | Validation level | Main evidence | Current gaps / notes |
|---|---|---|---|
| PK subroutines `ADVAN1/2/3` | **Primary** | closed-form formula checks and ODE cross-validation in `test_pk_subroutines_reference.py` | strong on analytical PK; more advanced ODE families still need broader external breadth |
| ODE PK / sensitivity workflows (`ADVAN6`, `ADVAN13`) | **Secondary** | unit tests, sensitivity tests, analytical cross-checks where available | performance and sensitivity-default coverage still lag the strongest compiled tools |
| TTE / count / categorical / CTMC / HMM / TMDD | **Primary** | unit checks plus external validation against `scipy` closed forms or limit-case reductions | fit-baseline breadth can still grow, but numerical footing is strong |
| NCA | **Primary** | analytic one-compartment references, public PKNCA summaries, WinNonlin-backed Indometh tables | report polish and full workflow parity still lag Phoenix |
| VPC / pcVPC / NPDE / NPC | **Primary** | unit, regression, and external formula/coverage checks | broader cross-software parity remains future work |

## Workflow surfaces

| Surface | Validation level | Main evidence | Current gaps / notes |
|---|---|---|---|
| NONMEM-style control-stream parsing/runtime | **Secondary** | parser tests, integration tests for supported runtime subsets, executable support notes in `control_stream.md` | parser breadth still exceeds full runtime/export breadth; writer/round-trip contract still incomplete |
| GUI review and comparison workflows | **Secondary** | unit tests around persistence, result review, and comparison flow | credible workflow surface now exists, but empirical user-flow polish still trails WinNonlin / Monolix |
| Benchmark / profiling harness | **Secondary** | checked-in baseline artifacts, benchmark scripts, benchmark-output tests | estimation breadth is better than before, but mixed-endpoint and simulation profiling still need expansion |

## Release-gating guidance

Current practical release posture:

- **Primary release-gated surfaces**
  - `FO`, `FOCE`, `FOCEI`, `Laplacian`
  - analytical PK subroutines with strong reference checks
  - NCA
  - core diagnostics (`VPC`, `pcVPC`, `NPDE`, `NPC`)
  - TTE / count / categorical / CTMC / TMDD analytical checks
- **Secondary release-gated surfaces**
  - `SAEM`
  - `IMP` / `IMPMAP`
  - `BAYES(Laplace)`
  - `NONPARAMETRIC`
  - supported control-stream runtime subsets
- **Second-tier or environment-sensitive surfaces**
  - `BAYES(NUTS)`
  - reduced mixed-endpoint advanced-estimator empirical paths beyond the current reduced `BAYES(Laplace)` benchmark
  - broader external-validation suites that require dedicated environments or slower third-party tooling

## What should happen when a method is not yet strong enough

The project should prefer:

1. **narrower claims with stronger evidence**
2. **explicit support-boundary notes**
3. **known-gap tests that fail or `xfail` honestly**

The project should avoid:

1. loosening concordance tolerances to hide defects
2. presenting a secondary or experimental method as if it had FOCE-level validation breadth
3. treating smoke tests as external validation

## Highest-value next additions

1. Add another broader external empirical family for `SAEM`, `IMP` / `IMPMAP`, and `NONPARAMETRIC`, focusing on model-family breadth rather than merely reaching a second anchor count.
2. Promote one more mixed-endpoint PK/PD advanced-estimator benchmark beyond the new reduced `BAYES(Laplace)` and `SAEM` warfarin PK/PD tests, ideally `IMPMAP` once the runtime/diagnostic surface is honest enough.
3. Broaden ODE-heavy validation for advanced estimators so support boundaries are based on measured evidence rather than caution alone.
4. Keep this matrix synchronized with `todo/remaining_priorities.md` and `external_validation_benchmarks.md` whenever a method changes support level.
