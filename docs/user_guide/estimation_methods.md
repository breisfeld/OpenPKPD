# Estimation Methods

`ModelBuilder.estimation()` and the control-stream runner both dispatch into the
native estimator router in `openpkpd.estimation`.

## Choosing a method

| Method | `method=` | When to use |
|--------|-----------|-------------|
| First Order | `"FO"` | fast screening, simple models |
| FOCE | `"FOCE"` | default choice for many PK models |
| FOCEI | `"FOCEI"` or `"FOCE"` + `interaction=True` | proportional/combined error models |
| Laplacian | `"LAPLACIAN"` | non-Gaussian likelihood approximations |
| SAEM | `"SAEM"` | irregular or highly nonlinear models |
| IMP | `"IMP"` | importance sampling likelihood refinement |
| IMPMAP | `"IMPMAP"` | IMP-family workflow with MAP-style routing |
| Bayesian | `"BAYES"` | posterior sampling / Bayesian summaries |
| Nonparametric | `"NONPARAMETRIC"` | NPML/NPEM-style support-point estimation |

## FO — First Order

Linearises the model around `ETA = 0`. It is fast, but least accurate when the
model is strongly nonlinear or IIV is large.

```python
.estimation(method="FO", maxeval=500)
```

## FOCE and FOCEI

FOCE linearises around each subject's conditional mode (their EBE).

```python
.estimation(method="FOCE", maxeval=9999)
```

FOCEI adds ETA–EPS interaction and is usually the right choice for
proportional or combined error models.

```python
.estimation(method="FOCEI", maxeval=9999)
# or:
.estimation(method="FOCE", interaction=True, maxeval=9999)
```

FOCE/FOCEI also accept optimizer controls that are useful on hard likelihood
surfaces:

| Argument | Meaning |
|----------|---------|
| `n_starts` | number of perturbed restarts; best OFV retained |
| `gtol` | outer optimizer gradient tolerance |
| `outer_optimizer` | primary outer optimizer, e.g. `"L-BFGS-B"` |
| `outer_fallback_optimizer` | optional follow-up polish optimizer, e.g. `"Powell"` |
| `outer_fallback_maxeval` | evaluation budget for the fallback optimizer |
| `retain_best_iterate` | keep the best point visited even if the terminal iterate is worse |
| `retry_on_abnormal` | rerun FOCEI from structured alternate starts after abnormal termination |
| `retry_omega_scales` | OMEGA scaling factors used for structured retries |

Example:

```python
.estimation(
    method="FOCEI",
    maxeval=200,
    n_starts=3,
    outer_optimizer="L-BFGS-B",
    outer_fallback_optimizer="Powell",
    outer_fallback_maxeval=40,
    retain_best_iterate=True,
    retry_on_abnormal=True,
    retry_omega_scales=(0.5, 0.25, 0.1),
)
```

## Laplacian

Laplacian extends FOCE with a Hessian correction term and is the main native
approximation for more non-Gaussian observation models.

```python
.estimation(method="LAPLACIAN", maxeval=9999)
```

## SAEM

SAEM is implemented as a stochastic approximation EM workflow using a
single-chain Metropolis-Hastings style inner sampler.

```python
.estimation(method="SAEM")
```

SAEM runs in two phases: a stochastic exploration phase and a smoothing phase.
The number of iterations in each phase is controlled via `SAEMMethod` directly:

```python
from openpkpd.estimation.saem import SAEMMethod

saem = SAEMMethod(n_iter_phase1=200, n_iter_phase2=100, n_workers=4)
result = saem.estimate(pop_model, params)
```

> **Note:** `maxeval` and `n_parallel` are not accepted by `SAEMMethod`.
> Use `n_iter_phase1` / `n_iter_phase2` to control iteration counts and
> `n_workers` for thread-level parallelism in the E-step.

It is useful for harder models, but the current implementation is still less
mature than specialized SAEM toolchains.

## IMP and IMPMAP

IMP refines the marginal likelihood through importance sampling.

```python
.estimation(method="IMP", maxeval=9999)
```

`IMPMAP` runs the same IMP outer objective, but first uses a short FOCEI warm
start to seed the population parameters into a better basin before the IMP
optimization:

```python
.estimation(method="IMPMAP", maxeval=9999)
```

In practice, prefer `IMPMAP` over raw `IMP` when the model is basin-sensitive
or when a direct IMP run tends to stick near the initial THETA values.

## Bayesian estimation

`BAYES` uses the best available backend in this order:

1. PyMC if `openpkpd[bayes]` is installed
2. built-in pure-NumPy NUTS otherwise

```python
.estimation(method="BAYES", n_samples=1000, n_chains=2, tune=500)
```

You can also request a backend explicitly:

```python
.estimation(method="BAYES", backend="pymc")
.estimation(method="BAYES", backend="nuts")
.estimation(method="BAYES", backend="laplace")
```

Passing any other backend name now raises a hard error instead of silently
falling back to a different path.

Backend guidance:

| Backend | Current role | Notes |
|---------|--------------|-------|
| `"pymc"` | best-supported full MCMC backend | strongest diagnostics and most complete posterior workflow |
| `"nuts"` | built-in second-tier MCMC backend | multi-chain diagnostics are available through `BAYESMethod`; currently samples THETA only and empirical population-model runs can be slow |
| `"laplace"` | fast approximation fallback | MAP + Hessian-based Gaussian posterior approximation |

Important current limitations:

- the built-in NUTS backend currently samples **THETA only**; `OMEGA` and
  `SIGMA` remain fixed at their starting values
- the built-in NUTS path now uses the cached symbolic analytical theta-gradient
  path on the supported analytical PK subset, but still falls back outside that
  envelope and can be slow on larger ODE-heavy models
- NUTS runs now expose sampler and posterior-evaluation diagnostics in
  `result.diagnostics["nuts"]`, including log-probability call counts,
  exact-cache hit/miss counts, warm-start hit counts, FOCE inner/outer call
  counts, step-size summaries, acceptance statistics, and tree-depth summaries
- on a bounded synthetic 6-subject oral-PK probe (`n_samples=12`, `tune=8`,
  `n_chains=2`), the compiled symbolic path now takes about `8.5 s`, but the
  empirical theophylline benchmark still needs a larger budget (`24/16`) and
  about `28 s` to bring max `R-hat` down to roughly `1.19`, so it should still
  be treated as a dependency-free second-tier path rather than the default
  empirical benchmark route
- the standalone low-level helper `nuts_estimate()` is **single-chain only** and
  therefore does **not** return meaningful `R-hat`; use
  `.estimation(method="BAYES", backend="nuts", n_chains>=2)` when you want
  multi-chain diagnostics
- Laplace is useful for fast Bayesian summaries, but should not be described as
  equivalent to full MCMC on difficult posterior geometries
- `IMP` results expose optimizer-stop and final effective-sample-size details in
  `result.diagnostics`, which is the intended surface for telling apart
  iteration-budget exhaustion, plateauing, and poor importance-sampling
  coverage

## Nonparametric estimation

OpenPKPD also exposes a nonparametric support-point estimator.

```python
.estimation(method="NONPARAMETRIC", base_method="FOCE", max_iter=100)
```

This runs a base parametric fit first, then optimizes support-point weights.

Current support note:

- the operationally strongest current path is `base_method="FOCEI"` on
  population PK models with a modest number of ETAs
- the repository now includes an empirical phenobarbital benchmark against
  Pharmpy's bundled `pheno` dataset and a runnable example in
  `examples/32_nonparametric_support_points.py`
- external validation is still narrower than the FO/FOCEI surface, so treat
  nonparametric estimation as a real but still selectively benchmarked path

## Common options

Common arguments are passed through to the selected estimator:

| Argument | Typical use |
|----------|-------------|
| `maxeval` | optimizer / outer-loop budget |
| `interaction` | FOCEI-style ETA–EPS interaction |
| `n_starts`, `gtol` | FOCE/FOCEI multi-start and gradient controls |
| `outer_optimizer`, `outer_fallback_optimizer`, `outer_fallback_maxeval` | FOCE/FOCEI outer-optimizer selection and polish |
| `retain_best_iterate`, `retry_on_abnormal`, `retry_omega_scales` | FOCEI robustness controls for unstable/non-convex fits |
| `n_samples`, `n_chains`, `tune`, `backend` | Bayesian configuration |
| `base_method`, `n_support_points`, `max_iter` | nonparametric configuration |

## Result objects

Most methods return an `EstimationResult`:

```python
result.ofv
result.theta_final
result.omega_final
result.sigma_final
result.converged
result.post_hoc_etas
result.ofv_history
result.warnings

result.compute_shrinkage()
print(result.summary())
```

### ETA shrinkage and de-shrinkage

After calling `compute_shrinkage()`, per-ETA shrinkage fractions are stored in
`result.eta_shrinkage`:

```python
result.compute_shrinkage()
# result.eta_shrinkage: array of shrinkage fractions, e.g. [0.62]
# Shrinkage > 30% triggers a warning and is flagged in the HTML report.
```

EBEs (Empirical Bayes Estimates, i.e. post-hoc ETAs) from FOCE are
systematically shrunk toward zero.  When shrinkage is high, covariate plots
and ETA distributions based on raw EBEs underrepresent between-subject
variability.  The Combes (2013) rescaling correction adjusts each subject's
ETA vector so that `SD(eta_k) = sqrt(omega_kk)` exactly:

```python
adjusted_etas = result.compute_deshrinkage_etas()
# Returns dict[subject_id → adjusted_eta_vector]
# adjusted_etas[1]  # de-shrunken ETA for subject 1
```

The correction factor per random effect `k` is `1 / (1 − shrinkage_k)`.
Subjects retain their relative ordering; only the dispersion is corrected.
De-shrunken ETAs are appropriate for covariate plots and ETA histograms when
FOCE shrinkage exceeds roughly 30%.  SAEM EBEs typically have lower shrinkage
and often need less correction.

> **Reference:** Combes F-P, Retout S, Frey N, Mentré F (2013).
> *Prediction of shrinkage of individual parameters using the Bayesian
> information matrix in nonlinear mixed-effects models.*
> Pharm Res 30:2355–2367.

`method="BAYES"` returns a `BayesianResult`, which adds posterior-specific
fields such as:

```python
result.posterior_samples
result.r_hat
result.n_effective
result.posterior_ci_lo
result.posterior_ci_hi
result.backend_used
```

`method="NONPARAMETRIC"` returns a `NonparametricResult` with support points and
support weights.

## Model selection

For nested model comparison, use `ΔOFV` with a chi-squared reference where the
assumptions are appropriate:

```python
from scipy.stats import chi2

delta_ofv = result_base.ofv - result_full.ofv
p = chi2.sf(delta_ofv, df=1)
```

## Convergence and warnings

If `result.converged` is `False`, inspect `result.warnings` and the OFV history.
Typical causes are poor initials, near-singular OMEGA blocks, too-small
`maxeval`, or structural non-identifiability.

## Parallel execution

FOCE, SAEM, and IMP support parallel subject-level computation via `n_parallel`.

| Method | Executor | Parallelised work |
|--------|----------|-------------------|
| FOCE/FOCEI | `ProcessPoolExecutor` | Per-subject η optimisation (true multi-core) |
| SAEM | `ThreadPoolExecutor` | Per-subject Metropolis–Hastings E-step |
| IMP | `ThreadPoolExecutor` | Per-subject importance sampling |

```python
from openpkpd.estimation import get_estimation_method

# Use 4 worker processes for FOCE inner loop
method = get_estimation_method("FOCE", n_parallel=4)
result = method.estimate(pop_model, params)

# n_parallel=0 auto-detects core count
method = get_estimation_method("SAEM", n_parallel=0)
```

`n_parallel=1` is the default and runs serially. FO, Laplacian, BAYES, and
Nonparametric are inherently serial and ignore `n_parallel`.

For parallel VPC simulation, see `SimulationEngine(n_parallel=N)` in the
analysis tools guide.
