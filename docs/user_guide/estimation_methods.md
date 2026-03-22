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

`IMPMAP` is also routed by the estimator registry:

```python
.estimation(method="IMPMAP", maxeval=9999)
```

## Bayesian estimation

`BAYES` uses the best available backend in this order:

1. PyMC if `openpkpd[bayes]` is installed
2. NumPyro/JAX if `openpkpd[jax]` is installed
3. Laplace approximation fallback otherwise

```python
.estimation(method="BAYES", n_samples=1000, n_chains=2, tune=500)
```

You can also request a backend explicitly:

```python
.estimation(method="BAYES", backend="pymc")
.estimation(method="BAYES", backend="numpyro")
.estimation(method="BAYES", backend="laplace")
```

## Nonparametric estimation

OpenPKPD also exposes a nonparametric support-point estimator.

```python
.estimation(method="NONPARAMETRIC", base_method="FOCE", max_iter=100)
```

This runs a base parametric fit first, then optimizes support-point weights.

## Common options

Common arguments are passed through to the selected estimator:

| Argument | Typical use |
|----------|-------------|
| `maxeval` | optimizer / outer-loop budget |
| `interaction` | FOCEI-style ETA–EPS interaction |
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
