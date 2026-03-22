# Covariance Step

The covariance step computes standard errors, confidence intervals, and the
correlation matrix of parameter estimates via the sandwich (R/S) estimator.

## Enabling the covariance step

```python
result = (
    ModelBuilder()
    ...
    .estimation(method="FOCE", interaction=True)
    .covariance()              # Enable with default R/S sandwich
    .build()
    .fit()
)
```

Or with explicit matrix choice:

```python
.covariance(matrix="R")   # R matrix only (inverse Hessian)
.covariance(matrix="SR")  # Default: sandwich R/S estimator
```

## Accessing results

```python
cov = result.covariance_result

cov.se_theta       # Standard errors of THETA, shape (n_theta,)
cov.se_omega_diag  # Standard errors of diagonal OMEGA elements
cov.se_sigma_diag  # Standard errors of diagonal SIGMA elements

cov.cov_matrix     # Full covariance matrix of all free parameters
cov.cor_matrix     # Correlation matrix
cov.condition_number  # Condition number of R matrix

# 95% confidence intervals for THETA
import numpy as np
ci_low  = result.theta_final - 1.96 * cov.se_theta
ci_high = result.theta_final + 1.96 * cov.se_theta
```

## Output files

When running from a `.ctl` file the covariance step produces:

| File | Contents |
|------|----------|
| `.cov` | Covariance matrix (NONMEM 7.x format) |
| `.cor` | Correlation matrix |

## The R and S matrices

The sandwich estimator is:

```
Cov(θ) = R⁻¹ · S · R⁻¹
```

where:

- **R** is the Hessian of the OFV with respect to free parameters (expected
  information matrix)
- **S** is the cross-product of first derivatives (observed information matrix)

Both are computed by numerical finite differences in OpenPKPD.

## Condition number

A high condition number (> 1000) indicates near-collinearity between parameters,
typically caused by:

- Overparameterisation (too many parameters for the data)
- Correlation between THETA and OMEGA (model misspecification)
- Starting values close to a boundary

## Interpreting warnings

| Warning | Likely cause |
|---------|-------------|
| `R matrix not positive definite` | Convergence issue; OFV at a saddle point |
| `Correlation > 0.95` | Near-collinear parameters |
| `Condition number > 1000` | Check parameter identifiability |
