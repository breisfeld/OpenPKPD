"""
Residual error model utilities.

Provides:
  - ResidualModelType enum for common error model families
  - AR(1) autocorrelation fitting for population model residuals
  - Log-normal log-likelihood for multiplicative residual models
  - Power residual variance function

References:
    Karlsson MO, Sheiner LB. (1993). The importance of modeling interoccasion
        variability in population pharmacokinetic analyses.
        J Pharmacokinet Biopharm 21(6):735-750.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class ResidualModelType(StrEnum):
    """Enumeration of supported residual error model types."""

    ADDITIVE = "additive"
    PROPORTIONAL = "proportional"
    COMBINED = "combined"
    LOG_NORMAL = "log_normal"
    POWER = "power"


@dataclass
class AR1ResidualResult:
    """
    Result from AR(1) residual model fitting.

    Attributes:
        rho:             Estimated autocorrelation coefficient (|rho| < 1).
        sigma2:          Estimated innovation variance.
        log_likelihood:  Log-likelihood of the fitted model.
        n_subjects:      Number of subjects used in fitting.
        n_observations:  Total number of observations.
    """

    rho: float
    sigma2: float
    log_likelihood: float
    n_subjects: int = 0
    n_observations: int = 0


def fit_ar1_residuals(
    residuals: np.ndarray,
    subject_ids: np.ndarray,
    times: np.ndarray,
) -> AR1ResidualResult:
    """
    Fit an AR(1) model to CWRES (or WRES) residuals per subject.

    Each subject's residuals are assumed to follow:
        e(t_j) = rho^(t_j - t_{j-1}) * e(t_{j-1}) + innovation(t_j)

    where innovation ~ N(0, sigma2 * (1 - rho^2)).

    Uses Yule-Walker estimation (method of moments) for rho, then estimates
    sigma2 from the innovation variance.

    Args:
        residuals:    Array of residuals (CWRES or WRES), length n.
        subject_ids:  Array of subject identifiers, same length n.
        times:        Array of observation times, same length n.

    Returns:
        AR1ResidualResult with rho, sigma2, and log-likelihood.

    Notes:
        - Subjects with fewer than 3 observations are skipped.
        - rho is estimated by pooling lag-1 autocorrelation across subjects.
        - sigma2 is estimated as the variance of the innovations.
    """
    residuals = np.asarray(residuals, dtype=float)
    subject_ids = np.asarray(subject_ids)
    times = np.asarray(times, dtype=float)

    if not (residuals.shape == subject_ids.shape == times.shape):
        raise ValueError("residuals, subject_ids, and times must have the same length.")

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    # Pool lag-1 products and variances across subjects
    sum_lag1_product = 0.0
    sum_lag0_var = 0.0
    innovations: list[float] = []
    n_obs_total = 0

    for subj in unique_subjects:
        mask = subject_ids == subj
        t_s = times[mask]
        r_s = residuals[mask]

        # Sort by time
        sort_idx = np.argsort(t_s)
        t_s = t_s[sort_idx]
        r_s = r_s[sort_idx]

        if len(r_s) < 3:
            continue

        # Remove NaN
        valid = np.isfinite(r_s)
        t_s = t_s[valid]
        r_s = r_s[valid]
        if len(r_s) < 3:
            continue

        n_obs_total += len(r_s)
        # Simple AR(1): rho estimated via lag-1 autocorrelation of residuals
        # (assumes constant time spacing; for irregular times use discrete AR)
        r_mean = np.mean(r_s)
        r_centered = r_s - r_mean
        # Lag-0 and lag-1 contributions
        sum_lag0_var += float(np.sum(r_centered[:-1] ** 2))
        sum_lag1_product += float(np.sum(r_centered[:-1] * r_centered[1:]))

    # Pooled Yule-Walker estimate
    if sum_lag0_var > 0:
        rho = float(np.clip(sum_lag1_product / sum_lag0_var, -0.999, 0.999))
    else:
        rho = 0.0

    # Compute innovations and estimate sigma2
    for subj in unique_subjects:
        mask = subject_ids == subj
        t_s = times[mask]
        r_s = residuals[mask]
        sort_idx = np.argsort(t_s)
        r_s = r_s[sort_idx]
        valid = np.isfinite(r_s)
        r_s = r_s[valid]
        if len(r_s) < 3:
            continue
        # Innovation = e_t - rho * e_{t-1}
        innov = r_s[1:] - rho * r_s[:-1]
        innovations.extend(innov.tolist())

    if innovations:
        sigma2 = float(np.var(innovations, ddof=1))
    else:
        sigma2 = float(np.var(residuals[np.isfinite(residuals)], ddof=1))

    # Log-likelihood
    if sigma2 > 0 and n_obs_total > 0:
        log_lik = -0.5 * n_obs_total * np.log(2.0 * np.pi * sigma2)
        log_lik -= 0.5 / sigma2 * float(np.sum(np.array(innovations) ** 2)) if innovations else 0.0
    else:
        log_lik = float("nan")

    return AR1ResidualResult(
        rho=rho,
        sigma2=sigma2,
        log_likelihood=float(log_lik),
        n_subjects=n_subjects,
        n_observations=n_obs_total,
    )


def log_normal_log_likelihood(
    y_obs: float,
    mu: float,
    sigma2: float,
) -> float:
    """
    Log-likelihood for a log-normal observation.

    Model: log(Y) ~ N(log(mu), sigma2)
    Includes the Jacobian term -log(y_obs).

    ℓ = -0.5*(log(2π) + log(σ²) + (log(Y) - log(μ))²/σ²) - log(Y)

    Args:
        y_obs:   Observed value (must be > 0).
        mu:      Predicted value (location parameter on natural scale, > 0).
        sigma2:  Variance on log scale (> 0).

    Returns:
        Log-likelihood contribution. Returns -inf for invalid inputs.
    """
    if y_obs <= 0 or mu <= 0 or sigma2 <= 0:
        return float("-inf")

    log_diff = np.log(y_obs) - np.log(mu)
    ell = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + log_diff**2 / sigma2) - np.log(y_obs)
    return float(ell)


def power_residual_variance(
    f: float,
    sigma: float,
    theta: float,
) -> float:
    """
    Compute the power-model residual variance.

    var(Y) = sigma² * f^(2*theta)

    Args:
        f:      Model prediction (IPRED).
        sigma:  Residual standard deviation parameter.
        theta:  Power exponent.

    Returns:
        Variance as a float. Returns 0 if f <= 0 and theta > 0.
    """
    if f <= 0 and theta > 0:
        return np.finfo(float).tiny
    return float(sigma**2 * abs(f) ** (2.0 * theta))
