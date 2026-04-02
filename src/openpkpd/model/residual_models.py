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

import warnings
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import minimize_scalar


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


def _ct_ar1_negloglik(
    log_phi: float,
    residuals_by_subject: list[np.ndarray],
    times_by_subject: list[np.ndarray],
) -> float:
    """Negative log-likelihood for continuous-time AR(1) across subjects."""
    phi = np.exp(log_phi)
    if phi <= 0 or phi >= 1:
        return 1e10
    total_nll = 0.0
    for resids, times in zip(residuals_by_subject, times_by_subject):
        n = len(resids)
        if n < 2:
            continue
        dt = np.abs(np.subtract.outer(times, times))
        Sigma = phi**dt  # correlation matrix (sigma^2 factored out)
        try:
            L = np.linalg.cholesky(Sigma)
            alpha = np.linalg.solve(L, resids)
            total_nll += 0.5 * (
                n * np.log(2 * np.pi)
                + 2 * np.sum(np.log(np.diag(L)))
                + np.dot(alpha, alpha)
            )
        except np.linalg.LinAlgError:
            return 1e10
    return total_nll


def fit_ar1_residuals_ct(
    residuals_by_subject: list[np.ndarray],
    times_by_subject: list[np.ndarray],
) -> tuple[float, float]:
    """
    Fit a continuous-time AR(1) model by MLE.

    The correlation between observations at times t_i and t_j is φ^|t_i - t_j|.
    Fit φ by maximizing the log-likelihood across all subjects.

    Args:
        residuals_by_subject:  List of per-subject residual arrays.
        times_by_subject:      List of per-subject time arrays (same order).

    Returns:
        Tuple (phi_hat, sigma2_hat).
    """
    result = minimize_scalar(
        _ct_ar1_negloglik,
        bounds=(-10, -1e-6),
        method="bounded",
        args=(residuals_by_subject, times_by_subject),
    )
    phi_hat = float(np.exp(result.x))
    all_resids = np.concatenate(residuals_by_subject) if residuals_by_subject else np.array([])
    sigma2_hat = float(np.var(all_resids)) if len(all_resids) > 0 else 0.0
    return phi_hat, sigma2_hat


def fit_ar1_residuals_yw(
    residuals: np.ndarray,
    subject_ids: np.ndarray,
    times: np.ndarray,
) -> AR1ResidualResult:
    """
    Fit an AR(1) model using Yule-Walker estimation (deprecated).

    .. deprecated::
        Use :func:`fit_ar1_residuals` (continuous-time MLE) instead.
        Yule-Walker ignores actual time differences and is only correct for
        equally-spaced observations.
    """
    warnings.warn(
        "fit_ar1_residuals_yw is deprecated; use fit_ar1_residuals (continuous-time MLE) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _fit_ar1_residuals_yw_impl(residuals, subject_ids, times)


def _fit_ar1_residuals_yw_impl(
    residuals: np.ndarray,
    subject_ids: np.ndarray,
    times: np.ndarray,
) -> AR1ResidualResult:
    """Internal Yule-Walker implementation (no deprecation warning)."""
    residuals = np.asarray(residuals, dtype=float)
    subject_ids = np.asarray(subject_ids)
    times = np.asarray(times, dtype=float)

    if not (residuals.shape == subject_ids.shape == times.shape):
        raise ValueError("residuals, subject_ids, and times must have the same length.")

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    sum_lag1_product = 0.0
    sum_lag0_var = 0.0
    innovations: list[float] = []
    n_obs_total = 0

    for subj in unique_subjects:
        mask = subject_ids == subj
        t_s = times[mask]
        r_s = residuals[mask]
        sort_idx = np.argsort(t_s)
        t_s = t_s[sort_idx]
        r_s = r_s[sort_idx]
        if len(r_s) < 3:
            continue
        valid = np.isfinite(r_s)
        t_s = t_s[valid]
        r_s = r_s[valid]
        if len(r_s) < 3:
            continue
        n_obs_total += len(r_s)
        r_mean = np.mean(r_s)
        r_centered = r_s - r_mean
        sum_lag0_var += float(np.sum(r_centered[:-1] ** 2))
        sum_lag1_product += float(np.sum(r_centered[:-1] * r_centered[1:]))

    if sum_lag0_var > 0:
        rho = float(np.clip(sum_lag1_product / sum_lag0_var, -0.999, 0.999))
    else:
        rho = 0.0

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
        innov = r_s[1:] - rho * r_s[:-1]
        innovations.extend(innov.tolist())

    if innovations:
        sigma2 = float(np.var(innovations, ddof=1))
    else:
        sigma2 = float(np.var(residuals[np.isfinite(residuals)], ddof=1))

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


def fit_ar1_residuals(
    residuals: np.ndarray,
    subject_ids: np.ndarray,
    times: np.ndarray,
) -> AR1ResidualResult:
    """
    Fit a continuous-time AR(1) model to CWRES (or WRES) residuals per subject.

    The correlation between observations at times t_i and t_j is φ^|t_i - t_j|.
    φ is estimated by MLE across all subjects. This correctly handles irregular
    observation times.

    Args:
        residuals:    Array of residuals (CWRES or WRES), length n.
        subject_ids:  Array of subject identifiers, same length n.
        times:        Array of observation times, same length n.

    Returns:
        AR1ResidualResult with rho, sigma2, and log-likelihood.

    Notes:
        - Subjects with fewer than 2 observations are skipped (single obs cannot
          contribute to the AR(1) likelihood).
        - phi is estimated by continuous-time MLE (accounts for irregular times).
        - sigma2 is estimated as the pooled variance of the residuals.
    """
    residuals = np.asarray(residuals, dtype=float)
    subject_ids = np.asarray(subject_ids)
    times = np.asarray(times, dtype=float)

    if not (residuals.shape == subject_ids.shape == times.shape):
        raise ValueError("residuals, subject_ids, and times must have the same length.")

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    # Build per-subject arrays, skipping subjects with < 2 valid observations
    residuals_by_subject: list[np.ndarray] = []
    times_by_subject: list[np.ndarray] = []
    n_obs_total = 0

    for subj in unique_subjects:
        mask = subject_ids == subj
        t_s = np.asarray(times[mask], dtype=float)
        r_s = np.asarray(residuals[mask], dtype=float)
        sort_idx = np.argsort(t_s)
        t_s = t_s[sort_idx]
        r_s = r_s[sort_idx]
        valid = np.isfinite(r_s)
        t_s = t_s[valid]
        r_s = r_s[valid]
        if len(r_s) < 2:
            continue
        residuals_by_subject.append(r_s)
        times_by_subject.append(t_s)
        n_obs_total += len(r_s)

    if not residuals_by_subject:
        return AR1ResidualResult(
            rho=0.0,
            sigma2=float(np.var(residuals[np.isfinite(residuals)], ddof=1))
            if np.any(np.isfinite(residuals))
            else 0.0,
            log_likelihood=float("nan"),
            n_subjects=n_subjects,
            n_observations=n_obs_total,
        )

    # Fit continuous-time AR(1) by MLE
    phi_hat, sigma2_hat = fit_ar1_residuals_ct(residuals_by_subject, times_by_subject)

    # Compute log-likelihood at the MLE
    log_lik = float(-_ct_ar1_negloglik(np.log(phi_hat), residuals_by_subject, times_by_subject))

    return AR1ResidualResult(
        rho=phi_hat,
        sigma2=sigma2_hat,
        log_likelihood=log_lik,
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


_POWER_VAR_FLOOR: float = 1e-6
"""Minimum |f| used in power-variance to prevent a hard discontinuity at f=0.

Without a floor the function is discontinuous: for theta>0 the limit as
|f|→0 is 0, but the old implementation snapped to ``np.finfo(float).tiny``
exactly at f=0, creating a step-change of ~2e-308 vs ~0.  The floor makes the
variance a smooth function of f for all f with |f| < ``_POWER_VAR_FLOOR``.
"""


def power_residual_variance(
    f: float,
    sigma: float,
    theta: float,
) -> float:
    """
    Compute the power-model residual variance.

    var(Y) = sigma² * max(|f|, floor)^(2*theta)

    A small floor (``_POWER_VAR_FLOOR`` = 1e-6) is applied to ``|f|`` when
    ``theta > 0`` to keep the variance function continuous and strictly
    positive near f = 0, avoiding both division-by-zero in downstream
    likelihood calculations and a discontinuous jump at the boundary.

    Args:
        f:      Model prediction (IPRED).
        sigma:  Residual standard deviation parameter.
        theta:  Power exponent.

    Returns:
        Variance as a float; always > 0 when sigma > 0 and theta >= 0.
    """
    f_abs = abs(f)
    if theta > 0:
        f_abs = max(f_abs, _POWER_VAR_FLOOR)
    return float(sigma**2 * f_abs ** (2.0 * theta))
