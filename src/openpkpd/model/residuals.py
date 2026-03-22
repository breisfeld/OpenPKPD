"""
Residual error model helpers.

Supports additive, proportional, combined additive+proportional, and
user-defined residual models from compiled $ERROR blocks.
"""

from __future__ import annotations

import math

import numpy as np

from openpkpd.utils.constants import LOG2PI


def log_likelihood_normal(y: float, mu: float, sigma2: float) -> float:
    """
    Log-likelihood of observation y ~ N(mu, sigma2).

    ℓ = -0.5 * [log(2π) + log(σ²) + (y - μ)² / σ²]
    """
    if sigma2 <= 0:
        return -1e30
    return -0.5 * (LOG2PI + math.log(sigma2) + (y - mu) ** 2 / sigma2)


def compute_residual_variance(
    f: float,
    sigma: np.ndarray,
    error_type: str = "combined",
    eps_val: float | None = None,
) -> float:
    """
    Compute the residual variance at a given predicted value.

    Models:
      additive:      Var(Y|f) = sigma[0,0]
      proportional:  Var(Y|f) = (f * sigma[0,0])²  or  f² * sigma[0,0]
      combined:      Var(Y|f) = sigma[0,0] + (f * sigma[1,1])²
                                (sigma has 2+ elements)
    """
    if error_type == "additive":
        return float(sigma[0, 0])
    elif error_type == "proportional":
        return float(f**2 * sigma[0, 0])
    elif error_type == "combined":
        if sigma.shape[0] >= 2:
            return float(sigma[0, 0] + f**2 * sigma[1, 1])
        return float(sigma[0, 0] + f**2 * sigma[0, 0])
    else:
        raise ValueError(f"Unknown error_type: {error_type!r}")


def compute_wres(
    dv: np.ndarray,
    pred: np.ndarray,
    c_i: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted residuals: WRES = C_i^{-1/2} * (DV - PRED).

    Args:
        dv:   Observed values, shape (n_obs,).
        pred: Population predictions, shape (n_obs,).
        c_i:  Marginal variance-covariance matrix, shape (n_obs, n_obs).

    Returns:
        WRES array of shape (n_obs,).
    """
    from scipy.linalg import cholesky as scipy_chol
    from scipy.linalg import solve_triangular

    try:
        L = scipy_chol(c_i, lower=True)
    except Exception:
        # Fallback: use diagonal
        return (dv - pred) / np.sqrt(np.diag(c_i))
    return solve_triangular(L, dv - pred, lower=True)


def compute_iwres(
    dv: np.ndarray,
    ipred: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Compute individual weighted residuals: IWRES = (DV - IPRED) / W.

    W is the per-observation residual standard deviation.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(w > 0, (dv - ipred) / w, 0.0)


def compute_cwres(
    dv: np.ndarray,
    pred: np.ndarray,
    ipred: np.ndarray,
    wres: np.ndarray,
    iwres: np.ndarray,
) -> np.ndarray:
    """
    Conditional weighted residuals (CWRES) — NONMEM approximation.

    CWRES ≈ WRES + (IPRED - PRED) / SD

    where SD = sqrt of per-observation residual variance estimated from IWRES:
    SD_i = |DV_i - IPRED_i| / |IWRES_i| when |IWRES_i| > 0, else 1.

    This approximation is used when the full FOCE C_i matrix is not available.
    For the exact form see plots/diagnostics.py::_cwres_subject().
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ires = dv - ipred
        # Estimate per-observation SD from IWRES: SD = |IRES| / |IWRES|
        sd_est = np.where(
            np.abs(iwres) > 1e-8,
            np.abs(ires) / np.abs(iwres),
            1.0,
        )
        sd_est = np.where(sd_est > 0, sd_est, 1.0)
        correction = (ipred - pred) / sd_est
        return wres + correction
