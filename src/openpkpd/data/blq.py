"""
BLQ (Below Limit of Quantification) data handling methods M1-M7.

Implements likelihood contributions for censored observations per
Beal (2001) and Ahn et al. (2008) methods.

References:
    Beal, S.L. (2001). Ways to fit a PK model with some data below the
    quantification limit. Journal of Pharmacokinetics and Pharmacodynamics,
    28(5), 481-504.

    Ahn, J.E., Karlsson, M.O., Dunne, A., & Ludden, T.M. (2008).
    Likelihood based approaches to handling data below the quantification
    limit using NONMEM VI. Journal of Pharmacokinetics and Pharmacodynamics,
    35(4), 401-421.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from openpkpd.utils.constants import BLQMethod


def blq_log_likelihood(
    y_obs: float,
    mu: float,
    sigma2: float,
    lloq: float,
    method: str,
) -> float:
    """
    Compute log-likelihood contribution for a BLQ observation.

    Each method handles the censored observation differently:

    - M1: Exclude — return 0.0 (caller should skip this observation).
    - M2: log P(Y < LLOQ) = log Phi((LLOQ - mu) / sigma).
    - M3: log P(Y < LLOQ) = log Phi((LLOQ - mu) / sigma) — same formula
          as M2 but conceptually represents full likelihood censoring; the
          distinction between M2 and M3 lies in implementation context
          (M2 sets MDV=1 for non-first BLQ; M3 applies uniformly).
    - M4: M3 + normalization for truncated normal (Y >= 0).
          log P(Y < LLOQ | Y >= 0) = log [Phi(z_lloq) - Phi(z_0)] where
          z = (x - mu) / sigma, with re-normalization by 1 - Phi(-mu/sigma).
    - M5: Normal log-likelihood evaluated at DV = LLOQ / 2.
    - M6: Normal log-likelihood evaluated at DV = LLOQ / 2 (identical to M5;
          the M6 distinction — first BLQ vs. discard rest — is managed by
          the caller).
    - M7: Normal log-likelihood evaluated at DV = 0.

    Args:
        y_obs:  Observed (censored) value. Used only for M5/M6/M7 imputation
                context; the actual imputed value is derived from lloq or 0.
        mu:     Model prediction (IPRED) at this observation time.
        sigma2: Residual variance (must be > 0).
        lloq:   Lower limit of quantification.
        method: One of the BLQMethod constants (M1–M7).

    Returns:
        Log-likelihood contribution (a non-positive float, or 0.0 for M1).

    Raises:
        ValueError: If sigma2 <= 0 or method is unrecognized.
    """
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")

    sigma = math.sqrt(sigma2)

    if method == BLQMethod.M1:
        # Exclude: caller should skip BLQ observations
        return 0.0

    elif method in (BLQMethod.M2, BLQMethod.M3):
        # Censored likelihood: P(Y < LLOQ) = Phi((LLOQ - mu) / sigma)
        z = (lloq - mu) / sigma
        log_prob = stats.norm.logcdf(z)
        return float(log_prob)

    elif method == BLQMethod.M4:
        # Truncated normal (Y >= 0): P(LLOQ > Y >= 0) / P(Y >= 0)
        # = [Phi(z_lloq) - Phi(z_0)] / [1 - Phi(-mu/sigma)]
        z_lloq = (lloq - mu) / sigma
        z_0 = (0.0 - mu) / sigma
        # P(0 <= Y < LLOQ) in standard normal space
        prob_window = stats.norm.cdf(z_lloq) - stats.norm.cdf(z_0)
        # P(Y >= 0) for normalisation
        prob_pos = 1.0 - stats.norm.cdf(z_0)
        if prob_pos <= 0.0:
            return -1e30
        if prob_window <= 0.0:
            return -1e30
        return float(math.log(prob_window) - math.log(prob_pos))

    elif method == BLQMethod.M5:
        # Impute DV = LLOQ / 2, use normal likelihood
        y_imputed = lloq / 2.0
        return _normal_log_likelihood(y_imputed, mu, sigma2)

    elif method == BLQMethod.M6:
        # Same imputation as M5 for the first BLQ; caller handles subsequent
        y_imputed = lloq / 2.0
        return _normal_log_likelihood(y_imputed, mu, sigma2)

    elif method == BLQMethod.M7:
        # Impute DV = 0, use normal likelihood
        return _normal_log_likelihood(0.0, mu, sigma2)

    else:
        raise ValueError(
            f"Unrecognized BLQ method {method!r}. Use one of: M1, M2, M3, M4, M5, M6, M7."
        )


def _normal_log_likelihood(y: float, mu: float, sigma2: float) -> float:
    """
    Log-likelihood of y ~ N(mu, sigma2).

    ell = -0.5 * [log(2*pi) + log(sigma2) + (y - mu)^2 / sigma2]
    """
    return float(-0.5 * (math.log(2.0 * math.pi) + math.log(sigma2) + (y - mu) ** 2 / sigma2))


def is_blq(dv: float, lloq: float) -> bool:
    """
    Return True if the observation is below the quantification limit.

    An observation is considered BLQ when DV < LLOQ (strictly less than).

    Args:
        dv:   Observed dependent variable value.
        lloq: Lower limit of quantification.

    Returns:
        True if dv < lloq, False otherwise.
    """
    return dv < lloq


def apply_m5_imputation(
    df: pd.DataFrame,
    lloq_col: str = "LLOQ",
    dv_col: str = "DV",
) -> pd.DataFrame:
    """
    Replace BLQ values with LLOQ/2 (M5 method).

    Observations where DV < LLOQ are replaced by LLOQ/2 in-place on a
    copy of the DataFrame. Non-BLQ observations and rows without a valid
    LLOQ are left unchanged.

    Args:
        df:       Input DataFrame containing at least ``dv_col`` and
                  ``lloq_col`` columns.
        lloq_col: Name of the LLOQ column (default ``"LLOQ"``).
        dv_col:   Name of the observed DV column (default ``"DV"``).

    Returns:
        A new DataFrame with BLQ values replaced by LLOQ/2.

    Raises:
        KeyError: If ``dv_col`` or ``lloq_col`` are not in ``df``.
    """
    if dv_col not in df.columns:
        raise KeyError(f"DV column {dv_col!r} not found in DataFrame.")
    if lloq_col not in df.columns:
        raise KeyError(f"LLOQ column {lloq_col!r} not found in DataFrame.")

    df = df.copy()
    lloq = pd.to_numeric(df[lloq_col], errors="coerce")
    dv = pd.to_numeric(df[dv_col], errors="coerce")
    blq_mask = (dv < lloq) & lloq.notna() & dv.notna()
    df.loc[blq_mask, dv_col] = lloq[blq_mask] / 2.0
    return df


def apply_m7_imputation(
    df: pd.DataFrame,
    lloq_col: str = "LLOQ",
    dv_col: str = "DV",
) -> pd.DataFrame:
    """
    Replace BLQ values with 0 (M7 method).

    Observations where DV < LLOQ are replaced by 0.0 in a copy of the
    DataFrame. Non-BLQ observations and rows without a valid LLOQ are
    left unchanged.

    Args:
        df:       Input DataFrame containing at least ``dv_col`` and
                  ``lloq_col`` columns.
        lloq_col: Name of the LLOQ column (default ``"LLOQ"``).
        dv_col:   Name of the observed DV column (default ``"DV"``).

    Returns:
        A new DataFrame with BLQ values replaced by 0.0.

    Raises:
        KeyError: If ``dv_col`` or ``lloq_col`` are not in ``df``.
    """
    if dv_col not in df.columns:
        raise KeyError(f"DV column {dv_col!r} not found in DataFrame.")
    if lloq_col not in df.columns:
        raise KeyError(f"LLOQ column {lloq_col!r} not found in DataFrame.")

    df = df.copy()
    lloq = pd.to_numeric(df[lloq_col], errors="coerce")
    dv = pd.to_numeric(df[dv_col], errors="coerce")
    blq_mask = (dv < lloq) & lloq.notna() & dv.notna()
    df.loc[blq_mask, dv_col] = 0.0
    return df


def flag_blq_observations(
    df: pd.DataFrame,
    lloq_col: str = "LLOQ",
    dv_col: str = "DV",
    blq_flag_col: str = "BLQ",
) -> pd.DataFrame:
    """
    Add a boolean BLQ flag column to the DataFrame.

    Sets ``blq_flag_col`` to 1 where DV < LLOQ, and 0 otherwise. Rows
    with missing DV or LLOQ are flagged as 0.

    Args:
        df:           Input DataFrame.
        lloq_col:     Name of the LLOQ column (default ``"LLOQ"``).
        dv_col:       Name of the observed DV column (default ``"DV"``).
        blq_flag_col: Name of the new BLQ indicator column (default ``"BLQ"``).

    Returns:
        A new DataFrame with an integer BLQ indicator column added.
    """
    df = df.copy()
    lloq = pd.to_numeric(df.get(lloq_col, pd.Series(np.nan, index=df.index)), errors="coerce")
    dv = pd.to_numeric(df.get(dv_col, pd.Series(np.nan, index=df.index)), errors="coerce")
    blq_mask = (dv < lloq) & lloq.notna() & dv.notna()
    df[blq_flag_col] = blq_mask.astype(int)
    return df
