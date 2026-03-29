"""
Standalone MCMC convergence diagnostics.

Implements split-R-hat (Vehtari et al. 2021) and autocorrelation-based
effective sample size that work without ArviZ, so all backends (NumPyro,
Laplace) can compute diagnostics consistently.

References:
    Vehtari, A. et al. (2021). Rank-normalization, folding, and localization:
    An improved R-hat for assessing convergence of MCMC. Bayesian Analysis 16(2).
    https://doi.org/10.1214/20-BA1221

    Gelman, A. & Rubin, D. (1992). Inference from iterative simulation using
    multiple sequences. Statistical Science 7(4), 457-472.
"""

from __future__ import annotations

import numpy as np


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    """
    Rank-normalise samples across all chains jointly (Vehtari et al. 2021).

    Args:
        chains: shape (n_chains, n_draws, n_params)

    Returns:
        Rank-normalised array of same shape.
    """
    n_chains, n_draws, n_params = chains.shape
    out = np.empty_like(chains, dtype=float)
    total = n_chains * n_draws
    for p in range(n_params):
        flat = chains[:, :, p].ravel()
        from scipy.stats import rankdata  # type: ignore[import]

        ranks = rankdata(flat, method="average")
        # Map ranks to N(0,1) via normal quantile of (rank - 3/8) / (N + 1/4)
        from scipy.special import ndtri  # type: ignore[import]

        z = ndtri((ranks - 3.0 / 8.0) / (total + 0.25))
        out[:, :, p] = z.reshape(n_chains, n_draws)
    return out


def compute_rhat(chains: np.ndarray) -> np.ndarray:
    """
    Split-R-hat convergence diagnostic (Vehtari et al. 2021).

    Splits each chain in half (giving 2·M sub-chains), rank-normalises, then
    applies the Gelman-Rubin formula. Values ≤ 1.01 indicate good convergence;
    values > 1.1 suggest chains have not mixed.

    Args:
        chains: array of shape (n_chains, n_draws, n_params).

    Returns:
        R-hat per parameter, shape (n_params,).
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[np.newaxis, :, :]  # treat as single chain

    n_chains, n_draws, n_params = chains.shape
    # Split each chain in half → 2*n_chains sub-chains
    half = n_draws // 2
    if half < 2:
        return np.ones(n_params)

    left = chains[:, :half, :]
    right = chains[:, half : 2 * half, :]
    split = np.concatenate([left, right], axis=0)  # (2*M, half, n_params)

    # Rank-normalise
    split_rn = _rank_normalize(split)

    n_m = split_rn.shape[0]  # 2*M
    n_n = split_rn.shape[1]  # half

    chain_mean = split_rn.mean(axis=1)           # (2M, n_params)
    grand_mean = chain_mean.mean(axis=0)         # (n_params,)

    B = n_n / (n_m - 1) * np.sum((chain_mean - grand_mean) ** 2, axis=0)
    W = np.mean(np.var(split_rn, axis=1, ddof=1), axis=0)

    var_hat = (n_n - 1) / n_n * W + B / n_n
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.where(W > 0, np.sqrt(var_hat / W), np.ones(n_params))
    return np.clip(rhat, 1.0, None)


def compute_ess(chains: np.ndarray) -> np.ndarray:
    """
    Effective sample size via rank-normalised autocorrelation.

    Uses the initial positive sequence estimator (Geyer 1992) on the
    rank-normalised split chains.

    Args:
        chains: array of shape (n_chains, n_draws, n_params).

    Returns:
        ESS per parameter, shape (n_params,).
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[np.newaxis, :, :]

    n_chains, n_draws, n_params = chains.shape
    half = n_draws // 2
    if half < 2:
        return np.full(n_params, float(n_chains * n_draws))

    left = chains[:, :half, :]
    right = chains[:, half : 2 * half, :]
    split = np.concatenate([left, right], axis=0)
    split_rn = _rank_normalize(split)

    n_m, n_n, _ = split_rn.shape
    ess = np.empty(n_params)

    for p in range(n_params):
        data = split_rn[:, :, p]          # (n_m, n_n)
        # Compute mean autocorrelation across chains at each lag
        acov = np.array([
            np.mean([
                np.cov(data[m, : n_n - lag], data[m, lag:], ddof=1)[0, 1]
                if lag < n_n - 1 else 0.0
                for m in range(n_m)
            ])
            for lag in range(n_n)
        ])
        var_plus = acov[0]
        if var_plus <= 0:
            ess[p] = float(n_m * n_n)
            continue
        # Normalise autocorrelations
        rho = acov / var_plus
        # Initial positive sequence estimator
        sum_rho = 0.0
        for t in range(1, n_n - 1, 2):
            pair = rho[t] + rho[t + 1]
            if pair < 0:
                break
            sum_rho += pair
        ess[p] = max(1.0, (n_m * n_n) / (1.0 + 2.0 * sum_rho))

    return ess


def compute_autocorr(chain: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Sample autocorrelation at each lag for a single chain.

    Args:
        chain:   shape (n_draws,) or (n_draws, n_params).
        max_lag: Maximum lag to compute.

    Returns:
        Autocorrelation array of shape (max_lag + 1, n_params) or
        (max_lag + 1,) if chain is 1-D.
    """
    chain = np.asarray(chain, dtype=float)
    squeeze = chain.ndim == 1
    if squeeze:
        chain = chain[:, np.newaxis]
    n_draws, n_params = chain.shape
    max_lag = min(max_lag, n_draws - 2)
    result = np.empty((max_lag + 1, n_params))
    for p in range(n_params):
        x = chain[:, p] - chain[:, p].mean()
        c0 = float(np.dot(x, x)) / n_draws
        result[0, p] = 1.0
        for lag in range(1, max_lag + 1):
            result[lag, p] = float(np.dot(x[: n_draws - lag], x[lag:])) / (n_draws * c0) if c0 > 0 else 0.0
    return result[:, 0] if squeeze else result
