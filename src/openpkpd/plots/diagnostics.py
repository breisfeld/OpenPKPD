"""
Compute diagnostic DataFrame from a fitted PopulationModel + EstimationResult.

This is the central data source for all plot functions.

Columns returned (EVID=0, MDV=0 rows only):
  ID, TIME, DV, PRED, IPRED, RES, IRES, WRES, IWRES, CWRES,
  ETA1..ETAn, MDV, EVID
  + any covariate columns present in population_model.dataset.df

Additional functions:
  compute_npde(): Add Normalized Prediction Distribution Errors (NPDE) column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.linalg import cholesky as scipy_chol
from scipy.linalg import solve_triangular

if TYPE_CHECKING:
    from openpkpd.estimation.base import EstimationResult
    from openpkpd.model.population import PopulationModel


def _finite_diff_jacobian(
    indiv,
    theta: np.ndarray,
    eta: np.ndarray,
    sigma: np.ndarray,
    trans: int,
    h: float = 1e-5,
) -> np.ndarray:
    """
    Compute Jacobian R_i = d(IPRED)/d(eta) at given eta via central differences.

    Returns array of shape (n_obs, n_eta).
    """
    n_eta = len(eta)
    ipred0, obs_mask, _ = indiv.evaluate(theta, eta, sigma, trans=trans)
    n_obs = int(np.sum(obs_mask))
    if n_obs == 0 or n_eta == 0:
        return np.zeros((n_obs, n_eta))

    R = np.zeros((n_obs, n_eta))
    for k in range(n_eta):
        eta_p = eta.copy()
        eta_m = eta.copy()
        eta_p[k] += h
        eta_m[k] -= h
        ip_p, _, _ = indiv.evaluate(theta, eta_p, sigma, trans=trans)
        ip_m, _, _ = indiv.evaluate(theta, eta_m, sigma, trans=trans)
        R[:, k] = (ip_p[obs_mask] - ip_m[obs_mask]) / (2 * h)

    return R


def _prediction_eta_jacobian(
    indiv,
    theta: np.ndarray,
    eta: np.ndarray,
    sigma: np.ndarray,
    trans: int,
) -> np.ndarray:
    """Use a subject-native Jacobian when available; otherwise finite-difference it."""
    supports = getattr(indiv, "supports_prediction_eta_jacobian", None)
    jac_fn = getattr(indiv, "prediction_eta_jacobian", None)
    try:
        if callable(supports) and supports(trans) and callable(jac_fn):
            jac = np.asarray(jac_fn(theta, eta, sigma, trans=trans), dtype=float)
            _, obs_mask, _ = indiv.evaluate(theta, eta, sigma, trans=trans)
            if jac.shape == (int(np.sum(obs_mask)), len(eta)):
                return jac
    except Exception:
        pass
    return _finite_diff_jacobian(indiv, theta, eta, sigma, trans)


def _cwres_subject(
    dv: np.ndarray,
    pred: np.ndarray,
    ipred: np.ndarray,
    eta_hat: np.ndarray,
    R_i: np.ndarray,
    omega: np.ndarray,
    sigma_diag: np.ndarray,
) -> np.ndarray:
    """
    Compute CWRES for one subject.

    CWRES = C_i^{-1/2} * (DV - PRED - R_i @ eta_hat)

    C_i = R_i @ Omega @ R_i^T + diag(sigma_diag)
    """
    n_obs = len(dv)
    if n_obs == 0:
        return np.array([])

    # Clamp sigma_diag to avoid degenerate C_i
    sigma_diag_clamped = np.maximum(sigma_diag, 1e-10)

    # Build C_i
    C_i = R_i @ omega @ R_i.T + np.diag(sigma_diag_clamped)

    # Cholesky of C_i
    try:
        L = scipy_chol(C_i, lower=True)
        resid = dv - pred - R_i @ eta_hat
        cwres = solve_triangular(L, resid, lower=True)
    except Exception:
        # Fallback to IWRES
        w = np.sqrt(np.maximum(sigma_diag_clamped, 1e-10))
        cwres = (dv - ipred) / w

    return cwres


def compute_diagnostics(
    population_model: PopulationModel,
    result: EstimationResult,
) -> pd.DataFrame:
    """
    Compute diagnostic DataFrame from a fitted model.

    Args:
        population_model: Assembled PopulationModel with dataset.
        result: EstimationResult from model.fit().

    Returns:
        DataFrame with ID, TIME, DV, PRED, IPRED, RES, IRES, WRES, IWRES, CWRES,
        ETA1..ETAn, MDV, EVID, and any covariate columns.
        Only rows with EVID=0 and MDV=0 are included.
    """
    theta = result.theta_final
    omega = result.omega_final
    sigma = result.sigma_final
    n_eta = omega.shape[0]
    trans = population_model.trans

    rows: list[dict] = []

    for sid in population_model.subject_ids():
        indiv = population_model.individual_model(sid)
        eta_hat = result.post_hoc_etas.get(sid, np.zeros(n_eta))
        eta_zero = np.zeros(n_eta)

        # PRED: evaluate at eta=0
        pred_arr, obs_mask, _ = indiv.evaluate(theta, eta_zero, sigma, trans=trans)
        # IPRED: evaluate at eta_hat
        ipred_arr, obs_mask2, _ = indiv.evaluate(theta, eta_hat, sigma, trans=trans)

        if not np.any(obs_mask):
            continue

        dv = indiv.subject_events.obs_dv[obs_mask]
        times = indiv.subject_events.obs_times[obs_mask]
        pred = pred_arr[obs_mask]
        ipred = ipred_arr[obs_mask]

        # Residual variance per observation (proportional component)
        sigma_val = float(sigma[0, 0]) if sigma.size > 0 else 1.0
        sigma_diag = ipred**2 * sigma_val

        # Jacobian R_i
        R_i = _prediction_eta_jacobian(indiv, theta, eta_hat, sigma, trans)

        # WRES (uses C_i at eta=0)
        sigma_diag_pred = pred**2 * sigma_val
        sigma_diag_pred = np.maximum(sigma_diag_pred, 1e-10)
        if n_eta > 0:
            R_pred = _prediction_eta_jacobian(indiv, theta, eta_zero, sigma, trans)
            C_i_pred = R_pred @ omega @ R_pred.T + np.diag(sigma_diag_pred)
        else:
            C_i_pred = np.diag(sigma_diag_pred)

        try:
            L_pred = scipy_chol(C_i_pred, lower=True)
            wres = solve_triangular(L_pred, dv - pred, lower=True)
        except Exception:
            wres = (dv - pred) / np.sqrt(sigma_diag_pred)

        # IWRES
        w = np.sqrt(np.maximum(sigma_diag, 1e-10))
        iwres = (dv - ipred) / w

        # CWRES
        if n_eta > 0:
            cwres = _cwres_subject(dv, pred, ipred, eta_hat, R_i, omega, sigma_diag)
        else:
            cwres = wres.copy()

        n_obs = len(dv)
        for j in range(n_obs):
            row: dict = {
                "ID": sid,
                "TIME": float(times[j]),
                "DV": float(dv[j]),
                "PRED": float(pred[j]),
                "IPRED": float(ipred[j]),
                "RES": float(dv[j] - pred[j]),
                "IRES": float(dv[j] - ipred[j]),
                "WRES": float(wres[j]),
                "IWRES": float(iwres[j]),
                "CWRES": float(cwres[j]) if len(cwres) > j else float(iwres[j]),
                "MDV": 0,
                "EVID": 0,
            }
            for k in range(n_eta):
                row[f"ETA{k + 1}"] = float(eta_hat[k])
            rows.append(row)

    df = pd.DataFrame(rows)

    # Merge covariate columns from dataset
    if len(df) > 0 and population_model.covariate_columns:
        available_covariates = [
            c
            for c in population_model.covariate_columns
            if c in population_model.dataset.df.columns
        ]
        if available_covariates:
            source_df = population_model.dataset.df
            if "TIME" in source_df.columns and "TIME" in df.columns:
                cov_df = source_df.copy()
                if "EVID" in cov_df.columns:
                    cov_df = cov_df[cov_df["EVID"].fillna(0) == 0]
                if "MDV" in cov_df.columns:
                    cov_df = cov_df[cov_df["MDV"].fillna(0) == 0]
                cov_df = cov_df[["ID", "TIME", *available_covariates]].copy()
                df = df.copy()
                df["_OBSSEQ"] = df.groupby(["ID", "TIME"], sort=False).cumcount()
                cov_df["_OBSSEQ"] = cov_df.groupby(["ID", "TIME"], sort=False).cumcount()
                df = df.merge(cov_df, on=["ID", "TIME", "_OBSSEQ"], how="left", sort=False)
                df.drop(columns=["_OBSSEQ"], inplace=True)
            else:
                cov_df = source_df[["ID", *available_covariates]].drop_duplicates(subset=["ID"])
                df = df.merge(cov_df, on="ID", how="left")

    return df


def compute_npde(
    population_model: PopulationModel,
    result: EstimationResult,
    n_simulations: int = 1000,
    seed: int = 42,
    decorrelate: bool = True,
) -> pd.DataFrame:
    """
    Compute Normalised Prediction Distribution Errors (NPDE).

    Delegates to :class:`~openpkpd.simulation.npde.NPDEEngine` which
    implements the full Brendel 2006 algorithm including within-subject
    Cholesky decorrelation.

    Under a correctly specified model, NPDE ~ N(0, 1) independently across
    all observations.  Systematic deviations indicate model misspecification.

    Args:
        population_model: Assembled PopulationModel with dataset.
        result:           EstimationResult from model.fit().
        n_simulations:    Number of Monte Carlo replicates (default 1000).
                          Use ≥ 500 for reliable results.
        seed:             Random seed for reproducibility (default 42).
        decorrelate:      Apply within-subject Cholesky decorrelation
                          (Brendel 2006 step 5).  Default True.

    Returns:
        DataFrame identical to compute_diagnostics() output with additional
        columns ``PDE`` (before decorrelation) and ``NPDE`` (decorrelated).
    """
    from openpkpd.simulation.engine import SimulationEngine
    from openpkpd.simulation.npde import NPDEEngine

    diag_df = compute_diagnostics(population_model, result)

    if len(diag_df) == 0:
        diag_df["PDE"] = pd.Series(dtype=float)
        diag_df["NPDE"] = pd.Series(dtype=float)
        return diag_df

    sim_engine = SimulationEngine(population_model, result, seed=seed)
    npde_engine = NPDEEngine(sim_engine)
    npde_result = npde_engine.compute(
        n_replicates=n_simulations,
        seed=seed,
        decorrelate=decorrelate,
    )

    # Preserve repeated same-time observations by merging with within-time
    # occurrence order rather than performing a many-to-many ID/TIME join.
    diag_df = diag_df.copy()
    diag_df["_OBSSEQ"] = diag_df.groupby(["ID", "TIME"], sort=False).cumcount()

    npde_df = npde_result.df[["ID", "TIME", "PDE", "NPDE"]].copy()
    npde_df["_OBSSEQ"] = npde_df.groupby(["ID", "TIME"], sort=False).cumcount()

    diag_df = diag_df.merge(
        npde_df,
        on=["ID", "TIME", "_OBSSEQ"],
        how="left",
        sort=False,
    )
    diag_df.drop(columns=["_OBSSEQ"], inplace=True)

    return diag_df
