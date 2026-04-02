"""
Normalised Prediction Distribution Errors (NPDE).

Implements the full Brendel 2006 algorithm including within-subject
decorrelation of the prediction distribution errors.

Algorithm (Brendel et al., AAPS J 2006; Comets et al., Comput Methods
Programs Biomed 2008):

  For each subject i with n_i observations and K simulation replicates:

  1. Assemble the (n_i × K) matrix Y_sim_i of simulated DV values.
  2. Compute the marginal predictive CDF at each observed value:
       pd_{ij} = (#{r : Y_sim_{ij}^r < y_{obs,ij}} + 0.5 * #{r : equal}) / K
  3. Estimate the within-subject correlation matrix of the predictive
     distribution from the simulated replicates:
       C_i = Corr(Y_sim_i)   [correlation across K replicates, shape n_i × n_i]
  4. Transform pd_i to the normal scale:
       npd_i = Phi^{-1}(pd_i)   (prediction distribution errors, PDE)
  5. Decorrelate using the Cholesky factor of C_i:
       npde_i = Chol(C_i)^{-T} @ npd_i
     If C_i is unavailable or n_i = 1, skip decorrelation (npde_i = npd_i).

Under a correctly specified model, NPDE ~ N(0, 1) independently across
all subjects and observations.

Reference:
    Brendel K, Comets E, Laffont CM, Laveille C, Mentré F (2006).
    Metrics for external model evaluation with an application to the
    population pharmacokinetics of gliclazide. Pharm Res 23(9):2036-49.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm
from scipy.stats import shapiro

if TYPE_CHECKING:
    from openpkpd.simulation.engine import SimulationEngine


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NPDEResult:
    """
    Result of a full NPDE computation.

    Attributes:
        df:           DataFrame with columns ID, TIME, DV, PDE, NPDE.
                      PDE = Phi^{-1}(pd_i) before decorrelation.
                      NPDE = decorrelated PDE (Brendel 2006).
        n_replicates: Number of simulation replicates used.
        mean_npde:    Empirical mean of NPDE (should be ~0).
        var_npde:     Empirical variance of NPDE (should be ~1).
        sw_stat:      Shapiro-Wilk test statistic for normality of NPDE.
        sw_pvalue:    Shapiro-Wilk p-value (low → non-normal → model misspecification).
    """

    df: pd.DataFrame
    n_replicates: int
    mean_npde: float = 0.0
    var_npde: float = 1.0
    sw_stat: float = float("nan")
    sw_pvalue: float = float("nan")

    def summary(self) -> str:
        lines = [
            f"NPDE Summary  (K={self.n_replicates} replicates, N={len(self.df)} observations)",
            f"  Mean NPDE : {self.mean_npde:+.4f}  (expected 0)",
            f"  Var  NPDE : {self.var_npde:.4f}  (expected 1)",
        ]
        if np.isfinite(self.sw_pvalue):
            flag = "  ✗ non-normal" if self.sw_pvalue < 0.05 else "  ✓ normal"
            lines.append(f"  Shapiro-Wilk: W={self.sw_stat:.4f}, p={self.sw_pvalue:.4f}{flag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class NPDEEngine:
    """
    Compute Normalised Prediction Distribution Errors (NPDE).

    Uses a :class:`~openpkpd.simulation.engine.SimulationEngine` to
    generate K replicate datasets, then applies the Brendel 2006
    algorithm with within-subject decorrelation.

    Args:
        simulation_engine: A configured and ready SimulationEngine.
    """

    def __init__(self, simulation_engine: SimulationEngine) -> None:
        self.sim_engine = simulation_engine

    # ── Public API ────────────────────────────────────────────────────────

    def compute(
        self,
        n_replicates: int = 1000,
        seed: int = 42,
        decorrelate: bool = True,
    ) -> NPDEResult:
        """
        Run NPDE computation.

        Args:
            n_replicates: Number of Monte Carlo replicates.  More replicates
                          give smoother NPDEs; ≥ 500 recommended.
            seed:         Random seed passed to the SimulationEngine.
            decorrelate:  If True (default), apply within-subject Cholesky
                          decorrelation (Brendel 2006 step 5).  Set False to
                          return raw PDE (= NPDE without decorrelation).

        Returns:
            NPDEResult with NPDE values and summary statistics.
        """
        if hasattr(self.sim_engine, "seed"):
            with contextlib.suppress(Exception):
                self.sim_engine.seed = seed
        if hasattr(self.sim_engine, "rng"):
            with contextlib.suppress(Exception):
                self.sim_engine.rng = np.random.default_rng(seed)

        # Generate K replicate datasets
        sim_result = self.sim_engine.simulate(n_replicates=n_replicates)
        full_df = sim_result.simulated_df

        observed_df = full_df[full_df["REP"] == 0].copy()
        simulated_df = full_df[full_df["REP"] >= 1].copy()

        # Drop dosing rows
        for df_ in (observed_df, simulated_df):
            if "MDV" in df_.columns:
                df_.drop(df_[df_["MDV"] != 0].index, inplace=True)

        observed_df["_OBSSEQ"] = (
            observed_df.groupby(["ID", "TIME"], sort=False).cumcount()
            if not observed_df.empty
            else pd.Series(index=observed_df.index, dtype=int)
        )
        simulated_df["_OBSSEQ"] = (
            simulated_df.groupby(["ID", "REP", "TIME"], sort=False).cumcount()
            if not simulated_df.empty
            else pd.Series(index=simulated_df.index, dtype=int)
        )

        sim_grouped = simulated_df.groupby("ID")
        frames: list[pd.DataFrame] = []

        for sid, obs_grp in observed_df.groupby("ID"):
            obs_grp = obs_grp.sort_values(["TIME", "_OBSSEQ"], kind="mergesort").reset_index(
                drop=True
            )
            try:
                sim_grp = sim_grouped.get_group(sid)
            except KeyError:
                sim_grp = simulated_df.iloc[0:0]

            obs_index = obs_grp[["TIME", "_OBSSEQ"]].copy()
            obs_times = obs_index["TIME"].to_numpy(dtype=float)
            obs_dv = obs_grp["DV"].to_numpy(dtype=float)
            n_i = len(obs_times)

            # Build (n_i, K) simulated DV matrix
            Y_sim = self._build_sim_matrix(sim_grp, obs_index, n_replicates)

            # Step 2 — marginal predictive CDF per observation
            pd_i = _compute_pd(obs_dv, Y_sim)

            # Step 4 — transform to normal scale (PDE)
            pde_i = _pd_to_normal(pd_i)

            # Step 3 + 5 — within-subject decorrelation
            npde_i = _decorrelate(pde_i, Y_sim) if decorrelate and n_i > 1 else pde_i.copy()

            frames.append(
                pd.DataFrame(
                    {
                        "ID": np.full(n_i, sid),
                        "TIME": obs_times,
                        "DV": obs_dv,
                        "PDE": pde_i,
                        "NPDE": npde_i,
                    }
                )
            )

        if frames:
            df_out = pd.concat(frames, ignore_index=True)
        else:
            df_out = pd.DataFrame(columns=["ID", "TIME", "DV", "PDE", "NPDE"])

        # Summary statistics
        npde_valid = df_out["NPDE"].dropna().to_numpy(dtype=float)
        mean_npde = float(np.mean(npde_valid)) if len(npde_valid) else float("nan")
        var_npde = float(np.var(npde_valid, ddof=1)) if len(npde_valid) > 1 else float("nan")

        sw_stat, sw_pvalue = float("nan"), float("nan")
        if 3 <= len(npde_valid) <= 5000:
            try:
                sw_stat, sw_pvalue = shapiro(npde_valid)
                sw_stat, sw_pvalue = float(sw_stat), float(sw_pvalue)
            except Exception:
                pass

        return NPDEResult(
            df=df_out,
            n_replicates=n_replicates,
            mean_npde=mean_npde,
            var_npde=var_npde,
            sw_stat=sw_stat,
            sw_pvalue=sw_pvalue,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_sim_matrix(
        self,
        sim_grp: pd.DataFrame,
        obs_index: pd.DataFrame,
        n_replicates: int,
    ) -> np.ndarray:
        """
        Build a (n_i, K) matrix of simulated DV values.

        For each observation time t_j, collect the simulated DV across all
        K replicates.  If a replicate has no row at time t_j (e.g. due to
        floating-point mismatch) the cell is filled with NaN.

        Args:
            sim_grp:      Simulated rows for one subject (all REP ≥ 1).
            obs_index:    Observed TIME/_OBSSEQ table, shape (n_i, 2).
            n_replicates: Number of replicates K.

        Returns:
            Array of shape (n_i, K).  May contain NaN for missing cells.
        """
        n_i = len(obs_index)
        Y_sim = np.full((n_i, n_replicates), np.nan)
        if n_i == 0 or sim_grp.empty or n_replicates <= 0:
            return Y_sim

        obs_lookup = obs_index.reset_index(drop=True).copy()
        obs_lookup["_OBSROW"] = np.arange(n_i, dtype=int)
        obs_lookup["_TIME_KEY"] = np.round(obs_lookup["TIME"].to_numpy(dtype=float), 9)

        sim_lookup = sim_grp[["REP", "TIME", "DV", "_OBSSEQ"]].copy()
        sim_lookup["_TIME_KEY"] = np.round(sim_lookup["TIME"].to_numpy(dtype=float), 9)

        merged = sim_lookup.merge(
            obs_lookup[["_TIME_KEY", "_OBSSEQ", "_OBSROW"]],
            on=["_TIME_KEY", "_OBSSEQ"],
            how="inner",
            sort=False,
        )
        if merged.empty:
            return Y_sim

        rep_idx = merged["REP"].to_numpy(dtype=int) - 1
        row_idx = merged["_OBSROW"].to_numpy(dtype=int)
        dv_vals = merged["DV"].to_numpy(dtype=float)
        valid = (rep_idx >= 0) & (rep_idx < n_replicates)
        Y_sim[row_idx[valid], rep_idx[valid]] = dv_vals[valid]

        return Y_sim


# ---------------------------------------------------------------------------
# Pure-function helpers (no state)
# ---------------------------------------------------------------------------


def _compute_pd(obs_dv: np.ndarray, Y_sim: np.ndarray) -> np.ndarray:
    """
    Compute marginal predictive CDF values pd_{ij} for one subject.

    pd_{ij} = (#{r : Y_sim[j,r] < obs_dv[j]} + 0.5 * #{equal}) / K

    NaN columns in Y_sim (missing replicates) are excluded from the count.

    Args:
        obs_dv: Observed DV, shape (n_i,).
        Y_sim:  Simulated DV matrix, shape (n_i, K).

    Returns:
        pd array, shape (n_i,), values in (0, 1).
    """
    if Y_sim.size == 0:
        return np.full(Y_sim.shape[0], np.nan)

    valid = ~np.isnan(Y_sim)
    k_counts = valid.sum(axis=1)
    obs_col = obs_dv[:, None]
    n_less = np.sum((Y_sim < obs_col) & valid, axis=1, dtype=float)
    n_eq = np.sum((Y_sim == obs_col) & valid, axis=1, dtype=float)

    pd_vals = np.full(Y_sim.shape[0], np.nan)
    valid_rows = k_counts > 0
    if not np.any(valid_rows):
        return pd_vals

    k_float = k_counts[valid_rows].astype(float)
    p = (n_less[valid_rows] + 0.5 * n_eq[valid_rows]) / k_float
    lower = 0.5 / k_float
    pd_vals[valid_rows] = np.clip(p, lower, 1.0 - lower)
    return pd_vals


def _pd_to_normal(pd_vals: np.ndarray) -> np.ndarray:
    """Transform pd values to normal scale: PDE = Phi^{-1}(pd)."""
    out = np.full_like(pd_vals, np.nan)
    finite = np.isfinite(pd_vals)
    out[finite] = sp_norm.ppf(pd_vals[finite])
    return out


def _decorrelate(pde_i: np.ndarray, Y_sim: np.ndarray) -> np.ndarray:
    """
    Within-subject Cholesky decorrelation (Brendel 2006 step 5).

    Estimate the correlation matrix C_i of the predictive distribution
    from Y_sim, then return L^{-T} @ pde_i where L = Chol(C_i).

    Args:
        pde_i: Raw PDE vector for one subject, shape (n_i,).
        Y_sim: Simulated DV matrix, shape (n_i, K).

    Returns:
        Decorrelated NPDE vector, shape (n_i,).
    """
    n_i = pde_i.shape[0]

    # Use only columns (replicates) that are complete for this subject
    complete_cols = ~np.any(np.isnan(Y_sim), axis=0)
    Y_complete = Y_sim[:, complete_cols]
    K_avail = Y_complete.shape[1]

    if K_avail < n_i + 2:
        import warnings

        warnings.warn(
            f"NPDE: insufficient replicates for subject with {n_i} observations "
            f"(need \u2265 {n_i + 2}, have {K_avail}). "
            "Decorrelation skipped; raw PDE values returned.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pde_i.copy()

    # Correlation matrix of predictive distribution (n_i × n_i)
    # Each row of Y_complete is one observation time; Pearson across K replicates
    try:
        C_i = np.corrcoef(Y_complete)  # shape (n_i, n_i)
    except Exception:
        return pde_i.copy()

    # Regularise: add jitter if near-singular
    min_eig = float(np.linalg.eigvalsh(C_i).min())
    if min_eig < 1e-8:
        C_i += np.eye(n_i) * (abs(min_eig) + 1e-6)

    # Cholesky and back-substitution: solve L @ x = pde_i for x
    try:
        from scipy.linalg import cholesky, solve_triangular

        L = cholesky(C_i, lower=True)
        # Handle NaN in pde_i: replace with 0, decorrelate, restore NaN
        nan_mask = ~np.isfinite(pde_i)
        pde_safe = pde_i.copy()
        pde_safe[nan_mask] = 0.0
        npde_i = solve_triangular(L, pde_safe, lower=True)
        npde_i[nan_mask] = np.nan
        return npde_i
    except Exception:
        return pde_i.copy()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = ["NPDEEngine", "NPDEResult"]
