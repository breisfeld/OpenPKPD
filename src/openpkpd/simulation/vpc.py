"""
VPCEngine: Visual Predictive Check computation from simulation results.

Computes observed and simulated percentile bands for VPC plots.
Supports:
  - Standard VPC (quantiles of observed vs. simulated concentrations)
  - Prediction-corrected VPC (pcVPC)
  - Stratification by a covariate column
  - Configurable time bins and quantile levels

References:
  Karlsson MO, Holford N. A Tutorial on Visual Predictive Checks.
  PAGE 2008. Abstr 1434.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openpkpd.simulation.engine import SimulationEngine


@dataclass
class VPCResult:
    """
    Result from a VPC computation.

    Attributes:
        observed_df:      Observed data (REP=0 rows from simulation).
        simulated_df:     All simulated replicates (REP>=1).
        obs_bins:         Bin edges used for time binning.
        obs_percentiles:  Observed percentiles per bin.
                          Columns: bin_mid, p5, p50, p95.
        sim_percentiles:  Simulated percentile confidence intervals per bin.
                          Columns: bin_mid, p5_lo, p5_mid, p5_hi,
                                   p50_lo, p50_mid, p50_hi,
                                   p95_lo, p95_mid, p95_hi.
        n_replicates:     Number of simulated replicates used.
    """

    observed_df: pd.DataFrame
    simulated_df: pd.DataFrame
    obs_bins: np.ndarray
    obs_percentiles: pd.DataFrame
    sim_percentiles: pd.DataFrame
    n_replicates: int
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95)


def _quantile_label(q: float) -> str:
    """Return a stable column label for a quantile in [0, 1]."""
    pct_str = f"{100.0 * float(q):g}".replace(".", "_")
    return f"p{pct_str}"


def _validate_quantiles(quantiles: tuple[float, ...]) -> tuple[float, float, float]:
    """Validate and unpack the three requested VPC quantiles."""
    if len(quantiles) != 3:
        raise ValueError(
            f"quantiles must have exactly 3 elements (low, mid, high), got {len(quantiles)}"
        )

    q_lo, q_mid, q_hi = (float(q) for q in quantiles)
    if any(q < 0.0 or q > 1.0 for q in (q_lo, q_mid, q_hi)):
        raise ValueError("quantiles must lie in [0, 1]")
    if not (q_lo < q_mid < q_hi):
        raise ValueError("quantiles must be strictly increasing")
    return q_lo, q_mid, q_hi


class VPCEngine:
    """
    Compute Visual Predictive Check (VPC) statistics.

    Runs the simulation engine to generate replicate datasets, then
    bins the data in time and computes observed/simulated percentile bands.

    Args:
        simulation_engine: A configured SimulationEngine instance.
    """

    def __init__(self, simulation_engine: SimulationEngine) -> None:
        self.sim_engine = simulation_engine

    def compute(
        self,
        n_replicates: int = 500,
        n_bins: int = 10,
        quantiles: tuple[float, ...] = (0.05, 0.50, 0.95),
        stratify_by: str | None = None,
        prediction_corrected: bool = False,
    ) -> VPCResult:
        """
        Run VPC simulation and compute quantile statistics.

        Args:
            n_replicates:         Number of Monte Carlo replicates (default 500).
            n_bins:               Number of time bins for stratification (default 10).
            quantiles:            Tuple of quantile levels (default (0.05, 0.50, 0.95)).
            stratify_by:          Optional column name to stratify (e.g., 'DOSE').
                                  If None, no stratification.
            prediction_corrected: If True, compute prediction-corrected VPC (pcVPC)
                                  where DV and predictions are normalized by PRED.

        Returns:
            VPCResult with observed and simulated percentile bands.

        Raises:
            ValueError: If quantiles tuple does not have exactly 3 elements.
        """
        q_lo, q_mid, q_hi = _validate_quantiles(quantiles)

        # Generate simulated data (includes REP=0 observed)
        sim_result = self.sim_engine.simulate(n_replicates=n_replicates)
        full_df = sim_result.simulated_df

        # Separate observed (REP=0) and simulated (REP>=1)
        observed_df = full_df[full_df["REP"] == 0].copy()
        simulated_df = full_df[full_df["REP"] >= 1].copy()

        # Filter out MDV=1 rows (missing observations)
        if "MDV" in observed_df.columns:
            observed_df = observed_df[observed_df["MDV"] == 0].copy()
        if "MDV" in simulated_df.columns:
            simulated_df = simulated_df[simulated_df["MDV"] == 0].copy()

        # Build time bins from observed data (before any correction so bin
        # edges are based on the original observation times)
        obs_times = observed_df["TIME"].values
        bins = _make_bins(obs_times, n_bins)

        # Apply prediction-correction if requested (needs bins for median PRED)
        if prediction_corrected:
            observed_df, simulated_df = _apply_prediction_correction(
                observed_df, simulated_df, bins, stratify_by=stratify_by
            )

        # Compute observed percentiles per bin
        obs_percentiles = _compute_obs_percentiles(
            observed_df, bins, q_lo, q_mid, q_hi, stratify_by
        )

        # Compute simulated percentile confidence intervals per bin
        sim_percentiles = _compute_sim_percentiles(
            simulated_df, bins, q_lo, q_mid, q_hi, stratify_by, n_replicates
        )

        return VPCResult(
            observed_df=observed_df,
            simulated_df=simulated_df,
            obs_bins=bins,
            obs_percentiles=obs_percentiles,
            sim_percentiles=sim_percentiles,
            n_replicates=n_replicates,
            quantiles=(q_lo, q_mid, q_hi),
        )


def _make_bins(times: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Build time bin edges using equal-quantile (percentile-based) binning.

    Equal-quantile binning ensures approximately equal numbers of observations
    per bin, which is more robust than equal-width binning for PK data.

    Args:
        times:  Array of observation times.
        n_bins: Target number of bins.

    Returns:
        Array of bin edges, shape (n_bins + 1,).
    """
    if len(times) == 0:
        return np.array([0.0, 1.0])

    percentile_edges = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(times, percentile_edges))

    if len(edges) < 2:
        logger.debug("VPC: equal-quantile binning failed, falling back to linear binning")
        t_min = float(np.min(times))
        t_max = float(np.max(times))
        edges = np.linspace(t_min, t_max + 1e-6, n_bins + 1)

    # Ensure first edge is at or before min time, last edge after max time
    edges[0] = min(edges[0], float(np.min(times))) - 1e-9
    edges[-1] = max(edges[-1], float(np.max(times))) + 1e-9

    return edges


def _bin_data(df: pd.DataFrame, bins: np.ndarray) -> pd.Series:
    """
    Assign each row to a time bin.

    Args:
        df:   DataFrame with 'TIME' column.
        bins: Array of bin edges.

    Returns:
        Series of bin indices (integer codes).
    """
    return pd.cut(df["TIME"], bins=bins, labels=False, include_lowest=True)


def _group_slices(
    df: pd.DataFrame, group_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Sort by grouping columns and return contiguous group slices."""
    work = df.sort_values(group_cols, kind="mergesort").reset_index(drop=True)
    if work.empty:
        empty = np.array([], dtype=int)
        return work, empty, empty

    key_frame = work[group_cols]
    start_mask = key_frame.ne(key_frame.shift()).any(axis=1).to_numpy()
    starts = np.flatnonzero(start_mask)
    ends = np.r_[starts[1:], len(work)]
    return work, starts, ends


def _compute_group_percentiles(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    percentiles: list[float],
    output_cols: list[str],
    *,
    include_count: bool = False,
) -> pd.DataFrame:
    """Compute exact NumPy percentiles for each group without per-group DataFrames."""
    work, starts, ends = _group_slices(df, group_cols)
    if work.empty:
        cols = list(group_cols) + output_cols
        if include_count:
            cols.append("n")
        return pd.DataFrame(columns=cols)

    values = work[value_col].to_numpy(dtype=float)
    keys = work[group_cols].iloc[starts].reset_index(drop=True)
    counts = ends - starts
    if len(counts) > 0 and np.all(counts == counts[0]):
        group_size = int(counts[0])
        grouped = values.reshape(len(starts), group_size)
        out = np.percentile(grouped, percentiles, axis=1).T
    else:
        out = np.empty((len(starts), len(percentiles)), dtype=float)
        for idx, (start, end) in enumerate(zip(starts, ends, strict=False)):
            out[idx, :] = np.percentile(values[start:end], percentiles)

    result = keys.copy()
    for col_idx, col in enumerate(output_cols):
        result[col] = out[:, col_idx]
    if include_count:
        result["n"] = counts
    return result


def _compute_obs_percentiles(
    observed_df: pd.DataFrame,
    bins: np.ndarray,
    q_lo: float,
    q_mid: float,
    q_hi: float,
    stratify_by: str | None,
) -> pd.DataFrame:
    """
    Compute observed percentiles per time bin.

    Args:
        observed_df:  Observed data DataFrame (REP=0).
        bins:         Time bin edges.
        q_lo, q_mid, q_hi: Quantile levels.
        stratify_by:  Optional stratification column.

    Returns:
        DataFrame with columns: bin_mid, pX, pY, pZ
        (or stratify_by + those columns if stratified), where the labels follow
        the requested quantiles.
    """
    df = observed_df.copy()
    df["_bin"] = _bin_data(df, bins)
    bin_mids = (bins[:-1] + bins[1:]) / 2
    q_lo_col = _quantile_label(q_lo)
    q_mid_col = _quantile_label(q_mid)
    q_hi_col = _quantile_label(q_hi)

    group_cols = ["_bin"]
    if stratify_by is not None and stratify_by in df.columns:
        group_cols = [stratify_by, "_bin"]

    valid = df["_bin"].notna() & df["DV"].notna()
    if not valid.any():
        return pd.DataFrame(columns=["bin_mid", q_lo_col, q_mid_col, q_hi_col, "n"])

    result = _compute_group_percentiles(
        df.loc[valid, group_cols + ["DV"]],
        group_cols,
        "DV",
        [q_lo * 100.0, q_mid * 100.0, q_hi * 100.0],
        [q_lo_col, q_mid_col, q_hi_col],
        include_count=True,
    )
    result["bin_mid"] = bin_mids[result["_bin"].to_numpy(dtype=int)]
    result.drop(columns=["_bin"], inplace=True)

    sort_cols = ["bin_mid"]
    if stratify_by is not None and stratify_by in result.columns:
        sort_cols = [stratify_by, "bin_mid"]
    return result.sort_values(sort_cols).reset_index(drop=True)


def _compute_sim_percentiles(
    simulated_df: pd.DataFrame,
    bins: np.ndarray,
    q_lo: float,
    q_mid: float,
    q_hi: float,
    stratify_by: str | None,
    n_replicates: int,
) -> pd.DataFrame:
    """
    Compute simulated percentile confidence intervals per time bin.

    For each bin and each replicate, compute the q_lo, q_mid, q_hi percentile
    of DV. Then across replicates, compute the 5th, 50th, 95th percentile of
    each quantile's distribution (the 90% CI band).

    Args:
        simulated_df:  Simulated data (REP>=1).
        bins:          Time bin edges.
        q_lo, q_mid, q_hi: Primary quantile levels.
        stratify_by:   Optional stratification column.
        n_replicates:  Number of replicates.

    Returns:
        DataFrame with columns following the requested quantile labels, e.g.
        bin_mid, p5_lo, p5_mid, p5_hi, p50_lo, ... for the default quantiles.
    """
    df = simulated_df.copy()
    df["_bin"] = _bin_data(df, bins)
    bin_mids = (bins[:-1] + bins[1:]) / 2
    q_lo_col = _quantile_label(q_lo)
    q_mid_col = _quantile_label(q_mid)
    q_hi_col = _quantile_label(q_hi)

    group_cols = ["REP", "_bin"]
    if stratify_by is not None and stratify_by in df.columns:
        group_cols = [stratify_by, "REP", "_bin"]

    valid = df["_bin"].notna() & df["DV"].notna()
    if not valid.any():
        cols = [
            "bin_mid",
            f"{q_lo_col}_lo",
            f"{q_lo_col}_mid",
            f"{q_lo_col}_hi",
            f"{q_mid_col}_lo",
            f"{q_mid_col}_mid",
            f"{q_mid_col}_hi",
            f"{q_hi_col}_lo",
            f"{q_hi_col}_mid",
            f"{q_hi_col}_hi",
        ]
        return pd.DataFrame(columns=cols)

    rep_df = _compute_group_percentiles(
        df.loc[valid, group_cols + ["DV"]],
        group_cols,
        "DV",
        [q_lo * 100.0, q_mid * 100.0, q_hi * 100.0],
        [q_lo_col, q_mid_col, q_hi_col],
    )

    # Aggregate across replicates per bin
    agg_group_cols = ["_bin"]
    if stratify_by is not None and stratify_by in rep_df.columns:
        agg_group_cols = [stratify_by, "_bin"]

    records = []
    for keys, grp in rep_df.groupby(agg_group_cols, observed=True):
        if isinstance(keys, tuple):
            bin_idx = int(keys[-1])
            strat_val = keys[0] if len(keys) > 1 else None
        else:
            bin_idx = int(keys)
            strat_val = None

        if bin_idx < 0 or bin_idx >= len(bin_mids):
            continue

        rec: dict = {
            "bin_mid": float(bin_mids[bin_idx]),
            f"{q_lo_col}_lo": float(np.percentile(grp[q_lo_col], 5)),
            f"{q_lo_col}_mid": float(np.percentile(grp[q_lo_col], 50)),
            f"{q_lo_col}_hi": float(np.percentile(grp[q_lo_col], 95)),
            f"{q_mid_col}_lo": float(np.percentile(grp[q_mid_col], 5)),
            f"{q_mid_col}_mid": float(np.percentile(grp[q_mid_col], 50)),
            f"{q_mid_col}_hi": float(np.percentile(grp[q_mid_col], 95)),
            f"{q_hi_col}_lo": float(np.percentile(grp[q_hi_col], 5)),
            f"{q_hi_col}_mid": float(np.percentile(grp[q_hi_col], 50)),
            f"{q_hi_col}_hi": float(np.percentile(grp[q_hi_col], 95)),
        }
        if stratify_by is not None and strat_val is not None:
            rec[stratify_by] = strat_val
        records.append(rec)

    if not records:
        return pd.DataFrame(
            columns=[
                "bin_mid",
                f"{q_lo_col}_lo",
                f"{q_lo_col}_mid",
                f"{q_lo_col}_hi",
                f"{q_mid_col}_lo",
                f"{q_mid_col}_mid",
                f"{q_mid_col}_hi",
                f"{q_hi_col}_lo",
                f"{q_hi_col}_mid",
                f"{q_hi_col}_hi",
            ]
        )

    result = pd.DataFrame(records)

    sort_cols = ["bin_mid"]
    if stratify_by is not None and stratify_by in result.columns:
        sort_cols = [stratify_by, "bin_mid"]
    return result.sort_values(sort_cols).reset_index(drop=True)


def _apply_prediction_correction(
    observed_df: pd.DataFrame,
    simulated_df: pd.DataFrame,
    bins: np.ndarray,
    stratify_by: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply prediction correction (Bergstrand et al. 2011 pcVPC method).

    For each time bin b, compute the median observed PRED (median_PRED_b)
    and scale each DV as::

        DV_pc = DV * median_PRED_b / PRED_i

    This preserves the shape of variability while eliminating dose- or
    design-related scale differences between bins.  The same bin-level
    reference (median observed PRED) is used for both observed and simulated
    data, so the correction is internally consistent.

    If PRED is absent from the DataFrames, the function returns them
    unchanged with a UserWarning.

    Args:
        observed_df:  Observed DataFrame (REP=0) with DV and PRED columns.
        simulated_df: Simulated DataFrame (REP≥1) with DV and PRED columns.
        bins:         Time bin edges from :func:`_make_bins`.
        stratify_by:  Optional stratification column. When provided and present,
                      the reference median PRED is computed within each
                      stratum-by-bin group.

    Returns:
        Tuple of (corrected_observed_df, corrected_simulated_df).

    Reference:
        Bergstrand M, Hooker AC, Wallin JE, Karlsson MO (2011).
        Prediction-corrected visual predictive checks for diagnosing
        nonlinear mixed-effects models. AAPS J 13(2):143-51.
    """
    import warnings

    obs = observed_df.copy()
    sim = simulated_df.copy()

    if "PRED" not in obs.columns:
        warnings.warn(
            "pcVPC: PRED column not found in observed data; prediction correction skipped.",
            UserWarning,
            stacklevel=3,
        )
        return obs, sim

    # Assign each observed row to a time bin
    obs["_bin"] = pd.cut(obs["TIME"], bins=bins, labels=False, include_lowest=True)

    # Median observed PRED per bin (the reference level for each bin)
    median_pred_per_bin: dict[int, float] = (
        obs.groupby("_bin", observed=True)["PRED"].median().to_dict()
    )
    median_pred_per_stratum_bin: dict[tuple[object, int], float] = {}
    if stratify_by is not None and stratify_by in obs.columns:
        median_pred_per_stratum_bin = (
            obs.groupby([stratify_by, "_bin"], observed=True)["PRED"].median().to_dict()
        )

    def _correct(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_bin"] = pd.cut(df["TIME"], bins=bins, labels=False, include_lowest=True)
        pred_col = "PRED" if "PRED" in df.columns else None
        group_cols = ["_bin"]
        if stratify_by is not None and stratify_by in df.columns and median_pred_per_stratum_bin:
            group_cols = [stratify_by, "_bin"]
        rows_corrected = []
        for keys, grp in df.groupby(group_cols, observed=True):
            if isinstance(keys, tuple):
                strat_val, bin_idx = keys[0], keys[-1]
                ref = median_pred_per_stratum_bin.get(
                    (strat_val, int(bin_idx)),
                    median_pred_per_bin.get(int(bin_idx), np.nan),
                )
            else:
                bin_idx = keys
                ref = median_pred_per_bin.get(int(bin_idx), np.nan)
            if not np.isfinite(ref) or ref == 0.0:
                rows_corrected.append(grp)
                continue
            grp = grp.copy()
            if pred_col is not None:
                pred_vals = grp[pred_col].replace(0, np.nan)
                grp["DV"] = grp["DV"] * ref / pred_vals
            rows_corrected.append(grp)

        if not rows_corrected:
            return df.drop(columns=["_bin"])
        result = pd.concat(rows_corrected).sort_index()
        result = result.dropna(subset=["DV"])
        return result.drop(columns=["_bin"])

    obs = _correct(obs)
    sim = _correct(sim)

    return obs, sim
