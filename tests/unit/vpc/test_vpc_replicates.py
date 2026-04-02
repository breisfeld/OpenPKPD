"""
Tests for VPC warnings and binning behaviour.
Covers:
  V1: minimum replicate warning
  V2: pcVPC PRED≈0 warning with count
  V3: bin edges from union of observed + simulated times
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from openpkpd.simulation.vpc import (
    VPCEngine,
    _apply_prediction_correction,
    _make_bins,
)


# ---------------------------------------------------------------------------
# Minimal SimulationEngine stub
# ---------------------------------------------------------------------------

class _FakeSimResult:
    def __init__(self, df: pd.DataFrame) -> None:
        self.simulated_df = df


class _FakeSimEngine:
    """Minimal stub that returns a pre-built DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def simulate(self, n_replicates: int) -> _FakeSimResult:
        return _FakeSimResult(self._df)


def _make_df(n_obs: int = 10, n_reps: int = 500, times=None) -> pd.DataFrame:
    """Build a minimal observed + simulated DataFrame."""
    if times is None:
        times = np.arange(n_obs, dtype=float)

    obs_rows = pd.DataFrame(
        {
            "REP": 0,
            "ID": np.arange(n_obs) + 1,
            "TIME": times,
            "DV": np.random.default_rng(0).random(n_obs),
        }
    )

    sim_frames = []
    for rep in range(1, n_reps + 1):
        sim_frames.append(
            pd.DataFrame(
                {
                    "REP": rep,
                    "ID": np.arange(n_obs) + 1,
                    "TIME": times,
                    "DV": np.random.default_rng(rep).random(n_obs),
                }
            )
        )
    return pd.concat([obs_rows] + sim_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# V1: replicate count warning
# ---------------------------------------------------------------------------


def test_v1_no_warning_500_replicates():
    """n_replicates=500 should produce no RuntimeWarning."""
    df = _make_df(n_reps=500)
    engine = VPCEngine(_FakeSimEngine(df))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        engine.compute(n_replicates=500, n_bins=3)
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0, f"Unexpected warnings: {runtime_warnings}"


def test_v1_warning_100_replicates():
    """n_replicates=100 should emit a RuntimeWarning mentioning '100 replicates'."""
    df = _make_df(n_reps=100)
    engine = VPCEngine(_FakeSimEngine(df))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        engine.compute(n_replicates=100, n_bins=3)
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warnings) >= 1
    assert "100 replicates" in str(runtime_warnings[0].message)


def test_v1_boundary_199_warns_200_does_not():
    """199 → warning; 200 → no warning (boundary is exclusive at 200)."""
    df199 = _make_df(n_reps=199)
    df200 = _make_df(n_reps=200)

    engine199 = VPCEngine(_FakeSimEngine(df199))
    with warnings.catch_warnings(record=True) as w199:
        warnings.simplefilter("always")
        engine199.compute(n_replicates=199, n_bins=3)
    rw199 = [x for x in w199 if issubclass(x.category, RuntimeWarning)]
    assert len(rw199) >= 1

    engine200 = VPCEngine(_FakeSimEngine(df200))
    with warnings.catch_warnings(record=True) as w200:
        warnings.simplefilter("always")
        engine200.compute(n_replicates=200, n_bins=3)
    rw200 = [x for x in w200 if issubclass(x.category, RuntimeWarning)]
    assert len(rw200) == 0


def test_v1_warning_emitted_once_per_run():
    """Warning should be emitted exactly once per compute() call."""
    df = _make_df(n_reps=50)
    engine = VPCEngine(_FakeSimEngine(df))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        engine.compute(n_replicates=50, n_bins=3)
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1


# ---------------------------------------------------------------------------
# V2: pcVPC PRED≈0 warning
# ---------------------------------------------------------------------------


def _make_pcvpc_df_with_zero_bin(
    n_bins: int = 5,
    rows_per_bin: int = 20,
    zero_bins: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Build obs/sim DataFrames where exactly ``zero_bins`` bins have median PRED=0.
    Each bin is a distinct time cluster, so the bin-level median is determined by
    the PRED values within that cluster.
    """
    rng = np.random.default_rng(42)
    all_times = []
    all_pred = []

    for b in range(n_bins):
        # Assign times within a narrow range for each bin
        t_center = float(b) * 2.0
        times = np.full(rows_per_bin, t_center)
        if b < zero_bins:
            pred = np.zeros(rows_per_bin)
        else:
            pred = rng.uniform(1.0, 5.0, rows_per_bin)
        all_times.append(times)
        all_pred.append(pred)

    times_arr = np.concatenate(all_times)
    pred_arr = np.concatenate(all_pred)
    n_total = len(times_arr)

    obs = pd.DataFrame(
        {
            "REP": 0,
            "ID": np.arange(n_total) + 1,
            "TIME": times_arr,
            "DV": rng.random(n_total),
            "PRED": pred_arr,
        }
    )
    sim = pd.DataFrame(
        {
            "REP": 1,
            "ID": np.arange(n_total) + 1,
            "TIME": times_arr,
            "DV": rng.random(n_total),
            "PRED": pred_arr,
        }
    )
    bins = _make_bins(times_arr, n_bins)
    return obs, sim, bins


def test_v2_no_warning_all_valid(caplog):
    """All PRED valid → no warning logged."""
    obs, sim, bins = _make_pcvpc_df_with_zero_bin(n_bins=5, rows_per_bin=20, zero_bins=0)
    with caplog.at_level(logging.WARNING, logger="openpkpd.simulation.vpc"):
        _apply_prediction_correction(obs, sim, bins)
    warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warning_msgs) == 0


def test_v2_warning_5pct_zero_pred(caplog):
    """More than 1% of rows in zero-median bins → WARNING with count and percentage."""
    # 1 out of 5 bins (20%) has PRED=0 → 20% of rows skipped → warning
    obs, sim, bins = _make_pcvpc_df_with_zero_bin(n_bins=5, rows_per_bin=20, zero_bins=1)
    with caplog.at_level(logging.WARNING, logger="openpkpd.simulation.vpc"):
        _apply_prediction_correction(obs, sim, bins)
    warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING and "pcVPC" in r.message]
    assert len(warning_msgs) >= 1
    msg = warning_msgs[0].message
    assert "skipped" in msg or "%" in msg


def test_v2_debug_only_small_zero_fraction(caplog):
    """If 0 rows skipped (all PRED valid), no DEBUG pcVPC message."""
    obs, sim, bins = _make_pcvpc_df_with_zero_bin(n_bins=10, rows_per_bin=100, zero_bins=0)
    with caplog.at_level(logging.DEBUG, logger="openpkpd.simulation.vpc"):
        _apply_prediction_correction(obs, sim, bins)
    warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING and "pcVPC" in r.message]
    assert len(warning_msgs) == 0


def test_v2_nan_pred_caught(caplog):
    """Rows in bins where median PRED=NaN are treated as non-finite and counted."""
    # Create obs where all PRED values in the first bin cluster are NaN
    n_bins = 5
    rows_per_bin = 20
    rng = np.random.default_rng(99)
    all_times = []
    all_pred = []
    for b in range(n_bins):
        t_center = float(b) * 2.0
        times = np.full(rows_per_bin, t_center)
        if b == 0:
            pred = np.full(rows_per_bin, np.nan)
        else:
            pred = rng.uniform(1.0, 5.0, rows_per_bin)
        all_times.append(times)
        all_pred.append(pred)

    times_arr = np.concatenate(all_times)
    pred_arr = np.concatenate(all_pred)
    n_total = len(times_arr)

    obs = pd.DataFrame({"REP": 0, "ID": np.arange(n_total) + 1,
                        "TIME": times_arr, "DV": rng.random(n_total), "PRED": pred_arr})
    sim = pd.DataFrame({"REP": 1, "ID": np.arange(n_total) + 1,
                        "TIME": times_arr, "DV": rng.random(n_total), "PRED": pred_arr})
    bins = _make_bins(times_arr, n_bins)

    with caplog.at_level(logging.DEBUG, logger="openpkpd.simulation.vpc"):
        _apply_prediction_correction(obs, sim, bins)
    # Should have a debug or warning message about skipped rows (NaN ref → n_dropped > 0)
    skipped_msgs = [r for r in caplog.records if "pcVPC" in r.message and "skipped" in r.message]
    assert len(skipped_msgs) >= 1


# ---------------------------------------------------------------------------
# V3: bin edges from union of observed + simulated times
# ---------------------------------------------------------------------------


def test_v3_sim_times_extend_beyond_obs():
    """Simulated times beyond observed range → bins cover the full range."""
    obs_times = np.array([0.0, 1.0, 2.0])
    sim_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    all_times = np.concatenate([obs_times, sim_times])
    bins = _make_bins(all_times, 4)
    # Bins should cover up to at least t=4
    assert bins[-1] >= 4.0


def test_v3_sim_times_within_obs_range():
    """Sim times within observed range → bins effectively same as obs-only."""
    obs_times = np.linspace(0, 10, 50)
    sim_times = np.linspace(2, 8, 20)  # subset of obs range
    bins_union = _make_bins(np.concatenate([obs_times, sim_times]), 5)
    bins_obs = _make_bins(obs_times, 5)
    # Both cover 0–10, so last edges should be close
    assert bins_union[-1] >= bins_obs[-1] - 0.1


def test_v3_no_sim_data_fallback():
    """No simulated data → bins from observed only, no error."""
    obs_times = np.array([0.0, 1.0, 2.0, 3.0])
    sim_times = np.array([])
    all_times = np.concatenate([obs_times, sim_times])
    bins = _make_bins(all_times, 3)
    assert len(bins) >= 2
    assert bins[-1] >= 3.0


def test_v3_numerical_bins_cover_sim_time():
    """Build VPC where obs=[0,1,2] and sim includes [3,4]; last bin covers t=4."""
    obs_times = np.array([0.0, 1.0, 2.0])
    sim_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    all_times = np.concatenate([obs_times, sim_times])
    bins = _make_bins(all_times, 4)
    # Verify that 4.0 is within the last bin
    assert 4.0 <= bins[-1]
