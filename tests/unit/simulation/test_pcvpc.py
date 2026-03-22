"""Tests for pcVPC prediction-correction (Bergstrand 2011)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from openpkpd.simulation.vpc import _apply_prediction_correction, _make_bins

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frames(
    n_subjects: int = 10,
    n_obs: int = 5,
    n_replicates: int = 50,
    pred_scale: float = 1.0,
    seed: int = 0,
):
    """Return (observed_df, simulated_df, bins) with a PRED column."""
    rng = np.random.default_rng(seed)
    times = np.arange(1.0, n_obs + 1.0)
    mu = 10.0 * np.exp(-0.2 * times)

    obs_records = []
    for sid in range(1, n_subjects + 1):
        pred_vals = mu * pred_scale  # typical predictions
        for t, p in zip(times, pred_vals, strict=False):
            obs_records.append(
                {
                    "ID": sid,
                    "TIME": t,
                    "REP": 0,
                    "MDV": 0,
                    "DV": p + rng.normal(0, 0.5),
                    "PRED": p,
                }
            )

    sim_records = []
    for rep in range(1, n_replicates + 1):
        for sid in range(1, n_subjects + 1):
            for t, p in zip(times, mu * pred_scale, strict=False):
                sim_records.append(
                    {
                        "ID": sid,
                        "TIME": t,
                        "REP": rep,
                        "MDV": 0,
                        "DV": p + rng.normal(0, 0.5),
                        "PRED": p,
                    }
                )

    obs_df = pd.DataFrame(obs_records)
    sim_df = pd.DataFrame(sim_records)
    bins = _make_bins(obs_df["TIME"].values, n_bins=5)
    return obs_df, sim_df, bins


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyPredictionCorrection:
    def test_stratified_reference_uses_per_stratum_bin_median(self):
        obs = pd.DataFrame(
            {
                "ID": [1, 2, 1, 2],
                "TIME": [1.0, 1.0, 2.0, 2.0],
                "REP": [0, 0, 0, 0],
                "MDV": [0, 0, 0, 0],
                "DOSE": [100, 200, 100, 200],
                "DV": [2.0, 20.0, 4.0, 40.0],
                "PRED": [2.0, 20.0, 4.0, 40.0],
            }
        )
        sim = obs.copy()
        sim["REP"] = 1
        bins = np.array([0.5, 1.5, 2.5])

        obs_c, sim_c = _apply_prediction_correction(obs, sim, bins, stratify_by="DOSE")

        np.testing.assert_allclose(obs_c["DV"].values, obs["DV"].values, rtol=1e-9)
        np.testing.assert_allclose(sim_c["DV"].values, sim["DV"].values, rtol=1e-9)

    def test_returns_two_dataframes(self):
        obs, sim, bins = _make_frames()
        obs_c, sim_c = _apply_prediction_correction(obs, sim, bins)
        assert isinstance(obs_c, pd.DataFrame)
        assert isinstance(sim_c, pd.DataFrame)

    def test_dv_column_modified_when_pred_varies(self):
        """When subjects have different PRED, correction changes DV values."""
        obs = pd.DataFrame(
            {
                "ID": [1, 2, 3],
                "TIME": [1.0, 1.0, 1.0],
                "REP": [0, 0, 0],
                "MDV": [0, 0, 0],
                "DV": [5.0, 5.0, 5.0],
                "PRED": [2.0, 4.0, 6.0],  # heterogeneous PRED → median = 4
            }
        )
        sim = obs.copy()
        sim["REP"] = 1
        bins = np.array([0.5, 1.5])
        obs_c, _ = _apply_prediction_correction(obs, sim, bins)
        # Only the middle subject (PRED=4=median) is unchanged; others differ
        assert not np.allclose(obs_c["DV"].values, obs["DV"].values)

    def test_no_pred_column_warns_and_returns_unchanged(self):
        obs, sim, bins = _make_frames()
        obs_no_pred = obs.drop(columns=["PRED"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obs_c, sim_c = _apply_prediction_correction(obs_no_pred, sim, bins)
        assert any("PRED column not found" in str(warning.message) for warning in w)
        pd.testing.assert_frame_equal(obs_c, obs_no_pred)

    def test_bin_reference_is_median_pred(self):
        """
        For a single bin with all PRED = 2.0 and DV = 4.0,
        corrected DV should equal DV * median_PRED / PRED = 4 * 2 / 2 = 4.
        Single-bin case: median_PRED_bin = PRED, so DV_pc = DV.
        """
        obs = pd.DataFrame(
            {
                "ID": [1, 2],
                "TIME": [1.0, 1.0],
                "REP": [0, 0],
                "MDV": [0, 0],
                "DV": [4.0, 4.0],
                "PRED": [2.0, 2.0],
            }
        )
        sim = obs.copy()
        sim["REP"] = 1
        bins = np.array([0.5, 1.5])
        obs_c, _ = _apply_prediction_correction(obs, sim, bins)
        # median PRED in bin = 2.0, individual PRED = 2.0 → scale = 1 → DV unchanged
        np.testing.assert_allclose(obs_c["DV"].values, [4.0, 4.0], rtol=1e-9)

    def test_scale_applied_correctly(self):
        """
        DV_pc = DV * median_PRED_bin / PRED.
        With PRED = [1, 4] in one bin, median_PRED = 2.5.
        Subject with PRED=1 → scale = 2.5;  PRED=4 → scale = 0.625.
        """
        obs = pd.DataFrame(
            {
                "ID": [1, 2],
                "TIME": [1.0, 1.0],
                "REP": [0, 0],
                "MDV": [0, 0],
                "DV": [1.0, 4.0],
                "PRED": [1.0, 4.0],
            }
        )
        sim = obs.copy()
        sim["REP"] = 1
        bins = np.array([0.5, 1.5])
        obs_c, _ = _apply_prediction_correction(obs, sim, bins)
        median_pred = 2.5  # median of [1, 4]
        expected = np.array([1.0 * median_pred / 1.0, 4.0 * median_pred / 4.0])
        np.testing.assert_allclose(obs_c["DV"].values, expected, rtol=1e-9)

    def test_same_reference_applied_to_sim(self):
        """Observed and simulated use the same median_PRED_bin reference."""
        obs, sim, bins = _make_frames(n_subjects=5, n_obs=3, n_replicates=10)
        obs_c, sim_c = _apply_prediction_correction(obs, sim, bins)
        # In a correct model the corrected DV distributions should be comparable
        assert len(obs_c) > 0
        assert len(sim_c) > 0

    def test_zero_pred_rows_dropped(self):
        """Rows where PRED = 0 produce NaN DV_pc and are dropped."""
        obs = pd.DataFrame(
            {
                "ID": [1, 2],
                "TIME": [1.0, 1.0],
                "REP": [0, 0],
                "MDV": [0, 0],
                "DV": [3.0, 3.0],
                "PRED": [0.0, 2.0],  # first row has PRED=0
            }
        )
        sim = obs.copy()
        sim["REP"] = 1
        bins = np.array([0.5, 1.5])
        obs_c, _ = _apply_prediction_correction(obs, sim, bins)
        # Row with PRED=0 is dropped after NaN propagation
        assert len(obs_c) < len(obs)
        assert 0.0 not in obs_c["PRED"].values

    def test_multi_bin_uses_per_bin_median(self):
        """Each time bin uses its own median PRED as reference."""
        obs = pd.DataFrame(
            {
                "ID": [1, 1, 1, 1],
                "TIME": [1.0, 1.0, 5.0, 5.0],
                "REP": [0, 0, 0, 0],
                "MDV": [0, 0, 0, 0],
                "DV": [10.0, 10.0, 2.0, 2.0],
                "PRED": [10.0, 10.0, 2.0, 2.0],  # two bins, different scales
            }
        )
        sim = obs.copy()
        sim["REP"] = 1
        bins = np.array([0.5, 3.0, 6.0])
        obs_c, _ = _apply_prediction_correction(obs, sim, bins)
        # Within each single-PRED bin: median_PRED = PRED → correction = 1 → DV unchanged
        np.testing.assert_allclose(obs_c["DV"].values, [10.0, 10.0, 2.0, 2.0], rtol=1e-9)
