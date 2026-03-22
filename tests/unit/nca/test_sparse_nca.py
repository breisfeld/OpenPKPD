"""
Tests for SparseNCAEngine.

Uses a simple 1-compartment IV bolus model as the population model.
Sparse samples (3 points) are drawn from the true profile; the engine
should recover AUC_inf within 10% of the analytical truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.nca.sparse import SparseNCAEngine

# ---------------------------------------------------------------------------
# Minimal population model stub (1-cmt IV bolus: CL=5 L/h, V=50 L)
# ---------------------------------------------------------------------------


class _Simple1CmtModel:
    """Stub that implements the fallback API used by SparseNCAEngine."""

    CL = 5.0  # L/h
    V = 50.0  # L

    def __init__(self) -> None:
        self.theta = np.array([self.CL, self.V])
        self.omega = np.diag([0.09, 0.09])  # 30% CV on CL and V
        self.sigma = np.diag([0.01])

    # Intentionally do NOT implement get_individual_model() so that
    # SparseNCAEngine falls back to the built-in 1-cmt analytical solution.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _true_profile(times: np.ndarray, dose: float = 100.0) -> np.ndarray:
    """Analytical 1-cmt IV bolus: C(t) = Dose/V * exp(-CL/V * t)."""
    CL, V = _Simple1CmtModel.CL, _Simple1CmtModel.V
    return dose / V * np.exp(-CL / V * np.array(times, dtype=float))


def _auc_inf_analytical(dose: float = 100.0) -> float:
    """AUC_inf = Dose / CL for 1-cmt IV model."""
    return dose / _Simple1CmtModel.CL


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSparseNCAEngine:
    @pytest.fixture()
    def engine(self):
        return SparseNCAEngine(_Simple1CmtModel())

    @pytest.fixture()
    def sparse_data(self):
        """Three sparse samples from the true 1-cmt profile."""
        times = np.array([1.0, 4.0, 12.0])
        conc = _true_profile(times, dose=100.0)
        return times, conc

    def test_compute_subject_returns_nca_params(self, engine, sparse_data):
        from openpkpd.nca.nca import NCAParameters

        times, conc = sparse_data
        result = engine.compute_subject(
            subject_id=1, sparse_times=times, sparse_conc=conc, dose=100.0, route="IV"
        )
        assert isinstance(result, NCAParameters)

    def test_auc_within_10pct(self, engine, sparse_data):
        """AUC_inf from sparse NCA should be within 10% of analytical truth."""
        times, conc = sparse_data
        result = engine.compute_subject(
            subject_id=1, sparse_times=times, sparse_conc=conc, dose=100.0, route="IV"
        )
        truth = _auc_inf_analytical(dose=100.0)
        # Use auc_last or auc_inf, whichever is finite
        auc = result.auc_inf if np.isfinite(result.auc_inf) else result.auc_last
        assert np.isfinite(auc), "AUC estimate must be finite"
        rel_err = abs(auc - truth) / truth
        assert rel_err < 0.10, f"AUC relative error {rel_err:.1%} exceeds 10%"

    def test_compute_dataset(self, engine):
        """compute_dataset returns one row per subject."""
        import pandas as pd

        times = [1.0, 4.0, 12.0] * 3
        subjs = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        concs = list(_true_profile(np.array([1.0, 4.0, 12.0]))) * 3

        df = pd.DataFrame({"ID": subjs, "TIME": times, "DV": concs})
        results = engine.compute_dataset(df, dose=100.0, route="IV")
        assert len(results) == 3
        assert "auc_last" in results.columns or "auc_inf" in results.columns

    def test_subject_id_preserved(self, engine, sparse_data):
        times, conc = sparse_data
        result = engine.compute_subject(
            subject_id="SUBJ_X", sparse_times=times, sparse_conc=conc, dose=100.0
        )
        assert result.subject_id == "SUBJ_X"

    def test_custom_dense_times(self, sparse_data):
        custom_t = np.linspace(0, 24, 50)
        engine = SparseNCAEngine(_Simple1CmtModel(), dense_times=custom_t)
        times, conc = sparse_data
        result = engine.compute_subject(1, times, conc, dose=100.0)
        assert np.isfinite(result.cmax)

    def test_predict_matches_analytical_fallback(self, engine):
        """Fallback 1-cmt analytical predictor should match the closed form."""
        times = np.array([0.0, 1.0, 4.0, 12.0])
        eta = np.array([np.log(1.5), np.log(0.8)])
        pred = engine._predict(eta, times, dose=100.0)

        cl = _Simple1CmtModel.CL * np.exp(eta[0])
        v = _Simple1CmtModel.V * np.exp(eta[1])
        expected = 100.0 / v * np.exp(-(cl / v) * times)
        np.testing.assert_allclose(pred, expected, rtol=1e-12, atol=1e-12)

    def test_compute_subject_preserves_exact_dose_scaling_under_analytical_fallback(self, engine):
        times = np.array([1.0, 4.0, 12.0])
        conc_100 = _true_profile(times, dose=100.0)
        conc_200 = _true_profile(times, dose=200.0)

        result_100 = engine.compute_subject(1, times, conc_100, dose=100.0, route="IV")
        result_200 = engine.compute_subject(1, times, conc_200, dose=200.0, route="IV")

        auc_100 = result_100.auc_inf if np.isfinite(result_100.auc_inf) else result_100.auc_last
        auc_200 = result_200.auc_inf if np.isfinite(result_200.auc_inf) else result_200.auc_last
        assert auc_200 / auc_100 == pytest.approx(2.0, rel=1e-3)
        assert result_200.cmax / result_100.cmax == pytest.approx(2.0, rel=1e-3)
