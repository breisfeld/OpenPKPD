"""Tests for urinary excretion NCA."""

import numpy as np
import pytest

from openpkpd.nca import NCAParameters
from openpkpd.nca.urine import UrineNCAEngine


class TestUrineNCAEngine:
    def test_basic_computation(self):
        """Basic urinary NCA computation."""
        engine = UrineNCAEngine()
        # 4 collection intervals: 0-4h, 4-8h, 8-12h, 12-24h
        collection_times = np.array([0, 4, 8, 12, 24], dtype=float)
        delta_amounts = np.array([20.0, 15.0, 10.0, 5.0], dtype=float)

        params = engine.compute_subject(
            subject_id=1,
            dose=100.0,
            collection_times=collection_times,
            delta_amounts=delta_amounts,
        )

        assert params.ae_last == pytest.approx(50.0)
        assert params.ae_inf == pytest.approx(50.0)  # no extrapolation without plasma
        assert params.fe == pytest.approx(0.5)
        assert len(params.intervals) == 4

    def test_interval_rates(self):
        """Rate midpoints computed correctly."""
        engine = UrineNCAEngine()
        collection_times = np.array([0, 4, 8], dtype=float)
        delta_amounts = np.array([20.0, 10.0], dtype=float)

        params = engine.compute_subject(1, 100.0, collection_times, delta_amounts)

        assert params.intervals[0]["rate_mid"] == pytest.approx(20.0 / 4.0)
        assert params.intervals[1]["rate_mid"] == pytest.approx(10.0 / 4.0)

    def test_renal_clearance_with_plasma(self):
        """Renal clearance computed when plasma AUC_inf provided."""
        engine = UrineNCAEngine()
        collection_times = np.array([0, 4, 8, 12, 24], dtype=float)
        delta_amounts = np.array([20.0, 15.0, 10.0, 5.0], dtype=float)

        # Create a minimal NCAParameters with auc_inf and lambda_z
        plasma_nca = NCAParameters(subject_id=1, dose=100.0)
        plasma_nca.auc_inf = 200.0
        plasma_nca.lambda_z = 0.2

        params = engine.compute_subject(
            subject_id=1,
            dose=100.0,
            collection_times=collection_times,
            delta_amounts=delta_amounts,
            plasma_nca=plasma_nca,
        )

        # cl_renal = Ae_inf / AUC_inf
        assert np.isfinite(params.cl_renal)
        assert params.cl_renal > 0

    def test_wrong_collection_times_length(self):
        """Raises ValueError if collection_times is wrong length."""
        engine = UrineNCAEngine()
        with pytest.raises(ValueError):
            engine.compute_subject(
                1,
                100.0,
                collection_times=np.array([0, 4, 8]),  # needs n+1=3 but delta has 3
                delta_amounts=np.array([20.0, 15.0, 10.0]),
            )

    def test_to_dict(self):
        """to_dict returns expected keys."""
        engine = UrineNCAEngine()
        collection_times = np.array([0, 4, 8], dtype=float)
        delta_amounts = np.array([20.0, 10.0], dtype=float)
        params = engine.compute_subject(1, 100.0, collection_times, delta_amounts)
        d = params.to_dict()
        assert "subject_id" in d
        assert "ae_last" in d
        assert "fe" in d
        assert "cl_renal" in d

    def test_exact_extrapolation_and_renal_clearance_formula(self):
        """Ae_inf and CLrenal should follow the closed-form urine equations."""
        engine = UrineNCAEngine()
        collection_times = np.array([0.0, 2.0, 4.0])
        delta_amounts = np.array([8.0, 4.0])

        plasma_nca = NCAParameters(subject_id=1, dose=100.0)
        plasma_nca.lambda_z = 0.25
        plasma_nca.auc_inf = 100.0

        params = engine.compute_subject(
            subject_id=1,
            dose=100.0,
            collection_times=collection_times,
            delta_amounts=delta_amounts,
            plasma_nca=plasma_nca,
        )

        expected_ae_last = 12.0
        expected_rate_last = 4.0 / 2.0
        expected_ae_inf = expected_ae_last + expected_rate_last / plasma_nca.lambda_z
        expected_cl_renal = expected_ae_inf / plasma_nca.auc_inf

        assert params.ae_last == pytest.approx(expected_ae_last)
        assert params.ae_inf == pytest.approx(expected_ae_inf)
        assert params.fe == pytest.approx(expected_ae_inf / 100.0)
        assert params.cl_renal == pytest.approx(expected_cl_renal)

    @pytest.mark.parametrize("lambda_z", [0.0, -0.25])
    def test_nonpositive_lambda_z_falls_back_to_ae_last_without_nan(self, lambda_z):
        engine = UrineNCAEngine()
        collection_times = np.array([0.0, 2.0, 4.0])
        delta_amounts = np.array([8.0, 4.0])

        plasma_nca = NCAParameters(subject_id=1, dose=100.0)
        plasma_nca.lambda_z = lambda_z
        plasma_nca.auc_inf = 100.0

        params = engine.compute_subject(
            subject_id=1,
            dose=100.0,
            collection_times=collection_times,
            delta_amounts=delta_amounts,
            plasma_nca=plasma_nca,
        )

        assert params.ae_last == pytest.approx(12.0)
        assert params.ae_inf == pytest.approx(12.0)
        assert params.fe == pytest.approx(0.12)
        assert params.cl_renal == pytest.approx(0.12)
