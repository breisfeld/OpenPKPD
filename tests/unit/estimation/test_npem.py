"""
Tests for NPEMMethod (estimation/nonparametric.py) — full NPEM with
joint support-point location optimisation.

Uses a lightweight mock population model so tests run in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.nonparametric import NonparametricResult, NPEMMethod

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _SubjectEvents:
    def __init__(self, dv):
        self._dv = np.asarray(dv)

    def observation_mask(self):
        return np.ones(len(self._dv), dtype=bool)

    @property
    def obs_dv(self):
        return self._dv


class _IndividualModel:
    def __init__(self, sid, dv):
        self.subject_events = _SubjectEvents(dv)
        self._sid = sid

    def evaluate(self, theta, eta, sigma, trans=None):
        # Simple: prediction = theta[0] * exp(eta[0])
        import numpy as np

        f = np.full(len(self.subject_events.obs_dv), float(theta[0]) * np.exp(float(eta[0])))
        return None, None, f

    def log_likelihood(self, theta, eta, sigma, trans=None):
        return 0.0

    def obj_eta(self, eta, theta, omega, sigma, trans=None):
        return 0.0


class _MockPopulationModel:
    def __init__(self, n_subjects=5):
        self._n = n_subjects
        self._indivs = {
            i: _IndividualModel(i, dv=np.array([1.0, 0.8, 0.6])) for i in range(1, n_subjects + 1)
        }
        self.trans = 1

    def subject_ids(self):
        return list(self._indivs.keys())

    def individual_model(self, sid):
        return self._indivs[sid]

    def ofv_fo(self, params):
        return float(np.sum((params.theta - 1.0) ** 2))

    def post_hoc_etas(self, params=None):
        return {sid: np.zeros(1) for sid in self.subject_ids()}


class _MockParams:
    def __init__(self):
        self.theta = np.array([1.0])
        self.omega = np.eye(1) * 0.1
        self.sigma = np.eye(1) * 0.05
        self.theta_specs = []
        self.omega_specs = []
        self.sigma_specs = []

    def n_eta(self):
        return 1


class _MockBaseResult:
    def __init__(self, n_subjects):
        self.theta_final = np.array([1.0])
        self.omega_final = np.eye(1) * 0.1
        self.sigma_final = np.eye(1) * 0.05
        self.ofv = 100.0
        self.converged = True
        self.post_hoc_etas = {i: np.array([0.01 * i]) for i in range(1, n_subjects + 1)}
        self.ofv_history = [100.0]
        self.n_function_evals = 10
        self.elapsed_time = 0.1
        self.method = "FOCE"
        self.message = "OK"
        self.warnings = []


# ---------------------------------------------------------------------------
# NPEMMethod tests (patched estimate so we don't need a real FOCE)
# ---------------------------------------------------------------------------


class TestNPEMMethod:
    @pytest.fixture()
    def method(self):
        return NPEMMethod(
            max_iter=5,
            max_location_iter=5,
            optimise_locations=True,
            n_support_points=4,
        )

    @pytest.fixture()
    def method_no_location(self):
        return NPEMMethod(
            max_iter=5,
            optimise_locations=False,
            n_support_points=4,
        )

    def test_method_name(self, method):
        assert method.method_name == "NPEM"

    def test_optimise_locations_flag(self, method, method_no_location):
        assert method.optimise_locations is True
        assert method_no_location.optimise_locations is False

    def _run_with_mock(self, method, n_subjects=5):
        """Run NPEM estimate with mocked base method."""
        pop_model = _MockPopulationModel(n_subjects)
        params = _MockParams()
        mock_result = _MockBaseResult(n_subjects)

        import unittest.mock as mock

        with mock.patch("openpkpd.estimation.get_estimation_method") as mock_get:
            mock_est = mock.MagicMock()
            mock_est.estimate.return_value = mock_result
            mock_get.return_value = mock_est
            return method.estimate(pop_model, params)

    def test_returns_nonparametric_result(self, method):
        result = self._run_with_mock(method)
        assert isinstance(result, NonparametricResult)

    def test_support_points_shape(self, method):
        result = self._run_with_mock(method)
        K = result.support_points.shape[0]
        assert K == 4  # n_support_points
        assert result.support_points.shape[1] == 1  # n_eta

    def test_weights_sum_to_one(self, method):
        result = self._run_with_mock(method)
        assert abs(result.support_weights.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, method):
        result = self._run_with_mock(method)
        assert np.all(result.support_weights >= 0)

    def test_omega_final_shape(self, method):
        result = self._run_with_mock(method)
        assert result.omega_final.shape == (1, 1)

    def test_method_label(self, method):
        result = self._run_with_mock(method)
        assert result.method == "NPEM"

    def test_no_location_opt_produces_result(self, method_no_location):
        result = self._run_with_mock(method_no_location)
        assert isinstance(result, NonparametricResult)
        assert abs(result.support_weights.sum() - 1.0) < 1e-6

    def test_summary_includes_support_info(self, method):
        result = self._run_with_mock(method)
        summary = result.summary()
        assert "NPEM" in summary or "support" in summary.lower()

    def test_location_optimization_moves_support_point_to_likelihood_optimum(self):
        pop_model = _MockPopulationModel(n_subjects=1)
        pop_model._indivs[1] = _IndividualModel(1, dv=np.array([2.0, 2.0, 2.0]))
        params = _MockParams()
        mock_result = _MockBaseResult(1)
        mock_result.post_hoc_etas = {1: np.array([0.0])}

        method = NPEMMethod(
            max_iter=5,
            max_location_iter=50,
            optimise_locations=True,
            n_support_points=1,
        )

        import unittest.mock as mock

        with mock.patch("openpkpd.estimation.get_estimation_method") as mock_get:
            mock_est = mock.MagicMock()
            mock_est.estimate.return_value = mock_result
            mock_get.return_value = mock_est
            result = method.estimate(pop_model, params)

        assert result.support_points.shape == (1, 1)
        assert result.support_points[0, 0] == pytest.approx(np.log(2.0), abs=1e-2)
