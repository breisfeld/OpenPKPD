"""
Unit tests for EstimationResult (estimation/base.py).

Covers: AIC/BIC/n_parameters, compute_n_parameters, compute_shrinkage,
        summary(), to_html(), _make_result helper.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationMethod, EstimationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    theta=None,
    omega=None,
    sigma=None,
    ofv=100.0,
    converged=True,
    n_obs=0,
    n_subjects=0,
    post_hoc_etas=None,
) -> EstimationResult:
    if theta is None:
        theta = np.array([1.0, 2.0, 3.0])
    if omega is None:
        omega = np.diag([0.1, 0.2])
    if sigma is None:
        sigma = np.diag([0.05])
    return EstimationResult(
        theta_final=theta,
        omega_final=omega,
        sigma_final=sigma,
        ofv=ofv,
        converged=converged,
        n_observations=n_obs,
        n_subjects=n_subjects,
        post_hoc_etas=post_hoc_etas or {},
    )


# ---------------------------------------------------------------------------
# n_parameters
# ---------------------------------------------------------------------------


class TestNParameters:
    def test_inferred_from_shapes(self):
        # 3 THETA + 2x2 lower tri (3 elements) + 1x1 lower tri (1 element)
        r = _make_result(
            theta=np.array([1.0, 2.0, 3.0]),
            omega=np.diag([0.1, 0.2]),
            sigma=np.diag([0.05]),
        )
        assert r.n_parameters == 3 + 3 + 1  # 7

    def test_explicit_n_parameters_overrides(self):
        r = _make_result()
        r._n_parameters = 42
        assert r.n_parameters == 42

    def test_scalar_omega_sigma(self):
        r = _make_result(
            theta=np.array([1.0]),
            omega=np.array([[0.1]]),
            sigma=np.array([[0.05]]),
        )
        assert r.n_parameters == 1 + 1 + 1  # 3

    def test_compute_n_parameters_from_specs(self):
        """compute_n_parameters respects fixed flags on spec objects."""

        class _Spec:
            def __init__(self, fixed=False):
                self.fixed = fixed

        r = _make_result(theta=np.array([1.0, 2.0, 3.0]))
        # 2 free theta, 1 fixed; omega 1 element (free); sigma 1 element (fixed)
        r.compute_n_parameters(
            theta_specs=[_Spec(False), _Spec(True), _Spec(False)],
            omega_specs=[_Spec(False)],
            sigma_specs=[_Spec(True)],
        )
        assert r.n_parameters == 2 + 1 + 0  # 3

    def test_compute_n_parameters_no_specs(self):
        """Without specs, falls back to matrix shapes."""
        r = _make_result(
            theta=np.array([1.0, 2.0]),
            omega=np.diag([0.1]),
            sigma=np.diag([0.05]),
        )
        r.compute_n_parameters()
        assert r.n_parameters == 2 + 1 + 1  # 4

    def test_compute_n_parameters_partial_specs(self):
        """Only theta_specs provided; omega/sigma inferred."""

        class _Spec:
            def __init__(self, fixed=False):
                self.fixed = fixed

        r = _make_result(
            theta=np.array([1.0, 2.0, 3.0]),
            omega=np.diag([0.1, 0.2]),
            sigma=np.diag([0.05]),
        )
        r.compute_n_parameters(theta_specs=[_Spec(True), _Spec(False), _Spec(False)])
        # 2 free THETA + 3 OMEGA elements + 1 SIGMA element
        assert r.n_parameters == 2 + 3 + 1  # 6


# ---------------------------------------------------------------------------
# AIC / BIC
# ---------------------------------------------------------------------------


class TestAicBic:
    def test_aic_formula(self):
        r = _make_result(ofv=200.0)
        # AIC = OFV + 2 * n_parameters
        assert r.aic == pytest.approx(200.0 + 2 * r.n_parameters)

    def test_bic_formula(self):
        r = _make_result(ofv=200.0, n_obs=100)
        # BIC = OFV + ln(n_obs) * n_parameters
        expected = 200.0 + math.log(100) * r.n_parameters
        assert r.bic == pytest.approx(expected)

    def test_bic_no_observations_is_inf(self):
        r = _make_result(ofv=100.0, n_obs=0)
        assert r.bic == float("inf")

    def test_aic_equals_ofv_plus_two_k(self):
        r = _make_result(ofv=50.0)
        assert r.aic - r.ofv == pytest.approx(2 * r.n_parameters)

    def test_aic_bic_with_explicit_n_parameters(self):
        r = _make_result(ofv=100.0, n_obs=50)
        r._n_parameters = 5
        assert r.aic == pytest.approx(100.0 + 10.0)
        assert r.bic == pytest.approx(100.0 + math.log(50) * 5)


# ---------------------------------------------------------------------------
# compute_shrinkage
# ---------------------------------------------------------------------------


class TestComputeShrinkage:
    def _result_with_etas(
        self, eta_matrix: np.ndarray, omega_diag: list[float]
    ) -> EstimationResult:
        omega_diag.__len__()
        omega = np.diag(omega_diag)
        sigma = np.diag([0.05])
        theta = np.array([1.0])
        post_hoc = {i + 1: eta_matrix[i] for i in range(len(eta_matrix))}
        return EstimationResult(
            theta_final=theta,
            omega_final=omega,
            sigma_final=sigma,
            ofv=100.0,
            post_hoc_etas=post_hoc,
        )

    def test_zero_shrinkage_when_sd_equals_omega_sd(self):
        # If sd(EBE) == sqrt(omega_kk), shrinkage = 0
        np.random.seed(42)
        omega_var = 0.25  # sd = 0.5
        etas = np.random.normal(0, 0.5, 30).reshape(30, 1)
        r = self._result_with_etas(etas, [omega_var])
        r.compute_shrinkage()
        assert len(r.eta_shrinkage) == 1
        # Shrinkage can be slightly off due to random sample but should be near 0
        assert abs(r.eta_shrinkage[0]) < 0.3

    def test_high_shrinkage_when_etas_near_zero(self):
        # All ETAs near zero → sd(EBE) ≈ 0 → shrinkage ≈ 1
        etas = np.zeros((20, 1)) + 1e-6 * np.random.randn(20, 1)
        r = self._result_with_etas(etas, [0.25])
        r.compute_shrinkage()
        assert r.eta_shrinkage[0] > 0.9

    def test_shrinkage_length_matches_n_eta(self):
        etas = np.random.randn(10, 2) * 0.1
        r = self._result_with_etas(etas, [0.1, 0.2])
        r.compute_shrinkage()
        assert len(r.eta_shrinkage) == 2

    def test_no_post_hoc_etas_returns_early(self):
        r = _make_result()
        r.compute_shrinkage()
        assert len(r.eta_shrinkage) == 0

    def test_eps_shrinkage_from_iwres(self):
        iwres = np.ones(50) * 0.8 + np.random.randn(50) * 0.05
        r = _make_result()
        r.post_hoc_etas = {1: np.array([0.0])}
        r.compute_shrinkage(iwres=iwres)
        # EPS shrinkage = 1 - sd(IWRES)
        assert len(r.eps_shrinkage) == 1
        expected = 1.0 - float(np.std(iwres, ddof=1))
        assert r.eps_shrinkage[0] == pytest.approx(expected, abs=1e-6)

    def test_eps_shrinkage_not_updated_when_iwres_none(self):
        r = _make_result()
        r.post_hoc_etas = {1: np.array([0.0])}
        r.eps_shrinkage = np.array([0.5])
        r.compute_shrinkage(iwres=None)
        # eps_shrinkage should remain as-is
        assert r.eps_shrinkage[0] == pytest.approx(0.5)

    def test_shrinkage_warning_emitted_above_30pct(self):
        etas = np.zeros((20, 1))  # 100% shrinkage
        r = self._result_with_etas(etas, [0.25])
        with pytest.warns(UserWarning, match="shrinkage"):
            r.compute_shrinkage()
        assert len(r.shrinkage_warnings) == 1
        assert ">30%" in r.shrinkage_warnings[0]

    def test_omega_zero_gives_zero_shrinkage(self):
        etas = np.random.randn(10, 1) * 0.3
        r = self._result_with_etas(etas, [0.0])
        r.compute_shrinkage()
        assert r.eta_shrinkage[0] == 0.0


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_contains_key_fields(self):
        r = _make_result(ofv=150.3, converged=True)
        s = r.summary()
        assert "OFV" in s
        assert "150.3" in s
        assert "AIC" in s
        assert "BIC" in s
        assert "Converged" in s

    def test_summary_includes_shrinkage(self):
        r = _make_result()
        r.eta_shrinkage = np.array([0.15, 0.25])
        s = r.summary()
        assert "shrinkage" in s.lower()

    def test_summary_includes_warnings(self):
        r = _make_result()
        r.warnings = ["Test warning message"]
        s = r.summary()
        assert "Test warning message" in s

    def test_summary_no_shrinkage_when_empty(self):
        r = _make_result()
        r.eta_shrinkage = np.array([])
        s = r.summary()
        assert "ETA shrinkage" not in s

    def test_summary_returns_string(self):
        r = _make_result()
        assert isinstance(r.summary(), str)


# ---------------------------------------------------------------------------
# to_html()
# ---------------------------------------------------------------------------


class TestToHtml:
    def test_to_html_writes_file(self):
        r = _make_result(ofv=200.0, n_obs=50)
        r.method = "FOCE"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            r.to_html(path)
            assert os.path.exists(path)
            content = Path(path).read_text()
            assert "<!DOCTYPE html>" in content
            assert "200." in content  # OFV

    def test_to_html_with_params(self):
        from openpkpd.model.parameters import ParameterSet, ThetaSpec

        r = _make_result(
            theta=np.array([1.5, 0.08, 30.0]),
            omega=np.diag([0.2, 0.1]),
            sigma=np.diag([0.05]),
            ofv=180.0,
            n_obs=100,
        )
        r.method = "FOCE"

        theta_specs = [
            ThetaSpec(init=1.5, label="KA"),
            ThetaSpec(init=0.08, label="CL"),
            ThetaSpec(init=30.0, label="V"),
        ]
        params = ParameterSet.from_specs(theta_specs, [], [])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            r.to_html(path, params=params)
            content = Path(path).read_text()
            assert "KA" in content
            assert "CL" in content
            assert "V" in content

    def test_to_html_without_params_creates_minimal(self):
        r = _make_result(ofv=100.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "minimal.html")
            r.to_html(path)
            assert os.path.exists(path)
            content = Path(path).read_text()
            assert "THETA" in content


# ---------------------------------------------------------------------------
# EstimationResult dataclass defaults
# ---------------------------------------------------------------------------


class TestEstimationResultDefaults:
    def test_default_eta_shrinkage_is_empty_array(self):
        r = EstimationResult(
            theta_final=np.array([1.0]),
            omega_final=np.diag([0.1]),
            sigma_final=np.diag([0.05]),
            ofv=100.0,
        )
        assert len(r.eta_shrinkage) == 0

    def test_default_converged_is_false(self):
        r = EstimationResult(
            theta_final=np.array([1.0]),
            omega_final=np.diag([0.1]),
            sigma_final=np.diag([0.05]),
            ofv=100.0,
        )
        assert r.converged is False

    def test_warnings_default_empty_list(self):
        r = EstimationResult(
            theta_final=np.array([1.0]),
            omega_final=np.diag([0.1]),
            sigma_final=np.diag([0.05]),
            ofv=100.0,
        )
        assert r.warnings == []

    def test_post_hoc_etas_default_empty_dict(self):
        r = EstimationResult(
            theta_final=np.array([1.0]),
            omega_final=np.diag([0.1]),
            sigma_final=np.diag([0.05]),
            ofv=100.0,
        )
        assert r.post_hoc_etas == {}

    def test_mutable_defaults_not_shared(self):
        r1 = EstimationResult(
            theta_final=np.array([1.0]),
            omega_final=np.diag([0.1]),
            sigma_final=np.diag([0.05]),
            ofv=1.0,
        )
        r2 = EstimationResult(
            theta_final=np.array([1.0]),
            omega_final=np.diag([0.1]),
            sigma_final=np.diag([0.05]),
            ofv=2.0,
        )
        r1.warnings.append("test")
        assert r2.warnings == []


# ---------------------------------------------------------------------------
# EstimationMethod._make_result
# ---------------------------------------------------------------------------


class TestEstimationMethodMakeResult:
    def test_make_result_constructs_correctly(self):
        """_make_result builds an EstimationResult with correct fields."""

        class _DummyMethod(EstimationMethod):
            method_name = "TEST"

            def estimate(self, population_model, init_params, **kwargs):
                pass

        method = _DummyMethod()
        theta = np.array([1.0, 2.0])
        omega = np.diag([0.1])
        sigma = np.diag([0.05])
        r = method._make_result(theta, omega, sigma, ofv=99.9, converged=True)

        assert r.method == "TEST"
        assert r.ofv == pytest.approx(99.9)
        assert r.converged is True
        np.testing.assert_array_equal(r.theta_final, theta)
        np.testing.assert_array_equal(r.omega_final, omega)
        np.testing.assert_array_equal(r.sigma_final, sigma)
