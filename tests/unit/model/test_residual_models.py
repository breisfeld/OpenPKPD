"""Tests for residual error model utilities."""

import numpy as np
import pytest

from openpkpd.model.residual_models import (
    AR1ResidualResult,
    ResidualModelType,
    fit_ar1_residuals,
    log_normal_log_likelihood,
    power_residual_variance,
)


class TestResidualModelType:
    def test_enum_values(self):
        assert ResidualModelType.ADDITIVE == "additive"
        assert ResidualModelType.PROPORTIONAL == "proportional"
        assert ResidualModelType.COMBINED == "combined"
        assert ResidualModelType.LOG_NORMAL == "log_normal"
        assert ResidualModelType.POWER == "power"


class TestFitAR1Residuals:
    def setup_method(self):
        """Create synthetic AR(1) residuals."""
        rng = np.random.default_rng(42)
        self.rho_true = 0.5
        self.sigma2_true = 1.0
        n_subj = 10
        n_per_subj = 10

        residuals = []
        subject_ids = []
        times = []

        for i in range(n_subj):
            r = [rng.normal(0, 1)]
            for _ in range(n_per_subj - 1):
                innov = rng.normal(0, np.sqrt(self.sigma2_true * (1 - self.rho_true**2)))
                r.append(self.rho_true * r[-1] + innov)
            residuals.extend(r)
            subject_ids.extend([i] * n_per_subj)
            times.extend(range(n_per_subj))

        self.residuals = np.array(residuals)
        self.subject_ids = np.array(subject_ids)
        self.times = np.array(times, dtype=float)

    def test_returns_ar1_result(self):
        result = fit_ar1_residuals(self.residuals, self.subject_ids, self.times)
        assert isinstance(result, AR1ResidualResult)

    def test_rho_in_range(self):
        result = fit_ar1_residuals(self.residuals, self.subject_ids, self.times)
        assert -1.0 < result.rho < 1.0

    def test_sigma2_positive(self):
        result = fit_ar1_residuals(self.residuals, self.subject_ids, self.times)
        assert result.sigma2 > 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            fit_ar1_residuals(
                np.array([1.0, 2.0]),
                np.array([1]),  # wrong length
                np.array([0.0, 1.0]),
            )

    def test_n_subjects_count(self):
        result = fit_ar1_residuals(self.residuals, self.subject_ids, self.times)
        assert result.n_subjects == 10

    def test_known_zero_autocorrelation(self):
        """Independent residuals → rho close to 0."""
        rng = np.random.default_rng(99)
        n = 5
        n_subj = 20
        r = rng.normal(0, 1, n * n_subj)
        s = np.repeat(np.arange(n_subj), n)
        t = np.tile(np.arange(n, dtype=float), n_subj)
        result = fit_ar1_residuals(r, s, t)
        assert abs(result.rho) < 0.5  # Should be near 0


class TestLogNormalLogLikelihood:
    def test_positive_likelihood(self):
        ell = log_normal_log_likelihood(5.0, 5.0, 0.1)
        assert np.isfinite(ell)

    def test_invalid_inputs(self):
        assert log_normal_log_likelihood(0.0, 5.0, 0.1) == float("-inf")
        assert log_normal_log_likelihood(5.0, 0.0, 0.1) == float("-inf")
        assert log_normal_log_likelihood(5.0, 5.0, 0.0) == float("-inf")

    def test_mode_at_true_value(self):
        """Likelihood should be higher when prediction matches observation."""
        ell_exact = log_normal_log_likelihood(5.0, 5.0, 0.5)
        ell_far = log_normal_log_likelihood(5.0, 15.0, 0.5)
        assert ell_exact > ell_far

    def test_known_value(self):
        """Test against manually computed value."""
        import math

        y, mu, s2 = 1.0, 1.0, 1.0
        expected = -0.5 * (math.log(2 * math.pi) + math.log(s2) + 0.0) - math.log(y)
        result = log_normal_log_likelihood(y, mu, s2)
        assert abs(result - expected) < 1e-10


class TestPowerResidualVariance:
    def test_sigma1_theta0(self):
        """theta=0 → var = sigma^2 regardless of f."""
        var = power_residual_variance(f=5.0, sigma=2.0, theta=0.0)
        assert var == pytest.approx(4.0)  # 2^2 * 5^0 = 4

    def test_theta_half(self):
        """theta=0.5 → var = sigma^2 * f."""
        var = power_residual_variance(f=4.0, sigma=1.0, theta=0.5)
        assert var == pytest.approx(4.0)  # 1^2 * 4^1 = 4

    def test_theta_one(self):
        """theta=1 → proportional: var = sigma^2 * f^2."""
        var = power_residual_variance(f=3.0, sigma=1.0, theta=1.0)
        assert var == pytest.approx(9.0)

    def test_zero_f_positive_theta(self):
        """f=0 with positive theta → returns a small but continuous positive value.

        Before the M-04 fix, the function snapped to np.finfo(float).tiny at
        exactly f=0, creating a discontinuity.  After the fix it uses a smooth
        floor of _POWER_VAR_FLOOR = 1e-6, so var = sigma^2 * floor^(2*theta).
        """
        from openpkpd.model.residual_models import _POWER_VAR_FLOOR

        var = power_residual_variance(f=0.0, sigma=1.0, theta=1.0)
        assert var > 0.0
        # Smooth floor: sigma^2 * floor^(2*1) = 1.0 * 1e-12
        expected = 1.0 * _POWER_VAR_FLOOR ** (2 * 1.0)
        assert var == pytest.approx(expected, rel=1e-9)
