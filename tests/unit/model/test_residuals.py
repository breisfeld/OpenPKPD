"""Direct tests for residual helper numerics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openpkpd.model.residuals import (
    compute_cwres,
    compute_iwres,
    compute_residual_variance,
    compute_wres,
    log_likelihood_normal,
)
from openpkpd.utils.constants import LOG2PI


@pytest.mark.unit
class TestLogLikelihoodNormal:
    def test_matches_closed_form_value(self):
        y = 2.5
        mu = 2.0
        sigma2 = 4.0

        expected = -0.5 * (LOG2PI + math.log(sigma2) + (y - mu) ** 2 / sigma2)

        assert log_likelihood_normal(y, mu, sigma2) == pytest.approx(expected)

    def test_nonpositive_variance_returns_large_negative_sentinel(self):
        assert log_likelihood_normal(1.0, 1.0, 0.0) == pytest.approx(-1e30)
        assert log_likelihood_normal(1.0, 1.0, -0.5) == pytest.approx(-1e30)


@pytest.mark.unit
class TestComputeResidualVariance:
    def test_additive_variance_uses_first_diagonal_entry(self):
        sigma = np.array([[0.25]])
        assert compute_residual_variance(3.0, sigma, error_type="additive") == pytest.approx(0.25)

    def test_proportional_variance_scales_with_f_squared(self):
        sigma = np.array([[0.5]])
        assert compute_residual_variance(4.0, sigma, error_type="proportional") == pytest.approx(
            8.0
        )

    def test_combined_variance_uses_two_diagonal_entries_when_available(self):
        sigma = np.diag([0.25, 0.5])
        assert compute_residual_variance(4.0, sigma, error_type="combined") == pytest.approx(8.25)

    def test_combined_variance_falls_back_to_single_entry_when_needed(self):
        sigma = np.array([[0.25]])
        assert compute_residual_variance(4.0, sigma, error_type="combined") == pytest.approx(4.25)

    def test_unknown_error_type_raises(self):
        with pytest.raises(ValueError, match="Unknown error_type"):
            compute_residual_variance(1.0, np.array([[1.0]]), error_type="mystery")


@pytest.mark.unit
class TestComputeWRES:
    def test_whitens_residuals_with_cholesky_factor(self):
        dv = np.array([3.0, 5.0])
        pred = np.array([1.0, 2.0])
        c_i = np.array([[4.0, 0.0], [0.0, 9.0]])

        result = compute_wres(dv, pred, c_i)

        np.testing.assert_allclose(result, np.array([1.0, 1.0]), rtol=1e-12, atol=1e-12)

    def test_falls_back_to_diagonal_scaling_when_cholesky_fails(self):
        dv = np.array([5.0, 7.0])
        pred = np.array([1.0, 4.0])
        c_i = np.array([[4.0, 5.0], [5.0, 4.0]])

        result = compute_wres(dv, pred, c_i)

        expected = (dv - pred) / np.sqrt(np.diag(c_i))
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
class TestComputeIWRES:
    def test_zero_or_negative_weights_return_zero(self):
        dv = np.array([5.0, 6.0, 7.0])
        ipred = np.array([4.0, 5.0, 8.0])
        w = np.array([2.0, 0.0, -1.0])

        result = compute_iwres(dv, ipred, w)

        np.testing.assert_allclose(result, np.array([0.5, 0.0, 0.0]), rtol=1e-12, atol=1e-12)


@pytest.mark.unit
class TestComputeCWRES:
    def test_matches_documented_approximation_when_iwres_is_nonzero(self):
        dv = np.array([11.0, 13.0])
        pred = np.array([10.0, 10.0])
        ipred = np.array([12.0, 11.0])
        wres = np.array([0.5, -0.25])
        iwres = np.array([-1.0, 2.0])

        result = compute_cwres(dv, pred, ipred, wres, iwres)

        ires = dv - ipred
        sd_est = np.abs(ires) / np.abs(iwres)
        expected = wres + (ipred - pred) / sd_est
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_uses_unit_sd_fallback_when_iwres_is_zero(self):
        dv = np.array([10.0])
        pred = np.array([8.0])
        ipred = np.array([9.0])
        wres = np.array([0.25])
        iwres = np.array([0.0])

        result = compute_cwres(dv, pred, ipred, wres, iwres)

        np.testing.assert_allclose(result, np.array([1.25]), rtol=1e-12, atol=1e-12)
