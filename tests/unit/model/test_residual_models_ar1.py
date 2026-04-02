"""Tests for continuous-time AR(1) residual model fitting (RM1)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from openpkpd.model.residual_models import (
    AR1ResidualResult,
    _ct_ar1_negloglik,
    fit_ar1_residuals,
    fit_ar1_residuals_ct,
    fit_ar1_residuals_yw,
)


def _generate_ct_ar1_data(
    phi: float,
    sigma2: float,
    times_per_subject: list[float],
    n_subjects: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate synthetic continuous-time AR(1) data."""
    residuals_by_subject: list[np.ndarray] = []
    times_by_subject: list[np.ndarray] = []
    times = np.array(times_per_subject, dtype=float)
    for _ in range(n_subjects):
        n = len(times)
        resids = np.zeros(n)
        # Draw from the stationary distribution at t0
        resids[0] = rng.normal(0.0, np.sqrt(sigma2))
        for j in range(1, n):
            dt = times[j] - times[j - 1]
            rho_j = phi**dt
            noise_var = sigma2 * (1.0 - rho_j**2)
            resids[j] = rho_j * resids[j - 1] + rng.normal(0.0, np.sqrt(max(noise_var, 0.0)))
        residuals_by_subject.append(resids)
        times_by_subject.append(times.copy())
    return residuals_by_subject, times_by_subject


class TestCtAR1NegLogLik:
    """Unit tests for _ct_ar1_negloglik."""

    def test_phi_out_of_range_returns_large(self):
        resids = [np.array([0.1, -0.2, 0.3])]
        times = [np.array([0.0, 1.0, 2.0])]
        # log_phi = 0 => phi = 1.0 (boundary) should return 1e10
        val = _ct_ar1_negloglik(0.0, resids, times)
        assert val >= 1e10

    def test_negative_log_phi_returns_finite(self):
        resids = [np.array([0.1, -0.2, 0.3])]
        times = [np.array([0.0, 1.0, 2.0])]
        val = _ct_ar1_negloglik(-1.0, resids, times)
        assert np.isfinite(val)
        assert val > 0


class TestFitAR1ResidualsCT:
    """Tests for fit_ar1_residuals_ct."""

    def test_numerical_accuracy(self):
        """MLE estimate of phi should be within +-0.05 of 0.8, sigma2 within +-0.15 of 1.0."""
        rng = np.random.default_rng(42)
        phi_true = 0.8
        sigma2_true = 1.0
        times = [0.0, 1.0, 3.0, 7.0, 10.0]
        resids, times_list = _generate_ct_ar1_data(phi_true, sigma2_true, times, 50, rng)
        phi_hat, sigma2_hat = fit_ar1_residuals_ct(resids, times_list)
        assert abs(phi_hat - phi_true) < 0.05, f"phi_hat={phi_hat:.4f}, expected ~{phi_true}"
        assert abs(sigma2_hat - sigma2_true) < 0.15, (
            f"sigma2_hat={sigma2_hat:.4f}, expected ~{sigma2_true}"
        )

    def test_single_obs_subjects_skipped(self):
        """Subjects with a single observation should be skipped without error."""
        resids = [np.array([0.5]), np.array([0.1, -0.2, 0.3])]
        times = [np.array([0.0]), np.array([0.0, 1.0, 2.0])]
        phi_hat, sigma2_hat = fit_ar1_residuals_ct(resids, times)
        assert np.isfinite(phi_hat)
        assert 0.0 < phi_hat < 1.0
        assert np.isfinite(sigma2_hat)


class TestFitAR1Residuals:
    """Tests for fit_ar1_residuals (the public interface, now CT MLE)."""

    def _make_flat_arrays(
        self,
        residuals_by_subject: list[np.ndarray],
        times_by_subject: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten per-subject lists into the flat arrays expected by fit_ar1_residuals."""
        all_resids = []
        all_times = []
        all_ids = []
        for sid, (r, t) in enumerate(zip(residuals_by_subject, times_by_subject)):
            all_resids.append(r)
            all_times.append(t)
            all_ids.append(np.full(len(r), sid))
        return (
            np.concatenate(all_resids),
            np.concatenate(all_ids),
            np.concatenate(all_times),
        )

    def test_returns_ar1_result(self):
        rng = np.random.default_rng(7)
        times = [0.0, 1.0, 3.0, 7.0, 10.0]
        resids, times_list = _generate_ct_ar1_data(0.7, 0.5, times, 20, rng)
        flat_r, flat_ids, flat_t = self._make_flat_arrays(resids, times_list)
        result = fit_ar1_residuals(flat_r, flat_ids, flat_t)
        assert isinstance(result, AR1ResidualResult)
        assert 0.0 < result.rho < 1.0
        assert result.sigma2 > 0

    def test_equal_spacing_ct_recovers_phi(self):
        """With equal-spacing data, CT MLE should recover phi close to truth."""
        rng = np.random.default_rng(123)
        phi_true = 0.6
        # Equal spacing: [0, 1, 2, 3, 4]
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        resids, times_list = _generate_ct_ar1_data(phi_true, 0.8, times, 60, rng)
        flat_r, flat_ids, flat_t = self._make_flat_arrays(resids, times_list)

        result_ct = fit_ar1_residuals(flat_r, flat_ids, flat_t)

        # CT MLE should recover phi reasonably well even for equal spacing
        assert abs(result_ct.rho - phi_true) < 0.12, (
            f"CT phi={result_ct.rho:.4f}, expected close to {phi_true}"
        )

    def test_irregular_times_yw_biased(self):
        """Highly irregular times: YW gives biased phi, CT gives more correct phi.

        We verify the two methods differ by > 0.1 when time gaps are very irregular.
        """
        rng = np.random.default_rng(999)
        # Highly irregular: some 0.1-hr gaps, one 24-hr gap
        times = [0.0, 0.1, 0.2, 24.2]
        resids, times_list = _generate_ct_ar1_data(0.9, 1.0, times, 60, rng)
        flat_r, flat_ids, flat_t = self._make_flat_arrays(resids, times_list)

        result_ct = fit_ar1_residuals(flat_r, flat_ids, flat_t)

        from openpkpd.model.residual_models import _fit_ar1_residuals_yw_impl

        result_yw = _fit_ar1_residuals_yw_impl(flat_r, flat_ids, flat_t)

        diff = abs(result_ct.rho - result_yw.rho)
        assert diff > 0.1, (
            f"Expected CT and YW to diverge (diff > 0.1) for irregular times, "
            f"got CT phi={result_ct.rho:.4f}, YW rho={result_yw.rho:.4f}, diff={diff:.4f}"
        )

    def test_single_obs_subjects_skipped_gracefully(self):
        """Single-observation subjects should be skipped; function should not raise."""
        resids = np.array([0.5, 0.1, -0.2, 0.3])
        subject_ids = np.array([0, 1, 1, 1])
        times = np.array([0.0, 0.0, 1.0, 2.0])
        result = fit_ar1_residuals(resids, subject_ids, times)
        assert isinstance(result, AR1ResidualResult)
        # Should still run with the one valid subject
        assert np.isfinite(result.rho)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_ar1_residuals(np.array([1.0, 2.0]), np.array([1]), np.array([0.0, 1.0]))


class TestFitAR1ResidualsYW:
    """Tests that fit_ar1_residuals_yw still works and emits DeprecationWarning."""

    def test_emits_deprecation_warning(self):
        resids = np.array([0.1, -0.2, 0.3, 0.1, -0.1, 0.2])
        ids = np.array([0, 0, 0, 1, 1, 1])
        times = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fit_ar1_residuals_yw(resids, ids, times)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "deprecated" in str(deprecation_warnings[0].message).lower()
        assert isinstance(result, AR1ResidualResult)

    def test_still_returns_valid_result(self):
        rng = np.random.default_rng(55)
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        resids_list, times_list = _generate_ct_ar1_data(0.5, 1.0, times, 10, rng)
        all_r = np.concatenate(resids_list)
        all_ids = np.concatenate([np.full(len(r), i) for i, r in enumerate(resids_list)])
        all_t = np.concatenate(times_list)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = fit_ar1_residuals_yw(all_r, all_ids, all_t)
        assert isinstance(result, AR1ResidualResult)
        assert np.isfinite(result.rho)
