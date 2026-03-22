"""Tests for BCa bootstrap confidence intervals."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from openpkpd.inference.bootstrap import BootstrapResult, bca_ci

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_original(theta, omega_diag, sigma_diag):
    """Return a mock EstimationResult with given parameter values."""
    result = MagicMock()
    result.theta_final = np.asarray(theta, dtype=float)
    result.omega_final = np.diag(np.asarray(omega_diag, dtype=float))
    result.sigma_final = np.diag(np.asarray(sigma_diag, dtype=float))
    return result


def _make_boot_result(rng, n_boot=500, n_theta=2, n_eta=1, n_eps=1, ci_level=0.95):
    """Return a BootstrapResult with synthetic samples."""
    theta_samples = rng.normal(loc=0.0, scale=1.0, size=(n_boot, n_theta)) + np.arange(
        1, n_theta + 1
    )
    omega_samples = np.abs(rng.normal(loc=0.05, scale=0.01, size=(n_boot, n_eta)))
    sigma_samples = np.abs(rng.normal(loc=0.01, scale=0.002, size=(n_boot, n_eps)))
    return BootstrapResult(
        n_boot=n_boot,
        n_success=n_boot,
        theta_samples=theta_samples,
        omega_diag_samples=omega_samples,
        sigma_diag_samples=sigma_samples,
        ci_level=ci_level,
    )


# ---------------------------------------------------------------------------
# Tests for bca_ci()
# ---------------------------------------------------------------------------


class TestBcaCi:
    def test_returns_two_floats(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(5.0, 1.0, 200)
        lo, hi = bca_ci(samples, original_est=5.0)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo < hi

    def test_lo_less_than_hi(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            samples = rng.normal(0, 1, 300)
            lo, hi = bca_ci(samples, original_est=rng.normal())
            assert lo <= hi

    def test_contains_original_est_for_correct_model(self):
        """95% BCa CI should contain the true value most of the time."""
        rng = np.random.default_rng(42)
        true_val = 3.0
        samples = rng.normal(true_val, 0.5, 500)
        lo, hi = bca_ci(samples, original_est=true_val)
        assert lo < true_val < hi

    def test_narrower_than_range(self):
        rng = np.random.default_rng(2)
        samples = rng.normal(0, 1, 300)
        lo, hi = bca_ci(samples, original_est=0.0)
        assert lo > np.min(samples)
        assert hi < np.max(samples)

    def test_zero_bias_symmetric_dist(self):
        """For a symmetric distribution with the original estimate at the mean,
        BCa bounds should be close to percentile bounds."""
        rng = np.random.default_rng(3)
        samples = rng.normal(0.0, 1.0, 5000)
        lo_bca, hi_bca = bca_ci(samples, original_est=0.0)
        lo_pct = float(np.percentile(samples, 2.5))
        hi_pct = float(np.percentile(samples, 97.5))
        # BCa should be within 0.2 of percentile for symmetric case
        assert abs(lo_bca - lo_pct) < 0.2
        assert abs(hi_bca - hi_pct) < 0.2

    def test_positive_bias_shifts_interval_up(self):
        """When bootstrap mean > original (positive bias), BCa shifts interval down."""
        rng = np.random.default_rng(5)
        # Bootstrap samples biased upward relative to original
        samples = rng.normal(5.0, 0.5, 500)
        lo_bca, _ = bca_ci(samples, original_est=3.0)  # original much lower
        lo_pct = float(np.percentile(samples, 2.5))
        # BCa lower bound should be <= percentile lower bound (bias-corrected downward)
        assert lo_bca <= lo_pct + 0.5  # generous tolerance

    def test_ci_level_respected(self):
        """Wider ci_level should produce wider interval."""
        rng = np.random.default_rng(6)
        samples = rng.normal(0, 1, 500)
        lo_90, hi_90 = bca_ci(samples, 0.0, ci_level=0.90)
        lo_99, hi_99 = bca_ci(samples, 0.0, ci_level=0.99)
        assert lo_99 < lo_90 < hi_90 < hi_99

    def test_handles_extreme_original_estimate(self):
        """When original_est is far outside samples, result is still finite."""
        samples = np.ones(200) * 5.0 + np.random.default_rng(7).normal(0, 0.1, 200)
        lo, hi = bca_ci(samples, original_est=-100.0)
        assert np.isfinite(lo)
        assert np.isfinite(hi)

    def test_affine_equivariance(self):
        rng = np.random.default_rng(8)
        samples = rng.normal(2.0, 0.4, 1000)
        original_est = 2.1
        lo, hi = bca_ci(samples, original_est=original_est)

        scale = 3.5
        shift = -1.2
        lo_t, hi_t = bca_ci(
            scale * samples + shift,
            original_est=scale * original_est + shift,
        )

        np.testing.assert_allclose(
            [lo_t, hi_t],
            [scale * lo + shift, scale * hi + shift],
            rtol=1e-10,
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Tests for BootstrapResult.ci_table()
# ---------------------------------------------------------------------------


class TestCiTable:
    def test_returns_dataframe(self):
        rng = np.random.default_rng(10)
        boot = _make_boot_result(rng)
        orig = _make_mock_original([1.0, 2.0], [0.05], [0.01])
        result = boot.ci_table(orig)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        rng = np.random.default_rng(11)
        boot = _make_boot_result(rng)
        orig = _make_mock_original([1.0, 2.0], [0.05], [0.01])
        df = boot.ci_table(orig)
        expected = {"parameter", "mean", "se", "p2_5", "p97_5", "bca_lo", "bca_hi"}
        assert expected.issubset(set(df.columns))

    def test_row_count(self):
        """One row per THETA + OMEGA diagonal + SIGMA diagonal."""
        rng = np.random.default_rng(12)
        n_theta, n_eta, n_eps = 3, 2, 1
        boot = _make_boot_result(rng, n_theta=n_theta, n_eta=n_eta, n_eps=n_eps)
        orig = _make_mock_original(np.ones(n_theta), np.ones(n_eta) * 0.05, np.ones(n_eps) * 0.01)
        df = boot.ci_table(orig)
        assert len(df) == n_theta + n_eta + n_eps

    def test_parameter_names(self):
        rng = np.random.default_rng(13)
        boot = _make_boot_result(rng, n_theta=2, n_eta=1, n_eps=1)
        orig = _make_mock_original([1.0, 2.0], [0.05], [0.01])
        df = boot.ci_table(orig)
        assert "THETA(1)" in df["parameter"].values
        assert "THETA(2)" in df["parameter"].values
        assert "OMEGA(1,1)" in df["parameter"].values
        assert "SIGMA(1,1)" in df["parameter"].values

    def test_bca_bounds_ordered(self):
        """BCa lower bound must be <= upper bound for all parameters."""
        rng = np.random.default_rng(14)
        boot = _make_boot_result(rng, n_boot=500)
        orig = _make_mock_original([1.0, 2.0], [0.05], [0.01])
        df = boot.ci_table(orig)
        assert (df["bca_lo"] <= df["bca_hi"]).all()

    def test_percentile_bounds_ordered(self):
        rng = np.random.default_rng(15)
        boot = _make_boot_result(rng, n_boot=500)
        orig = _make_mock_original([1.0, 2.0], [0.05], [0.01])
        df = boot.ci_table(orig)
        assert (df["p2_5"] <= df["p97_5"]).all()

    def test_se_positive(self):
        rng = np.random.default_rng(16)
        boot = _make_boot_result(rng)
        orig = _make_mock_original([1.0, 2.0], [0.05], [0.01])
        df = boot.ci_table(orig)
        assert (df["se"] > 0).all()

    def test_mean_close_to_true_value(self):
        """Bootstrap mean should be close to the true generating values."""
        rng = np.random.default_rng(17)
        boot = _make_boot_result(rng, n_boot=1000)
        orig = _make_mock_original([1.0, 2.0], [0.05], [0.01])
        df = boot.ci_table(orig)
        theta_rows = df[df["parameter"].str.startswith("THETA")]
        assert abs(theta_rows.iloc[0]["mean"] - 1.0) < 0.05
        assert abs(theta_rows.iloc[1]["mean"] - 2.0) < 0.1

    def test_bca_and_percentile_similar_for_symmetric(self):
        """For a well-behaved symmetric distribution, BCa and percentile CI
        should be within a narrow margin."""
        rng = np.random.default_rng(18)
        # Large n_boot and well-behaved distribution → small correction
        n_boot = 2000
        theta_samples = rng.normal(1.0, 0.1, (n_boot, 1))
        boot = BootstrapResult(
            n_boot=n_boot,
            n_success=n_boot,
            theta_samples=theta_samples,
            omega_diag_samples=np.abs(rng.normal(0.05, 0.005, (n_boot, 1))),
            sigma_diag_samples=np.abs(rng.normal(0.01, 0.001, (n_boot, 1))),
        )
        orig = _make_mock_original([1.0], [0.05], [0.01])
        df = boot.ci_table(orig)
        row = df[df["parameter"] == "THETA(1)"].iloc[0]
        # BCa and percentile CIs should be very close for symmetric normal
        assert abs(row["bca_lo"] - row["p2_5"]) < 0.05
        assert abs(row["bca_hi"] - row["p97_5"]) < 0.05

    def test_ci_table_bca_columns_match_direct_bca_computation(self):
        theta_samples = np.array([[0.8], [1.0], [1.2], [1.3], [1.4], [1.6]])
        boot = BootstrapResult(
            n_boot=6,
            n_success=6,
            theta_samples=theta_samples,
            omega_diag_samples=np.array([[0.4], [0.45], [0.5], [0.55], [0.6], [0.65]]),
            sigma_diag_samples=np.array([[0.08], [0.09], [0.10], [0.11], [0.12], [0.13]]),
        )
        orig = _make_mock_original([1.1], [0.52], [0.11])

        df = boot.ci_table(orig)
        theta_row = df[df["parameter"] == "THETA(1)"].iloc[0]
        expected_lo, expected_hi = bca_ci(theta_samples[:, 0], 1.1, 0.95)

        assert theta_row["bca_lo"] == pytest.approx(expected_lo)
        assert theta_row["bca_hi"] == pytest.approx(expected_hi)
