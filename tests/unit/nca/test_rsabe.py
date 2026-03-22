"""Tests for Reference-Scaled ABE."""

import numpy as np
import pytest

from openpkpd.nca.bioequivalence import (
    RSABEResult,
    average_bioequivalence,
    reference_scaled_abe,
)


class TestRSABE:
    def test_low_variability_falls_back_to_abe(self):
        """When sigma_wr <= sigma_w0, standard ABE is used."""
        rng = np.random.default_rng(42)
        # Generate highly correlated reference data → low within-subject variability
        r1 = rng.lognormal(mean=3.0, sigma=0.1, size=24)
        r2 = r1 * rng.lognormal(mean=0.0, sigma=0.05, size=24)  # tight
        t = r1 * rng.lognormal(mean=0.0, sigma=0.05, size=24)

        result = reference_scaled_abe(t, r1, r2, sigma_w0=0.25)
        # sigma_wr should be small → used_abe = True
        assert isinstance(result, RSABEResult)
        if result.sigma_wr <= 0.25:
            assert result.used_abe is True

    def test_high_variability_uses_rsabe(self):
        """When sigma_wr > sigma_w0, RSABE criterion used."""
        rng = np.random.default_rng(99)
        # Generate high within-subject variability
        r1 = rng.lognormal(mean=3.0, sigma=0.4, size=24)
        r2 = r1 * rng.lognormal(mean=0.0, sigma=0.4, size=24)  # wide
        t = r1 * rng.lognormal(mean=0.0, sigma=0.1, size=24)

        result = reference_scaled_abe(t, r1, r2, sigma_w0=0.25)
        if result.sigma_wr > 0.25:
            assert result.used_abe is False
            assert np.isfinite(result.scaled_criterion)
            assert np.isfinite(result.upper_bound_ci)

    def test_result_fields(self):
        """RSABEResult has all expected fields."""
        rng = np.random.default_rng(7)
        r1 = rng.lognormal(mean=3.0, sigma=0.3, size=12)
        r2 = r1 * rng.lognormal(mean=0.0, sigma=0.3, size=12)
        t = r1 * rng.lognormal(mean=0.0, sigma=0.1, size=12)

        result = reference_scaled_abe(t, r1, r2)
        assert hasattr(result, "sigma_wr")
        assert hasattr(result, "gmr")
        assert hasattr(result, "bioequivalent")
        assert hasattr(result, "method")
        assert result.gmr > 0
        assert result.sigma_wr >= 0

    def test_invalid_inputs(self):
        """Raises ValueError for invalid inputs."""
        with pytest.raises(ValueError):
            reference_scaled_abe(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),  # wrong length
                np.array([1.0, 2.0, 3.0]),
            )

    def test_ema_variant(self):
        """EMA variant uses sigma_w0=0.294."""
        rng = np.random.default_rng(42)
        r1 = rng.lognormal(mean=3.0, sigma=0.3, size=12)
        r2 = r1 * rng.lognormal(mean=0.0, sigma=0.3, size=12)
        t = r1 * rng.lognormal(mean=0.0, sigma=0.1, size=12)

        result_fda = reference_scaled_abe(t, r1, r2, regulatory="FDA")
        result_ema = reference_scaled_abe(t, r1, r2, regulatory="EMA")
        assert result_ema.sigma_w0 == pytest.approx(0.294)
        assert result_fda.sigma_w0 == pytest.approx(0.25)

    def test_abe_fallback_matches_average_bioequivalence(self):
        """Low-variability fallback should match the paired ABE calculation."""
        rng = np.random.default_rng(123)
        r1 = rng.lognormal(mean=3.0, sigma=0.08, size=18)
        r2 = r1 * rng.lognormal(mean=0.0, sigma=0.03, size=18)
        t = np.sqrt(r1 * r2) * rng.lognormal(mean=0.0, sigma=0.04, size=18)

        result = reference_scaled_abe(t, r1, r2, sigma_w0=0.25)
        ref_geom_mean = np.sqrt(r1 * r2)
        abe = average_bioequivalence(t, ref_geom_mean)

        assert result.used_abe is True
        assert result.gmr == pytest.approx(abe.gmr)
        assert result.gmr_ci_lo == pytest.approx(abe.gmr_ci_lo)
        assert result.gmr_ci_hi == pytest.approx(abe.gmr_ci_hi)
        assert result.bioequivalent == abe.bioequivalent

    def test_abe_fallback_exact_lower_limit_is_accepted(self):
        """Fallback ABE should accept an exact 80% lower-bound CI."""
        r1 = np.array([100.0, 120.0, 90.0, 110.0, 95.0])
        r2 = r1.copy()
        t = 0.80 * np.sqrt(r1 * r2)

        result = reference_scaled_abe(t, r1, r2, sigma_w0=0.25)

        assert result.used_abe is True
        assert result.gmr == pytest.approx(0.80)
        assert result.gmr_ci_lo == pytest.approx(0.80)
        assert result.gmr_ci_hi == pytest.approx(0.80)
        assert result.bioequivalent

    def test_high_variability_rsabe_accepts_exact_lower_gmr_boundary(self):
        """High-variability RSABE should not fail exact 80% GMR due to roundoff."""
        base = np.array([0.45, -0.45, 0.225, -0.225] * 3, dtype=float)
        r1 = 100.0 * np.exp(base)
        r2 = 100.0 * np.exp(-base)
        t = 0.80 * np.sqrt(r1 * r2)

        result = reference_scaled_abe(t, r1, r2, sigma_w0=0.25)

        assert result.used_abe is False
        assert result.upper_bound_ci < 0.0
        assert result.gmr == pytest.approx(0.80)
        assert result.bioequivalent
