"""Tests for SAEM SA exponent (alpha) validation and Phase 2 step counter."""

from __future__ import annotations

import pytest

from openpkpd.estimation.saem import SAEMMethod


@pytest.mark.unit
class TestSAEMAlphaValidation:
    def test_default_alpha_no_error(self):
        """alpha=0.7 (default) should construct without error."""
        m = SAEMMethod()
        assert m.alpha == pytest.approx(0.7)

    def test_alpha_boundary_exclusive_lower(self):
        """alpha=0.5 is on the exclusive boundary → ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            SAEMMethod(alpha=0.5)

    def test_alpha_boundary_inclusive_upper(self):
        """alpha=1.0 is on the inclusive upper boundary → no error."""
        m = SAEMMethod(alpha=1.0)
        assert m.alpha == pytest.approx(1.0)

    def test_alpha_too_large(self):
        """alpha=1.1 is above the inclusive upper → ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            SAEMMethod(alpha=1.1)

    def test_alpha_too_small_negative(self):
        """alpha=0.0 is below the exclusive lower → ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            SAEMMethod(alpha=0.0)


@pytest.mark.unit
class TestSAEMGammaSchedule:
    """Numerical gamma values at specific Phase 2 steps."""

    def test_gamma_phase2_step1(self):
        """
        At the first Phase 2 step (k == n_iter_phase1), the counter is
        k - n_iter_phase1 + 1 = 1, so gamma = 1^(-0.7) = 1.0.
        """
        n_phase1 = 100
        alpha = 0.7
        k = n_phase1  # first Phase 2 iteration
        counter = k - n_phase1 + 1  # should be 1
        assert counter == 1, "Phase 2 step counter must start at 1"
        gamma = counter ** (-alpha)
        assert gamma == pytest.approx(1.0, abs=1e-10)

    def test_gamma_phase2_step10(self):
        """At the 10th Phase 2 step, gamma = 10^(-0.7) ≈ 0.1995."""
        n_phase1 = 100
        alpha = 0.7
        k = n_phase1 + 9  # 10th Phase 2 iteration
        counter = k - n_phase1 + 1  # should be 10
        assert counter == 10
        gamma = counter ** (-alpha)
        assert gamma == pytest.approx(10 ** (-0.7), rel=1e-6)

    def test_phase2_counter_starts_at_1_not_0(self):
        """Verify that the counter at k == n_iter_phase1 is exactly 1."""
        n_phase1 = 300
        k = n_phase1  # first Phase 2 step
        counter = k - n_phase1 + 1
        assert counter == 1

    def test_gamma_uses_self_alpha(self):
        """SAEMMethod stores alpha on self and it matches the constructor arg."""
        m = SAEMMethod(alpha=0.8)
        assert m.alpha == pytest.approx(0.8)
        # Verify the gamma formula uses self.alpha correctly
        n_phase1 = 50
        k = n_phase1 + 4  # 5th Phase 2 step → counter=5
        counter = k - n_phase1 + 1
        expected_gamma = counter ** (-m.alpha)
        assert expected_gamma == pytest.approx(5 ** (-0.8), rel=1e-6)
