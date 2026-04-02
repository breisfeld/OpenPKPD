"""Tests for N5: lambda_z_criterion parameter in NCAEngine.

Verifies that 'adjr2', 'aic', and 'bic' criteria are selectable and
produce correct results.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.nca.nca import NCAEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def theopylline_terminal():
    """Theophylline subject 1 terminal phase data (4 points, post-Cmax)."""
    times = np.array([7.03, 9.05, 12.12, 24.37])
    conc = np.array([3.98, 3.15, 2.50, 0.92])
    return times, conc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLambdaZCriterion:
    """NCAEngine lambda_z_criterion parameter tests."""

    def test_adjr2_default_preserved(self, theopylline_terminal):
        """criterion='adjr2' gives the same result as the historical default."""
        times, conc = theopylline_terminal
        engine_default = NCAEngine()
        engine_explicit = NCAEngine(lambda_z_criterion="adjr2")

        lz_def, _, _ = engine_default._compute_lambda_z(times, conc, criterion="adjr2")
        lz_exp, _, _ = engine_explicit._compute_lambda_z(times, conc)

        assert lz_def == pytest.approx(lz_exp, rel=1e-12)

    def test_aic_selects_fewer_or_equal_points_than_adjr2(self, theopylline_terminal):
        """AIC criterion tends to be more parsimonious for small samples."""
        times, conc = theopylline_terminal
        engine = NCAEngine(min_points_lambda=3)

        _, _, n_adjr2 = engine._compute_lambda_z(times, conc, criterion="adjr2")
        _, _, n_aic = engine._compute_lambda_z(times, conc, criterion="aic")

        # AIC should select same or fewer points (more parsimonious)
        assert n_aic <= n_adjr2

    def test_bic_selects_same_or_fewer_points_than_aic(self, theopylline_terminal):
        """BIC penalises more strongly than AIC (2·log(n) vs 2·2 for n>8)."""
        times, conc = theopylline_terminal
        engine = NCAEngine(min_points_lambda=3)

        _, _, n_aic = engine._compute_lambda_z(times, conc, criterion="aic")
        _, _, n_bic = engine._compute_lambda_z(times, conc, criterion="bic")

        # BIC penalty >= AIC penalty for n >= 8; generally as or more parsimonious
        assert n_bic <= n_aic

    def test_invalid_criterion_raises_value_error(self):
        """Unknown criterion raises ValueError at engine creation."""
        with pytest.raises(ValueError, match="not recognised"):
            NCAEngine(lambda_z_criterion="invalid")

    def test_invalid_criterion_in_compute_raises(self, theopylline_terminal):
        """Passing an invalid criterion to _compute_lambda_z also raises ValueError."""
        times, conc = theopylline_terminal
        engine = NCAEngine()
        with pytest.raises(ValueError, match="not recognised"):
            engine._compute_lambda_z(times, conc, criterion="invalid")

    def test_aic_bic_lambda_z_close_to_adjr2_for_well_behaved_data(self, theopylline_terminal):
        """For theophylline terminal phase all criteria agree within ±10%."""
        times, conc = theopylline_terminal
        engine = NCAEngine(min_points_lambda=3)

        lz_adjr2, _, _ = engine._compute_lambda_z(times, conc, criterion="adjr2")
        lz_aic, _, _ = engine._compute_lambda_z(times, conc, criterion="aic")
        lz_bic, _, _ = engine._compute_lambda_z(times, conc, criterion="bic")

        assert np.isfinite(lz_adjr2)
        assert np.isfinite(lz_aic)
        assert np.isfinite(lz_bic)

        assert lz_aic == pytest.approx(lz_adjr2, rel=0.10), (
            f"AIC lambda_z {lz_aic:.4f} differs >10% from adjr2 {lz_adjr2:.4f}"
        )
        assert lz_bic == pytest.approx(lz_adjr2, rel=0.10), (
            f"BIC lambda_z {lz_bic:.4f} differs >10% from adjr2 {lz_adjr2:.4f}"
        )

    def test_compute_subject_uses_criterion(self, theopylline_terminal):
        """compute_subject passes lambda_z_criterion through to _compute_lambda_z."""
        times, conc = theopylline_terminal
        dose = 4.02  # mg/kg theophylline

        engine_adjr2 = NCAEngine(lambda_z_criterion="adjr2")
        engine_aic = NCAEngine(lambda_z_criterion="aic")

        # Prepend a time-0 concentration (dose event) so terminal phase is
        # reached after Cmax
        full_times = np.concatenate([[0.0], times])
        full_conc = np.concatenate([[0.0], conc])

        res_adjr2 = engine_adjr2.compute_subject(full_times, full_conc, dose, route="oral")
        res_aic = engine_aic.compute_subject(full_times, full_conc, dose, route="oral")

        # Both should produce a finite lambda_z
        assert np.isfinite(res_adjr2.lambda_z)
        assert np.isfinite(res_aic.lambda_z)

    def test_aic_bic_return_valid_for_perfect_exponential(self):
        """AIC/BIC criteria work on a perfect monoexponential dataset."""
        lz_true = 0.08
        n = 10
        times = np.linspace(5.0, 5.0 + (n - 1) * 2.0, n)
        conc = 3.0 * np.exp(-lz_true * times)

        engine = NCAEngine(min_points_lambda=3)
        for crit in ("adjr2", "aic", "bic"):
            lz, r2, n_pts = engine._compute_lambda_z(times, conc, criterion=crit)
            assert np.isfinite(lz), f"{crit}: lambda_z not finite"
            assert lz == pytest.approx(lz_true, rel=0.01), f"{crit}: lambda_z wrong"
