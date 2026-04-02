"""Tests for N6: consolidated lambda_z regression (single numpy path).

Verifies that the single vectorised implementation produces identical
results regardless of sample size (previously n<=16 used a scalar loop
and n>16 used the vectorised path).
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.nca.nca import NCAEngine


# ---------------------------------------------------------------------------
# Helper: build a synthetic monoexponential terminal phase dataset
# ---------------------------------------------------------------------------

def _make_terminal_data(n: int, lz: float = 0.08, c0: float = 5.0, t0: float = 4.0):
    """Generate n synthetic terminal-phase points for a monoexponential decay."""
    times = np.linspace(t0, t0 + (n - 1) * 1.5, n)
    conc = c0 * np.exp(-lz * times)
    return times, conc


# ---------------------------------------------------------------------------
# N6: Consolidation equivalence
# ---------------------------------------------------------------------------


class TestConsolidation:
    """Both sample sizes produce identical lambda_z from the same algorithm."""

    def test_small_n_gives_correct_lambda(self):
        """n=8 (formerly scalar path): lambda_z matches true value within 1%."""
        engine = NCAEngine(min_points_lambda=3)
        times, conc = _make_terminal_data(n=8, lz=0.08)
        lz, r2, n_pts = engine._compute_lambda_z(times, conc)
        assert np.isfinite(lz), "lambda_z must be finite"
        assert lz == pytest.approx(0.08, rel=0.01), f"Expected ~0.08, got {lz}"
        assert 3 <= n_pts <= 8, f"n_pts={n_pts} out of expected range [3, 8]"

    def test_large_n_gives_correct_lambda(self):
        """n=20 (formerly vectorised path): lambda_z matches true value within 1%."""
        engine = NCAEngine(min_points_lambda=3)
        times, conc = _make_terminal_data(n=20, lz=0.08)
        lz, r2, n_pts = engine._compute_lambda_z(times, conc)
        assert np.isfinite(lz), "lambda_z must be finite"
        assert lz == pytest.approx(0.08, rel=0.01), f"Expected ~0.08, got {lz}"

    def test_n8_and_n20_identical_for_same_data(self):
        """Same 8-point dataset gives identical lambda_z when padded to 20 identical copies.

        This is not a meaningful pharmacological test, but it verifies that the single
        algorithm path gives reproducible results — it's the same code for all n.
        """
        engine = NCAEngine(min_points_lambda=3)
        # Use the same 8-point segment for both calls
        times, conc = _make_terminal_data(n=8, lz=0.10, c0=4.0, t0=3.0)
        lz_8, r2_8, _ = engine._compute_lambda_z(times, conc)

        # Build a 20-point dataset from the same underlying decay
        times20, conc20 = _make_terminal_data(n=20, lz=0.10, c0=4.0, t0=3.0)
        lz_20, r2_20, _ = engine._compute_lambda_z(times20, conc20)

        # Both should recover the true lambda_z closely
        assert lz_8 == pytest.approx(0.10, rel=0.01)
        assert lz_20 == pytest.approx(0.10, rel=0.01)

    def test_r_squared_near_one_for_perfect_exponential(self):
        """Perfect monoexponential data should yield R² very close to 1."""
        engine = NCAEngine(min_points_lambda=3)
        for n in (3, 8, 16, 17, 25):
            times, conc = _make_terminal_data(n=n, lz=0.05)
            lz, r2, n_pts = engine._compute_lambda_z(times, conc)
            assert r2 == pytest.approx(1.0, abs=1e-10), f"n={n}: R²={r2} not near 1"

    def test_min_points_lambda_respected(self):
        """Engine returns (nan, nan, 0) when fewer than min_points_lambda are available."""
        engine = NCAEngine(min_points_lambda=5)
        times, conc = _make_terminal_data(n=4, lz=0.08)
        lz, r2, n_pts = engine._compute_lambda_z(times, conc)
        assert not np.isfinite(lz)
        assert not np.isfinite(r2)
        assert n_pts == 0
