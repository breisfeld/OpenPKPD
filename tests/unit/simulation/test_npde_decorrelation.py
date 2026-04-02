"""Tests for NPDE decorrelation with insufficient replicates (NP2)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from openpkpd.simulation.npde import _decorrelate


class TestNPDEDecorrelation:
    """Tests for _decorrelate() behavior with sufficient/insufficient replicates."""

    def test_sufficient_replicates_no_warning(self):
        """With K_avail > n_i + 2 complete replicates, no RuntimeWarning should be raised."""
        n_i = 3
        K = 20  # K_avail = 20 > n_i + 2 = 5
        rng = np.random.default_rng(42)
        pde_i = rng.normal(0, 1, n_i)
        Y_sim = rng.normal(0, 1, (n_i, K))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _decorrelate(pde_i, Y_sim)

        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, (
            f"Unexpected RuntimeWarning with sufficient replicates: {[str(x.message) for x in runtime_warnings]}"
        )
        assert result.shape == (n_i,)

    def test_insufficient_replicates_raises_runtime_warning(self):
        """With K_avail < n_i + 2, RuntimeWarning mentioning counts should be raised."""
        n_i = 5
        K = 3  # K_avail = 3 < n_i + 2 = 7
        rng = np.random.default_rng(99)
        pde_i = rng.normal(0, 1, n_i)
        Y_sim = rng.normal(0, 1, (n_i, K))

        with pytest.warns(RuntimeWarning, match="insufficient replicates"):
            result = _decorrelate(pde_i, Y_sim)

    def test_insufficient_replicates_warning_contains_counts(self):
        """RuntimeWarning message should mention both n_i and K_avail counts."""
        n_i = 4
        K = 2  # K_avail = 2 < n_i + 2 = 6
        rng = np.random.default_rng(77)
        pde_i = rng.normal(0, 1, n_i)
        Y_sim = rng.normal(0, 1, (n_i, K))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _decorrelate(pde_i, Y_sim)

        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) >= 1
        msg = str(runtime_warnings[0].message)
        # Should mention the number of observations
        assert str(n_i) in msg, f"Expected n_i={n_i} in warning message: {msg}"
        # Should mention needed count
        needed = n_i + 2
        assert str(needed) in msg or str(K) in msg, (
            f"Expected count info in warning message: {msg}"
        )

    def test_insufficient_replicates_returns_raw_pde(self):
        """When decorrelation is skipped, the returned values should equal the raw pde_i."""
        n_i = 4
        K = 2  # K_avail = 2 < n_i + 2 = 6
        rng = np.random.default_rng(13)
        pde_i = np.array([0.5, -1.2, 0.3, 2.1])
        Y_sim = rng.normal(0, 1, (n_i, K))

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _decorrelate(pde_i, Y_sim)

        np.testing.assert_array_equal(result, pde_i), (
            "Expected raw pde_i to be returned when decorrelation is skipped"
        )
        # Should NOT be NaN
        assert np.all(np.isfinite(result)), "Raw pde_i should not contain NaN/Inf"

    def test_nan_columns_excluded_from_k_avail(self):
        """NaN-containing columns should be excluded from K_avail count."""
        n_i = 2
        # With 6 columns but 4 containing NaN, K_avail = 2 < n_i + 2 = 4 → warning
        K = 6
        rng = np.random.default_rng(55)
        pde_i = np.array([0.1, -0.2])
        Y_sim = rng.normal(0, 1, (n_i, K))
        # Introduce NaN in 4 of the 6 columns
        Y_sim[:, 2:] = np.nan

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _decorrelate(pde_i, Y_sim)

        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) >= 1, (
            "Expected RuntimeWarning when NaN columns reduce K_avail below threshold"
        )
        np.testing.assert_array_equal(result, pde_i)

    def test_exactly_threshold_k_avail_raises_warning(self):
        """K_avail == n_i + 1 (strictly < n_i + 2) should raise warning."""
        n_i = 3
        K = n_i + 1  # exactly n_i + 1, which is < n_i + 2
        rng = np.random.default_rng(0)
        pde_i = rng.normal(0, 1, n_i)
        Y_sim = rng.normal(0, 1, (n_i, K))

        with pytest.warns(RuntimeWarning):
            _decorrelate(pde_i, Y_sim)

    def test_exactly_sufficient_k_avail_no_warning(self):
        """K_avail == n_i + 2 (the minimum sufficient value) should NOT raise warning."""
        n_i = 3
        K = n_i + 2  # exactly n_i + 2, the minimum to proceed
        rng = np.random.default_rng(11)
        pde_i = rng.normal(0, 1, n_i)
        Y_sim = rng.normal(0, 1, (n_i, K))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _decorrelate(pde_i, Y_sim)

        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, (
            f"Unexpected warning at K_avail == n_i + 2: {[str(x.message) for x in runtime_warnings]}"
        )
