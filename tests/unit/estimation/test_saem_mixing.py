"""Tests for SAEM Phase 1 mixing diagnostic warning."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from openpkpd.estimation.saem import ConvergenceWarning, SAEMMethod


def _build_stable_history(n: int = 200, n_params: int = 2) -> list[np.ndarray]:
    """History where the last two windows are nearly identical (well-mixed)."""
    rng = np.random.default_rng(0)
    # Constant value + tiny noise → negligible relative change
    base = np.array([1.0, 0.5])
    return [base + rng.normal(0, 1e-6, n_params) for _ in range(n)]


def _build_unstable_history(n: int = 200, n_params: int = 2) -> list[np.ndarray]:
    """History where the last two windows differ by > phi_tol (not mixed)."""
    # Linearly drifting parameters → large relative change between windows
    return [np.array([1.0 + i * 0.01, 0.5 + i * 0.005]) for i in range(n)]


@pytest.mark.unit
class TestSAEMPhase1Mixing:
    def test_stable_history_no_warning(self):
        """Well-mixed Phase 1 (stable parameter history) → no ConvergenceWarning."""
        saem = SAEMMethod(n_iter_phase1=200, phi_tol=1e-3, seed=0)
        history = _build_stable_history(n=200)

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            # Should not raise
            saem._check_phase1_mixing(history)

    def test_unstable_history_warns(self):
        """Unstable Phase 1 history → ConvergenceWarning mentioning 'n_burn'."""
        saem = SAEMMethod(n_iter_phase1=200, phi_tol=1e-3, seed=0)
        history = _build_unstable_history(n=200)

        with pytest.warns(ConvergenceWarning, match="n_burn"):
            saem._check_phase1_mixing(history)

    def test_warning_content_mentions_mixing(self):
        """Warning message mentions 'Phase 1 may not have mixed'."""
        saem = SAEMMethod(n_iter_phase1=200, phi_tol=1e-3, seed=0)
        history = _build_unstable_history(n=200)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            saem._check_phase1_mixing(history)

        assert len(caught) == 1
        msg = str(caught[0].message)
        assert "Phase 1" in msg
        assert "n_burn" in msg

    def test_warning_emitted_once_at_transition(self):
        """
        _check_phase1_mixing is called exactly once at the transition.
        Verify that calling it once emits at most one warning.
        """
        saem = SAEMMethod(n_iter_phase1=100, phi_tol=1e-3, seed=0)
        history = _build_unstable_history(n=100)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            saem._check_phase1_mixing(history)

        assert len(caught) <= 1

    def test_insufficient_history_no_warning(self):
        """Too few iterations for the window check → no spurious warning."""
        saem = SAEMMethod(n_iter_phase1=200, phi_tol=1e-3, seed=0)
        # Only 5 entries, far below 2*(200//4)=100 needed
        history = _build_unstable_history(n=5)

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            saem._check_phase1_mixing(history)  # should not raise
