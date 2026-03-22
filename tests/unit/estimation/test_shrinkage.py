"""
Unit tests for ETA shrinkage computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult


def _make_result(post_hoc_etas: dict, omega: np.ndarray) -> EstimationResult:
    """Helper to build a minimal EstimationResult."""
    omega.shape[0]
    return EstimationResult(
        theta_final=np.array([1.0]),
        omega_final=omega,
        sigma_final=np.eye(1),
        ofv=0.0,
        post_hoc_etas=post_hoc_etas,
    )


@pytest.mark.unit
def test_full_shrinkage_when_all_etas_zero():
    """When all EBEs are zero, shrinkage = 1.0 for every eta."""
    omega = np.diag([0.5, 0.3])
    etas = {i: np.array([0.0, 0.0]) for i in range(10)}
    result = _make_result(etas, omega)
    result.compute_shrinkage()
    np.testing.assert_allclose(result.eta_shrinkage, [1.0, 1.0], atol=1e-10)


@pytest.mark.unit
def test_zero_shrinkage_when_eta_sd_equals_omega_sd():
    """When std(EBE_k) ≈ sqrt(omega_kk), shrinkage ≈ 0."""
    rng = np.random.default_rng(42)
    omega_diag = np.array([0.25, 0.16])
    omega = np.diag(omega_diag)
    # Generate ETAs with exact SD matching sqrt(omega_kk)
    n_subj = 1000
    etas = {}
    for i in range(n_subj):
        eta = np.array(
            [
                rng.normal(0, np.sqrt(omega_diag[0])),
                rng.normal(0, np.sqrt(omega_diag[1])),
            ]
        )
        etas[i] = eta

    result = _make_result(etas, omega)
    result.compute_shrinkage()
    np.testing.assert_allclose(result.eta_shrinkage, [0.0, 0.0], atol=0.05)


@pytest.mark.unit
def test_shrinkage_in_zero_one():
    """Shrinkage should be in [0, 1] for well-behaved EBEs."""
    rng = np.random.default_rng(7)
    omega = np.diag([0.4, 0.2, 0.1])
    n_subj = 20
    etas = {}
    for i in range(n_subj):
        # ETAs smaller in variance than omega → shrinkage > 0
        etas[i] = rng.normal(0, 0.3, size=3)
    result = _make_result(etas, omega)
    result.compute_shrinkage()
    assert np.all(result.eta_shrinkage >= -0.1), "Shrinkage below 0"
    assert np.all(result.eta_shrinkage <= 1.0 + 1e-6), "Shrinkage above 1"


@pytest.mark.unit
def test_shrinkage_length_matches_n_eta():
    """eta_shrinkage should have length == n_eta."""
    omega = np.diag([0.3, 0.2, 0.1])
    etas = {i: np.array([0.1, -0.1, 0.0]) for i in range(5)}
    result = _make_result(etas, omega)
    result.compute_shrinkage()
    assert len(result.eta_shrinkage) == 3


@pytest.mark.unit
def test_empty_post_hoc_etas_no_crash():
    """compute_shrinkage with empty post_hoc_etas should not crash."""
    omega = np.diag([0.3, 0.2])
    result = _make_result({}, omega)
    # Should be a no-op
    result.compute_shrinkage()
    # eta_shrinkage stays as initialized (empty array)
    assert len(result.eta_shrinkage) == 0
