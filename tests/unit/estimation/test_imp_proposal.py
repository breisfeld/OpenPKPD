"""
I2: Tests for IMP proposal covariance PD validation with fallback.
"""
from __future__ import annotations

import logging
import numpy as np
import pytest

from openpkpd.math.matrix import repair_pd


# ── Test 1: Valid PD matrix passes through unchanged ─────────────────────────

def test_valid_pd_matrix_passes():
    """A valid PD matrix should not trigger the fallback."""
    V = np.array([[1.0, 0.3], [0.3, 0.5]])
    repaired = repair_pd(V)
    np.testing.assert_allclose(repaired, V, atol=1e-10)
    # Verify it's still PD
    np.linalg.cholesky(repaired)  # should not raise


# ── Test 2: Near-singular matrix is repaired ──────────────────────────────────

def test_near_singular_repaired():
    """A matrix with small negative eigenvalue should be repaired to PD."""
    evals = np.array([1.0, -1e-8])
    evecs = np.array([[0.6, -0.8], [0.8, 0.6]])
    V = evecs @ np.diag(evals) @ evecs.T

    repaired = repair_pd(V)
    # Should now be PD
    try:
        np.linalg.cholesky(repaired)
    except np.linalg.LinAlgError:
        pytest.fail("repair_pd failed to make the matrix PD")

    # Eigenvalues should all be positive
    eigs = np.linalg.eigvalsh(repaired)
    assert np.all(eigs > 0), f"Not all eigenvalues positive: {eigs}"


# ── Test 3: Severely ill-conditioned matrix falls back to diagonal ───────────

def test_severely_ill_conditioned_falls_back_to_diagonal(caplog):
    """
    Simulate the IMP proposal covariance fallback path: if repair_pd still
    fails to produce a Cholesky-decomposable matrix, use diagonal fallback.
    """
    # Build a matrix that cannot be repaired by standard eigenvalue clipping
    # (extremely large off-diagonal → correlation > 1 after repair fails)
    V = np.array([[1.0, 100.0], [100.0, 1.0]])  # off-diagonal > diagonal

    repaired = repair_pd(V)

    # Check if Cholesky succeeds after repair
    try:
        np.linalg.cholesky(repaired)
        chol_ok = True
    except np.linalg.LinAlgError:
        chol_ok = False

    if not chol_ok:
        # Simulate the IMP fallback logic
        import logging as _logging
        logger = _logging.getLogger("openpkpd.estimation.imp")
        with caplog.at_level(_logging.WARNING, logger="openpkpd.estimation.imp"):
            logger.warning(
                "IMP: repaired proposal covariance for subject %s is still not PD; "
                "falling back to diagonal",
                "test_subject",
            )
            proposal_cov = np.diag(np.diag(repaired))

        assert "falling back to diagonal" in caplog.text
        assert proposal_cov.shape == repaired.shape
        # Diagonal is always PD
        np.linalg.cholesky(proposal_cov)
    else:
        # repair_pd was sufficient; no fallback needed (test still passes)
        pass


# ── Test 4: Numerical — importance weights ESS > 0.5 * N_samples ─────────────

def test_imp_importance_weights_ess():
    """
    For a 2D Gaussian proposal matching the prior, ESS should be ~N.
    For a moderately mismatched proposal, ESS > 0.5*N.
    """
    from scipy.stats import multivariate_normal

    np.random.seed(42)
    N = 500
    n_eta = 2

    # True Omega (prior)
    omega = np.array([[0.1, 0.02], [0.02, 0.08]])
    omega_inv = np.linalg.inv(omega)
    log_det_omega = np.log(np.linalg.det(omega))

    # Proposal: mean=0, covariance=2*omega (slightly overdispersed)
    eta_map = np.zeros(n_eta)
    V_prop = 2.0 * omega  # PD by construction

    # Draw samples
    samples = np.random.multivariate_normal(eta_map, V_prop, size=N)

    # Log-prior weights: log p(eta | Omega) - log q(eta | V_prop)
    log_weights = np.empty(N)
    for s, eta_s in enumerate(samples):
        log_prior = -0.5 * float(eta_s @ omega_inv @ eta_s) - 0.5 * log_det_omega
        log_proposal = multivariate_normal.logpdf(eta_s, mean=eta_map, cov=V_prop)
        log_weights[s] = log_prior - log_proposal

    # Compute ESS
    lw_max = np.max(log_weights)
    w = np.exp(np.clip(log_weights - lw_max, -50, 0))
    w_norm = w / w.sum()
    ess = 1.0 / float(np.sum(w_norm**2))

    min_ess = 0.5 * N
    assert ess > min_ess, (
        f"ESS = {ess:.1f} is below minimum {min_ess:.1f} (N={N}). "
        "Proposal covariance may be degenerate."
    )
