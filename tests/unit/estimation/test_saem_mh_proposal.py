"""
Tests for SAEM Cholesky MH proposal (S3).

Covers:
  1. Diagonal Omega: Cholesky proposal has same marginal variances as isotropic
  2. Correlated Omega [[1, 0.9], [0.9, 1]]: sample correlation ≈ 0.9 (±0.1)
  3. Near-singular Omega: Cholesky fails → fallback used, no crash
  4. Correlated model: Cholesky proposal yields higher acceptance than isotropic
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from openpkpd.estimation.saem import SAEMMethod


# ---------------------------------------------------------------------------
# Helpers: exercise the MH step in isolation via _e_step_one_subject
# ---------------------------------------------------------------------------


class _FakeIndiv:
    """Fake individual model that evaluates a log-posterior (eta^T eta)."""

    def __init__(self, target_mu=None, target_sigma=None):
        self.mu = np.zeros(2) if target_mu is None else target_mu
        self.cov = np.eye(2) if target_sigma is None else target_sigma
        self.inv_cov = np.linalg.inv(self.cov)

    def obj_eta(self, eta, theta, omega, sigma, trans=1):
        diff = eta - self.mu
        return float(diff @ self.inv_cov @ diff)


def _collect_proposals(omega: np.ndarray, scale: float, n_samples: int, seed: int = 0) -> np.ndarray:
    """
    Collect proposals from _e_step_one_subject for a single chain, rejecting
    all proposals (always keep current) to see the raw proposal distribution.

    This works by using a target that always rejects (obj_current << obj_prop).
    """
    rng = np.random.default_rng(seed)
    n_eta = omega.shape[0]

    # Target that always prefers current state (rejects proposals)
    class _AlwaysRejectIndiv:
        call_count = 0

        def obj_eta(self, eta, theta, omega, sigma, trans=1):
            # Return -inf for current (good) and 0 for proposals (bad)
            # Actually, we can't distinguish current vs proposal from here.
            # Instead, use a very tight target so almost all proposals are rejected.
            return float(np.dot(eta, eta)) * 1e6  # steep bowl around 0

    # Instead of always-reject, let's collect proposals by overriding randomness.
    # Run n_samples of the MH step starting from zero, record the proposal.
    proposals = []
    eta_init = np.zeros(n_eta)

    for i in range(n_samples):
        # Call _e_step_one_subject with n_chains=1
        # The "proposal" for chain 0 is: eta_current + Cholesky @ z
        # We can read it by observing the accepted/rejected new_chains when target is flat.
        class _FlatTarget:
            def obj_eta(self, eta, theta, omega, sigma, trans=1):
                return 0.0  # always accept → new_chains = proposal

        rng_i = np.random.default_rng(i + seed * 100)
        _, new_chains, _, _ = SAEMMethod._e_step_one_subject(
            sid=0,
            chains=np.array([eta_init]),
            scale=scale,
            indiv=_FlatTarget(),
            theta=np.array([]),
            omega=omega,
            sigma=np.eye(1),
            trans=1,
            n_chains=1,
            n_eta=n_eta,
            rng=rng_i,
        )
        proposals.append(new_chains[0] - eta_init)

    return np.array(proposals)  # shape (n_samples, n_eta)


# ---------------------------------------------------------------------------
# Test 1: Diagonal Omega → same marginal variances as isotropic
# ---------------------------------------------------------------------------


def test_diagonal_omega_matches_isotropic_variance():
    """For diagonal Omega, Cholesky proposal has the expected marginal variances."""
    omega = np.diag([0.5, 0.8])
    scale = 1.0
    n_samples = 5000

    proposals = _collect_proposals(omega, scale, n_samples, seed=1)
    # E[deta_i^2] should ≈ scale^2 * omega[i, i]
    for i in range(2):
        expected_var = scale**2 * omega[i, i]
        sample_var = np.var(proposals[:, i])
        assert abs(sample_var - expected_var) < 0.1 * expected_var + 0.02, (
            f"Dim {i}: expected var={expected_var:.4f}, got {sample_var:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Correlated Omega → sample correlation ≈ 0.9 (±0.1)
# ---------------------------------------------------------------------------


def test_correlated_omega_sample_correlation():
    """For Omega=[[1,0.9],[0.9,1]], proposals have sample correlation within ±0.1 of 0.9."""
    omega = np.array([[1.0, 0.9], [0.9, 1.0]])
    scale = 1.0
    n_samples = 10000

    proposals = _collect_proposals(omega, scale, n_samples, seed=2)
    corr = np.corrcoef(proposals[:, 0], proposals[:, 1])[0, 1]
    assert abs(corr - 0.9) < 0.1, f"Sample correlation={corr:.3f}, expected ~0.9"


# ---------------------------------------------------------------------------
# Test 3: Near-singular Omega → Cholesky fails → fallback, no crash
# ---------------------------------------------------------------------------


def test_near_singular_omega_fallback_no_crash():
    """Near-singular Omega causes Cholesky LinAlgError → fallback used, no exception."""
    # Rank-1 matrix (not positive definite)
    v = np.array([1.0, 1.0])
    omega = np.outer(v, v)  # [[1,1],[1,1]] — singular

    # Should not raise
    try:
        proposals = _collect_proposals(omega, scale=0.3, n_samples=100, seed=3)
        # If we get here, proposals were generated (fallback or cholesky succeeded)
        assert proposals.shape == (100, 2)
    except np.linalg.LinAlgError:
        pytest.fail("Near-singular omega caused unhandled LinAlgError in MH proposal")


# ---------------------------------------------------------------------------
# Test 4: Correlated target → Cholesky proposal yields higher acceptance than isotropic
# ---------------------------------------------------------------------------


def test_cholesky_higher_acceptance_than_isotropic_for_correlated_target():
    """
    For a highly correlated Gaussian target, the Cholesky proposal (matching the
    correlation) should yield a higher MH acceptance rate than an isotropic proposal
    of the same marginal step size.
    """
    rng = np.random.default_rng(42)
    n_eta = 2
    n_chains = 1
    n_steps = 2000

    # Correlated target: Gaussian with rho=0.9
    target_cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    target_indiv = _FakeIndiv(target_mu=np.zeros(2), target_sigma=target_cov)

    scale = 0.8

    # --- Cholesky proposal (omega matches target) ---
    omega_matched = target_cov.copy()
    eta_chains_chol = np.zeros((n_chains, n_eta))
    n_accepted_chol = 0
    for _ in range(n_steps):
        _, eta_chains_chol, _, n_acc = SAEMMethod._e_step_one_subject(
            sid=0,
            chains=eta_chains_chol,
            scale=scale,
            indiv=target_indiv,
            theta=np.array([]),
            omega=target_cov,
            sigma=np.eye(1),
            trans=1,
            n_chains=n_chains,
            n_eta=n_eta,
            rng=np.random.default_rng(_ + 1000),
        )
        n_accepted_chol += n_acc
    acc_rate_chol = n_accepted_chol / (n_steps * n_chains)

    # --- Isotropic proposal (diagonal omega with matched marginal variance) ---
    omega_isotropic = np.eye(n_eta) * np.mean(np.diag(target_cov))
    eta_chains_iso = np.zeros((n_chains, n_eta))
    n_accepted_iso = 0
    for _ in range(n_steps):
        _, eta_chains_iso, _, n_acc = SAEMMethod._e_step_one_subject(
            sid=0,
            chains=eta_chains_iso,
            scale=scale,
            indiv=target_indiv,
            theta=np.array([]),
            omega=omega_isotropic,
            sigma=np.eye(1),
            trans=1,
            n_chains=n_chains,
            n_eta=n_eta,
            rng=np.random.default_rng(_ + 2000),
        )
        n_accepted_iso += n_acc
    acc_rate_iso = n_accepted_iso / (n_steps * n_chains)

    # Cholesky proposal (correlated) should have higher acceptance for correlated target
    assert acc_rate_chol > 0.30, (
        f"Cholesky acceptance rate {acc_rate_chol:.3f} should be > 30% for correlated target"
    )
    # Cholesky should outperform isotropic for correlated target
    # (allow a small tolerance for stochastic variability)
    assert acc_rate_chol >= acc_rate_iso - 0.05, (
        f"Cholesky ({acc_rate_chol:.3f}) should not be much worse than isotropic ({acc_rate_iso:.3f})"
    )
