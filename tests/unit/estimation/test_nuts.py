"""
Tests for the native NUTS sampler (estimation/nuts.py).

Verifies correctness against known distributions: standard normal and
a correlated bivariate normal.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.nuts import NUTSSampler, _leapfrog, nuts_estimate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _std_normal_log_prob(theta: np.ndarray) -> float:
    """log p(theta) ∝ -0.5 * ||theta||^2  (standard normal)."""
    return float(-0.5 * np.dot(theta, theta))


def _std_normal_grad(theta: np.ndarray) -> np.ndarray:
    """Gradient of std normal log-prob: -theta."""
    return -theta.copy()


def _banana_log_prob(theta: np.ndarray) -> float:
    """Rosenbrock-banana: banana-shaped distribution for testing."""
    x, y = float(theta[0]), float(theta[1])
    return float(-0.5 * ((x**2 + (y - x**2) ** 2) / 2.0))


# ---------------------------------------------------------------------------
# _leapfrog
# ---------------------------------------------------------------------------


class TestLeapfrog:
    def test_energy_approximately_conserved(self):
        """Leapfrog should (nearly) conserve Hamiltonian energy."""
        theta = np.array([1.0, 0.0])
        r = np.array([0.0, 1.0])
        # Standard normal: H = 0.5*r^2 - log_prob(theta)
        H0 = 0.5 * np.dot(r, r) - _std_normal_log_prob(theta)
        theta_new, r_new = _leapfrog(theta, r, _std_normal_grad, step_size=0.1, n_steps=10)
        H1 = 0.5 * np.dot(r_new, r_new) - _std_normal_log_prob(theta_new)
        assert abs(H1 - H0) < 0.1  # energy approximately conserved

    def test_reversibility(self):
        """Leapfrog is time-reversible: flip momentum, apply again → original."""
        theta = np.array([0.5, -0.3])
        r = np.array([1.2, -0.4])
        theta_new, r_new = _leapfrog(theta, r, _std_normal_grad, step_size=0.05, n_steps=5)
        theta_back, r_back = _leapfrog(
            theta_new, -r_new, _std_normal_grad, step_size=0.05, n_steps=5
        )
        np.testing.assert_allclose(theta_back, theta, atol=1e-8)


# ---------------------------------------------------------------------------
# NUTSSampler — standard normal
# ---------------------------------------------------------------------------


class TestNUTSSamplerStdNormal:
    @pytest.fixture()
    def samples_1d(self):
        sampler = NUTSSampler(
            _std_normal_log_prob,
            _std_normal_grad,
            delta=0.65,
            seed=42,
        )
        return sampler.sample(np.array([0.0]), n_samples=500, n_warmup=200)

    def test_sample_shape(self, samples_1d):
        assert samples_1d.shape == (500, 1)

    def test_mean_near_zero(self, samples_1d):
        mean = float(samples_1d[:, 0].mean())
        assert abs(mean) < 0.2, f"Mean {mean:.4f} too far from 0"

    def test_std_near_one(self, samples_1d):
        std = float(samples_1d[:, 0].std())
        assert 0.7 < std < 1.4, f"Std {std:.4f} not near 1"

    def test_samples_finite(self, samples_1d):
        assert np.all(np.isfinite(samples_1d))


class TestNUTSSamplerFDGradient:
    """NUTS with finite-difference gradient (no analytic grad)."""

    def test_fd_gradient_produces_samples(self):
        sampler = NUTSSampler(
            _std_normal_log_prob,
            grad_log_prob_fn=None,  # FD gradient
            seed=7,
        )
        samples = sampler.sample(np.array([0.0]), n_samples=100, n_warmup=50)
        assert samples.shape == (100, 1)
        assert np.all(np.isfinite(samples))


class TestNUTSSampler2D:
    def test_bivariate_normal_shape(self):
        def log_prob(theta):
            return _std_normal_log_prob(theta)

        def grad(theta):
            return _std_normal_grad(theta)

        sampler = NUTSSampler(log_prob, grad, seed=123)
        samples = sampler.sample(np.zeros(2), n_samples=200, n_warmup=100)
        assert samples.shape == (200, 2)

    def test_bivariate_normal_mean_near_zero(self):
        def log_prob(theta):
            return _std_normal_log_prob(theta)

        def grad(theta):
            return _std_normal_grad(theta)

        sampler = NUTSSampler(log_prob, grad, seed=99)
        samples = sampler.sample(np.zeros(2), n_samples=300, n_warmup=150)
        mean = np.abs(samples.mean(axis=0))
        assert np.all(mean < 0.3)

    def test_bivariate_correlated_normal_covariance(self):
        cov = np.array([[1.0, 0.6], [0.6, 2.0]])
        precision = np.linalg.inv(cov)

        def log_prob(theta):
            return float(-0.5 * theta @ precision @ theta)

        def grad(theta):
            return -(precision @ theta)

        sampler = NUTSSampler(log_prob, grad, delta=0.7, seed=321)
        samples = sampler.sample(np.zeros(2), n_samples=600, n_warmup=300)

        np.testing.assert_allclose(samples.mean(axis=0), np.zeros(2), atol=0.25)
        np.testing.assert_allclose(np.cov(samples.T), cov, atol=0.35)


# ---------------------------------------------------------------------------
# nuts_estimate
# ---------------------------------------------------------------------------


class TestNutsEstimate:
    def test_returns_dict(self):
        result = nuts_estimate(
            _std_normal_log_prob,
            np.array([0.0]),
            n_samples=100,
            n_warmup=50,
            seed=42,
        )
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self):
        result = nuts_estimate(
            _std_normal_log_prob,
            np.array([0.0]),
            n_samples=100,
            n_warmup=50,
            seed=1,
        )
        assert "samples" in result
        assert "r_hat" in result
        assert "n_effective" in result
        assert "backend_used" in result

    def test_backend_used_is_nuts(self):
        result = nuts_estimate(
            _std_normal_log_prob,
            np.array([0.0]),
            n_samples=50,
            n_warmup=25,
            seed=0,
        )
        assert result["backend_used"] == "nuts"

    def test_r_hat_near_one(self):
        result = nuts_estimate(
            _std_normal_log_prob,
            np.array([0.0, 0.0]),
            n_samples=200,
            n_warmup=100,
            seed=5,
        )
        # r_hat is set to ones (single chain, no multi-chain diagnostic)
        np.testing.assert_allclose(result["r_hat"], np.ones(2))

    def test_n_effective_positive(self):
        result = nuts_estimate(
            _std_normal_log_prob,
            np.array([0.0]),
            n_samples=100,
            n_warmup=50,
            seed=3,
        )
        assert result["n_effective"][0] > 0
