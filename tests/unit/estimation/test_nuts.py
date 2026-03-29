"""
Tests for the native NUTS sampler (estimation/nuts.py).

Verifies correctness at three levels:

1. Unit — _leapfrog and _build_tree internals (deterministic, analytic).
2. Statistical — NUTSSampler recovers known distributions (mean, std,
   covariance, quantile coverage, autocorrelation efficiency).
3. Integration — nuts_estimate() wrapper returns correct structure and
   meaningful diagnostics.

Each statistical test uses a fixed seed and enough samples so that
failures reflect genuine algorithmic bugs rather than sampling noise.
Tolerances are deliberately loose (3–4σ of the Monte Carlo estimator).
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.nuts import NUTSSampler, _build_tree, _leapfrog, nuts_estimate

# ---------------------------------------------------------------------------
# Shared analytic targets
# ---------------------------------------------------------------------------


def _std_normal_log_prob(theta: np.ndarray) -> float:
    """log p(theta) ∝ -0.5 * ||theta||^2  (standard normal, any dim)."""
    return float(-0.5 * np.dot(theta, theta))


def _std_normal_grad(theta: np.ndarray) -> np.ndarray:
    """Gradient of std normal log-prob: -theta."""
    return -theta.copy()


def _normal_log_prob(mu: float, sd: float):
    """Factory: returns log p for N(mu, sd^2)."""
    def _lp(theta: np.ndarray) -> float:
        return float(-0.5 * ((theta[0] - mu) / sd) ** 2)
    return _lp


def _normal_grad(mu: float, sd: float):
    """Factory: returns gradient of N(mu, sd^2) log-prob."""
    def _g(theta: np.ndarray) -> np.ndarray:
        return np.array([-(theta[0] - mu) / sd ** 2])
    return _g


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

    def test_volume_preservation_1d(self):
        """Leapfrog is symplectic: the Jacobian det of (θ,r)→(θ',r') is 1.

        Symplecticity is the key property that makes HMC/NUTS asymptotically
        exact.  A finite-difference Jacobian is sufficient to verify it.
        """
        step_size = 0.1
        n_steps = 5
        eps_jac = 1e-6

        x0 = np.array([0.5, 0.3])  # [theta, r] stacked for 1-D case

        def step(x: np.ndarray) -> np.ndarray:
            th, rv = _leapfrog(x[:1], x[1:], _std_normal_grad, step_size, n_steps)
            return np.concatenate([th, rv])

        f0 = step(x0)
        J = np.zeros((2, 2))
        for i in range(2):
            ei = np.zeros(2)
            ei[i] = eps_jac
            J[:, i] = (step(x0 + ei) - f0) / eps_jac

        det = np.linalg.det(J)
        assert abs(det - 1.0) < 0.01, (
            f"Leapfrog Jacobian det = {det:.6f}, expected 1.0 (symplecticity)"
        )

    def test_negative_step_direction(self):
        """A negative step size moves the trajectory backward in time."""
        theta = np.array([1.0])
        r = np.array([1.0])
        th_fwd, _ = _leapfrog(theta, r, _std_normal_grad, step_size=+0.2, n_steps=1)
        th_bwd, _ = _leapfrog(theta, r, _std_normal_grad, step_size=-0.2, n_steps=1)
        # Forward step increases theta (r > 0); backward decreases it
        assert th_fwd[0] > theta[0], "Forward step should increase theta"
        assert th_bwd[0] < theta[0], "Backward step should decrease theta"


# ---------------------------------------------------------------------------
# _build_tree — unit tests for the NUTS tree builder
# ---------------------------------------------------------------------------


class TestBuildTree:
    """Direct unit tests for _build_tree with the standard-normal target."""

    # Shared joint log-prob (std normal + kinetic energy)
    @staticmethod
    def _joint(theta, r):
        return float(_std_normal_log_prob(theta)) - 0.5 * float(np.dot(r, r))

    def test_base_case_returns_nine_elements(self):
        """Base case (j=0) must return a 9-tuple."""
        result = _build_tree(
            np.array([0.0]), np.array([0.0]),
            log_u=-5.0, v=1, j=0,
            step_size=0.1,
            log_prob=_std_normal_log_prob,
            grad_log_prob=_std_normal_grad,
            joint_log_prob=self._joint,
        )
        assert len(result) == 9, f"Expected 9-tuple, got {len(result)} elements"

    def test_base_case_n_prime_1_when_in_slice(self):
        """n′=1 when the new joint log-prob exceeds the slice threshold log_u."""
        # At theta=0, r=0, joint=0. With log_u=-5 (well below), n_prime should be 1.
        *_, n_prime, s_prime, alpha_prime, n_alpha_prime = _build_tree(
            np.array([0.0]), np.array([0.0]),
            log_u=-5.0, v=1, j=0,
            step_size=0.1,
            log_prob=_std_normal_log_prob,
            grad_log_prob=_std_normal_grad,
            joint_log_prob=self._joint,
        )
        # Unpack correctly: result = (tm, rm, tp, rp, t_prime, n_prime, s_prime, alpha, n_alpha)
        result = _build_tree(
            np.array([0.0]), np.array([0.0]),
            log_u=-5.0, v=1, j=0,
            step_size=0.1,
            log_prob=_std_normal_log_prob,
            grad_log_prob=_std_normal_grad,
            joint_log_prob=self._joint,
        )
        n_prime = result[5]
        assert n_prime == 1, f"Expected n_prime=1, got {n_prime}"

    def test_base_case_n_prime_0_when_outside_slice(self):
        """n′=0 when log_u is above the joint log-prob of the proposed point.

        We can force this by starting at a high-energy position and slicing
        at a threshold above what the leapfrog step can reach.
        """
        # Start at theta=5 (log_p≈-12.5), r=0 (kinetic=0), joint≈-12.5
        # With log_u=-5 (above -12.5 + step), n_prime should be 0.
        theta_start = np.array([5.0])
        r_start = np.array([0.0])
        joint_start = self._joint(theta_start, r_start)  # ≈ -12.5

        result = _build_tree(
            theta_start, r_start,
            log_u=joint_start + 5.0,   # slice well above starting joint
            v=1, j=0,
            step_size=0.05,
            log_prob=_std_normal_log_prob,
            grad_log_prob=_std_normal_grad,
            joint_log_prob=self._joint,
        )
        n_prime = result[5]
        assert n_prime == 0, f"Expected n_prime=0 (outside slice), got {n_prime}"

    def test_base_case_s_prime_0_on_energy_divergence(self):
        """s′=0 (stop) when energy diverges beyond delta_max."""
        # With delta_max=0 and log_u=-0.1, any step that reduces energy → s=0.
        theta_start = np.array([0.0])
        r_start = np.array([0.0])
        result = _build_tree(
            theta_start, r_start,
            log_u=0.5,     # log_u > possible joint → forces s_prime=0
            v=1, j=0,
            step_size=0.1,
            log_prob=_std_normal_log_prob,
            grad_log_prob=_std_normal_grad,
            joint_log_prob=self._joint,
            delta_max=0.0,  # any deviation triggers stop
        )
        s_prime = result[6]
        assert s_prime == 0, f"Expected s_prime=0 (diverged), got {s_prime}"

    def test_base_case_alpha_between_0_and_1(self):
        """α′ (Metropolis acceptance prob) must be in [0, 1]."""
        result = _build_tree(
            np.array([0.5]), np.array([0.3]),
            log_u=-2.0, v=1, j=0,
            step_size=0.1,
            log_prob=_std_normal_log_prob,
            grad_log_prob=_std_normal_grad,
            joint_log_prob=self._joint,
        )
        alpha_prime = result[7]
        assert 0.0 <= alpha_prime <= 1.0, f"alpha_prime={alpha_prime} out of [0,1]"

    def test_base_case_n_alpha_is_one(self):
        """n_alpha′ should always be 1 at the base case (one leaf)."""
        result = _build_tree(
            np.array([0.0]), np.array([0.0]),
            log_u=-2.0, v=1, j=0,
            step_size=0.1,
            log_prob=_std_normal_log_prob,
            grad_log_prob=_std_normal_grad,
            joint_log_prob=self._joint,
        )
        n_alpha_prime = result[8]
        assert n_alpha_prime == 1, f"Expected n_alpha_prime=1, got {n_alpha_prime}"

    def test_depth_1_n_alpha_is_two(self):
        """At depth j=1, the tree makes 2 leapfrog steps → n_alpha_prime = 2."""
        result = _build_tree(
            np.array([0.0]), np.array([0.0]),
            log_u=-10.0, v=1, j=1,
            step_size=0.1,
            log_prob=_std_normal_log_prob,
            grad_log_prob=_std_normal_grad,
            joint_log_prob=self._joint,
        )
        n_alpha_prime = result[8]
        assert n_alpha_prime == 2, (
            f"Depth-1 tree should visit 2 leaves, got n_alpha={n_alpha_prime}"
        )

    def test_backward_direction_updates_minus_endpoint(self):
        """v=-1 (backward) should update the minus endpoint, not the plus."""
        theta0 = np.array([0.0])
        r0 = np.array([1.0])

        res_fwd = _build_tree(
            theta0, r0, log_u=-5.0, v=+1, j=0, step_size=0.2,
            log_prob=_std_normal_log_prob,
            grad_log_prob=_std_normal_grad,
            joint_log_prob=self._joint,
        )
        res_bwd = _build_tree(
            theta0, r0, log_u=-5.0, v=-1, j=0, step_size=0.2,
            log_prob=_std_normal_log_prob,
            grad_log_prob=_std_normal_grad,
            joint_log_prob=self._joint,
        )
        # For v=+1 and v=-1 with same |step|: proposed theta_prime should differ
        theta_prime_fwd = res_fwd[4]
        theta_prime_bwd = res_bwd[4]
        assert not np.allclose(theta_prime_fwd, theta_prime_bwd), (
            "Forward and backward steps should produce different proposed positions"
        )


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

    def test_fd_gradient_recovers_correct_mean(self):
        """FD-based sampler must recover the correct mode, not just produce finite samples."""
        mu, sd = 2.0, 0.8
        sampler = NUTSSampler(
            _normal_log_prob(mu, sd),
            grad_log_prob_fn=None,  # uses internal FD
            seed=11,
        )
        samples = sampler.sample(np.array([0.0]), n_samples=300, n_warmup=150)
        mean = float(samples[:, 0].mean())
        assert abs(mean - mu) < 0.3, (
            f"FD sampler mean {mean:.3f} too far from true mean {mu}"
        )

    def test_fd_gradient_recovers_correct_std(self):
        """FD-based sampler recovers the correct standard deviation."""
        mu, sd = 2.0, 0.8
        sampler = NUTSSampler(
            _normal_log_prob(mu, sd),
            grad_log_prob_fn=None,
            seed=12,
        )
        samples = sampler.sample(np.array([0.0]), n_samples=300, n_warmup=150)
        std = float(samples[:, 0].std())
        assert abs(std - sd) < 0.25, (
            f"FD sampler std {std:.3f} too far from true std {sd}"
        )


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
# NUTSSampler — statistical accuracy and efficiency
# ---------------------------------------------------------------------------


class TestNUTSSamplerAccuracy:
    """
    Statistical tests verifying that NUTSSampler correctly samples from
    known target distributions.

    Each test uses a fixed seed and a sample count large enough that failures
    indicate algorithmic bugs, not sampling noise.  Tolerances are set at
    3–4 Monte Carlo standard errors.
    """

    def test_non_centred_gaussian_mean(self):
        """Sampler must recover the mode of N(3.0, 0.5²), not just N(0,1)."""
        mu, sd = 3.0, 0.5
        sampler = NUTSSampler(_normal_log_prob(mu, sd), _normal_grad(mu, sd), seed=20)
        samples = sampler.sample(np.array([mu]), n_samples=400, n_warmup=200)
        mean = float(samples[:, 0].mean())
        assert abs(mean - mu) < 0.15, f"Mean {mean:.3f} far from true {mu}"

    def test_non_centred_gaussian_std(self):
        """Sampler must recover the spread of N(3.0, 0.5²)."""
        mu, sd = 3.0, 0.5
        sampler = NUTSSampler(_normal_log_prob(mu, sd), _normal_grad(mu, sd), seed=21)
        samples = sampler.sample(np.array([mu]), n_samples=400, n_warmup=200)
        std = float(samples[:, 0].std())
        assert abs(std - sd) < 0.15, f"Std {std:.3f} far from true {sd}"

    def test_empirical_95_coverage(self):
        """90–99% of N(0,1) samples should fall within the 95% CI [-1.96, 1.96].

        True coverage = 95.4%.  With 500 samples the MC std of the fraction is
        √(0.954·0.046/500) ≈ 0.009, so [90%, 99%] is a ≥4σ band.
        """
        sampler = NUTSSampler(_std_normal_log_prob, _std_normal_grad, seed=30)
        samples = sampler.sample(np.array([0.0]), n_samples=500, n_warmup=200)
        col = samples[:, 0]
        frac = float(np.mean((col >= -1.96) & (col <= 1.96)))
        assert 0.90 <= frac <= 0.99, (
            f"95% CI coverage = {frac:.3f}, expected 0.90–0.99"
        )

    def test_seed_reproducibility(self):
        """Two NUTSSampler instances with the same seed must produce identical draws."""
        sampler_a = NUTSSampler(_std_normal_log_prob, _std_normal_grad, seed=99)
        draws_a = sampler_a.sample(np.array([0.0]), n_samples=80, n_warmup=30)

        sampler_b = NUTSSampler(_std_normal_log_prob, _std_normal_grad, seed=99)
        draws_b = sampler_b.sample(np.array([0.0]), n_samples=80, n_warmup=30)

        np.testing.assert_array_equal(
            draws_a, draws_b,
            err_msg="Identical seeds must produce identical sample arrays",
        )

    def test_different_seeds_differ(self):
        """Two samplers with different seeds must (almost certainly) produce different draws."""
        sampler_a = NUTSSampler(_std_normal_log_prob, _std_normal_grad, seed=1)
        draws_a = sampler_a.sample(np.array([0.0]), n_samples=50, n_warmup=20)

        sampler_b = NUTSSampler(_std_normal_log_prob, _std_normal_grad, seed=2)
        draws_b = sampler_b.sample(np.array([0.0]), n_samples=50, n_warmup=20)

        assert not np.array_equal(draws_a, draws_b), (
            "Different seeds should yield different samples"
        )

    def test_high_dimensional_mean(self):
        """5-D N(0, I): posterior mean should be near zero in every dimension."""
        sampler = NUTSSampler(_std_normal_log_prob, _std_normal_grad, seed=50)
        samples = sampler.sample(np.zeros(5), n_samples=500, n_warmup=250)
        assert samples.shape == (500, 5)
        mean = np.abs(samples.mean(axis=0))
        assert np.all(mean < 0.3), (
            f"5-D mean too large: {mean.tolist()}"
        )

    def test_high_dimensional_std(self):
        """5-D N(0, I): marginal standard deviations should each be near 1."""
        sampler = NUTSSampler(_std_normal_log_prob, _std_normal_grad, seed=51)
        samples = sampler.sample(np.zeros(5), n_samples=500, n_warmup=250)
        std = samples.std(axis=0)
        assert np.all((std > 0.65) & (std < 1.45)), (
            f"5-D stds out of range: {std.tolist()}"
        )

    def test_low_lag1_autocorrelation(self):
        """NUTS should produce nearly independent draws for N(0,1).

        Lag-1 autocorrelation should be well below 0.5 for a well-adapted
        sampler.  Values near 1.0 would indicate the chain is stuck.
        """
        sampler = NUTSSampler(_std_normal_log_prob, _std_normal_grad, seed=60)
        samples = sampler.sample(np.array([0.0]), n_samples=400, n_warmup=200)
        col = samples[:, 0]
        col_c = col - col.mean()
        ac1 = float(np.corrcoef(col_c[:-1], col_c[1:])[0, 1])
        assert ac1 < 0.5, (
            f"Lag-1 autocorrelation {ac1:.3f} is too high — chain may be stuck"
        )

    def test_quantiles_match_normal(self):
        """Empirical quantiles of N(0,1) samples should match theoretical values."""
        sampler = NUTSSampler(_std_normal_log_prob, _std_normal_grad, seed=70)
        samples = sampler.sample(np.array([0.0]), n_samples=600, n_warmup=300)
        col = samples[:, 0]
        # 10th and 90th percentile of N(0,1) are ≈ -1.28 and +1.28
        p10 = float(np.percentile(col, 10))
        p90 = float(np.percentile(col, 90))
        assert abs(p10 - (-1.28)) < 0.3, f"10th percentile {p10:.3f} ≠ -1.28"
        assert abs(p90 - (+1.28)) < 0.3, f"90th percentile {p90:.3f} ≠ +1.28"

    def test_max_tree_depth_respected(self):
        """With max_tree_depth=1, no sample should require more than 2 leapfrog steps.

        We verify this indirectly: sampling still completes and produces finite
        results even with a very shallow tree.
        """
        sampler = NUTSSampler(
            _std_normal_log_prob, _std_normal_grad,
            max_tree_depth=1, seed=80,
        )
        samples = sampler.sample(np.array([0.0]), n_samples=100, n_warmup=50)
        assert samples.shape == (100, 1)
        assert np.all(np.isfinite(samples)), "Samples must be finite with max_tree_depth=1"


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

    def test_ess_meaningful_for_simple_gaussian(self):
        """For N(0,1) with 200 draws, ESS should be at least 20% of n_samples.

        NUTS produces nearly independent samples, so ESS ≥ n_samples * 0.2
        is a conservative lower bound even with serial NumPy RNG overhead.
        An ESS of 1 would indicate the chain never moved.
        """
        n_samples = 200
        result = nuts_estimate(
            _std_normal_log_prob,
            np.array([0.0]),
            n_samples=n_samples,
            n_warmup=100,
            seed=9,
        )
        ess = int(result["n_effective"][0])
        assert ess >= n_samples // 5, (
            f"ESS={ess} is suspiciously low for N(0,1) with {n_samples} draws "
            f"(expected ≥ {n_samples // 5})"
        )

    def test_samples_accuracy(self):
        """nuts_estimate samples must approximate the correct distribution."""
        result = nuts_estimate(
            _normal_log_prob(2.5, 0.6),
            np.array([0.0]),
            n_samples=300,
            n_warmup=150,
            seed=10,
        )
        col = result["samples"][:, 0]
        assert abs(col.mean() - 2.5) < 0.25, (
            f"nuts_estimate mean {col.mean():.3f} far from 2.5"
        )
        assert abs(col.std() - 0.6) < 0.2, (
            f"nuts_estimate std {col.std():.3f} far from 0.6"
        )

    def test_samples_shape_matches_request(self):
        """nuts_estimate must return exactly n_samples rows."""
        result = nuts_estimate(
            _std_normal_log_prob,
            np.array([0.0, 0.0, 0.0]),
            n_samples=150,
            n_warmup=75,
            seed=6,
        )
        assert result["samples"].shape == (150, 3)

    def test_all_samples_finite(self):
        """nuts_estimate must not produce NaN or Inf in samples."""
        result = nuts_estimate(
            _std_normal_log_prob,
            np.array([0.0]),
            n_samples=100,
            n_warmup=50,
            seed=7,
        )
        assert np.all(np.isfinite(result["samples"])), "nuts_estimate produced non-finite samples"
