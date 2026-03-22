"""
External-validation tests for the sandwich covariance estimator.

Validates against the analytic HC0 heteroscedasticity-consistent covariance
formula (White 1982):

    Ĉ_sandwich = R⁻¹ · S · R⁻¹
    R_ij = Σ_subj  ∂²f_subj/∂θ_i∂θ_j   (Hessian of OFV)
    S     = Σ_subj  (∂f_subj/∂θ)(∂f_subj/∂θ)ᵀ  (outer-product of gradients)

For a quadratic objective f_i(θ) = (θ − cᵢ)ᵀ Wᵢ (θ − cᵢ), the analytic
R and S are computable in closed form and serve as an independent reference.

References
----------
White H (1982). Maximum likelihood estimation of misspecified models.
Econometrica 50:1-25.
Efron B, Hinkley DV (1978). Assessing the accuracy of the MLE. Biometrika.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.covariance.sandwich import SandwichCovariance
from openpkpd.model.parameters import ParameterSet, ThetaSpec

# ---------------------------------------------------------------------------
# Mock infrastructure (mirrors tests/unit/covariance/test_sandwich.py)
# ---------------------------------------------------------------------------


class _QuadraticIndividual:
    def __init__(self, center, weight) -> None:
        self.center = np.asarray(center, dtype=float)
        self.weight = np.asarray(weight, dtype=float)

    def obj_eta(self, eta, theta, omega, sigma, trans=2) -> float:
        d = np.asarray(theta, dtype=float) - self.center
        return float(d @ self.weight @ d)


class _QuadraticPopulation:
    trans = 1

    def __init__(self, individuals) -> None:
        self._i = {i + 1: ind for i, ind in enumerate(individuals)}

    def subject_ids(self):
        return sorted(self._i)

    def individual_model(self, sid):
        return self._i[sid]


def _theta_params(theta: np.ndarray, labels=None) -> ParameterSet:
    labels = labels or [f"P{i}" for i in range(len(theta))]
    return ParameterSet.from_specs(
        theta_specs=[
            ThetaSpec(init=float(t), lower=-float("inf"), label=lbl)
            for t, lbl in zip(theta, labels, strict=False)
        ],
        omega_specs=[],
        sigma_specs=[],
    )


# ---------------------------------------------------------------------------
# Symmetry and PSD properties (required by any valid covariance matrix)
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestCovarianceStructuralProperties:
    """
    A valid covariance matrix must be symmetric and positive semi-definite.
    These tests verify that SandwichCovariance satisfies both properties
    for three different model configurations.
    """

    def _run_estimator(self, n_subjects, n_params, seed):
        rng = np.random.default_rng(seed)
        theta = rng.standard_normal(n_params)
        individuals = []
        for _ in range(n_subjects):
            center = rng.standard_normal(n_params)
            raw_W = rng.standard_normal((n_params, n_params))
            weight = raw_W.T @ raw_W + np.eye(n_params) * 0.1  # PD
            individuals.append(_QuadraticIndividual(center, weight))
        pop = _QuadraticPopulation(individuals)
        params = _theta_params(theta)
        result = SandwichCovariance(matrix="SR").compute(pop, params, eta_hat={})
        return result

    @pytest.mark.parametrize(
        "n_subjects,n_params,seed",
        [
            (5, 2, 0),
            (10, 3, 1),
            (20, 2, 2),
        ],
    )
    def test_cov_matrix_is_symmetric(self, n_subjects, n_params, seed):
        """Ĉ must be symmetric to 1e-10."""
        result = self._run_estimator(n_subjects, n_params, seed)
        np.testing.assert_allclose(result.cov_matrix, result.cov_matrix.T, atol=1e-10)

    @pytest.mark.parametrize(
        "n_subjects,n_params,seed",
        [
            (5, 2, 3),
            (10, 3, 4),
            (15, 2, 5),
        ],
    )
    def test_cov_matrix_is_psd(self, n_subjects, n_params, seed):
        """All eigenvalues of Ĉ must be ≥ 0."""
        result = self._run_estimator(n_subjects, n_params, seed)
        eigvals = np.linalg.eigvalsh(result.cov_matrix)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min():.2e}"

    @pytest.mark.parametrize(
        "n_subjects,n_params,seed",
        [
            (6, 2, 6),
            (8, 3, 7),
        ],
    )
    def test_se_equals_sqrt_diagonal(self, n_subjects, n_params, seed):
        """se must equal sqrt(diag(cov_matrix)) exactly."""
        result = self._run_estimator(n_subjects, n_params, seed)
        expected_se = np.sqrt(np.diag(result.cov_matrix))
        np.testing.assert_allclose(result.se, expected_se, rtol=1e-10)

    @pytest.mark.parametrize(
        "n_subjects,n_params,seed",
        [
            (5, 2, 8),
            (10, 3, 9),
        ],
    )
    def test_cor_diagonal_is_one(self, n_subjects, n_params, seed):
        """Diagonal of correlation matrix must be exactly 1.0."""
        result = self._run_estimator(n_subjects, n_params, seed)
        np.testing.assert_allclose(np.diag(result.cor_matrix), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# HC0 analytic reference for a specific linear-regression-like problem
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestSandwichHC0Reference:
    """
    For a quadratic OFV with known weights Wᵢ and centres cᵢ, the analytic
    sandwich covariance is:

        R = Σ 2Wᵢ
        gᵢ = 2Wᵢ(θ − cᵢ)
        S = Σ gᵢ gᵢᵀ
        Ĉ_SR = R⁻¹ S R⁻¹

    This matches the HC0 formula from White (1982) applied to OLS residuals.
    """

    def test_r_matrix_matches_analytic_hessian(self):
        """R = Σ 2Wᵢ (Hessian of total OFV for quadratic objective)."""
        theta = np.array([0.4, -0.2])
        centers = [np.array([0.1, -0.3]), np.array([-0.2, 0.5])]
        weights = [
            np.array([[2.0, 0.3], [0.3, 1.2]]),
            np.array([[1.5, 0.2], [0.2, 0.9]]),
        ]
        pop = _QuadraticPopulation(
            [_QuadraticIndividual(c, w) for c, w in zip(centers, weights, strict=False)]
        )
        params = _theta_params(theta, labels=["CL", "V"])
        result = SandwichCovariance(matrix="SR").compute(pop, params, eta_hat={})

        R_ref = sum(2.0 * w for w in weights)
        np.testing.assert_allclose(result.r_matrix, R_ref, rtol=1e-5, atol=1e-7)

    def test_s_matrix_matches_outer_product_of_gradients(self):
        """S = Σ gᵢgᵢᵀ where gᵢ = 2Wᵢ(θ−cᵢ)."""
        theta = np.array([1.0, 0.5])
        centers = [np.array([0.3, 0.1]), np.array([0.8, -0.4]), np.array([-0.2, 0.7])]
        weights = [
            np.array([[3.0, 0.5], [0.5, 2.0]]),
            np.array([[1.0, 0.1], [0.1, 1.5]]),
            np.array([[2.5, 0.3], [0.3, 0.8]]),
        ]
        pop = _QuadraticPopulation(
            [_QuadraticIndividual(c, w) for c, w in zip(centers, weights, strict=False)]
        )
        params = _theta_params(theta)
        result = SandwichCovariance(matrix="SR").compute(pop, params, eta_hat={})

        grads = [2.0 * w @ (theta - c) for c, w in zip(centers, weights, strict=False)]
        S_ref = sum(np.outer(g, g) for g in grads)
        np.testing.assert_allclose(result.s_matrix, S_ref, rtol=1e-5, atol=1e-7)

    def test_sandwich_matches_rinv_s_rinv(self):
        """Ĉ_SR = R⁻¹ · S · R⁻¹ exactly."""
        theta = np.array([0.7, -0.3])
        centers = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([-0.3, 0.3]),
            np.array([0.2, -0.2]),
        ]
        W = np.array([[2.0, 0.4], [0.4, 1.8]])
        pop = _QuadraticPopulation([_QuadraticIndividual(c, W) for c in centers])
        params = _theta_params(theta, labels=["CL", "V"])
        result = SandwichCovariance(matrix="SR").compute(pop, params, eta_hat={})

        R = sum(2.0 * W for _ in centers)
        grads = [2.0 * W @ (theta - c) for c in centers]
        S = sum(np.outer(g, g) for g in grads)
        R_inv = np.linalg.inv(R)
        C_ref = R_inv @ S @ R_inv

        np.testing.assert_allclose(result.cov_matrix, C_ref, rtol=1e-5, atol=1e-7)

    def test_r_only_matches_inverse_hessian(self):
        """matrix='R': Ĉ_R = R⁻¹ (inverse Hessian)."""
        theta = np.array([1.0, 2.0])
        W = np.array([[4.0, 0.6], [0.6, 3.0]])
        centers = [np.array([0.5, 1.0]), np.array([1.5, 3.0])]
        pop = _QuadraticPopulation([_QuadraticIndividual(c, W) for c in centers])
        params = _theta_params(theta)
        result = SandwichCovariance(matrix="R").compute(pop, params, eta_hat={})

        R_ref = sum(2.0 * W for _ in centers)
        np.testing.assert_allclose(result.cov_matrix, np.linalg.inv(R_ref), rtol=1e-5, atol=1e-7)

    def test_scale_invariance(self):
        """
        The SR sandwich estimator Ĉ = R⁻¹SR⁻¹ is scale-invariant.

        Scaling all objectives by k: R→k·R, S→k²·S.
        Ĉ_scaled = (kR)⁻¹(k²S)(kR)⁻¹ = k⁻¹·k²·k⁻¹ · R⁻¹SR⁻¹ = R⁻¹SR⁻¹ = Ĉ_base.

        Separately, R-only Ĉ = R⁻¹ → (kR)⁻¹ = k⁻¹·R⁻¹, so R-only scales as 1/k.
        """
        theta = np.array([0.3, 0.8])
        centers = [np.array([0.0, 0.5]), np.array([0.6, 0.2]), np.array([-0.4, 1.0])]
        W_base = np.array([[2.0, 0.3], [0.3, 1.5]])
        k = 3.0

        class _ScaledIndividual:
            def __init__(self, center, weight, scale):
                self.center = np.asarray(center, dtype=float)
                self.weight = np.asarray(weight, dtype=float) * scale

            def obj_eta(self, eta, theta, omega, sigma, trans=2):
                d = np.asarray(theta, dtype=float) - self.center
                return float(d @ self.weight @ d)

        pop_base = _QuadraticPopulation([_QuadraticIndividual(c, W_base) for c in centers])
        pop_scaled = _QuadraticPopulation([_ScaledIndividual(c, W_base, k) for c in centers])
        params = _theta_params(theta)

        res_base = SandwichCovariance(matrix="SR").compute(pop_base, params, {})
        res_scaled = SandwichCovariance(matrix="SR").compute(pop_scaled, params, {})

        # SR is scale-invariant: Ĉ_SR does not change with scaling
        np.testing.assert_allclose(res_scaled.cov_matrix, res_base.cov_matrix, rtol=1e-5, atol=1e-8)

        # R-only Ĉ_R = R⁻¹ → scales as 1/k
        res_base_r = SandwichCovariance(matrix="R").compute(pop_base, params, {})
        res_scaled_r = SandwichCovariance(matrix="R").compute(pop_scaled, params, {})
        np.testing.assert_allclose(
            res_scaled_r.cov_matrix, res_base_r.cov_matrix / k, rtol=1e-5, atol=1e-8
        )
