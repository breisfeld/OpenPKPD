"""Exact numerical tests for the sandwich covariance estimator."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.covariance.sandwich import SandwichCovariance, _build_param_names
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec


class _QuadraticIndividual:
    def __init__(self, center: np.ndarray, weight: np.ndarray) -> None:
        self.center = np.asarray(center, dtype=float)
        self.weight = np.asarray(weight, dtype=float)

    def obj_eta(
        self,
        eta: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> float:
        diff = np.asarray(theta, dtype=float) - self.center
        return float(diff @ self.weight @ diff)


class _QuadraticPopulation:
    def __init__(self, individuals: list[_QuadraticIndividual]) -> None:
        self._individuals = {i + 1: indiv for i, indiv in enumerate(individuals)}
        self.trans = 1

    def subject_ids(self) -> list[int]:
        return sorted(self._individuals)

    def individual_model(self, subject_id: int) -> _QuadraticIndividual:
        return self._individuals[subject_id]


def _make_theta_only_params(theta: np.ndarray) -> ParameterSet:
    return ParameterSet.from_specs(
        theta_specs=[
            ThetaSpec(init=float(theta[0]), lower=-float("inf"), label="CL"),
            ThetaSpec(init=float(theta[1]), lower=-float("inf"), label="V"),
        ],
        omega_specs=[],
        sigma_specs=[],
    )


@pytest.mark.unit
@pytest.mark.parametrize("matrix_kind", ["S", "R", "SR"])
def test_sandwich_matches_closed_form_quadratic_case(matrix_kind: str):
    theta = np.array([0.4, -0.2])
    center_1 = np.array([0.1, -0.3])
    center_2 = np.array([-0.2, 0.5])
    weight_1 = np.array([[2.0, 0.3], [0.3, 1.2]])
    weight_2 = np.array([[1.5, 0.2], [0.2, 0.9]])

    params = _make_theta_only_params(theta)
    population = _QuadraticPopulation(
        [
            _QuadraticIndividual(center_1, weight_1),
            _QuadraticIndividual(center_2, weight_2),
        ]
    )

    grad_1 = 2.0 * weight_1 @ (theta - center_1)
    grad_2 = 2.0 * weight_2 @ (theta - center_2)
    r_expected = 2.0 * (weight_1 + weight_2)
    s_expected = np.outer(grad_1, grad_1) + np.outer(grad_2, grad_2)
    cov_expected = np.linalg.inv(r_expected)

    if matrix_kind == "S":
        cov_expected = s_expected
    elif matrix_kind == "R":
        cov_expected = cov_expected
    else:
        cov_expected = cov_expected @ s_expected @ cov_expected

    estimator = SandwichCovariance(eps=1e-6, matrix=matrix_kind)
    result = estimator.compute(population, params, eta_hat={})

    np.testing.assert_allclose(result.r_matrix, r_expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(result.s_matrix, s_expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(result.cov_matrix, cov_expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(result.se, np.sqrt(np.diag(cov_expected)), rtol=1e-4)

    se_outer = np.outer(np.sqrt(np.diag(cov_expected)), np.sqrt(np.diag(cov_expected)))
    cor_expected = np.where(se_outer > 0, cov_expected / se_outer, 0.0)
    np.fill_diagonal(cor_expected, 1.0)
    np.testing.assert_allclose(result.cor_matrix, cor_expected, rtol=1e-4, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sandwich_sr_cov_is_symmetric(seed: int) -> None:
    """SR covariance matrix must be symmetric to 1e-10."""
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(2)
    individuals = []
    for _ in range(6):
        c = rng.standard_normal(2)
        raw_W = rng.standard_normal((2, 2))
        W = raw_W.T @ raw_W + np.eye(2) * 0.1
        individuals.append(_QuadraticIndividual(c, W))
    pop = _QuadraticPopulation(individuals)
    params = _make_theta_only_params(theta)
    result = SandwichCovariance(eps=1e-6, matrix="SR").compute(pop, params, eta_hat={})
    np.testing.assert_allclose(result.cov_matrix, result.cov_matrix.T, atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("seed", [4, 5, 6])
def test_sandwich_sr_cov_is_psd(seed: int) -> None:
    """SR covariance matrix must have all eigenvalues ≥ 0."""
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(2)
    individuals = []
    for _ in range(8):
        c = rng.standard_normal(2)
        raw_W = rng.standard_normal((2, 2))
        W = raw_W.T @ raw_W + np.eye(2) * 0.2
        individuals.append(_QuadraticIndividual(c, W))
    pop = _QuadraticPopulation(individuals)
    params = _make_theta_only_params(theta)
    result = SandwichCovariance(eps=1e-6, matrix="SR").compute(pop, params, eta_hat={})
    eigvals = np.linalg.eigvalsh(result.cov_matrix)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min():.2e}"


@pytest.mark.unit
def test_sandwich_se_equals_sqrt_of_diagonal() -> None:
    """se must equal sqrt(diag(cov_matrix)) to machine precision."""
    theta = np.array([0.4, -0.2])
    centers = [np.array([0.1, -0.3]), np.array([-0.2, 0.5])]
    weight = np.array([[2.0, 0.3], [0.3, 1.2]])
    pop = _QuadraticPopulation([_QuadraticIndividual(c, weight) for c in centers])
    params = _make_theta_only_params(theta)
    result = SandwichCovariance(eps=1e-6, matrix="SR").compute(pop, params, eta_hat={})
    expected_se = np.sqrt(np.diag(result.cov_matrix))
    np.testing.assert_allclose(result.se, expected_se, rtol=1e-10)


@pytest.mark.unit
def test_sandwich_cor_diagonal_is_one() -> None:
    """Diagonal of correlation matrix must be exactly 1.0."""
    theta = np.array([1.0, 2.0])
    W = np.array([[4.0, 0.6], [0.6, 3.0]])
    centers = [np.array([0.5, 1.0]), np.array([-0.5, 3.0]), np.array([1.5, 2.5])]
    pop = _QuadraticPopulation([_QuadraticIndividual(c, W) for c in centers])
    params = _make_theta_only_params(theta)
    result = SandwichCovariance(eps=1e-6, matrix="SR").compute(pop, params, eta_hat={})
    np.testing.assert_allclose(np.diag(result.cor_matrix), 1.0, atol=1e-10)


@pytest.mark.unit
def test_build_param_names_uses_theta_labels_and_lower_triangle_order():
    params = ParameterSet.from_specs(
        theta_specs=[
            ThetaSpec(init=1.0, lower=0.0, label="CL"),
            ThetaSpec(init=20.0, lower=0.0, fixed=True, label="V"),
        ],
        omega_specs=[OmegaSpec(block_size=2, values=[0.1, 0.02, 0.3])],
        sigma_specs=[SigmaSpec(block_size=1, values=[0.4])],
    )

    assert _build_param_names(params) == [
        "CL",
        "OMEGA(1,1)",
        "OMEGA(2,1)",
        "OMEGA(2,2)",
        "SIGMA(1,1)",
    ]
