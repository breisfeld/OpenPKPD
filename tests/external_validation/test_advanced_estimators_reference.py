"""External-validation tests for advanced estimators using exact analytic references."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.estimation.bayes import BAYESMethod
from openpkpd.estimation.nonparametric import NonparametricMethod
from openpkpd.estimation.nuts import NUTSSampler
from openpkpd.model.parameters import ParameterSet


@pytest.mark.external_validation
class TestNUTSAgainstExactGaussian:
    """NUTS samples should recover the exact mean/covariance of a Gaussian target."""

    def test_bivariate_gaussian_mean_and_covariance(self):
        cov = np.array([[1.0, 0.6], [0.6, 2.0]])
        precision = np.linalg.inv(cov)

        def log_prob(theta: np.ndarray) -> float:
            return float(-0.5 * theta @ precision @ theta)

        def grad(theta: np.ndarray) -> np.ndarray:
            return -(precision @ theta)

        samples = NUTSSampler(log_prob, grad, delta=0.7, seed=321).sample(
            np.zeros(2), n_samples=800, n_warmup=400
        )

        np.testing.assert_allclose(samples.mean(axis=0), np.zeros(2), atol=0.15)
        np.testing.assert_allclose(np.cov(samples.T), cov, atol=0.20)


@pytest.mark.external_validation
def test_bayes_laplace_matches_exact_quadratic_posterior(monkeypatch: pytest.MonkeyPatch) -> None:
    """BAYES(Laplace) should recover the exact Gaussian posterior of a quadratic OFV."""
    import openpkpd.estimation.foce as foce_module

    theta_map = np.array([2.0])
    target_var = 9.0

    class _DummyFOCE:
        def __init__(self) -> None:
            self._current_eta_hat = {}

        def estimate(self, population_model, init_params):
            return EstimationResult(
                theta_final=theta_map.copy(),
                omega_final=np.eye(1),
                sigma_final=np.eye(1),
                ofv=7.0,
                converged=True,
                post_hoc_etas={},
                ofv_history=[7.0],
                n_function_evals=1,
                elapsed_time=0.0,
                method="FOCE",
                message="quadratic mock",
            )

        def _inner_loop(self, population_model, params):
            return {}

        def _outer_ofv(self, population_model, params, eta_hat):
            diff = float(params.theta[0] - theta_map[0])
            return 7.0 + diff**2 / target_var

    monkeypatch.setattr(foce_module, "FOCEMethod", _DummyFOCE)

    result = BAYESMethod(backend="laplace", n_samples=4000, seed=123, prior_sd_theta=1e8).estimate(
        population_model=None,
        init_params=ParameterSet(theta=theta_map.copy(), omega=np.eye(1), sigma=np.eye(1)),
    )

    theta_samples = result.posterior_samples["theta"][:, 0]
    assert result.backend_used == "laplace"
    assert float(theta_samples.mean()) == pytest.approx(theta_map[0], abs=0.10)
    assert float(theta_samples.var(ddof=1)) == pytest.approx(target_var, rel=0.10)


@pytest.mark.external_validation
def test_nonparametric_weight_em_matches_exact_two_support_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NPML weight EM should recover the exact optimum for a two-subject likelihood matrix."""

    class _DummyPopulation:
        def subject_ids(self):
            return [1, 2]

    likelihood_matrix = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    expected_w1 = 0.58 / 0.96  # solve d/dw Σ log(w L_i1 + (1-w) L_i2) = 0

    method = NonparametricMethod(max_iter=200, tol=1e-12)
    monkeypatch.setattr(
        method,
        "_compute_likelihood_matrix",
        lambda support_points, population_model, init_params, base_result, subject_ids: (
            likelihood_matrix
        ),
    )

    weights = method._optimize_weights(
        support_points=np.array([[0.0], [1.0]]),
        population_model=_DummyPopulation(),
        init_params=None,
        base_result=EstimationResult(
            theta_final=np.array([1.0]),
            omega_final=np.eye(1),
            sigma_final=np.eye(1),
            ofv=0.0,
        ),
    )

    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(weights, [expected_w1, 1.0 - expected_w1], atol=1e-6)
