"""
Unit tests for Nonparametric estimation (NonparametricMethod, NonparametricResult).

Tests cover:
  - Class construction and method_name
  - NonparametricResult dataclass fields and derived statistics
  - Weight optimisation correctness (weights sum to 1, all non-negative)
  - Likelihood matrix computation shape
  - get_estimation_method routing
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.estimation.nonparametric import NonparametricMethod, NonparametricResult

# ---------------------------------------------------------------------------
# NonparametricMethod construction
# ---------------------------------------------------------------------------


def test_nonparametric_method_name() -> None:
    """method_name must be 'NONPARAMETRIC'."""
    method = NonparametricMethod()
    assert method.method_name == "NONPARAMETRIC"


def test_nonparametric_method_creation() -> None:
    """Default construction should set sensible attribute values."""
    method = NonparametricMethod(base_method="FOCE")
    assert method.base_method == "FOCE"
    assert method.n_support_points is None
    assert method.max_iter == 100
    assert method.tol == pytest.approx(1e-5)


def test_nonparametric_method_custom_params() -> None:
    """Custom arguments should be stored correctly."""
    method = NonparametricMethod(
        base_method="FO",
        n_support_points=20,
        max_iter=50,
        tol=1e-4,
    )
    assert method.base_method == "FO"
    assert method.n_support_points == 20
    assert method.max_iter == 50
    assert method.tol == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# NonparametricResult construction
# ---------------------------------------------------------------------------


def test_nonparametric_result_fields() -> None:
    """NonparametricResult should store support_points and support_weights."""
    result = NonparametricResult(
        theta_final=np.array([1.0]),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=100.0,
        support_points=np.zeros((5, 1)),
        support_weights=np.ones(5) / 5,
    )
    assert result.support_points.shape == (5, 1)
    assert result.support_weights.shape == (5,)
    assert np.isclose(result.support_weights.sum(), 1.0)


def test_nonparametric_result_defaults() -> None:
    """Default NonparametricResult should have empty arrays for support fields."""
    result = NonparametricResult(
        theta_final=np.array([1.0, 2.0]),
        omega_final=np.eye(2),
        sigma_final=np.eye(1),
        ofv=200.0,
    )
    assert result.support_points.shape == (0,)
    assert result.support_weights.shape == (0,)


# ---------------------------------------------------------------------------
# Derived statistics
# ---------------------------------------------------------------------------


def test_empirical_mean_uniform_weights() -> None:
    """With uniform weights, empirical mean = arithmetic mean of support points."""
    sp = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    w = np.ones(3) / 3
    result = NonparametricResult(
        theta_final=np.zeros(2),
        omega_final=np.eye(2),
        sigma_final=np.eye(1),
        ofv=0.0,
        support_points=sp,
        support_weights=w,
    )
    mean = result.empirical_mean()
    expected = np.array([2.0, 3.0])  # arithmetic mean of [0,2,4] and [1,3,5]
    np.testing.assert_allclose(mean, expected, atol=1e-10)


def test_empirical_variance_all_same_point() -> None:
    """When all support points are identical, variance must be zero."""
    sp = np.array([[1.0], [1.0], [1.0]])
    w = np.array([0.3, 0.4, 0.3])
    result = NonparametricResult(
        theta_final=np.ones(1),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=0.0,
        support_points=sp,
        support_weights=w,
    )
    var = result.empirical_variance()
    np.testing.assert_allclose(var, np.zeros(1), atol=1e-12)


def test_empirical_variance_two_points() -> None:
    """Known two-point distribution: variance = p*(1-p)*gap^2."""
    # Support: ETA in {-1, +1} with equal weights
    sp = np.array([[-1.0], [1.0]])
    w = np.array([0.5, 0.5])
    result = NonparametricResult(
        theta_final=np.zeros(1),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=0.0,
        support_points=sp,
        support_weights=w,
    )
    var = result.empirical_variance()
    # mean = 0, var = 0.5*((-1-0)^2) + 0.5*((1-0)^2) = 1.0
    np.testing.assert_allclose(var, np.array([1.0]), atol=1e-10)


def test_empirical_mean_empty() -> None:
    """empirical_mean() with no support points returns empty array."""
    result = NonparametricResult(
        theta_final=np.array([1.0]),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=0.0,
    )
    mean = result.empirical_mean()
    assert mean.shape == (0,)


# ---------------------------------------------------------------------------
# _optimize_weights (internal)
# ---------------------------------------------------------------------------


class _LikelihoodSubjectEvents:
    def __init__(self, dv: float) -> None:
        self.obs_dv = np.array([dv], dtype=float)

    def observation_mask(self):
        return np.array([True])


class _LikelihoodIndividualModel:
    def __init__(self, dv: float) -> None:
        self.subject_events = _LikelihoodSubjectEvents(dv)

    def evaluate(self, theta, eta, sigma, trans=None):
        pred = np.array([float(theta[0]) * np.exp(float(eta[0]))])
        return pred, None, pred


class _LikelihoodPopulationModel:
    trans = None

    def __init__(self, dvs: list[float]) -> None:
        self._indivs = {sid: _LikelihoodIndividualModel(dv) for sid, dv in enumerate(dvs, start=1)}

    def subject_ids(self):
        return list(self._indivs.keys())

    def individual_model(self, sid):
        return self._indivs[sid]


class _LikelihoodParams:
    theta = np.array([1.0])
    omega = np.eye(1) * 0.1
    sigma = np.eye(1) * 0.01


def _likelihood_base_result() -> EstimationResult:
    return EstimationResult(
        theta_final=np.array([1.0]),
        omega_final=np.eye(1) * 0.1,
        sigma_final=np.eye(1) * 0.01,
        ofv=0.0,
    )


def test_optimize_weights_sum_to_one() -> None:
    """EM optimised weights must always sum to 1.0."""
    # Create a mock population model and parameter set for the weight optimiser
    # We test using a simple stub to avoid needing a full population model.

    method = NonparametricMethod(max_iter=50)

    # Build a minimal stub population model
    class StubIndiv:
        def __init__(self, dv: np.ndarray) -> None:
            self.dv = dv

        def evaluate(self, theta, eta, sigma, trans=None):
            # Return (ipred, _, f) where f = dv (perfect fit at one support point)
            return self.dv, None, self.dv

        @property
        def subject_events(self):
            dv_ref = self.dv  # capture in closure before entering SE scope

            class SE:
                obs_dv = dv_ref
                obs_times = np.arange(len(dv_ref), dtype=float)

                def observation_mask(self_se):
                    return np.ones(len(dv_ref), dtype=bool)

            return SE()

    class StubModel:
        trans = None

        def subject_ids(self):
            return [1, 2, 3, 4, 5]

        def individual_model(self, sid):
            return StubIndiv(np.array([5.0, 3.0, 1.0]))

    # Minimal ParameterSet stub
    class StubParams:
        theta = np.array([1.0])
        omega = np.eye(1) * 0.1
        sigma = np.eye(1) * 1.0

    stub_model = StubModel()
    stub_params = StubParams()

    # Minimal EstimationResult stub
    from openpkpd.estimation.base import EstimationResult

    base_result = EstimationResult(
        theta_final=np.array([1.0]),
        omega_final=np.eye(1) * 0.1,
        sigma_final=np.eye(1) * 1.0,
        ofv=100.0,
    )

    # 3 support points, 1 ETA
    support_points = np.array([[0.0], [0.5], [-0.5]])
    weights = method._optimize_weights(
        support_points=support_points,
        population_model=stub_model,
        init_params=stub_params,
        base_result=base_result,
    )

    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6), f"weights.sum() = {weights.sum()}"
    assert np.all(weights >= 0), "All weights must be non-negative"


def test_compute_likelihood_matrix_prefers_matching_support_points() -> None:
    """Likelihood matrix should favour the support point matching each subject."""
    method = NonparametricMethod(max_iter=20)
    support_points = np.array([[0.0], [np.log(2.0)]])
    pop_model = _LikelihoodPopulationModel([1.0, 2.0])

    L = method._compute_likelihood_matrix(
        support_points=support_points,
        population_model=pop_model,
        init_params=_LikelihoodParams(),
        base_result=_likelihood_base_result(),
        subject_ids=pop_model.subject_ids(),
    )

    assert L.shape == (2, 2)
    assert L[0, 0] > L[0, 1]
    assert L[1, 1] > L[1, 0]


def test_compute_likelihood_matrix_parallel_matches_serial() -> None:
    support_points = np.array([[0.0], [np.log(2.0)]])
    pop_model = _LikelihoodPopulationModel([1.0, 2.0, 1.0, 2.0])

    serial = NonparametricMethod(max_iter=20, n_parallel=1)._compute_likelihood_matrix(
        support_points=support_points,
        population_model=pop_model,
        init_params=_LikelihoodParams(),
        base_result=_likelihood_base_result(),
        subject_ids=pop_model.subject_ids(),
    )
    parallel = NonparametricMethod(max_iter=20, n_parallel=2)._compute_likelihood_matrix(
        support_points=support_points,
        population_model=pop_model,
        init_params=_LikelihoodParams(),
        base_result=_likelihood_base_result(),
        subject_ids=pop_model.subject_ids(),
    )

    np.testing.assert_allclose(parallel, serial, atol=1e-12)


def test_optimize_weights_recovers_symmetric_two_support_distribution() -> None:
    """Symmetric exact matches should recover approximately equal weights."""
    method = NonparametricMethod(max_iter=50, tol=1e-10)
    support_points = np.array([[0.0], [np.log(2.0)]])
    pop_model = _LikelihoodPopulationModel([1.0, 2.0])

    weights = method._optimize_weights(
        support_points=support_points,
        population_model=pop_model,
        init_params=_LikelihoodParams(),
        base_result=_likelihood_base_result(),
    )

    np.testing.assert_allclose(weights, np.array([0.5, 0.5]), atol=5e-2)


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_returns_string() -> None:
    """summary() must return a non-empty string."""
    result = NonparametricResult(
        theta_final=np.array([1.0, 2.0]),
        omega_final=np.eye(2),
        sigma_final=np.eye(1),
        ofv=150.0,
        method="NONPARAMETRIC",
        support_points=np.array([[0.1, -0.1], [0.2, 0.3]]),
        support_weights=np.array([0.6, 0.4]),
    )
    s = result.summary()
    assert isinstance(s, str)
    assert "NONPARAMETRIC" in s
    assert "n_support_points" in s


# ---------------------------------------------------------------------------
# get_estimation_method routing
# ---------------------------------------------------------------------------


def test_get_estimation_method_nonparametric() -> None:
    """get_estimation_method('NONPARAMETRIC') must return NonparametricMethod."""
    from openpkpd.estimation import get_estimation_method

    m = get_estimation_method("NONPARAMETRIC")
    assert isinstance(m, NonparametricMethod)
    assert m.method_name == "NONPARAMETRIC"


def test_get_estimation_method_nonparm_alias() -> None:
    """'NONPARM' should also route to NonparametricMethod."""
    from openpkpd.estimation import get_estimation_method

    m = get_estimation_method("NONPARM")
    assert isinstance(m, NonparametricMethod)


def test_get_estimation_method_np_alias() -> None:
    """'NP' should also route to NonparametricMethod."""
    from openpkpd.estimation import get_estimation_method

    m = get_estimation_method("NP")
    assert isinstance(m, NonparametricMethod)


def test_get_estimation_method_nonparametric_kwargs() -> None:
    """kwargs should be forwarded to NonparametricMethod."""
    from openpkpd.estimation import get_estimation_method

    m = get_estimation_method("NONPARAMETRIC", base_method="FO", max_iter=200)
    assert isinstance(m, NonparametricMethod)
    assert m.base_method == "FO"
    assert m.max_iter == 200
