"""
Unit tests for Population Fisher Information Matrix (PFIM) and optimal design.

Tests cover:
  - PFIMEngine instantiation (including with None model/params for basic use)
  - DesignResult dataclass construction and summary()
  - FIM symmetry and positive semi-definiteness
  - Criterion helpers (_a_criterion, _condition_number, _se_from_fim)
  - efficiency() function
  - D/A/E criterion_value helpers
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.design.pfim import DesignResult, PFIMEngine

# ---------------------------------------------------------------------------
# PFIMEngine instantiation
# ---------------------------------------------------------------------------


def test_pfim_instantiation_none() -> None:
    """PFIMEngine should instantiate successfully with None model and params."""
    engine = PFIMEngine(population_model=None, init_params=None)
    assert engine is not None
    assert engine.population_model is None
    assert engine.init_params is None


def test_pfim_instantiation_with_times() -> None:
    """PFIMEngine should store provided sampling_times."""
    times = np.array([1.0, 2.0, 4.0, 8.0])
    engine = PFIMEngine(population_model=None, init_params=None, sampling_times=times)
    np.testing.assert_array_equal(engine.sampling_times, times)


def test_pfim_compute_fim_requires_model() -> None:
    """compute_fim must raise RuntimeError when population_model is None."""
    engine = PFIMEngine(population_model=None, init_params=None)
    with pytest.raises(RuntimeError, match="non-None"):
        engine.compute_fim(np.array([1.0, 2.0]))


def test_pfim_optimize_design_requires_model() -> None:
    """optimize_design must raise RuntimeError when population_model is None."""
    engine = PFIMEngine(population_model=None, init_params=None)
    with pytest.raises(RuntimeError, match="non-None"):
        engine.optimize_design(n_samples=3)


# ---------------------------------------------------------------------------
# DesignResult construction
# ---------------------------------------------------------------------------


def test_design_result_fields() -> None:
    """DesignResult should store all fields correctly."""
    result = DesignResult(
        sampling_times=np.array([1.0, 2.0, 4.0]),
        information_matrix=np.eye(3),
        d_efficiency=1.0,
        a_efficiency=3.0,
        condition_number=1.0,
        se_theta=np.ones(3),
    )
    assert len(result.sampling_times) == 3
    assert result.information_matrix.shape == (3, 3)
    assert result.d_efficiency == pytest.approx(1.0)
    assert result.a_efficiency == pytest.approx(3.0)
    assert result.condition_number == pytest.approx(1.0)
    assert result.se_theta.shape == (3,)


def test_design_result_summary() -> None:
    """summary() should return a non-empty descriptive string."""
    result = DesignResult(
        sampling_times=np.array([1.0, 4.0, 8.0]),
        information_matrix=np.eye(3),
        d_efficiency=0.95,
        a_efficiency=2.5,
        condition_number=10.0,
        se_theta=np.array([0.1, 0.2, 0.15]),
    )
    s = result.summary()
    assert isinstance(s, str)
    assert len(s) > 0
    assert "D-efficiency" in s
    assert "A-criterion" in s


# ---------------------------------------------------------------------------
# Criterion helpers (tested via the engine on a standalone FIM)
# ---------------------------------------------------------------------------


def _make_engine_no_model() -> PFIMEngine:
    """Helper: create engine with no model, useful for testing helpers only."""
    return PFIMEngine(population_model=None, init_params=None)


def test_a_criterion_identity() -> None:
    """trace(I^{-1}) = trace(I) = n for the identity matrix."""
    engine = _make_engine_no_model()
    fim = np.eye(4)
    # FIM = I, so FIM^{-1} = I, trace = 4
    a_val = engine._a_criterion(fim)
    assert a_val == pytest.approx(4.0)


def test_a_criterion_scaled_identity() -> None:
    """For c*I, trace((cI)^{-1}) = n/c."""
    engine = _make_engine_no_model()
    c = 5.0
    n = 3
    fim = c * np.eye(n)
    a_val = engine._a_criterion(fim)
    assert a_val == pytest.approx(n / c, rel=1e-8)


def test_condition_number_identity() -> None:
    """Condition number of the identity matrix must be 1."""
    engine = _make_engine_no_model()
    cond = engine._condition_number(np.eye(5))
    assert cond == pytest.approx(1.0)


def test_condition_number_singular() -> None:
    """Condition number of a singular matrix must be inf."""
    engine = _make_engine_no_model()
    fim = np.zeros((3, 3))
    cond = engine._condition_number(fim)
    assert np.isinf(cond)


def test_condition_number_diagonal() -> None:
    """For diag(a, b) with a > b > 0, condition number = a/b."""
    engine = _make_engine_no_model()
    fim = np.diag([10.0, 2.0])
    cond = engine._condition_number(fim)
    assert cond == pytest.approx(5.0, rel=1e-8)


def test_se_from_fim_identity() -> None:
    """SE = sqrt(diag(FIM^{-1})) = 1 for identity FIM."""
    engine = _make_engine_no_model()
    se = engine._se_from_fim(np.eye(3))
    np.testing.assert_allclose(se, np.ones(3), atol=1e-10)


def test_se_from_fim_scaled() -> None:
    """For c*I, SE = 1/sqrt(c)."""
    engine = _make_engine_no_model()
    c = 4.0
    se = engine._se_from_fim(c * np.eye(3))
    np.testing.assert_allclose(se, np.full(3, 1.0 / np.sqrt(c)), rtol=1e-8)


def test_se_from_fim_singular_returns_zeros() -> None:
    """SE computation on singular FIM should return zeros gracefully."""
    engine = _make_engine_no_model()
    fim = np.zeros((3, 3))
    se = engine._se_from_fim(fim)
    assert se.shape == (3,)
    np.testing.assert_array_equal(se, np.zeros(3))


# ---------------------------------------------------------------------------
# _criterion_value
# ---------------------------------------------------------------------------


def test_criterion_value_d_identity() -> None:
    """D-criterion for identity FIM = log(det(I))/p = 0."""
    engine = _make_engine_no_model()
    fim = np.eye(4)
    val = engine._criterion_value(fim, "D")
    assert val == pytest.approx(0.0)


def test_criterion_value_d_scaled() -> None:
    """D-criterion for c*I_p = log(c^p)/p = log(c)."""
    engine = _make_engine_no_model()
    c = 3.0
    p = 3
    fim = c * np.eye(p)
    val = engine._criterion_value(fim, "D")
    assert val == pytest.approx(np.log(c), rel=1e-8)


def test_criterion_value_a_identity() -> None:
    """A-criterion for n*I equals -n (negated for maximisation)."""
    engine = _make_engine_no_model()
    n = 5
    fim = np.eye(n)
    # A-value = -trace(I^{-1}) = -n
    val = engine._criterion_value(fim, "A")
    assert val == pytest.approx(-float(n))


def test_criterion_value_e_identity() -> None:
    """E-criterion for identity FIM = min eigenvalue = 1."""
    engine = _make_engine_no_model()
    fim = np.eye(3)
    val = engine._criterion_value(fim, "E")
    assert val == pytest.approx(1.0)


def test_criterion_value_unknown_raises() -> None:
    """Unknown criterion should raise ValueError."""
    engine = _make_engine_no_model()
    with pytest.raises(ValueError, match="Unknown criterion"):
        engine._criterion_value(np.eye(2), "X")


# ---------------------------------------------------------------------------
# efficiency()
# ---------------------------------------------------------------------------


def _make_minimal_pfim_engine() -> PFIMEngine:
    """
    Helper: build a PFIMEngine with a minimal stub model.

    The stub model simulates monoexponential decay:
        C(t) = exp(-theta[0] * t)
    with one subject and obs_times matching the evaluate() output length.
    """
    OBS_TIMES = np.array([1.0, 2.0, 4.0, 8.0])
    K_TRUE = 0.1

    class SE:
        obs_times = OBS_TIMES
        obs_dv = np.exp(-K_TRUE * OBS_TIMES)
        obs_cmt = np.zeros(len(OBS_TIMES), dtype=int)
        obs_mdv = np.zeros(len(OBS_TIMES), dtype=int)

        def observation_mask(self_se):
            return np.ones(len(OBS_TIMES), dtype=bool)

    class Indiv:
        subject_events = SE()

        def evaluate(self_i, theta, eta, sigma, trans=None):
            f = np.exp(-theta[0] * OBS_TIMES)
            return f, None, f

    class MinimalModel:
        trans = None

        def subject_ids(self_m):
            return [1]

        def individual_model(self_m, sid):
            return Indiv()

    class MinimalParams:
        theta = np.array([K_TRUE])
        omega = np.eye(1) * 0.05
        sigma = np.eye(1) * 0.01

    return PFIMEngine(
        population_model=MinimalModel(),
        init_params=MinimalParams(),
    )


def _make_linear_gaussian_pfim_engine() -> tuple[PFIMEngine, np.ndarray, float]:
    """Closed-form linear-Gaussian model with no random effects."""
    obs_times_arr = np.array([0.5, 1.5, 3.0])
    sigma_var = 4.0

    class SE:
        obs_times = obs_times_arr

    class Indiv:
        subject_events = SE()

        def evaluate(self, theta, eta, sigma, trans=None):
            f = theta[0] + theta[1] * obs_times_arr
            return f, None, f

    class Model:
        trans = None

        def subject_ids(self):
            return [1]

        def individual_model(self, sid):
            return Indiv()

    class Params:
        theta = np.array([2.0, -0.3])
        omega = np.zeros((0, 0))
        sigma = np.array([[sigma_var]])

    return (
        PFIMEngine(population_model=Model(), init_params=Params()),
        obs_times_arr,
        sigma_var,
    )


def _make_eta_rank_one_pfim_engine() -> tuple[PFIMEngine, np.ndarray, float, float, float]:
    """Closed-form one-theta / one-eta model for exact FO FIM checks."""
    obs_times_arr = np.array([1.0, 2.0, 4.0])
    theta0 = 2.0
    omega0 = 0.2
    sigma0 = 0.5
    decay = 0.3

    class SE:
        obs_times = obs_times_arr

    class Indiv:
        subject_events = SE()

        def evaluate(self, theta, eta, sigma, trans=None):
            base = np.exp(-decay * obs_times_arr)
            f = theta[0] * np.exp(eta[0]) * base
            return f, None, f

    class Model:
        trans = None

        def subject_ids(self):
            return [1]

        def individual_model(self, sid):
            return Indiv()

    class Params:
        theta = np.array([theta0])
        omega = np.array([[omega0]])
        sigma = np.array([[sigma0]])

    return (
        PFIMEngine(population_model=Model(), init_params=Params()),
        obs_times_arr,
        theta0,
        omega0,
        sigma0,
    )


def _make_quadratic_time_pfim_engine() -> tuple[PFIMEngine, np.ndarray, float]:
    """Model whose prediction must be re-evaluated on the requested time grid."""
    obs_times_arr = np.array([0.5, 1.5, 3.0])
    sigma_var = 2.0

    class SE:
        def __init__(self, obs_times):
            self.obs_times = np.asarray(obs_times, dtype=float)
            self.obs_dv = np.full(len(self.obs_times), np.nan)
            self.obs_cmt = np.ones(len(self.obs_times), dtype=int)
            self.obs_mdv = np.zeros(len(self.obs_times), dtype=int)

        def observation_mask(self):
            return np.ones(len(self.obs_times), dtype=bool)

    class Indiv:
        def __init__(self):
            self.subject_events = SE(obs_times_arr)

        def evaluate(self, theta, eta, sigma, trans=None):
            times = np.asarray(self.subject_events.obs_times, dtype=float)
            f = theta[0] + theta[1] * times**2
            return f, np.ones(len(times), dtype=bool), f

    class Model:
        trans = None

        def subject_ids(self):
            return [1]

        def individual_model(self, sid):
            return Indiv()

    class Params:
        theta = np.array([1.0, 0.5])
        omega = np.zeros((0, 0))
        sigma = np.array([[sigma_var]])

    return PFIMEngine(population_model=Model(), init_params=Params()), obs_times_arr, sigma_var


def test_efficiency_same_design() -> None:
    """D-efficiency of a design compared to itself must be 1.0."""
    engine = _make_minimal_pfim_engine()
    times = np.array([1.0, 2.0, 4.0, 8.0])
    eff = engine.efficiency(times, times, criterion="D")
    assert eff == pytest.approx(1.0, rel=1e-5), f"Self D-efficiency should be 1.0, got {eff}"


def test_efficiency_a_criterion_self() -> None:
    """A-efficiency of a design compared to itself must be 1.0."""
    engine = _make_minimal_pfim_engine()
    times = np.array([1.0, 2.0, 4.0, 8.0])
    eff = engine.efficiency(times, times, criterion="A")
    # A-efficiency: trace(FIM_ref^{-1}) / trace(FIM_test^{-1}) = 1
    assert eff == pytest.approx(1.0, rel=1e-5)


def test_compute_fim_matches_closed_form_linear_gaussian_no_eta() -> None:
    engine, times, sigma_var = _make_linear_gaussian_pfim_engine()

    fim = engine.compute_fim(times, n_subjects=1)
    G = np.column_stack([np.ones(len(times)), times])
    expected = (G.T @ G) / sigma_var

    np.testing.assert_allclose(fim, expected, rtol=1e-5, atol=1e-7)


def test_compute_fim_scales_linearly_with_subject_count() -> None:
    engine, times, _ = _make_linear_gaussian_pfim_engine()

    fim_1 = engine.compute_fim(times, n_subjects=1)
    fim_7 = engine.compute_fim(times, n_subjects=7)

    np.testing.assert_allclose(fim_7, 7.0 * fim_1, rtol=1e-10)


def test_compute_fim_scales_inversely_with_sigma() -> None:
    engine, times, _ = _make_linear_gaussian_pfim_engine()

    fim_small_sigma = engine.compute_fim(times, sigma=np.array([[2.0]]))
    fim_large_sigma = engine.compute_fim(times, sigma=np.array([[8.0]]))

    np.testing.assert_allclose(fim_large_sigma, 0.25 * fim_small_sigma, rtol=1e-5)


def test_compute_fim_rejects_multi_endpoint_diagonal_sigma() -> None:
    engine, times, _ = _make_linear_gaussian_pfim_engine()

    with pytest.raises(ValueError, match="scalar residual variance"):
        engine.compute_fim(times, sigma=np.diag([2.0, 3.0]))


def test_compute_fim_rejects_correlated_sigma() -> None:
    engine, times, _ = _make_linear_gaussian_pfim_engine()

    with pytest.raises(ValueError, match="correlated residual structures"):
        engine.compute_fim(times, sigma=np.array([[2.0, 0.5], [0.5, 2.0]]))


def test_compute_fim_accepts_sigma_with_only_leading_scalar_term() -> None:
    engine, times, _ = _make_linear_gaussian_pfim_engine()

    fim_reference = engine.compute_fim(times, sigma=np.array([[2.0]]))
    fim_with_trailing_zeros = engine.compute_fim(
        times,
        sigma=np.array([[2.0, 0.0], [0.0, 0.0]]),
    )

    np.testing.assert_allclose(fim_with_trailing_zeros, fim_reference, rtol=1e-10, atol=1e-12)


def test_compute_fim_rejects_non_square_sigma() -> None:
    engine, times, _ = _make_linear_gaussian_pfim_engine()

    with pytest.raises(ValueError, match="square SIGMA matrices"):
        engine.compute_fim(times, sigma=np.array([[2.0, 0.0]]))


@pytest.mark.parametrize("bad_sigma", [np.array([[0.0]]), np.array([[-1.0]]), np.array([[np.nan]])])
def test_compute_fim_rejects_nonpositive_or_nonfinite_scalar_sigma(
    bad_sigma: np.ndarray,
) -> None:
    engine, times, _ = _make_linear_gaussian_pfim_engine()

    with pytest.raises(ValueError, match="finite positive scalar residual variance"):
        engine.compute_fim(times, sigma=bad_sigma)


def test_compute_fim_matches_closed_form_with_random_effect_variance() -> None:
    engine, times, theta0, omega0, sigma0 = _make_eta_rank_one_pfim_engine()

    fim = engine.compute_fim(times)
    base = np.exp(-0.3 * times)[:, None]
    G = base
    Z = theta0 * base
    V = omega0 * (Z @ Z.T) + sigma0 * np.eye(len(times))
    expected = G.T @ np.linalg.inv(V) @ G

    np.testing.assert_allclose(fim, expected, rtol=1e-5, atol=1e-7)


def test_interpolate_predictions_re_evaluates_requested_times() -> None:
    engine, _, _ = _make_quadratic_time_pfim_engine()
    indiv = engine.population_model.individual_model(1)
    theta = engine.init_params.theta
    sigma = engine.init_params.sigma
    target_times = np.array([0.25, 2.0, 4.0])

    pred = engine._interpolate_predictions(
        indiv,
        theta,
        np.zeros(0),
        sigma,
        target_times,
    )

    expected = theta[0] + theta[1] * target_times**2
    np.testing.assert_allclose(pred, expected, rtol=1e-10, atol=1e-12)


def test_compute_fim_uses_requested_times_beyond_subject_grid() -> None:
    engine, _, sigma_var = _make_quadratic_time_pfim_engine()
    times = np.array([0.25, 2.0, 4.0])

    fim = engine.compute_fim(times, n_subjects=1)
    G = np.column_stack([np.ones(len(times)), times**2])
    expected = (G.T @ G) / sigma_var

    np.testing.assert_allclose(fim, expected, rtol=1e-5, atol=1e-7)


def test_d_efficiency_is_invariant_to_subject_count() -> None:
    engine = _make_minimal_pfim_engine()
    times_test = np.array([1.0, 2.0, 4.0, 8.0])
    times_ref = np.array([1.0, 3.0, 6.0, 12.0])

    eff_1 = engine.efficiency(times_test, times_ref, criterion="D", n_subjects=1)
    eff_9 = engine.efficiency(times_test, times_ref, criterion="D", n_subjects=9)

    assert eff_9 == pytest.approx(eff_1, rel=1e-10)
