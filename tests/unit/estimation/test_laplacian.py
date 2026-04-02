from __future__ import annotations

import math

import numpy as np
import pytest

from openpkpd.estimation.laplacian import LaplacianMethod


class _DummySubjectEvents:
    obs_dv = np.array([1.0])

    def observation_mask(self) -> np.ndarray:
        return np.array([True])


class _ObservationModelIndividual:
    def __init__(self) -> None:
        self.subject_events = _DummySubjectEvents()

    def evaluate(self, theta, eta, sigma, trans=2):
        structural = np.array([1.8])
        return structural, np.array([True]), structural

    def evaluate_observation_model(self, theta, eta, sigma, trans=2):
        pred = np.array([1.2])
        var = np.array([0.1])
        return pred, np.array([True]), pred, pred, var

    def obj_eta(self, eta, theta, omega, sigma, trans=2) -> float:
        eta_arr = np.asarray(eta, dtype=float)
        target = np.array([0.25])
        return float(np.sum((eta_arr - target) ** 2))


class _NativeHessianObservationModelIndividual(_ObservationModelIndividual):
    def __init__(self) -> None:
        super().__init__()
        self.hessian_calls = 0

    def eta_objective_hessian(self, theta, eta, omega, sigma, trans=2):
        self.hessian_calls += 1
        return np.array([[2.0]])


class _ObservationModelPopulation:
    trans = 2

    def subject_ids(self):
        return [1]

    def individual_model(self, sid):
        assert sid == 1
        return _ObservationModelIndividual()


class _NativeHessianObservationModelPopulation(_ObservationModelPopulation):
    def __init__(self) -> None:
        self._indiv = _NativeHessianObservationModelIndividual()

    def individual_model(self, sid):
        assert sid == 1
        return self._indiv


class _Params:
    theta = np.array([1.0])
    omega = np.array([[0.2]])
    sigma = np.array([[0.1]])

    def n_eta(self) -> int:
        return 1


def test_laplacian_outer_ofv_matches_foce_plus_logdet_hessian() -> None:
    eta = np.array([0.25])
    ofv = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(
        _ObservationModelPopulation(),
        _Params(),
        {1: eta},
    )

    expected = (
        math.log(2.0 * math.pi)
        + math.log(0.1)          # log|C_i| (residual variance)
        + math.log(0.2)          # log|Ω|   (H-08: Ω changes each outer iter)
        + (1.0 - 1.2) ** 2 / 0.1
        + eta[0] ** 2 / 0.2
        + math.log(2.0)
    )
    assert ofv == pytest.approx(expected, abs=1e-10)


def test_laplacian_uses_native_eta_hessian_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    pop = _NativeHessianObservationModelPopulation()

    def _unexpected_numerical_hessian(*args, **kwargs):
        raise AssertionError("numerical_hessian should not be used")

    monkeypatch.setattr(
        "openpkpd.estimation.laplacian.numerical_hessian", _unexpected_numerical_hessian
    )
    ofv = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(
        pop, _Params(), {1: np.array([0.25])}
    )
    assert np.isfinite(ofv)
    assert pop._indiv.hessian_calls == 1


def test_laplacian_outer_ofv_parallel_matches_serial_for_multi_subject_population() -> None:
    class _TwoSubjectPopulation:
        trans = 2

        def __init__(self) -> None:
            self._indiv = {
                1: _ObservationModelIndividual(),
                2: _ObservationModelIndividual(),
            }

        def subject_ids(self):
            return [1, 2]

        def individual_model(self, sid):
            return self._indiv[sid]

    eta_hat = {
        1: np.array([0.25]),
        2: np.array([0.25]),
    }
    serial = LaplacianMethod(interaction=False, maxeval=1, n_parallel=1)._outer_ofv(
        _TwoSubjectPopulation(),
        _Params(),
        eta_hat,
    )
    parallel = LaplacianMethod(interaction=False, maxeval=1, n_parallel=2)._outer_ofv(
        _TwoSubjectPopulation(),
        _Params(),
        eta_hat,
    )

    assert parallel == pytest.approx(serial, abs=1e-12)


# ---------------------------------------------------------------------------
# Additional unit tests for Laplacian accuracy
# ---------------------------------------------------------------------------


class _ScaledIndividual(_ObservationModelIndividual):
    """Individual with controllable pred/var for parametric OFV tests."""

    def __init__(self, obs: float, pred: float, var: float, omega_var: float) -> None:
        self._obs = obs
        self._pred = pred
        self._var = var
        self._omega_var = omega_var
        self.subject_events = type(
            "E", (), {"obs_dv": np.array([obs]), "observation_mask": lambda s: np.array([True])}
        )()

    def evaluate_observation_model(self, theta, eta, sigma, trans=2):
        pred = np.array([self._pred])
        var = np.array([self._var])
        return pred, np.array([True]), pred, pred, var

    def obj_eta(self, eta, theta, omega, sigma, trans=2) -> float:
        eta_v = float(np.asarray(eta)[0])
        om_v = float(omega[0, 0])
        sig_v = float(sigma[0, 0])
        return float(
            math.log(2 * math.pi * sig_v) + (self._obs - eta_v) ** 2 / sig_v + eta_v**2 / om_v
        )


class _ScaledParams:
    def __init__(self, omega_var: float, sigma_var: float) -> None:
        self.theta = np.array([1.0])
        self.omega = np.array([[omega_var]])
        self.sigma = np.array([[sigma_var]])

    def n_eta(self) -> int:
        return 1


def _make_pop(obs: float, pred: float, var: float, omega_var: float):
    indiv = _ScaledIndividual(obs, pred, var, omega_var)

    class _Pop:
        trans = 2

        def subject_ids(self):
            return [1]

        def individual_model(self, sid):
            return indiv

    return _Pop()


@pytest.mark.unit
@pytest.mark.parametrize(
    "obs,pred,var,omega_var,sigma_var",
    [
        (0.5, 0.4, 0.1, 0.2, 0.1),
        (2.0, 1.8, 0.5, 0.4, 0.5),
        (0.0, 0.0, 1.0, 1.0, 1.0),
        (1.5, 1.5, 0.2, 0.3, 0.2),
    ],
)
def test_laplacian_ofv_increases_with_pred_error(obs, pred, var, omega_var, sigma_var) -> None:
    """Larger |obs − pred| gives larger Laplacian OFV."""
    params = _ScaledParams(omega_var, sigma_var)
    # Build a population where pred matches obs exactly, then shift pred by 1
    pop_good = _make_pop(obs, obs, var, omega_var)
    pop_bad = _make_pop(obs, obs + 1.0, var, omega_var)
    eta = {1: np.array([0.0])}
    ofv_good = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(pop_good, params, eta)
    ofv_bad = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(pop_bad, params, eta)
    assert ofv_bad > ofv_good


@pytest.mark.unit
def test_laplacian_ofv_linear_gaussian_formula() -> None:
    """
    Verify the Laplacian OFV formula at the MAP estimate for linear-Gaussian.

    The code implements:
        OFV = FOCE_base(η̂) + log|H_obj_eta|
    where FOCE_base = log(2πσ) + log|Ω| + (y−η̂)²/σ + η̂²/ω
    and   H_obj_eta = d²obj_eta/dη² = 2/σ + 2/ω
    MAP eta: η̂ = y·ω/(ω+σ)

    The log|Ω| term is included because Ω changes each outer iteration, so
    the gradient w.r.t. OMEGA is only correct when log|Ω| is part of the OFV
    (H-08 fix).
    """
    dv, omega_v, sigma_v = 1.2, 0.4, 0.6
    eta_map = dv * omega_v / (omega_v + sigma_v)
    foce_base = (
        math.log(2 * math.pi * sigma_v)
        + math.log(omega_v)  # log|Ω| — required for correct OMEGA gradient
        + (dv - eta_map) ** 2 / sigma_v
        + eta_map**2 / omega_v
    )
    H = 2.0 / sigma_v + 2.0 / omega_v
    expected_ofv = foce_base + math.log(H)

    params = _ScaledParams(omega_v, sigma_v)
    pop = _make_pop(dv, eta_map, sigma_v, omega_v)
    eta_hat = {1: np.array([eta_map])}

    ofv = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(pop, params, eta_hat)
    assert ofv == pytest.approx(expected_ofv, abs=1e-4)


@pytest.mark.unit
def test_laplacian_ofv_is_finite_for_small_omega() -> None:
    """Laplacian OFV should remain finite even for near-zero OMEGA."""
    params = _ScaledParams(omega_var=1e-4, sigma_var=0.1)
    pop = _make_pop(obs=1.0, pred=1.0, var=0.1, omega_var=1e-4)
    eta_hat = {1: np.array([0.0])}
    ofv = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(pop, params, eta_hat)
    assert np.isfinite(ofv)


@pytest.mark.unit
def test_laplacian_ofv_scale_with_sigma() -> None:
    """
    Doubling σ changes both the log(σ) term and the log|H| term.

    OFV = log(2πσ) + (obs−pred)²/σ + η²/ω + log(2/σ + 2/ω)
    With zero residual and eta=0, OFV = log(2πσ) + log(2/σ + 2/ω).
    The expected change when σ doubles is computed analytically.
    """
    obs, pred = 1.0, 1.0  # zero residual so residual term doesn't change
    omega_v = 0.3
    sigma1 = 0.2
    sigma2 = 0.4  # double

    def _expected_ofv(sigma, omega):
        return math.log(2 * math.pi * sigma) + math.log(2.0 / sigma + 2.0 / omega)

    expected_delta = _expected_ofv(sigma2, omega_v) - _expected_ofv(sigma1, omega_v)

    params1 = _ScaledParams(omega_v, sigma1)
    params2 = _ScaledParams(omega_v, sigma2)
    pop1 = _make_pop(obs, pred, sigma1, omega_v)
    pop2 = _make_pop(obs, pred, sigma2, omega_v)
    eta_hat = {1: np.array([0.0])}

    ofv1 = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(pop1, params1, eta_hat)
    ofv2 = LaplacianMethod(interaction=False, maxeval=1)._outer_ofv(pop2, params2, eta_hat)
    assert ofv2 - ofv1 == pytest.approx(expected_delta, abs=1e-4)
