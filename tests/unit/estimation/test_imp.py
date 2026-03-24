from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.imp import IMPMethod


class _GaussianSubjectEvents:
    def __init__(self, dv: float) -> None:
        self.obs_dv = np.array([dv], dtype=float)

    def observation_mask(self) -> np.ndarray:
        return np.array([True])


class _GaussianIndividualModel:
    def __init__(self, dv: float) -> None:
        self.dv = float(dv)
        self.subject_events = _GaussianSubjectEvents(dv)

    def log_likelihood(self, theta, eta, sigma, trans=None) -> float:
        eta_val = float(np.asarray(eta, dtype=float)[0])
        sigma_var = float(sigma[0, 0])
        return float(np.log(2.0 * np.pi * sigma_var) + (self.dv - eta_val) ** 2 / sigma_var)

    def obj_eta(self, eta, theta, omega, sigma, trans=None) -> float:
        eta_val = float(np.asarray(eta, dtype=float)[0])
        omega_var = float(omega[0, 0])
        return float(
            self.log_likelihood(theta, np.array([eta_val]), sigma, trans=trans)
            + eta_val**2 / omega_var
        )


class _NativeHessianGaussianIndividualModel(_GaussianIndividualModel):
    def __init__(self, dv: float) -> None:
        super().__init__(dv)
        self.hessian_calls = 0

    def eta_objective_hessian(self, theta, eta, omega, sigma, trans=None):
        self.hessian_calls += 1
        sigma_var = float(sigma[0, 0])
        omega_var = float(omega[0, 0])
        return np.array([[2.0 / sigma_var + 2.0 / omega_var]])


class _GaussianPopulationModel:
    trans = 2

    def __init__(self, dv: float) -> None:
        self._indiv = {1: _GaussianIndividualModel(dv)}

    def subject_ids(self):
        return [1]

    def individual_model(self, sid):
        return self._indiv[sid]


class _NativeHessianGaussianPopulationModel(_GaussianPopulationModel):
    def __init__(self, dv: float) -> None:
        self._indiv = {1: _NativeHessianGaussianIndividualModel(dv)}


class _GaussianParams:
    def __init__(self, omega_var: float, sigma_var: float) -> None:
        self.theta = np.array([0.0])
        self.omega = np.array([[omega_var]], dtype=float)
        self.sigma = np.array([[sigma_var]], dtype=float)

    def n_eta(self) -> int:
        return 1


def _implemented_log_marginal_gaussian(dv: float, sigma_var: float, omega_var: float) -> float:
    """Closed-form log p(y) = log N(y; 0, sigma_var + omega_var) — fully normalised."""
    total_var = sigma_var + omega_var
    return float(-0.5 * (np.log(2.0 * np.pi * total_var) + dv**2 / total_var))


def test_importance_sample_matches_closed_form_gaussian_log_marginal() -> None:
    pop_model = _GaussianPopulationModel(dv=1.25)
    params = _GaussianParams(omega_var=0.6, sigma_var=0.4)

    log_marg = IMPMethod(isample=32, seed=0)._importance_sample(pop_model, params, 1)

    assert log_marg == pytest.approx(
        _implemented_log_marginal_gaussian(1.25, 0.4, 0.6),
        abs=1e-8,
    )


def test_compute_imp_ofv_matches_closed_form_gaussian_case() -> None:
    pop_model = _GaussianPopulationModel(dv=1.25)
    params = _GaussianParams(omega_var=0.6, sigma_var=0.4)

    ofv = IMPMethod(isample=32, seed=0)._compute_imp_ofv(pop_model, params)

    expected = -2.0 * _implemented_log_marginal_gaussian(1.25, 0.4, 0.6)
    assert ofv == pytest.approx(expected, abs=1e-8)


def test_importance_sample_uses_native_eta_hessian_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pop_model = _NativeHessianGaussianPopulationModel(dv=1.25)
    params = _GaussianParams(omega_var=0.6, sigma_var=0.4)

    def _unexpected_numerical_hessian(*args, **kwargs):
        raise AssertionError("numerical_hessian should not be used")

    monkeypatch.setattr("openpkpd.estimation.imp.numerical_hessian", _unexpected_numerical_hessian)
    log_marg = IMPMethod(isample=32, seed=0)._importance_sample(pop_model, params, 1)
    assert log_marg == pytest.approx(
        _implemented_log_marginal_gaussian(1.25, 0.4, 0.6),
        abs=1e-8,
    )
    assert pop_model.individual_model(1).hessian_calls == 1


# ---------------------------------------------------------------------------
# Additional unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "dv,omega_var,sigma_var",
    [
        (0.5, 0.3, 0.2),
        (2.0, 1.0, 0.5),
        (0.0, 0.8, 0.8),
        (-1.5, 0.4, 0.6),
    ],
)
def test_imp_ofv_finite_for_various_params(dv, omega_var, sigma_var) -> None:
    """IMP OFV must be finite for a range of Gaussian parameter combinations."""
    pop = _GaussianPopulationModel(dv=dv)
    params = _GaussianParams(omega_var=omega_var, sigma_var=sigma_var)
    ofv = IMPMethod(isample=200, seed=7)._compute_imp_ofv(pop, params)
    assert np.isfinite(ofv), f"IMP OFV not finite for dv={dv}"


@pytest.mark.unit
def test_imp_ofv_is_positive() -> None:
    """IMP OFV = −2*log-marginal; must be > 0 for this Gaussian model."""
    pop = _GaussianPopulationModel(dv=1.0)
    params = _GaussianParams(omega_var=0.5, sigma_var=0.5)
    ofv = IMPMethod(isample=500, seed=9)._compute_imp_ofv(pop, params)
    assert ofv > 0.0


@pytest.mark.unit
def test_imp_sensitivity_to_seed() -> None:
    """Different seeds give slightly different OFVs; both close to analytic."""
    dv, omega_var, sigma_var = 1.5, 0.4, 0.6
    pop = _GaussianPopulationModel(dv=dv)
    params = _GaussianParams(omega_var=omega_var, sigma_var=sigma_var)
    analytic = -2.0 * _implemented_log_marginal_gaussian(dv, sigma_var, omega_var)

    ofv_a = IMPMethod(isample=500, seed=1)._compute_imp_ofv(pop, params)
    ofv_b = IMPMethod(isample=500, seed=2)._compute_imp_ofv(pop, params)

    assert abs(ofv_a - analytic) < 0.5
    assert abs(ofv_b - analytic) < 0.5
    assert ofv_a != ofv_b  # stochastic → different seeds → different values


@pytest.mark.unit
def test_imp_error_decreases_with_larger_sample() -> None:
    """|IMP(N=2000) − analytic| ≤ |IMP(N=50) − analytic| + small_tol."""
    dv, omega_var, sigma_var = 1.0, 0.5, 0.5
    pop = _GaussianPopulationModel(dv=dv)
    params = _GaussianParams(omega_var=omega_var, sigma_var=sigma_var)
    analytic = -2.0 * _implemented_log_marginal_gaussian(dv, sigma_var, omega_var)

    err_small = abs(IMPMethod(isample=50, seed=0)._compute_imp_ofv(pop, params) - analytic)
    err_large = abs(IMPMethod(isample=2000, seed=0)._compute_imp_ofv(pop, params) - analytic)
    assert err_large <= err_small + 0.01
