"""Parity checks for Rust-accelerated paths against Python fallback behavior."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openpkpd._native import import_core_symbol
from openpkpd.data.event_processor import DoseEvent, SubjectEvents
from openpkpd.model.individual import IndividualModel
from openpkpd.model.individual import _likelihood as likelihood_mod
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan2 import ADVAN2


try:
    _NEG2LL_NATIVE = import_core_symbol("neg2ll_obs_loop")
except ImportError:
    _NEG2LL_NATIVE = None

try:
    _ANALYTIC_1CMT_ORAL_NATIVE = import_core_symbol("analytic_1cmt_oral_probe_multidose")
except ImportError:
    _ANALYTIC_1CMT_ORAL_NATIVE = None

try:
    _ANALYTIC_1CMT_IV_INF_NATIVE = import_core_symbol("analytic_1cmt_iv_infusion_probe_multidose")
except ImportError:
    _ANALYTIC_1CMT_IV_INF_NATIVE = None

try:
    _NATIVE_2CMT_IV_SENS = import_core_symbol("native_cvodes_2cmt_iv_sensitivity_probe_multidose")
except ImportError:
    _NATIVE_2CMT_IV_SENS = None


class _StubLikelihoodModel(likelihood_mod.LikelihoodMixin):
    def __init__(
        self,
        dv: np.ndarray,
        pred: np.ndarray,
        var: np.ndarray,
        obs_mask: np.ndarray,
        *,
        blq_method: str = "M1",
        lloq: float | np.ndarray | None = None,
    ) -> None:
        self.subject_events = SimpleNamespace(obs_dv=np.asarray(dv, dtype=float))
        self._pred = np.asarray(pred, dtype=float)
        self._var = np.asarray(var, dtype=float)
        self._obs_mask = np.asarray(obs_mask, dtype=bool)
        self.blq_method = blq_method
        self.lloq = lloq

    def evaluate_observation_model(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
        eps_val: np.ndarray | None = None,
    ):
        ipred = self._pred[self._obs_mask]
        return ipred, self._obs_mask.copy(), self._pred.copy(), self._pred.copy(), self._var.copy()


def _sample_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 16
    pred = rng.uniform(0.3, 5.0, n)
    var = rng.uniform(0.05, 0.6, n)
    dv = pred + rng.normal(0.0, 0.25, n)
    obs_mask = rng.random(n) > 0.15
    lloq = np.full(n, np.nan)
    return dv, pred, var, obs_mask, lloq


def _build_advan2_individual() -> IndividualModel:
    obs_times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0], dtype=float)
    events = SubjectEvents(
        subject_id=1,
        dose_events=[
            DoseEvent(time=0.0, amount=320.0, compartment=1),
            DoseEvent(time=12.0, amount=320.0, compartment=1),
        ],
        obs_times=obs_times,
        obs_dv=np.full(len(obs_times), np.nan),
        obs_cmt=np.ones(len(obs_times), dtype=int),
        obs_mdv=np.zeros(len(obs_times), dtype=int),
    )

    def pk_callable(theta, eta, t=0.0, covariates=None):
        return {
            "KA": float(theta[0]) * np.exp(float(eta[0])),
            "CL": float(theta[1]) * np.exp(float(eta[1])),
            "V": float(theta[2]),
        }

    def error_callable(theta, eta, eps, f, ipred, y, t, a=None, covariates=None, sigma=None):
        return {"Y": f * (1.0 + eps[0]), "IPRED": f}

    error_callable._source = "Y = F * (1 + EPS[0])"

    return IndividualModel(
        subject_events=events,
        pk_subroutine=ADVAN2(),
        pk_callable=pk_callable,
        error_callable=error_callable,
        n_eps=1,
    )


def _build_advan1_infusion_individual() -> IndividualModel:
    _advan1_impl = ADVAN1()

    class _Advan1Inf:
        advan = 1
        n_compartments = 1

        def solve(self, pk_params, dose_events, times, **kwargs):
            kwargs.pop("return_amounts", None)
            return _advan1_impl.solve(pk_params, dose_events, times, **kwargs)

        def apply_trans(self, pk_params, trans):
            return _advan1_impl.apply_trans(pk_params, trans)

    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0], dtype=float)
    events = SubjectEvents(
        subject_id=1,
        dose_events=[
            DoseEvent(time=0.0, amount=100.0, compartment=1, rate=10.0),
            DoseEvent(time=18.0, amount=50.0, compartment=1, rate=5.0),
        ],
        obs_times=obs_times,
        obs_dv=np.full(len(obs_times), np.nan),
        obs_cmt=np.ones(len(obs_times), dtype=int),
        obs_mdv=np.zeros(len(obs_times), dtype=int),
    )

    def pk_callable(theta, eta, t=0.0, covariates=None):
        return {
            "CL": float(theta[0]) * np.exp(float(eta[0])),
            "V": float(theta[1]) * np.exp(float(eta[1])),
        }

    def error_callable(theta, eta, eps, f, ipred, y, t, a=None, covariates=None, sigma=None):
        return {"Y": f * (1.0 + eps[0]), "IPRED": f}

    error_callable._source = "Y = F * (1 + EPS[0])"

    return IndividualModel(
        subject_events=events,
        pk_subroutine=_Advan1Inf(),
        pk_callable=pk_callable,
        error_callable=error_callable,
        n_eps=1,
    )


def _build_native_2cmt_individual() -> IndividualModel:
    class _Advan6TwoCmt:
        advan = 6
        n_compartments = 2

        def solve(self, pk_params, dose_events, times, **kwargs):  # pragma: no cover
            raise NotImplementedError("native path should be used")

        def apply_trans(self, pk_params, trans):  # pragma: no cover
            return pk_params

    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0], dtype=float)
    events = SubjectEvents(
        subject_id=1,
        dose_events=[DoseEvent(time=0.0, amount=100.0, compartment=1)],
        obs_times=obs_times,
        obs_dv=np.full(len(obs_times), np.nan),
        obs_cmt=np.ones(len(obs_times), dtype=int),
        obs_mdv=np.zeros(len(obs_times), dtype=int),
    )

    def pk_callable(theta, eta, t=0.0, covariates=None):
        return {
            "CL": float(theta[0]) * np.exp(float(eta[0])),
            "V1": float(theta[1]) * np.exp(float(eta[1])),
            "Q": float(theta[2]),
            "V2": float(theta[3]),
        }

    def error_callable(theta, eta, eps, f, ipred, y, t, a=None, covariates=None, sigma=None):
        return {"Y": f * (1.0 + eps[0]), "IPRED": f}

    error_callable._source = "Y = F * (1 + EPS[0])"

    return IndividualModel(
        subject_events=events,
        pk_subroutine=_Advan6TwoCmt(),
        pk_callable=pk_callable,
        error_callable=error_callable,
        n_eps=1,
    )


@pytest.mark.skipif(_NEG2LL_NATIVE is None, reason="native _core extension not available")
@pytest.mark.parametrize("blq_method", ["M1", "M3", "M4", "M5", "M6", "M7"])
def test_log_likelihood_native_matches_python_fallback_across_blq_methods(
    monkeypatch: pytest.MonkeyPatch,
    blq_method: str,
) -> None:
    dv, pred, var, obs_mask, lloq = _sample_case(100 + hash(blq_method) % 1000)
    if blq_method != "M1":
        # Force a mix of BLQ and non-BLQ observations.
        lloq[:] = 1.0
        dv[:5] = np.array([0.2, 0.4, 0.6, 0.8, 0.9])

    model = _StubLikelihoodModel(dv, pred, var, obs_mask, blq_method=blq_method, lloq=lloq)

    monkeypatch.setattr(likelihood_mod, "_RUST_CORE_AVAILABLE", True)
    monkeypatch.setattr(likelihood_mod, "_neg2ll_obs_loop_rust", _NEG2LL_NATIVE)
    native_value = model.log_likelihood(np.array([1.0]), np.zeros(1), np.array([[0.1]]))

    monkeypatch.setattr(likelihood_mod, "_RUST_CORE_AVAILABLE", False)
    python_value = model.log_likelihood(np.array([1.0]), np.zeros(1), np.array([[0.1]]))

    assert native_value == pytest.approx(python_value, rel=1e-10, abs=1e-10)


@pytest.mark.skipif(_NEG2LL_NATIVE is None, reason="native _core extension not available")
def test_log_likelihood_native_matches_python_fallback_with_scalar_lloq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dv, pred, var, obs_mask, _ = _sample_case(4242)
    dv[:4] = np.array([0.25, 0.5, 0.75, 0.95])
    model = _StubLikelihoodModel(dv, pred, var, obs_mask, blq_method="M3", lloq=1.0)

    monkeypatch.setattr(likelihood_mod, "_RUST_CORE_AVAILABLE", True)
    monkeypatch.setattr(likelihood_mod, "_neg2ll_obs_loop_rust", _NEG2LL_NATIVE)
    native_value = model.log_likelihood(np.array([1.0]), np.zeros(1), np.array([[0.1]]))

    monkeypatch.setattr(likelihood_mod, "_RUST_CORE_AVAILABLE", False)
    python_value = model.log_likelihood(np.array([1.0]), np.zeros(1), np.array([[0.1]]))

    assert native_value == pytest.approx(python_value, rel=1e-10, abs=1e-10)


@pytest.mark.skipif(_NEG2LL_NATIVE is None, reason="native _core extension not available")
def test_log_likelihood_native_matches_python_fallback_for_zero_variance_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dv = np.array([1.0, 0.5, np.nan])
    pred = np.array([1.2, 0.6, 0.0])
    var = np.array([0.0, 0.3, 1.0])
    obs_mask = np.array([True, True, False])
    model = _StubLikelihoodModel(dv, pred, var, obs_mask, blq_method="M1", lloq=None)

    monkeypatch.setattr(likelihood_mod, "_RUST_CORE_AVAILABLE", True)
    monkeypatch.setattr(likelihood_mod, "_neg2ll_obs_loop_rust", _NEG2LL_NATIVE)
    native_value = model.log_likelihood(np.array([1.0]), np.zeros(1), np.array([[0.1]]))

    monkeypatch.setattr(likelihood_mod, "_RUST_CORE_AVAILABLE", False)
    python_value = model.log_likelihood(np.array([1.0]), np.zeros(1), np.array([[0.1]]))

    assert native_value == pytest.approx(python_value, rel=1e-10, abs=1e-10)


@pytest.mark.skipif(
    _ANALYTIC_1CMT_ORAL_NATIVE is None,
    reason="analytic 1cmt oral native probe not available",
)
def test_advan2_evaluate_native_matches_forced_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.array([1.4, 2.8, 32.0], dtype=float)
    eta = np.array([0.15, -0.10], dtype=float)
    sigma = np.array([[0.04]], dtype=float)

    native_indiv = _build_advan2_individual()
    python_indiv = _build_advan2_individual()
    monkeypatch.setattr(python_indiv, "_try_native_pk_backend", lambda pk_params, obs_times: None)

    native_ipred, native_mask, native_f = native_indiv.evaluate(theta, eta, sigma, trans=2)
    python_ipred, python_mask, python_f = python_indiv.evaluate(theta, eta, sigma, trans=2)

    np.testing.assert_array_equal(native_mask, python_mask)
    np.testing.assert_allclose(native_ipred, python_ipred, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(native_f, python_f, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    _ANALYTIC_1CMT_ORAL_NATIVE is None,
    reason="analytic 1cmt oral native probe not available",
)
def test_advan2_observation_model_native_matches_forced_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.array([1.2, 3.1, 28.0], dtype=float)
    eta = np.array([-0.20, 0.25], dtype=float)
    sigma = np.array([[0.09]], dtype=float)

    native_indiv = _build_advan2_individual()
    python_indiv = _build_advan2_individual()
    monkeypatch.setattr(python_indiv, "_try_native_pk_backend", lambda pk_params, obs_times: None)

    native_out = native_indiv.evaluate_observation_model(theta, eta, sigma, trans=2)
    python_out = python_indiv.evaluate_observation_model(theta, eta, sigma, trans=2)

    for native_arr, python_arr in zip(native_out, python_out, strict=False):
        np.testing.assert_allclose(np.asarray(native_arr), np.asarray(python_arr), rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    _ANALYTIC_1CMT_IV_INF_NATIVE is None,
    reason="analytic 1cmt iv infusion native probe not available",
)
def test_advan1_infusion_evaluate_native_matches_forced_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.array([2.2, 18.0], dtype=float)
    eta = np.array([0.10, -0.15], dtype=float)
    sigma = np.array([[0.04]], dtype=float)

    native_indiv = _build_advan1_infusion_individual()
    python_indiv = _build_advan1_infusion_individual()
    monkeypatch.setattr(python_indiv, "_try_native_pk_backend", lambda pk_params, obs_times: None)

    native_ipred, native_mask, native_f = native_indiv.evaluate(theta, eta, sigma, trans=2)
    python_ipred, python_mask, python_f = python_indiv.evaluate(theta, eta, sigma, trans=2)

    np.testing.assert_array_equal(native_mask, python_mask)
    np.testing.assert_allclose(native_ipred, python_ipred, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(native_f, python_f, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    _NATIVE_2CMT_IV_SENS is None,
    reason="native 2cmt iv sensitivity probe not available",
)
def test_native_2cmt_eta_jacobian_matches_generic_prediction_eta_jacobian() -> None:
    indiv = _build_native_2cmt_individual()
    theta = np.array([1.8, 12.0, 0.9, 18.0], dtype=float)
    eta = np.array([0.12, -0.08], dtype=float)
    sigma = np.array([[0.04]], dtype=float)
    obs_mask = indiv.subject_events.observation_mask()

    native_jac = indiv.native_advan6_prediction_eta_jacobian(theta, eta, obs_mask, n_eta=2)
    assert native_jac is not None

    generic_jac = indiv.prediction_eta_jacobian(theta, eta, sigma, trans=2)

    np.testing.assert_allclose(native_jac, generic_jac, rtol=1e-4, atol=1e-8)
