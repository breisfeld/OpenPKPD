"""Focused unit tests for IndividualModel and PopulationModel numerics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.blq import blq_log_likelihood
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.data.event_processor import DoseEvent, SubjectEvents
from openpkpd.model.derivative_kernels import DerivativeKernelCapabilities
from openpkpd.model.individual import IndividualModel
from openpkpd.model.parameters import ParameterSet
from openpkpd.model.population import PopulationModel
from openpkpd.model.residuals import log_likelihood_normal
from openpkpd.parser.code_compiler import NMTRANCompiler
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.pk.analytical.advan4 import ADVAN4
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.constants import BLQMethod


class _DummyPK(PKSubroutine):
    advan = 1

    def __init__(self) -> None:
        self.solve_calls: list[dict[str, object]] = []
        self.last_solve_kwargs: dict[str, object] = {}

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        pk_callable=None,
        des_callable=None,
        **kwargs,
    ) -> PKSolution:
        obs_times = np.asarray(obs_times, dtype=float)
        self.last_solve_kwargs = dict(kwargs)
        self.solve_calls.append(
            {"pk_params": dict(pk_params), "obs_times": obs_times.copy(), "kwargs": dict(kwargs)}
        )
        base = float(pk_params.get("BASE", 0.0))
        slope = float(pk_params.get("SLOPE", 1.0))
        f_shift = float(pk_params.get("F_SHIFT", 0.0))
        ipred = base + slope * obs_times
        return PKSolution(
            times=obs_times,
            amounts=ipred[:, None],
            ipred=ipred,
            f=ipred + f_shift,
        )

    def apply_trans(self, raw_params: dict[str, float], trans: int) -> dict[str, float]:
        return dict(raw_params)


def _make_subject_events(
    obs_times: list[float],
    obs_dv: list[float],
    *,
    covariate_df: pd.DataFrame | None = None,
    obs_covariates: list[dict[str, object]] | None = None,
) -> SubjectEvents:
    n = len(obs_times)
    return SubjectEvents(
        subject_id=1,
        obs_times=np.asarray(obs_times, dtype=float),
        obs_dv=np.asarray(obs_dv, dtype=float),
        obs_cmt=np.ones(n, dtype=int),
        obs_mdv=np.zeros(n, dtype=int),
        obs_covariates=obs_covariates,
        covariate_df=covariate_df,
    )


def _make_params(
    theta: list[float], omega_diag: list[float], sigma_diag: list[float]
) -> ParameterSet:
    return ParameterSet(
        theta=np.asarray(theta, dtype=float),
        omega=np.diag(np.asarray(omega_diag, dtype=float)),
        sigma=np.diag(np.asarray(sigma_diag, dtype=float)),
    )


def _finite_difference_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        offset = np.zeros_like(x)
        offset[i] = eps
        grad[i] = (f(x + offset) - f(x - offset)) / (2.0 * eps)
    return grad


def _finite_difference_hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    hess = np.zeros((len(x), len(x)), dtype=float)
    for i in range(len(x)):
        for j in range(i, len(x)):
            x_pp = x.copy()
            x_pp[i] += eps
            x_pp[j] += eps
            x_pm = x.copy()
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp = x.copy()
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm = x.copy()
            x_mm[i] -= eps
            x_mm[j] -= eps
            hess[i, j] = hess[j, i] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * eps * eps)
    return hess


def _finite_difference_jacobian(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x), dtype=float)
    jac = np.zeros((len(f0), len(x)), dtype=float)
    for i in range(len(x)):
        offset = np.zeros_like(x)
        offset[i] = eps
        jac[:, i] = (
            np.asarray(f(x + offset), dtype=float) - np.asarray(f(x - offset), dtype=float)
        ) / (2.0 * eps)
    return jac


def _make_symbolic_advan2_model(
    error_code: str,
    *,
    n_eps: int,
    obs_times: np.ndarray,
    obs_dv: np.ndarray,
    dose_events: list[DoseEvent],
    pk_code: str | None = None,
    covariate_df: pd.DataFrame | None = None,
) -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=SubjectEvents(
            subject_id=1,
            dose_events=dose_events,
            obs_times=np.asarray(obs_times, dtype=float),
            obs_dv=np.asarray(obs_dv, dtype=float),
            obs_cmt=np.ones(len(obs_times), dtype=int),
            obs_mdv=np.zeros(len(obs_times), dtype=int),
            covariate_df=covariate_df,
        ),
        pk_subroutine=ADVAN2(),
        pk_callable=compiler.compile_pk(
            pk_code
            or "KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV = THETA(3)*EXP(ETA(3))"
        ),
        error_callable=compiler.compile_error(error_code),
        n_eps=n_eps,
    )


def _make_symbolic_advan1_model(
    error_code: str,
    *,
    n_eps: int,
    obs_times: np.ndarray,
    obs_dv: np.ndarray,
    dose_events: list[DoseEvent],
) -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=SubjectEvents(
            subject_id=1,
            dose_events=dose_events,
            obs_times=np.asarray(obs_times, dtype=float),
            obs_dv=np.asarray(obs_dv, dtype=float),
            obs_cmt=np.ones(len(obs_times), dtype=int),
            obs_mdv=np.zeros(len(obs_times), dtype=int),
        ),
        pk_subroutine=ADVAN1(),
        pk_callable=compiler.compile_pk("CL = THETA(1)*EXP(ETA(1))\nV = THETA(2)*EXP(ETA(2))"),
        error_callable=compiler.compile_error(error_code),
        n_eps=n_eps,
    )


def _make_symbolic_advan4_model(
    error_code: str,
    *,
    n_eps: int,
    obs_times: np.ndarray,
    obs_dv: np.ndarray,
    dose_events: list[DoseEvent],
    pk_code: str | None = None,
) -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=SubjectEvents(
            subject_id=1,
            dose_events=dose_events,
            obs_times=np.asarray(obs_times, dtype=float),
            obs_dv=np.asarray(obs_dv, dtype=float),
            obs_cmt=np.ones(len(obs_times), dtype=int),
            obs_mdv=np.zeros(len(obs_times), dtype=int),
        ),
        pk_subroutine=ADVAN4(),
        pk_callable=compiler.compile_pk(
            pk_code
            or (
                "KA = THETA(1)*EXP(ETA(1))\n"
                "CL = THETA(2)*EXP(ETA(2))\n"
                "V2 = THETA(3)*EXP(ETA(3))\n"
                "Q = THETA(4)\n"
                "V3 = THETA(5)\n"
                "K = CL/V2\n"
                "K12 = Q/V2\n"
                "K21 = Q/V3"
            )
        ),
        error_callable=compiler.compile_error(error_code),
        n_eps=n_eps,
    )


def _make_symbolic_advan3_model(
    pk_code: str,
    error_code: str,
    *,
    n_eps: int,
    obs_times: np.ndarray,
    obs_dv: np.ndarray,
    dose_events: list[DoseEvent],
) -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=SubjectEvents(
            subject_id=1,
            dose_events=dose_events,
            obs_times=np.asarray(obs_times, dtype=float),
            obs_dv=np.asarray(obs_dv, dtype=float),
            obs_cmt=np.ones(len(obs_times), dtype=int),
            obs_mdv=np.zeros(len(obs_times), dtype=int),
        ),
        pk_subroutine=ADVAN3(),
        pk_callable=compiler.compile_pk(pk_code),
        error_callable=compiler.compile_error(error_code),
        n_eps=n_eps,
    )


def _assert_symbolic_derivatives_match_fd(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: IndividualModel,
    theta: np.ndarray,
    eta: np.ndarray,
    omega: np.ndarray,
    sigma: np.ndarray,
    case_name: str,
    grad_eps: float = 1e-6,
    jac_eps: float | None = None,
    hess_eps: float = 1e-4,
    grad_rtol: float = 2e-4,
    grad_atol: float = 2e-5,
    jac_rtol: float = 2e-4,
    jac_atol: float = 2e-5,
    hess_rtol: float = 4e-3,
    hess_atol: float = 4e-4,
    value_rtol: float = 1e-9,
    value_atol: float = 1e-9,
    trans: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if jac_eps is None:
        jac_eps = grad_eps

    kernel = model.get_subject_derivative_kernel(trans)
    assert kernel is not None, case_name
    # Verify minimum required capabilities are present; allow additional
    # capabilities (e.g. theta_data_objective_gradient) without failing.
    assert kernel.capabilities.eta_objective_gradient, f"{case_name}: eta_objective_gradient"
    assert kernel.capabilities.eta_objective_hessian, f"{case_name}: eta_objective_hessian"
    assert kernel.capabilities.prediction_eta_jacobian, f"{case_name}: prediction_eta_jacobian"

    symbolic_value, symbolic_grad = model.symbolic_obj_eta_value_grad(
        eta, theta, omega, sigma, trans=trans
    )
    symbolic_jacobian = model.prediction_eta_jacobian(theta, eta, sigma, trans=trans)
    symbolic_hessian = model.eta_objective_hessian(theta, eta, omega, sigma, trans=trans)
    assert np.allclose(symbolic_hessian, symbolic_hessian.T, rtol=1e-10, atol=1e-10), case_name

    monkeypatch.setattr(model, "get_subject_derivative_kernel", lambda trans_value: None)
    obs_mask = model.subject_events.observation_mask()

    def obj_of_eta(eta_value: np.ndarray) -> float:
        return float(model.obj_eta(eta_value, theta, omega, sigma, trans=trans))

    def pred_of_eta(eta_value: np.ndarray) -> np.ndarray:
        _, _, _, pred, _ = model.evaluate_observation_model(theta, eta_value, sigma, trans=trans)
        return np.asarray(pred[obs_mask], dtype=float)

    generic_value = obj_of_eta(eta)
    generic_grad = _finite_difference_gradient(obj_of_eta, eta, eps=grad_eps)
    generic_jacobian = _finite_difference_jacobian(pred_of_eta, eta, eps=jac_eps)
    generic_hessian = _finite_difference_hessian(obj_of_eta, eta, eps=hess_eps)

    assert symbolic_value == pytest.approx(generic_value, rel=value_rtol, abs=value_atol), case_name
    np.testing.assert_allclose(
        symbolic_grad, generic_grad, rtol=grad_rtol, atol=grad_atol, err_msg=case_name
    )
    np.testing.assert_allclose(
        symbolic_jacobian, generic_jacobian, rtol=jac_rtol, atol=jac_atol, err_msg=case_name
    )
    np.testing.assert_allclose(
        symbolic_hessian, generic_hessian, rtol=hess_rtol, atol=hess_atol, err_msg=case_name
    )
    return symbolic_grad, symbolic_jacobian, symbolic_hessian


class TestIndividualModel:
    def test_evaluate_stitches_iov_predictions_by_occasion(self) -> None:
        events = _make_subject_events([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0])
        occasion_indices = np.array([1, 2, 1, 3], dtype=int)
        pk = _DummyPK()

        def pk_callable(theta, eta, t=0.0, covariates=None):
            occ = float(covariates["OCC"])
            return {"BASE": 10.0 * occ, "SLOPE": 1.0}

        model = IndividualModel(
            subject_events=events,
            pk_subroutine=pk,
            pk_callable=pk_callable,
            error_callable=None,
            occasion_indices=occasion_indices,
        )

        ipred, obs_mask, f = model.evaluate(np.array([0.0]), np.array([0.0]), np.array([[1.0]]))

        np.testing.assert_allclose(ipred, [11.0, 22.0, 13.0, 34.0])
        np.testing.assert_array_equal(obs_mask, np.array([True, True, True, True]))
        np.testing.assert_allclose(f, ipred)
        assert len(pk.solve_calls) == 3

    def test_evaluate_passes_time_varying_covariate_fn_to_solver(self) -> None:
        covariate_df = pd.DataFrame([{"TIME": 0.0, "WT": 70.0}, {"TIME": 5.0, "WT": 80.0}])
        events = _make_subject_events([1.0, 6.0], [0.0, 0.0], covariate_df=covariate_df)
        pk = _DummyPK()

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": float(covariates["WT"]), "SLOPE": 0.0}

        model = IndividualModel(
            subject_events=events,
            pk_subroutine=pk,
            pk_callable=pk_callable,
            error_callable=None,
            des_callable=lambda t, a, pk_params, theta, eta: a,
        )

        model.evaluate(np.array([0.0]), np.array([0.0]), np.array([[1.0]]))

        assert pk.last_solve_kwargs["covariate_change_times"] == [0.0, 5.0]
        covariate_fn = pk.last_solve_kwargs["covariate_fn"]
        assert callable(covariate_fn)
        assert covariate_fn(2.0)["BASE"] == pytest.approx(70.0)
        assert covariate_fn(6.0)["BASE"] == pytest.approx(80.0)

    def test_evaluate_requests_no_amounts_when_prediction_only(self) -> None:
        events = _make_subject_events([1.0, 2.0], [0.0, 0.0])
        pk = _DummyPK()

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": 1.0, "SLOPE": 2.0}

        model = IndividualModel(events, pk, pk_callable, error_callable=None)

        model.evaluate(np.array([0.0]), np.array([0.0]), np.array([[1.0]]))

        assert pk.last_solve_kwargs["return_amounts"] is False

    def test_evaluate_observation_model_uses_error_output_y_and_w(self) -> None:
        events = _make_subject_events([1.0, 2.0], [10.0, np.nan])
        pk = _DummyPK()
        calls: list[tuple[float, float, float, float]] = []

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": 1.0, "SLOPE": 2.0, "F_SHIFT": 10.0}

        def error_callable(theta, eta, eps, f, ipred=None, dv=None, t=0.0, **kwargs):
            calls.append((float(f), float(ipred), float(dv), float(t)))
            return {"Y": float(f) + 0.5, "W": 2.0}

        model = IndividualModel(events, pk, pk_callable, error_callable)

        ipred, obs_mask, f, pred, var = model.evaluate_observation_model(
            np.array([0.0]), np.array([0.0]), np.array([[9.0]])
        )

        np.testing.assert_allclose(ipred, [3.0, 5.0])
        np.testing.assert_array_equal(obs_mask, np.array([True, True]))
        np.testing.assert_allclose(f, [13.0, 15.0])
        np.testing.assert_allclose(pred, [13.5, 15.5])
        np.testing.assert_allclose(var, [36.0, 36.0])
        assert calls[0] == pytest.approx((13.0, 3.0, 10.0, 1.0))
        assert calls[1][0:2] == pytest.approx((15.0, 5.0))
        assert np.isnan(calls[1][2])
        assert calls[1][3] == pytest.approx(2.0)

    def test_evaluate_observation_model_passes_amounts_and_per_observation_covariates(self) -> None:
        events = _make_subject_events(
            [1.0, 1.0],
            [10.0, 20.0],
            obs_covariates=[{"DVID": 1.0}, {"DVID": 2.0}],
        )
        events.obs_cmt = np.array([2, 3], dtype=int)
        pk = _DummyPK()
        seen: list[tuple[float, float, float]] = []

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": 1.0, "SLOPE": 2.0, "F_SHIFT": 10.0}

        def error_callable(
            theta, eta, eps, f, ipred=None, dv=None, t=0.0, a=None, covariates=None, **kwargs
        ):
            seen.append((float(a[0]), float(covariates["DVID"]), float(dv)))
            return {"Y": float(f)}

        model = IndividualModel(events, pk, pk_callable, error_callable)

        _, _, _, pred, _ = model.evaluate_observation_model(
            np.array([0.0]), np.array([0.0]), np.array([[1.0]])
        )

        np.testing.assert_allclose(pred, [13.0, 13.0])
        assert seen[0] == (3.0, 1.0, 10.0)
        assert seen[2] == (3.0, 2.0, 20.0)

    def test_evaluate_observation_model_infers_proportional_variance_from_eps_sensitivity(
        self,
    ) -> None:
        events = _make_subject_events([1.0, 2.0], [10.0, 20.0])

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": 1.0, "SLOPE": 2.0, "F_SHIFT": 10.0}

        def error_callable(theta, eta, eps, f, ipred=None, dv=None, t=0.0, sigma=None, **kwargs):
            return {"Y": float(f) * (1.0 + float(eps[0]))}

        model = IndividualModel(events, _DummyPK(), pk_callable, error_callable)

        _, _, f, pred, var = model.evaluate_observation_model(
            np.array([0.0]), np.array([0.0]), np.array([[0.25]])
        )

        np.testing.assert_allclose(pred, f)
        np.testing.assert_allclose(var, [13.0**2 * 0.25, 15.0**2 * 0.25])

    def test_evaluate_observation_model_infers_combined_variance_from_multiple_eps_terms(
        self,
    ) -> None:
        events = _make_subject_events([1.0, 2.0], [10.0, 20.0])

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": 1.0, "SLOPE": 2.0, "F_SHIFT": 10.0}

        def error_callable(theta, eta, eps, f, ipred=None, dv=None, t=0.0, sigma=None, **kwargs):
            return {"Y": float(f) + float(f) * float(eps[0]) + float(eps[1])}

        model = IndividualModel(events, _DummyPK(), pk_callable, error_callable, n_eps=2)

        _, _, f, pred, var = model.evaluate_observation_model(
            np.array([0.0]), np.array([0.0]), np.diag([0.04, 0.25])
        )

        np.testing.assert_allclose(pred, f)
        np.testing.assert_allclose(var, [13.0**2 * 0.04 + 0.25, 15.0**2 * 0.04 + 0.25])

    def test_evaluate_observation_model_caches_error_signature_fallback(self) -> None:
        events = _make_subject_events([1.0, 2.0], [10.0, 20.0])

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": 1.0, "SLOPE": 2.0, "F_SHIFT": 10.0}

        class _FallbackErrorCallable:
            def __init__(self) -> None:
                self.full_attempts = 0
                self.reduced_attempts = 0

            def __call__(self, theta, eta, eps, f, ipred=None, dv=None, t=0.0, **kwargs):
                if "sigma" in kwargs:
                    self.full_attempts += 1
                    raise TypeError("unexpected keyword argument 'sigma'")
                self.reduced_attempts += 1
                return {"Y": float(f) * (1.0 + float(eps[0]))}

        error_callable = _FallbackErrorCallable()
        model = IndividualModel(events, _DummyPK(), pk_callable, error_callable)

        _, _, f, pred, var = model.evaluate_observation_model(
            np.array([0.0]), np.array([0.0]), np.array([[0.25]])
        )

        np.testing.assert_allclose(pred, f)
        np.testing.assert_allclose(var, [13.0**2 * 0.25, 15.0**2 * 0.25])
        assert error_callable.full_attempts == 1
        assert error_callable.reduced_attempts == 4

    def test_evaluate_observation_model_uses_compiled_error_raw_fast_path(self) -> None:
        from openpkpd.parser.code_compiler import NMTRANCompiler

        events = _make_subject_events([1.0, 2.0], [10.0, 20.0])
        pk = _DummyPK()

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": 1.0, "SLOPE": 2.0, "F_SHIFT": 10.0}

        error_callable = NMTRANCompiler().compile_error("W = THETA(1)*F\nY = F + W*EPS(1)")
        model = IndividualModel(events, pk, pk_callable, error_callable)

        _, _, f, pred, var = model.evaluate_observation_model(
            np.array([0.1]), np.array([0.0]), np.array([[0.25]])
        )

        np.testing.assert_allclose(pred, f)
        np.testing.assert_allclose(var, [(0.1 * 13.0) ** 2 * 0.25, (0.1 * 15.0) ** 2 * 0.25])
        assert pk.last_solve_kwargs["return_amounts"] is False

    def test_evaluate_observation_model_keeps_amounts_when_error_uses_a(self) -> None:
        from openpkpd.parser.code_compiler import NMTRANCompiler

        events = _make_subject_events([1.0, 2.0], [10.0, 20.0])
        pk = _DummyPK()

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": 1.0, "SLOPE": 2.0, "F_SHIFT": 10.0}

        error_callable = NMTRANCompiler().compile_error("Y = F + A(1)*EPS(1)")
        model = IndividualModel(events, pk, pk_callable, error_callable)

        _, _, f, pred, _ = model.evaluate_observation_model(
            np.array([0.0]), np.array([0.0]), np.array([[0.25]])
        )

        np.testing.assert_allclose(pred, f)
        assert "return_amounts" not in pk.last_solve_kwargs

    def test_evaluate_observation_model_uses_mixed_dvid_fast_path_with_amounts(self) -> None:
        events = _make_subject_events(
            [1.0, 2.0],
            [10.0, 20.0],
            obs_covariates=[{"DVID": 1.0}, {"DVID": 2.0}],
        )
        compiler = NMTRANCompiler()

        class _MixedPK(PKSubroutine):
            advan = 6

            def __init__(self) -> None:
                self.last_solve_kwargs: dict[str, object] = {}

            def solve(
                self,
                pk_params: dict[str, float],
                dose_events: list,
                obs_times: np.ndarray,
                pk_callable=None,
                des_callable=None,
                **kwargs,
            ) -> PKSolution:
                self.last_solve_kwargs = dict(kwargs)
                obs_times = np.asarray(obs_times, dtype=float)
                f = np.array([10.0, 20.0], dtype=float)
                amounts = np.array(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                    ],
                    dtype=float,
                )
                return PKSolution(times=obs_times, amounts=amounts, ipred=f, f=f)

            def apply_trans(self, raw_params: dict[str, float], trans: int) -> dict[str, float]:
                return dict(raw_params)

        error_callable = compiler.compile_error(
            "PKPROP = THETA(9)\n"
            "PKADD = THETA(10)\n"
            "PDADD = THETA(11)\n"
            "IPRED = THETA(8) + A(4)\n"
            "W = PDADD\n"
            "Y = IPRED + W*EPS(2)\n"
            "IF (DVID == 1) W = SQRT((PKPROP*F)**2 + PKADD**2)\n"
            "IF (DVID == 1) Y = F + W*EPS(1)"
        )
        model = IndividualModel(events, _MixedPK(), None, error_callable, n_eps=2)
        theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.2, 3.0, 5.0], dtype=float)
        sigma = np.diag([0.25, 0.04])

        _, _, f, pred, var = model.evaluate_observation_model(theta, np.array([]), sigma)

        assert model._common_error_model == ("mixed_pkpd_dvid_theta", (7, 3, 8, 9, 10))
        np.testing.assert_allclose(f, [10.0, 20.0])
        np.testing.assert_allclose(pred, [10.0, 108.0])
        np.testing.assert_allclose(var, [((0.2 * 10.0) ** 2 + 3.0**2) * 0.25, 5.0**2 * 0.04])

    def test_log_likelihood_m6_counts_only_first_blq(self) -> None:
        events = _make_subject_events([1.0, 2.0, 3.0], [0.25, 0.10, 2.0])
        pk = _DummyPK()

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": 1.0, "SLOPE": 0.0}

        model = IndividualModel(
            subject_events=events,
            pk_subroutine=pk,
            pk_callable=pk_callable,
            error_callable=None,
            blq_method=BLQMethod.M6,
            lloq=0.5,
        )

        result = model.log_likelihood(np.array([0.0]), np.array([0.0]), np.array([[1.0]]))

        expected_ll = blq_log_likelihood(0.25, 1.0, 1.0, 0.5, BLQMethod.M6)
        expected_ll += log_likelihood_normal(2.0, 1.0, 1.0)
        assert result == pytest.approx(-2.0 * expected_ll)

    def test_evaluate_observation_model_uses_native_advan6_mixed_pkpd_probe_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        compiler = NMTRANCompiler()
        events = SubjectEvents(
            subject_id=1,
            dose_events=[DoseEvent(time=0.0, amount=100.0, compartment=1)],
            obs_times=np.array([0.5, 24.0], dtype=float),
            obs_dv=np.array([0.0, 44.0], dtype=float),
            obs_cmt=np.array([3, 4], dtype=int),
            obs_mdv=np.zeros(2, dtype=int),
            obs_covariates=[{"DVID": 1.0}, {"DVID": 2.0}],
        )

        class _NativeMixedPK(PKSubroutine):
            advan = 6
            n_compartments = 10

            def solve(self, *args, **kwargs):
                raise AssertionError("native probe should bypass python solve()")

            def apply_trans(self, raw_params: dict[str, float], trans: int) -> dict[str, float]:
                return dict(raw_params)

        calls: list[tuple[list[float], float, list[float]]] = []

        # The code prefers the multidose probe; patch it with the
        # multidose signature (obs_times, dose_times, dose_amts, theta).
        def _fake_multidose_probe(
            times: list[float],
            dose_times: list[float],
            dose_amts: list[float],
            theta8: list[float],
        ) -> list[list[float]]:
            calls.append((list(times), list(dose_times), list(dose_amts), list(theta8)))
            return [
                [80.0, 10.0, 30.0, -5.0],
                [0.1, 0.2, 12.0, -7.5],
            ]

        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_multidose_rust",
            _fake_multidose_probe,
        )
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],  # kept so contract builder finds it non-None
        )

        error_callable = compiler.compile_error(
            "PKPROP = THETA(9)\n"
            "PKADD = THETA(10)\n"
            "PDADD = THETA(11)\n"
            "IPRED = THETA(8) + A(4)\n"
            "W = PDADD\n"
            "Y = IPRED + W*EPS(2)\n"
            "IF (DVID == 1) W = SQRT((PKPROP*F)**2 + PKADD**2)\n"
            "IF (DVID == 1) Y = F + W*EPS(1)"
        )

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {
                "KTR": 0.8968,
                "KA": 0.8887,
                "CL": 0.1337,
                "V": 8.6756,
                "EMAX": 0.999,
                "EC50": 1.5735,
                "KOUT": 0.0552,
                "E0": 101.3225,
                "PCMT": 3.0,
            }

        model = IndividualModel(
            subject_events=events,
            pk_subroutine=_NativeMixedPK(),
            pk_callable=pk_callable,
            error_callable=error_callable,
            n_eps=2,
        )

        theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 101.3225, 0.2, 3.0, 5.0], dtype=float)
        sigma = np.diag([0.25, 0.04])
        ipred, _, f, pred, _ = model.evaluate_observation_model(theta, np.array([]), sigma, trans=1)

        assert len(calls) == 1
        # Multidose probe signature: (obs_times, dose_times, dose_amts, theta)
        assert calls[0][0] == [0.5, 24.0]          # obs_times
        assert calls[0][1] == [0.0]                 # dose_times: single dose at t=0
        assert calls[0][2] == pytest.approx([100.0])  # dose_amts
        np.testing.assert_allclose(calls[0][3], [0.8968, 0.8887, 0.1337, 8.6756, 0.999, 1.5735, 0.0552, 101.3225])
        np.testing.assert_allclose(ipred, [30.0 / 8.6756, 12.0 / 8.6756])
        np.testing.assert_allclose(f, ipred)
        np.testing.assert_allclose(pred, [30.0 / 8.6756, 101.3225 - 7.5])

    def test_obj_eta_with_iov_etas_sums_block_penalties(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events = _make_subject_events([1.0, 2.0], [0.0, 0.0])
        model = IndividualModel(
            subject_events=events,
            pk_subroutine=_DummyPK(),
            pk_callable=None,
            error_callable=None,
            occasion_indices=np.array([1, 2], dtype=int),
        )
        monkeypatch.setattr(model, "log_likelihood", lambda *args, **kwargs: 7.0)

        eta = np.array([1.0, 2.0, 3.0, 4.0])
        omega = np.diag([2.0, 8.0])

        result = model.obj_eta(eta, np.array([0.0]), omega, np.array([[1.0]]))

        expected_penalty = (1.0**2) / 2.0 + (2.0**2) / 8.0 + (3.0**2) / 2.0 + (4.0**2) / 8.0
        assert result == pytest.approx(7.0 + expected_penalty)

    @pytest.mark.parametrize(
        ("case_name", "error_code", "theta", "sigma", "n_eps", "eta"),
        [
            (
                "proportional",
                "Y = F*(1 + EPS(1))",
                np.array([1.2, 2.5, 30.0]),
                np.array([[0.01]]),
                1,
                np.array([0.1, -0.2, 0.05]),
            ),
            (
                "additive",
                "Y = F + EPS(1)",
                np.array([1.2, 2.5, 30.0]),
                np.array([[0.09]]),
                1,
                np.array([0.1, -0.2, 0.05]),
            ),
            (
                "combined_two_eps",
                "Y = F + EPS(1) + F*EPS(2)",
                np.array([1.2, 2.5, 30.0]),
                np.array([[0.04, 0.01], [0.01, 0.09]]),
                2,
                np.array([0.1, -0.2, 0.05]),
            ),
            (
                "combined_theta",
                "W = SQRT(THETA(4)**2 + (F*THETA(5))**2)\nY = F + W*EPS(1)\nIRES = DV - F\nIWRES = IRES / W",
                np.array([1.2, 2.5, 30.0, 0.6, 0.15]),
                np.array([[1.0]]),
                1,
                np.array([0.1, -0.2, 0.05]),
            ),
        ],
        ids=["proportional", "additive", "combined_two_eps", "combined_theta"],
    )
    def test_symbolic_advan2_eta_derivatives_match_independent_finite_differences(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case_name: str,
        error_code: str,
        theta: np.ndarray,
        sigma: np.ndarray,
        n_eps: int,
        eta: np.ndarray,
    ) -> None:
        pytest.importorskip("sympy")
        obs_times = np.array([0.5, 1.0, 2.0, 4.0, 7.0, 12.0, 24.0], dtype=float)
        obs_dv = np.array([7.5, 8.2, 7.1, 5.0, 3.2, 1.6, 0.3], dtype=float)
        model = _make_symbolic_advan2_model(
            error_code,
            n_eps=n_eps,
            obs_times=obs_times,
            obs_dv=obs_dv,
            dose_events=[
                DoseEvent(time=0.0, amount=320.0, compartment=1),
                DoseEvent(time=6.0, amount=160.0, compartment=1),
            ],
        )
        omega = np.diag([0.04, 0.02, 0.02])
        _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name=case_name,
        )

    def test_symbolic_advan2_eta_derivatives_match_finite_differences_in_ka_equals_k_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan2_model(
            "Y = F*(1 + EPS(1))",
            n_eps=1,
            obs_times=np.array([0.25, 0.75, 1.5, 3.0, 6.0, 12.0], dtype=float),
            obs_dv=np.array([7.0, 8.4, 7.5, 5.2, 2.9, 0.9], dtype=float),
            dose_events=[DoseEvent(time=0.0, amount=300.0, compartment=1)],
        )
        theta = np.array([1.0, 10.0, 10.0])
        eta = np.zeros(3, dtype=float)
        omega = np.diag([0.05, 0.03, 0.02])
        sigma = np.array([[0.02]])
        _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name="ka_equals_k_limit",
            grad_eps=2e-6,
            jac_eps=2e-6,
            hess_eps=2e-4,
            grad_rtol=1e-3,
            grad_atol=1e-4,
            jac_rtol=1e-3,
            jac_atol=1e-4,
            hess_rtol=1e-2,
            hess_atol=2e-3,
        )

    @pytest.mark.parametrize(
        (
            "case_name",
            "error_code",
            "theta",
            "sigma",
            "n_eps",
            "eta",
            "obs_times",
            "obs_dv",
            "dose_events",
            "kwargs",
        ),
        [
            (
                "proportional_alias",
                "Y = F + F*EPS(1)",
                np.array([1.15, 2.4, 28.0]),
                np.array([[0.02]]),
                1,
                np.array([0.08, -0.12, 0.03]),
                np.array([0.4, 1.2, 2.5, 5.0, 8.0, 16.0]),
                np.array([6.8, 7.1, 6.2, 4.4, 2.6, 0.8]),
                [DoseEvent(time=0.0, amount=280.0, compartment=1)],
                {},
            ),
            (
                "additive_theta_weighted",
                "W = THETA(4)\nY = F + W*EPS(1)",
                np.array([1.05, 2.1, 24.0, 0.7]),
                np.array([[0.6]]),
                1,
                np.array([-0.05, 0.07, 0.11]),
                np.array([0.5, 1.5, 3.0, 6.0, 10.0]),
                np.array([7.0, 6.4, 4.8, 2.8, 1.1]),
                [DoseEvent(time=0.0, amount=250.0, compartment=1)],
                {},
            ),
            (
                "combined_theta_short_form",
                "W = SQRT(THETA(4)**2 + (F*THETA(5))**2)\nY = F + W*EPS(1)",
                np.array([1.25, 2.8, 31.0, 0.5, 0.12]),
                np.array([[0.8]]),
                1,
                np.array([0.06, -0.09, 0.04]),
                np.array([0.5, 1.0, 2.0, 4.0, 8.0, 14.0]),
                np.array([7.9, 8.0, 6.7, 4.7, 2.5, 0.9]),
                [DoseEvent(time=0.0, amount=300.0, compartment=1)],
                {},
            ),
            (
                "combined_two_eps_negative_covariance",
                "Y = F + EPS(1) + F*EPS(2)",
                np.array([1.18, 2.35, 27.0]),
                np.array([[0.09, -0.015], [-0.015, 0.04]]),
                2,
                np.array([0.09, -0.07, 0.02]),
                np.array([0.6, 1.5, 3.0, 7.0, 12.0]),
                np.array([6.9, 7.2, 5.8, 2.9, 1.0]),
                [DoseEvent(time=0.0, amount=260.0, compartment=1)],
                {},
            ),
            (
                "exact_dose_rows",
                "Y = F + EPS(1)",
                np.array([1.1, 2.2, 26.0]),
                np.array([[0.04]]),
                1,
                np.array([0.03, -0.04, 0.08]),
                np.array([0.0, 0.25, 4.0, 4.0, 4.5, 9.0]),
                np.array([0.2, 5.8, 0.1, 0.1, 4.1, 1.3]),
                [
                    DoseEvent(time=0.0, amount=300.0, compartment=1),
                    DoseEvent(time=4.0, amount=150.0, compartment=1),
                ],
                {},
            ),
            (
                "near_limit_small_gap",
                "Y = F*(1 + EPS(1))",
                np.array([1.0, 10.0, 10.0]),
                np.array([[0.03]]),
                1,
                np.array([2e-6, 0.0, 0.0]),
                np.array([0.3, 0.9, 2.0, 4.0, 8.0]),
                np.array([6.9, 8.0, 6.1, 3.8, 1.4]),
                [DoseEvent(time=0.0, amount=290.0, compartment=1)],
                {
                    "grad_eps": 2e-7,
                    "jac_eps": 5e-7,
                    "hess_eps": 5e-5,
                    "grad_rtol": 2e-3,
                    "grad_atol": 2e-4,
                    "jac_rtol": 2e-3,
                    "jac_atol": 7e-4,
                    "hess_rtol": 2e-2,
                    "hess_atol": 4e-3,
                },
            ),
            (
                "near_limit_moderate_gap",
                "Y = F*(1 + EPS(1))",
                np.array([1.0, 10.0, 10.0]),
                np.array([[0.03]]),
                1,
                np.array([2e-5, 0.0, 0.0]),
                np.array([0.3, 0.9, 2.0, 4.0, 8.0]),
                np.array([6.9, 8.0, 6.1, 3.8, 1.4]),
                [DoseEvent(time=0.0, amount=290.0, compartment=1)],
                {
                    "grad_eps": 2e-7,
                    "jac_eps": 2e-7,
                    "hess_eps": 5e-5,
                    "grad_rtol": 2e-3,
                    "grad_atol": 2e-4,
                    "jac_rtol": 2e-3,
                    "jac_atol": 5e-4,
                    "hess_rtol": 2e-2,
                    "hess_atol": 4e-3,
                },
            ),
        ],
        ids=[
            "proportional_alias",
            "additive_theta_weighted",
            "combined_theta_short_form",
            "combined_two_eps_negative_covariance",
            "exact_dose_rows",
            "near_limit_small_gap",
            "near_limit_moderate_gap",
        ],
    )
    def test_symbolic_advan2_eta_derivatives_cover_edge_case_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case_name: str,
        error_code: str,
        theta: np.ndarray,
        sigma: np.ndarray,
        n_eps: int,
        eta: np.ndarray,
        obs_times: np.ndarray,
        obs_dv: np.ndarray,
        dose_events: list[DoseEvent],
        kwargs: dict[str, float],
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan2_model(
            error_code,
            n_eps=n_eps,
            obs_times=obs_times,
            obs_dv=obs_dv,
            dose_events=dose_events,
        )
        omega = np.diag([0.04, 0.02, 0.02])
        symbolic_grad, symbolic_jacobian, _symbolic_hessian = _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name=case_name,
            **kwargs,
        )

        if case_name == "exact_dose_rows":
            np.testing.assert_allclose(symbolic_jacobian[[0], :], 0.0, atol=1e-10)
        if case_name.startswith("near_limit"):
            assert np.all(np.isfinite(symbolic_grad)), case_name
            assert np.all(np.isfinite(symbolic_jacobian)), case_name

    def test_symbolic_advan2_eta_derivatives_support_static_power_covariate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan2_model(
            "Y = F*(1 + EPS(1))",
            n_eps=1,
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float),
            obs_dv=np.array([6.8, 7.7, 6.0, 3.7, 1.5], dtype=float),
            dose_events=[DoseEvent(time=0.0, amount=290.0, compartment=1)],
            pk_code=(
                "KA = THETA(1)*EXP(ETA(1))\n"
                "CL = THETA(2)*EXP(ETA(2))\n"
                "V = THETA(3)*EXP(ETA(3))\n"
                "CL = CL * (WT/70.0)**THETA(4)"
            ),
            covariate_df=pd.DataFrame({"TIME": [0.0], "WT": [82.0]}),
        )
        theta = np.array([1.3, 1.9, 13.0, 0.75])
        eta = np.array([0.04, -0.05, 0.06])
        omega = np.diag([0.04, 0.03, 0.02])
        sigma = np.array([[0.03]])

        _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name="advan2_static_power_covariate",
            trans=2,
            jac_rtol=5e-4,
            jac_atol=5e-5,
            hess_rtol=6e-3,
            hess_atol=7e-4,
        )

    @pytest.mark.parametrize(
        ("case_name", "error_code", "theta", "sigma", "n_eps", "eta"),
        [
            (
                "proportional",
                "Y = F*(1 + EPS(1))",
                np.array([2.4, 28.0]),
                np.array([[0.02]]),
                1,
                np.array([0.08, -0.12]),
            ),
            (
                "additive",
                "Y = F + EPS(1)",
                np.array([2.1, 24.0]),
                np.array([[0.09]]),
                1,
                np.array([-0.04, 0.1]),
            ),
            (
                "combined_two_eps",
                "Y = F + EPS(1) + F*EPS(2)",
                np.array([2.2, 26.0]),
                np.array([[0.06, -0.01], [-0.01, 0.03]]),
                2,
                np.array([0.06, -0.08]),
            ),
            (
                "ipred_prop_theta",
                "IPRED = F\nW = IPRED * THETA(3)\nY = IPRED + W * EPS(1)",
                np.array([2.0, 25.0, 0.18]),
                np.array([[1.0]]),
                1,
                np.array([0.03, -0.05]),
            ),
            (
                "ipred_combined_theta",
                "IPRED = F\nW = SQRT(THETA(3)**2 + (IPRED*THETA(4))**2)\nY = IPRED + W * EPS(1)",
                np.array([2.0, 25.0, 0.4, 0.12]),
                np.array([[0.8]]),
                1,
                np.array([0.03, -0.05]),
            ),
        ],
        ids=[
            "proportional",
            "additive",
            "combined_two_eps",
            "ipred_prop_theta",
            "ipred_combined_theta",
        ],
    )
    def test_symbolic_advan1_eta_derivatives_match_independent_finite_differences(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case_name: str,
        error_code: str,
        theta: np.ndarray,
        sigma: np.ndarray,
        n_eps: int,
        eta: np.ndarray,
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan1_model(
            error_code,
            n_eps=n_eps,
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=float),
            obs_dv=np.array([7.2, 6.8, 5.7, 4.0, 2.1, 0.9], dtype=float),
            dose_events=[
                DoseEvent(time=0.0, amount=280.0, compartment=1),
                DoseEvent(time=6.0, amount=140.0, compartment=1),
            ],
        )
        omega = np.diag([0.04, 0.02])
        _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name=f"advan1_{case_name}",
        )

    @pytest.mark.parametrize(
        (
            "case_name",
            "error_code",
            "theta",
            "sigma",
            "n_eps",
            "eta",
            "obs_times",
            "obs_dv",
            "dose_events",
            "kwargs",
        ),
        [
            (
                "exact_dose_rows",
                "Y = F + EPS(1)",
                np.array([2.0, 22.0]),
                np.array([[0.05]]),
                1,
                np.array([0.04, -0.03]),
                np.array([0.0, 0.5, 4.0, 4.0, 6.0, 12.0]),
                np.array([0.1, 6.2, 4.0, 4.0, 5.0, 1.8]),
                [
                    DoseEvent(time=0.0, amount=260.0, compartment=1),
                    DoseEvent(time=4.0, amount=120.0, compartment=1),
                ],
                {},
            ),
            (
                "multiple_bolus_superposition",
                "IPRED = F\nW = IPRED * THETA(3)\nY = IPRED + W * EPS(1)",
                np.array([1.8, 20.0, 0.15]),
                np.array([[1.0]]),
                1,
                np.array([0.02, 0.05]),
                np.array([0.5, 1.5, 3.0, 5.0, 8.0, 12.0]),
                np.array([7.8, 6.5, 5.0, 4.1, 2.6, 1.2]),
                [
                    DoseEvent(time=0.0, amount=180.0, compartment=1),
                    DoseEvent(time=2.5, amount=90.0, compartment=1),
                    DoseEvent(time=7.0, amount=60.0, compartment=1),
                ],
                {},
            ),
            (
                "negative_covariance_combined_eps",
                "Y = F + EPS(1) + F*EPS(2)",
                np.array([2.3, 27.0]),
                np.array([[0.08, -0.02], [-0.02, 0.05]]),
                2,
                np.array([-0.03, 0.09]),
                np.array([0.4, 1.0, 2.5, 5.0, 9.0]),
                np.array([6.9, 6.2, 4.7, 2.9, 1.1]),
                [DoseEvent(time=0.0, amount=240.0, compartment=1)],
                {},
            ),
            (
                "slow_elimination_long_tail",
                "Y = F*(1 + EPS(1))",
                np.array([0.35, 30.0]),
                np.array([[0.015]]),
                1,
                np.array([0.04, -0.02]),
                np.array([0.5, 4.0, 12.0, 24.0, 48.0, 72.0]),
                np.array([9.0, 8.2, 6.7, 4.8, 2.5, 1.4]),
                [DoseEvent(time=0.0, amount=320.0, compartment=1)],
                {},
            ),
        ],
        ids=[
            "exact_dose_rows",
            "multiple_bolus_superposition",
            "negative_covariance_combined_eps",
            "slow_elimination_long_tail",
        ],
    )
    def test_symbolic_advan1_eta_derivatives_cover_edge_case_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case_name: str,
        error_code: str,
        theta: np.ndarray,
        sigma: np.ndarray,
        n_eps: int,
        eta: np.ndarray,
        obs_times: np.ndarray,
        obs_dv: np.ndarray,
        dose_events: list[DoseEvent],
        kwargs: dict[str, float],
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan1_model(
            error_code,
            n_eps=n_eps,
            obs_times=obs_times,
            obs_dv=obs_dv,
            dose_events=dose_events,
        )
        omega = np.diag([0.04, 0.02])
        symbolic_grad, symbolic_jacobian, _symbolic_hessian = _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name=f"advan1_{case_name}",
            **kwargs,
        )

        if case_name == "exact_dose_rows":
            np.testing.assert_allclose(symbolic_jacobian[[0], :], 0.0, atol=1e-10)
        assert np.all(np.isfinite(symbolic_grad)), case_name
        assert np.all(np.isfinite(symbolic_jacobian)), case_name

    @pytest.mark.parametrize(
        ("case_name", "pk_code", "error_code", "theta", "sigma", "n_eps", "eta", "omega"),
        [
            (
                "library_prop_theta",
                "CL = THETA(1)*EXP(ETA(1))\nV1 = THETA(2)*EXP(ETA(2))\nQ = THETA(3)\nV2 = THETA(4)",
                "IPRED = F\nW = IPRED * THETA(5)\nY = IPRED + W * EPS(1)",
                np.array([1.7, 12.0, 0.65, 18.0, 0.12]),
                np.array([[1.0]]),
                1,
                np.array([0.05, -0.08]),
                np.diag([0.04, 0.03]),
            ),
            (
                "library_additive",
                "CL = THETA(1)*EXP(ETA(1))\nV1 = THETA(2)*EXP(ETA(2))\nQ = THETA(3)\nV2 = THETA(4)",
                "Y = F + EPS(1)",
                np.array([1.5, 11.0, 0.55, 17.0]),
                np.array([[0.05]]),
                1,
                np.array([-0.04, 0.06]),
                np.diag([0.04, 0.03]),
            ),
            (
                "all_eta_combined_eps",
                "CL = THETA(1)*EXP(ETA(1))\nV1 = THETA(2)*EXP(ETA(2))\nQ = THETA(3)*EXP(ETA(3))\nV2 = THETA(4)*EXP(ETA(4))",
                "Y = F + EPS(1) + F*EPS(2)",
                np.array([1.6, 11.5, 0.6, 16.5]),
                np.array([[0.07, -0.015], [-0.015, 0.03]]),
                2,
                np.array([0.04, -0.05, 0.03, -0.02]),
                np.diag([0.04, 0.03, 0.02, 0.02]),
            ),
            (
                "library_combined_theta",
                "CL = THETA(1)*EXP(ETA(1))\nV1 = THETA(2)*EXP(ETA(2))\nQ = THETA(3)\nV2 = THETA(4)",
                "IPRED = F\nW = SQRT(THETA(5)**2 + (IPRED*THETA(6))**2)\nY = IPRED + W * EPS(1)",
                np.array([1.7, 12.0, 0.65, 18.0, 0.25, 0.1]),
                np.array([[0.8]]),
                1,
                np.array([0.03, -0.05]),
                np.diag([0.04, 0.03]),
            ),
        ],
        ids=[
            "library_prop_theta",
            "library_additive",
            "all_eta_combined_eps",
            "library_combined_theta",
        ],
    )
    def test_symbolic_advan3_eta_derivatives_match_independent_finite_differences(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case_name: str,
        pk_code: str,
        error_code: str,
        theta: np.ndarray,
        sigma: np.ndarray,
        n_eps: int,
        eta: np.ndarray,
        omega: np.ndarray,
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan3_model(
            pk_code,
            error_code,
            n_eps=n_eps,
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=float),
            obs_dv=np.array([7.0, 6.4, 5.1, 3.5, 1.9, 0.8], dtype=float),
            dose_events=[
                DoseEvent(time=0.0, amount=250.0, compartment=1),
                DoseEvent(time=10.0, amount=120.0, compartment=1),
            ],
        )
        _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name=f"advan3_{case_name}",
            trans=4,
            hess_rtol=8e-3,
            hess_atol=1e-3,
        )

    def test_symbolic_advan3_trans1_eta_derivatives_match_independent_finite_differences(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan3_model(
            "CL = THETA(1)*EXP(ETA(1))\nV1 = THETA(2)*EXP(ETA(2))\nQ = THETA(3)\nV2 = THETA(4)\nK = CL/V1\nK12 = Q/V1\nK21 = Q/V2",
            "Y = F*(1 + EPS(1))",
            n_eps=1,
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=float),
            obs_dv=np.array([7.0, 6.4, 5.1, 3.5, 1.9, 0.8], dtype=float),
            dose_events=[
                DoseEvent(time=0.0, amount=250.0, compartment=1),
                DoseEvent(time=10.0, amount=120.0, compartment=1),
            ],
        )
        theta = np.array([1.7, 12.0, 0.65, 18.0])
        eta = np.array([0.05, -0.08])
        omega = np.diag([0.04, 0.03])
        sigma = np.array([[0.02]])
        _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name="advan3_trans1_macro_explicit",
            trans=1,
            hess_rtol=8e-3,
            hess_atol=1e-3,
        )

    @pytest.mark.parametrize(
        (
            "case_name",
            "pk_code",
            "error_code",
            "theta",
            "sigma",
            "n_eps",
            "eta",
            "omega",
            "obs_times",
            "obs_dv",
            "dose_events",
            "kwargs",
        ),
        [
            (
                "exact_dose_rows",
                "CL = THETA(1)*EXP(ETA(1))\nV1 = THETA(2)*EXP(ETA(2))\nQ = THETA(3)\nV2 = THETA(4)",
                "Y = F + EPS(1)",
                np.array([1.5, 11.0, 0.55, 17.0]),
                np.array([[0.05]]),
                1,
                np.array([0.02, -0.03]),
                np.diag([0.04, 0.03]),
                np.array([0.0, 0.5, 6.0, 6.0, 12.0, 24.0]),
                np.array([0.1, 6.9, 4.0, 4.0, 2.0, 0.9]),
                [
                    DoseEvent(time=0.0, amount=260.0, compartment=1),
                    DoseEvent(time=6.0, amount=130.0, compartment=1),
                ],
                {"hess_rtol": 8e-3, "hess_atol": 1e-3},
            ),
            (
                "multiple_bolus_superposition",
                "CL = THETA(1)*EXP(ETA(1))\nV1 = THETA(2)*EXP(ETA(2))\nQ = THETA(3)\nV2 = THETA(4)",
                "IPRED = F\nW = IPRED * THETA(5)\nY = IPRED + W * EPS(1)",
                np.array([1.7, 12.0, 0.65, 18.0, 0.14]),
                np.array([[1.0]]),
                1,
                np.array([0.03, 0.04]),
                np.diag([0.04, 0.03]),
                np.array([0.5, 1.5, 3.0, 5.0, 8.0, 12.0, 20.0]),
                np.array([7.1, 6.5, 5.7, 4.8, 3.4, 2.5, 1.1]),
                [
                    DoseEvent(time=0.0, amount=180.0, compartment=1),
                    DoseEvent(time=3.0, amount=90.0, compartment=1),
                    DoseEvent(time=9.0, amount=60.0, compartment=1),
                ],
                {"hess_rtol": 1e-2, "hess_atol": 2e-3},
            ),
            (
                "negative_covariance_combined_eps",
                "CL = THETA(1)*EXP(ETA(1))\nV1 = THETA(2)*EXP(ETA(2))\nQ = THETA(3)*EXP(ETA(3))\nV2 = THETA(4)*EXP(ETA(4))",
                "Y = F + EPS(1) + F*EPS(2)",
                np.array([1.6, 11.5, 0.6, 16.5]),
                np.array([[0.08, -0.02], [-0.02, 0.04]]),
                2,
                np.array([0.04, -0.05, 0.02, 0.03]),
                np.diag([0.04, 0.03, 0.02, 0.02]),
                np.array([0.4, 1.0, 2.5, 5.0, 9.0, 18.0]),
                np.array([6.8, 6.1, 4.9, 3.1, 1.8, 0.7]),
                [DoseEvent(time=0.0, amount=240.0, compartment=1)],
                {"hess_rtol": 1e-2, "hess_atol": 2e-3},
            ),
            (
                "slow_tail",
                "CL = THETA(1)*EXP(ETA(1))\nV1 = THETA(2)*EXP(ETA(2))\nQ = THETA(3)\nV2 = THETA(4)",
                "Y = F*(1 + EPS(1))",
                np.array([0.45, 14.0, 0.18, 25.0]),
                np.array([[0.015]]),
                1,
                np.array([0.05, -0.02]),
                np.diag([0.04, 0.03]),
                np.array([0.5, 2.0, 6.0, 12.0, 24.0, 48.0]),
                np.array([5.4, 5.9, 4.8, 3.5, 1.9, 0.9]),
                [DoseEvent(time=0.0, amount=300.0, compartment=1)],
                {"hess_rtol": 1.2e-2, "hess_atol": 2e-3},
            ),
        ],
        ids=[
            "exact_dose_rows",
            "multiple_bolus_superposition",
            "negative_covariance_combined_eps",
            "slow_tail",
        ],
    )
    def test_symbolic_advan3_eta_derivatives_cover_edge_case_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case_name: str,
        pk_code: str,
        error_code: str,
        theta: np.ndarray,
        sigma: np.ndarray,
        n_eps: int,
        eta: np.ndarray,
        omega: np.ndarray,
        obs_times: np.ndarray,
        obs_dv: np.ndarray,
        dose_events: list[DoseEvent],
        kwargs: dict[str, float],
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan3_model(
            pk_code,
            error_code,
            n_eps=n_eps,
            obs_times=obs_times,
            obs_dv=obs_dv,
            dose_events=dose_events,
        )
        symbolic_grad, symbolic_jacobian, _symbolic_hessian = _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name=f"advan3_{case_name}",
            trans=4,
            **kwargs,
        )

        if case_name == "exact_dose_rows":
            np.testing.assert_allclose(symbolic_jacobian[[0], :], 0.0, atol=1e-10)
        assert np.all(np.isfinite(symbolic_grad)), case_name
        assert np.all(np.isfinite(symbolic_jacobian)), case_name

    @pytest.mark.parametrize(
        ("case_name", "error_code", "theta", "sigma", "n_eps", "eta"),
        [
            (
                "proportional",
                "Y = F*(1 + EPS(1))",
                np.array([1.3, 1.8, 12.0, 0.7, 18.0]),
                np.array([[0.02]]),
                1,
                np.array([0.05, -0.08, 0.04]),
            ),
            (
                "additive",
                "Y = F + EPS(1)",
                np.array([1.1, 1.7, 11.0, 0.6, 16.0]),
                np.array([[0.05]]),
                1,
                np.array([-0.03, 0.07, -0.02]),
            ),
            (
                "combined_two_eps",
                "Y = F + EPS(1) + F*EPS(2)",
                np.array([1.2, 1.9, 13.0, 0.65, 17.0]),
                np.array([[0.06, -0.01], [-0.01, 0.025]]),
                2,
                np.array([0.04, -0.05, 0.06]),
            ),
            (
                "ipred_prop_theta",
                "IPRED = F\nW = IPRED * THETA(6)\nY = IPRED + W * EPS(1)",
                np.array([1.25, 1.85, 12.5, 0.7, 19.0, 0.14]),
                np.array([[1.0]]),
                1,
                np.array([0.02, -0.04, 0.03]),
            ),
            (
                "ipred_combined_theta",
                "IPRED = F\nW = SQRT(THETA(6)**2 + (IPRED*THETA(7))**2)\nY = IPRED + W * EPS(1)",
                np.array([1.25, 1.85, 12.5, 0.7, 19.0, 0.3, 0.12]),
                np.array([[0.8]]),
                1,
                np.array([0.02, -0.04, 0.03]),
            ),
        ],
        ids=[
            "proportional",
            "additive",
            "combined_two_eps",
            "ipred_prop_theta",
            "ipred_combined_theta",
        ],
    )
    def test_symbolic_advan4_eta_derivatives_match_independent_finite_differences(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case_name: str,
        error_code: str,
        theta: np.ndarray,
        sigma: np.ndarray,
        n_eps: int,
        eta: np.ndarray,
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan4_model(
            error_code,
            n_eps=n_eps,
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0, 14.0], dtype=float),
            obs_dv=np.array([5.8, 7.1, 7.6, 6.0, 3.2, 1.5], dtype=float),
            dose_events=[
                DoseEvent(time=0.0, amount=220.0, compartment=1),
                DoseEvent(time=9.0, amount=110.0, compartment=1),
            ],
        )
        omega = np.diag([0.04, 0.03, 0.02])
        _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name=f"advan4_{case_name}",
            trans=1,
            hess_rtol=8e-3,
            hess_atol=1e-3,
        )

    def test_symbolic_advan4_trans4_eta_derivatives_match_independent_finite_differences(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan4_model(
            "Y = F*(1 + EPS(1))",
            n_eps=1,
            obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0, 14.0], dtype=float),
            obs_dv=np.array([5.8, 7.1, 7.6, 6.0, 3.2, 1.5], dtype=float),
            dose_events=[
                DoseEvent(time=0.0, amount=220.0, compartment=1),
                DoseEvent(time=9.0, amount=110.0, compartment=1),
            ],
            pk_code=(
                "KA = THETA(1)*EXP(ETA(1))\n"
                "CL = THETA(2)*EXP(ETA(2))\n"
                "V2 = THETA(3)*EXP(ETA(3))\n"
                "Q = THETA(4)\n"
                "V3 = THETA(5)"
            ),
        )
        theta = np.array([1.2, 1.9, 13.0, 0.65, 17.0])
        eta = np.array([0.04, -0.05, 0.06])
        omega = np.diag([0.04, 0.03, 0.02])
        sigma = np.array([[0.02]])
        _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name="advan4_trans4_macro",
            trans=4,
            hess_rtol=8e-3,
            hess_atol=1e-3,
        )

    @pytest.mark.parametrize(
        (
            "case_name",
            "error_code",
            "theta",
            "sigma",
            "n_eps",
            "eta",
            "obs_times",
            "obs_dv",
            "dose_events",
            "kwargs",
        ),
        [
            (
                "exact_dose_rows",
                "Y = F + EPS(1)",
                np.array([1.2, 1.7, 11.0, 0.6, 16.0]),
                np.array([[0.05]]),
                1,
                np.array([0.03, -0.02, 0.04]),
                np.array([0.0, 0.5, 6.0, 6.0, 10.0, 18.0]),
                np.array([0.1, 5.6, 3.4, 3.4, 4.6, 1.3]),
                [
                    DoseEvent(time=0.0, amount=240.0, compartment=1),
                    DoseEvent(time=6.0, amount=120.0, compartment=1),
                ],
                {"hess_rtol": 8e-3, "hess_atol": 1e-3},
            ),
            (
                "multiple_bolus_superposition",
                "IPRED = F\nW = IPRED * THETA(6)\nY = IPRED + W * EPS(1)",
                np.array([1.15, 1.9, 12.0, 0.7, 18.0, 0.15]),
                np.array([[1.0]]),
                1,
                np.array([0.02, 0.06, -0.05]),
                np.array([0.5, 1.5, 3.0, 5.0, 8.0, 12.0, 20.0]),
                np.array([5.5, 7.0, 7.4, 6.5, 4.4, 2.8, 1.2]),
                [
                    DoseEvent(time=0.0, amount=180.0, compartment=1),
                    DoseEvent(time=3.5, amount=90.0, compartment=1),
                    DoseEvent(time=11.0, amount=60.0, compartment=1),
                ],
                {"hess_rtol": 1e-2, "hess_atol": 2e-3},
            ),
            (
                "negative_covariance_combined_eps",
                "Y = F + EPS(1) + F*EPS(2)",
                np.array([1.3, 1.8, 12.5, 0.75, 19.0]),
                np.array([[0.08, -0.02], [-0.02, 0.04]]),
                2,
                np.array([-0.04, 0.05, 0.03]),
                np.array([0.4, 1.0, 2.5, 5.0, 9.0, 16.0]),
                np.array([4.8, 6.6, 7.3, 5.9, 3.4, 1.4]),
                [DoseEvent(time=0.0, amount=230.0, compartment=1)],
                {"hess_rtol": 1e-2, "hess_atol": 2e-3},
            ),
            (
                "slow_tail",
                "Y = F*(1 + EPS(1))",
                np.array([0.55, 1.0, 20.0, 0.25, 25.0]),
                np.array([[0.015]]),
                1,
                np.array([0.04, -0.01, 0.02]),
                np.array([0.5, 3.0, 8.0, 16.0, 32.0, 48.0]),
                np.array([3.5, 5.2, 4.8, 3.5, 1.8, 0.9]),
                [DoseEvent(time=0.0, amount=300.0, compartment=1)],
                {"hess_rtol": 1.5e-2, "hess_atol": 3e-3},
            ),
        ],
        ids=[
            "exact_dose_rows",
            "multiple_bolus_superposition",
            "negative_covariance_combined_eps",
            "slow_tail",
        ],
    )
    def test_symbolic_advan4_eta_derivatives_cover_edge_case_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case_name: str,
        error_code: str,
        theta: np.ndarray,
        sigma: np.ndarray,
        n_eps: int,
        eta: np.ndarray,
        obs_times: np.ndarray,
        obs_dv: np.ndarray,
        dose_events: list[DoseEvent],
        kwargs: dict[str, float],
    ) -> None:
        pytest.importorskip("sympy")
        model = _make_symbolic_advan4_model(
            error_code,
            n_eps=n_eps,
            obs_times=obs_times,
            obs_dv=obs_dv,
            dose_events=dose_events,
        )
        omega = np.diag([0.04, 0.03, 0.02])
        symbolic_grad, symbolic_jacobian, _symbolic_hessian = _assert_symbolic_derivatives_match_fd(
            monkeypatch,
            model=model,
            theta=theta,
            eta=eta,
            omega=omega,
            sigma=sigma,
            case_name=f"advan4_{case_name}",
            trans=1,
            **kwargs,
        )

        if case_name == "exact_dose_rows":
            np.testing.assert_allclose(symbolic_jacobian[[0], :], 0.0, atol=1e-10)
        assert np.all(np.isfinite(symbolic_grad)), case_name
        assert np.all(np.isfinite(symbolic_jacobian)), case_name


class TestPopulationModel:
    def test_evaluate_individual_uses_subject_specific_observation_grid(self) -> None:
        df = pd.DataFrame(
            [
                {"ID": 1, "TIME": 1.0, "AMT": 0.0, "DV": 0.0, "EVID": 0, "MDV": 0},
                {"ID": 1, "TIME": 2.0, "AMT": 0.0, "DV": 0.0, "EVID": 0, "MDV": 0},
                {"ID": 2, "TIME": 1.0, "AMT": 0.0, "DV": 0.0, "EVID": 0, "MDV": 0},
                {"ID": 2, "TIME": 3.0, "AMT": 0.0, "DV": 0.0, "EVID": 0, "MDV": 0},
            ]
        )
        dataset = NONMEMDataset.from_dataframe(df)
        params = _make_params(theta=[10.0, 2.0], omega_diag=[1.0], sigma_diag=[1.0])
        pk = _DummyPK()

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"BASE": float(theta[0]) + float(eta[0]), "SLOPE": float(theta[1])}

        pop = PopulationModel(
            dataset=dataset,
            pk_subroutine=pk,
            params=params,
            pk_callable=pk_callable,
            trans=1,
        )

        ipred, obs_mask, f = pop.evaluate_individual(2, np.array([3.0]))

        np.testing.assert_allclose(ipred, [15.0, 19.0])
        np.testing.assert_array_equal(obs_mask, np.array([True, True]))
        np.testing.assert_allclose(f, ipred)

    def test_ofv_foce_aggregates_subject_objectives_and_defaults_missing_eta_to_zero(self) -> None:
        df = pd.DataFrame([{"ID": 1, "TIME": 1.0, "AMT": 0.0, "DV": 0.0, "EVID": 0, "MDV": 0}])
        pop = PopulationModel(
            dataset=NONMEMDataset.from_dataframe(df),
            pk_subroutine=_DummyPK(),
            params=_make_params(theta=[1.0], omega_diag=[1.0, 4.0], sigma_diag=[1.0]),
            trans=1,
        )

        class _StubIndiv:
            def __init__(self) -> None:
                self.calls: list[np.ndarray] = []

            def obj_eta(self, eta, theta, omega, sigma, trans=2):
                eta_arr = np.asarray(eta, dtype=float)
                self.calls.append(eta_arr.copy())
                return 10.0 + float(np.sum(eta_arr))

        subj1 = _StubIndiv()
        subj2 = _StubIndiv()
        pop._subject_events = {1: None, 2: None}  # type: ignore[assignment]
        pop._individual_models = {1: subj1, 2: subj2}  # type: ignore[assignment]

        result = pop.ofv_foce(pop.params, eta_hat={1: np.array([1.0, 2.0])})

        assert result == pytest.approx(23.0)
        np.testing.assert_allclose(subj1.calls[0], [1.0, 2.0])
        np.testing.assert_allclose(subj2.calls[0], [0.0, 0.0])



# ===========================================================================
# Detection-logic gate tests for _build_native_advan6_mixed_pkpd_contract
# ===========================================================================
# These tests verify that every blocking condition in the native-path
# contract builder returns None correctly, and that the "happy path"
# yields a non-None contract.  Each test exercises exactly one gate.
#
# The "valid" mixed-PK/PD error code below matches the
# mixed_pkpd_dvid_theta pattern required by the contract builder.
# ===========================================================================

_MIXED_PKPD_ERROR_CODE = (
    "PKPROP = THETA(9)\n"
    "PKADD = THETA(10)\n"
    "PDADD = THETA(11)\n"
    "IPRED = THETA(8) + A(4)\n"
    "W = PDADD\n"
    "Y = IPRED + W*EPS(2)\n"
    "IF (DVID == 1) W = SQRT((PKPROP*F)**2 + PKADD**2)\n"
    "IF (DVID == 1) Y = F + W*EPS(1)"
)
_SIMPLE_PROP_ERROR_CODE = "Y = F*(1+EPS(1))"
_SINGLE_DOSE = [DoseEvent(time=0.0, amount=100.0, compartment=1)]
_OBS_TIMES = np.array([0.5, 1.0, 4.0, 24.0], dtype=float)
_OBS_DV = np.array([10.0, 9.0, 7.0, 3.0], dtype=float)
_DVID_COVARIATES = [{"DVID": 1.0}, {"DVID": 1.0}, {"DVID": 2.0}, {"DVID": 2.0}]


class _FakeADVAN6(PKSubroutine):
    """Minimal PKSubroutine that declares advan=6."""
    advan = 6
    n_compartments = 4

    def solve(self, *args, **kwargs):
        raise AssertionError("native probe should bypass Python solve()")

    def apply_trans(self, raw_params, trans):
        return dict(raw_params)


class _FakeADVAN2(PKSubroutine):
    """Minimal PKSubroutine that declares advan=2 (non-ODE)."""
    advan = 2
    n_compartments = 2

    def solve(self, *args, **kwargs):
        raise AssertionError("should not be called")

    def apply_trans(self, raw_params, trans):
        return dict(raw_params)


def _build_mixed_pkpd_model(
    *,
    dose_events=None,
    error_code=_MIXED_PKPD_ERROR_CODE,
    pk_subroutine=None,
    obs_times=None,
    obs_dv=None,
    obs_covariates=None,
    covariate_df=None,
    occasion_indices=None,
) -> IndividualModel:
    compiler = NMTRANCompiler()
    n_eps = 1 if error_code == _SIMPLE_PROP_ERROR_CODE else 2
    dose_events = dose_events if dose_events is not None else _SINGLE_DOSE
    obs_times = obs_times if obs_times is not None else _OBS_TIMES
    obs_dv = obs_dv if obs_dv is not None else _OBS_DV
    obs_covariates = obs_covariates if obs_covariates is not None else _DVID_COVARIATES
    pk_sub = pk_subroutine if pk_subroutine is not None else _FakeADVAN6()

    def _pk_callable(theta, eta, t=0.0, covariates=None):
        return {
            "KTR": 1.0, "KA": 0.5, "CL": 0.134, "V": 8.11,
            "EMAX": 0.8, "EC50": 1.0, "KOUT": 0.0174, "E0": 100.0, "PCMT": 3.0,
        }

    events = SubjectEvents(
        subject_id=1,
        dose_events=dose_events,
        obs_times=obs_times,
        obs_dv=obs_dv,
        obs_cmt=np.ones(len(obs_times), dtype=int),
        obs_mdv=np.zeros(len(obs_times), dtype=int),
        obs_covariates=obs_covariates,
        covariate_df=covariate_df,
    )
    model = IndividualModel(
        subject_events=events,
        pk_subroutine=pk_sub,
        pk_callable=_pk_callable,
        error_callable=compiler.compile_error(error_code),
        n_eps=n_eps,
        occasion_indices=occasion_indices,
    )
    return model


class TestNativeAdvan6DetectionGates:
    """
    One test per blocking condition in _build_native_advan6_mixed_pkpd_contract.

    Each test patches away the native module if needed and asserts that the
    contract is exactly None when a gate condition is violated.
    """

    def test_happy_path_builds_contract(self) -> None:
        model = _build_mixed_pkpd_model()
        contract = model._native_advan6_mixed_pkpd_contract
        if contract is None:
            pytest.skip("native-cvodes extension not compiled in")
        assert isinstance(contract, dict)
        assert "dose_amount" in contract
        assert contract["dose_amount"] == pytest.approx(100.0)
        assert "required_names" in contract
        assert "KTR" in contract["required_names"]

    def test_gate_no_native_module_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            None,
        )
        model = _build_mixed_pkpd_model()
        assert model._native_advan6_mixed_pkpd_contract is None

    def test_gate_wrong_advan_number_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        model = _build_mixed_pkpd_model(pk_subroutine=_FakeADVAN2())
        assert model._native_advan6_mixed_pkpd_contract is None

    def test_gate_wrong_error_model_returns_none(self, monkeypatch) -> None:
        """An error model that is not in _NATIVE_SUPPORTED_ERROR_MODELS returns None.

        proportional / additive / combined_* are now accepted; a custom
        multi-EPS pattern that doesn't match any known template is still rejected.
        """
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        compiler = NMTRANCompiler()
        # This 4-line error block doesn't match any recognized template.
        unknown_error = (
            "W1 = THETA(1)\n"
            "W2 = THETA(2)\n"
            "Y = F + W1*EPS(1) + W2*EPS(2)\n"
            "IPRED2 = F*THETA(3)"
        )
        events = SubjectEvents(
            subject_id=1,
            dose_events=_SINGLE_DOSE,
            obs_times=_OBS_TIMES,
            obs_dv=_OBS_DV,
            obs_cmt=np.ones(4, dtype=int),
            obs_mdv=np.zeros(4, dtype=int),
        )
        model = IndividualModel(
            subject_events=events,
            pk_subroutine=_FakeADVAN6(),
            pk_callable=lambda theta, eta, t=0.0, covariates=None: {},
            error_callable=compiler.compile_error(unknown_error),
            n_eps=2,
        )
        assert model._native_advan6_mixed_pkpd_contract is None

    def test_proportional_error_model_activates_native_path(self, monkeypatch) -> None:
        """After P1b: proportional error model now builds a non-None contract."""
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        model = _build_mixed_pkpd_model(
            error_code=_SIMPLE_PROP_ERROR_CODE,
            obs_covariates=None,
        )
        contract = model._native_advan6_mixed_pkpd_contract
        assert contract is not None
        assert contract["is_mixed_pkpd"] is False

    def test_gate_occasion_indices_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        model = _build_mixed_pkpd_model(
            occasion_indices=np.array([1, 1, 2, 2]),
        )
        assert model._native_advan6_mixed_pkpd_contract is None

    def test_gate_extra_covariate_column_returns_none(self, monkeypatch) -> None:
        import pandas as pd
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        cov_df = pd.DataFrame({"TIME": _OBS_TIMES, "DVID": [1, 1, 2, 2], "WT": [70, 70, 70, 70]})
        model = _build_mixed_pkpd_model(covariate_df=cov_df)
        assert model._native_advan6_mixed_pkpd_contract is None

    def test_gate_zero_doses_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        model = _build_mixed_pkpd_model(dose_events=[])
        assert model._native_advan6_mixed_pkpd_contract is None

    def test_multiple_doses_activate_native_path(self, monkeypatch) -> None:
        """After P1a: multiple IV bolus doses now build a non-None contract."""
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        two_doses = [
            DoseEvent(time=0.0, amount=100.0, compartment=1),
            DoseEvent(time=24.0, amount=100.0, compartment=1),
        ]
        model = _build_mixed_pkpd_model(dose_events=two_doses)
        contract = model._native_advan6_mixed_pkpd_contract
        assert contract is not None
        assert contract["dose_times"] == [0.0, 24.0]
        assert contract["dose_amts"] == [100.0, 100.0]

    def test_gate_infusion_dose_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        infusion = [DoseEvent(time=0.0, amount=100.0, compartment=1, rate=10.0)]
        model = _build_mixed_pkpd_model(dose_events=infusion)
        assert model._native_advan6_mixed_pkpd_contract is None

    def test_late_dose_activates_native_path(self, monkeypatch) -> None:
        """After P1a: a dose at any time (not just t=0) builds a contract."""
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        late = [DoseEvent(time=1.0, amount=100.0, compartment=1)]
        model = _build_mixed_pkpd_model(dose_events=late)
        contract = model._native_advan6_mixed_pkpd_contract
        assert contract is not None
        assert contract["dose_times"] == [1.0]

    def test_gate_wrong_dose_compartment_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        wrong_cmt = [DoseEvent(time=0.0, amount=100.0, compartment=2)]
        model = _build_mixed_pkpd_model(dose_events=wrong_cmt)
        assert model._native_advan6_mixed_pkpd_contract is None

    def test_gate_unsorted_obs_times_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "openpkpd.model.individual._native_cvodes_advan6_mixed_pkpd_probe_rust",
            lambda *a, **kw: [],
        )
        unsorted_times = np.array([4.0, 1.0, 0.5, 24.0], dtype=float)
        unsorted_dv = np.array([7.0, 9.0, 10.0, 3.0], dtype=float)
        model = _build_mixed_pkpd_model(
            obs_times=unsorted_times, obs_dv=unsorted_dv
        )
        assert model._native_advan6_mixed_pkpd_contract is None

    def test_probe_gate_missing_required_pk_param_returns_none(self, monkeypatch) -> None:
        """_try_native_advan6_mixed_pkpd_probe returns None if a required name is absent."""
        compiler = NMTRANCompiler()

        def _bad_pk(theta, eta, t=0.0, covariates=None):
            # Missing EMAX, EC50, KOUT, E0 — PCMT present
            return {"KTR": 1.0, "KA": 0.5, "CL": 0.134, "V": 8.11, "PCMT": 3.0}

        events = SubjectEvents(
            subject_id=1,
            dose_events=_SINGLE_DOSE,
            obs_times=_OBS_TIMES,
            obs_dv=_OBS_DV,
            obs_cmt=np.ones(4, dtype=int),
            obs_mdv=np.zeros(4, dtype=int),
            obs_covariates=_DVID_COVARIATES,
        )
        model = IndividualModel(
            subject_events=events,
            pk_subroutine=_FakeADVAN6(),
            pk_callable=_bad_pk,
            error_callable=compiler.compile_error(_MIXED_PKPD_ERROR_CODE),
            n_eps=2,
        )
        if model._native_advan6_mixed_pkpd_contract is None:
            pytest.skip("native-cvodes extension not compiled in")
        pk_params = _bad_pk(np.array([]), np.array([]))
        result = model._try_native_advan6_mixed_pkpd_probe(pk_params, _OBS_TIMES)
        assert result is None

    def test_probe_gate_wrong_pcmt_returns_none(self, monkeypatch) -> None:
        """_try_native_advan6_mixed_pkpd_probe returns None when PCMT != 3."""
        compiler = NMTRANCompiler()

        def _pcmt2_pk(theta, eta, t=0.0, covariates=None):
            return {
                "KTR": 1.0, "KA": 0.5, "CL": 0.134, "V": 8.11,
                "EMAX": 0.8, "EC50": 1.0, "KOUT": 0.0174, "E0": 100.0,
                "PCMT": 2.0,  # wrong — must be 3
            }

        events = SubjectEvents(
            subject_id=1,
            dose_events=_SINGLE_DOSE,
            obs_times=_OBS_TIMES,
            obs_dv=_OBS_DV,
            obs_cmt=np.ones(4, dtype=int),
            obs_mdv=np.zeros(4, dtype=int),
            obs_covariates=_DVID_COVARIATES,
        )
        model = IndividualModel(
            subject_events=events,
            pk_subroutine=_FakeADVAN6(),
            pk_callable=_pcmt2_pk,
            error_callable=compiler.compile_error(_MIXED_PKPD_ERROR_CODE),
            n_eps=2,
        )
        if model._native_advan6_mixed_pkpd_contract is None:
            pytest.skip("native-cvodes extension not compiled in")
        pk_params = _pcmt2_pk(np.array([]), np.array([]))
        result = model._try_native_advan6_mixed_pkpd_probe(pk_params, _OBS_TIMES)
        assert result is None
