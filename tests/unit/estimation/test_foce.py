from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openpkpd.estimation.foce import (
    FOCEMethod,
    _compute_G_i,
    _compute_G_i_via_sensitivity,
    _can_skip_eta_optimization,
    _eta_optimizer_bounds,
    _estimate_gradient_norm,
)
from openpkpd.model.derivative_kernels import DerivativeKernelCapabilities
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.utils.constants import LOG2PI
from openpkpd.utils.errors import WarningCode


class _DummySubjectEvents:
    obs_dv = np.array([1.0])

    def observation_mask(self) -> np.ndarray:
        return np.array([True])


class _DummyIndividualModel:
    def __init__(self) -> None:
        self.subject_events = _DummySubjectEvents()
        self.batch_calls = 0

    def obj_eta(
        self,
        eta: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> float:
        target = np.array([0.25])
        return float(np.sum((eta - target) ** 2))

    def obj_eta_many(
        self,
        eta_batch: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        self.batch_calls += 1
        eta_arr = np.asarray(eta_batch, dtype=float)
        if eta_arr.ndim == 1:
            eta_arr = eta_arr[None, :]
        target = np.array([0.25])
        return np.sum((eta_arr - target) ** 2, axis=1)

    def evaluate_observation_model(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pred = np.array([1.2])
        var = np.array([0.1])
        return pred, pred, pred, pred, var


class _DummyPopulationModel:
    trans = 2

    def __init__(self) -> None:
        self._indiv = _DummyIndividualModel()

    def subject_ids(self) -> list[int]:
        return [1]

    def n_subjects(self) -> int:
        return len(self.subject_ids())

    def individual_model(self, subject_id: int) -> _DummyIndividualModel:
        assert subject_id == 1
        return self._indiv


class _TwoSubjectPopulationModel(_DummyPopulationModel):
    def __init__(self) -> None:
        self._indiv_by_sid = {1: _DummyIndividualModel(), 2: _DummyIndividualModel()}

    def subject_ids(self) -> list[int]:
        return [1, 2]

    def individual_model(self, subject_id: int) -> _DummyIndividualModel:
        assert subject_id in {1, 2}
        return self._indiv_by_sid[subject_id]


class _AlwaysFailingIndividualModel(_DummyIndividualModel):
    def obj_eta(
        self,
        eta: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> float:
        raise OverflowError("simulated numerical overflow")


class _FailingPopulationModel(_DummyPopulationModel):
    def individual_model(self, subject_id: int) -> _AlwaysFailingIndividualModel:
        assert subject_id == 1
        return _AlwaysFailingIndividualModel()


class _DummyParams:
    theta = np.array([1.0])
    omega = np.array([[0.0]])
    sigma = np.array([[0.1]])

    def n_eta(self) -> int:
        return 1


class _PositiveOmegaParams(_DummyParams):
    omega = np.array([[0.2]])


class _SymbolicIndividualModel(_DummyIndividualModel):
    def __init__(self) -> None:
        super().__init__()
        self.value_grad_calls = 0
        self._kernel = _QuadraticEtaKernel(self)

    def supports_eta_objective_gradient(self, trans: int = 2) -> bool:
        return True

    def eta_objective_value_grad(
        self,
        eta: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> tuple[float, np.ndarray]:
        return self._kernel.eta_data_objective_value_grad(theta, eta, sigma)

    def get_subject_derivative_kernel(self, trans: int = 2) -> object:
        return self._kernel


class _QuadraticEtaKernel:
    capabilities = DerivativeKernelCapabilities(eta_objective_gradient=True)

    def __init__(self, owner: _SymbolicIndividualModel) -> None:
        self.owner = owner

    def eta_data_objective_value_grad(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        self.owner.value_grad_calls += 1
        target = np.array([0.25])
        eta_arr = np.asarray(eta, dtype=float)
        diff = eta_arr - target
        return float(np.sum(diff**2)), 2.0 * diff

    def eta_data_objective_values(
        self,
        theta: np.ndarray,
        eta_batch: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta_batch, dtype=float)
        if eta_arr.ndim == 1:
            eta_arr = eta_arr[None, :]
        target = np.array([0.25])
        return np.sum((eta_arr - target) ** 2, axis=1)


class _SymbolicPopulationModel(_DummyPopulationModel):
    def __init__(self) -> None:
        self._indiv = _SymbolicIndividualModel()


class _InteractionIndividualModel(_DummyIndividualModel):
    def evaluate_observation_model(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pred = np.array([2.0])
        sigma_val = float(sigma[0, 0]) if sigma.size > 0 else 0.25
        # Proportional error: var = F^2 * sigma (FOCEI interaction)
        var = pred**2 * sigma_val
        return pred, pred, pred, pred, var


class _InteractionPopulationModel(_DummyPopulationModel):
    def individual_model(self, subject_id: int) -> _InteractionIndividualModel:
        assert subject_id == 1
        return _InteractionIndividualModel()


class _InteractionParams(_DummyParams):
    omega = np.array([[1.0]])
    sigma = np.array([[0.25]])


def test_inner_loop_tolerates_singular_omega_when_individual_objective_handles_it() -> None:
    method = FOCEMethod(maxeval=1)

    eta_hat = method._inner_loop(_DummyPopulationModel(), _DummyParams())

    assert 1 in eta_hat
    assert np.all(np.isfinite(eta_hat[1]))
    assert eta_hat[1][0] == pytest.approx(0.25, abs=1e-3)


def test_outer_ofv_tolerates_singular_omega_when_penalty_fallback_is_needed() -> None:
    method = FOCEMethod(maxeval=1)

    ofv = method._outer_ofv(
        _DummyPopulationModel(),
        _DummyParams(),
        {1: np.array([0.25])},
    )

    assert np.isfinite(ofv)


def test_inner_loop_tolerates_individual_objective_exceptions() -> None:
    method = FOCEMethod(maxeval=1)

    eta_hat = method._inner_loop(_FailingPopulationModel(), _DummyParams())

    assert 1 in eta_hat
    assert np.all(np.isfinite(eta_hat[1]))


def test_inner_loop_skips_eta_optimization_for_fixed_near_zero_omega(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = ParameterSet.from_specs(
        [ThetaSpec(init=1.0, lower=0.0, upper=5.0)],
        [OmegaSpec(block_size=1, values=[1e-8], fixed=True)],
        [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
    )
    method = FOCEMethod(maxeval=1)

    def fail_minimize(*args, **kwargs):
        raise AssertionError("inner-loop minimize should be skipped")

    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fail_minimize)

    eta_hat = method._inner_loop(_DummyPopulationModel(), params)

    assert _can_skip_eta_optimization(params)
    assert eta_hat.keys() == {1}
    assert np.allclose(eta_hat[1], 0.0)


def test_outer_ofv_matches_closed_form_for_diagonal_variance_case() -> None:
    method = FOCEMethod(maxeval=1)

    eta = np.array([0.25])
    ofv = method._outer_ofv(_DummyPopulationModel(), _PositiveOmegaParams(), {1: eta})

    # Formula: n_obs*LOG2PI + log|R_i| + quad + eta_penalty + log|Omega|
    expected = (
        LOG2PI  # 1 observation
        + np.log(0.1)  # log|C_i|
        + (1.0 - 1.2) ** 2 / 0.1  # quad
        + eta[0] ** 2 / 0.2  # eta_penalty
        + np.log(0.2)  # log|Omega|
    )
    assert ofv == pytest.approx(expected, abs=1e-10)


def test_outer_ofv_parallel_matches_serial_for_multi_subject_population() -> None:
    params = _PositiveOmegaParams()
    eta_hat = {
        1: np.array([0.25]),
        2: np.array([0.25]),
    }

    serial = FOCEMethod(maxeval=1, n_parallel=1)._outer_ofv(
        _TwoSubjectPopulationModel(),
        params,
        eta_hat,
    )
    parallel = FOCEMethod(maxeval=1, n_parallel=2)._outer_ofv(
        _TwoSubjectPopulationModel(),
        params,
        eta_hat,
    )

    assert parallel == pytest.approx(serial, abs=1e-12)


def test_outer_ofv_focei_interaction_uses_prediction_scaled_variance() -> None:
    method = FOCEMethod(interaction=True, maxeval=1)

    eta = np.array([0.5])
    ofv = method._outer_ofv(
        _InteractionPopulationModel(),
        _InteractionParams(),
        {1: eta},
    )

    # Formula: (n_obs-n_eta)*LOG2PI + log|C_i| + quad_R + eta_penalty
    # G_i=0 (pred constant=2.0 regardless of eta), so C_i = R_i = diag(expected_var)
    # Woodbury: log|C_i| = log|R_i| + log|Omega| + log|M| where M = Omega^{-1} (G=0)
    expected_var = 2.0**2 * 0.25
    # log|M| = log|Omega^{-1}| = -log|Omega| = 0 (Omega=1.0)
    expected = (
        + np.log(expected_var)  # log|R_i|
        + np.log(1.0)  # log|Omega|
        + np.log(1.0)  # log|M| = -log|Omega| = 0
        + (1.0 - 2.0) ** 2 / expected_var  # data quadratic under R^{-1}
        + eta[0] ** 2  # eta_penalty (omega=1.0)
    )
    assert ofv == pytest.approx(expected, abs=1e-10)


class _InteractionParamsHalfOmega(_DummyParams):
    omega = np.array([[0.5]])
    sigma = np.array([[0.25]])


def test_outer_ofv_focei_interaction_does_not_double_count_log_omega() -> None:
    method = FOCEMethod(interaction=True, maxeval=1)

    eta = np.array([0.5])
    ofv = method._outer_ofv(
        _InteractionPopulationModel(),
        _InteractionParamsHalfOmega(),
        {1: eta},
    )

    expected_var = 2.0**2 * 0.25
    expected = (
        + np.log(expected_var)
        + np.log(0.5)
        + np.log(2.0)
        + (1.0 - 2.0) ** 2 / expected_var
        + eta[0] ** 2 / 0.5
    )
    assert ofv == pytest.approx(expected, abs=1e-10)


class _NoSubjectPopulation:
    trans = 2

    def subject_ids(self) -> list[int]:
        return []

    def n_subjects(self) -> int:
        return 0


def test_focei_enables_powell_fallback_by_default() -> None:
    assert FOCEMethod(interaction=True).outer_fallback_optimizer == "Powell"
    assert FOCEMethod(interaction=False).outer_fallback_optimizer is None


def test_run_outer_optimizer_accepts_control_stream_style_uppercase_names() -> None:
    method = FOCEMethod(interaction=True, maxeval=1)
    objective = lambda x: float(np.sum(np.square(x)))
    x0 = np.array([1.0], dtype=float)
    bounds = [(0.0, 2.0)]

    result = method._run_outer_optimizer(
        objective,
        x0,
        bounds,
        optimizer="POWELL",
        maxeval=1,
    )

    assert np.isfinite(float(result.fun))


def test_run_single_uses_fallback_outer_optimizer_when_it_improves_ofv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = ParameterSet.from_specs(
        [ThetaSpec(init=2.0, lower=0.0, upper=5.0)],
        [OmegaSpec(block_size=1, values=[0.2], fixed=True)],
        [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
    )
    method = FOCEMethod(
        interaction=True,
        maxeval=3,
        outer_fallback_maxeval=4,
        print_interval=999,
    )
    primary_x = params.to_vector()
    fallback_x = ParameterSet(
        theta=np.array([1.0]),
        omega=params.omega,
        sigma=params.sigma,
        theta_specs=params.theta_specs,
        omega_specs=params.omega_specs,
        sigma_specs=params.sigma_specs,
    ).to_vector()

    def fake_outer_ofv(_population_model, cur_params, _eta_hat):
        return float(cur_params.theta[0])

    monkeypatch.setattr(method, "_outer_ofv", fake_outer_ofv)
    monkeypatch.setattr(method, "_inner_loop", lambda population_model, cur_params: {})

    def fake_minimize(fun, x0, *args, **kwargs):
        if kwargs["method"] == "L-BFGS-B":
            trial = np.asarray(primary_x, dtype=float)
            fun(trial)
            return SimpleNamespace(x=trial, success=True, message="lbfgsb")
        if kwargs["method"] == "Powell":
            trial = np.asarray(fallback_x, dtype=float)
            fun(trial)
            return SimpleNamespace(x=trial, success=True, message="powell")
        raise AssertionError(f"unexpected optimizer {kwargs['method']}")

    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fake_minimize)

    result, final_params, _eta_hat, final_ofv = method._run_single(
        params.to_vector(),
        params,
        _NoSubjectPopulation(),
    )

    assert result.message == "powell"
    assert final_ofv == pytest.approx(1.0)
    assert float(final_params.theta[0]) == pytest.approx(1.0)


def test_run_single_retains_best_iterate_when_terminal_point_is_worse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = ParameterSet.from_specs(
        [ThetaSpec(init=2.0, lower=0.0, upper=5.0)],
        [OmegaSpec(block_size=1, values=[0.2], fixed=True)],
        [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
    )
    method = FOCEMethod(
        interaction=True,
        maxeval=3,
        outer_fallback_optimizer=None,
        print_interval=999,
    )
    better_x = ParameterSet(
        theta=np.array([1.0]),
        omega=params.omega,
        sigma=params.sigma,
        theta_specs=params.theta_specs,
        omega_specs=params.omega_specs,
        sigma_specs=params.sigma_specs,
    ).to_vector()
    worse_x = params.to_vector()

    def fake_outer_ofv(_population_model, cur_params, _eta_hat):
        return float(cur_params.theta[0])

    monkeypatch.setattr(method, "_outer_ofv", fake_outer_ofv)
    monkeypatch.setattr(method, "_inner_loop", lambda population_model, cur_params: {})

    def fake_minimize(fun, x0, *args, **kwargs):
        fun(np.asarray(better_x, dtype=float))
        return SimpleNamespace(x=np.asarray(worse_x, dtype=float), success=True, message="lbfgsb")

    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fake_minimize)

    result, final_params, _eta_hat, final_ofv = method._run_single(
        params.to_vector(),
        params,
        _NoSubjectPopulation(),
    )

    assert "[best-iterate]" in result.message
    assert final_ofv == pytest.approx(1.0)
    assert float(final_params.theta[0]) == pytest.approx(1.0)


def test_best_iterate_promotion_preserves_hessian_for_near_identical_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = ParameterSet.from_specs(
        [ThetaSpec(init=2.0, lower=0.0, upper=5.0)],
        [OmegaSpec(block_size=1, values=[0.2], fixed=True)],
        [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
    )
    method = FOCEMethod(interaction=True, maxeval=3, outer_fallback_optimizer=None)
    best_x = params.to_vector()
    method._best_outer_x = best_x.copy()
    method._best_outer_ofv = 1.0
    hess_inv = np.array([[3.0]])
    near_x = best_x + np.array([5e-5])
    monkeypatch.setattr(method, "_inner_loop", lambda population_model, cur_params: {})
    monkeypatch.setattr(
        method,
        "_outer_ofv",
        lambda population_model, cur_params, eta_hat: 1.0,
    )

    result, final_params, _eta_hat, final_ofv = method._maybe_promote_best_iterate(
        SimpleNamespace(x=near_x, success=False, message="lbfgsb", hess_inv=hess_inv),
        params,
        _NoSubjectPopulation(),
        final_params=params,
        final_eta_hat={},
        final_ofv=1.0 + 5e-6,
    )

    assert "[best-iterate]" in result.message
    np.testing.assert_allclose(result.hess_inv, hess_inv)
    assert final_ofv == pytest.approx(1.0)
    assert float(final_params.theta[0]) == pytest.approx(2.0)


def test_run_single_reuses_cached_outer_evaluation_for_terminal_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = ParameterSet.from_specs(
        [ThetaSpec(init=2.0, lower=0.0, upper=5.0)],
        [OmegaSpec(block_size=1, values=[0.2], fixed=True)],
        [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
    )
    method = FOCEMethod(
        interaction=False,
        maxeval=3,
        outer_fallback_optimizer=None,
        print_interval=999,
    )
    terminal_x = params.to_vector()
    calls = {"inner": 0, "outer": 0}

    def fake_inner_loop(_population_model, _params):
        calls["inner"] += 1
        return {1: np.array([0.25])}

    def fake_outer_ofv(_population_model, cur_params, _eta_hat):
        calls["outer"] += 1
        return float(cur_params.theta[0])

    def fake_minimize(fun, x0, *args, **kwargs):
        trial = np.asarray(terminal_x, dtype=float)
        assert float(fun(trial)) == pytest.approx(2.0)
        return SimpleNamespace(x=trial, success=True, message="lbfgsb")

    monkeypatch.setattr(method, "_inner_loop", fake_inner_loop)
    monkeypatch.setattr(method, "_outer_ofv", fake_outer_ofv)
    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fake_minimize)

    result, final_params, _eta_hat, final_ofv = method._run_single(
        terminal_x,
        params,
        _NoSubjectPopulation(),
    )

    assert result.message == "lbfgsb"
    assert final_ofv == pytest.approx(2.0)
    assert float(final_params.theta[0]) == pytest.approx(2.0)
    assert calls == {"inner": 1, "outer": 1}


def test_run_single_structured_retry_improves_abnormal_focei_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = ParameterSet.from_specs(
        [ThetaSpec(init=2.0, lower=0.0, upper=5.0)],
        [OmegaSpec(block_size=1, values=[0.2], fixed=True)],
        [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
    )
    method = FOCEMethod(
        interaction=True,
        maxeval=3,
        outer_fallback_optimizer=None,
        retry_omega_scales=(0.5,),
        print_interval=999,
    )
    base_x = params.to_vector()
    retry_x = ParameterSet(
        theta=np.array([1.0]),
        omega=params.omega,
        sigma=params.sigma,
        theta_specs=params.theta_specs,
        omega_specs=params.omega_specs,
        sigma_specs=params.sigma_specs,
    ).to_vector()

    def fake_outer_ofv(_population_model, cur_params, _eta_hat):
        return float(cur_params.theta[0])

    monkeypatch.setattr(method, "_outer_ofv", fake_outer_ofv)
    monkeypatch.setattr(method, "_inner_loop", lambda population_model, cur_params: {})
    monkeypatch.setattr(
        method,
        "_structured_retry_vectors",
        lambda init_params, final_params: [np.asarray(retry_x, dtype=float)],
    )

    call_count = {"n": 0}

    def fake_minimize(fun, x0, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            fun(np.asarray(base_x, dtype=float))
            return SimpleNamespace(x=np.asarray(base_x, dtype=float), success=False, message="ABNORMAL")
        fun(np.asarray(retry_x, dtype=float))
        return SimpleNamespace(x=np.asarray(retry_x, dtype=float), success=True, message="retry-ok")

    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fake_minimize)

    result, final_params, _eta_hat, final_ofv = method._run_single(
        params.to_vector(),
        params,
        _NoSubjectPopulation(),
    )

    assert call_count["n"] == 2
    assert result.message == "retry-ok"
    assert final_ofv == pytest.approx(1.0)
    assert float(final_params.theta[0]) == pytest.approx(1.0)


def test_estimate_reuses_parallel_worker_pool_across_outer_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executors: list[SimpleNamespace] = []

    class _FakeFuture:
        def __init__(self, value: tuple[int, np.ndarray]) -> None:
            self._value = value

        def result(self) -> tuple[int, np.ndarray]:
            return self._value

    class _FakeExecutor:
        def __init__(self, max_workers=None) -> None:
            self.max_workers = max_workers
            self.shutdown_calls = 0
            executors.append(self)

        def submit(self, fn, sid, indiv, eta0, eta_bounds, theta, omega, sigma, trans, maxiter):
            return _FakeFuture((sid, np.asarray(eta0, dtype=float)))

        def shutdown(self, wait: bool = True) -> None:
            self.shutdown_calls += 1

    def fake_as_completed(futures):
        return list(futures)

    def fake_minimize(fun, x0, *args, **kwargs):
        assert kwargs.get("bounds") is not None
        x = np.asarray(x0, dtype=float)
        fun(x)
        fun(x)
        return SimpleNamespace(x=x, success=True, message="ok")

    monkeypatch.setattr("openpkpd.estimation.foce.ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr("openpkpd.estimation.foce.as_completed", fake_as_completed)
    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fake_minimize)

    params = ParameterSet.from_specs(
        [ThetaSpec(init=1.0, lower=0.1, upper=5.0)],
        [OmegaSpec(block_size=1, values=[0.2], fixed=True)],
        [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
    )

    result = FOCEMethod(maxeval=2, n_parallel=2, print_interval=999).estimate(
        _TwoSubjectPopulationModel(),
        params,
    )

    assert result.converged
    assert len(executors) == 1
    assert executors[0].max_workers == 2
    assert executors[0].shutdown_calls == 1


def test_inner_loop_passes_batched_workers_to_scipy_and_uses_obj_eta_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = _DummyPopulationModel()
    seen_workers: list[object] = []

    def fake_minimize(fun, x0, *args, **kwargs):
        workers = kwargs["options"].get("workers")
        seen_workers.append(workers)
        assert callable(workers)
        pts = [
            np.asarray(x0, dtype=float) + np.array([1e-5]),
            np.asarray(x0, dtype=float) + np.array([2e-5]),
        ]
        out = workers(fun, pts)
        assert len(out) == 2
        return SimpleNamespace(x=np.asarray(x0, dtype=float), success=True, message="ok")

    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fake_minimize)

    eta_hat = FOCEMethod(maxeval=1)._inner_loop(population, _DummyParams())

    assert 1 in eta_hat
    assert len(seen_workers) == 1
    assert population._indiv.batch_calls == 1


def test_inner_loop_uses_symbolic_jac_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = _SymbolicPopulationModel()

    def fake_minimize(fun, x0, *args, **kwargs):
        jac = kwargs.get("jac")
        options = kwargs.get("options") or {}
        assert callable(jac)
        assert "workers" not in options
        x = np.asarray(x0, dtype=float)
        fun(x)
        grad = jac(x)
        np.testing.assert_allclose(grad, 2.0 * (x - np.array([0.25])))
        return SimpleNamespace(x=np.array([0.25]), success=True, message="ok")

    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fake_minimize)

    eta_hat = FOCEMethod(maxeval=1)._inner_loop(population, _PositiveOmegaParams())

    assert 1 in eta_hat
    np.testing.assert_allclose(eta_hat[1], [0.25])
    assert population._indiv.value_grad_calls == 1


def test_inner_loop_passes_finite_eta_bounds_to_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = _DummyPopulationModel()
    params = _PositiveOmegaParams()
    expected_bounds = _eta_optimizer_bounds(params)

    def fake_minimize(fun, x0, *args, **kwargs):
        bounds = kwargs.get("bounds")
        assert bounds == expected_bounds
        return SimpleNamespace(x=np.asarray(x0, dtype=float), success=True, message="ok")

    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fake_minimize)

    eta_hat = FOCEMethod(maxeval=1)._inner_loop(population, params)

    assert 1 in eta_hat


def test_eta_optimizer_bounds_use_theta_support_when_available() -> None:
    params = ParameterSet.from_specs(
        [
            ThetaSpec(init=1.5, lower=0.3, upper=8.0),
            ThetaSpec(init=3.0, lower=0.5, upper=15.0),
            ThetaSpec(init=35.0, lower=10.0, upper=80.0),
        ],
        [OmegaSpec(block_size=1, values=[0.1])] * 3,
        [SigmaSpec(block_size=1, values=[0.1])],
    )

    bounds = _eta_optimizer_bounds(params)

    assert len(bounds) == 3
    assert bounds[0][0] == pytest.approx(np.log(0.3 / 1.5))
    assert bounds[0][1] == pytest.approx(np.log(8.0 / 1.5))
    assert bounds[1][0] == pytest.approx(np.log(0.5 / 3.0))
    assert bounds[1][1] == pytest.approx(np.log(15.0 / 3.0))
    assert bounds[2][0] == pytest.approx(np.log(10.0 / 35.0))
    assert bounds[2][1] == pytest.approx(np.log(80.0 / 35.0))


def test_multistart_returns_best_ofv(monkeypatch: pytest.MonkeyPatch) -> None:
    """n_starts>1 returns the run with the lowest OFV."""
    from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec

    call_count = [0]

    def fake_minimize(fun, x0, *args, **kwargs):
        call_count[0] += 1
        # Return a different x each call so OFV varies
        x = np.asarray(x0, dtype=float) + call_count[0] * 0.1
        fun(x)
        return SimpleNamespace(x=x, success=True, message="ok")

    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fake_minimize)

    params = ParameterSet.from_specs(
        [ThetaSpec(init=1.0, lower=0.1, upper=5.0)],
        [OmegaSpec(block_size=1, values=[0.2], fixed=True)],
        [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
    )

    result = FOCEMethod(
        maxeval=2,
        n_starts=3,
        seed=0,
        outer_fallback_optimizer=None,
        retain_best_iterate=False,
    ).estimate(_DummyPopulationModel(), params)

    # 3 starts: outer minimize was called 3 times and each objective evaluation
    # triggers one inner-loop minimize. The terminal-point reevaluation is now
    # served from the exact outer-evaluation cache, so total calls = 3 starts ×
    # (1 outer + 1 inner) = 6.
    assert call_count[0] == 6
    assert result.converged
    assert np.isfinite(result.ofv)


def test_multistart_gtol_is_passed_to_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gtol parameter is forwarded to the scipy minimize call."""
    seen_gtol: list[float] = []
    seen_jac_callable: list[bool] = []

    def fake_minimize(fun, x0, *args, **kwargs):
        opts = kwargs.get("options", {})
        seen_gtol.append(opts.get("gtol", -1.0))
        seen_jac_callable.append(callable(kwargs.get("jac")))
        x = np.asarray(x0, dtype=float)
        fun(x)
        return SimpleNamespace(x=x, success=True, message="ok")

    monkeypatch.setattr("openpkpd.estimation.foce.minimize", fake_minimize)

    from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec

    params = ParameterSet.from_specs(
        [ThetaSpec(init=1.0, lower=0.1, upper=5.0)],
        [OmegaSpec(block_size=1, values=[0.2], fixed=True)],
        [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
    )

    FOCEMethod(maxeval=1, gtol=1e-7).estimate(_DummyPopulationModel(), params)

    assert seen_gtol, "minimize was never called"
    assert seen_gtol[0] == pytest.approx(1e-7)
    assert seen_jac_callable[0] is True


# ─────────────────────────────────────────────────────────────────────────────
# P0.2 — Trust-region outer optimizer
# ─────────────────────────────────────────────────────────────────────────────


class TestTrustConstrOptimizer:
    """trust-constr must be accepted and must find the minimum of a quadratic OFV."""

    def _make_quadratic_params(self) -> ParameterSet:
        return ParameterSet.from_specs(
            [ThetaSpec(init=2.0, lower=0.5, upper=5.0)],
            [OmegaSpec(block_size=1, values=[0.1], fixed=True)],
            [SigmaSpec(block_size=1, values=[0.05], fixed=True)],
        )

    def test_trust_constr_accepted_without_error(self):
        params = self._make_quadratic_params()
        foce = FOCEMethod(
            maxeval=30,
            outer_optimizer="trust-constr",
            outer_fallback_optimizer=None,
            retain_best_iterate=False,
        )
        # Should not raise ValueError
        result = foce.estimate(_DummyPopulationModel(), params)
        assert np.isfinite(result.ofv)

    def test_invalid_optimizer_raises(self):
        params = self._make_quadratic_params()
        foce = FOCEMethod(outer_optimizer="bogus", outer_fallback_optimizer=None)
        with pytest.raises(ValueError, match="Unsupported"):
            foce.estimate(_DummyPopulationModel(), params)

    def test_trust_constr_finds_optimum_of_quadratic(self):
        """
        OFV = (theta - 1.5)^2.  The trust-constr optimizer should converge to
        theta ≈ 1.5 given enough function evaluations.
        """
        class _QuadraticIndividual:
            subject_events = type(
                "E", (), {"obs_dv": np.array([1.5]), "observation_mask": lambda s: np.array([True])}
            )()

            def obj_eta(self, eta, theta, omega, sigma, trans=2):
                return float((float(theta[0]) - 1.5) ** 2)

            def log_likelihood(self, theta, eta, sigma, trans=2):
                return -(float(theta[0]) - 1.5) ** 2

            def evaluate_observation_model(self, theta, eta, sigma, trans=2):
                p = np.array([float(theta[0])])
                return p, np.array([True]), p, p, np.array([0.05])

        class _QuadraticPop:
            trans = 2
            _indiv = _QuadraticIndividual()

            def subject_ids(self): return [1]
            def n_subjects(self): return 1
            def individual_model(self, sid): return self._indiv

        params = self._make_quadratic_params()
        result = FOCEMethod(
            maxeval=200,
            outer_optimizer="trust-constr",
            outer_fallback_optimizer=None,
            retain_best_iterate=False,
        ).estimate(_QuadraticPop(), params)
        assert result.theta_final[0] == pytest.approx(1.5, abs=0.05)

    def test_lbfgsb_still_works_as_default(self):
        params = self._make_quadratic_params()
        result = FOCEMethod(
            maxeval=30,
            outer_optimizer="L-BFGS-B",
            outer_fallback_optimizer=None,
            retain_best_iterate=False,
        ).estimate(_DummyPopulationModel(), params)
        assert np.isfinite(result.ofv)

    def test_powell_accepted(self):
        params = self._make_quadratic_params()
        result = FOCEMethod(
            maxeval=10,
            outer_optimizer="Powell",
            outer_fallback_optimizer=None,
            retain_best_iterate=False,
        ).estimate(_DummyPopulationModel(), params)
        assert np.isfinite(result.ofv)


# ─────────────────────────────────────────────────────────────────────────────
# P0.2/P0.5 — _estimate_gradient_norm
# ─────────────────────────────────────────────────────────────────────────────


class TestEstimateGradientNorm:
    def test_returns_none_when_no_jac(self):
        result = SimpleNamespace(x=np.array([1.0]), success=True, message="ok")
        assert _estimate_gradient_norm(result) is None

    def test_reads_jac_from_lbfgsb_result(self):
        result = SimpleNamespace(
            x=np.array([1.0]),
            jac=np.array([0.002, -0.01, 0.005]),
            success=True,
            message="ok",
        )
        norm = _estimate_gradient_norm(result)
        assert norm == pytest.approx(0.01)

    def test_reads_grad_from_trust_constr_result(self):
        result = SimpleNamespace(
            x=np.array([1.0]),
            grad=np.array([0.5, -0.3]),
            success=True,
            message="ok",
        )
        norm = _estimate_gradient_norm(result)
        assert norm == pytest.approx(0.5)

    def test_grad_takes_precedence_over_jac(self):
        result = SimpleNamespace(
            x=np.array([1.0]),
            grad=np.array([0.9]),
            jac=np.array([0.1]),
            success=True,
            message="ok",
        )
        # grad is checked first
        assert _estimate_gradient_norm(result) == pytest.approx(0.9)


# ─────────────────────────────────────────────────────────────────────────────
# P0.5 — Structured warnings emitted by FOCE estimate()
# ─────────────────────────────────────────────────────────────────────────────


class TestFOCEStructuredWarnings:
    def _make_params(self, omega_diag=(0.1,)) -> ParameterSet:
        return ParameterSet.from_specs(
            [ThetaSpec(init=1.0, lower=0.1, upper=5.0)],
            [OmegaSpec(block_size=1, values=[v]) for v in omega_diag],
            [SigmaSpec(block_size=1, values=[0.05], fixed=True)],
        )

    def test_well_conditioned_produces_no_conditioning_warning(self):
        params = self._make_params(omega_diag=(0.2,))
        result = FOCEMethod(
            maxeval=5,
            outer_fallback_optimizer=None,
            retain_best_iterate=False,
        ).estimate(_DummyPopulationModel(), params)
        codes = {w.code for w in result.structured_warnings}
        assert WarningCode.WARN_001 not in codes
        assert WarningCode.WARN_002 not in codes


# ─────────────────────────────────────────────────────────────────────────────
# P0.3 — _compute_G_i and sensitivity path
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeGi:
    """Verify _compute_G_i produces the correct Jacobian."""

    def _make_linear_indiv(self, slope: float = 2.0):
        """IPRED(eta) = slope * eta[0].  G_i = [[slope]]."""
        class _LinearIndiv:
            def evaluate_observation_model(self, theta, eta, sigma, trans=2):
                pred = np.array([slope * float(eta[0])])
                var = np.array([0.1])
                return pred, np.array([True]), pred, pred, var
        return _LinearIndiv()

    def test_fd_gradient_correct_for_linear_model(self):
        indiv = self._make_linear_indiv(slope=3.0)
        eta = np.array([0.5])
        pred0 = np.array([3.0 * 0.5])   # 1.5
        obs_mask = np.array([True])
        G = _compute_G_i(indiv, np.array([]), eta, np.eye(1), 2, obs_mask, pred0)
        assert G.shape == (1, 1)
        np.testing.assert_allclose(G[0, 0], 3.0, rtol=1e-3)

    def test_fd_gradient_correct_for_two_eta_model(self):
        """IPRED(eta) = 2*eta[0] + 4*eta[1].  G = [[2, 4]]."""
        class _MultiEtaIndiv:
            def evaluate_observation_model(self, theta, eta, sigma, trans=2):
                pred = np.array([2.0 * float(eta[0]) + 4.0 * float(eta[1])])
                return pred, np.array([True]), pred, pred, np.array([0.1])

        eta = np.array([1.0, 1.0])
        pred0 = np.array([6.0])
        obs_mask = np.array([True])
        G = _compute_G_i(_MultiEtaIndiv(), np.array([]), eta, np.eye(1), 2, obs_mask, pred0)
        assert G.shape == (1, 2)
        np.testing.assert_allclose(G[0, 0], 2.0, rtol=1e-3)
        np.testing.assert_allclose(G[0, 1], 4.0, rtol=1e-3)

    def test_fd_gradient_falls_back_gracefully_when_evaluate_raises(self):
        """If evaluate_observation_model raises, the column should stay zero."""
        class _FlakyIndiv:
            def evaluate_observation_model(self, theta, eta, sigma, trans=2):
                if eta[0] != 0.5:
                    raise RuntimeError("boom")
                return np.array([1.0]), np.array([True]), np.array([1.0]), np.array([1.0]), np.array([0.1])

        eta = np.array([0.5])
        pred0 = np.array([1.0])
        G = _compute_G_i(_FlakyIndiv(), np.array([]), eta, np.eye(1), 2, np.array([True]), pred0)
        assert G.shape == (1, 1)
        assert float(G[0, 0]) == pytest.approx(0.0)

    def test_sensitivity_path_matches_fd_for_diagonal_chain_rule(self):
        """
        When pk_sub.solve_with_sensitivity is present and well-behaved,
        _compute_G_i should route through the sensitivity path and the result
        must agree with the FD result to within integration tolerance.

        Model: pk_param = exp(eta[0])  (one PK param, one ETA)
               amounts[t] = dose * exp(-pk_param * t)  (mono-exponential decay)
               IPRED = amounts / V   (observation = compartment 1 / V)
        At eta=0: pk_param=1, amounts[t=1] = dose*exp(-1), IPRED = dose*exp(-1)/V
        Sensitivity: ∂amounts/∂pk_param = -t*amounts
        ∂pk_param/∂eta = exp(eta) = 1 at eta=0
        ∂IPRED/∂eta = (1/V) * (-t) * amounts = -t * IPRED
        """
        dose, V, t_obs = 10.0, 2.0, 1.0
        # Nominal prediction
        pk0 = np.exp(0.0)          # = 1.0
        amt0 = dose * np.exp(-pk0 * t_obs)
        ipred0 = amt0 / V

        class _SensSubroutine:
            class _Sol:
                def __init__(self, y, sensitivity):
                    self.y = y                   # shape (1, 1)
                    self.sensitivity = sensitivity  # shape (1, 1, 1)

            def solve_with_sensitivity(self, theta, eta, trans=2):
                pk = np.exp(float(eta[0]))
                amt = dose * np.exp(-pk * t_obs)
                d_amt_dpk = -t_obs * amt        # ∂amounts/∂pk
                return self._Sol(
                    y=np.array([[amt]]),
                    sensitivity=np.array([[[d_amt_dpk]]]),
                )

        class _SensIndiv:
            pk_subroutine = _SensSubroutine()

            def pk_callable(self, theta, eta, trans=2):
                return np.array([np.exp(float(eta[0]))])

            def evaluate_observation_model(self, theta, eta, sigma, trans=2, _amounts=None):
                if _amounts is not None:
                    amt = float(np.asarray(_amounts).ravel()[0])
                else:
                    pk = np.exp(float(eta[0]))
                    amt = dose * np.exp(-pk * t_obs)
                pred = np.array([amt / V])
                return pred, np.array([True]), pred, pred, np.array([0.05])

        indiv = _SensIndiv()
        eta = np.array([0.0])
        obs_mask = np.array([True])
        pred0_obs = np.array([ipred0])

        G_fd = _compute_G_i(indiv, np.array([]), eta, np.eye(1), 2, obs_mask, pred0_obs, h=1e-5)
        G_sens = _compute_G_i_via_sensitivity(
            indiv, indiv.pk_subroutine, np.array([]), eta, np.eye(1), 2, obs_mask, pred0_obs, h=1e-5
        )
        # Both paths should agree to within ~1 % relative tolerance
        np.testing.assert_allclose(G_sens, G_fd, rtol=0.02,
            err_msg=f"Sensitivity path G={G_sens} vs FD path G={G_fd}")
