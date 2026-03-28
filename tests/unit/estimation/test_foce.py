from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openpkpd.estimation.foce import FOCEMethod
from openpkpd.model.derivative_kernels import DerivativeKernelCapabilities
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.utils.constants import LOG2PI


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

        def submit(self, fn, sid, indiv, eta0, theta, omega, sigma, trans, maxiter):
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

    # 3 starts: outer minimize was called 3 times (inner loop also calls minimize,
    # so total calls = 3 starts × (1 outer + 2 inner) = 9)
    assert call_count[0] == 9
    assert result.converged
    assert np.isfinite(result.ofv)


def test_multistart_gtol_is_passed_to_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gtol parameter is forwarded to the scipy minimize call."""
    seen_gtol: list[float] = []

    def fake_minimize(fun, x0, *args, **kwargs):
        opts = kwargs.get("options", {})
        seen_gtol.append(opts.get("gtol", -1.0))
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
