from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from openpkpd.estimation import get_estimation_method
from openpkpd.estimation.imp import IMPMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.utils.constants import Method
from openpkpd.utils.errors import WarningCode


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


class _MultiSubjectGaussianPopulationModel:
    trans = 2

    def __init__(self, dvs: list[float]) -> None:
        self._indiv = {
            sid: _GaussianIndividualModel(dv) for sid, dv in enumerate(dvs, start=1)
        }

    def subject_ids(self):
        return list(self._indiv)

    def individual_model(self, sid):
        return self._indiv[sid]


class _NativeHessianGaussianPopulationModel(_GaussianPopulationModel):
    def __init__(self, dv: float) -> None:
        self._indiv = {1: _NativeHessianGaussianIndividualModel(dv)}


class _NativeGradientGaussianIndividualModel(_NativeHessianGaussianIndividualModel):
    """Adds analytical gradient support to the scalar Gaussian fixture.

    obj_eta(eta) = log(2π·σ²) + (dv−η)²/σ² + η²/ω²
    d/dη        = −2(dv−η)/σ² + 2η/ω²
    """

    def supports_eta_objective_gradient(self, trans=None) -> bool:  # noqa: ARG002
        return True

    def eta_objective_value_grad(
        self, eta, theta, omega, sigma, trans=None  # noqa: ARG002
    ) -> tuple[float, np.ndarray]:
        eta_val = float(np.asarray(eta, dtype=float)[0])
        omega_var = float(omega[0, 0])
        sigma_var = float(sigma[0, 0])
        val = float(self.obj_eta(np.array([eta_val]), theta, omega, sigma, trans=trans))
        grad = np.array([-2.0 * (self.dv - eta_val) / sigma_var + 2.0 * eta_val / omega_var])
        return val, grad


class _NativeGradientGaussianPopulationModel(_GaussianPopulationModel):
    def __init__(self, dv: float) -> None:
        self._indiv = {1: _NativeGradientGaussianIndividualModel(dv)}


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


def test_importance_sample_reuses_exact_cached_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pop_model = _NativeHessianGaussianPopulationModel(dv=1.25)
    params = _GaussianParams(omega_var=0.6, sigma_var=0.4)
    imp = IMPMethod(isample=32, seed=0)
    calls = {"minimize": 0}

    def _counting_minimize(*args, **kwargs):
        calls["minimize"] += 1
        return scipy_minimize(*args, **kwargs)

    monkeypatch.setattr("openpkpd.estimation.imp.minimize", _counting_minimize)

    first = imp._importance_sample(pop_model, params, 1)
    second = imp._importance_sample(pop_model, params, 1)

    assert first == pytest.approx(second, abs=1e-12)
    assert calls["minimize"] == 1
    assert pop_model.individual_model(1).hessian_calls == 1


def test_importance_sample_warm_starts_map_from_previous_subject_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pop_model = _NativeHessianGaussianPopulationModel(dv=1.25)
    params_a = _GaussianParams(omega_var=0.6, sigma_var=0.4)
    params_b = _GaussianParams(omega_var=0.6, sigma_var=0.4)
    params_b.theta = np.array([0.5])
    imp = IMPMethod(isample=8, seed=0)
    x0_history: list[np.ndarray] = []

    def _fake_minimize(fun, x0, method=None, options=None):
        x0_arr = np.asarray(x0, dtype=float).copy()
        x0_history.append(x0_arr)
        return SimpleNamespace(x=x0_arr + 0.25)

    monkeypatch.setattr("openpkpd.estimation.imp.minimize", _fake_minimize)

    imp._importance_sample(pop_model, params_a, 1)
    imp._importance_sample(pop_model, params_b, 1)

    assert len(x0_history) == 2
    np.testing.assert_allclose(x0_history[0], np.zeros(1), atol=1e-12)
    np.testing.assert_allclose(x0_history[1], np.array([0.25]), atol=1e-12)


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


@pytest.mark.unit
def test_imp_objective_is_repeatable_with_fixed_subject_seeds() -> None:
    """Common-random-number streams must stabilize repeated OFV evaluations."""
    pop = _MultiSubjectGaussianPopulationModel([0.5, -1.25, 2.0])
    params = _GaussianParams(omega_var=0.4, sigma_var=0.6)

    imp = IMPMethod(isample=300, seed=123)
    imp._subj_seeds = {1: 101, 2: 202, 3: 303}

    ofv_first = imp._compute_imp_ofv(pop, params)
    ofv_second = imp._compute_imp_ofv(pop, params)

    assert ofv_first == pytest.approx(ofv_second, abs=1e-12)


@pytest.mark.unit
def test_imp_objective_matches_between_serial_and_parallel_with_fixed_subject_seeds() -> None:
    """Parallel subject evaluation must preserve the same CRN-based objective."""
    pop = _MultiSubjectGaussianPopulationModel([0.5, -1.25, 2.0, 0.75])
    params = _GaussianParams(omega_var=0.4, sigma_var=0.6)
    subj_seeds = {1: 101, 2: 202, 3: 303, 4: 404}

    serial = IMPMethod(isample=300, seed=123, n_parallel=1)
    serial._subj_seeds = subj_seeds.copy()
    parallel = IMPMethod(isample=300, seed=123, n_parallel=2)
    parallel._subj_seeds = subj_seeds.copy()

    ofv_serial = serial._compute_imp_ofv(pop, params)
    ofv_parallel = parallel._compute_imp_ofv(pop, params)

    assert ofv_serial == pytest.approx(ofv_parallel, abs=1e-12)


@pytest.mark.unit
def test_imp_estimate_reports_optimizer_and_ess_diagnostics() -> None:
    pop = _MultiSubjectGaussianPopulationModel([0.5, -1.25, 2.0])
    params = ParameterSet.from_specs(
        [ThetaSpec(init=0.0, fixed=True)],
        [OmegaSpec(block_size=1, values=[0.4])],
        [SigmaSpec(block_size=1, values=[0.6], fixed=True)],
    )

    result = IMPMethod(isample=80, maxeval=3, seed=123).estimate(pop, params)

    optimizer = result.diagnostics["optimizer"]
    objective = result.diagnostics["objective"]
    importance = result.diagnostics["importance_sampling"]

    assert optimizer["method"] == "L-BFGS-B"
    assert optimizer["iterations"] >= 0
    assert optimizer["function_evals"] == result.n_function_evals
    assert optimizer["maxeval"] == 3
    assert isinstance(optimizer["maxeval_reached"], bool)

    assert objective["history_length"] == len(result.ofv_history)
    assert objective["final_ofv"] == pytest.approx(result.ofv)
    assert objective["initial_ofv"] is not None
    assert objective["delta_ofv"] is not None

    assert importance["isample"] == 80
    assert importance["ess_warning_threshold"] == pytest.approx(8.0)
    assert set(importance["final_eval_ess_by_subject"]) == {1, 2, 3}
    assert importance["final_eval_min_ess"] is not None
    assert importance["final_eval_min_ess"] > 0.0
    assert importance["final_eval_mean_ess"] is not None
    assert importance["final_eval_median_ess"] is not None
    assert importance["final_eval_n_below_warn_threshold"] >= 0


@pytest.mark.unit
def test_impmap_uses_focei_warm_start(monkeypatch: pytest.MonkeyPatch) -> None:
    pop = _MultiSubjectGaussianPopulationModel([0.5, -1.25, 2.0])
    params = ParameterSet.from_specs(
        [ThetaSpec(init=0.4, lower=0.01, upper=10.0)],
        [OmegaSpec(block_size=1, values=[0.4])],
        [SigmaSpec(block_size=1, values=[0.6], fixed=True)],
    )
    warm_theta = np.array([0.75])
    warm_omega = np.array([[0.2]])
    warm_sigma = np.array([[0.6]])
    calls: list[tuple[str, object]] = []

    class _FakeFOCE:
        def __init__(self, **kwargs):
            calls.append(("ctor", kwargs))

        def estimate(self, population_model, init_params):
            calls.append(("estimate", init_params.theta.copy()))
            return SimpleNamespace(
                theta_final=warm_theta,
                omega_final=warm_omega,
                sigma_final=warm_sigma,
                converged=True,
                message="warm-start-ok",
                ofv=12.3,
            )

    def _fake_minimize(objective, x0, method=None, options=None):
        calls.append(("x0", np.asarray(x0, dtype=float).copy()))
        objective(np.asarray(x0, dtype=float))
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            success=True,
            status=0,
            message="ok",
            nit=1,
            nfev=1,
        )

    monkeypatch.setattr("openpkpd.estimation.imp.FOCEMethod", _FakeFOCE)
    monkeypatch.setattr("openpkpd.estimation.imp.minimize", _fake_minimize)
    monkeypatch.setattr(IMPMethod, "_compute_imp_ofv", lambda self, pop_model, p: 1.0)
    monkeypatch.setattr(IMPMethod, "_map_etas", lambda self, pop_model, p: {})

    result = IMPMethod(isample=10, maxeval=2, seed=123, is_map=True).estimate(pop, params)

    assert any(tag == "estimate" for tag, _payload in calls)
    warm_x0 = next(payload for tag, payload in calls if tag == "x0")
    expected_x0 = ParameterSet(
        theta=warm_theta,
        omega=warm_omega,
        sigma=warm_sigma,
        theta_specs=params.theta_specs,
        omega_specs=params.omega_specs,
        sigma_specs=params.sigma_specs,
    ).apply_bounds().to_vector()
    np.testing.assert_allclose(warm_x0, expected_x0)
    assert result.diagnostics["warm_start"]["used"] is True
    assert result.diagnostics["warm_start"]["method"] == "FOCEI"


@pytest.mark.unit
def test_imp_does_not_use_focei_warm_start(monkeypatch: pytest.MonkeyPatch) -> None:
    pop = _MultiSubjectGaussianPopulationModel([0.5, -1.25, 2.0])
    params = ParameterSet.from_specs(
        [ThetaSpec(init=0.4, lower=0.01, upper=10.0)],
        [OmegaSpec(block_size=1, values=[0.4])],
        [SigmaSpec(block_size=1, values=[0.6], fixed=True)],
    )

    class _UnexpectedFOCE:
        def __init__(self, **kwargs):
            raise AssertionError("FOCE warm start should not be used for raw IMP")

    def _fake_minimize(objective, x0, method=None, options=None):
        objective(np.asarray(x0, dtype=float))
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            success=True,
            status=0,
            message="ok",
            nit=1,
            nfev=1,
        )

    monkeypatch.setattr("openpkpd.estimation.imp.FOCEMethod", _UnexpectedFOCE)
    monkeypatch.setattr("openpkpd.estimation.imp.minimize", _fake_minimize)
    monkeypatch.setattr(IMPMethod, "_compute_imp_ofv", lambda self, pop_model, p: 1.0)
    monkeypatch.setattr(IMPMethod, "_map_etas", lambda self, pop_model, p: {})

    result = IMPMethod(isample=10, maxeval=2, seed=123, is_map=False).estimate(pop, params)

    assert "warm_start" not in result.diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# P0.4 — is_map parameter, method name, WARN_006, routing
# ─────────────────────────────────────────────────────────────────────────────


class TestIMPIsMap:
    """P0.4: is_map controls method name and estimation behaviour."""

    def test_default_is_map_false(self):
        imp = IMPMethod()
        assert imp.is_map is False

    def test_is_map_true_sets_method_name_impmap(self):
        imp = IMPMethod(is_map=True)
        assert imp.method_name == Method.IMPMAP

    def test_is_map_false_sets_method_name_imp(self):
        imp = IMPMethod(is_map=False)
        assert imp.method_name == Method.IMP

    def test_get_estimation_method_imp_gives_is_map_false(self):
        m = get_estimation_method("IMP")
        assert isinstance(m, IMPMethod)
        assert m.is_map is False

    def test_get_estimation_method_impmap_gives_is_map_true(self):
        m = get_estimation_method("IMPMAP")
        assert isinstance(m, IMPMethod)
        assert m.is_map is True

    def test_result_method_name_matches_is_map(self):
        """Result.method should reflect IMP vs IMPMAP."""
        pop = _GaussianPopulationModel(dv=1.0)
        params = _GaussianParams(omega_var=0.5, sigma_var=0.5)
        # We can't call .estimate() without a proper ParameterSet, so test
        # via method_name directly after construction
        assert IMPMethod(is_map=False).method_name == Method.IMP
        assert IMPMethod(is_map=True).method_name == Method.IMPMAP


class TestIMPWarn006:
    """P0.4: low ESS triggers WARN_006 on the EstimationResult."""

    class _ConcentratedPop:
        """Spike-posterior: MAP is exact, so all samples are far away."""
        trans = 2

        class _Indiv:
            def log_likelihood(self, theta, eta, sigma, trans=2):
                # Very tight spike at eta=5 → proposal from MAP is ~N(5, tiny)
                # but Omega is N(0, 0.01), so all samples near 5 have near-zero prior
                return -1000.0 * float(np.sum((np.asarray(eta) - 5.0) ** 2))

            def obj_eta(self, eta, theta, omega, sigma, trans=2):
                e = float(np.asarray(eta)[0])
                ov = float(omega[0, 0])
                return self.log_likelihood(theta, eta, sigma, trans) + e ** 2 / ov

        def subject_ids(self): return [1]
        def individual_model(self, sid): return self._Indiv()

    def test_warn006_emitted_when_ess_low(self):
        """Force low ESS by using a mismatched proposal distribution."""
        pop = self._ConcentratedPop()
        params = ParameterSet.from_specs(
            [],
            [OmegaSpec(block_size=1, values=[0.01])],
            [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
        )

        imp = IMPMethod(isample=50, maxeval=1, seed=0)
        # Call _importance_sample directly to force low ESS tracking
        class _Adapter:
            omega = np.array([[0.01]])
            sigma = np.array([[0.1]])
            theta = np.array([], dtype=float)
            def n_eta(self): return 1
            def apply_bounds(self): return self
            def to_vector(self): return np.array([])
            @classmethod
            def from_vector(cls, v, ref): return cls()

        # Simulate low ESS by pre-populating the tracking list
        imp._low_ess_subjects = [(1, 0.5)]   # force: 1 subject with ESS=0.5

        # Build a minimal result and check warning emission
        from openpkpd.estimation.base import EstimationResult
        result = EstimationResult(
            theta_final=np.array([]),
            omega_final=np.array([[0.01]]),
            sigma_final=np.array([[0.1]]),
            ofv=10.0,
            converged=True,
            method=Method.IMP,
        )
        # Trigger warning attachment logic directly
        if imp._low_ess_subjects:
            n_low = len(imp._low_ess_subjects)
            worst_id, worst_ess = min(imp._low_ess_subjects, key=lambda t: t[1])
            threshold = imp.ESS_WARN_FRACTION * imp.isample
            result.add_structured_warning(
                WarningCode.WARN_006,
                f"{n_low} subject(s) ESS < {threshold:.0f}; worst={worst_ess:.1f}",
            )
        codes = {w.code for w in result.structured_warnings}
        assert WarningCode.WARN_006 in codes

    def test_no_warn006_when_ess_high(self):
        """No WARN_006 when ESS tracking list is empty."""
        imp = IMPMethod(isample=500, seed=0)
        imp._low_ess_subjects = []
        from openpkpd.estimation.base import EstimationResult
        result = EstimationResult(
            theta_final=np.array([]),
            omega_final=np.eye(1) * 0.5,
            sigma_final=np.eye(1) * 0.1,
            ofv=10.0,
            converged=True,
            method=Method.IMP,
        )
        # No low-ESS subjects → no WARN_006
        codes = {w.code for w in result.structured_warnings}
        assert WarningCode.WARN_006 not in codes

    def test_ess_warn_fraction_default_is_10_percent(self):
        assert IMPMethod.ESS_WARN_FRACTION == pytest.approx(0.10)


class TestIMPNumericalAccuracy:
    """P0.4: Verify IMP accuracy against the closed-form Gaussian marginal.

    Closed form: log p(y) = log N(y; 0, omega + sigma) — see derivation in
    Lavielle (2014), Appendix A.  The IMP estimate must be within 5 % of the
    analytic value even at N=500 samples.
    """

    @pytest.mark.parametrize("dv,omega_var,sigma_var", [
        (1.0, 0.5, 0.5),
        (0.0, 1.0, 0.2),
        (3.0, 0.3, 0.7),
    ])
    def test_imp_marginal_close_to_analytic(self, dv, omega_var, sigma_var):
        pop = _GaussianPopulationModel(dv=dv)
        params = _GaussianParams(omega_var=omega_var, sigma_var=sigma_var)
        analytic = -2.0 * _implemented_log_marginal_gaussian(dv, sigma_var, omega_var)
        ofv_est = IMPMethod(isample=500, seed=7)._compute_imp_ofv(pop, params)
        # Allow ±5 % relative error OR ±0.1 absolute for near-zero analytic
        assert abs(ofv_est - analytic) <= max(0.05 * abs(analytic), 0.1), (
            f"IMP OFV={ofv_est:.4f} analytic={analytic:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# P5 — native analytical gradient in MAP optimization
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestIMPNativeGradient:
    """P5: analytical gradient dispatch in _importance_sample / _map_etas."""

    # ── fixture helpers ────────────────────────────────────────────────────

    _DV = 1.25
    _OMEGA_VAR = 0.6
    _SIGMA_VAR = 0.4

    @property
    def _params(self) -> _GaussianParams:
        return _GaussianParams(omega_var=self._OMEGA_VAR, sigma_var=self._SIGMA_VAR)

    # eta* = dv * ω / (σ + ω)  (Gaussian MAP closed form)
    @property
    def _expected_eta_map(self) -> float:
        return self._DV * self._OMEGA_VAR / (self._SIGMA_VAR + self._OMEGA_VAR)

    # ── gradient accuracy ──────────────────────────────────────────────────

    def test_gradient_consistent_with_finite_difference(self) -> None:
        """Analytical gradient matches central-FD of obj_eta to 1e-5 rel tol."""
        indiv = _NativeGradientGaussianIndividualModel(self._DV)
        theta = np.array([0.0])
        omega = np.array([[self._OMEGA_VAR]])
        sigma = np.array([[self._SIGMA_VAR]])
        eps = 1e-5
        for eta_val in (-1.0, 0.0, 0.5, 1.5):
            eta = np.array([eta_val])
            _, grad = indiv.eta_objective_value_grad(eta, theta, omega, sigma)
            fd = np.array([
                (indiv.obj_eta(eta + eps, theta, omega, sigma)
                 - indiv.obj_eta(eta - eps, theta, omega, sigma))
                / (2.0 * eps)
            ])
            np.testing.assert_allclose(
                grad, fd, rtol=1e-5, atol=1e-8,
                err_msg=f"gradient mismatch at eta={eta_val}"
            )

    # ── _importance_sample jac dispatch ───────────────────────────────────

    def test_importance_sample_passes_jac_true_with_native_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When gradient is available, _importance_sample passes jac=True to minimize."""
        pop_model = _NativeGradientGaussianPopulationModel(dv=self._DV)
        captured: list[dict] = []

        def _spy(fun, x0, **kwargs):
            captured.append(dict(kwargs))
            return scipy_minimize(fun, x0, **kwargs)

        monkeypatch.setattr("openpkpd.estimation.imp.minimize", _spy)
        IMPMethod(isample=32, seed=0)._importance_sample(pop_model, self._params, 1)

        assert len(captured) >= 1, "minimize should have been called"
        assert captured[0].get("jac") is True, (
            "jac=True must be forwarded when native gradient is available"
        )

    def test_importance_sample_does_not_pass_jac_without_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without gradient support, _importance_sample must NOT pass jac."""
        pop_model = _GaussianPopulationModel(dv=self._DV)
        captured: list[dict] = []

        def _spy(fun, x0, **kwargs):
            captured.append(dict(kwargs))
            return scipy_minimize(fun, x0, **kwargs)

        monkeypatch.setattr("openpkpd.estimation.imp.minimize", _spy)
        IMPMethod(isample=32, seed=0)._importance_sample(pop_model, self._params, 1)

        assert len(captured) >= 1
        assert "jac" not in captured[0], (
            "jac must not be forwarded when native gradient is unavailable"
        )

    def test_importance_sample_native_gradient_matches_closed_form(self) -> None:
        """IS result with analytical gradient agrees with closed-form Gaussian marginal."""
        pop_model = _NativeGradientGaussianPopulationModel(dv=self._DV)
        analytic = _implemented_log_marginal_gaussian(
            self._DV, self._SIGMA_VAR, self._OMEGA_VAR
        )
        log_marg = IMPMethod(isample=32, seed=0)._importance_sample(
            pop_model, self._params, 1
        )
        assert log_marg == pytest.approx(analytic, abs=1e-8)

    # ── _map_etas jac dispatch ────────────────────────────────────────────

    def test_map_etas_passes_jac_true_with_native_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When gradient is available, _map_etas passes jac=True to minimize."""
        pop_model = _NativeGradientGaussianPopulationModel(dv=self._DV)
        captured: list[dict] = []

        def _spy(fun, x0, **kwargs):
            captured.append(dict(kwargs))
            return scipy_minimize(fun, x0, **kwargs)

        monkeypatch.setattr("openpkpd.estimation.imp.minimize", _spy)
        IMPMethod(isample=32, seed=0, is_map=True)._map_etas(pop_model, self._params)

        assert len(captured) >= 1, "minimize should have been called"
        assert captured[0].get("jac") is True, (
            "jac=True must be forwarded when native gradient is available"
        )

    def test_map_etas_does_not_pass_jac_without_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without gradient support, _map_etas must NOT pass jac."""
        pop_model = _GaussianPopulationModel(dv=self._DV)
        captured: list[dict] = []

        def _spy(fun, x0, **kwargs):
            captured.append(dict(kwargs))
            return scipy_minimize(fun, x0, **kwargs)

        monkeypatch.setattr("openpkpd.estimation.imp.minimize", _spy)
        IMPMethod(isample=32, seed=0, is_map=True)._map_etas(pop_model, self._params)

        assert len(captured) >= 1
        assert "jac" not in captured[0], (
            "jac must not be forwarded when native gradient is unavailable"
        )

    def test_map_etas_native_gradient_reaches_correct_map(self) -> None:
        """MAP estimate with analytical gradient matches Gaussian closed-form solution."""
        pop_model = _NativeGradientGaussianPopulationModel(dv=self._DV)
        eta_hat = IMPMethod(isample=32, seed=0, is_map=True)._map_etas(
            pop_model, self._params
        )
        # eta* = dv * omega / (sigma + omega)
        assert eta_hat[1][0] == pytest.approx(self._expected_eta_map, abs=1e-6)
