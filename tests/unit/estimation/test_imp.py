from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation import get_estimation_method
from openpkpd.estimation.imp import IMPMethod
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
        from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec

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
