"""
Unit tests for Bayesian estimation (BAYESMethod, BayesianResult).

These tests verify:
  - Class construction and method_name attribute
  - Backend selection logic
  - BayesianResult dataclass construction and posterior_summary formatting
  - _compute_posterior_summary helper
  - Laplace fallback can be instantiated without MCMC backends
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.estimation.bayes import BayesianResult, BAYESMethod
from openpkpd.model.parameters import ParameterSet

# ---------------------------------------------------------------------------
# BAYESMethod construction
# ---------------------------------------------------------------------------


def test_bayes_method_name() -> None:
    """method_name class attribute must be 'BAYES'."""
    method = BAYESMethod()
    assert method.method_name == "BAYES"


def test_bayes_method_defaults() -> None:
    """Default constructor should set sensible hyperparameters."""
    method = BAYESMethod()
    assert method.n_samples == 1000
    assert method.n_chains == 2
    assert method.tune == 500
    assert method.target_accept == pytest.approx(0.85)
    assert method.seed == 42
    assert method.backend == "auto"
    assert method.prior_sd_theta == pytest.approx(2.0)


def test_bayes_method_custom_params() -> None:
    """Constructor should store custom arguments."""
    method = BAYESMethod(
        n_samples=500,
        n_chains=4,
        tune=200,
        target_accept=0.9,
        seed=99,
        backend="laplace",
        prior_sd_theta=1.5,
    )
    assert method.n_samples == 500
    assert method.n_chains == 4
    assert method.tune == 200
    assert method.target_accept == pytest.approx(0.9)
    assert method.seed == 99
    assert method.backend == "laplace"
    assert method.prior_sd_theta == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_backend_selection_explicit_laplace() -> None:
    """Explicit 'laplace' backend should return 'laplace'."""
    method = BAYESMethod(backend="laplace")
    assert method._select_backend() == "laplace"


def test_backend_selection_auto_falls_to_laplace(monkeypatch: pytest.MonkeyPatch) -> None:
    """When neither pymc nor numpyro is importable, fall back to laplace."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[override]
        if name in ("pymc", "numpyro"):
            raise ImportError(f"Fake ImportError for {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    method = BAYESMethod(backend="auto")
    assert method._select_backend() == "laplace"


def test_backend_selection_explicit_pymc_when_available() -> None:
    """When backend='pymc' is requested explicitly, return 'pymc'."""
    method = BAYESMethod(backend="pymc")
    assert method._select_backend() == "pymc"


def test_backend_selection_explicit_numpyro() -> None:
    """Explicit 'numpyro' backend should return 'numpyro'."""
    method = BAYESMethod(backend="numpyro")
    assert method._select_backend() == "numpyro"


# ---------------------------------------------------------------------------
# BayesianResult construction
# ---------------------------------------------------------------------------


def test_bayesian_result_minimal_construction() -> None:
    """BayesianResult should construct with just the required EstimationResult fields."""
    result = BayesianResult(
        theta_final=np.array([1.0, 2.0]),
        omega_final=np.eye(2),
        sigma_final=np.eye(1),
        ofv=100.0,
    )
    assert result.theta_final.shape == (2,)
    assert result.ofv == pytest.approx(100.0)
    assert isinstance(result.posterior_samples, dict)
    assert len(result.posterior_samples) == 0
    assert result.r_hat.shape == (0,)
    assert result.n_effective.shape == (0,)


def test_bayesian_result_with_posterior_samples() -> None:
    """BayesianResult should store and expose posterior samples correctly."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((200, 3))

    result = BayesianResult(
        theta_final=np.array([1.0, 2.0, 3.0]),
        omega_final=np.eye(2),
        sigma_final=np.eye(1),
        ofv=50.0,
        posterior_samples={"theta": samples},
        r_hat=np.array([1.01, 1.00, 1.02]),
        n_effective=np.array([180, 190, 175]),
    )
    assert "theta" in result.posterior_samples
    assert result.posterior_samples["theta"].shape == (200, 3)
    assert result.r_hat.shape == (3,)
    assert result.n_effective.shape == (3,)


def test_bayesian_result_inherits_estimation_result_properties() -> None:
    """BayesianResult inherits AIC, BIC, n_parameters from EstimationResult."""
    result = BayesianResult(
        theta_final=np.array([1.0, 2.0]),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=200.0,
        n_observations=50,
    )
    # n_parameters = 2 (theta) + 1 (omega 1x1 lower tri) + 1 (sigma 1x1) = 4
    assert result.n_parameters == 4
    assert result.aic == pytest.approx(200.0 + 2 * 4)
    assert result.bic == pytest.approx(200.0 + np.log(50) * 4)


# ---------------------------------------------------------------------------
# posterior_summary output
# ---------------------------------------------------------------------------


def test_posterior_summary_returns_string() -> None:
    """posterior_summary() should return a non-empty string."""
    rng = np.random.default_rng(1)
    samples = rng.standard_normal((100, 2)) + np.array([3.0, 10.0])

    result = BayesianResult(
        theta_final=np.array([3.0, 10.0]),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=80.0,
        method="BAYES(Laplace)",
        posterior_samples={"theta": samples},
        r_hat=np.array([1.0, 1.0]),
        n_effective=np.array([100, 100]),
        backend_used="laplace",
    )
    summary = result.posterior_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "THETA(1)" in summary
    assert "THETA(2)" in summary


def test_posterior_summary_no_samples() -> None:
    """posterior_summary() should not crash when there are no posterior samples."""
    result = BayesianResult(
        theta_final=np.array([1.0, 2.0]),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=120.0,
        posterior_ci_lo=np.array([0.8, 1.6]),
        posterior_ci_hi=np.array([1.2, 2.4]),
    )
    summary = result.posterior_summary()
    assert isinstance(summary, str)
    assert "THETA(1)" in summary


# ---------------------------------------------------------------------------
# _compute_posterior_summary helper
# ---------------------------------------------------------------------------


def test_compute_posterior_summary_bounds() -> None:
    """CI bounds should bracket the true mean for known samples."""
    method = BAYESMethod()
    rng = np.random.default_rng(42)
    # 1000 samples from N(5, 1)
    samples = rng.normal(5.0, 1.0, size=(1000, 1))
    lo, hi = method._compute_posterior_summary(samples, ci=0.95)

    # With 1000 N(5,1) samples, 95% CI should contain ~4 to ~6
    assert lo[0] < 5.0 < hi[0], "True mean should be within 95% CI"
    assert lo[0] > 3.0, "CI lower should be > 3"
    assert hi[0] < 7.0, "CI upper should be < 7"


def test_compute_posterior_summary_shape() -> None:
    """Output shape should match number of parameters."""
    method = BAYESMethod()
    samples = np.ones((500, 4))
    lo, hi = method._compute_posterior_summary(samples, ci=0.90)
    assert lo.shape == (4,)
    assert hi.shape == (4,)


def test_compute_posterior_summary_1d_input() -> None:
    """1-D input should be treated as single parameter."""
    method = BAYESMethod()
    samples = np.linspace(0, 1, 100)
    lo, hi = method._compute_posterior_summary(samples, ci=0.80)
    assert lo.shape == (1,)
    assert hi.shape == (1,)
    assert lo[0] < hi[0]


def test_approx_hessian_covariance_respects_ofv_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Laplace covariance should be 2 * H^-1 for OFV-like objectives."""
    import openpkpd.estimation.foce as foce_module

    theta_map = np.array([2.0])
    target_var = 9.0

    class _DummyFOCE:
        def __init__(self) -> None:
            self._current_eta_hat = {}

        def _inner_loop(self, population_model, params):
            return {}

        def _outer_ofv(self, population_model, params, eta_hat):
            diff = float(params.theta[0] - theta_map[0])
            return 7.0 + diff**2 / target_var

    monkeypatch.setattr(foce_module, "FOCEMethod", _DummyFOCE)

    method = BAYESMethod(backend="laplace", prior_sd_theta=1e8)
    init_params = ParameterSet(theta=theta_map.copy(), omega=np.eye(1), sigma=np.eye(1))
    foce_result = EstimationResult(
        theta_final=theta_map.copy(),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=7.0,
    )

    cov = method._approx_hessian_covariance(
        population_model=None,
        init_params=init_params,
        theta_map=theta_map,
        foce_result=foce_result,
    )

    assert cov.shape == (1, 1)
    assert cov[0, 0] == pytest.approx(target_var, rel=1e-2, abs=1e-2)


# ---------------------------------------------------------------------------
# get_estimation_method routing
# ---------------------------------------------------------------------------


def test_get_estimation_method_bayes() -> None:
    """get_estimation_method('BAYES') must return a BAYESMethod instance."""
    from openpkpd.estimation import get_estimation_method

    m = get_estimation_method("BAYES")
    assert isinstance(m, BAYESMethod)
    assert m.method_name == "BAYES"


def test_get_estimation_method_bayes_kwargs() -> None:
    """get_estimation_method should pass kwargs to BAYESMethod."""
    from openpkpd.estimation import get_estimation_method

    m = get_estimation_method("BAYES", n_samples=200, backend="laplace")
    assert isinstance(m, BAYESMethod)
    assert m.n_samples == 200
    assert m.backend == "laplace"
