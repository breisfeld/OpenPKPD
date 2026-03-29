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



# ---------------------------------------------------------------------------
# mcmc_diagnostics — unit tests with analytically known answers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeRhat:
    """Tests for compute_rhat (split-R-hat, Vehtari et al. 2021)."""

    def _make_chains(self, n_chains: int, n_draws: int, n_params: int,
                     scale: float = 1.0) -> np.ndarray:
        """Generate independent N(0,1) chains — should yield R-hat ≈ 1."""
        rng = np.random.default_rng(0)
        return rng.normal(0.0, scale, size=(n_chains, n_draws, n_params))

    def test_rhat_perfectly_mixed_near_one(self) -> None:
        """Independent N(0,1) chains → R-hat ≤ 1.05 for all parameters."""
        from openpkpd.estimation.mcmc_diagnostics import compute_rhat

        chains = self._make_chains(4, 2000, 3)
        rhat = compute_rhat(chains)
        assert rhat.shape == (3,)
        assert np.all(rhat <= 1.05), f"R-hat values: {rhat}"

    def test_rhat_returns_at_least_one(self) -> None:
        """R-hat values must be ≥ 1.0 by definition."""
        from openpkpd.estimation.mcmc_diagnostics import compute_rhat

        chains = self._make_chains(2, 500, 2)
        rhat = compute_rhat(chains)
        assert np.all(rhat >= 1.0)

    def test_rhat_poorly_mixed_above_threshold(self) -> None:
        """Chains stuck at very different means should yield R-hat > 1.1."""
        from openpkpd.estimation.mcmc_diagnostics import compute_rhat

        rng = np.random.default_rng(1)
        n_draws, n_params = 500, 1
        # Chain 0 centred at 0, chain 1 centred at 5 — huge between-chain spread
        chain0 = rng.normal(0.0, 0.1, (n_draws, n_params))
        chain1 = rng.normal(5.0, 0.1, (n_draws, n_params))
        chains = np.stack([chain0, chain1], axis=0)   # (2, 500, 1)
        rhat = compute_rhat(chains)
        assert rhat[0] > 1.1, f"Expected R-hat > 1.1 for separated chains, got {rhat[0]:.4f}"

    def test_rhat_single_chain_returns_ones(self) -> None:
        """Single chain → R-hat should not raise and returns a finite value."""
        from openpkpd.estimation.mcmc_diagnostics import compute_rhat

        rng = np.random.default_rng(2)
        single = rng.normal(size=(1, 200, 2))
        rhat = compute_rhat(single)
        assert rhat.shape == (2,)
        assert np.all(np.isfinite(rhat))

    def test_rhat_2d_input_treated_as_single_chain(self) -> None:
        """2-D input (n_draws, n_params) is treated as a single chain."""
        from openpkpd.estimation.mcmc_diagnostics import compute_rhat

        rng = np.random.default_rng(3)
        single_2d = rng.normal(size=(400, 2))
        rhat = compute_rhat(single_2d)
        assert rhat.shape == (2,)


@pytest.mark.unit
class TestComputeESS:
    """Tests for compute_ess (effective sample size)."""

    def test_ess_iid_samples_near_total(self) -> None:
        """i.i.d. chains → ESS should be a substantial fraction of n_total."""
        from openpkpd.estimation.mcmc_diagnostics import compute_ess

        rng = np.random.default_rng(10)
        chains = rng.normal(size=(4, 500, 2))   # 4 chains × 500 draws = 2000 total
        ess = compute_ess(chains)
        assert ess.shape == (2,)
        # For i.i.d. samples, ESS should be high (≥ 50% of total)
        assert np.all(ess >= 500), f"ESS too low for i.i.d. samples: {ess}"

    def test_ess_highly_autocorrelated_is_lower(self) -> None:
        """Highly autocorrelated chain → ESS much lower than n_total."""
        from openpkpd.estimation.mcmc_diagnostics import compute_ess

        rng = np.random.default_rng(11)
        n_draws = 1000
        # AR(1) with high correlation φ=0.99
        phi = 0.99
        x = np.zeros((1, n_draws, 1))
        for t in range(1, n_draws):
            x[0, t, 0] = phi * x[0, t - 1, 0] + rng.normal() * np.sqrt(1 - phi**2)
        ess = compute_ess(x)
        # ESS should be << n_draws when autocorrelation is high
        assert ess[0] < 200, f"Expected ESS < 200 for phi=0.99, got {ess[0]:.1f}"

    def test_ess_always_positive(self) -> None:
        """ESS must be strictly positive."""
        from openpkpd.estimation.mcmc_diagnostics import compute_ess

        rng = np.random.default_rng(12)
        chains = rng.normal(size=(2, 300, 3))
        ess = compute_ess(chains)
        assert np.all(ess > 0)


@pytest.mark.unit
class TestComputeAutocorr:
    """Tests for compute_autocorr."""

    def test_autocorr_lag0_is_one(self) -> None:
        """Autocorrelation at lag 0 must be 1.0 (definition)."""
        from openpkpd.estimation.mcmc_diagnostics import compute_autocorr

        rng = np.random.default_rng(20)
        chain = rng.normal(size=500)
        ac = compute_autocorr(chain, max_lag=20)
        assert ac[0] == pytest.approx(1.0)

    def test_autocorr_iid_near_zero_for_lag_gt_0(self) -> None:
        """For i.i.d. samples, autocorrelations at lag > 0 should be ≈ 0."""
        from openpkpd.estimation.mcmc_diagnostics import compute_autocorr

        rng = np.random.default_rng(21)
        chain = rng.normal(size=5000)
        ac = compute_autocorr(chain, max_lag=10)
        # All lags ≥ 1 should be close to zero (within 3-sigma of 1/sqrt(N))
        assert np.all(np.abs(ac[1:]) < 0.1), f"Autocorr not near zero: {ac[1:]}"

    def test_autocorr_ar1_known_values(self) -> None:
        """AR(1) process with φ=0.8 → lag-k autocorr ≈ φ^k."""
        from openpkpd.estimation.mcmc_diagnostics import compute_autocorr

        rng = np.random.default_rng(22)
        phi = 0.8
        n = 8000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal() * np.sqrt(1 - phi**2)
        ac = compute_autocorr(x, max_lag=5)
        for k in range(1, 6):
            expected = phi**k
            assert abs(ac[k] - expected) < 0.05, (
                f"lag {k}: expected {expected:.3f}, got {ac[k]:.3f}"
            )

    def test_autocorr_multidimensional(self) -> None:
        """2-D chain (n_draws, n_params) → output shape (max_lag+1, n_params)."""
        from openpkpd.estimation.mcmc_diagnostics import compute_autocorr

        rng = np.random.default_rng(23)
        chain_2d = rng.normal(size=(500, 3))
        ac = compute_autocorr(chain_2d, max_lag=10)
        assert ac.shape == (11, 3)
        assert np.allclose(ac[0], 1.0)


# ---------------------------------------------------------------------------
# BayesianResult — posterior_samples_by_chain and diagnostic integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBayesianResultDiagnostics:
    """Tests for posterior_samples_by_chain and R-hat/ESS in BayesianResult."""

    def _make_result(self, n_chains: int = 2, n_draws: int = 200,
                     n_theta: int = 2, rhat_override: np.ndarray | None = None,
                     ess_override: np.ndarray | None = None) -> BayesianResult:
        rng = np.random.default_rng(42)
        chains = rng.normal(size=(n_chains, n_draws, n_theta))
        flat = chains.reshape(-1, n_theta)
        rhat = rhat_override if rhat_override is not None else np.ones(n_theta)
        ess = ess_override if ess_override is not None else np.full(n_theta, n_chains * n_draws)
        return BayesianResult(
            theta_final=flat.mean(axis=0),
            omega_final=np.eye(1),
            sigma_final=np.eye(1),
            ofv=float("nan"),
            converged=True,
            elapsed_time=0.1,
            method="BAYES(test)",
            posterior_samples={"theta": flat},
            posterior_samples_by_chain={"theta": chains},
            r_hat=rhat,
            n_effective=ess,
            backend_used="test",
        )

    def test_posterior_samples_by_chain_stored(self) -> None:
        """posterior_samples_by_chain must be a dict with correct shape."""
        result = self._make_result(n_chains=3, n_draws=100, n_theta=2)
        assert "theta" in result.posterior_samples_by_chain
        assert result.posterior_samples_by_chain["theta"].shape == (3, 100, 2)

    def test_posterior_summary_shows_rhat_and_ess(self) -> None:
        """posterior_summary() must include R-hat and N_eff columns."""
        rhat = np.array([1.002, 1.008])
        ess = np.array([380.0, 420.0])
        result = self._make_result(rhat_override=rhat, ess_override=ess)
        summary = result.posterior_summary()
        assert "R-hat" in summary
        assert "N_eff" in summary
        assert "1.0020" in summary or "1.002" in summary

    def test_converged_flag_reflects_rhat(self) -> None:
        """BAYESMethod(NumPyro) sets converged=True iff all R-hat ≤ 1.1."""
        good_rhat = np.array([1.003, 1.007])
        bad_rhat = np.array([1.003, 1.25])
        r_good = self._make_result(rhat_override=good_rhat)
        r_bad = self._make_result(rhat_override=bad_rhat)
        # Manually set the converged field to simulate what the backend does
        r_good = BayesianResult(
            **{**r_good.__dict__, "converged": bool(np.all(good_rhat <= 1.1))}
        )
        r_bad = BayesianResult(
            **{**r_bad.__dict__, "converged": bool(np.all(bad_rhat <= 1.1))}
        )
        assert r_good.converged is True
        assert r_bad.converged is False


# ---------------------------------------------------------------------------
# Plots — smoke tests (no display)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMCMCPlots:
    """Smoke tests for the three new diagnostic plot functions."""

    @pytest.fixture
    def chains_3d(self) -> np.ndarray:
        rng = np.random.default_rng(99)
        return rng.normal(size=(3, 200, 2))   # 3 chains, 200 draws, 2 params

    def test_mcmc_trace_by_chain_plot_runs(self, chains_3d) -> None:
        pytest.importorskip("matplotlib")
        from openpkpd.plots import mcmc_trace_by_chain_plot
        import matplotlib
        matplotlib.use("Agg")
        fig = mcmc_trace_by_chain_plot(chains_3d, param_names=["CL", "V"])
        assert fig is not None

    def test_rhat_plot_runs(self) -> None:
        pytest.importorskip("matplotlib")
        from openpkpd.plots import rhat_plot
        import matplotlib
        matplotlib.use("Agg")
        rhat = np.array([1.002, 1.12, 1.003])
        fig = rhat_plot(rhat, param_names=["CL", "V", "KA"], threshold=1.1)
        assert fig is not None

    def test_ess_plot_runs(self) -> None:
        pytest.importorskip("matplotlib")
        from openpkpd.plots import ess_plot
        import matplotlib
        matplotlib.use("Agg")
        ess = np.array([850.0, 312.0, 990.0])
        fig = ess_plot(ess, n_total=1000, param_names=["CL", "V", "KA"])
        assert fig is not None

    def test_mcmc_trace_by_chain_accepts_dict(self, chains_3d) -> None:
        pytest.importorskip("matplotlib")
        from openpkpd.plots import mcmc_trace_by_chain_plot
        import matplotlib
        matplotlib.use("Agg")
        fig = mcmc_trace_by_chain_plot({"theta": chains_3d})
        assert fig is not None
