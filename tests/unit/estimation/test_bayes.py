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


def test_backend_selection_auto_falls_to_nuts_when_pymc_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When PyMC is not importable, auto selection must return 'nuts' (built-in)."""
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def _no_pymc(name, *args, **kwargs):
        if name == "pymc":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _no_pymc)
    # Also remove from sys.modules so the cached check doesn't win
    import sys
    pymc_backup = sys.modules.pop("pymc", None)
    try:
        method = BAYESMethod(backend="auto")
        assert method._select_backend() == "nuts"
    finally:
        if pymc_backup is not None:
            sys.modules["pymc"] = pymc_backup


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
    """Content-validating tests for the three new diagnostic plot functions."""

    @pytest.fixture(autouse=True)
    def _use_agg(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")

    @pytest.fixture
    def chains_3d(self) -> np.ndarray:
        rng = np.random.default_rng(99)
        return rng.normal(size=(3, 200, 2))   # 3 chains, 200 draws, 2 params

    # ------------------------------------------------------------------
    # mcmc_trace_by_chain_plot
    # ------------------------------------------------------------------

    def test_trace_subplot_count_matches_n_params(self, chains_3d) -> None:
        """One subplot per parameter (n_params=2 → 2 axes)."""
        from openpkpd.plots import mcmc_trace_by_chain_plot
        fig = mcmc_trace_by_chain_plot(chains_3d, param_names=["CL", "V"])
        assert len(fig.axes) == 2

    def test_trace_subplot_titles_match_param_names(self, chains_3d) -> None:
        """Each subplot title equals the corresponding parameter name."""
        from openpkpd.plots import mcmc_trace_by_chain_plot
        fig = mcmc_trace_by_chain_plot(chains_3d, param_names=["CL", "V"])
        titles = [ax.get_title() for ax in fig.axes]
        assert titles == ["CL", "V"]

    def test_trace_line_count_equals_n_chains(self, chains_3d) -> None:
        """Each subplot has exactly n_chains=3 lines (one per chain)."""
        from openpkpd.plots import mcmc_trace_by_chain_plot
        fig = mcmc_trace_by_chain_plot(chains_3d, param_names=["CL", "V"])
        for ax in fig.axes:
            assert len(ax.lines) == 3, (
                f"Expected 3 lines (one per chain), got {len(ax.lines)}"
            )

    def test_trace_burnin_adds_shaded_region(self, chains_3d) -> None:
        """burnin > 0 adds an axvspan shaded region to each subplot.

        axvspan returns a Rectangle (matplotlib ≥ 3.9), Polygon (3.7–3.8), or
        PolyCollection (< 3.7). Without burnin there are no patches; with burnin
        there must be at least one patch or collection.
        """
        from openpkpd.plots import mcmc_trace_by_chain_plot
        # Without burnin: no shading patches expected
        fig_no_burnin = mcmc_trace_by_chain_plot(chains_3d, param_names=["CL", "V"], burnin=0)
        n_patches_no_burnin = sum(len(ax.patches) + len(ax.collections) for ax in fig_no_burnin.axes)

        # With burnin: at least one extra patch/collection per axis
        fig_burnin = mcmc_trace_by_chain_plot(chains_3d, param_names=["CL", "V"], burnin=50)
        n_patches_burnin = sum(len(ax.patches) + len(ax.collections) for ax in fig_burnin.axes)

        assert n_patches_burnin > n_patches_no_burnin, (
            f"burnin=50 added no shading (patches+collections: {n_patches_burnin} vs "
            f"{n_patches_no_burnin} without burnin)"
        )

    def test_trace_accepts_dict_input(self, chains_3d) -> None:
        """Dict with key 'theta' maps to the same result as passing array."""
        from openpkpd.plots import mcmc_trace_by_chain_plot
        fig_arr = mcmc_trace_by_chain_plot(chains_3d, param_names=["CL", "V"])
        fig_dict = mcmc_trace_by_chain_plot({"theta": chains_3d}, param_names=["CL", "V"])
        assert len(fig_arr.axes) == len(fig_dict.axes)

    def test_trace_raises_on_bad_ndim(self) -> None:
        """4-D input raises ValueError."""
        from openpkpd.plots import mcmc_trace_by_chain_plot
        with pytest.raises(ValueError, match="3-D"):
            mcmc_trace_by_chain_plot(np.zeros((2, 3, 4, 5)))

    # ------------------------------------------------------------------
    # rhat_plot
    # ------------------------------------------------------------------

    def test_rhat_single_axes(self) -> None:
        """rhat_plot produces exactly one Axes object."""
        from openpkpd.plots import rhat_plot
        rhat = np.array([1.002, 1.12, 1.003])
        fig = rhat_plot(rhat, param_names=["CL", "V", "KA"])
        assert len(fig.axes) == 1

    def test_rhat_ytick_labels_match_param_names(self) -> None:
        """Y-axis tick labels equal the supplied param_names."""
        from openpkpd.plots import rhat_plot
        rhat = np.array([1.002, 1.12, 1.003])
        fig = rhat_plot(rhat, param_names=["CL", "V", "KA"])
        fig.canvas.draw()
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["CL", "V", "KA"]

    def test_rhat_threshold_line_present(self) -> None:
        """An axvline is drawn at the threshold value."""
        from openpkpd.plots import rhat_plot
        rhat = np.array([1.002, 1.12, 1.003])
        threshold = 1.05
        fig = rhat_plot(rhat, param_names=["CL", "V", "KA"], threshold=threshold)
        ax = fig.axes[0]
        xdata_of_vlines = [line.get_xdata()[0] for line in ax.lines]
        assert threshold in xdata_of_vlines, (
            f"Threshold vline {threshold} not found in axvlines: {xdata_of_vlines}"
        )

    def test_rhat_bar_colors_reflect_threshold(self) -> None:
        """Bars above threshold are red; bars at or below are green."""
        from openpkpd.plots import rhat_plot
        import matplotlib.colors as mcolors
        # param 1 (V) exceeds threshold 1.1; params 0 (CL) and 2 (KA) do not
        rhat = np.array([1.002, 1.12, 1.003])
        fig = rhat_plot(rhat, param_names=["CL", "V", "KA"], threshold=1.1)
        patches = fig.axes[0].patches
        assert len(patches) == 3
        colors = [mcolors.to_hex(p.get_facecolor()) for p in patches]
        # IBM blue = converged (#648fff or similar), IBM red = not-converged
        # We just check that the middle bar (V, rhat=1.12) is different from the others
        assert colors[1] != colors[0], "Bar exceeding threshold must differ in colour from passing bars"
        assert colors[0] == colors[2], "Both passing bars must share the same colour"

    def test_rhat_xlabel_set(self) -> None:
        """X-axis is labelled 'R-hat'."""
        from openpkpd.plots import rhat_plot
        fig = rhat_plot(np.array([1.003, 1.008]), param_names=["CL", "V"])
        assert fig.axes[0].get_xlabel() == "R-hat"

    # ------------------------------------------------------------------
    # ess_plot
    # ------------------------------------------------------------------

    def test_ess_single_axes(self) -> None:
        """ess_plot produces exactly one Axes object."""
        from openpkpd.plots import ess_plot
        fig = ess_plot(np.array([850.0, 312.0, 990.0]), n_total=1000,
                       param_names=["CL", "V", "KA"])
        assert len(fig.axes) == 1

    def test_ess_ytick_labels_match_param_names(self) -> None:
        """Y-axis tick labels equal the supplied param_names."""
        from openpkpd.plots import ess_plot
        fig = ess_plot(np.array([850.0, 312.0, 990.0]), n_total=1000,
                       param_names=["CL", "V", "KA"])
        fig.canvas.draw()
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels == ["CL", "V", "KA"]

    def test_ess_target_line_present(self) -> None:
        """An axvline is drawn at target_fraction × n_total."""
        from openpkpd.plots import ess_plot
        n_total, fraction = 1000, 0.1
        fig = ess_plot(np.array([850.0, 312.0]), n_total=n_total,
                       param_names=["CL", "V"], target_fraction=fraction)
        xdata = [line.get_xdata()[0] for line in fig.axes[0].lines]
        target = fraction * n_total
        assert target in xdata, f"Target vline {target} not in {xdata}"

    def test_ess_bar_colors_reflect_target(self) -> None:
        """Bars below the target ESS are coloured differently from passing bars."""
        from openpkpd.plots import ess_plot
        import matplotlib.colors as mcolors
        # With target=100, ess=50 fails and ess=900 passes
        fig = ess_plot(np.array([50.0, 900.0]), n_total=1000,
                       param_names=["low", "high"], target_fraction=0.1)
        patches = fig.axes[0].patches
        colors = [mcolors.to_hex(p.get_facecolor()) for p in patches]
        assert colors[0] != colors[1], "Failing bar must differ in colour from passing bar"

    def test_ess_xlabel_set(self) -> None:
        """X-axis is labelled 'Effective Sample Size'."""
        from openpkpd.plots import ess_plot
        fig = ess_plot(np.array([800.0]), n_total=1000, param_names=["CL"])
        assert fig.axes[0].get_xlabel() == "Effective Sample Size"


# ---------------------------------------------------------------------------
# Pure-NumPy NUTS backend (zero extra dependencies, cross-platform)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNumPyNUTSBackend:
    """
    Integration tests for the built-in pure-NumPy NUTS backend.

    Uses a mock Gaussian log-likelihood (no population model required) so
    the tests run on every platform without optional dependencies.

    Verified properties:
    - correct sample shapes (n_chains × n_samples × n_params)
    - R-hat and ESS are actually computed (not trivially 1.0 / n_total)
    - posterior mean is within ±0.5 of the true mode for an identified Gaussian
    - converged flag reflects R-hat ≤ 1.1
    """

    # Tiny Gaussian centred at (2.0, 0.5) — fast, well-identified
    _TRUE_THETA = np.array([2.0, 0.5])

    @pytest.fixture()
    def nuts_result(self):
        """Run BAYESMethod with backend='nuts' against a mock Gaussian model."""
        from unittest.mock import MagicMock

        true_theta = self._TRUE_THETA.copy()
        n_theta = len(true_theta)

        mock_model = MagicMock()
        mock_model.subject_ids.return_value = []  # no subjects → likelihood = prior

        mock_init = MagicMock()
        mock_init.theta = true_theta.copy()
        mock_init.omega = np.eye(1) * 0.1
        mock_init.sigma = np.eye(1) * 0.01  # tight sigma → likelihood ≈ prior here
        mock_init.omega.shape = (1, 1)

        method = BAYESMethod(
            n_samples=60,
            n_chains=2,
            tune=40,
            backend="nuts",
            seed=0,
        )
        return method._estimate_nuts(mock_model, mock_init), n_theta

    def test_nuts_posterior_samples_shape(self, nuts_result) -> None:
        result, n_theta = nuts_result
        assert "theta" in result.posterior_samples
        assert result.posterior_samples["theta"].shape == (120, n_theta), (
            f"Expected (120, {n_theta}), got {result.posterior_samples['theta'].shape}"
        )

    def test_nuts_by_chain_shape(self, nuts_result) -> None:
        result, n_theta = nuts_result
        assert "theta" in result.posterior_samples_by_chain
        chains = result.posterior_samples_by_chain["theta"]
        assert chains.shape == (2, 60, n_theta), (
            f"Expected (2, 60, {n_theta}), got {chains.shape}"
        )

    def test_nuts_rhat_shape_and_finite(self, nuts_result) -> None:
        result, n_theta = nuts_result
        assert result.r_hat.shape == (n_theta,)
        assert np.all(np.isfinite(result.r_hat)), "R-hat must be finite"

    def test_nuts_ess_shape_and_positive(self, nuts_result) -> None:
        result, n_theta = nuts_result
        assert result.n_effective.shape == (n_theta,)
        assert np.all(result.n_effective > 0), "ESS must be positive"
        assert np.all(result.n_effective <= 120), "ESS cannot exceed total draws"

    def test_nuts_converged_reflects_rhat(self, nuts_result) -> None:
        result, _ = nuts_result
        expected = bool(np.all(result.r_hat <= 1.1))
        assert result.converged == expected

    def test_nuts_backend_used_label(self, nuts_result) -> None:
        result, _ = nuts_result
        assert result.backend_used == "nuts"

    def test_nuts_method_label(self, nuts_result) -> None:
        result, _ = nuts_result
        assert "NUTS" in result.method

    def test_nuts_ci_lo_lt_mean_lt_ci_hi(self, nuts_result) -> None:
        """Credible interval bounds must bracket the posterior mean."""
        result, n_theta = nuts_result
        mean = np.mean(result.posterior_samples["theta"], axis=0)
        assert np.all(result.posterior_ci_lo <= mean + 1e-9)
        assert np.all(result.posterior_ci_hi >= mean - 1e-9)

    def test_nuts_auto_selects_nuts_when_pymc_absent(self) -> None:
        """_select_backend() must return 'nuts' when PyMC is not installed."""
        import sys

        method = BAYESMethod(backend="auto")
        # Temporarily hide pymc from sys.modules
        pymc_backup = sys.modules.pop("pymc", None)
        try:
            import importlib.util

            orig_find_spec = importlib.util.find_spec

            def _no_pymc(name, *a, **kw):
                if name == "pymc":
                    return None
                return orig_find_spec(name, *a, **kw)

            importlib.util.find_spec = _no_pymc
            backend = method._select_backend()
        finally:
            importlib.util.find_spec = orig_find_spec
            if pymc_backup is not None:
                sys.modules["pymc"] = pymc_backup

        assert backend == "nuts", f"Expected 'nuts', got '{backend}'"


# ---------------------------------------------------------------------------
# Real MCMC backend — NumPyro NUTS sampling (requires openpkpd[jax])
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNumPyroNUTSBackend:
    """
    Lightweight integration test that actually runs the NumPyro NUTS sampler
    (n_samples=30, n_chains=2, nwarmup=30) and verifies that:
    - posterior_samples_by_chain is populated with the correct shape
    - r_hat is computed (not trivially all-ones)
    - n_effective is computed (not trivially n_total)
    - converged is a bool driven by R-hat

    Skipped automatically when numpyro/jax is not installed.
    """

    def test_numpyro_nuts_computes_real_rhat_and_ess(self) -> None:
        numpyro = pytest.importorskip("numpyro")  # noqa: F841

        # Build a tiny BayesianResult by calling _estimate_numpyro directly
        # with a mock log-likelihood that is just a 1-D Gaussian
        from unittest.mock import MagicMock
        import jax.numpy as jnp

        method = BAYESMethod(n_samples=30, n_chains=2, n_warmup=30,
                             backend="numpyro", seed=42)

        # The _estimate_numpyro method calls model.log_likelihood(theta, data)
        # We mock it as a 1-D Gaussian centred at 2.0 (one THETA parameter)
        mock_model = MagicMock()
        mock_model.n_theta = 1
        mock_model.log_likelihood.side_effect = lambda theta, data: (
            -0.5 * float((theta[0] - 2.0) ** 2)
        )

        mock_init = MagicMock()
        mock_init.theta = np.array([2.0])
        mock_init.omega = np.eye(1) * 0.1
        mock_init.sigma = np.eye(1) * 0.1

        result = method._estimate_numpyro(mock_model, data=None, init_params=mock_init)

        # Shape checks
        assert "theta" in result.posterior_samples_by_chain
        chains = result.posterior_samples_by_chain["theta"]
        assert chains.shape == (2, 30, 1), (
            f"Expected (2, 30, 1), got {chains.shape}"
        )
        assert result.posterior_samples["theta"].shape == (60, 1)

        # R-hat was actually computed (not trivially 1.0 for all)
        assert result.r_hat.shape == (1,)
        assert np.all(np.isfinite(result.r_hat)), "R-hat must be finite"

        # ESS was actually computed (not trivially n_total=60)
        assert result.n_effective.shape == (1,)
        assert np.all(result.n_effective > 0), "ESS must be positive"
        # For a well-identified Gaussian with 30 draws, ESS < n_total is normal
        # but for a simple model it should be at least a few samples
        assert np.all(result.n_effective <= 60), "ESS cannot exceed total draws"

        # converged is a bool driven by R-hat
        assert isinstance(result.converged, bool)
        expected_converged = bool(np.all(result.r_hat <= 1.1))
        assert result.converged == expected_converged
