"""
Unit tests for count data models.

Tests cover:
  - PoissonModel: log-PMF correctness, mean_rate, log-likelihood, fit.
  - NegativeBinomialModel: log-PMF, convergence to Poisson for large r, fit.
  - ZeroInflatedPoissonModel: log-PMF at k=0 vs k>0, zero-inflation, fit.
  - CountData dataclass construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import nbinom as scipy_nbinom
from scipy.stats import poisson as scipy_poisson

from openpkpd.models.count import (
    CountData,
    CountResult,
    NegativeBinomialModel,
    PoissonModel,
    ZeroInflatedPoissonModel,
)

# ---------------------------------------------------------------------------
# PoissonModel
# ---------------------------------------------------------------------------


class TestPoissonModel:
    """Tests for PoissonModel."""

    def test_log_pmf_matches_scipy(self) -> None:
        """log_pmf must match scipy.stats.poisson.logpmf."""
        model = PoissonModel()
        for k in range(6):
            for mu in [0.5, 1.0, 3.0, 10.0]:
                assert model.log_pmf(k, mu) == pytest.approx(
                    scipy_poisson.logpmf(k, mu), abs=1e-10
                ), f"Mismatch at k={k}, mu={mu}"

    def test_log_pmf_k3_mu3(self) -> None:
        """P(Y=3 | Poisson(3)) ≈ 0.2240."""
        model = PoissonModel()
        ll = model.log_pmf(3, mu=3.0)
        assert ll == pytest.approx(scipy_poisson.logpmf(3, 3.0), abs=1e-10)

    def test_log_pmf_k0_gives_log_exp_minus_mu(self) -> None:
        """P(Y=0 | Poisson(mu)) = exp(-mu)."""
        model = PoissonModel()
        mu = 2.5
        expected = -mu
        assert model.log_pmf(0, mu) == pytest.approx(expected, abs=1e-10)

    def test_mean_rate_intercept_only(self) -> None:
        """log-linear model with intercept only: mu = exp(params[0])."""
        model = PoissonModel()
        log_rate = np.log(3.0)
        mu_arr = model.mean_rate(np.array([log_rate]))
        assert float(mu_arr[0]) == pytest.approx(3.0, abs=1e-8)

    def test_mean_rate_with_covariate(self) -> None:
        """mu = exp(alpha + beta * x)."""
        model = PoissonModel()
        alpha, beta, x = 0.5, 1.0, 2.0
        params = np.array([alpha, beta])
        cov = {"dose": np.array([x])}
        mu_arr = model.mean_rate(params, cov)
        expected = np.exp(alpha + beta * x)
        assert float(mu_arr[0]) == pytest.approx(expected, abs=1e-8)

    def test_log_likelihood_negative(self) -> None:
        """Log-likelihood should be <= 0."""
        model = PoissonModel()
        data = [
            CountData(
                subject_id=1,
                counts=np.array([0, 1, 2, 3]),
                times=np.array([0.0, 1.0, 2.0, 3.0]),
            )
        ]
        params = np.array([np.log(2.0)])  # mu = 2
        ll = model.log_likelihood(data, params)
        assert ll <= 0.0
        assert np.isfinite(ll)

    def test_log_likelihood_multiple_subjects(self) -> None:
        """LL over multiple subjects should equal sum of individual LLs."""
        model = PoissonModel()
        params = np.array([np.log(3.0)])
        subjects = [
            CountData(subject_id=i, counts=np.array([i]), times=np.array([0.0])) for i in range(5)
        ]
        ll_all = model.log_likelihood(subjects, params)
        ll_sum = sum(model.log_likelihood([s], params) for s in subjects)
        assert ll_all == pytest.approx(ll_sum, abs=1e-10)

    def test_log_likelihood_matches_manual_sum_with_covariate_and_offset(self) -> None:
        """Full Poisson likelihood should match the hand-computed sum."""
        model = PoissonModel()
        alpha, beta = 0.2, 0.4
        x = np.array([-1.0, 0.5, 2.0])
        exposure = np.array([0.5, 1.0, 2.0])
        counts = np.array([0, 2, 4])
        mu = exposure * np.exp(alpha + beta * x)
        data = [
            CountData(
                subject_id=1,
                counts=counts,
                times=np.array([0.0, 1.0, 2.0]),
                offsets=np.log(exposure),
                covariates={"x": x},
            )
        ]

        ll = model.log_likelihood(data, np.array([alpha, beta]))
        expected = float(np.sum(scipy_poisson.logpmf(counts, mu)))

        assert ll == pytest.approx(expected, abs=1e-10)

    def test_fit_recovers_intercept(self) -> None:
        """Fitting to count data should recover the true log-rate."""
        rng = np.random.default_rng(42)
        true_mu = 4.0
        counts = rng.poisson(true_mu, size=200)
        data = [
            CountData(
                subject_id=i,
                counts=np.array([counts[i]]),
                times=np.array([0.0]),
            )
            for i in range(len(counts))
        ]
        model = PoissonModel()
        result = model.fit(data, np.array([1.0]))
        fitted_mu = float(np.exp(result.rate_params[0]))
        assert fitted_mu == pytest.approx(true_mu, rel=0.10)

    def test_fit_result_fields(self) -> None:
        """CountResult fields should have correct types."""
        model = PoissonModel()
        data = [
            CountData(
                subject_id=1,
                counts=np.array([2, 3]),
                times=np.array([0.0, 1.0]),
            )
        ]
        result = model.fit(data, np.array([0.5]))
        assert isinstance(result, CountResult)
        assert isinstance(result.rate_params, np.ndarray)
        assert isinstance(result.ofv, float)
        assert isinstance(result.aic, float)
        assert isinstance(result.converged, bool)
        assert result.dispersion is None

    def test_aic_formula(self) -> None:
        """AIC = OFV + 2 * n_params."""
        model = PoissonModel()
        data = [
            CountData(
                subject_id=1,
                counts=np.array([1]),
                times=np.array([0.0]),
            )
        ]
        result = model.fit(data, np.array([0.0]))
        expected_aic = result.ofv + 2.0 * len(result.rate_params)
        assert result.aic == pytest.approx(expected_aic, abs=1e-6)

    def test_offset_shifts_log_rate(self) -> None:
        """An offset of log(2) should double the expected count."""
        model = PoissonModel()
        params = np.array([0.0])  # log(mu) = 0 => mu = 1 without offset
        mu_no_offset = float(model.mean_rate(params, offset=0.0)[0])
        mu_with_offset = float(model.mean_rate(params, offset=np.log(2.0))[0])
        assert mu_with_offset == pytest.approx(2.0 * mu_no_offset, abs=1e-10)

    def test_fit_recovers_covariate_slope_with_offset(self) -> None:
        """Poisson fit should recover intercept/slope under offsets."""
        rng = np.random.default_rng(123)
        n = 2000
        alpha = 0.3
        beta = 0.6
        x = rng.normal(0.0, 0.8, size=n)
        exposure = rng.uniform(0.5, 2.0, size=n)
        mu = exposure * np.exp(alpha + beta * x)
        counts = rng.poisson(mu)

        data = [
            CountData(
                subject_id=1,
                counts=counts,
                times=np.arange(n, dtype=float),
                offsets=np.log(exposure),
                covariates={"x": x},
            )
        ]

        model = PoissonModel()
        result = model.fit(data, np.array([0.0, 0.0]))

        assert result.converged
        assert result.rate_params[0] == pytest.approx(alpha, abs=0.12)
        assert result.rate_params[1] == pytest.approx(beta, abs=0.12)


# ---------------------------------------------------------------------------
# NegativeBinomialModel
# ---------------------------------------------------------------------------


class TestNegativeBinomialModel:
    """Tests for NegativeBinomialModel."""

    def test_log_pmf_large_r_approaches_poisson(self) -> None:
        """NegBin with large r should approximate Poisson."""
        nb = NegativeBinomialModel(r=1e6)
        pois = PoissonModel()
        for k in [0, 1, 2, 5, 10]:
            ll_nb = nb.log_pmf(k, mu=3.0, r=1e6)
            ll_p = pois.log_pmf(k, mu=3.0)
            assert ll_nb == pytest.approx(ll_p, abs=0.01), (
                f"NB and Poisson differ at k={k}: NB={ll_nb:.6f}, P={ll_p:.6f}"
            )

    def test_log_pmf_r_one(self) -> None:
        """NegBin(r=1) = geometric distribution."""
        nb = NegativeBinomialModel(r=1.0)
        mu = 2.0
        p_geom = 1.0 / (1.0 + mu)
        # P(Y=0 | Geom(p)) = p
        ll = nb.log_pmf(0, mu=mu, r=1.0)
        assert ll == pytest.approx(np.log(p_geom), abs=1e-8)

    def test_dispersion_extracted_when_fixed(self) -> None:
        """Fixed r should appear in CountResult.dispersion."""
        nb = NegativeBinomialModel(r=2.5)
        data = [
            CountData(
                subject_id=1,
                counts=np.array([1, 3, 2]),
                times=np.array([0.0, 1.0, 2.0]),
            )
        ]
        result = nb.fit(data, np.array([0.5]))
        assert result.dispersion == pytest.approx(2.5, abs=1e-6)

    def test_dispersion_estimated(self) -> None:
        """When r is None, dispersion should be estimated and positive."""
        rng = np.random.default_rng(7)
        counts = rng.negative_binomial(n=3, p=0.5, size=100)
        data = [
            CountData(
                subject_id=i,
                counts=np.array([counts[i]]),
                times=np.array([0.0]),
            )
            for i in range(len(counts))
        ]
        nb = NegativeBinomialModel(r=None)
        # init_params: [log_rate, log_r]
        result = nb.fit(data, np.array([0.0, 0.0]))
        assert result.dispersion is not None
        assert result.dispersion > 0.0

    def test_log_likelihood_finite(self) -> None:
        """log_likelihood should be finite for valid inputs."""
        nb = NegativeBinomialModel(r=2.0)
        data = [
            CountData(
                subject_id=1,
                counts=np.array([0, 1, 4]),
                times=np.array([0.0, 1.0, 2.0]),
            )
        ]
        ll = nb.log_likelihood(data, np.array([np.log(2.0)]))
        assert np.isfinite(ll)

    def test_log_likelihood_matches_manual_sum_when_dispersion_is_estimated(self) -> None:
        """NB likelihood should match SciPy when log(r) is in params."""
        mu = 1.7
        r = 2.5
        counts = np.array([0, 1, 4])
        data = [
            CountData(
                subject_id=1,
                counts=counts,
                times=np.array([0.0, 1.0, 2.0]),
            )
        ]
        model = NegativeBinomialModel(r=None)
        params = np.array([np.log(mu), np.log(r)])
        p = r / (r + mu)

        ll = model.log_likelihood(data, params)
        expected = float(np.sum(scipy_nbinom.logpmf(counts, n=r, p=p)))

        assert ll == pytest.approx(expected, abs=1e-10)

    def test_overdispersion_increases_zero_prob(self) -> None:
        """Higher dispersion (lower r) should increase variance and P(Y=0)."""
        # For NB: P(Y=0 | mu, r) = (r / (r+mu))^r
        # As r decreases toward 0, P(Y=0) → 1
        nb_low_r = NegativeBinomialModel(r=0.1)
        nb_high_r = NegativeBinomialModel(r=100.0)
        mu = 2.0
        p_zero_low = np.exp(nb_low_r.log_pmf(0, mu=mu, r=0.1))
        p_zero_high = np.exp(nb_high_r.log_pmf(0, mu=mu, r=100.0))
        assert p_zero_low > p_zero_high

    def test_fit_recovers_mean_and_dispersion(self) -> None:
        """NB fit should recover both mean rate and dispersion on synthetic data."""
        rng = np.random.default_rng(19)
        true_mu = 2.5
        true_r = 4.0
        p = true_r / (true_r + true_mu)
        counts = rng.negative_binomial(n=true_r, p=p, size=2500)
        data = [
            CountData(
                subject_id=1,
                counts=counts,
                times=np.arange(len(counts), dtype=float),
            )
        ]

        model = NegativeBinomialModel(r=None)
        result = model.fit(data, np.array([np.log(2.0), np.log(2.0)]))

        assert result.converged
        assert np.exp(result.rate_params[0]) == pytest.approx(true_mu, rel=0.10)
        assert result.dispersion == pytest.approx(true_r, rel=0.25)


# ---------------------------------------------------------------------------
# ZeroInflatedPoissonModel
# ---------------------------------------------------------------------------


class TestZeroInflatedPoissonModel:
    """Tests for ZeroInflatedPoissonModel."""

    def test_log_pmf_k0_formula(self) -> None:
        """P(Y=0|ZIP) = pi + (1-pi)*exp(-mu)."""
        model = ZeroInflatedPoissonModel()
        pi, mu = 0.3, 2.0
        expected = pi + (1.0 - pi) * np.exp(-mu)
        assert np.exp(model.log_pmf(0, mu=mu, zero_prob=pi)) == pytest.approx(expected, abs=1e-10)

    def test_log_pmf_k_positive(self) -> None:
        """P(Y=k|ZIP) = (1-pi)*Poisson(k|mu) for k > 0."""
        model = ZeroInflatedPoissonModel()
        pi, mu, k = 0.2, 3.0, 2
        expected = (1.0 - pi) * np.exp(scipy_poisson.logpmf(k, mu))
        assert np.exp(model.log_pmf(k, mu=mu, zero_prob=pi)) == pytest.approx(expected, abs=1e-10)

    def test_zero_prob_zero_equals_poisson(self) -> None:
        """ZIP with pi=0 reduces to Poisson."""
        zip_model = ZeroInflatedPoissonModel()
        pois_model = PoissonModel()
        mu = 4.0
        for k in range(5):
            ll_zip = zip_model.log_pmf(k, mu=mu, zero_prob=0.0)
            ll_pois = pois_model.log_pmf(k, mu=mu)
            assert ll_zip == pytest.approx(ll_pois, abs=1e-10)

    def test_zero_inflation_increases_p_zero(self) -> None:
        """Higher pi should increase P(Y=0)."""
        model = ZeroInflatedPoissonModel()
        mu = 2.0
        p_low_pi = np.exp(model.log_pmf(0, mu=mu, zero_prob=0.1))
        p_high_pi = np.exp(model.log_pmf(0, mu=mu, zero_prob=0.6))
        assert p_high_pi > p_low_pi

    def test_log_likelihood_finite(self) -> None:
        """log_likelihood should be finite for valid inputs."""
        model = ZeroInflatedPoissonModel(zero_prob=0.3)
        data = [
            CountData(
                subject_id=1,
                counts=np.array([0, 0, 1, 3, 0]),
                times=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            )
        ]
        ll = model.log_likelihood(data, np.array([np.log(2.0)]))
        assert np.isfinite(ll)

    def test_log_likelihood_matches_manual_zip_mixture_when_pi_is_estimated(self) -> None:
        """ZIP likelihood should match the exact zero/positive mixture formula."""
        pi, mu = 0.35, 2.2
        counts = np.array([0, 2, 0, 1])
        model = ZeroInflatedPoissonModel(zero_prob=None)
        data = [
            CountData(
                subject_id=1,
                counts=counts,
                times=np.arange(len(counts), dtype=float),
            )
        ]
        params = np.array([np.log(mu), np.log(pi / (1.0 - pi))])

        ll = model.log_likelihood(data, params)
        expected_terms = [
            np.log(pi + (1.0 - pi) * np.exp(-mu)),
            np.log(1.0 - pi) + scipy_poisson.logpmf(2, mu),
            np.log(pi + (1.0 - pi) * np.exp(-mu)),
            np.log(1.0 - pi) + scipy_poisson.logpmf(1, mu),
        ]
        expected = float(np.sum(expected_terms))

        assert ll == pytest.approx(expected, abs=1e-10)

    def test_fit_fixed_zero_prob(self) -> None:
        """Fit with fixed pi should return a CountResult."""
        rng = np.random.default_rng(11)
        counts = np.where(rng.uniform(size=100) < 0.3, 0, rng.poisson(2.0, 100))
        data = [
            CountData(
                subject_id=i,
                counts=np.array([counts[i]]),
                times=np.array([0.0]),
            )
            for i in range(len(counts))
        ]
        model = ZeroInflatedPoissonModel(zero_prob=0.3)
        result = model.fit(data, np.array([0.0]))
        assert isinstance(result, CountResult)
        assert np.isfinite(result.ofv)

    def test_fit_fixed_zero_prob_recovers_mean_rate(self) -> None:
        """With pi fixed at truth, ZIP should recover the count-component mean."""
        rng = np.random.default_rng(20260308)
        true_pi, true_mu = 0.35, 2.4
        n = 1500
        counts = np.where(rng.uniform(size=n) < true_pi, 0, rng.poisson(true_mu, size=n))
        data = [
            CountData(
                subject_id=1,
                counts=counts,
                times=np.arange(n, dtype=float),
            )
        ]

        result = ZeroInflatedPoissonModel(zero_prob=true_pi).fit(data, np.array([np.log(1.5)]))

        assert result.converged
        assert np.exp(result.rate_params[0]) == pytest.approx(true_mu, rel=0.10)

    def test_fit_estimated_zero_prob(self) -> None:
        """Fit with estimated pi (None) should converge."""
        rng = np.random.default_rng(22)
        true_pi, true_mu = 0.4, 3.0
        zeros_mask = rng.uniform(size=150) < true_pi
        counts = np.where(zeros_mask, 0, rng.poisson(true_mu, 150))
        data = [
            CountData(
                subject_id=i,
                counts=np.array([counts[i]]),
                times=np.array([0.0]),
            )
            for i in range(len(counts))
        ]
        # init_params: [log_mu, logit_pi]
        model = ZeroInflatedPoissonModel(zero_prob=None)
        result = model.fit(data, np.array([np.log(true_mu), 0.0]))
        assert np.isfinite(result.ofv)

    def test_fit_estimated_zero_prob_beats_poisson_and_matches_grid_optimum(self) -> None:
        """Estimated-pi ZIP should outperform Poisson on excess-zero data."""
        rng = np.random.default_rng(314)
        true_pi, true_mu = 0.45, 2.8
        n = 900
        counts = np.where(rng.uniform(size=n) < true_pi, 0, rng.poisson(true_mu, size=n))
        data = [
            CountData(
                subject_id=1,
                counts=counts,
                times=np.arange(n, dtype=float),
            )
        ]

        zip_model = ZeroInflatedPoissonModel(zero_prob=None)
        zip_result = zip_model.fit(data, np.array([np.log(2.0), 0.0]))
        poisson_result = PoissonModel().fit(
            data, np.array([np.log(max(float(np.mean(counts)), 1e-6))])
        )

        log_mu_grid = np.linspace(np.log(1.4), np.log(4.2), 31)
        logit_pi_grid = np.linspace(-2.5, 2.5, 31)
        grid_best_ofv = min(
            -2.0 * zip_model.log_likelihood(data, np.array([log_mu, logit_pi]))
            for log_mu in log_mu_grid
            for logit_pi in logit_pi_grid
        )

        assert zip_result.converged
        assert np.exp(zip_result.rate_params[0]) == pytest.approx(true_mu, rel=0.15)
        assert zip_result.ofv <= grid_best_ofv + 2.0
        assert zip_result.ofv + 25.0 < poisson_result.ofv


# ---------------------------------------------------------------------------
# CountData dataclass
# ---------------------------------------------------------------------------


class TestCountData:
    """Tests for CountData dataclass."""

    def test_basic_construction(self) -> None:
        """CountData should store arrays correctly."""
        data = CountData(
            subject_id=5,
            counts=np.array([0, 1, 2]),
            times=np.array([0.0, 1.0, 2.0]),
        )
        assert data.subject_id == 5
        assert len(data.counts) == 3
        assert data.offsets is None
        assert data.covariates is None

    def test_with_offsets_and_covariates(self) -> None:
        """CountData with offsets and covariates."""
        data = CountData(
            subject_id=1,
            counts=np.array([3, 5]),
            times=np.array([0.0, 1.0]),
            offsets=np.log(np.array([2.0, 4.0])),
            covariates={"dose": np.array([100.0, 200.0])},
        )
        assert data.offsets is not None
        assert data.covariates is not None
        assert "dose" in data.covariates
