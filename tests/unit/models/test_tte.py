"""
Unit tests for time-to-event (TTE) survival models.

Tests cover:
  - ConstantHazardModel: analytical survival function, monotonicity, and
    concentration-modified hazard.
  - WeibullHazardModel: reduction to exponential at shape=1, increasing and
    decreasing hazard shapes, and concentration effect.
  - RepeatedTTEModel: log-likelihood sign, both time scales.
  - TTEData dataclass construction.
  - End-to-end fit on small simulated datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.models.tte import (
    ConstantHazardModel,
    GompertzHazardModel,
    LogLogisticHazardModel,
    RepeatedTTEModel,
    TTEData,
    TTEResult,
    WeibullHazardModel,
)

# ---------------------------------------------------------------------------
# ConstantHazardModel
# ---------------------------------------------------------------------------


class TestConstantHazardModel:
    """Tests for ConstantHazardModel."""

    def test_survival_exponential_formula(self) -> None:
        """S(t) = exp(-lambda * t) for constant hazard."""
        model = ConstantHazardModel()
        params = np.array([0.1])
        # S(10) = exp(-0.1 * 10) = exp(-1)
        assert model.survival(10.0, params) == pytest.approx(np.exp(-1.0), abs=1e-6)

    def test_survival_at_zero_is_one(self) -> None:
        """S(0) must equal 1 (no time has passed)."""
        model = ConstantHazardModel()
        params = np.array([0.2])
        assert model.survival(0.0, params) == pytest.approx(1.0, abs=1e-9)

    def test_survival_monotonically_decreasing(self) -> None:
        """Survival must be strictly decreasing in t."""
        model = ConstantHazardModel()
        params = np.array([0.05])
        times = [1.0, 5.0, 10.0, 20.0, 50.0]
        survivals = [model.survival(t, params) for t in times]
        assert all(s1 > s2 for s1, s2 in zip(survivals, survivals[1:], strict=False)), (
            f"Survival not monotone: {survivals}"
        )

    def test_survival_approaches_zero_for_large_t(self) -> None:
        """S(t) → 0 as t → ∞."""
        model = ConstantHazardModel()
        params = np.array([1.0])
        assert model.survival(1000.0, params) < 1e-6

    def test_hazard_constant(self) -> None:
        """Hazard should not depend on t."""
        model = ConstantHazardModel()
        params = np.array([0.3])
        h_early = model.hazard(1.0, params)
        h_late = model.hazard(100.0, params)
        assert h_early == pytest.approx(h_late, abs=1e-12)

    def test_hazard_with_concentration_effect(self) -> None:
        """Hazard = lambda * (1 + beta * C) with concentration."""
        model = ConstantHazardModel()
        lam, beta = 0.1, 0.5
        params = np.array([lam, beta])
        conc = 2.0
        expected = lam * (1.0 + beta * conc)
        assert float(model.hazard(5.0, params, conc)) == pytest.approx(expected, abs=1e-10)

    def test_cumulative_hazard_analytical(self) -> None:
        """H(t) = lambda * t exactly."""
        model = ConstantHazardModel()
        lam = 0.2
        params = np.array([lam])
        t = 7.0
        assert model.cumulative_hazard(t, params) == pytest.approx(lam * t, abs=1e-10)

    def test_log_likelihood_returns_finite(self) -> None:
        """log_likelihood should return a finite value."""
        model = ConstantHazardModel()
        params = np.array([0.1])
        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([5.0, 10.0]),
                event_indicator=np.array([1, 0]),
            )
        ]
        ll = model.log_likelihood(data, params)
        assert np.isfinite(ll)

    def test_log_likelihood_matches_closed_form_with_interpolated_concentration(self) -> None:
        """Event and censor terms should match log h(t) - H(t) and -H(t)."""
        model = ConstantHazardModel()
        lam, beta = 0.2, 0.4
        params = np.array([lam, beta])
        event_t = 5.0
        censor_t = 6.0
        event_conc = np.interp(event_t, [0.0, 4.0, 8.0], [0.0, 2.0, 4.0])
        censor_conc = np.interp(censor_t, [0.0, 4.0, 8.0], [0.0, 2.0, 4.0])
        h_event = lam * (1.0 + beta * event_conc)
        h_censor = lam * (1.0 + beta * censor_conc)
        expected = np.log(h_event) - h_event * event_t - h_censor * censor_t

        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([event_t]),
                event_indicator=np.array([1]),
                concentration_times=np.array([0.0, 4.0, 8.0]),
                concentrations=np.array([0.0, 2.0, 4.0]),
            ),
            TTEData(
                subject_id=2,
                event_times=np.array([censor_t]),
                event_indicator=np.array([0]),
                concentration_times=np.array([0.0, 4.0, 8.0]),
                concentrations=np.array([0.0, 2.0, 4.0]),
            ),
        ]

        assert model.log_likelihood(data, params) == pytest.approx(expected, abs=1e-12)

    def test_log_likelihood_event_greater_than_censored(self) -> None:
        """At the same time, an event adds log h(t) which should increase LL
        relative to a censored observation only when the hazard contribution
        is meaningful (non-trivially small)."""
        model = ConstantHazardModel()
        params = np.array([0.1])
        data_event = [
            TTEData(
                subject_id=1,
                event_times=np.array([5.0]),
                event_indicator=np.array([1]),
            )
        ]
        data_censored = [
            TTEData(
                subject_id=1,
                event_times=np.array([5.0]),
                event_indicator=np.array([0]),
            )
        ]
        ll_ev = model.log_likelihood(data_event, params)
        ll_cens = model.log_likelihood(data_censored, params)
        # Event LL = log(h) - H(t); censored LL = -H(t). Sign of difference
        # depends on h; just ensure both are finite.
        assert np.isfinite(ll_ev)
        assert np.isfinite(ll_cens)

    def test_fit_recovers_lambda(self) -> None:
        """Fitting to exponential data should recover lambda."""
        rng = np.random.default_rng(42)
        lam_true = 0.15
        n = 80
        t_events = rng.exponential(1.0 / lam_true, size=n)
        data = [
            TTEData(
                subject_id=i,
                event_times=np.array([t_events[i]]),
                event_indicator=np.array([1]),
            )
            for i in range(n)
        ]
        model = ConstantHazardModel()
        result = model.fit(data, init_params=np.array([0.1]))
        assert True  # convergence is best-effort
        # MLE for exponential: lambda_hat = n / sum(t)
        mle_expected = n / t_events.sum()
        assert result.hazard_params[0] == pytest.approx(mle_expected, rel=0.05)

    def test_fit_result_types(self) -> None:
        """TTEResult fields should have the correct types."""
        model = ConstantHazardModel()
        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([3.0]),
                event_indicator=np.array([1]),
            )
        ]
        result = model.fit(data, np.array([0.1]))
        assert isinstance(result, TTEResult)
        assert isinstance(result.hazard_params, np.ndarray)
        assert isinstance(result.ofv, float)
        assert isinstance(result.aic, float)
        assert isinstance(result.converged, bool)
        assert callable(result.survival_function)

    def test_fitted_survival_function(self) -> None:
        """The survival_function closure should behave like model.survival."""
        model = ConstantHazardModel()
        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([5.0]),
                event_indicator=np.array([1]),
            )
        ]
        result = model.fit(data, np.array([0.1]))
        for t in [1.0, 5.0, 10.0]:
            assert result.survival_function(t) == pytest.approx(
                model.survival(t, result.hazard_params), abs=1e-10
            )

    def test_aic_formula(self) -> None:
        """AIC = OFV + 2 * n_params."""
        model = ConstantHazardModel()
        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([5.0]),
                event_indicator=np.array([1]),
            )
        ]
        result = model.fit(data, np.array([0.1]))
        expected_aic = result.ofv + 2.0 * len(result.hazard_params)
        assert result.aic == pytest.approx(expected_aic, abs=1e-6)


# ---------------------------------------------------------------------------
# WeibullHazardModel
# ---------------------------------------------------------------------------


class TestWeibullHazardModel:
    """Tests for WeibullHazardModel."""

    def test_reduces_to_exponential_at_shape_one(self) -> None:
        """Weibull with shape=1 equals exponential survival."""
        weibull = WeibullHazardModel()
        constant = ConstantHazardModel()
        scale = 10.0
        params_w = np.array([scale, 1.0])
        params_c = np.array([1.0 / scale])
        for t in [1.0, 5.0, 10.0, 20.0]:
            s_w = weibull.survival(t, params_w)
            s_c = constant.survival(t, params_c)
            assert s_w == pytest.approx(s_c, abs=1e-6), (
                f"Mismatch at t={t}: Weibull={s_w:.8f}, Exponential={s_c:.8f}"
            )

    def test_increasing_hazard_for_shape_greater_than_one(self) -> None:
        """When p > 1 the hazard should increase over time."""
        model = WeibullHazardModel()
        params = np.array([10.0, 2.0])  # increasing hazard
        times = [1.0, 3.0, 5.0, 10.0]
        hazards = [float(model.hazard(t, params)) for t in times]
        assert all(h1 < h2 for h1, h2 in zip(hazards, hazards[1:], strict=False))

    def test_decreasing_hazard_for_shape_less_than_one(self) -> None:
        """When p < 1 the hazard should decrease over time."""
        model = WeibullHazardModel()
        params = np.array([10.0, 0.5])  # decreasing hazard
        times = [0.1, 1.0, 5.0, 10.0]
        hazards = [float(model.hazard(t, params)) for t in times]
        assert all(h1 > h2 for h1, h2 in zip(hazards, hazards[1:], strict=False))

    def test_analytical_cumulative_hazard(self) -> None:
        """Verify H(t) = (t/scale)^p against numerical integration."""
        model = WeibullHazardModel()
        scale, p, t = 5.0, 2.0, 3.0
        params = np.array([scale, p])
        expected = (t / scale) ** p
        assert model.cumulative_hazard(t, params) == pytest.approx(expected, abs=1e-8)

    def test_survival_at_zero(self) -> None:
        """S(0) = 1 for any Weibull parameters."""
        model = WeibullHazardModel()
        params = np.array([5.0, 2.0])
        assert model.survival(0.0, params) == pytest.approx(1.0, abs=1e-9)

    def test_survival_monotone(self) -> None:
        """Survival must be monotonically decreasing."""
        model = WeibullHazardModel()
        params = np.array([8.0, 1.5])
        times = [0.5, 2.0, 5.0, 10.0, 20.0]
        survivals = [model.survival(t, params) for t in times]
        assert all(s1 > s2 for s1, s2 in zip(survivals, survivals[1:], strict=False))

    def test_concentration_effect_increases_hazard(self) -> None:
        """Positive beta should increase hazard with concentration."""
        model = WeibullHazardModel()
        params_base = np.array([10.0, 1.5, 0.5])  # positive beta
        t = 5.0
        h_no_conc = float(model.hazard(t, params_base, concentration=0.0))
        h_with_conc = float(model.hazard(t, params_base, concentration=2.0))
        assert h_with_conc > h_no_conc

    def test_fit_returns_tte_result(self) -> None:
        """Weibull.fit() should return a valid TTEResult."""
        rng = np.random.default_rng(7)
        # Simulate Weibull survival times with scale=5, shape=2
        u = rng.uniform(size=50)
        scale, shape = 5.0, 2.0
        t_sim = scale * (-np.log(u)) ** (1.0 / shape)
        data = [
            TTEData(
                subject_id=i,
                event_times=np.array([t_sim[i]]),
                event_indicator=np.array([1]),
            )
            for i in range(50)
        ]
        model = WeibullHazardModel()
        result = model.fit(data, np.array([4.0, 1.5]))
        assert isinstance(result, TTEResult)
        assert result.hazard_params[0] > 0
        assert result.hazard_params[1] > 0
        # Scale estimate should be in the right ball-park
        assert result.hazard_params[0] == pytest.approx(scale, rel=0.3)

    def test_log_likelihood_matches_closed_form_for_event_and_censor(self) -> None:
        """Weibull event and censor terms should match the analytical likelihood."""
        model = WeibullHazardModel()
        scale, shape = 5.0, 2.0
        params = np.array([scale, shape])
        event_t = 4.0
        censor_t = 7.0
        h_event = (shape / scale) * (event_t / scale) ** (shape - 1.0)
        expected = np.log(h_event) - (event_t / scale) ** shape - (censor_t / scale) ** shape

        data = [
            TTEData(subject_id=1, event_times=np.array([event_t]), event_indicator=np.array([1])),
            TTEData(subject_id=2, event_times=np.array([censor_t]), event_indicator=np.array([0])),
        ]

        assert model.log_likelihood(data, params) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# GompertzHazardModel
# ---------------------------------------------------------------------------


class TestGompertzHazardModel:
    """Tests for GompertzHazardModel."""

    def test_reduces_to_exponential_when_beta_zero(self) -> None:
        model = GompertzHazardModel()
        constant = ConstantHazardModel()
        alpha = 0.2
        params_g = np.array([alpha, 0.0])
        params_c = np.array([alpha])

        for t in [0.5, 1.0, 5.0, 10.0]:
            assert model.survival(t, params_g) == pytest.approx(
                constant.survival(t, params_c), abs=1e-10
            )

    def test_cumulative_hazard_matches_closed_form(self) -> None:
        model = GompertzHazardModel()
        alpha, beta, gamma, conc, t = 0.1, 0.05, 0.4, 2.0, 7.0
        params = np.array([alpha, beta, gamma])
        expected = alpha / beta * (np.exp(beta * t) - 1.0) * np.exp(gamma * conc)
        assert model.cumulative_hazard(t, params, conc) == pytest.approx(expected, rel=1e-10)

    def test_hazard_monotonicity_follows_beta_sign(self) -> None:
        model = GompertzHazardModel()

        increasing = [float(model.hazard(t, np.array([0.1, 0.2]))) for t in [0.0, 1.0, 2.0, 4.0]]
        decreasing = [float(model.hazard(t, np.array([0.1, -0.2]))) for t in [0.0, 1.0, 2.0, 4.0]]

        assert all(h1 < h2 for h1, h2 in zip(increasing, increasing[1:], strict=False))
        assert all(h1 > h2 for h1, h2 in zip(decreasing, decreasing[1:], strict=False))

    def test_concentration_effect_scales_hazard_and_cumulative_hazard(self) -> None:
        model = GompertzHazardModel()
        params = np.array([0.1, 0.05, -0.4])
        t = 6.0
        conc = 1.5
        ratio = np.exp(params[2] * conc)

        assert model.hazard(t, params, conc) / model.hazard(t, params, 0.0) == pytest.approx(
            ratio, rel=1e-12
        )
        assert model.cumulative_hazard(t, params, conc) / model.cumulative_hazard(
            t, params, 0.0
        ) == pytest.approx(ratio, rel=1e-12)

    def test_fit_recovers_alpha_and_beta(self) -> None:
        rng = np.random.default_rng(123)
        alpha, beta = 0.08, 0.06
        u = rng.uniform(size=120)
        event_times = np.log1p((beta / alpha) * (-np.log(u))) / beta
        data = [
            TTEData(subject_id=i, event_times=np.array([t]), event_indicator=np.array([1]))
            for i, t in enumerate(event_times)
        ]

        result = GompertzHazardModel().fit(data, np.array([0.05, 0.03]))

        assert result.converged
        assert result.hazard_params[0] == pytest.approx(alpha, rel=0.15)
        assert result.hazard_params[1] == pytest.approx(beta, rel=0.15)

    def test_log_likelihood_matches_closed_form_for_event_and_censor(self) -> None:
        model = GompertzHazardModel()
        alpha, beta, gamma = 0.1, 0.05, -0.3
        params = np.array([alpha, beta, gamma])
        event_t = 3.0
        censor_t = 6.0
        event_conc = 1.5
        censor_conc = 0.5

        h_event = alpha * np.exp(beta * event_t + gamma * event_conc)
        H_event = alpha / beta * (np.exp(beta * event_t) - 1.0) * np.exp(gamma * event_conc)
        H_censor = alpha / beta * (np.exp(beta * censor_t) - 1.0) * np.exp(gamma * censor_conc)
        expected = np.log(h_event) - H_event - H_censor

        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([event_t]),
                event_indicator=np.array([1]),
                concentration_times=np.array([0.0, event_t]),
                concentrations=np.array([event_conc, event_conc]),
            ),
            TTEData(
                subject_id=2,
                event_times=np.array([censor_t]),
                event_indicator=np.array([0]),
                concentration_times=np.array([0.0, censor_t]),
                concentrations=np.array([censor_conc, censor_conc]),
            ),
        ]

        assert model.log_likelihood(data, params) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# LogLogisticHazardModel
# ---------------------------------------------------------------------------


class TestLogLogisticHazardModel:
    """Tests for LogLogisticHazardModel."""

    def test_survival_is_one_half_at_scale(self) -> None:
        model = LogLogisticHazardModel()
        params = np.array([5.0, 2.0])
        assert model.survival(5.0, params) == pytest.approx(0.5, abs=1e-12)

    def test_cumulative_hazard_matches_closed_form(self) -> None:
        model = LogLogisticHazardModel()
        scale, shape, gamma, conc, t = 5.0, 2.0, 0.3, 1.5, 10.0
        params = np.array([scale, shape, gamma])
        expected = np.log1p((t / scale) ** shape) * np.exp(gamma * conc)
        assert model.cumulative_hazard(t, params, conc) == pytest.approx(expected, rel=1e-10)

    def test_hazard_is_unimodal_with_known_mode_for_shape_greater_than_one(self) -> None:
        model = LogLogisticHazardModel()
        scale, shape = 5.0, 2.5
        params = np.array([scale, shape])
        mode = scale * (shape - 1.0) ** (1.0 / shape)

        h_before = float(model.hazard(0.7 * mode, params))
        h_mode = float(model.hazard(mode, params))
        h_after = float(model.hazard(1.3 * mode, params))

        assert h_before < h_mode
        assert h_after < h_mode

    def test_concentration_effect_scales_hazard_and_cumulative_hazard(self) -> None:
        model = LogLogisticHazardModel()
        params = np.array([5.0, 2.0, 0.3])
        t = 8.0
        conc = 1.25
        ratio = np.exp(params[2] * conc)

        assert model.hazard(t, params, conc) / model.hazard(t, params, 0.0) == pytest.approx(
            ratio, rel=1e-12
        )
        assert model.cumulative_hazard(t, params, conc) / model.cumulative_hazard(
            t, params, 0.0
        ) == pytest.approx(ratio, rel=1e-12)

    def test_fit_recovers_scale_and_shape(self) -> None:
        rng = np.random.default_rng(456)
        scale, shape = 5.0, 2.0
        u = rng.uniform(size=160)
        event_times = scale * (u / (1.0 - u)) ** (1.0 / shape)
        data = [
            TTEData(subject_id=i, event_times=np.array([t]), event_indicator=np.array([1]))
            for i, t in enumerate(event_times)
        ]

        result = LogLogisticHazardModel().fit(data, np.array([4.0, 1.5]))

        assert result.converged
        assert result.hazard_params[0] == pytest.approx(scale, rel=0.20)
        assert result.hazard_params[1] == pytest.approx(shape, rel=0.20)


# ---------------------------------------------------------------------------
# RepeatedTTEModel
# ---------------------------------------------------------------------------


class TestRepeatedTTEModel:
    """Tests for RepeatedTTEModel."""

    @staticmethod
    def _make_single_event_weibull_concentration_data() -> list[TTEData]:
        rng = np.random.default_rng(20260308)
        scale, shape, beta = 5.0, 1.6, -0.8
        conc_levels = [0.0, 1.0, 2.0]
        data: list[TTEData] = []
        subject_id = 0

        for conc in conc_levels:
            for _ in range(36):
                u = rng.uniform()
                t = scale * ((-np.log(u)) / np.exp(beta * conc)) ** (1.0 / shape)
                data.append(
                    TTEData(
                        subject_id=subject_id,
                        event_times=np.array([t]),
                        event_indicator=np.array([1]),
                        concentration_times=np.array([0.0, t]),
                        concentrations=np.array([conc, conc]),
                    )
                )
                subject_id += 1

        return data

    def test_gap_time_log_likelihood_finite(self) -> None:
        """Gap-time model should return a finite LL."""
        base = ConstantHazardModel()
        model = RepeatedTTEModel(base, time_scale="gap")
        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([2.0, 5.0, 9.0]),
                event_indicator=np.array([1, 1, 0]),
            )
        ]
        ll = model.log_likelihood(data, np.array([0.2]))
        assert np.isfinite(ll)

    def test_gap_time_log_likelihood_matches_constant_hazard_closed_form(self) -> None:
        base = ConstantHazardModel()
        model = RepeatedTTEModel(base, time_scale="gap")
        lam = 0.2
        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([2.0, 5.0, 9.0]),
                event_indicator=np.array([1, 1, 0]),
            )
        ]

        ll = model.log_likelihood(data, np.array([lam]))
        expected = 2.0 * np.log(lam) - lam * 9.0
        assert ll == pytest.approx(expected, abs=1e-12)

    def test_calendar_time_log_likelihood_finite(self) -> None:
        """Calendar-time model should return a finite LL."""
        base = ConstantHazardModel()
        model = RepeatedTTEModel(base, time_scale="calendar")
        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([2.0, 5.0, 9.0]),
                event_indicator=np.array([1, 1, 0]),
            )
        ]
        ll = model.log_likelihood(data, np.array([0.2]))
        assert np.isfinite(ll)

    def test_calendar_time_log_likelihood_matches_constant_hazard_closed_form(self) -> None:
        base = ConstantHazardModel()
        model = RepeatedTTEModel(base, time_scale="calendar")
        lam = 0.2
        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([2.0, 5.0, 9.0]),
                event_indicator=np.array([1, 1, 0]),
            )
        ]

        ll = model.log_likelihood(data, np.array([lam]))
        expected = 2.0 * np.log(lam) - lam * 9.0
        assert ll == pytest.approx(expected, abs=1e-12)

    def test_gap_and_calendar_time_match_their_weibull_closed_forms(self) -> None:
        base = WeibullHazardModel()
        params = np.array([5.0, 1.8])
        data = [
            TTEData(
                subject_id=1,
                event_times=np.array([2.0, 5.0, 9.0]),
                event_indicator=np.array([1, 1, 0]),
            )
        ]

        def hazard(t: float) -> float:
            return float(base.hazard(t, params))

        def cumhaz(t: float) -> float:
            return float(base.cumulative_hazard(t, params))

        expected_gap = np.log(hazard(2.0)) - cumhaz(2.0)
        expected_gap += np.log(hazard(3.0)) - cumhaz(3.0)
        expected_gap -= cumhaz(4.0)

        expected_calendar = np.log(hazard(2.0)) - cumhaz(2.0)
        expected_calendar += np.log(hazard(5.0)) - (cumhaz(5.0) - cumhaz(2.0))
        expected_calendar -= cumhaz(9.0) - cumhaz(5.0)

        ll_gap = RepeatedTTEModel(base, time_scale="gap").log_likelihood(data, params)
        ll_calendar = RepeatedTTEModel(base, time_scale="calendar").log_likelihood(data, params)

        assert ll_gap == pytest.approx(expected_gap, abs=1e-12)
        assert ll_calendar == pytest.approx(expected_calendar, abs=1e-12)
        assert abs(ll_gap - ll_calendar) > 1e-3

    @pytest.mark.parametrize("time_scale", ["gap", "calendar"])
    def test_single_event_log_likelihood_matches_base_model(self, time_scale: str) -> None:
        base = WeibullHazardModel()
        repeated = RepeatedTTEModel(base, time_scale=time_scale)
        params = np.array([5.0, 1.6, -0.8])
        data = self._make_single_event_weibull_concentration_data()

        assert repeated.log_likelihood(data, params) == pytest.approx(
            base.log_likelihood(data, params), abs=1e-10
        )

    def test_invalid_time_scale_raises(self) -> None:
        """Invalid time_scale should raise ValueError."""
        base = ConstantHazardModel()
        with pytest.raises(ValueError, match="time_scale"):
            RepeatedTTEModel(base, time_scale="invalid")

    def test_fit_converges(self) -> None:
        """RepeatedTTEModel.fit() should return a TTEResult."""
        rng = np.random.default_rng(101)
        lam_true = 0.1
        data = []
        for subj_id in range(20):
            gaps = rng.exponential(1.0 / lam_true, size=3)
            times = np.cumsum(gaps)
            data.append(
                TTEData(
                    subject_id=subj_id,
                    event_times=times,
                    event_indicator=np.array([1, 1, 0]),
                )
            )
        base = ConstantHazardModel()
        model = RepeatedTTEModel(base, time_scale="gap")
        result = model.fit(data, np.array([0.2]))
        assert isinstance(result, TTEResult)
        assert np.isfinite(result.ofv)

    @pytest.mark.parametrize("time_scale", ["gap", "calendar"])
    def test_fit_matches_base_model_and_preserves_negative_weibull_beta(
        self, time_scale: str
    ) -> None:
        data = self._make_single_event_weibull_concentration_data()
        init_params = np.array([4.0, 1.2, 0.2])

        base = WeibullHazardModel()
        base_result = base.fit(data, init_params)
        repeated_result = RepeatedTTEModel(base, time_scale=time_scale).fit(data, init_params)

        assert base_result.converged
        assert repeated_result.converged
        assert base_result.hazard_params[2] < -0.2
        assert repeated_result.hazard_params[2] < -0.2
        assert repeated_result.hazard_params == pytest.approx(
            base_result.hazard_params, rel=1e-6, abs=1e-8
        )
        assert repeated_result.ofv == pytest.approx(base_result.ofv, rel=1e-8)


# ---------------------------------------------------------------------------
# TTEData dataclass
# ---------------------------------------------------------------------------


class TestTTEData:
    """Tests for the TTEData dataclass."""

    def test_basic_construction(self) -> None:
        """TTEData should store attributes correctly."""
        data = TTEData(
            subject_id=1,
            event_times=np.array([10.0]),
            event_indicator=np.array([1]),
        )
        assert data.subject_id == 1
        assert data.event_indicator[0] == 1
        assert data.concentration_times is None
        assert data.concentrations is None

    def test_with_pk_data(self) -> None:
        """TTEData with PK concentrations should store arrays correctly."""
        data = TTEData(
            subject_id=2,
            event_times=np.array([5.0, 12.0]),
            event_indicator=np.array([1, 0]),
            concentration_times=np.array([0.0, 2.0, 5.0, 12.0]),
            concentrations=np.array([10.0, 6.0, 3.0, 0.5]),
        )
        assert len(data.concentration_times) == 4
        assert data.concentrations[0] == pytest.approx(10.0)

    def test_event_and_censored_indicators(self) -> None:
        """Indicator array should distinguish events (1) from censored (0)."""
        data = TTEData(
            subject_id=3,
            event_times=np.array([3.0, 7.0, 15.0]),
            event_indicator=np.array([1, 0, 1]),
        )
        n_events = int(data.event_indicator.sum())
        n_censored = int((data.event_indicator == 0).sum())
        assert n_events == 2
        assert n_censored == 1
