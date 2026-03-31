"""Tests for time-varying covariate LOCF accessor on SubjectEvents."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.event_processor import SubjectEvents


def _make_subject_events(cov_data: list[dict]) -> SubjectEvents:
    """Return a minimal SubjectEvents with a covariate_df."""
    cov_df = pd.DataFrame(cov_data)
    return SubjectEvents(
        subject_id=1,
        obs_times=np.array([0.0]),
        obs_dv=np.array([0.0]),
        obs_cmt=np.array([1]),
        obs_mdv=np.array([0]),
        covariate_df=cov_df,
    )


class TestCovariateAt:
    def test_returns_empty_when_no_covariate_df(self):
        se = SubjectEvents(subject_id=1)
        assert se.covariate_at(5.0) == {}

    def test_locf_baseline(self):
        """Before first measurement, returns first row."""
        se = _make_subject_events(
            [
                {"TIME": 1.0, "WT": 70.0},
                {"TIME": 5.0, "WT": 72.0},
            ]
        )
        result = se.covariate_at(0.0)
        assert result["WT"] == pytest.approx(70.0)

    def test_locf_at_measurement_time(self):
        """At the exact measurement time, returns that row."""
        se = _make_subject_events(
            [
                {"TIME": 1.0, "WT": 70.0},
                {"TIME": 5.0, "WT": 72.0},
            ]
        )
        assert se.covariate_at(1.0)["WT"] == pytest.approx(70.0)
        assert se.covariate_at(5.0)["WT"] == pytest.approx(72.0)

    def test_locf_between_measurements(self):
        """Between measurements, returns the most recent value."""
        se = _make_subject_events(
            [
                {"TIME": 0.0, "WT": 70.0},
                {"TIME": 10.0, "WT": 75.0},
                {"TIME": 20.0, "WT": 80.0},
            ]
        )
        assert se.covariate_at(5.0)["WT"] == pytest.approx(70.0)
        assert se.covariate_at(15.0)["WT"] == pytest.approx(75.0)
        assert se.covariate_at(25.0)["WT"] == pytest.approx(80.0)

    def test_multiple_covariates(self):
        se = _make_subject_events(
            [
                {"TIME": 0.0, "WT": 70.0, "CRCL": 90.0},
                {"TIME": 10.0, "WT": 72.0, "CRCL": 85.0},
            ]
        )
        cov = se.covariate_at(5.0)
        assert cov["WT"] == pytest.approx(70.0)
        assert cov["CRCL"] == pytest.approx(90.0)
        cov2 = se.covariate_at(15.0)
        assert cov2["WT"] == pytest.approx(72.0)
        assert cov2["CRCL"] == pytest.approx(85.0)

    def test_after_last_measurement(self):
        """After the last measurement, the last row is carried forward."""
        se = _make_subject_events(
            [
                {"TIME": 0.0, "WT": 70.0},
                {"TIME": 5.0, "WT": 75.0},
            ]
        )
        assert se.covariate_at(100.0)["WT"] == pytest.approx(75.0)

    def test_nan_values_excluded(self):
        """NaN covariate values should not appear in the result dict."""
        se = _make_subject_events(
            [
                {"TIME": 0.0, "WT": 70.0, "CRCL": float("nan")},
            ]
        )
        cov = se.covariate_at(0.0)
        assert "WT" in cov
        assert "CRCL" not in cov

    def test_returns_float_values(self):
        se = _make_subject_events([{"TIME": 0.0, "WT": 70}])
        cov = se.covariate_at(0.0)
        assert isinstance(cov["WT"], float)

    def test_preserves_non_numeric_covariates_for_routing(self):
        se = _make_subject_events([{"TIME": 0.0, "DVID": "cp", "WT": 70.0}])
        cov = se.covariate_at(0.0)
        assert cov["DVID"] == "cp"
        assert cov["WT"] == pytest.approx(70.0)


class TestCovariateChangeTimes:
    def test_returns_empty_when_no_covariate_df(self):
        se = SubjectEvents(subject_id=1)
        assert se.covariate_change_times() == []

    def test_single_row(self):
        se = _make_subject_events([{"TIME": 0.0, "WT": 70.0}])
        assert se.covariate_change_times() == [0.0]

    def test_multiple_rows_sorted(self):
        se = _make_subject_events(
            [
                {"TIME": 5.0, "WT": 72.0},
                {"TIME": 0.0, "WT": 70.0},
                {"TIME": 10.0, "WT": 75.0},
            ]
        )
        times = se.covariate_change_times()
        assert times == [0.0, 5.0, 10.0]

    def test_unique_times(self):
        """Duplicate times should appear only once."""
        se = _make_subject_events(
            [
                {"TIME": 0.0, "WT": 70.0},
                {"TIME": 0.0, "WT": 70.0},
                {"TIME": 5.0, "WT": 72.0},
            ]
        )
        times = se.covariate_change_times()
        assert times == [0.0, 5.0]


class TestAdvan6TimeVaryingCovariates:
    """Integration-level tests: ODE solver respects covariate_fn updates."""

    def test_covariate_fn_invoked_at_change_times(self):
        """Verify that covariate_fn is called when covariate_change_times are provided."""
        from openpkpd.data.event_processor import DoseEvent
        from openpkpd.pk.ode.advan6 import ADVAN6

        call_times: list[float] = []

        def des_callable(t, a, pk_params, theta, eta):
            k = pk_params.get("K", 0.1)
            return [-k * a[0]]

        def covariate_fn(t: float) -> dict:
            call_times.append(t)
            return {"K": 0.1, "V": 10.0}

        advan = ADVAN6(n_compartments=1)
        dose_events = [DoseEvent(time=0.0, amount=100.0, compartment=1)]
        obs_times = np.array([1.0, 5.0, 10.0])

        pk_params = {"K": 0.1, "V": 10.0}
        advan.solve(
            pk_params=pk_params,
            dose_events=dose_events,
            obs_times=obs_times,
            des_callable=des_callable,
            covariate_fn=covariate_fn,
            covariate_change_times=[3.0, 7.0],
        )
        # covariate_fn should have been called at t=3.0 and t=7.0
        assert any(abs(t - 3.0) < 0.01 for t in call_times)
        assert any(abs(t - 7.0) < 0.01 for t in call_times)

    def test_constant_covariates_give_same_result_as_no_covariates(self):
        """When covariate_fn returns unchanged pk_params, results should match
        the standard call with constant parameters."""
        from openpkpd.data.event_processor import DoseEvent
        from openpkpd.pk.ode.advan6 import ADVAN6

        advan = ADVAN6(n_compartments=1)
        dose_events = [DoseEvent(time=0.0, amount=100.0, compartment=1)]
        obs_times = np.linspace(0.5, 10.0, 20)
        pk_params = {"K": 0.15, "V": 12.0}

        def des_callable(t, a, pk, theta, eta):
            return [-pk["K"] * a[0]]

        # Solve without covariate_fn
        sol_plain = advan.solve(pk_params, dose_events, obs_times, des_callable=des_callable)

        # Solve with covariate_fn that always returns same params
        def covariate_fn(t):
            return {"K": 0.15, "V": 12.0}

        sol_with = advan.solve(
            pk_params,
            dose_events,
            obs_times,
            des_callable=des_callable,
            covariate_fn=covariate_fn,
            covariate_change_times=[2.0, 6.0],
        )

        np.testing.assert_allclose(sol_with.ipred, sol_plain.ipred, rtol=1e-4)

    def test_changing_elimination_rate_changes_ipred(self):
        """When K changes at a covariate breakpoint, IPRED should differ from constant-K."""
        from openpkpd.data.event_processor import DoseEvent
        from openpkpd.pk.ode.advan6 import ADVAN6

        advan = ADVAN6(n_compartments=1)
        dose_events = [DoseEvent(time=0.0, amount=100.0, compartment=1)]
        obs_times = np.array([5.0, 10.0])
        pk_params = {"K": 0.1, "V": 10.0}

        def des_callable(t, a, pk, theta, eta):
            return [-pk["K"] * a[0]]

        # Plain solve: K=0.1 throughout
        sol_plain = advan.solve(pk_params, dose_events, obs_times, des_callable=des_callable)

        # Solve with K doubling at t=3
        n_calls = [0]

        def covariate_fn(t: float) -> dict:
            n_calls[0] += 1
            return {"K": 0.2, "V": 10.0}  # faster elimination

        sol_cv = advan.solve(
            pk_params,
            dose_events,
            obs_times,
            des_callable=des_callable,
            covariate_fn=covariate_fn,
            covariate_change_times=[3.0],
        )

        # After the covariate change (K doubles), concentrations should be lower
        assert sol_cv.ipred[0] < sol_plain.ipred[0]  # at t=5
        assert sol_cv.ipred[1] < sol_plain.ipred[1]  # at t=10

    def test_repeated_solve_reuses_cached_schedule_for_identical_event_layout(self, monkeypatch):
        from openpkpd.data.event_processor import DoseEvent
        import openpkpd.pk.ode.advan6 as advan6_mod
        from openpkpd.pk.ode.advan6 import ADVAN6

        advan = ADVAN6(n_compartments=1)
        dose_events = [DoseEvent(time=0.0, amount=100.0, compartment=1)]
        obs_times = np.array([1.0, 5.0, 10.0])
        pk_params = {"K": 0.15, "V": 12.0}

        def des_callable(t, a, pk, theta, eta):
            return [-pk["K"] * a[0]]

        call_count = {"prepare": 0}
        real_prepare = advan6_mod._prepare_doses

        def counted_prepare(*args, **kwargs):
            call_count["prepare"] += 1
            return real_prepare(*args, **kwargs)

        monkeypatch.setattr(advan6_mod, "_prepare_doses", counted_prepare)

        sol_1 = advan.solve(pk_params, dose_events, obs_times, des_callable=des_callable)
        sol_2 = advan.solve(pk_params, dose_events, obs_times, des_callable=des_callable)

        np.testing.assert_allclose(sol_1.ipred, sol_2.ipred)
        assert call_count["prepare"] == 1
