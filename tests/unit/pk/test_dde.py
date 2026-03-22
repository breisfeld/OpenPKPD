"""Unit tests for DDESubroutine."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.ode.dde import DDESubroutine


def _bolus(time=0.0, amount=100.0, compartment=1):
    return DoseEvent(time=time, amount=amount, rate=0.0, duration=0.0, compartment=compartment)


def _one_cmt_des(t, A, pk_params, theta, eta):
    """Simple one-compartment DES: dA/dt = -CL/V * A (no delay)."""
    cl = pk_params.get("CL", 1.0)
    v = pk_params.get("V", 10.0)
    ke = cl / v
    return [-ke * A[0]]


def _one_cmt_dde(t, A, pk_params, theta, eta):
    """One-compartment DDE: dA/dt = -CL/V * A(t - tau) (uses history)."""
    cl = pk_params.get("CL", 1.0)
    v = pk_params.get("V", 10.0)
    tau = pk_params.get("TAU", 0.0)
    history = pk_params.get("_AHISTORY", None)
    if history is not None and tau > 0:
        A_lag = history(t - tau)
        ke = cl / v
        return [-ke * A_lag[0]]
    ke = cl / v
    return [-ke * A[0]]


def _expected_delayed_elimination_amount(dose: float, ke: float, tau: float, t) -> np.ndarray:
    """Exact solution for the first three method-of-steps intervals used in tests."""
    t_arr = np.asarray(t, dtype=float)
    amount = np.zeros_like(t_arr)

    seg0 = (t_arr >= 0.0) & (t_arr <= tau)
    amount[seg0] = dose

    seg1 = (t_arr > tau) & (t_arr <= 2.0 * tau)
    dt1 = t_arr[seg1] - tau
    amount[seg1] = dose * (1.0 - ke * dt1)

    seg2 = (t_arr > 2.0 * tau) & (t_arr <= 3.0 * tau)
    dt2 = t_arr[seg2] - 2.0 * tau
    amount[seg2] = dose * ((1.0 - ke * tau) - ke * dt2 + 0.5 * ke**2 * dt2**2)

    return amount


class TestDDESubroutineNoDelay:
    """When TAU=0, DDESubroutine should behave like ADVAN6 (plain ODE)."""

    def test_one_cmt_exponential_decay(self):
        solver = DDESubroutine(n_compartments=1)
        pk_params = {"CL": 2.0, "V": 10.0, "TAU": 0.0}
        obs_times = np.array([0.5, 1.0, 2.0, 4.0])
        dose_events = [_bolus(time=0.0, amount=100.0)]

        sol = solver.solve(
            pk_params=pk_params,
            dose_events=dose_events,
            obs_times=obs_times,
            des_callable=_one_cmt_des,
        )

        # Analytical: A(t) = 100 * exp(-CL/V * t); IPRED = A/V
        ke = 2.0 / 10.0
        expected_ipred = 100.0 * np.exp(-ke * obs_times) / 10.0
        np.testing.assert_allclose(sol.ipred, expected_ipred, rtol=1e-4)

    def test_empty_obs_times(self):
        solver = DDESubroutine(n_compartments=1)
        sol = solver.solve(
            pk_params={"CL": 1.0, "V": 5.0},
            dose_events=[_bolus()],
            obs_times=np.array([]),
            des_callable=_one_cmt_des,
        )
        assert sol.ipred.shape == (0,)
        assert sol.amounts.shape == (0, 1)

    def test_raises_without_des_callable(self):
        solver = DDESubroutine(n_compartments=1)
        with pytest.raises(Exception, match="des_callable"):
            solver.solve(
                pk_params={"CL": 1.0, "V": 5.0},
                dose_events=[_bolus()],
                obs_times=np.array([1.0, 2.0]),
                des_callable=None,
            )


class TestDDESubroutineWithDelay:
    """Tests with non-zero delay (uses history function)."""

    def test_history_function_available_in_des(self):
        """Verify that _AHISTORY is injected and callable."""
        history_calls = []

        def _des_that_checks_history(t, A, pk_params, theta, eta):
            if "_AHISTORY" in pk_params:
                hist = pk_params["_AHISTORY"]
                history_calls.append(hist(0.0))  # Always query at t=0
            ke = pk_params.get("CL", 1.0) / pk_params.get("V", 10.0)
            return [-ke * A[0]]

        solver = DDESubroutine(n_compartments=1)
        solver.solve(
            pk_params={"CL": 1.0, "V": 10.0, "TAU": 0.5},
            dose_events=[_bolus(amount=100.0)],
            obs_times=np.array([1.0, 2.0]),
            des_callable=_des_that_checks_history,
        )
        assert len(history_calls) > 0  # History was called at least once

    def test_history_before_dose_returns_zeros(self):
        """History queried at negative times must return initial condition (zeros)."""
        queried = []

        def _des_record_history(t, A, pk_params, theta, eta):
            hist = pk_params.get("_AHISTORY")
            if hist and t < 0.1:
                queried.append(list(hist(-1.0)))  # Pre-dose query
            ke = pk_params.get("CL", 1.0) / pk_params.get("V", 10.0)
            return [-ke * A[0]]

        solver = DDESubroutine(n_compartments=1)
        solver.solve(
            pk_params={"CL": 1.0, "V": 10.0, "TAU": 2.0},
            dose_events=[_bolus(amount=50.0)],
            obs_times=np.array([0.5, 1.0]),
            des_callable=_des_record_history,
        )
        if queried:
            assert queried[0][0] == pytest.approx(0.0)

    def test_amounts_non_negative(self):
        """Compartment amounts should always be >= 0 after solve."""
        solver = DDESubroutine(n_compartments=1)
        sol = solver.solve(
            pk_params={"CL": 1.0, "V": 5.0, "TAU": 0.5},
            dose_events=[_bolus(amount=100.0)],
            obs_times=np.linspace(0.1, 10.0, 20),
            des_callable=_one_cmt_dde,
        )
        assert np.all(sol.amounts >= 0.0)

    def test_multiple_doses(self):
        """Multiple dose events should not raise and produce finite IPRED."""
        solver = DDESubroutine(n_compartments=1)
        dose_events = [
            _bolus(time=0.0, amount=100.0),
            _bolus(time=6.0, amount=100.0),
            _bolus(time=12.0, amount=100.0),
        ]
        sol = solver.solve(
            pk_params={"CL": 1.0, "V": 10.0, "TAU": 1.0},
            dose_events=dose_events,
            obs_times=np.array([3.0, 6.0, 9.0, 12.0, 18.0]),
            des_callable=_one_cmt_dde,
        )
        assert np.all(np.isfinite(sol.ipred))
        assert np.all(sol.ipred >= 0.0)

    def test_output_shape(self):
        solver = DDESubroutine(n_compartments=2)

        def _two_cmt_des(t, A, pk_params, theta, eta):
            k12 = pk_params.get("K12", 0.1)
            k21 = pk_params.get("K21", 0.05)
            ke = pk_params.get("KE", 0.2)
            return [
                -k12 * A[0] + k21 * A[1] - ke * A[0],
                k12 * A[0] - k21 * A[1],
            ]

        obs = np.array([1.0, 2.0, 4.0, 8.0])
        sol = solver.solve(
            pk_params={"K12": 0.1, "K21": 0.05, "KE": 0.2, "V": 5.0, "TAU": 0.0},
            dose_events=[_bolus(amount=100.0, compartment=1)],
            obs_times=obs,
            des_callable=_two_cmt_des,
        )
        assert sol.amounts.shape == (4, 2)
        assert sol.ipred.shape == (4,)
        assert sol.times.shape == (4,)

    def test_delay_longer_than_observation_window_gives_exact_plateau(self):
        """If tau exceeds the window, delayed elimination never starts and the dose stays constant."""
        solver = DDESubroutine(n_compartments=1)
        obs_times = np.array([0.25, 0.5, 1.0, 1.5])

        sol = solver.solve(
            pk_params={"CL": 2.0, "V": 10.0, "TAU": 5.0},
            dose_events=[_bolus(amount=100.0)],
            obs_times=obs_times,
            des_callable=_one_cmt_dde,
        )

        np.testing.assert_allclose(sol.amounts[:, 0], 100.0, atol=1e-8)
        np.testing.assert_allclose(sol.ipred, 10.0, atol=1e-8)

    def test_delay_equation_matches_piecewise_exact_solution_over_first_segments(self):
        """The lagged-elimination toy model has an exact method-of-steps solution."""
        solver = DDESubroutine(n_compartments=1)
        dose = 80.0
        cl = 1.5
        v = 10.0
        tau = 1.0
        ke = cl / v
        obs_times = np.array([0.25, 1.0, 1.5, 2.0, 2.5, 2.75])

        sol = solver.solve(
            pk_params={"CL": cl, "V": v, "TAU": tau},
            dose_events=[_bolus(amount=dose)],
            obs_times=obs_times,
            des_callable=_one_cmt_dde,
        )

        expected_amounts = _expected_delayed_elimination_amount(dose, ke, tau, obs_times)
        np.testing.assert_allclose(sol.amounts[:, 0], expected_amounts, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(sol.ipred, expected_amounts / v, rtol=2e-4, atol=2e-4)

    def test_delayed_linear_system_obeys_exact_superposition(self):
        """The linear delayed system should superpose dose-by-dose."""
        solver = DDESubroutine(n_compartments=1)
        pk_params = {"CL": 1.2, "V": 8.0, "TAU": 1.5}
        obs_times = np.array([0.5, 1.0, 2.0, 2.5, 4.0, 6.0])
        doses = [
            _bolus(time=0.0, amount=40.0),
            _bolus(time=2.0, amount=25.0),
        ]

        sol_multi = solver.solve(pk_params, doses, obs_times, des_callable=_one_cmt_dde)
        summed = sum(
            solver.solve(pk_params, [dose], obs_times, des_callable=_one_cmt_dde).amounts[:, 0]
            for dose in doses
        )

        np.testing.assert_allclose(sol_multi.amounts[:, 0], summed, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(sol_multi.ipred, summed / pk_params["V"], rtol=2e-4, atol=2e-4)


class TestDDERegistered:
    """DDESubroutine is accessible via the standard pk dispatcher."""

    def test_get_advan_16(self):
        from openpkpd.pk import DDESubroutine as DDESub
        from openpkpd.pk import get_advan

        solver = get_advan(16)
        assert isinstance(solver, DDESub)
