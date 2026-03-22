"""Direct exact tests for the ODE solver helpers."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.data.event_processor import DoseEvent
from openpkpd.math.ode_solver import solve_ode_piecewise, solve_ode_scipy


def _decay_rhs(k: float):
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([-k * y[0]])

    return rhs


def _piecewise_decay_rhs(k: float):
    def rhs(t: float, y: np.ndarray, infusion_rates: np.ndarray) -> np.ndarray:
        return np.array([-k * y[0] + infusion_rates[0]])

    return rhs


@pytest.mark.unit
def test_solve_ode_scipy_matches_exact_exponential_decay():
    times = np.array([0.0, 1.0, 2.0, 4.0])
    _, y = solve_ode_scipy(
        _decay_rhs(0.2),
        (0.0, 4.0),
        np.array([100.0]),
        t_eval=times,
        method="BDF",
        rtol=1e-10,
        atol=1e-12,
    )

    expected = 100.0 * np.exp(-0.2 * times)
    np.testing.assert_allclose(y[0], expected, rtol=1e-7, atol=1e-9)


@pytest.mark.unit
def test_solve_ode_piecewise_records_state_at_observation_time_exactly():
    obs_times = np.array([0.0, 1.0, 2.0])
    amounts = solve_ode_piecewise(
        _piecewise_decay_rhs(0.2),
        [DoseEvent(time=0.0, amount=100.0, compartment=1)],
        obs_times,
        np.array([0.0]),
        method="BDF",
        rtol=1e-10,
        atol=1e-12,
    )

    expected = 100.0 * np.exp(-0.2 * obs_times)
    np.testing.assert_allclose(amounts[:, 0], expected, rtol=1e-6, atol=1e-8)


@pytest.mark.unit
def test_solve_ode_piecewise_applies_reset_between_observations():
    obs_times = np.array([1.0, 2.0])
    amounts = solve_ode_piecewise(
        _piecewise_decay_rhs(0.0),
        [
            DoseEvent(time=0.0, amount=100.0, compartment=1),
            DoseEvent(time=1.5, amount=0.0, compartment=1, reset=True),
        ],
        obs_times,
        np.array([0.0]),
        method="BDF",
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(amounts[:, 0], np.array([100.0, 0.0]), atol=1e-8)


@pytest.mark.unit
def test_solve_ode_piecewise_handles_empty_observation_grid():
    amounts = solve_ode_piecewise(
        _piecewise_decay_rhs(0.1),
        [DoseEvent(time=0.0, amount=100.0, compartment=1)],
        np.array([]),
        np.array([0.0]),
    )

    assert amounts.shape == (0, 1)
