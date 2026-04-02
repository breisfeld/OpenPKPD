"""Unified SciPy-based ODE solver wrapper for OpenPKPD."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp


def solve_ode_scipy(
    rhs: Callable,
    t_span: tuple[float, float],
    y0: np.ndarray,
    t_eval: np.ndarray | None = None,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    dense_output: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve an ODE system using scipy.

    Args:
        rhs:     Right-hand side function dy/dt = rhs(t, y).
        t_span:  (t_start, t_end).
        y0:      Initial condition array.
        t_eval:  Times at which to store solution.
        method:  Integration method ('BDF', 'Radau', 'RK45', 'DOP853').
        rtol:    Relative tolerance.
        atol:    Absolute tolerance.

    Returns:
        (t_out, y_out) where y_out has shape (len(y0), len(t_out)).
    """
    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
        **kwargs,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol.t, sol.y


def solve_ode_piecewise(
    rhs: Callable,
    dose_events: list,
    obs_times: np.ndarray,
    y0: np.ndarray,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> np.ndarray:
    """
    Solve ODE system with piecewise dosing events.

    The ODE is solved in segments between dosing events, with
    instantaneous state changes applied at dose times.

    Args:
        rhs:         dy/dt = rhs(t, y, dose_rate) callable.
        dose_events: List of DoseEvent objects sorted by time.
        obs_times:   Times at which to record the solution.
        y0:          Initial compartment amounts.
        method:      ODE integration method.
        rtol, atol:  Tolerances.

    Returns:
        y_out array of shape (len(obs_times), len(y0)).
    """
    from openpkpd.data.event_processor import DoseEvent

    if len(obs_times) == 0:
        return np.zeros((0, len(y0)), dtype=float)

    # Build event-time grid: union of all event times and obs_times.
    # Reset events must be included so their discontinuities are applied even
    # when they occur between observations.
    dose_times = [e.time for e in dose_events]
    all_times = sorted(set(dose_times + list(obs_times)))

    y = y0.copy().astype(float)

    # Map obs_times → all output indices at that time. This preserves
    # duplicate observation rows (for example same-time multi-endpoint data)
    # and avoids collapsing them to a single output slot.
    obs_indices: dict[float, list[int]] = {}
    for i, time in enumerate(np.asarray(obs_times, dtype=float)):
        obs_indices.setdefault(float(time), []).append(i)
    y_out = np.zeros((len(obs_times), len(y)))

    # Active infusion state: {compartment_idx: [(rate, end_time), ...]}
    active_infusions: dict[int, list[tuple[float, float]]] = {}

    def make_rhs_with_infusions(
        infusions: dict[int, list[tuple[float, float]]],
    ) -> Callable:
        def _rhs(t: float, y: np.ndarray) -> np.ndarray:
            # Add active infusion rates
            infusion_rates = np.zeros(len(y))
            for cmt_idx, entries in infusions.items():
                for rate, end_t in entries:
                    if t <= end_t:
                        infusion_rates[cmt_idx] += rate
            return rhs(t, y, infusion_rates)

        return _rhs

    def _record_state(t: float, state: np.ndarray) -> None:
        for idx in obs_indices.get(float(t), []):
            y_out[idx, :] = state

    event_iter = iter(dose_events)
    next_event: DoseEvent | None = next(event_iter, None)

    i = 0
    while i < len(all_times) - 1:
        t_start = all_times[i]
        t_end = all_times[i + 1]

        # Apply events at t_start
        while next_event is not None and abs(next_event.time - t_start) < 1e-12:
            ev = next_event
            next_event = next(event_iter, None)
            cmt_idx = ev.compartment - 1

            if ev.reset:
                y[:] = 0.0
                active_infusions.clear()
            elif ev.is_bolus:
                if 0 <= cmt_idx < len(y):
                    y[cmt_idx] += ev.amount
            elif ev.is_infusion:
                active_infusions.setdefault(cmt_idx, []).append((ev.rate, ev.infusion_end_time))

        # Remove expired infusions
        active_infusions = {
            k: [(rate, end_t) for rate, end_t in entries if end_t > t_start]
            for k, entries in active_infusions.items()
        }
        active_infusions = {k: entries for k, entries in active_infusions.items() if entries}

        # Record at t_start after applying discontinuous events, before solving
        # forward to the next time point.
        _record_state(t_start, y)

        if t_end > t_start + 1e-14:
            _rhs = make_rhs_with_infusions(active_infusions)
            eval_pts = sorted({float(t) for t in obs_times if t_start < t <= t_end})
            if eval_pts:
                _, y_seg = solve_ode_scipy(
                    _rhs,
                    (t_start, t_end),
                    y,
                    t_eval=np.array(eval_pts),
                    method=method,
                    rtol=rtol,
                    atol=atol,
                )
                for j, t_pt in enumerate(eval_pts):
                    _record_state(t_pt, y_seg[:, j])
            else:
                _, y_seg = solve_ode_scipy(
                    _rhs,
                    (t_start, t_end),
                    y,
                    t_eval=np.array([t_end]),
                    method=method,
                    rtol=rtol,
                    atol=atol,
                )
            y = y_seg[:, -1]

        i += 1

    # Apply events at the final grid time and then record the terminal state.
    if all_times:
        final_time = all_times[-1]
        while next_event is not None and abs(next_event.time - final_time) < 1e-12:
            ev = next_event
            next_event = next(event_iter, None)
            cmt_idx = ev.compartment - 1

            if ev.reset:
                y[:] = 0.0
                active_infusions.clear()
            elif ev.is_bolus:
                if 0 <= cmt_idx < len(y):
                    y[cmt_idx] += ev.amount
            elif ev.is_infusion:
                active_infusions.setdefault(cmt_idx, []).append((ev.rate, ev.infusion_end_time))

        _record_state(final_time, y)

    return y_out
