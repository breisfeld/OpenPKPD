"""
Delay Differential Equation (DDE) solver for PK/PD models.

Supports systems of the form:
    dA(t)/dt = f(t, A(t), A(t - τ), pk_params)

where τ is a (possibly parameter-dependent) lag/delay read from pk_params
under the key ``"TAU"`` or ``"DELAY"``.

The history function is available to the $DES callable via
``pk_params["_AHISTORY"]``, which is a callable ``history(t) -> list[float]``
returning compartment amounts at time *t*.  For *t* before the first dose the
history returns the initial-condition vector (all zeros by default, or values
from ``pk_params`` prefixed ``"A0_"``).

Integration
-----------
A dense-output piecewise RK45 integration stores the solution history in a
list of scipy ``OdeSolution`` objects.  Each call to ``history(t)`` queries the
correct dense segment.

Fallback
--------
If ``ddeint`` is installed it is *not* used — the built-in implementation is
preferred because it handles dose events natively.

Usage (via normal NONMEM-compatible $DES code)
----------------------------------------------
::

    # In $PK:  TAU = THETA(4)
    # In $DES (Python):
    #   history = PK_PARAMS["_AHISTORY"]
    #   A_lag   = history(T - PK_PARAMS["TAU"])
    #   DADT(1) = -KA * A(1) + ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.pk.ode.advan6 import (
    _apply_dose_event,
    _get_volume,
    _prepare_doses,
)
from openpkpd.utils.errors import PKError


class DDESubroutine(PKSubroutine):
    """
    Generic DDE solver for PK/PD models with constant or parameter-dependent
    delays.

    The delay value (τ) is read from ``pk_params`` as ``"TAU"`` or ``"DELAY"``.
    If neither is present, the solver degenerates to a plain ODE (τ = 0).

    The history function is injected into ``pk_params`` under the key
    ``"_AHISTORY"`` before each call to ``des_callable``, allowing the user's
    ``$DES`` code to look up past compartment states.

    Parameters
    ----------
    n_compartments : int
        Number of ODE compartments (default 10).
    rtol, atol : float
        Tolerances for the underlying ``solve_ivp`` (RK45).
    method : str
        scipy integration method (default ``"RK45"``).
    """

    advan: int = 16  # Analogous to NONMEM ADVAN16/17 (DDE extension)
    n_compartments: int = 10
    output_compartment: int = 1

    def __init__(
        self,
        n_compartments: int = 10,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        method: str = "RK45",
    ) -> None:
        self.n_compartments = n_compartments
        self.rtol = rtol
        self.atol = atol
        self.method = method

    # ------------------------------------------------------------------
    # PKSubroutine interface
    # ------------------------------------------------------------------

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        """
        Integrate the DDE system for a single subject.

        The ``des_callable`` is called as::

            dadt = des_callable(t, a_list, pk_params_with_history, [], [])

        where ``pk_params_with_history`` is *pk_params* augmented with the
        key ``"_AHISTORY"``.

        Parameters
        ----------
        pk_params :
            PK parameter dict (e.g., ``{"CL": 3.0, "V": 10.0, "TAU": 2.0}``).
        dose_events :
            Dose events for this subject (bolus or infusion).
        obs_times :
            Observation times (unsorted is OK).
        pk_callable :
            Unused; included for interface compatibility.
        des_callable :
            REQUIRED compiled ``$DES`` callable with signature
            ``(t, A_list, pk_params, theta, eta) -> dAdt_list``.

        Returns
        -------
        PKSolution

        Raises
        ------
        PKError
            If ``des_callable`` is None or ODE integration fails.
        """
        if des_callable is None:
            raise PKError(
                "DDESubroutine requires a compiled $DES callable. "
                "Provide des_callable=... in solve()."
            )

        if len(obs_times) == 0:
            empty = np.zeros((0, self.n_compartments))
            return PKSolution(times=obs_times.copy(), amounts=empty, ipred=np.zeros(0))

        delay = float(pk_params.get("TAU", pk_params.get("DELAY", 0.0)))

        # Initial compartment values from pk_params["A0_1"] etc.
        y0 = np.zeros(self.n_compartments)
        for i in range(1, self.n_compartments + 1):
            a0_val = pk_params.get(f"A0_{i}")
            if a0_val is not None:
                y0[i - 1] = float(a0_val)

        v = _get_volume(pk_params)
        out_cmt_idx = self.output_compartment - 1

        # Bioavailability and lag times
        f_factors: dict[int, float] = {}
        alag_map: dict[int, float] = {}
        for i in range(1, self.n_compartments + 1):
            fval = pk_params.get(f"F{i}")
            if fval is not None:
                f_factors[i] = float(fval)
            aval = pk_params.get(f"ALAG{i}")
            if aval is not None:
                alag_map[i] = float(aval)

        active_doses = _prepare_doses(dose_events, f_factors, alag_map)

        t_start = 0.0
        t_end = float(np.max(obs_times))

        # Build breakpoints. For delayed systems, method-of-steps requires each
        # integration segment to be no longer than the delay so that history
        # lookups always fall in already-completed segments.
        event_times: list[float] = [t_start]
        for d in active_doses:
            event_times.append(float(d.time))
            if d.is_infusion:
                event_times.append(float(d.infusion_end_time))
        event_times.append(t_end)
        if delay > 1e-14:
            step_time = t_start + delay
            while step_time < t_end - 1e-12:
                event_times.append(step_time)
                step_time += delay
        all_breakpoints = sorted({t for t in event_times if t_start <= t <= t_end + 1e-12})

        # History storage: list of (t_arr, y_arr) pairs for past intervals
        # For each completed segment we store dense (t, y) so we can interpolate.
        history_segments: list[tuple[np.ndarray, np.ndarray]] = []

        # Observation output array
        obs_sorted_idx = np.argsort(obs_times)
        obs_sorted = obs_times[obs_sorted_idx]
        y_out = np.zeros((len(obs_times), self.n_compartments))
        obs_queue = list(zip(obs_sorted, obs_sorted_idx, strict=False))
        obs_ptr = 0

        active_infusions: dict[int, tuple[float, float]] = {}
        y_current = y0.copy()

        # Apply doses at t=0
        for d in active_doses:
            if abs(d.time - t_start) < 1e-12:
                _apply_dose_event(d, y_current, active_infusions)

        initial_history_state = y_current.copy()

        def _history(t_query: float) -> np.ndarray:
            """Return compartment amounts at time t_query from stored history."""
            if t_query < t_start - 1e-12:
                return y0.copy()
            if t_query <= t_start + 1e-12:
                return initial_history_state.copy()
            for t_seg, y_seg in history_segments:
                if t_seg[0] <= t_query <= t_seg[-1] + 1e-12:
                    if len(t_seg) == 1:
                        return y_seg[:, 0].copy()
                    # Linear interpolation along dense grid
                    interp = interp1d(
                        t_seg,
                        y_seg,
                        axis=1,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(y_seg[:, 0], y_seg[:, -1]),
                    )
                    return np.asarray(interp(t_query))
            # Before first segment or after last — return the start state or last state.
            if history_segments:
                last_t, last_y = history_segments[-1]
                if t_query > last_t[-1]:
                    return last_y[:, -1].copy()
            return initial_history_state.copy()

        # Augment pk_params with history accessor
        pk_params_aug: dict[str, Any] = dict(pk_params)
        pk_params_aug["_AHISTORY"] = _history
        pk_params_aug["_DELAY"] = delay

        prev_t = t_start

        for bp in all_breakpoints:
            if bp <= prev_t + 1e-14:
                continue

            t0_interval = prev_t
            t1_interval = bp

            # Collect obs times in this interval
            eval_times_in_interval: list[float] = []
            eval_indices: list[int] = []
            while obs_ptr < len(obs_queue):
                t_obs, orig_idx = obs_queue[obs_ptr]
                if t_obs <= t1_interval + 1e-12 and t_obs > t0_interval - 1e-12:
                    if t_obs > t0_interval + 1e-14:
                        eval_times_in_interval.append(float(t_obs))
                        eval_indices.append(orig_idx)
                    obs_ptr += 1
                elif t_obs <= t0_interval + 1e-12:
                    y_out[orig_idx, :] = y_current
                    obs_ptr += 1
                else:
                    break

            if t1_interval > t0_interval + 1e-14:
                infusion_snap = dict(active_infusions)

                def _rhs(t: float, y: np.ndarray, _snap: dict = infusion_snap) -> np.ndarray:
                    a_list = list(y)
                    try:
                        dadt = des_callable(t, a_list, pk_params_aug, [], [])
                    except Exception as exc:
                        raise PKError(f"$DES evaluation failed at t={t:.6g}: {exc}") from exc
                    dydt = np.array(dadt, dtype=float)
                    for cmt_idx, (rate, end_t) in _snap.items():
                        if t <= end_t + 1e-14 and 0 <= cmt_idx < len(dydt):
                            dydt[cmt_idx] += rate
                    return dydt

                # Dense t_eval for history storage
                n_dense = max(20, int((t1_interval - t0_interval) / max(delay, 0.1) * 10))
                t_dense = np.linspace(t0_interval, t1_interval, n_dense)
                if eval_times_in_interval:
                    t_eval = np.union1d(t_dense, np.array(eval_times_in_interval))
                else:
                    t_eval = t_dense

                try:
                    sol = solve_ivp(
                        _rhs,
                        (t0_interval, t1_interval),
                        y_current.copy(),
                        method=self.method,
                        t_eval=t_eval,
                        rtol=self.rtol,
                        atol=self.atol,
                        dense_output=False,
                    )
                    if not sol.success:
                        raise PKError(f"DDE solver failed: {sol.message}")

                    # Store history segment
                    history_segments.append((sol.t.copy(), sol.y.copy()))

                    # Record outputs at requested obs times
                    for orig_idx in eval_indices:
                        t_obs_f = obs_times[orig_idx]
                        # Find closest index in sol.t
                        idx_closest = int(np.argmin(np.abs(sol.t - t_obs_f)))
                        y_out[orig_idx, :] = sol.y[:, idx_closest]

                    y_current = sol.y[:, -1].copy()

                except Exception as exc:
                    if not isinstance(exc, PKError):
                        raise PKError(f"DDESubroutine integration failed: {exc}") from exc
                    raise

            # Apply dose events at this breakpoint
            for d in active_doses:
                if abs(d.time - bp) < 1e-12:
                    _apply_dose_event(d, y_current, active_infusions)

            # Remove expired infusions
            active_infusions = {k: v for k, v in active_infusions.items() if v[1] > bp + 1e-14}

            prev_t = bp

        # Fill any remaining obs times
        while obs_ptr < len(obs_queue):
            _, orig_idx = obs_queue[obs_ptr]
            y_out[orig_idx, :] = y_current
            obs_ptr += 1

        y_out = np.maximum(y_out, 0.0)
        ipred = y_out[:, out_cmt_idx] / v

        return PKSolution(times=obs_times.copy(), amounts=y_out, ipred=ipred)
