"""
ADVAN6 — Generic nonstiff ODE model via scipy solve_ivp (RK45).

The user provides a $DES block that defines dA(n)/dt for n compartments.
The ODE is integrated piecewise, restarting at each dose event to handle
bolus injections (instantaneous state changes) and infusions (rate additions
during the infusion window).

Observation times that fall before the first dose return IPRED = 0.

IPRED is computed as A[output_cmt - 1] / V, where V is looked up from
pk_params as 'V', 'V1', 'V2', or 1.0 (if not specified).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.errors import PKError


def _get_volume(pk_params: dict[str, float]) -> float:
    """
    Extract the output compartment volume from pk_params.

    Tries V, V1, V2 in order; falls back to 1.0 if none found.
    """
    for key in ("V", "V1", "V2"):
        v = pk_params.get(key)
        if v is not None and v > 0:
            return float(v)
    return 1.0


class ADVAN6(PKSubroutine):
    """
    Generic nonstiff ODE model (ADVAN6).

    Integrates a user-supplied $DES right-hand side using scipy solve_ivp
    with the RK45 method. The ODE is solved piecewise, restarting at each
    dose event to apply instantaneous state changes (bolus) or to update
    the active infusion rate (zero-order infusion).

    Parameters:
        n_compartments: Number of ODE compartments (default 10; set dynamically
                        if the user's model specifies fewer).
        rtol:           Relative tolerance for the ODE solver (default 1e-6).
        atol:           Absolute tolerance for the ODE solver (default 1e-8).
        method:         scipy solve_ivp method (default 'RK45').
    """

    n_compartments: int = 10
    advan: int = 6
    output_compartment: int = 1

    def __init__(
        self,
        n_compartments: int = 10,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        method: str = "RK45",
    ) -> None:
        """
        Initialize the ODE solver settings.

        Args:
            n_compartments: Number of ODE compartments.
            rtol:           Relative tolerance.
            atol:           Absolute tolerance.
            method:         scipy solve_ivp integration method.
        """
        self.n_compartments = n_compartments
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
        covariate_fn: Callable | None = None,
        covariate_change_times: list[float] | None = None,
    ) -> PKSolution:
        """
        Solve the ODE system for a single subject.

        The des_callable must be provided and will be called as:
            dadt = des_callable(t, a_list, pk_params, theta=[], eta=[])

        The ODE is integrated piecewise between consecutive event times.
        At each dose event:
          - Bolus: adds dose.amount to a[dose.compartment - 1].
          - Infusion: adds dose.rate to the derivative of a[dose.compartment - 1]
            during [dose.time, dose.infusion_end_time].

        Time-varying covariates:
          When ``covariate_fn`` and ``covariate_change_times`` are provided,
          covariate change times are added as extra breakpoints and ``pk_params``
          is updated (via ``covariate_fn``) at the start of each new interval.
          This implements piecewise-constant (LOCF) covariate handling within
          the ODE integration.

        Args:
            pk_params:               Dict of model parameters (already evaluated from $PK).
            dose_events:             List of DoseEvent objects for this subject.
            obs_times:               1-D array of times at which to record IPRED.
            pk_callable:             Unused (params already evaluated).
            des_callable:            REQUIRED compiled $DES callable.
            covariate_fn:            Optional callable ``(t) -> pk_params_dict`` that
                                     returns updated (TRANS-applied) PK parameters at
                                     time ``t`` using LOCF covariates.  When provided,
                                     ``pk_params`` is refreshed at each covariate
                                     change breakpoint.
            covariate_change_times:  Times at which covariates change; added to the
                                     piecewise-integration breakpoints.

        Returns:
            PKSolution with amounts shape (n_times, n_compartments) and IPRED.

        Raises:
            PKError: If des_callable is None or ODE integration fails.
        """
        if des_callable is None:
            raise PKError(
                "ADVAN6 requires a compiled $DES callable. "
                "Provide des_callable=... in solve() call."
            )

        if len(obs_times) == 0:
            empty = np.zeros((0, self.n_compartments))
            return PKSolution(
                times=obs_times.copy(),
                amounts=empty,
                ipred=np.zeros(0),
            )

        v = _get_volume(pk_params)
        out_cmt_idx = int(self.get_output_compartment(pk_params)) - 1

        # Bioavailability factors from pk_params (F1, F2, etc.)
        f_factors: dict[int, float] = {}
        for i in range(1, self.n_compartments + 1):
            f_key = f"F{i}"
            fval = pk_params.get(f_key)
            if fval is not None:
                f_factors[i] = float(fval)

        # Lag times from pk_params (ALAG1, ALAG2, etc.)
        alag: dict[int, float] = {}
        for i in range(1, self.n_compartments + 1):
            alag_key = f"ALAG{i}"
            alag_val = pk_params.get(alag_key)
            if alag_val is not None:
                alag[i] = float(alag_val)

        # Filter reset events and apply bioavailability/lag
        active_doses = _prepare_doses(dose_events, f_factors, alag)

        # Build sorted list of breakpoints: dose times + observation times + infusion end times
        t_start = 0.0
        t_end = float(np.max(obs_times))

        # Collect all event times
        event_times: list[float] = [t_start]
        for d in active_doses:
            event_times.append(float(d.time))
            if d.is_infusion:
                event_times.append(float(d.infusion_end_time))
        event_times.append(t_end)

        # Add covariate change times as extra breakpoints (time-varying covariates)
        if covariate_change_times:
            event_times.extend(float(tc) for tc in covariate_change_times)

        # Build a set of times at which pk_params should be refreshed
        cov_change_set: set[float] = set()
        if covariate_fn is not None and covariate_change_times:
            cov_change_set = {float(tc) for tc in covariate_change_times}

        # Build piecewise integration grid
        all_breakpoints = sorted(set(event_times))
        # Filter to valid range
        all_breakpoints = [t for t in all_breakpoints if t_start <= t <= t_end + 1e-12]

        # Initial state
        y_current = np.zeros(self.n_compartments)

        # Storage for results at obs_times
        obs_times_sorted_idx = np.argsort(obs_times)
        obs_times_sorted = obs_times[obs_times_sorted_idx]

        y_out = np.zeros((len(obs_times), self.n_compartments))

        # Map obs time → output index (use sorted obs times)
        obs_queue = list(zip(obs_times_sorted, obs_times_sorted_idx, strict=False))
        obs_ptr = 0

        # Track active infusions: {cmt_idx: (rate, end_time)}
        active_infusions: dict[int, tuple[float, float]] = {}

        # Iterate over piecewise intervals
        prev_t = t_start

        # Apply any doses at t=0 before starting integration
        for d in active_doses:
            if abs(d.time - t_start) < 1e-12:
                _apply_dose_event(d, y_current, active_infusions)

        for bp in all_breakpoints:
            if bp <= prev_t + 1e-14:
                continue

            t_interval_start = prev_t
            t_interval_end = bp

            # Find obs times in this interval (strictly after t_interval_start, up to bp)
            eval_times_in_interval: list[float] = []
            eval_indices: list[int] = []

            while obs_ptr < len(obs_queue):
                t_obs, orig_idx = obs_queue[obs_ptr]
                if t_obs <= t_interval_end + 1e-12 and t_obs > t_interval_start - 1e-12:
                    if t_obs > t_interval_start + 1e-14:
                        eval_times_in_interval.append(float(t_obs))
                        eval_indices.append(orig_idx)
                    obs_ptr += 1
                elif t_obs <= t_interval_start + 1e-12:
                    # Observation at or before current start: record current state
                    y_out[orig_idx, :] = y_current
                    obs_ptr += 1
                else:
                    break

            if t_interval_end > t_interval_start + 1e-14:
                # Build RHS for this interval (with active infusions)
                rhs = _make_rhs(des_callable, pk_params, active_infusions, self.n_compartments)

                if eval_times_in_interval:
                    t_eval = np.array(eval_times_in_interval)
                    try:
                        unique_t_eval, inverse = np.unique(t_eval, return_inverse=True)
                        sol = solve_ivp(
                            rhs,
                            (t_interval_start, t_interval_end),
                            y_current.copy(),
                            method=self.method,
                            t_eval=unique_t_eval,
                            rtol=self.rtol,
                            atol=self.atol,
                            dense_output=False,
                        )
                        if not sol.success:
                            raise PKError(f"ODE solver failed: {sol.message}")
                        # Store results at eval times
                        for k, orig_idx in enumerate(eval_indices):
                            y_out[orig_idx, :] = sol.y[:, inverse[k]]
                        y_current = sol.y[:, -1].copy()
                    except Exception as exc:
                        if not isinstance(exc, PKError):
                            raise PKError(f"ADVAN6 ODE integration failed: {exc}") from exc
                        raise
                else:
                    # Integrate to end of interval (no obs to record)
                    try:
                        sol = solve_ivp(
                            rhs,
                            (t_interval_start, t_interval_end),
                            y_current.copy(),
                            method=self.method,
                            t_eval=np.array([t_interval_end]),
                            rtol=self.rtol,
                            atol=self.atol,
                        )
                        if not sol.success:
                            raise PKError(f"ODE solver failed: {sol.message}")
                        y_current = sol.y[:, -1].copy()
                    except Exception as exc:
                        if not isinstance(exc, PKError):
                            raise PKError(f"ADVAN6 ODE integration failed: {exc}") from exc
                        raise

            # Apply dose events and infusion state changes at breakpoint bp
            for d in active_doses:
                if abs(d.time - bp) < 1e-12:
                    _apply_dose_event(d, y_current, active_infusions)

            # Remove expired infusions
            active_infusions = {
                cmt_idx: (r, end_t)
                for cmt_idx, (r, end_t) in active_infusions.items()
                if end_t > bp + 1e-14
            }

            # Refresh pk_params at covariate change breakpoints (time-varying covariates)
            if covariate_fn is not None and any(abs(bp - tc) < 1e-12 for tc in cov_change_set):
                import contextlib

                with contextlib.suppress(Exception):
                    pk_params = covariate_fn(bp)

            prev_t = bp

        # Any remaining obs times after the last breakpoint get current state
        while obs_ptr < len(obs_queue):
            _, orig_idx = obs_queue[obs_ptr]
            y_out[orig_idx, :] = y_current
            obs_ptr += 1

        # Ensure non-negative amounts (numerical noise)
        y_out = np.maximum(y_out, 0.0)

        # IPRED = A[output_cmt - 1] / V
        ipred = y_out[:, out_cmt_idx] / v

        return PKSolution(
            times=obs_times.copy(),
            amounts=y_out,
            ipred=ipred,
        )


def _prepare_doses(
    dose_events: list[DoseEvent],
    f_factors: dict[int, float],
    alag: dict[int, float],
) -> list[DoseEvent]:
    """
    Filter reset events, apply bioavailability and lag time adjustments.

    Args:
        dose_events: Raw list of DoseEvent objects.
        f_factors:   Bioavailability per compartment {cmt: F}.
        alag:        Lag times per compartment {cmt: ALAG}.

    Returns:
        Adjusted list of non-reset DoseEvent objects.
    """
    result: list[DoseEvent] = []
    for d in dose_events:
        if d.reset:
            continue
        cmt = d.compartment
        f = f_factors.get(cmt, 1.0)
        lag = alag.get(cmt, 0.0)

        adjusted = DoseEvent(
            time=d.time + lag,
            amount=d.amount * f,
            rate=d.rate,
            duration=d.duration,
            compartment=cmt,
            ss=d.ss,
            ii=d.ii,
            reset=False,
        )
        result.append(adjusted)
    return sorted(result, key=lambda e: e.time)


def _apply_dose_event(
    d: DoseEvent,
    y: np.ndarray,
    active_infusions: dict[int, tuple[float, float]],
) -> None:
    """
    Apply a single dose event to the current state vector in-place.

    For bolus: adds amount to y[cmt-1].
    For infusion: registers the rate in active_infusions.

    Args:
        d:                The dose event to apply.
        y:                Current compartment amounts (modified in-place).
        active_infusions: Dict of {cmt_idx: (rate, end_time)} (modified in-place).
    """
    cmt_idx = d.compartment - 1
    if not (0 <= cmt_idx < len(y)):
        return

    if d.is_bolus:
        y[cmt_idx] += d.amount
    elif d.is_infusion:
        end_t = d.infusion_end_time
        # Accumulate if multiple infusions into same compartment
        existing_rate = active_infusions.get(cmt_idx, (0.0, end_t))[0]
        active_infusions[cmt_idx] = (existing_rate + d.rate, end_t)


def _make_rhs(
    des_callable: Callable,
    pk_params: dict[str, float],
    active_infusions: dict[int, tuple[float, float]],
    n_compartments: int,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Build the ODE right-hand side function for the current integration interval.

    The RHS calls des_callable(t, a_list, pk_params, theta=[], eta=[]) to get
    the user's dA/dt values, then adds any active infusion rates.

    Args:
        des_callable:     Compiled $DES callable.
        pk_params:        Current PK parameters.
        active_infusions: Dict {cmt_idx: (rate, end_time)} for current interval.
        n_compartments:   Number of ODE compartments.

    Returns:
        A callable rhs(t, y) -> np.ndarray for use with solve_ivp.
    """
    # Snapshot of infusions at the time the RHS is created
    infusion_snapshot = dict(active_infusions)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        a_list = list(y)
        try:
            dadt = des_callable(t, a_list, pk_params, [], [])
        except Exception as exc:
            raise PKError(f"$DES evaluation failed at t={t:.6g}: {exc}") from exc

        dydt = np.array(dadt, dtype=float)

        # Add active infusion contributions
        for cmt_idx, (rate, end_t) in infusion_snapshot.items():
            if t <= end_t + 1e-14 and 0 <= cmt_idx < len(dydt):
                dydt[cmt_idx] += rate

        return dydt

    return rhs
