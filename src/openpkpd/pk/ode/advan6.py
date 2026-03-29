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

import warnings
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
        jit: str = "scipy",
        max_steps: int = 100_000,
    ) -> None:
        """
        Initialize the ODE solver settings.

        Args:
            n_compartments: Number of ODE compartments.
            rtol:           Relative tolerance.
            atol:           Absolute tolerance.
            method:         scipy solve_ivp integration method (used when
                            jit='scipy' or as fallback).
            jit:            JIT acceleration tier.  Default is ``'scipy'`` for
                            full backward compatibility.  Opt-in to faster tiers:
                            ``'numpy'``  — pure-NumPy Dormand-Prince RK45 (no new deps).
                            ``'numba'``  — Numba @njit DES + NumPy-RK45; requires numba.
                            ``'llc'``    — Pure-Numba RK45 + @njit DES; both the
                                           integrator and RHS run as native code
                                           with zero Python overhead per step.
                                           Fastest option; requires numba.
                            ``'auto'``   — ``'llc'`` if numba installed, else ``'numpy'``.
            max_steps:      Maximum number of RK45 steps per integration segment
                            for the explicit tiers (numpy/numba/llc).  If exceeded,
                            a warning is issued and the solver falls back to
                            ``scipy.integrate.solve_ivp`` with ``self.method``.
                            Has no effect on the ``'scipy'`` tier (which uses
                            scipy's own step limit).  Default 100,000.
        """
        self.n_compartments = n_compartments
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.jit = jit
        self.max_steps = max_steps

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

        # Resolve effective JIT tier once per solve() call
        jit_tier = self._resolve_jit(des_callable, pk_params)

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

                t_eval_arr = (
                    np.array(eval_times_in_interval) if eval_times_in_interval
                    else np.array([t_interval_end])
                )
                try:
                    y_seg = self._integrate_segment(
                        rhs, t_interval_start, t_interval_end,
                        y_current.copy(), t_eval_arr, jit_tier,
                        des_callable, pk_params, active_infusions,
                    )
                    if eval_times_in_interval:
                        for k, orig_idx in enumerate(eval_indices):
                            y_out[orig_idx, :] = y_seg[k]
                    y_current = y_seg[-1].copy()
                except PKError:
                    raise
                except Exception as exc:
                    raise PKError(f"ADVAN6 ODE integration failed: {exc}") from exc

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


    # ── JIT helpers ────────────────────────────────────────────────────────────

    def _resolve_jit(self, des_callable: Any, pk_params: dict[str, float]) -> str:
        """Return the effective JIT tier string for this solve."""
        from openpkpd.pk.ode.jit import NUMBA_AVAILABLE  # noqa: PLC0415

        tier = self.jit
        if tier == "auto":
            # 'llc' (cfunc + LowLevelCallable) is the fastest numba path
            return "llc" if NUMBA_AVAILABLE else "numpy"
        if tier in ("numba", "llc") and not NUMBA_AVAILABLE:
            raise ImportError(
                f"jit={tier!r} requested but numba is not installed. "
                "Run: pip install numba"
            )
        return tier

    def _integrate_segment(
        self,
        rhs: Any,
        t0: float,
        tf: float,
        y0: np.ndarray,
        t_eval: np.ndarray,
        jit_tier: str,
        des_callable: Any,
        pk_params: dict[str, float],
        active_infusions: dict[int, tuple[float, float]],
    ) -> np.ndarray:
        """Integrate one piecewise segment; returns shape (len(t_eval), n_cmt)."""
        from openpkpd.pk.ode.jit import (  # noqa: PLC0415
            make_llc_rhs,
            make_numba_rhs,
            numpy_rk45_solve,
        )

        # ── Tier 3: pure-Numba RK45 + njit DES (zero Python per RHS call) ────────
        if jit_tier == "llc":
            solver = make_llc_rhs(
                des_callable, pk_params, active_infusions, self.n_compartments
            )
            if solver is not None:
                unique_t, inv = np.unique(t_eval, return_inverse=True)
                try:
                    return solver(t0, tf, y0, unique_t, self.rtol, self.atol)[inv]
                except RuntimeError as exc:
                    return self._stiff_fallback(rhs, t0, tf, y0, t_eval, exc)
            # DES compilation failed — fall through to njit path
            jit_tier = "numba"

        # ── Tier 2: @njit DES + NumPy-RK45 ────────────────────────────────────────
        if jit_tier == "numba":
            nb_result = make_numba_rhs(des_callable, pk_params, active_infusions, self.n_compartments)
            if nb_result is not None:
                rhs_nb, _, _ = nb_result
                unique_t, inv = np.unique(t_eval, return_inverse=True)
                try:
                    return numpy_rk45_solve(
                        rhs_nb, t0, tf, y0, unique_t, self.rtol, self.atol, self.max_steps
                    )[inv]
                except RuntimeError as exc:
                    return self._stiff_fallback(rhs, t0, tf, y0, t_eval, exc)
            # @njit compilation failed — fall through to numpy path
            jit_tier = "numpy"

        # ── Tier 1: pure-NumPy RK45 ────────────────────────────────────────────────
        if jit_tier == "numpy":
            unique_t, inv = np.unique(t_eval, return_inverse=True)
            try:
                return numpy_rk45_solve(
                    rhs, t0, tf, y0, unique_t, self.rtol, self.atol, self.max_steps
                )[inv]
            except RuntimeError as exc:
                return self._stiff_fallback(rhs, t0, tf, y0, t_eval, exc)

        # ── Tier 0: scipy solve_ivp (baseline) ─────────────────────────────────────
        return self._scipy_solve(rhs, t0, tf, y0, t_eval)

    def _scipy_solve(
        self,
        rhs: Any,
        t0: float,
        tf: float,
        y0: np.ndarray,
        t_eval: np.ndarray,
    ) -> np.ndarray:
        """Run scipy solve_ivp with self.method and return shape (n_eval, n_cmt)."""
        unique_t, inv = np.unique(t_eval, return_inverse=True)
        sol = solve_ivp(
            rhs, (t0, tf), y0,
            method=self.method, t_eval=unique_t,
            rtol=self.rtol, atol=self.atol, dense_output=False,
        )
        if not sol.success:
            raise PKError(f"ODE solver failed: {sol.message}")
        return sol.y[:, inv].T  # shape (n_eval, n_cmt)

    def _stiff_fallback(
        self,
        rhs: Any,
        t0: float,
        tf: float,
        y0: np.ndarray,
        t_eval: np.ndarray,
        exc: RuntimeError,
    ) -> np.ndarray:
        """
        Called when an explicit-RK45 tier hits its step limit.

        Issues a :class:`UserWarning` advising the user that the ODE may be stiff,
        then retries with ``scipy.integrate.solve_ivp`` using ``self.method``.
        If ``self.method`` is still ``'RK45'``, the warning also suggests switching
        to ``'Radau'`` or ``'BDF'``.
        """
        method_hint = (
            " Consider setting method='Radau' or method='BDF' on your ADVAN6/ADVAN8 "
            "instance for stiff models."
            if self.method == "RK45"
            else ""
        )
        warnings.warn(
            f"ODE step-limit exceeded (likely stiff ODE): {exc}. "
            f"Falling back to scipy solve_ivp (method={self.method!r}).{method_hint}",
            stacklevel=4,
        )
        return self._scipy_solve(rhs, t0, tf, y0, t_eval)


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
