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
        self._schedule_cache: dict[
            tuple[object, ...],
            tuple[
                list[float],
                dict[float, list[DoseEvent]],
                list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]],
                np.ndarray,
                set[float],
            ],
        ] = {}

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

        (
            all_breakpoints,
            doses_by_breakpoint,
            interval_plans,
            trailing_obs_indices,
            cov_change_set,
        ) = self._build_schedule(
            dose_events,
            obs_times,
            f_factors=f_factors,
            alag=alag,
            covariate_change_times=covariate_change_times,
            refresh_covariates=(covariate_fn is not None),
        )
        t_start = 0.0

        # Initial state: for SS doses, compute the periodic-orbit pre-dose state.
        # Only the first SS dose establishes the initial condition; subsequent doses
        # are applied normally in the main integration loop.
        # Scan dose_events directly (no extra _prepare_doses call) and apply the
        # F-factor and lag-time adjustments on-the-fly so the schedule cache is not
        # invalidated.
        y_current = np.zeros(self.n_compartments)
        for _d in dose_events:
            if _d.reset or not (_d.ss and _d.ii > 0):
                continue
            cmt = int(_d.compartment)
            f = f_factors.get(cmt, 1.0)
            lag = alag.get(cmt, 0.0)
            # Build an F/lag-adjusted copy for _find_ss_state
            _d_adj = DoseEvent(
                time=_d.time + lag,
                amount=_d.amount * f,
                rate=_d.rate,
                duration=_d.duration,
                compartment=cmt,
                ss=False,
                ii=_d.ii,
                reset=False,
            )
            try:
                y_current = self._find_ss_state(_d_adj, des_callable, pk_params, jit_tier)
            except Exception as exc:
                warnings.warn(
                    f"ADVAN6: SS initial-state computation failed ({exc}); "
                    "using y=0 as starting state.",
                    UserWarning,
                    stacklevel=3,
                )
            break  # only the first SS dose sets the initial condition

        y_out = np.zeros((len(obs_times), self.n_compartments))

        # Track active infusions: {cmt_idx: (rate, end_time)}
        active_infusions: dict[int, tuple[float, float]] = {}

        # Apply any doses at t=0 before starting integration
        for d in doses_by_breakpoint.get(t_start, []):
            if abs(d.time - t_start) < 1e-12:
                _apply_dose_event(d, y_current, active_infusions)

        for t_interval_start, t_interval_end, t_eval_arr, eval_indices, prefill_indices in interval_plans:
            bp = t_interval_end
            if prefill_indices.size:
                y_out[prefill_indices, :] = y_current

            if t_interval_end > t_interval_start + 1e-14:
                # Build RHS for this interval (with active infusions)
                rhs = _make_rhs(des_callable, pk_params, active_infusions, self.n_compartments)
                try:
                    y_seg = self._integrate_segment(
                        rhs, t_interval_start, t_interval_end,
                        y_current.copy(), t_eval_arr, jit_tier,
                        des_callable, pk_params, active_infusions,
                    )
                    if eval_indices.size:
                        y_out[eval_indices, :] = y_seg[: len(eval_indices)]
                    y_current = y_seg[-1].copy()
                except PKError:
                    raise
                except Exception as exc:
                    raise PKError(f"ADVAN6 ODE integration failed: {exc}") from exc

            # Apply dose events and infusion state changes at breakpoint bp
            for d in doses_by_breakpoint.get(bp, []):
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

        # Any remaining obs times after the last breakpoint get current state
        if trailing_obs_indices.size:
            y_out[trailing_obs_indices, :] = y_current

        # Only clip tiny negative roundoff. Some ODE states, especially PD
        # deviation states, are legitimately signed and must not be forced to 0.
        clip_tol = max(float(self.atol) * 10.0, 1e-12)
        y_out = np.where((y_out < 0.0) & (np.abs(y_out) <= clip_tol), 0.0, y_out)

        # IPRED = A[output_cmt - 1] / V
        ipred = y_out[:, out_cmt_idx] / v

        return PKSolution(
            times=obs_times.copy(),
            amounts=y_out,
            ipred=ipred,
        )

    def _find_ss_state(
        self,
        ss_dose: "DoseEvent",
        des_callable: "Callable",
        pk_params: dict[str, float],
        jit_tier: str,
        max_iter: int = 500,
        rtol: float = 1e-5,
    ) -> np.ndarray:
        """
        Find the periodic-orbit pre-dose state for a steady-state dose event.

        Iterates: apply dose → integrate one dosing interval τ → repeat until
        the state at the end of each interval converges to the state at its start.

        This is the standard NONMEM SS=1 initialisation for ODE models:
        integrate multiple cycles starting from y=0 until the system reaches
        its periodic orbit, then return the converged pre-dose state y_pre.

        The caller's main loop then applies the dose normally (adding dose.amount
        to the appropriate compartment) before proceeding with the observation-
        window integration.

        Args:
            ss_dose:     The DoseEvent with ss=True and ii>0 (already F-adjusted).
            des_callable: Compiled $DES callable.
            pk_params:   PK parameters for the RHS.
            jit_tier:    JIT tier string.
            max_iter:    Maximum iteration cycles (default 500).
            rtol:        Relative convergence tolerance (default 1e-5).

        Returns:
            y_pre: state vector just before the dose at the periodic orbit.
        """
        tau = float(ss_dose.ii)
        n = self.n_compartments
        y_pre = np.zeros(n)

        for _ in range(max_iter):
            y_post = y_pre.copy()
            active_inf: dict[int, tuple[float, float]] = {}

            # Apply the dose (bolus: add to compartment; infusion: register rate)
            if ss_dose.is_bolus:
                cmt_idx = ss_dose.compartment - 1
                if 0 <= cmt_idx < n:
                    y_post[cmt_idx] += ss_dose.amount
                rhs = _make_rhs(des_callable, pk_params, {}, n)
                seg = self._integrate_segment(
                    rhs, 0.0, tau, y_post, np.array([tau]),
                    jit_tier, des_callable, pk_params, {}
                )
                y_next = seg[-1].copy()
            else:
                # Infusion: apply rate for duration d, then free decay for remainder
                dur = ss_dose.amount / ss_dose.rate
                cmt_idx = ss_dose.compartment - 1
                end_t = min(dur, tau)
                if 0 <= cmt_idx < n:
                    active_inf[cmt_idx] = (ss_dose.rate, end_t)
                rhs_on = _make_rhs(des_callable, pk_params, active_inf, n)
                seg_on = self._integrate_segment(
                    rhs_on, 0.0, end_t, y_post, np.array([end_t]),
                    jit_tier, des_callable, pk_params, active_inf
                )
                y_mid = seg_on[-1].copy()
                if dur < tau:
                    rhs_off = _make_rhs(des_callable, pk_params, {}, n)
                    seg_off = self._integrate_segment(
                        rhs_off, dur, tau, y_mid, np.array([tau]),
                        jit_tier, des_callable, pk_params, {}
                    )
                    y_next = seg_off[-1].copy()
                else:
                    y_next = y_mid

            norm_next = float(np.linalg.norm(y_next))
            if norm_next > 0.0 and float(np.linalg.norm(y_next - y_pre)) / norm_next < rtol:
                return y_next
            y_pre = y_next

        return y_pre

    def _build_schedule(
        self,
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
        *,
        f_factors: dict[int, float],
        alag: dict[int, float],
        covariate_change_times: list[float] | None,
        refresh_covariates: bool,
    ) -> tuple[
        list[float],
        dict[float, list[DoseEvent]],
        list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]],
        np.ndarray,
        set[float],
    ]:
        """Return cached schedule structures for a subject/event layout."""
        key = _schedule_cache_key(
            dose_events,
            obs_times,
            f_factors=f_factors,
            alag=alag,
            covariate_change_times=covariate_change_times,
            refresh_covariates=refresh_covariates,
        )
        cached = self._schedule_cache.get(key)
        if cached is not None:
            return cached

        active_doses = _prepare_doses(dose_events, f_factors, alag)
        t_start = 0.0
        t_end = float(np.max(obs_times))

        event_times: list[float] = [t_start]
        doses_by_breakpoint: dict[float, list[DoseEvent]] = {}
        for d in active_doses:
            t_dose = float(d.time)
            event_times.append(t_dose)
            doses_by_breakpoint.setdefault(t_dose, []).append(d)
            if d.is_infusion:
                event_times.append(float(d.infusion_end_time))
        event_times.append(t_end)

        cov_change_set: set[float] = set()
        if covariate_change_times:
            cov_times = [float(tc) for tc in covariate_change_times]
            event_times.extend(cov_times)
            if refresh_covariates:
                cov_change_set = set(cov_times)

        all_breakpoints = sorted(set(event_times))
        all_breakpoints = [t for t in all_breakpoints if t_start <= t <= t_end + 1e-12]

        obs_times_sorted_idx = np.argsort(obs_times)
        obs_times_sorted = np.asarray(obs_times[obs_times_sorted_idx], dtype=float)

        interval_plans: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = []
        obs_queue = list(zip(obs_times_sorted, obs_times_sorted_idx, strict=False))
        obs_ptr = 0
        prev_t = t_start
        for bp in all_breakpoints:
            if bp <= prev_t + 1e-14:
                continue
            prefill_indices: list[int] = []
            eval_times_in_interval: list[float] = []
            eval_indices: list[int] = []
            while obs_ptr < len(obs_queue):
                t_obs, orig_idx = obs_queue[obs_ptr]
                if t_obs <= bp + 1e-12 and t_obs > prev_t - 1e-12:
                    if t_obs > prev_t + 1e-14:
                        eval_times_in_interval.append(float(t_obs))
                        eval_indices.append(int(orig_idx))
                    else:
                        prefill_indices.append(int(orig_idx))
                    obs_ptr += 1
                elif t_obs <= prev_t + 1e-12:
                    prefill_indices.append(int(orig_idx))
                    obs_ptr += 1
                else:
                    break
            t_eval_arr = (
                np.asarray(eval_times_in_interval, dtype=float)
                if eval_times_in_interval
                else np.asarray([bp], dtype=float)
            )
            interval_plans.append(
                (
                    prev_t,
                    bp,
                    t_eval_arr,
                    np.asarray(eval_indices, dtype=int),
                    np.asarray(prefill_indices, dtype=int),
                )
            )
            prev_t = bp
        trailing_obs_indices = np.asarray([int(orig_idx) for _, orig_idx in obs_queue[obs_ptr:]], dtype=int)

        value = (
            all_breakpoints,
            doses_by_breakpoint,
            interval_plans,
            trailing_obs_indices,
            cov_change_set,
        )
        self._schedule_cache[key] = value
        return value


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


def _schedule_cache_key(
    dose_events: list[DoseEvent],
    obs_times: np.ndarray,
    *,
    f_factors: dict[int, float],
    alag: dict[int, float],
    covariate_change_times: list[float] | None,
    refresh_covariates: bool,
) -> tuple[object, ...]:
    dose_key = tuple(
        (
            float(d.time),
            float(d.amount),
            float(d.rate),
            float(d.duration),
            int(d.compartment),
            bool(d.ss),
            float(d.ii),
            bool(d.reset),
        )
        for d in dose_events
    )
    obs_key = tuple(float(t) for t in np.asarray(obs_times, dtype=float).tolist())
    f_key = tuple(sorted((int(cmt), float(val)) for cmt, val in f_factors.items()))
    alag_key = tuple(sorted((int(cmt), float(val)) for cmt, val in alag.items()))
    cov_key = tuple(float(t) for t in (covariate_change_times or []))
    return (dose_key, obs_key, f_key, alag_key, cov_key, bool(refresh_covariates))


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
