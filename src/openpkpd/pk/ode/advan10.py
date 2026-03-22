"""
ADVAN10 — 1-compartment Michaelis-Menten elimination model.

The nonlinear ODE for the central compartment amount A is:

    dA/dt = -Vmax * A / (Km * V + A)

where:
    Vmax : Maximum elimination rate (mass/time, e.g., mg/h).
    Km   : Michaelis constant (concentration at half-maximal rate, e.g., mg/L).
    V    : Volume of distribution (L).

This is a special-case ODE that cannot be solved analytically in closed form
for arbitrary initial conditions. The equation is solved numerically using
scipy solve_ivp (LSODA by default for potential stiffness at high concentrations).

IPRED = A(t) / V.

For multiple doses, superposition does NOT apply (nonlinear system);
instead the ODE is integrated continuously with dose events applied
as instantaneous state changes (bolus) or infusion rate additions.

Parameters expected in pk_params:
    Vmax : Maximum elimination rate.
    Km   : Michaelis-Menten constant.
    V    : Volume of distribution.
    F1   : Bioavailability (optional, default 1.0).
    ALAG1: Lag time (optional, default 0.0).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.pk.ode.advan6 import _apply_dose_event, _prepare_doses
from openpkpd.utils.errors import PKError


def _get_primary_or_alias(
    pk_params: dict[str, float],
    primary: str,
    alias: str,
) -> float | None:
    """Return the primary value if present; otherwise fall back to an alias."""
    if primary in pk_params:
        return pk_params[primary]
    if alias in pk_params:
        return pk_params[alias]
    return None


class ADVAN10(PKSubroutine):
    """
    1-compartment Michaelis-Menten elimination model (ADVAN10).

    The ODE is: dA/dt = -Vmax * A / (Km * V + A)

    Integrated with scipy solve_ivp (LSODA). Supports bolus and infusion
    dosing with bioavailability (F1) and lag time (ALAG1).

    Parameters:
        rtol: Relative ODE tolerance (default 1e-6).
        atol: Absolute ODE tolerance (default 1e-9).
        method: Integration method (default 'LSODA').
    """

    n_compartments: int = 1
    advan: int = 10
    output_compartment: int = 1

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        method: str = "LSODA",
    ) -> None:
        """
        Initialize ADVAN10.

        Args:
            rtol:   Relative ODE tolerance.
            atol:   Absolute ODE tolerance.
            method: scipy solve_ivp method ('LSODA', 'Radau', 'RK45').
        """
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
    ) -> PKSolution:
        """
        Solve the Michaelis-Menten 1-compartment model.

        Args:
            pk_params:    Dict with Vmax, Km, V (plus optional F1, ALAG1).
            dose_events:  List of DoseEvent objects.
            obs_times:    1-D array of observation times.
            pk_callable:  Unused.
            des_callable: Unused (ADVAN10 has a built-in ODE).

        Returns:
            PKSolution with amounts shape (n_times, 1) and IPRED = A/V.

        Raises:
            PKError: If required parameters are missing or ODE fails.
        """
        vmax = _get_primary_or_alias(pk_params, "Vmax", "VMAX")
        km = _get_primary_or_alias(pk_params, "Km", "KM")
        v = _get_primary_or_alias(pk_params, "V", "V1")

        if vmax is None or vmax <= 0:
            raise PKError(f"ADVAN10 requires Vmax > 0; got Vmax={vmax}")
        if km is None or km <= 0:
            raise PKError(f"ADVAN10 requires Km > 0; got Km={km}")
        if v is None or v <= 0:
            raise PKError(f"ADVAN10 requires V > 0; got V={v}")

        if len(obs_times) == 0:
            return PKSolution(
                times=obs_times.copy(),
                amounts=np.zeros((0, 1)),
                ipred=np.zeros(0),
            )

        f_factors: dict[int, float] = {}
        f1 = pk_params.get("F1")
        if f1 is not None:
            f_factors[1] = float(f1)

        alag: dict[int, float] = {}
        alag1 = pk_params.get("ALAG1")
        if alag1 is not None:
            alag[1] = float(alag1)

        active_doses = _prepare_doses(dose_events, f_factors, alag)

        t_start = 0.0
        t_end = float(np.max(obs_times))

        # Collect event breakpoints
        event_times: list[float] = [t_start]
        for d in active_doses:
            event_times.append(float(d.time))
            if d.is_infusion:
                event_times.append(float(d.infusion_end_time))
        event_times.append(t_end)
        all_breakpoints = sorted({t for t in event_times if t_start <= t <= t_end + 1e-12})

        # Sort obs times
        obs_sorted_idx = np.argsort(obs_times)
        obs_sorted = obs_times[obs_sorted_idx]

        y_out = np.zeros(len(obs_times))  # amounts at obs times

        # Current state
        y_current = np.array([0.0])  # [A]
        active_infusions: dict[int, tuple[float, float]] = {}

        # Apply dose events at t=0
        for d in active_doses:
            if abs(d.time - t_start) < 1e-12:
                _apply_dose_event(d, y_current, active_infusions)

        obs_queue = list(zip(obs_sorted, obs_sorted_idx, strict=False))
        obs_ptr = 0
        prev_t = t_start

        def _make_rhs_mm(infusions: dict[int, tuple[float, float]]) -> Callable:
            """Build Michaelis-Menten RHS with optional infusion."""
            infusion_snap = dict(infusions)

            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                a = max(y[0], 0.0)
                # Elimination: -Vmax * A / (Km * V + A)
                dydt = -float(vmax) * a / (float(km) * float(v) + a + 1e-30)
                # Add infusion
                for cmt_idx, (rate, end_t) in infusion_snap.items():
                    if t <= end_t + 1e-14 and cmt_idx == 0:
                        dydt += rate
                return np.array([dydt])

            return rhs

        for bp in all_breakpoints:
            if bp <= prev_t + 1e-14:
                continue

            t_int_start = prev_t
            t_int_end = bp

            eval_times: list[float] = []
            eval_idx: list[int] = []

            while obs_ptr < len(obs_queue):
                t_obs, orig_idx = obs_queue[obs_ptr]
                if t_obs <= t_int_end + 1e-12 and t_obs > t_int_start - 1e-12:
                    if t_obs > t_int_start + 1e-14:
                        eval_times.append(float(t_obs))
                        eval_idx.append(orig_idx)
                    obs_ptr += 1
                elif t_obs <= t_int_start + 1e-12:
                    y_out[orig_idx] = y_current[0]
                    obs_ptr += 1
                else:
                    break

            if t_int_end > t_int_start + 1e-14:
                rhs = _make_rhs_mm(active_infusions)

                t_eval_arr: np.ndarray | None = None
                if eval_times:
                    t_eval_arr = np.array(eval_times)

                try:
                    sol = solve_ivp(
                        rhs,
                        (t_int_start, t_int_end),
                        y_current.copy(),
                        method=self.method,
                        t_eval=t_eval_arr if t_eval_arr is not None else np.array([t_int_end]),
                        rtol=self.rtol,
                        atol=self.atol,
                    )
                    if not sol.success:
                        raise PKError(f"ADVAN10 ODE solver failed: {sol.message}")

                    if eval_times and t_eval_arr is not None:
                        for k, orig_idx in enumerate(eval_idx):
                            y_out[orig_idx] = max(sol.y[0, k], 0.0)

                    y_current = sol.y[:, -1].copy()
                    y_current = np.maximum(y_current, 0.0)

                except Exception as exc:
                    if not isinstance(exc, PKError):
                        raise PKError(f"ADVAN10 ODE integration failed: {exc}") from exc
                    raise

            # Apply dose events at breakpoint bp
            for d in active_doses:
                if abs(d.time - bp) < 1e-12:
                    _apply_dose_event(d, y_current, active_infusions)

            # Remove expired infusions
            active_infusions = {
                ci: (r, et) for ci, (r, et) in active_infusions.items() if et > bp + 1e-14
            }

            prev_t = bp

        # Remaining obs times
        while obs_ptr < len(obs_queue):
            _, orig_idx = obs_queue[obs_ptr]
            y_out[orig_idx] = max(y_current[0], 0.0)
            obs_ptr += 1

        amounts = y_out.reshape(-1, 1)
        ipred = y_out / float(v)

        return PKSolution(
            times=obs_times.copy(),
            amounts=amounts,
            ipred=ipred,
        )
