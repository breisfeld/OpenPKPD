"""
Target-Mediated Drug Disposition (TMDD) models.

Implements full TMDD, quasi-steady-state (QSSA), and Michaelis-Menten
approximations as PKSubroutine ODE subclasses.

References:
    Levy G. (1994). Pharmacologic target-mediated drug disposition.
        Clin Pharmacol Ther 56(3):248-252.
    Mager DE, Jusko WJ. (2001). General pharmacokinetic model for drugs
        exhibiting target-mediated drug disposition. J Pharmacokinet Pharmacodyn
        28(6):507-532.
    Gibiansky L et al. (2008). Approximations of the target-mediated drug
        disposition model and identifiability of model parameters.
        J Pharmacokinet Pharmacodyn 35(5):573-591.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from openpkpd.pk.base import PKSolution, PKSubroutine


def _solve_ode_piecewise(
    odes_fn: Callable,
    A0: np.ndarray,
    dose_by_time: dict[float, np.ndarray],  # time -> delta_amounts per compartment
    obs_times: np.ndarray,
    t_max: float,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> dict[float, np.ndarray]:
    """
    Solve an ODE system piecewise, applying bolus doses at specified times.

    Returns a dict mapping time -> state vector at that time.
    """
    all_t = np.unique(np.concatenate([[0.0], obs_times, list(dose_by_time.keys())]))
    all_t = all_t[(all_t >= 0.0) & (all_t <= t_max + 1e-10)]
    all_t.sort()

    A_current = A0.copy()
    solution: dict[float, np.ndarray] = {0.0: A_current.copy()}

    # Apply initial dose if any
    if 0.0 in dose_by_time:
        A_current = A_current + dose_by_time[0.0]
        solution[0.0] = A_current.copy()

    prev_t = 0.0
    for t_next in all_t[1:]:
        if t_next <= prev_t + 1e-12:
            continue
        sol = solve_ivp(
            odes_fn,
            [prev_t, t_next],
            A_current,
            method="RK45",
            rtol=rtol,
            atol=atol,
            max_step=min(t_next - prev_t, 1.0),
        )
        if sol.success:
            A_current = sol.y[:, -1].copy()
        # Apply bolus dose at t_next
        if t_next in dose_by_time:
            A_current = A_current + dose_by_time[t_next]
        solution[t_next] = A_current.copy()
        prev_t = t_next

    return solution


def _extract_at_obs(
    solution: dict[float, np.ndarray],
    obs_times: np.ndarray,
    odes_fn: Callable,
    n_cmpt: int,
) -> np.ndarray:
    """Extract solution values at observation times."""
    stored = sorted(solution.keys())
    result = np.zeros((len(obs_times), n_cmpt))
    for i, t_obs in enumerate(obs_times):
        t_f = float(t_obs)
        if t_f in solution:
            result[i] = solution[t_f]
        else:
            earlier = [t for t in stored if t <= t_f]
            if earlier:
                t_from = max(earlier)
                sol = solve_ivp(
                    odes_fn,
                    [t_from, t_f],
                    solution[t_from].copy(),
                    method="RK45",
                    rtol=1e-6,
                    atol=1e-9,
                )
                result[i] = sol.y[:, -1] if sol.success else solution[t_from]
    return result


def _get_dose_events(
    dose_events: list,
    f1: float,
    cmt_idx: int,
    n_cmpt: int,
) -> dict[float, np.ndarray]:
    """Build dose_by_time dict for a single bolus compartment."""
    dose_by_time: dict[float, np.ndarray] = {}
    for de in dose_events:
        if hasattr(de, "time") and hasattr(de, "amount"):
            t = float(de.time)
            amt = float(de.amount) * f1
            delta = np.zeros(n_cmpt)
            delta[cmt_idx] = amt
            if t in dose_by_time:
                dose_by_time[t] = dose_by_time[t] + delta
            else:
                dose_by_time[t] = delta
    return dose_by_time


class FullTMDD(PKSubroutine):
    """
    Full TMDD model (Mager & Jusko 2001).

    Compartments (0-indexed):
        0: Drug (free drug, C = A[0]/V)
        1: Target (R, free receptor)
        2: Drug-target complex (RC)

    ODEs:
        dA[0]/dt = -(CL/V)*A[0] - kon*A[0]/V*A[1] + koff*A[2] + infusion
        dA[1]/dt =  Ksyn - Kdeg*A[1] - kon*A[0]/V*A[1] + koff*A[2]
        dA[2]/dt =  kon*A[0]/V*A[1] - koff*A[2] - kint*A[2]

    Parameters: CL, V, kon, koff, kint, Ksyn, Kdeg

    IPRED = A[0] / V  (free drug concentration)

    Note: A[1] at t=0 = R0 = Ksyn/Kdeg (steady state receptor)
    """

    advan = 0
    n_compartments = 3
    output_compartment = 1

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        cl = float(pk_params.get("CL", 1.0))
        v = float(pk_params.get("V", 10.0))
        kon = float(pk_params.get("kon", 0.1))
        koff = float(pk_params.get("koff", 0.01))
        kint = float(pk_params.get("kint", 0.1))
        ksyn = float(pk_params.get("Ksyn", 1.0))
        kdeg = float(pk_params.get("Kdeg", 0.1))
        f1 = float(pk_params.get("F1", 1.0))

        obs_times = np.asarray(obs_times, dtype=float)
        t_max = float(np.max(obs_times)) if len(obs_times) > 0 else 24.0

        r0 = ksyn / kdeg if kdeg > 0 else 0.0
        A0 = np.array([0.0, r0, 0.0])
        n_cmpt = 3

        def odes(t: float, y: np.ndarray) -> np.ndarray:
            c = max(y[0], 0.0) / v
            r = max(y[1], 0.0)
            rc = max(y[2], 0.0)
            dy0 = -(cl / v) * y[0] - kon * c * v * r + koff * rc
            dy1 = ksyn - kdeg * r - kon * c * v * r + koff * rc
            dy2 = kon * c * v * r - koff * rc - kint * rc
            return np.array([dy0, dy1, dy2])

        dose_by_time = _get_dose_events(dose_events, f1, 0, n_cmpt)
        solution = _solve_ode_piecewise(odes, A0, dose_by_time, obs_times, t_max)
        amounts = _extract_at_obs(solution, obs_times, odes, n_cmpt)
        amounts = np.maximum(amounts, 0.0)
        ipred = amounts[:, 0] / v

        return PKSolution(times=obs_times, amounts=amounts, ipred=ipred)


class QSSATMDDModel(PKSubroutine):
    """
    Quasi-steady-state (QSSA) TMDD approximation (Gibiansky 2008).

    Two-compartment ODE with quasi-steady-state for the drug-target complex:
        Kss = (koff + kint) / kon

        dC/dt = -(CL/V)*C - kint*Rtot*C/(Kss+C) + (Ksyn - Kdeg*Rtot + kint*Rtot*C/(Kss+C))/V
        (or equivalently, per the Gibiansky approximation)

    Compartments (0-indexed):
        0: Drug amount in central (C = A[0]/V)
        1: Total receptor (Rtot = R + RC)

    Parameters: CL, V, Kss, kint, Ksyn, Kdeg

    IPRED = A[0] / V
    """

    advan = 0
    n_compartments = 2
    output_compartment = 1

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        cl = float(pk_params.get("CL", 1.0))
        v = float(pk_params.get("V", 10.0))
        kss = float(pk_params.get("Kss", 1.0))
        kint = float(pk_params.get("kint", 0.1))
        ksyn = float(pk_params.get("Ksyn", 1.0))
        kdeg = float(pk_params.get("Kdeg", 0.1))
        f1 = float(pk_params.get("F1", 1.0))

        obs_times = np.asarray(obs_times, dtype=float)
        t_max = float(np.max(obs_times)) if len(obs_times) > 0 else 24.0

        rtot0 = ksyn / kdeg if kdeg > 0 else 0.0
        A0 = np.array([0.0, rtot0])
        n_cmpt = 2

        def odes(t: float, y: np.ndarray) -> np.ndarray:
            a_drug = max(y[0], 0.0)
            rtot = max(y[1], 0.0)
            c = a_drug / v
            rc_frac = c / (kss + c) if (kss + c) > 0 else 0.0
            da_drug = -(cl / v) * a_drug - kint * rtot * rc_frac * v
            drtot = ksyn - kdeg * rtot - kint * rtot * rc_frac
            return np.array([da_drug, drtot])

        dose_by_time = _get_dose_events(dose_events, f1, 0, n_cmpt)
        solution = _solve_ode_piecewise(odes, A0, dose_by_time, obs_times, t_max)
        amounts = _extract_at_obs(solution, obs_times, odes, n_cmpt)
        amounts = np.maximum(amounts, 0.0)
        ipred = amounts[:, 0] / v

        return PKSolution(times=obs_times, amounts=amounts, ipred=ipred)


class MichaelisMentenTMDD(PKSubroutine):
    """
    Michaelis-Menten (MM) TMDD approximation (1-compartment with saturable elimination).

    ODE in amount form:
        dA/dt = -(CL/V)*A - Vmax*A/(Km*V + A)

    Reduces to ADVAN10 when CL = 0.

    Parameters: CL, V, Vmax, Km

    IPRED = A[0] / V
    """

    advan = 0
    n_compartments = 1
    output_compartment = 1

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        cl = float(pk_params.get("CL", 0.0))
        v = float(pk_params.get("V", 10.0))
        vmax = float(pk_params.get("Vmax", 1.0))
        km = float(pk_params.get("Km", 1.0))
        f1 = float(pk_params.get("F1", 1.0))

        obs_times = np.asarray(obs_times, dtype=float)
        t_max = float(np.max(obs_times)) if len(obs_times) > 0 else 24.0

        A0 = np.array([0.0])
        n_cmpt = 1

        def odes(t: float, y: np.ndarray) -> np.ndarray:
            a = max(y[0], 0.0)
            c = a / v
            da = -(cl / v) * a - vmax * c / (km + c)
            return np.array([da])

        dose_by_time = _get_dose_events(dose_events, f1, 0, n_cmpt)
        solution = _solve_ode_piecewise(odes, A0, dose_by_time, obs_times, t_max)
        amounts = _extract_at_obs(solution, obs_times, odes, n_cmpt)
        amounts = np.maximum(amounts, 0.0)
        ipred = amounts[:, 0] / v

        return PKSolution(times=obs_times, amounts=amounts, ipred=ipred)
