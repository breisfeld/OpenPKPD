"""
Transit compartment absorption model.

Implements the Savic (2007) n-transit compartment absorption model feeding
into a 1- or 2-compartment disposition model.

Reference:
    Savic RM et al. (2007). Implementation of a transit compartment model for
    describing drug absorption in pharmacokinetic studies.
    J Pharmacokinet Pharmacodyn 34(5):711-726.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from openpkpd.pk.base import PKSolution, PKSubroutine


class TransitAbsorption(PKSubroutine):
    """
    Savic (2007) n-transit compartment absorption model.

    Compartments:
        1..n_transit:   Transit depot compartments
        n_transit+1:    Central (output) compartment
        n_transit+2:    (optional) Peripheral compartment if 2-cmt

    ODE system:
        dA(1)/dt  = -KTR * A(1)                           [first transit]
        dA(i)/dt  = KTR * A(i-1) - KTR * A(i)             [transit i]
        dA(N+1)/dt = KTR * A(N) - (CL/V) * A(N+1)         [central, 1-cmt]
        (for 2-cmt add inter-compartmental exchange)

    Parameters:
        KTR:       Transit rate constant (inverse time).
        N_TRANSIT: Number of transit compartments (positive integer).
        CL:        Clearance.
        V:         Volume of central compartment.
        K12:       (optional) Central to peripheral rate constant.
        K21:       (optional) Peripheral to central rate constant.
        V2:        (optional) Peripheral volume (used for amount conversion only if needed).
        F1:        (optional) Bioavailability fraction, default 1.0.

    IPRED = A(N_TRANSIT+1) / V
    """

    advan = 0
    output_compartment = 1  # central is compartment index N+1 in internal numbering

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        """Solve transit absorption model via ODE."""
        ktr = float(pk_params.get("KTR", pk_params.get("ktr", 1.0)))
        n_transit = int(pk_params.get("N_TRANSIT", pk_params.get("n_transit", 3)))
        cl = float(pk_params.get("CL", pk_params.get("cl", 1.0)))
        v = float(pk_params.get("V", pk_params.get("v", 10.0)))
        f1 = float(pk_params.get("F1", pk_params.get("f1", 1.0)))
        k12 = float(pk_params.get("K12", pk_params.get("k12", 0.0)))
        k21 = float(pk_params.get("K21", pk_params.get("k21", 0.0)))

        two_cmt = k12 > 0 and k21 > 0
        n_cmpt = n_transit + 1 + (1 if two_cmt else 0)

        obs_times = np.asarray(obs_times, dtype=float)

        # Collect dose amounts per time, applying bioavailability
        dose_by_time: dict[float, float] = {}
        for de in dose_events:
            t = float(de.time) if hasattr(de, "time") else 0.0
            amt = float(de.amount) * f1 if hasattr(de, "amount") else 0.0
            dose_by_time[t] = dose_by_time.get(t, 0.0) + amt

        def odes(t: float, y: np.ndarray) -> np.ndarray:
            dy = np.zeros(n_cmpt)
            # First transit compartment
            dy[0] = -ktr * y[0]
            # Intermediate transit compartments
            for i in range(1, n_transit):
                dy[i] = ktr * y[i - 1] - ktr * y[i]
            # Input into central from last transit compartment
            central_in = ktr * y[n_transit - 1] if n_transit > 0 else 0.0
            k_elim = cl / v
            if two_cmt:
                dy[n_transit] = (
                    central_in - k_elim * y[n_transit] - k12 * y[n_transit] + k21 * y[n_transit + 1]
                )
                dy[n_transit + 1] = k12 * y[n_transit] - k21 * y[n_transit + 1]
            else:
                dy[n_transit] = central_in - k_elim * y[n_transit]
            return dy

        # Initial state
        A0 = np.zeros(n_cmpt)

        # Apply dose at t=0 if present
        if 0.0 in dose_by_time:
            A0[0] += dose_by_time[0.0]

        # All time points to step through: obs times + additional dose times
        all_t_points = np.unique(np.concatenate([[0.0], obs_times, list(dose_by_time.keys())]))
        all_t_points.sort()

        # Step through time, storing solution at each point
        solution_store: dict[float, np.ndarray] = {0.0: A0.copy()}
        A_current = A0.copy()
        prev_t = 0.0

        for t_next in all_t_points[1:]:
            if t_next <= prev_t + 1e-14:
                continue
            sol = solve_ivp(
                odes,
                [prev_t, t_next],
                A_current,
                method="RK45",
                dense_output=False,
                rtol=1e-6,
                atol=1e-9,
                max_step=min(t_next - prev_t, 1.0),
            )
            if sol.success:
                A_current = sol.y[:, -1].copy()
            # Apply dose at t_next if any (after integrating to that time)
            if t_next in dose_by_time and t_next != 0.0:
                A_current[0] += dose_by_time[t_next]
            solution_store[t_next] = A_current.copy()
            prev_t = t_next

        # Extract solution at obs_times
        stored_times = sorted(solution_store.keys())
        n_obs = len(obs_times)
        result_amounts = np.zeros((n_obs, n_cmpt))

        for i, t_obs in enumerate(obs_times):
            t_obs_f = float(t_obs)
            if t_obs_f in solution_store:
                result_amounts[i] = solution_store[t_obs_f]
            else:
                # Find the closest earlier stored time and integrate forward
                earlier = [t for t in stored_times if t <= t_obs_f]
                if earlier:
                    t_from = max(earlier)
                    A_from = solution_store[t_from].copy()
                    sol = solve_ivp(
                        odes,
                        [t_from, t_obs_f],
                        A_from,
                        method="RK45",
                        rtol=1e-6,
                        atol=1e-9,
                    )
                    result_amounts[i] = sol.y[:, -1] if sol.success else A_from

        # IPRED = amount in central compartment / V
        central_idx = n_transit  # 0-indexed
        ipred = result_amounts[:, central_idx] / v
        ipred = np.maximum(ipred, 0.0)

        return PKSolution(
            times=obs_times,
            amounts=result_amounts,
            ipred=ipred,
        )
