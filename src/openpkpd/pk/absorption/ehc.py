"""
Enterohepatic recirculation (EHC) model.

Models gallbladder emptying at specified intervals re-introducing drug into the
central compartment. Uses scipy.integrate.solve_ivp for ODE integration.

Compartments:
    0: Depot (optional, used when KA > 0)
    1: Central (1-indexed: 1)
    2: Gallbladder (1-indexed: 2)

Parameters:
    CL:           Clearance from central compartment.
    V:            Volume of central compartment.
    KGB:          Gallbladder emptying rate constant (during emptying window).
    FGBMAX:       Fraction of hepatically cleared drug entering gallbladder.
    EHC_INTERVAL: Interval between gallbladder emptying events (time units).
    EHC_DURATION: Duration of each emptying window (default = EHC_INTERVAL / 10).
    KA:           Optional absorption rate (if oral dosing).
    F1:           Bioavailability.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from openpkpd.pk.base import PKSolution, PKSubroutine


class EnterohepatiCRecirculation(PKSubroutine):
    """
    Gallbladder EHC model with ODE integration.

    Drug cleared hepatically partially enters the gallbladder.
    At regular intervals, the gallbladder empties into the central compartment.
    """

    advan = 0
    n_compartments = 3  # depot (optional), central, gallbladder

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        """Solve EHC model via ODE."""
        cl = float(pk_params.get("CL", pk_params.get("cl", 1.0)))
        v = float(pk_params.get("V", pk_params.get("v", 10.0)))
        kgb = float(pk_params.get("KGB", pk_params.get("kgb", 0.5)))
        fgbmax = float(pk_params.get("FGBMAX", pk_params.get("fgbmax", 0.1)))
        ehc_interval = float(pk_params.get("EHC_INTERVAL", pk_params.get("ehc_interval", 6.0)))
        ehc_duration = float(
            pk_params.get(
                "EHC_DURATION",
                pk_params.get("ehc_duration", ehc_interval / 10.0),
            )
        )
        ka = float(pk_params.get("KA", pk_params.get("ka", 0.0)))
        f1 = float(pk_params.get("F1", pk_params.get("f1", 1.0)))

        obs_times = np.asarray(obs_times, dtype=float)
        k = cl / v  # elimination rate constant

        # Compartments: [depot (0), central (1), gallbladder (2)]
        use_depot = ka > 0
        n_cmpt = 3

        # Determine integration end time
        t_max = float(np.max(obs_times)) if len(obs_times) > 0 else 24.0

        # Collect dose events
        dose_by_time: dict[float, float] = {}
        for de in dose_events:
            if hasattr(de, "time") and hasattr(de, "amount"):
                t = float(de.time)
                amt = float(de.amount) * f1
                dose_by_time[t] = dose_by_time.get(t, 0.0) + amt

        def is_emptying(t: float) -> bool:
            """Return True if time t falls within a gallbladder emptying window."""
            if ehc_interval <= 0:
                return False
            phase = t % ehc_interval
            return phase < ehc_duration

        def odes(t: float, y: np.ndarray) -> np.ndarray:
            dy = np.zeros(n_cmpt)
            emptying = is_emptying(t)

            # Depot absorption into central
            if use_depot and ka > 0:
                dy[0] = -ka * y[0]
                central_input = ka * y[0]
            else:
                central_input = 0.0

            # Gallbladder empties into central during emptying window
            gb_to_central = kgb * y[2] if emptying else 0.0

            # Fraction of hepatic clearance diverted to gallbladder
            hepatic_cl_to_gb = k * y[1] * fgbmax

            # Central compartment ODE
            dy[1] = central_input + gb_to_central - k * y[1]

            # Gallbladder: accumulates hepatic fraction, empties periodically
            dy[2] = hepatic_cl_to_gb - gb_to_central

            return dy

        # Initial state
        A0 = np.zeros(n_cmpt)

        # Apply dose at t=0 if present
        if 0.0 in dose_by_time:
            if use_depot:
                A0[0] += dose_by_time[0.0]
            else:
                A0[1] += dose_by_time[0.0]

        # Build EHC breakpoints (start and end of each emptying window)
        ehc_breakpoints: list[float] = []
        if ehc_interval > 0 and t_max > 0:
            t_ehc = 0.0
            while t_ehc <= t_max + ehc_interval:
                ehc_breakpoints.append(t_ehc)
                ehc_breakpoints.append(t_ehc + ehc_duration)
                t_ehc += ehc_interval

        # Combine all breakpoints: observations, doses, EHC transitions
        all_t = np.unique(
            np.concatenate(
                [
                    [0.0],
                    obs_times,
                    list(dose_by_time.keys()),
                    ehc_breakpoints,
                ]
            )
        )
        all_t = all_t[(all_t >= 0.0) & (all_t <= t_max + 1e-10)]
        all_t.sort()

        # Simulate piecewise
        solution_store: dict[float, np.ndarray] = {0.0: A0.copy()}
        A_current = A0.copy()
        prev_t = 0.0

        for t_next in all_t[1:]:
            if t_next <= prev_t + 1e-12:
                continue
            sol = solve_ivp(
                odes,
                [prev_t, t_next],
                A_current,
                method="RK45",
                rtol=1e-6,
                atol=1e-9,
                max_step=min(t_next - prev_t, 0.5),
            )
            if sol.success:
                A_current = sol.y[:, -1].copy()
            # Apply any dose event at this breakpoint (excluding t=0 already handled)
            if t_next in dose_by_time and t_next != 0.0:
                if use_depot:
                    A_current[0] += dose_by_time[t_next]
                else:
                    A_current[1] += dose_by_time[t_next]
            solution_store[t_next] = A_current.copy()
            prev_t = t_next

        # Extract solution at observation times
        stored_times_sorted = sorted(solution_store.keys())
        n_obs = len(obs_times)
        result_amounts = np.zeros((n_obs, n_cmpt))

        for i, t_obs in enumerate(obs_times):
            t_f = float(t_obs)
            if t_f in solution_store:
                result_amounts[i] = solution_store[t_f]
            else:
                # Find the closest earlier stored time and integrate forward
                earlier = [t for t in stored_times_sorted if t <= t_f]
                if earlier:
                    t_from = max(earlier)
                    sol = solve_ivp(
                        odes,
                        [t_from, t_f],
                        solution_store[t_from].copy(),
                        method="RK45",
                        rtol=1e-6,
                        atol=1e-9,
                    )
                    result_amounts[i] = sol.y[:, -1] if sol.success else solution_store[t_from]

        # IPRED = central compartment amount / V
        ipred = result_amounts[:, 1] / v
        ipred = np.maximum(ipred, 0.0)

        return PKSolution(
            times=obs_times,
            amounts=result_amounts,
            ipred=ipred,
        )
