"""
ADVAN1 — 1-compartment IV bolus model.

Analytical solution:
    A(t) = DOSE * exp(-K * t)     (after a single bolus)
    C(t) = A(t) / V

For multiple doses, superposition is used:
    A(t) = Σ_i DOSE_i * exp(-K * (t - t_i))   for all t_i ≤ t

Infusions of duration D (rate = DOSE / D):
    A(t) =  rate/K * (1 - exp(-K*(t-t0)))           for t0 ≤ t ≤ t0+D
    A(t) =  rate/K * (1 - exp(-K*D)) * exp(-K*(t-t0-D))  for t > t0+D
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.errors import PKError


class ADVAN1(PKSubroutine):
    """1-compartment IV bolus model (ADVAN1)."""

    n_compartments = 1
    advan = 1
    output_compartment = 1

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        k = pk_params.get("K")
        v = pk_params.get("V")

        if k is None or k <= 0:
            raise PKError(f"ADVAN1 requires K > 0, got K={k}")
        if v is None or v <= 0:
            raise PKError(f"ADVAN1 requires V > 0, got V={v}")

        # Filter dosing events (ignore resets for now — reset to zero)
        doses = [e for e in dose_events if not e.reset]
        amounts = np.zeros(len(obs_times))

        for dose in doses:
            if dose.is_bolus:
                # A(t) = DOSE * exp(-K * (t - t_dose)) for t > t_dose
                # Strict inequality: pre-dose convention (observation at t_dose is pre-dose)
                dt = obs_times - dose.time
                mask = dt > 0
                amounts[mask] += dose.amount * np.exp(-k * dt[mask])
            else:
                # Infusion: rate R for duration D = AMT/RATE
                r = dose.rate
                d = dose.amount / r  # infusion duration
                t0 = dose.time
                # During infusion (t0 ≤ t ≤ t0+D)
                dt = obs_times - t0
                during = (dt >= 0) & (dt <= d)
                amounts[during] += r / k * (1 - np.exp(-k * dt[during]))
                # After infusion (t > t0+D)
                after = dt > d
                amounts[after] += r / k * (1 - np.exp(-k * d)) * np.exp(-k * (dt[after] - d))

        # Concentration = Amount / Volume
        ipred = amounts / v

        return PKSolution(
            times=obs_times.copy(),
            amounts=amounts.reshape(-1, 1),
            ipred=ipred,
        )
