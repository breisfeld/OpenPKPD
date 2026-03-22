"""
Parallel absorption model.

Combines zero-order (infusion-like) and first-order (Bateman) absorption
into a 1-compartment disposition model.

Parameters:
    KA:    First-order absorption rate constant.
    D1:    Zero-order duration (time units).
    F1_FO: Fraction of dose absorbed by first-order process.
    F1_ZO: Fraction of dose absorbed by zero-order process.
    CL:    Clearance.
    V:     Volume of distribution.
    F1:    Overall bioavailability (default 1.0).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from openpkpd.pk.base import PKSolution, PKSubroutine


class ParallelAbsorption(PKSubroutine):
    """
    Zero-order + first-order parallel absorption into 1-cmt disposition.

    The total dose is split:
      - First-order fraction: F1_FO * dose absorbed via depot with rate KA
      - Zero-order fraction:  F1_ZO * dose absorbed as constant infusion over D1

    Both contributions are superposed analytically at observation times.

    Analytical solution:
      First-order: C_fo(t) = F_fo*dose*KA/(V*(KA-K)) * (exp(-K*t) - exp(-KA*t))
      Zero-order (during infusion):
          C_zo(t) = R0/(V*K) * (1 - exp(-K*t))  for t <= D1
      Zero-order (post infusion):
          C_zo(t) = R0/(V*K) * (1 - exp(-K*D1)) * exp(-K*(t-D1))  for t > D1
      where R0 = F_zo * dose / D1, K = CL/V
    """

    advan = 0
    n_compartments = 2  # depot + central

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list,
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        """Solve parallel absorption analytically."""
        ka = float(pk_params.get("KA", pk_params.get("ka", 1.0)))
        d1 = float(pk_params.get("D1", pk_params.get("d1", 1.0)))
        f1_fo = float(pk_params.get("F1_FO", pk_params.get("f1_fo", 0.5)))
        f1_zo = float(pk_params.get("F1_ZO", pk_params.get("f1_zo", 0.5)))
        cl = float(pk_params.get("CL", pk_params.get("cl", 1.0)))
        v = float(pk_params.get("V", pk_params.get("v", 10.0)))
        f1 = float(pk_params.get("F1", pk_params.get("f1", 1.0)))

        obs_times = np.asarray(obs_times, dtype=float)
        k = cl / v  # elimination rate constant

        # Collect all dose events
        dose_times: list[float] = []
        dose_amts: list[float] = []
        for de in dose_events:
            if hasattr(de, "time") and hasattr(de, "amount"):
                dose_times.append(float(de.time))
                dose_amts.append(float(de.amount) * f1)

        if not dose_times:
            dose_times = [0.0]
            dose_amts = [0.0]

        ipred = np.zeros(len(obs_times))

        for t_dose, amt in zip(dose_times, dose_amts, strict=False):
            dose_fo = f1_fo * amt
            dose_zo = f1_zo * amt

            for i, t_obs in enumerate(obs_times):
                dt = float(t_obs) - t_dose
                if dt < 0:
                    continue

                # First-order contribution (Bateman equation)
                if abs(ka - k) > 1e-6 * max(ka, k, 1e-10):
                    c_fo = (dose_fo * ka) / (v * (ka - k)) * (np.exp(-k * dt) - np.exp(-ka * dt))
                else:
                    # Equal rates: apply L'Hopital limit
                    c_fo = (dose_fo * ka / v) * dt * np.exp(-k * dt)

                # Zero-order contribution
                if d1 > 0:
                    r0 = dose_zo / d1
                    if dt <= d1:
                        c_zo = (r0 / (v * k)) * (1.0 - np.exp(-k * dt))
                    else:
                        c_zo = (r0 / (v * k)) * (1.0 - np.exp(-k * d1)) * np.exp(-k * (dt - d1))
                else:
                    c_zo = 0.0

                ipred[i] += max(0.0, c_fo + c_zo)

        n_times = len(obs_times)
        amounts = np.zeros((n_times, 2))
        amounts[:, 1] = ipred * v  # central compartment amount

        return PKSolution(
            times=obs_times,
            amounts=amounts,
            ipred=ipred,
        )
