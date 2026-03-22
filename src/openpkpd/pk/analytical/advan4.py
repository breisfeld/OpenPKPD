"""
ADVAN4 — 2-compartment oral absorption model.

Three compartments: depot (cmt 1) → central (cmt 2) ↔ peripheral (cmt 3).

Solution combines ADVAN2 absorption with ADVAN3 disposition:
    A2(t) = F*DOSE * KA/(KA-λ1)/(KA-λ2) * [
               -λ1*(KA-λ2)*exp(-λ1*t)/... + ... ]
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan3 import (
    _eigenvalues,
    _one_minus_exp_over_lambda,
    _propagate_2cmt,
)
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.errors import PKError


class ADVAN4(PKSubroutine):
    """2-compartment oral model (ADVAN4)."""

    n_compartments = 3
    advan = 4
    output_compartment = 2  # central compartment

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        ka = pk_params.get("KA")
        k = pk_params.get("K")
        k12 = pk_params.get("K12")
        k21 = pk_params.get("K21")
        v2 = pk_params.get("V2") or pk_params.get("V1") or pk_params.get("V")
        f1 = pk_params.get("F1", 1.0)

        if any(x is None for x in [ka, k, k12, k21, v2]):
            raise PKError(
                f"ADVAN4 requires KA, K, K12, K21, V2; got params={list(pk_params.keys())}"
            )
        assert (
            ka is not None
            and k is not None
            and k12 is not None
            and k21 is not None
            and v2 is not None
        )

        lam1, lam2 = _eigenvalues(k, k12, k21)  # type: ignore[arg-type]

        doses = [e for e in dose_events if not e.reset]
        a1 = np.zeros(len(obs_times))  # depot
        a2 = np.zeros(len(obs_times))  # central
        a3 = np.zeros(len(obs_times))  # peripheral

        for dose in doses:
            amt = dose.amount * f1
            dt = obs_times - dose.time
            mask = dt > 0  # pre-dose convention
            t = dt[mask]

            if dose.is_bolus:
                da1, da2, da3 = _triexp_oral(amt, ka, k, k12, k21, lam1, lam2, t)  # type: ignore[arg-type]
                a1[mask] += da1
                a2[mask] += da2
                a3[mask] += da3
            else:
                # Infusion into depot
                r = dose.rate
                dur = amt / r
                during = t <= dur
                after = t > dur

                if np.any(during):
                    da1, da2, da3 = _infusion_triexp(r, ka, k, k12, k21, lam1, lam2, t[during])  # type: ignore[arg-type]
                    idx = np.where(mask)[0][during]
                    a1[idx] += da1
                    a2[idx] += da2
                    a3[idx] += da3

                if np.any(after):
                    a1e, a2e, a3e = _infusion_triexp(
                        r, ka, k, k12, k21, lam1, lam2, np.array([dur])
                    )  # type: ignore[arg-type]
                    t_post = t[after] - dur
                    # Free decay of depot
                    a1_post = a1e[0] * np.exp(-ka * t_post)  # type: ignore[index]
                    # Propagate (a2e, a3e) as initial conditions + contribution from decaying depot
                    a2_disp, a3_disp = _propagate_2cmt(
                        a2e[0], a3e[0], k, k12, k21, lam1, lam2, t_post
                    )  # type: ignore[index]
                    # Absorption from remaining depot (a1e is initial depot at t=dur)
                    da1_abs, da2_abs, da3_abs = _triexp_oral(
                        a1e[0], ka, k, k12, k21, lam1, lam2, t_post
                    )  # type: ignore[index]

                    idx = np.where(mask)[0][after]
                    a1[idx] += a1_post
                    a2[idx] += a2_disp + da2_abs
                    a3[idx] += a3_disp + da3_abs

        ipred = a2 / v2  # type: ignore[operator]
        return PKSolution(
            times=obs_times.copy(),
            amounts=np.column_stack([a1, a2, a3]),
            ipred=ipred,
        )


def _triexp_oral(
    dose: float,
    ka: float,
    k: float,
    k12: float,
    k21: float,
    lam1: float,
    lam2: float,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depot, central, peripheral amounts for oral bolus dose."""
    a1 = dose * np.exp(-ka * dt)

    dl = lam2 - lam1
    if abs(dl) < 1e-10:
        # Equal disposition eigenvalues are rare; keep a conservative fallback
        # rather than introducing a brittle closed-form limit.
        a2 = dose * ka / (ka - k) * (np.exp(-k * dt) - np.exp(-ka * dt))
        a3 = np.zeros_like(dt)
        return a1, a2, a3

    h1 = _decay_difference(lam1, ka, dt)
    h2 = _decay_difference(lam2, ka, dt)
    c1 = (k21 - lam1) / dl
    c2 = (lam2 - k21) / dl

    # Exact convolution of depot absorption with the ADVAN3 impulse response.
    a2 = dose * ka * (c1 * h1 + c2 * h2)
    a3 = dose * ka * k12 / dl * (h1 - h2)

    return a1, a2, a3


def _infusion_triexp(
    r: float, ka: float, k: float, k12: float, k21: float, lam1: float, lam2: float, dt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Amounts during constant-rate infusion into depot."""
    a1 = r * _one_minus_exp_over_lambda(ka, dt)

    dl = lam2 - lam1
    if abs(dl) < 1e-10:
        a2 = (
            r
            / ka
            * (
                ka / (ka - k) * (1 - np.exp(-k * dt)) / k
                - ka / (ka - k) * (1 - np.exp(-ka * dt)) / ka
            )
        )
        a3 = np.zeros_like(dt)
        return a1, a2, a3

    i1 = _integrated_decay_difference(lam1, ka, dt)
    i2 = _integrated_decay_difference(lam2, ka, dt)
    c1 = (k21 - lam1) / dl
    c2 = (lam2 - k21) / dl

    a2 = r * ka * (c1 * i1 + c2 * i2)
    a3 = r * ka * k12 / dl * (i1 - i2)
    return a1, a2, a3


def _decay_difference(a: float, b: float, dt: np.ndarray) -> np.ndarray:
    """Stable evaluation of (exp(-a t) - exp(-b t)) / (b - a)."""
    if abs(b - a) < 1e-10:
        return dt * np.exp(-a * dt)
    return (np.exp(-a * dt) - np.exp(-b * dt)) / (b - a)


def _integrated_decay_difference(a: float, b: float, dt: np.ndarray) -> np.ndarray:
    """Stable evaluation of ∫₀ᵗ (exp(-a u) - exp(-b u)) / (b - a) du."""
    if abs(b - a) < 1e-10:
        if abs(a) < 1e-12:
            return 0.5 * dt**2
        return (1.0 - np.exp(-a * dt) * (1.0 + a * dt)) / (a**2)
    return (_one_minus_exp_over_lambda(a, dt) - _one_minus_exp_over_lambda(b, dt)) / (b - a)
