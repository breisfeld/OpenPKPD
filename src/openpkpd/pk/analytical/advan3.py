"""
ADVAN3 — 2-compartment IV bolus model.

Biexponential solution via eigenvalue decomposition:
    A1(t) = DOSE * [ α_+ * exp(-λ1*t) + α_- * exp(-λ2*t) ]

where λ1, λ2 are eigenvalues of the 2×2 rate matrix:
    M = [-(K+K12),  K21;
          K12,     -K21]

Eigenvalues:
    S = K + K12 + K21
    D = sqrt(S^2 - 4*K*K21)
    λ1 = (S - D) / 2   (slower)
    λ2 = (S + D) / 2   (faster)

Coefficients for A1 (central compartment):
    A1(t) = DOSE * [(K21 - λ1)/(λ2-λ1) * exp(-λ1*t) + (λ2 - K21)/(λ2-λ1) * exp(-λ2*t)]
    A2(t) = DOSE * K12/(λ2-λ1) * [exp(-λ1*t) - exp(-λ2*t)]
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.errors import PKError


def _eigenvalues(k: float, k12: float, k21: float) -> tuple[float, float]:
    """Compute λ1 (slow), λ2 (fast) eigenvalues of 2-cmt rate matrix."""
    s = k + k12 + k21
    d = np.sqrt(max(s * s - 4 * k * k21, 0.0))
    lam1 = (s - d) / 2  # smaller = slower
    lam2 = (s + d) / 2  # larger = faster
    return lam1, lam2


def _one_minus_exp_over_lambda(lam: float, dt: np.ndarray) -> np.ndarray:
    """Stable evaluation of (1 - exp(-lam * dt)) / lam with the lam→0 limit."""
    if abs(lam) < 1e-14:
        return dt.astype(float, copy=True)
    return (1.0 - np.exp(-lam * dt)) / lam


def _biexp_central(
    dose: float, k: float, k12: float, k21: float, lam1: float, lam2: float, dt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute A1, A2 for a single IV bolus dose."""
    dl = lam2 - lam1
    if dl < 1e-10:
        # Equal eigenvalues: use limit form
        a1 = dose * np.exp(-lam1 * dt) * (1 - lam1 * dt + k21 * dt)
        a2 = dose * k12 * dt * np.exp(-lam1 * dt)
        return a1, a2

    a1 = dose * ((k21 - lam1) / dl * np.exp(-lam1 * dt) + (lam2 - k21) / dl * np.exp(-lam2 * dt))
    a2 = dose * k12 / dl * (np.exp(-lam1 * dt) - np.exp(-lam2 * dt))
    return a1, a2


def _biexp_central_ss(
    dose: float,
    k: float,
    k12: float,
    k21: float,
    lam1: float,
    lam2: float,
    dt: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """A1, A2 at periodic steady-state for an IV bolus dose (dosing interval tau).

    Per-eigenvalue SS accumulation factors (Rowland & Tozer, Ch. 17):
        A1_ss(t) = D * [(K21-λ1)/(λ2-λ1)*ss1*exp(-λ1*t) + (λ2-K21)/(λ2-λ1)*ss2*exp(-λ2*t)]
        A2_ss(t) = D * K12/(λ2-λ1) * [ss1*exp(-λ1*t) - ss2*exp(-λ2*t)]
    where ss_i = 1/(1-exp(-λ_i*τ)).
    """
    dl = lam2 - lam1
    ss1 = 1.0 / (1.0 - np.exp(-lam1 * tau))
    if dl < 1e-10:
        # Degenerate: scale the single-dose Jordan form by the common ss factor.
        e = np.exp(-lam1 * dt)
        a1 = dose * ss1 * e * (1.0 + (k21 - lam1) * dt)
        a2 = dose * ss1 * k12 * dt * e
        return a1, a2
    ss2 = 1.0 / (1.0 - np.exp(-lam2 * tau))
    a1 = dose * (
        (k21 - lam1) / dl * ss1 * np.exp(-lam1 * dt)
        + (lam2 - k21) / dl * ss2 * np.exp(-lam2 * dt)
    )
    a2 = dose * k12 / dl * (ss1 * np.exp(-lam1 * dt) - ss2 * np.exp(-lam2 * dt))
    return a1, a2


class ADVAN3(PKSubroutine):
    """2-compartment IV model (ADVAN3)."""

    n_compartments = 2
    advan = 3
    output_compartment = 1  # central compartment

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        k = pk_params.get("K")
        k12 = pk_params.get("K12")
        k21 = pk_params.get("K21")
        v1 = pk_params.get("V1")
        if v1 is None:
            v1 = pk_params.get("V")

        # TRANS4: CL, Q, V1, V2 → convert to micro-rate constants
        if any(x is None for x in [k, k12, k21]) and all(
            pk_params.get(p) is not None for p in ("CL", "Q", "V1", "V2")
        ):
            cl = float(pk_params["CL"])
            q = float(pk_params["Q"])
            v1 = float(pk_params["V1"])
            v2 = float(pk_params["V2"])
            k = cl / v1
            k12 = q / v1
            k21 = q / v2
        # TRANS3: CL, Q, V (=V1), V2
        elif any(x is None for x in [k, k12, k21]) and all(
            pk_params.get(p) is not None for p in ("CL", "Q", "V", "V2")
        ):
            cl = float(pk_params["CL"])
            q = float(pk_params["Q"])
            v1 = float(pk_params["V"])
            v2 = float(pk_params["V2"])
            k = cl / v1
            k12 = q / v1
            k21 = q / v2

        if any(x is None for x in [k, k12, k21, v1]):
            raise PKError(
                f"ADVAN3 requires K, K12, K21, V1 (or V); or CL, Q, V1, V2 (TRANS4); "
                f"got params={list(pk_params.keys())}"
            )
        # k, k12, k21, v1 are not None — the PKError raise above guarantees this
        if float(v1) <= 0.0:  # type: ignore[arg-type]
            raise PKError(f"ADVAN3 requires V1/V > 0, got V1={v1}")

        lam1, lam2 = _eigenvalues(k, k12, k21)  # type: ignore[arg-type]

        doses = [e for e in dose_events if not e.reset]
        a1 = np.zeros(len(obs_times))
        a2 = np.zeros(len(obs_times))

        ss_infusion_warned = False
        for dose in doses:
            dt = obs_times - dose.time
            mask = dt > 0  # pre-dose convention

            if dose.is_bolus:
                if dose.ss and dose.ii > 0:
                    da1, da2 = _biexp_central_ss(
                        dose.amount, k, k12, k21, lam1, lam2, dt[mask], dose.ii  # type: ignore[arg-type]
                    )
                else:
                    da1, da2 = _biexp_central(dose.amount, k, k12, k21, lam1, lam2, dt[mask])  # type: ignore[arg-type]
                a1[mask] += da1
                a2[mask] += da2
            else:
                if dose.ss and not ss_infusion_warned:
                    warnings.warn(
                        "ADVAN3: SS=1 with infusion dosing is not yet implemented; "
                        "predictions are computed from a single-dose infusion.",
                        UserWarning,
                        stacklevel=3,
                    )
                    ss_infusion_warned = True
                # Infusion: integrate biexponential over duration
                r = dose.rate
                dur = dose.amount / r
                t_on = dt[mask]

                during = t_on <= dur
                after_mask = t_on > dur

                if np.any(during):
                    da1, da2 = _infusion_biexp(
                        r,
                        k,
                        k12,
                        k21,
                        lam1,
                        lam2,
                        t_on[during],  # type: ignore[arg-type]
                    )
                    idx = np.where(mask)[0][during]
                    a1[idx] += da1
                    a2[idx] += da2

                if np.any(after_mask):
                    # Evaluate at end of infusion, then free decay
                    a1_end, a2_end = _infusion_biexp(r, k, k12, k21, lam1, lam2, np.array([dur]))  # type: ignore[arg-type]
                    t_post = t_on[after_mask] - dur
                    a1_full, a2_full = _propagate_2cmt(
                        a1_end[0],
                        a2_end[0],
                        k,
                        k12,
                        k21,
                        lam1,
                        lam2,
                        t_post,  # type: ignore[arg-type]
                    )
                    idx = np.where(mask)[0][after_mask]
                    a1[idx] += a1_full
                    a2[idx] += a2_full

        ipred = a1 / v1  # type: ignore[operator]
        return PKSolution(
            times=obs_times.copy(),
            amounts=np.column_stack([a1, a2]),
            ipred=ipred,
        )


def _infusion_biexp(
    r: float, k: float, k12: float, k21: float, lam1: float, lam2: float, dt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Central and peripheral compartment amounts during constant-rate infusion."""
    dl = lam2 - lam1
    if dl < 1e-10:
        a1 = r * _one_minus_exp_over_lambda(lam1, dt)
        a2 = np.zeros_like(dt, dtype=float)
        return a1, a2

    f1 = _one_minus_exp_over_lambda(lam1, dt)
    f2 = _one_minus_exp_over_lambda(lam2, dt)

    a1 = r * ((k21 - lam1) / dl * f1 + (lam2 - k21) / dl * f2)
    a2 = r * k12 / dl * (f1 - f2)
    return a1, a2


def _propagate_2cmt(
    a1_0: float,
    a2_0: float,
    k: float,
    k12: float,
    k21: float,
    lam1: float,
    lam2: float,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate 2-cmt system from initial conditions (a1_0, a2_0) over time dt.

    Using matrix exponential eigendecomposition.
    """
    dl = lam2 - lam1
    if dl < 1e-10:
        # Degenerate eigenvalue limit (λ1 = λ2 = λ).
        # Derived by taking dl→0 in the biexponential formulas:
        #   a1 = exp(-λt) * [a1_0*(1 + (k21-λ)*t) + a2_0*k21*t]
        #   a2 = exp(-λt) * [a2_0*(1 + (λ-k21)*t) + a1_0*k12*t]
        e = np.exp(-lam1 * dt)
        a1 = (a1_0 * (1.0 + (k21 - lam1) * dt) + a2_0 * k21 * dt) * e
        a2 = (a2_0 * (1.0 + (lam1 - k21) * dt) + a1_0 * k12 * dt) * e
        return a1, a2

    e1 = np.exp(-lam1 * dt)
    e2 = np.exp(-lam2 * dt)

    a1 = (a1_0 * (k21 - lam1) + a2_0 * k21) / dl * e1 + (a1_0 * (lam2 - k21) - a2_0 * k21) / dl * e2
    a2 = (a1_0 * k12 + a2_0 * (lam2 - k21)) / dl * e1 + (
        -a1_0 * k12 + a2_0 * (k21 - lam1)
    ) / dl * e2

    return a1, a2
