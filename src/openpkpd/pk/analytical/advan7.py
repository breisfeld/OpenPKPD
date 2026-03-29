"""
ADVAN7 — General N-Compartment Linear Model (matrix-exponential solver).

Identical compartment structure to ADVAN5 (user-supplied Kij / Ki0 rate
constants) but uses ``scipy.linalg.expm`` to compute the state-transition
matrix instead of eigendecomposition.  This is more numerically stable for
systems with repeated or near-zero eigenvalues.

The solution at each observation time t > t_dose is::

    A(t) = expm(M * dt) @ A0

where M is the N×N rate matrix built from the Kij / Ki0 keys in pk_params.

Key reuse from ADVAN5:
    _infer_n_and_parse_rates — parse Kij / Ki0 → N, kij_dict, ki0_dict
    _build_rate_matrix       — assemble the N×N matrix M
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import expm as _expm

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan5 import (
    _build_rate_matrix,
    _infer_n_and_parse_rates,
)
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.errors import PKError


def _propagate_expm(
    a0: np.ndarray,
    M: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Return compartment amounts after propagating initial state *a0* by *dt*."""
    return np.maximum(_expm(M * dt) @ a0, 0.0)


def _bolus_expm(
    dose: float,
    dose_cmt: int,
    n: int,
    M: np.ndarray,
    obs_dts: np.ndarray,
) -> np.ndarray:
    """Amounts at *obs_dts* (> 0) after a bolus of *dose* into *dose_cmt*."""
    a0 = np.zeros(n)
    a0[dose_cmt - 1] = dose
    out = np.empty((len(obs_dts), n))
    for k, dt in enumerate(obs_dts):
        out[k] = _propagate_expm(a0, M, dt)
    return out


def _infusion_expm(
    rate: float,
    duration: float,
    dose_cmt: int,
    n: int,
    M: np.ndarray,
    obs_dts: np.ndarray,
) -> np.ndarray:
    """Amounts at *obs_dts* (> 0) after a zero-order infusion into *dose_cmt*.

    Uses the analytical integral: ∫₀ᵗ expm(M*(t-s))*r ds.
    During infusion (dt ≤ duration): integral computed via (expm(M*dt) - I) @ M⁻¹ r
    if M is invertible; otherwise small-dt approximation is used.
    After infusion: propagate end-of-infusion state by remaining time.
    """
    r = np.zeros(n)
    r[dose_cmt - 1] = rate
    out = np.empty((len(obs_dts), n))

    # Compute end-of-infusion state once
    a_end = _infusion_state(r, M, n, duration)

    for k, dt in enumerate(obs_dts):
        if dt <= duration:
            out[k] = _infusion_state(r, M, n, dt)
        else:
            # Infusion ended; propagate a_end by (dt - duration)
            out[k] = _propagate_expm(a_end, M, dt - duration)
    return out


def _infusion_state(r: np.ndarray, M: np.ndarray, n: int, dt: float) -> np.ndarray:
    """Compartment amounts at time *dt* during infusion with rate vector *r*."""
    eM = _expm(M * dt)
    try:
        M_inv = np.linalg.inv(M)
        state = (eM - np.eye(n)) @ M_inv @ r
    except np.linalg.LinAlgError:
        # Singular M (e.g. no elimination): use numerical integration
        state = eM @ r * dt
    return np.maximum(state, 0.0)


class ADVAN7(PKSubroutine):
    """
    General N-Compartment Linear Model — matrix-exponential solver (ADVAN7).

    Accepts the same pk_params convention as ADVAN5 (Kij / Ki0 rate constants
    and V / V{n} volume).  Uses ``scipy.linalg.expm`` for the state transition,
    which is more robust to near-degenerate rate matrices.

    Constructor Args:
        output_compartment (int): 1-based compartment used for IPRED. Default = 1.
    """

    n_compartments = 0  # dynamic
    advan = 7

    def __init__(self, output_compartment: int = 1) -> None:
        self._output_cmt = output_compartment

    @property
    def output_compartment(self) -> int:  # type: ignore[override]
        return self._output_cmt

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        n, kij, ki0 = _infer_n_and_parse_rates(pk_params)
        out_cmt = self._output_cmt
        if not (1 <= out_cmt <= n):
            raise PKError(
                f"ADVAN7: output_compartment={out_cmt} out of range for {n}-cmt model"
            )
        volume = pk_params.get(f"V{out_cmt}") or pk_params.get("V")
        if volume is None:
            raise PKError(f"ADVAN7: volume not found in pk_params: {list(pk_params)}")
        if float(volume) <= 0.0:
            raise PKError(f"ADVAN7: volume must be > 0, got {volume!r}")

        M = _build_rate_matrix(n, kij, ki0)
        doses = [e for e in dose_events if not e.reset]
        n_times = len(obs_times)
        amounts = np.zeros((n_times, n))

        for dose in doses:
            dose_cmt = dose.compartment if dose.compartment else 1
            if not (1 <= dose_cmt <= n):
                raise PKError(f"ADVAN7: dose compartment {dose_cmt} out of range")
            dt = obs_times - dose.time
            mask = dt > 0
            if not np.any(mask):
                continue
            if dose.is_bolus:
                da = _bolus_expm(dose.amount, dose_cmt, n, M, dt[mask])
            else:
                dur = dose.amount / dose.rate
                da = _infusion_expm(dose.rate, dur, dose_cmt, n, M, dt[mask])
            amounts[mask, :] += da

        ipred = amounts[:, out_cmt - 1] / float(volume)
        return PKSolution(times=obs_times.copy(), amounts=amounts, ipred=ipred)
