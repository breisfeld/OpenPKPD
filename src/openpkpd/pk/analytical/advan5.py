"""
ADVAN5 — General N-Compartment Linear Model.

Analytical solution for an arbitrary N-compartment linear system of ODEs:

    dA/dt = M @ A

where M is the N×N rate matrix constructed from user-supplied Kij rate constants.
The solution uses eigendecomposition: A(t) = P @ diag(exp(-λ·t)) @ P⁻¹ @ A₀.

N is inferred automatically from the Kij/Ki0 keys present in pk_params:

    Key pattern    Meaning
    ─────────────  ──────────────────────────────────────────────────────────
    K{i}{j}        Transfer rate FROM compartment i TO compartment j
                   (single digits i, j ∈ 1–9, i ≠ j)
    K{i}0          Elimination rate FROM compartment i  (i ∈ 1–9)
    K              Alias for K10 (elimination from compartment 1)

N = max compartment index found across all Kij / Ki0 keys.
Compartment indices are limited to 1–9; use ADVAN6/8 for N > 9.

Output compartment defaults to 1 (central). Override via the output_compartment
constructor argument. Volume is looked up as V{output_cmt} then "V".

TRANS1 (identity, micro rate constants) is the only supported TRANS code.
Superposition handles multiple dosing; bolus and zero-order infusions into
any compartment are supported.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.errors import PKError

# Compiled regex patterns for Kij / Ki0 key parsing (case-normalised before matching)
_RE_KIJ = re.compile(r"^K([1-9])([1-9])$")  # K{i}{j}: transfer from i to j
_RE_KI0 = re.compile(r"^K([1-9])0$")         # K{i}0:   elimination from i


def _infer_n_and_parse_rates(
    pk_params: dict[str, float],
) -> tuple[int, dict[tuple[int, int], float], dict[int, float]]:
    """
    Parse Kij and Ki0 keys from pk_params to infer N and build rate dicts.

    Accepted key patterns (keys are upper-cased before matching):
      K{i}{j}  — transfer rate FROM i TO j  (i, j ∈ 1–9, i ≠ j)
      K{i}0    — elimination FROM compartment i  (i ∈ 1–9)
      K        — alias for K10 (elimination from compartment 1)

    Returns:
        n   : number of compartments (max index seen across all keys)
        kij : mapping (i, j) → rate (FROM i TO j)
        ki0 : mapping i → elimination rate FROM compartment i

    Raises:
        PKError if no rate-constant keys are found (n == 0).
    """
    kij: dict[tuple[int, int], float] = {}
    ki0: dict[int, float] = {}
    indices: set[int] = set()

    for raw_key, value in pk_params.items():
        key = raw_key.upper()

        # Plain "K" → alias for K10 (elimination from compartment 1)
        if key == "K":
            ki0[1] = float(value)
            indices.add(1)
            continue

        m = _RE_KIJ.match(key)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            if i != j:
                kij[(i, j)] = float(value)
                indices.update((i, j))
            continue

        m = _RE_KI0.match(key)
        if m:
            i = int(m.group(1))
            ki0[i] = float(value)
            indices.add(i)

    if not indices:
        raise PKError(
            "ADVAN5 requires at least one rate-constant key (Kij or Ki0). "
            "No matching keys found in pk_params. "
            f"Got: {list(pk_params.keys())}"
        )

    n = max(indices)
    return n, kij, ki0


def _build_rate_matrix(
    n: int,
    kij: dict[tuple[int, int], float],
    ki0: dict[int, float],
) -> np.ndarray:
    """
    Build the N×N rate matrix M for the compartmental ODE system dA/dt = M @ A.

    Convention (matches ADVAN11):
      For each Kij (transfer FROM i TO j):
        M[j-1, i-1] += Kij   (flow into compartment j from i)
        M[i-1, i-1] -= Kij   (outflow from compartment i)
      For each Ki0 (elimination FROM i):
        M[i-1, i-1] -= Ki0   (outflow from i to outside)

    Returns:
        N×N numpy float64 array.
    """
    M = np.zeros((n, n), dtype=float)
    for (i, j), rate in kij.items():
        M[j - 1, i - 1] += rate
        M[i - 1, i - 1] -= rate
    for i, rate in ki0.items():
        M[i - 1, i - 1] -= rate
    return M


def _eigendecomp_n(
    M: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Eigendecompose an N×N rate matrix M.

    For a stable compartment system, eigenvalues of M are ≤ 0.
    Returns lam = -eigenvalues.real (positive decay rates), sorted ascending,
    along with the eigenvector matrix P and its inverse P_inv such that
    M = P @ diag(-lam) @ P_inv.

    Falls back to np.linalg.pinv if M is degenerate (repeated eigenvalues).
    """
    eigenvalues, P = np.linalg.eig(M)
    idx = np.argsort(eigenvalues.real)   # ascending: most negative first
    eigenvalues = eigenvalues[idx]
    P = P[:, idx]
    lam = -eigenvalues.real              # positive decay rates, shape (N,)
    try:
        P_inv = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        P_inv = np.linalg.pinv(P)
    return lam, P.real, P_inv.real



def _safe_mode_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide modal coefficients safely, returning 0 where |den| < 1e-14."""
    return np.divide(
        num,
        den,
        out=np.zeros_like(num, dtype=float),
        where=np.abs(den) > 1e-14,
    )


def _propagate_ncmt(
    a0: np.ndarray,
    lam: np.ndarray,
    P: np.ndarray,
    P_inv: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Propagate N-compartment system from initial state a0 over time steps dt.

    A(t) = P @ diag(exp(-λ·t)) @ P⁻¹ @ a0

    Args:
        a0    : Initial state vector, shape (N,).
        lam   : Positive decay rates, shape (N,).
        P     : Right-eigenvector matrix, shape (N, N).
        P_inv : Inverse of P, shape (N, N).
        dt    : Positive time offsets, shape (n_times,).

    Returns:
        Array of shape (n_times, N) with non-negative compartment amounts.
    """
    c = P_inv @ a0                              # modal coordinates, shape (N,)
    exp_lam = np.exp(-np.outer(dt, lam))        # (n_times, N)
    weighted_exp = exp_lam * c[np.newaxis, :]   # (n_times, N)
    amounts = (P @ weighted_exp.T).T            # (n_times, N)
    return np.maximum(amounts, 0.0)


def _bolus_response_n(
    dose: float,
    dose_cmt: int,
    n: int,
    lam: np.ndarray,
    P: np.ndarray,
    P_inv: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Compartment amounts for a single IV bolus into compartment dose_cmt.

    Sets a0[dose_cmt-1] = dose, all others = 0, then propagates.

    Args:
        dose     : Bolus amount.
        dose_cmt : 1-based compartment receiving the dose.
        n        : Total number of compartments.
        lam, P, P_inv : Eigendecomposition of rate matrix.
        dt       : Positive time offsets from dose time, shape (n_times,).

    Returns:
        Array of shape (n_times, N).
    """
    a0 = np.zeros(n)
    a0[dose_cmt - 1] = dose
    return _propagate_ncmt(a0, lam, P, P_inv, dt)


def _infusion_response_n(
    rate: float,
    duration: float,
    dose_cmt: int,
    n: int,
    lam: np.ndarray,
    P: np.ndarray,
    P_inv: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Compartment amounts for a constant-rate infusion into compartment dose_cmt.

    The forcing vector b has b[dose_cmt-1] = rate, all others = 0.
    The particular solution is -M⁻¹ @ b = P @ diag(1/λ) @ P⁻¹ @ b.

    During infusion [0, duration]:
        A(t) = Σ_k P[:,k] · (c_p[k]/λ[k]) · (1 − exp(−λ[k]·t))
    After infusion [duration, ∞):
        Evaluate A(duration), then free decay via _propagate_ncmt.

    Args:
        rate     : Infusion rate (amount / time).
        duration : Total infusion duration.
        dose_cmt : 1-based compartment receiving the infusion.
        n        : Total number of compartments.
        lam, P, P_inv : Eigendecomposition of rate matrix.
        dt       : Positive time offsets from infusion start, shape (n_times,).

    Returns:
        Array of shape (n_times, N).
    """
    b = np.zeros(n)
    b[dose_cmt - 1] = rate

    c_p = P_inv @ b                              # forcing in eigenbasis, shape (N,)
    c_p_over_lam = _safe_mode_divide(c_p, lam)  # shape (N,)

    n_times = len(dt)
    amounts = np.zeros((n_times, n))

    during = dt <= duration
    after = dt > duration

    if np.any(during):
        t_on = dt[during]
        exp_lam = np.exp(-np.outer(t_on, lam))                    # (n_during, N)
        weighted = c_p_over_lam[np.newaxis, :] * (1.0 - exp_lam)  # (n_during, N)
        amounts[during, :] = (P @ weighted.T).T

    if np.any(after):
        exp_lam_end = np.exp(-lam * duration)              # (N,)
        c_end = c_p_over_lam * (1.0 - exp_lam_end)        # modal coords of A(duration)
        a_end = P @ c_end                                  # (N,)
        t_post = dt[after] - duration
        amounts[after, :] = _propagate_ncmt(a_end, lam, P, P_inv, t_post)

    return np.maximum(amounts, 0.0)


class ADVAN5(PKSubroutine):
    """
    General N-Compartment Linear Model (ADVAN5).

    Analytically solves an arbitrary N-compartment linear system of ODEs
    via eigendecomposition of the N×N rate matrix. N is inferred from the
    Kij/Ki0 keys supplied in pk_params.

    TRANS1 (micro rate constants) is the only supported TRANS code.

    Parameters expected in pk_params:
        K{i}{j}  Transfer rate FROM compartment i TO compartment j
                 (i, j ∈ 1–9, i ≠ j)
        K{i}0    Elimination rate FROM compartment i  (i ∈ 1–9)
        K        Alias for K10 (elimination from compartment 1)
        V{n}     Volume of output compartment n  (e.g. V1, V2)
        V        Fallback volume when V{n} is absent

    Constructor Args:
        output_compartment (int): 1-based compartment used for IPRED. Default = 1.

    Notes:
        - Compartment indices are limited to 1–9. Use ADVAN6/8 for N > 9.
        - Dosing into any compartment is supported via DoseEvent.compartment.
        - Superposition is used for multiple doses (bolus and infusion).
        - ALAG and bioavailability (F) must be applied to DoseEvent amounts
          before calling solve(), consistent with other analytical ADVANs.
    """

    n_compartments = 0   # dynamic: inferred from pk_params at solve time
    advan = 5

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
        """
        Solve the N-compartment linear model for a single subject.

        Args:
            pk_params    : Dict with Kij/Ki0 rate constants and volume (V/V{n}).
            dose_events  : List of DoseEvent objects (bolus or infusion).
            obs_times    : 1-D array of observation times.
            pk_callable  : Unused (params already evaluated).
            des_callable : Unused.

        Returns:
            PKSolution with amounts shape (n_times, N) and IPRED = A[out_cmt]/V.
        """
        # Infer N and parse rate constants
        n, kij, ki0 = _infer_n_and_parse_rates(pk_params)

        # Validate output compartment
        out_cmt = self._output_cmt
        if not (1 <= out_cmt <= n):
            raise PKError(
                f"ADVAN5: output_compartment={out_cmt} is out of range "
                f"for an {n}-compartment model (valid: 1–{n})"
            )

        # Resolve volume for the output compartment
        volume = pk_params.get(f"V{out_cmt}")
        if volume is None:
            volume = pk_params.get("V")
        if volume is None:
            raise PKError(
                f"ADVAN5: volume not found. "
                f"Expected 'V{out_cmt}' or 'V' in pk_params. "
                f"Got keys: {list(pk_params.keys())}"
            )
        if float(volume) <= 0.0:
            raise PKError(f"ADVAN5: volume must be > 0, got {volume!r}")

        # Build rate matrix and eigendecompose once per subject
        M = _build_rate_matrix(n, kij, ki0)
        lam, P, P_inv = _eigendecomp_n(M)

        # Accumulate contributions via superposition
        doses = [e for e in dose_events if not e.reset]
        n_times = len(obs_times)
        amounts = np.zeros((n_times, n))

        for dose in doses:
            dose_cmt = dose.compartment if dose.compartment else 1
            if not (1 <= dose_cmt <= n):
                raise PKError(
                    f"ADVAN5: dose compartment {dose_cmt} is out of range "
                    f"for an {n}-compartment model (valid: 1–{n})"
                )

            dt = obs_times - dose.time
            mask = dt > 0
            if not np.any(mask):
                continue

            if dose.is_bolus:
                da = _bolus_response_n(
                    dose.amount, dose_cmt, n, lam, P, P_inv, dt[mask]
                )
            else:
                dur = dose.amount / dose.rate
                da = _infusion_response_n(
                    dose.rate, dur, dose_cmt, n, lam, P, P_inv, dt[mask]
                )

            amounts[mask, :] += da

        ipred = amounts[:, out_cmt - 1] / float(volume)

        return PKSolution(
            times=obs_times.copy(),
            amounts=amounts,
            ipred=ipred,
        )
