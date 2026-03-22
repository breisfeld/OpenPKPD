"""
ADVAN11 — 3-compartment IV model.

Triexponential solution via numpy eigenvalue decomposition of the 3x3 rate matrix:

    M = [-(K+K12+K13),  K21,          K31        ]
        [ K12,          -(K21+K23),   K32        ]
        [ K13,           K23,         -(K31+K32) ]

For standard NONMEM ADVAN11 with TRANS4 (CL, V1, Q2, V2, Q3, V3):
    K   = CL / V1
    K12 = Q2 / V1
    K21 = Q2 / V2
    K13 = Q3 / V1
    K31 = Q3 / V3
    K23 = K32 = 0

The three eigenvalues λ1 ≤ λ2 ≤ λ3 (all positive for a stable system) are computed
from the characteristic polynomial using numpy.linalg.eig on the negated rate matrix.

Superposition is used for multiple doses (bolus and zero-order infusion).
Output compartment = 1 (central). Volume = V1.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.errors import PKError


def _rate_matrix(
    k: float,
    k12: float,
    k21: float,
    k13: float,
    k31: float,
    k23: float = 0.0,
    k32: float = 0.0,
) -> np.ndarray:
    """
    Build the 3x3 rate matrix M for the 3-compartment system.

    The ODE is dA/dt = M @ A, so M has negative diagonal entries.

    Args:
        k:   Elimination rate from central (=CL/V1).
        k12: Transfer rate from central to peripheral 1.
        k21: Transfer rate from peripheral 1 to central.
        k13: Transfer rate from central to peripheral 2.
        k31: Transfer rate from peripheral 2 to central.
        k23: Transfer from peripheral 1 to peripheral 2 (usually 0).
        k32: Transfer from peripheral 2 to peripheral 1 (usually 0).

    Returns:
        3x3 numpy array (the rate matrix M).
    """
    return np.array(
        [
            [-(k + k12 + k13), k21, k31],
            [k12, -(k21 + k23), k32],
            [k13, k23, -(k31 + k32)],
        ],
        dtype=float,
    )


def _eigendecomp(
    M: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute eigendecomposition of rate matrix M.

    Returns eigenvalues (decay rates, positive), eigenvector matrix P,
    and its inverse P_inv such that M = P @ diag(-lam) @ P_inv.

    Args:
        M: 3x3 rate matrix (with negative diagonal).

    Returns:
        (lam, P, P_inv): Positive decay rates, right-eigenvector matrix, its inverse.
    """
    # M has eigenvalues that are <= 0 (system is stable)
    # We want decay rates lam_j > 0, so eigenvalues of M = -lam_j
    eigenvalues, P = np.linalg.eig(M)

    # Sort by eigenvalue (ascending: most negative first = fastest decay last)
    idx = np.argsort(eigenvalues.real)  # ascending: [-lam3, -lam2, -lam1]
    eigenvalues = eigenvalues[idx]
    P = P[:, idx]

    # Decay rates (positive)
    lam = -eigenvalues.real  # shape (3,)

    try:
        P_inv = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        # Degenerate: add tiny perturbation to break degeneracy
        P_inv = np.linalg.pinv(P)

    return lam, P.real, P_inv.real


def _safe_mode_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide modal coefficients safely, using 0 where both numerator and denominator vanish."""
    return np.divide(
        num,
        den,
        out=np.zeros_like(num, dtype=float),
        where=np.abs(den) > 1e-14,
    )


def _propagate_3cmt(
    a0: np.ndarray,
    lam: np.ndarray,
    P: np.ndarray,
    P_inv: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Propagate 3-cmt system from initial state a0 over time steps dt.

    Uses matrix exponential: A(t) = P @ diag(exp(-lam*t)) @ P_inv @ a0

    Args:
        a0:   Initial state vector, shape (3,).
        lam:  Decay rate eigenvalues, shape (3,).
        P:    Right-eigenvector matrix, shape (3,3).
        P_inv: Inverse of P, shape (3,3).
        dt:   Time steps (positive), shape (n_times,).

    Returns:
        Array of shape (n_times, 3) with compartment amounts at each dt.
    """
    # c = P_inv @ a0: coordinates in eigenvector basis, shape (3,)
    c = P_inv @ a0

    # exp_lam[i, j] = exp(-lam[j] * dt[i]), shape (n_times, 3)
    exp_lam = np.exp(-np.outer(dt, lam))  # (n_times, 3)

    # Scale by c: weighted_exp[i, j] = c[j] * exp(-lam[j] * dt[i]), shape (n_times, 3)
    weighted_exp = exp_lam * c[np.newaxis, :]  # broadcast c over time axis

    # Transform back to compartment space: A(t) = P @ weighted_exp.T, shape (3, n_times)
    # Then transpose to (n_times, 3)
    amounts = (P @ weighted_exp.T).T  # shape (n_times, 3)
    return np.maximum(amounts, 0.0)


def _bolus_response(
    dose: float,
    lam: np.ndarray,
    P: np.ndarray,
    P_inv: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Compute compartment amounts for a single IV bolus into compartment 1 (central).

    A(0) = [dose, 0, 0]^T

    Args:
        dose: Bolus dose amount.
        lam:  Eigenvalues (positive decay rates).
        P:    Eigenvector matrix.
        P_inv: Inverse eigenvector matrix.
        dt:   Positive time offsets from dose time.

    Returns:
        Array of shape (len(dt), 3).
    """
    a0 = np.array([dose, 0.0, 0.0])
    return _propagate_3cmt(a0, lam, P, P_inv, dt)


def _infusion_response(
    rate: float,
    duration: float,
    lam: np.ndarray,
    P: np.ndarray,
    P_inv: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Compute compartment amounts for a constant-rate infusion into compartment 1.

    The infusion RHS is M @ A + [rate, 0, 0]^T.
    The particular solution (steady-state) is -M^{-1} @ [rate, 0, 0]^T.
    General solution: A(t) = Ap + exp(M*t) @ (A0 - Ap) where A0=0 at t=0.

    Args:
        rate:     Infusion rate (amount/time).
        duration: Total infusion duration.
        lam, P, P_inv: Eigendecomposition of rate matrix M.
        dt:       Positive time offsets from dose start time.

    Returns:
        Array of shape (len(dt), 3) with amounts during/after infusion.
    """
    # Particular solution: Ap = -M^{-1} @ [rate, 0, 0]
    # Equivalently: Ap = P @ diag(1/lam) @ P_inv @ [rate, 0, 0]
    b = np.array([rate, 0.0, 0.0])

    # c_p = P_inv @ b (forcing vector in eigenbasis)
    c_p = P_inv @ b
    c_p_over_lam = _safe_mode_divide(c_p, lam)

    # Particular solution in compartment space: Ap[j] = sum_k P[j,k] * c_p[k] / lam[k]
    P @ c_p_over_lam  # shape (3,)

    # During infusion (0 <= t <= duration): A(t) = Ap * (1 - exp(-lam * t)) (from A0=0)
    # General: A(t) = Ap + exp(M*t) @ (0 - Ap) = Ap - sum_k (P[:,k] c_p[k]/lam[k]) exp(-lam[k]*t)
    # = sum_k P[:,k] * c_p[k]/lam[k] * (1 - exp(-lam[k]*t))

    n_times = len(dt)
    amounts = np.zeros((n_times, 3))

    # Split into during and after infusion
    during = dt <= duration
    after = dt > duration

    if np.any(during):
        t_on = dt[during]
        exp_lam = np.exp(-np.outer(t_on, lam))  # (n_during, 3)
        # weighted: (n_during, 3), row i = (c_p / lam) * (1 - exp(-lam * t_on[i]))
        weighted = c_p_over_lam[np.newaxis, :] * (1.0 - exp_lam)
        amounts[during, :] = (P @ weighted.T).T

    if np.any(after):
        # State at end of infusion (t = duration)
        exp_lam_end = np.exp(-lam * duration)  # shape (3,)
        c_end = c_p_over_lam * (1.0 - exp_lam_end)  # coords of A(duration) in eigenbasis
        a_end = P @ c_end  # shape (3,)

        # Post-infusion: free decay from a_end
        t_post = dt[after] - duration
        amounts[after, :] = _propagate_3cmt(a_end, lam, P, P_inv, t_post)

    return np.maximum(amounts, 0.0)


class ADVAN11(PKSubroutine):
    """
    3-compartment IV model (ADVAN11).

    Uses numpy eigenvalue decomposition of the 3x3 rate matrix to compute
    the triexponential solution. Supports superposition for multiple doses
    (bolus and zero-order infusion).

    Parameters expected in pk_params:
        K    : Elimination rate from central compartment (CL/V1).
        K12  : Transfer rate from central to peripheral 1 (Q2/V1).
        K21  : Transfer rate from peripheral 1 to central (Q2/V2).
        K13  : Transfer rate from central to peripheral 2 (Q3/V1).
        K31  : Transfer rate from peripheral 2 to central (Q3/V3).
        V1   : Volume of central compartment (for concentration = A1/V1).
        K23  : Transfer from peripheral 1 to peripheral 2 (optional, default 0).
        K32  : Transfer from peripheral 2 to peripheral 1 (optional, default 0).
    """

    n_compartments = 3
    advan = 11
    output_compartment = 1  # central compartment

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        """
        Solve the 3-compartment IV model for a single subject.

        Args:
            pk_params:    Dict with K, K12, K21, K13, K31, V1 (plus optional K23, K32).
            dose_events:  List of DoseEvent objects (bolus or infusion into cmt 1).
            obs_times:    1-D array of times at which to evaluate the solution.
            pk_callable:  Unused (params already evaluated).
            des_callable: Unused.

        Returns:
            PKSolution with amounts shape (n_times, 3) and IPRED = A1/V1.
        """
        k = pk_params.get("K")
        k12 = pk_params.get("K12")
        k21 = pk_params.get("K21")
        k13 = pk_params.get("K13")
        k31 = pk_params.get("K31")
        v1 = pk_params.get("V1")
        if v1 is None:
            v1 = pk_params.get("V")
        k23 = pk_params.get("K23", 0.0) or 0.0
        k32 = pk_params.get("K32", 0.0) or 0.0

        if any(x is None for x in [k, k12, k21, k13, k31, v1]):
            missing = [
                name
                for name, val in [
                    ("K", k),
                    ("K12", k12),
                    ("K21", k21),
                    ("K13", k13),
                    ("K31", k31),
                    ("V1", v1),
                ]
                if val is None
            ]
            raise PKError(
                f"ADVAN11 requires K, K12, K21, K13, K31, V1; "
                f"missing: {missing}; got params={list(pk_params.keys())}"
            )
        assert k is not None and k12 is not None and k21 is not None
        assert k13 is not None and k31 is not None and v1 is not None
        if float(v1) <= 0.0:
            raise PKError(f"ADVAN11 requires V1/V > 0, got V1={v1}")

        # Build and decompose rate matrix
        M = _rate_matrix(k, k12, k21, k13, k31, k23, k32)  # type: ignore[arg-type]
        lam, P, P_inv = _eigendecomp(M)

        # Sum contributions from each non-reset dose event via superposition
        doses = [e for e in dose_events if not e.reset]
        n_times = len(obs_times)
        amounts = np.zeros((n_times, 3))

        for dose in doses:
            dt = obs_times - dose.time
            mask = dt > 0

            if not np.any(mask):
                continue

            if dose.is_bolus:
                da = _bolus_response(dose.amount, lam, P, P_inv, dt[mask])
                amounts[mask, :] += da
            else:
                # Zero-order infusion
                dur = dose.amount / dose.rate
                da = _infusion_response(dose.rate, dur, lam, P, P_inv, dt[mask])
                amounts[mask, :] += da

        ipred = amounts[:, 0] / float(v1)  # type: ignore[arg-type]

        return PKSolution(
            times=obs_times.copy(),
            amounts=amounts,
            ipred=ipred,
        )
