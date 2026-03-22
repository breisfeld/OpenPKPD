"""
ADVAN12 — 3-compartment oral absorption model.

Adds a depot (absorption) compartment to the 3-compartment IV system (ADVAN11).

4 compartments:
  1 — Depot (first-order absorption at rate KA into central)
  2 — Central (output compartment, concentration = A2/V2)
  3 — Peripheral 1 (slow redistribution)
  4 — Peripheral 2 (slow redistribution)

The disposition kinetics (cmts 2, 3, 4) are governed by the same 3x3 rate matrix
as ADVAN11 (using K, K12, K21, K13, K31 where K = CL/V2, K12 = Q2/V2, etc.).

Absorption from depot into central is first-order with rate KA.
The depot decays as A1(t) = F * dose * exp(-KA * t).

The central compartment receives KA * A1(t) as an input; this is treated as an
inhomogeneous forcing term solved analytically by superposition.

Parameters expected in pk_params:
    KA   : Absorption rate constant.
    K    : Elimination rate from central (= CL/V2).
    K12  : Transfer rate central → peripheral 1 (= Q2/V2).
    K21  : Transfer rate peripheral 1 → central (= Q2/V3 with TRANS4 mapping).
    K13  : Transfer rate central → peripheral 2 (= Q3/V2).
    K31  : Transfer rate peripheral 2 → central (= Q3/V4).
    V2   : Volume of central compartment (or V1).
    F1   : Bioavailability fraction (optional, default 1.0).
    K23, K32: Optional inter-peripheral transfers (default 0).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan11 import (
    _eigendecomp,
    _propagate_3cmt,
    _rate_matrix,
)
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.errors import PKError


def _oral_bolus_response(
    dose: float,
    ka: float,
    lam: np.ndarray,
    P: np.ndarray,
    P_inv: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Compute all 4-compartment amounts for an oral bolus dose (into depot).

    The depot decays: A1(t) = dose * exp(-KA * t)
    The forcing on the disposition system is KA * A1(t) = dose * KA * exp(-KA * t)
    into compartment 2 (central, index 1 in the 3-cmt subsystem).

    Particular solution for forcing b(t) = KA * dose * exp(-KA * t) * e_1
    (e_1 = [1, 0, 0] in the 3-cmt subsystem):
        Ap(t) = P @ diag(1/(lam - KA)) @ P_inv @ (dose * KA * e_1) * exp(-KA * t)
        (valid when lam_j ≠ KA for all j)

    Homogeneous solution from IC A(0) = 0:
        A(t) = Ap(0) - Ap(t) + Ap(0) * [e^{-lam*t} - 1] component-wise

    Full solution: A(t) = P @ [c_h * exp(-lam*t) + c_p * exp(-KA*t)]
    where:
        c_p = P_inv @ (dose * KA * e_1) / (lam - KA)   (particular)
        c_h = -c_p                                       (from A(0) = 0)

    So: A(t) = P @ c_p @ [exp(-KA*t) - exp(-lam*t)]

    Args:
        dose:    Bolus amount entering depot.
        ka:      Absorption rate constant.
        lam:     Positive decay rates of 3-cmt system (shape 3).
        P, P_inv: Eigenvector matrices.
        dt:      Positive time offsets from dose time.

    Returns:
        Array of shape (len(dt), 4): rows = [A1, A2, A3, A4].
    """
    n = len(dt)
    amounts_4 = np.zeros((n, 4))

    # Depot (compartment 1): A1(t) = dose * exp(-KA * t)
    amounts_4[:, 0] = dose * np.exp(-ka * dt)

    # Forcing vector in 3-cmt space (into central = index 0 of subsystem)
    b_vec = np.array([dose * ka, 0.0, 0.0])
    c_p_unscaled = P_inv @ b_vec  # shape (3,)

    # Avoid division by zero when ka ≈ lam_j
    lam_diff = lam - ka  # shape (3,)
    small = np.abs(lam_diff) < 1e-8
    safe_diff = np.where(small, 1e-8, lam_diff)

    c_p = c_p_unscaled / safe_diff  # shape (3,)

    # A_disp(t) = P @ c_p * [exp(-ka*t) - exp(-lam*t)]
    exp_ka = np.exp(-ka * dt)[:, np.newaxis]  # (n, 1)
    exp_lam = np.exp(-np.outer(dt, lam))  # (n, 3)

    # For degenerate ka ≈ lam_j, use limit: t * exp(-ka * t)
    if np.any(small):
        for j in np.where(small)[0]:
            # Limit of [exp(-ka*t) - exp(-lam_j*t)] / (lam_j - ka) as lam_j → ka
            # = t * exp(-ka * t)
            # So the contribution is: P[:, j] * c_p_unscaled[j] * t * exp(-ka * t)
            c_p[j] = 0.0  # zero out the problematic term (will be handled separately)
            limit_contrib = c_p_unscaled[j] * dt * np.exp(-ka * dt)  # shape (n,)
            # Add P[:, j] * limit_contrib for compartments 2, 3, 4 (index 1, 2, 3)
            for row in range(3):
                amounts_4[:, row + 1] += P[row, j] * limit_contrib

    # Disposition compartments (indices 1, 2, 3 in 4-cmt output)
    diff_exp = exp_ka - exp_lam  # (n, 3)
    weighted = diff_exp * c_p[np.newaxis, :]  # (n, 3)
    disp_amounts = (P @ weighted.T).T  # (n, 3)

    amounts_4[:, 1:] += np.maximum(disp_amounts, 0.0)
    amounts_4 = np.maximum(amounts_4, 0.0)

    return amounts_4


def _oral_infusion_response(
    rate: float,
    duration: float,
    ka: float,
    lam: np.ndarray,
    P: np.ndarray,
    P_inv: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Compute 4-compartment amounts for a zero-order infusion into depot.

    During infusion [0, dur]: constant rate R enters depot.
    After infusion [dur, inf]: depot decays exponentially from state at t=dur.

    Args:
        rate:     Infusion rate (amount/time).
        duration: Infusion duration.
        ka:       Absorption rate constant from depot.
        lam, P, P_inv: Eigendecomposition of 3-cmt rate matrix.
        dt:       Positive time offsets from infusion start.

    Returns:
        Array of shape (len(dt), 4).
    """
    n = len(dt)
    amounts_4 = np.zeros((n, 4))

    def _safe_mode_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(
            num,
            den,
            out=np.zeros_like(num, dtype=float),
            where=np.abs(den) > 1e-14,
        )

    during = dt <= duration
    after = dt > duration

    # --- During infusion ---
    if np.any(during):
        t_on = dt[during]

        # Depot: constant rate infusion with first-order removal at KA
        # dA1/dt = rate - KA * A1 → A1(t) = (rate/KA) * (1 - exp(-KA*t))
        a1_during = rate / ka * (1.0 - np.exp(-ka * t_on))
        amounts_4[during, 0] = a1_during

        # Disposition compartments: driven by KA * A1(t) into central
        # = rate * (1 - exp(-KA*t)) enters central
        # Use superposition: _infusion_response gives response to direct infusion into central,
        # but here the "effective rate" into central varies in time.
        # Correct approach: integrate analytically using the inhomogeneous ODE.
        #
        # The forcing is f(t) = KA * A1(t) = rate * (1 - exp(-KA*t)) into central.
        # This splits into:
        #   f1(t) = rate * (constant infusion at rate `rate` into central)
        #   f2(t) = -rate * exp(-KA*t) * [KA * exp(-KA*t) term] (exponential forcing)
        #
        # Particular solution for constant rate into central: use _infusion_response
        # but for rate = rate into central (not depot).
        # Particular solution for exponential forcing: similar to oral bolus.

        # Part 1: response to constant rate `rate` infusion into central
        # (using the 3-cmt infusion formula with forcing b = [rate, 0, 0])
        b_const = np.array([rate, 0.0, 0.0])
        c_const = P_inv @ b_const
        # Particular solution: Ap = P @ (c_const / lam) * (1 - exp(-lam * t))
        exp_lam = np.exp(-np.outer(t_on, lam))  # (n_during, 3)
        weighted_const = _safe_mode_divide(c_const, lam)[np.newaxis, :] * (1.0 - exp_lam)
        disp1 = (P @ weighted_const.T).T  # (n_during, 3)

        # Part 2: response to -rate * exp(-KA*t) forcing into central
        # (like an oral bolus with dose = rate/KA scaled, uses c_p formula)
        b_exp = np.array([-rate, 0.0, 0.0])  # forcing amplitude
        c_exp_unscaled = P_inv @ b_exp
        lam_diff = lam - ka
        small = np.abs(lam_diff) < 1e-8
        safe_diff = np.where(small, 1e-8, lam_diff)
        c_exp = c_exp_unscaled / safe_diff  # c_p for exponential part

        exp_ka_on = np.exp(-ka * t_on)[:, np.newaxis]  # (n_during, 1)
        # A_exp(t) = P @ c_exp * [exp(-KA*t) - exp(-lam*t)]  (IC A(0)=0)
        diff_exp = exp_ka_on - exp_lam
        weighted_exp = diff_exp * c_exp[np.newaxis, :]
        disp2 = (P @ weighted_exp.T).T  # (n_during, 3)

        disp_during = disp1 + disp2  # (n_during, 3)
        amounts_4[during, 1:] = np.maximum(disp_during, 0.0)

    # --- After infusion ---
    if np.any(after):
        # State at end of infusion: reuse during-infusion solution at t=duration
        np.array([duration])

        # Depot at end of infusion
        a1_end = rate / ka * (1.0 - np.exp(-ka * duration))

        # After infusion, depot evolves as: dA1/dt = -KA * A1  (no more infusion)
        # → A1(t) = a1_end * exp(-KA * (t - dur))
        t_post = dt[after] - duration
        amounts_4[after, 0] = a1_end * np.exp(-ka * t_post)

        # Disposition state at end of infusion
        b_const = np.array([rate, 0.0, 0.0])
        c_const = P_inv @ b_const
        exp_lam_end = np.exp(-lam * duration)
        weighted_const_end = _safe_mode_divide(c_const, lam) * (1.0 - exp_lam_end)
        b_exp = np.array([-rate, 0.0, 0.0])
        c_exp_unscaled = P_inv @ b_exp
        lam_diff = lam - ka
        small = np.abs(lam_diff) < 1e-8
        safe_diff = np.where(small, 1e-8, lam_diff)
        c_exp = c_exp_unscaled / safe_diff
        exp_ka_end = np.exp(-ka * duration)
        weighted_exp_end = c_exp * (exp_ka_end - exp_lam_end)
        a_disp_end = P @ (weighted_const_end + weighted_exp_end)  # shape (3,)

        # Post-infusion: free decay from (a1_end into central) + (a_disp_end in disp cmts)
        # The depot continues to absorb into central at rate KA * a1_end * exp(-KA * t_post)
        # Use _oral_bolus_response with dose = a1_end (depot amount at t=dur)
        disp_post = _propagate_3cmt(a_disp_end, lam, P, P_inv, t_post)  # from disp IC

        # Add absorption from remaining depot
        oral_post = _oral_bolus_response(a1_end, ka, lam, P, P_inv, t_post)
        # oral_post[:,0] = depot evolution (already handled above), oral_post[:,1:] = disp
        disp_from_depot = oral_post[:, 1:]

        amounts_4[after, 1:] = np.maximum(disp_post + disp_from_depot, 0.0)

    return np.maximum(amounts_4, 0.0)


class ADVAN12(PKSubroutine):
    """
    3-compartment oral absorption model (ADVAN12).

    4 compartments: depot (1) → central (2) ↔ peripheral1 (3) ↔ peripheral2 (4).
    Output compartment = 2 (central). Concentration = A2 / V2.

    Parameters expected in pk_params:
        KA   : Absorption rate constant (depot → central).
        K    : Elimination rate from central (= CL/V2).
        K12  : Transfer rate central → peripheral 1.
        K21  : Transfer rate peripheral 1 → central.
        K13  : Transfer rate central → peripheral 2.
        K31  : Transfer rate peripheral 2 → central.
        V2   : Volume of central compartment.
        F1   : Bioavailability (optional, default 1.0).
        K23, K32: Inter-peripheral transfers (optional, default 0).
    """

    n_compartments = 4
    advan = 12
    output_compartment = 2  # central compartment

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
    ) -> PKSolution:
        """
        Solve the 3-compartment oral model for a single subject.

        Args:
            pk_params:    Dict with KA, K, K12, K21, K13, K31, V2 (or V1).
            dose_events:  List of DoseEvent objects (dose into depot, cmt 1).
            obs_times:    1-D array of observation times.
            pk_callable:  Unused.
            des_callable: Unused.

        Returns:
            PKSolution with amounts shape (n_times, 4) and IPRED = A2/V2.
        """
        ka = pk_params.get("KA")
        k = pk_params.get("K")
        k12 = pk_params.get("K12")
        k21 = pk_params.get("K21")
        k13 = pk_params.get("K13")
        k31 = pk_params.get("K31")
        # V2 is the central volume for ADVAN12; also accept V1 or V
        v2 = pk_params.get("V2")
        if v2 is None:
            v2 = pk_params.get("V1")
        if v2 is None:
            v2 = pk_params.get("V")
        f1 = pk_params.get("F1", 1.0)
        k23 = pk_params.get("K23", 0.0)
        k32 = pk_params.get("K32", 0.0)

        if any(x is None for x in [ka, k, k12, k21, k13, k31, v2]):
            missing = [
                name
                for name, val in [
                    ("KA", ka),
                    ("K", k),
                    ("K12", k12),
                    ("K21", k21),
                    ("K13", k13),
                    ("K31", k31),
                    ("V2", v2),
                ]
                if val is None
            ]
            raise PKError(
                f"ADVAN12 requires KA, K, K12, K21, K13, K31, V2; "
                f"missing: {missing}; got params={list(pk_params.keys())}"
            )
        assert ka is not None and k is not None and k12 is not None and k21 is not None
        assert k13 is not None and k31 is not None and v2 is not None

        if float(v2) <= 0.0:
            raise PKError(f"ADVAN12 requires V2/V1/V > 0, got V2={v2}")

        # Build 3-cmt rate matrix for disposition (cmts 2, 3, 4)
        M = _rate_matrix(k, k12, k21, k13, k31, k23, k32)  # type: ignore[arg-type]
        lam, P, P_inv = _eigendecomp(M)

        doses = [e for e in dose_events if not e.reset]
        n_times = len(obs_times)
        amounts = np.zeros((n_times, 4))  # [depot, central, periph1, periph2]

        for dose in doses:
            amt = dose.amount * float(f1)
            dt = obs_times - dose.time
            mask = dt > 0

            if not np.any(mask):
                continue

            if dose.is_bolus:
                da = _oral_bolus_response(amt, float(ka), lam, P, P_inv, dt[mask])  # type: ignore[arg-type]
                amounts[mask, :] += da
            else:
                # Zero-order infusion into depot at given rate
                dur = amt / dose.rate
                da = _oral_infusion_response(
                    dose.rate,
                    dur,
                    float(ka),
                    lam,
                    P,
                    P_inv,
                    dt[mask],  # type: ignore[arg-type]
                )
                amounts[mask, :] += da

        # IPRED = central amount / V2
        ipred = amounts[:, 1] / float(v2)  # type: ignore[arg-type]

        return PKSolution(
            times=obs_times.copy(),
            amounts=amounts,
            ipred=ipred,
        )
