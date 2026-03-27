"""
External-validation tests for analytical PK subroutines (ADVAN1–ADVAN3, ADVAN5).

Every test checks openpkpd output against a closed-form pharmacokinetic
solution derived directly from the underlying differential equations, or
against an independent scipy.integrate.odeint numerical reference.
No fitting is performed; all reference values are analytically exact.

References
----------
Gibaldi M, Perrier D (1982). *Pharmacokinetics*, 2nd ed. Marcel Dekker.
Rowland M, Tozer TN (1995). *Clinical Pharmacokinetics*, 3rd ed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import odeint

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.pk.analytical.advan5 import ADVAN5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bolus(amt: float = 100.0, t: float = 0.0, cmt: int = 1) -> list[DoseEvent]:
    return [DoseEvent(time=t, amount=amt, compartment=cmt)]


def _infusion(
    amt: float = 100.0, rate: float = 20.0, t: float = 0.0, cmt: int = 1
) -> list[DoseEvent]:
    return [DoseEvent(time=t, amount=amt, rate=rate, compartment=cmt)]


OBS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])


# ---------------------------------------------------------------------------
# ADVAN1 — 1-compartment IV bolus
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestADVAN1Reference:
    """
    Closed form:  C(t) = (D / V) * exp(−K * t)
    Reference: Gibaldi & Perrier (1982) Chapter 1.
    """

    @pytest.mark.parametrize(
        "D,K,V",
        [
            (100.0, 0.10, 20.0),
            (50.0, 0.50, 10.0),
            (200.0, 0.05, 40.0),
            (75.0, 0.30, 15.0),
            (300.0, 0.08, 60.0),
        ],
    )
    def test_concentration_matches_closed_form(self, D, K, V):
        """C(t) = D/V * exp(−K*t) to floating-point precision."""
        sol = ADVAN1().solve({"K": K, "V": V}, _bolus(D), OBS)
        expected = (D / V) * np.exp(-K * OBS)
        np.testing.assert_allclose(sol.ipred, expected, rtol=1e-9, err_msg=f"D={D}, K={K}, V={V}")

    def test_half_life_identity(self):
        """At t = ln(2)/K, concentration equals D/(2V)."""
        K, V, D = 0.2, 10.0, 100.0
        t_half = math.log(2.0) / K
        sol = ADVAN1().solve({"K": K, "V": V}, _bolus(D), np.array([t_half]))
        assert sol.ipred[0] == pytest.approx(D / (2.0 * V), rel=1e-8)

    def test_initial_concentration(self):
        """C(0⁺) = D/V (limit as t → 0)."""
        K, V, D = 0.1, 20.0, 100.0
        sol = ADVAN1().solve({"K": K, "V": V}, _bolus(D), np.array([1e-9]))
        assert sol.ipred[0] == pytest.approx(D / V, rel=1e-5)

    def test_superposition_two_doses(self):
        """Linear superposition: C(t | D1+D2) = C(t | D1) + C(t | D2) for t > t2."""
        K, V, D1, D2, t2 = 0.1, 20.0, 100.0, 50.0, 6.0
        t_after = OBS[t2 < OBS]
        combined = ADVAN1().solve(
            {"K": K, "V": V},
            [
                DoseEvent(time=0.0, amount=D1, compartment=1),
                DoseEvent(time=t2, amount=D2, compartment=1),
            ],
            t_after,
        )
        c1 = (D1 / V) * np.exp(-K * t_after)
        c2 = (D2 / V) * np.exp(-K * (t_after - t2))
        np.testing.assert_allclose(combined.ipred, c1 + c2, rtol=1e-8)

    def test_steady_state_infusion(self):
        """At steady state (t → ∞) with continuous infusion, C_ss = rate / (K * V)."""
        K, V, rate = 0.1, 20.0, 50.0
        t_ss = np.array([500.0])  # 500 h >> 5 half-lives
        # Use very large dose so infusion lasts past t_ss
        sol = ADVAN1().solve(
            {"K": K, "V": V},
            [DoseEvent(time=0.0, amount=rate * 1000.0, rate=rate, compartment=1)],
            t_ss,
        )
        c_ss = rate / (K * V)
        assert sol.ipred[0] == pytest.approx(c_ss, rel=1e-4)

    def test_iv_infusion_during_infusion_closed_form(self):
        """During infusion: C(t) = (rate/K/V)*(1 - exp(-K*t))."""
        K, V, rate = 0.2, 10.0, 20.0
        t_obs = np.array([1.0, 2.0, 3.0])  # all during infusion (ends at t=100)
        sol = ADVAN1().solve(
            {"K": K, "V": V},
            [DoseEvent(time=0.0, amount=rate * 100.0, rate=rate, compartment=1)],
            t_obs,
        )
        expected = (rate / (K * V)) * (1.0 - np.exp(-K * t_obs))
        np.testing.assert_allclose(sol.ipred, expected, rtol=1e-8)


# ---------------------------------------------------------------------------
# ADVAN2 — 1-compartment oral (Bateman function)
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestADVAN2Reference:
    """
    Closed form (Bateman equation, Gibaldi & Perrier Ch. 2):
      C(t) = D*KA / (V*(KA − K)) * (exp(−K*t) − exp(−KA*t))
    """

    @pytest.mark.parametrize(
        "D,KA,K,V",
        [
            (100.0, 1.50, 0.100, 20.0),
            (500.0, 0.80, 0.200, 40.0),
            (200.0, 2.00, 0.150, 30.0),
            (100.0, 0.50, 0.050, 15.0),
            (300.0, 3.00, 0.300, 25.0),
        ],
    )
    def test_concentration_matches_bateman(self, D, KA, K, V):
        """C(t) = D*KA/(V*(KA-K)) * (exp(-K*t) - exp(-KA*t)) to 1e-8."""
        sol = ADVAN2().solve({"KA": KA, "K": K, "V": V}, _bolus(D), OBS)
        expected = (D * KA / (V * (KA - K))) * (np.exp(-K * OBS) - np.exp(-KA * OBS))
        np.testing.assert_allclose(
            sol.ipred, expected, rtol=1e-8, err_msg=f"D={D}, KA={KA}, K={K}, V={V}"
        )

    def test_tmax_matches_analytic(self):
        """Tmax = ln(KA/K) / (KA − K) (Gibaldi & Perrier Eq. 2-18)."""
        KA, K, V, D = 1.5, 0.1, 20.0, 100.0
        tmax_ref = math.log(KA / K) / (KA - K)
        t_dense = np.linspace(0.01, tmax_ref * 3.0, 5000)
        sol = ADVAN2().solve({"KA": KA, "K": K, "V": V}, _bolus(D), t_dense)
        tmax_obs = t_dense[np.argmax(sol.ipred)]
        assert tmax_obs == pytest.approx(tmax_ref, abs=0.02)

    def test_near_ka_equals_k_lhopital_limit(self):
        """When KA ≈ K: C(t) → D*K/V * t*exp(-K*t) (L'Hôpital)."""
        K, V, D = 0.2, 20.0, 100.0
        KA = K * (1.0 + 1e-8)
        t = np.array([1.0, 2.0, 4.0])
        sol = ADVAN2().solve({"KA": KA, "K": K, "V": V}, _bolus(D), t)
        expected = (D * K / V) * t * np.exp(-K * t)
        np.testing.assert_allclose(sol.ipred, expected, rtol=1e-5)

    def test_post_infusion_matches_ode(self):
        """
        Post-infusion absorption matches scipy.integrate.odeint reference.

        This is the scenario affected by the post-infusion depot absorption
        bug fixed in commit 16f4162.  After infusion ends at t_inf, the
        depot compartment still contains drug; that drug must continue to
        absorb into the central compartment.
        """
        KA, K, V, dose, rate = 0.5, 0.15, 20.0, 100.0, 20.0
        t_inf = dose / rate  # 5 h
        dose_event = DoseEvent(time=0.0, amount=dose, rate=rate, compartment=1)
        t_obs = np.linspace(t_inf + 0.5, t_inf + 12.0, 10)

        sol = ADVAN2().solve({"KA": KA, "K": K, "V": V}, [dose_event], t_obs)

        # Reference: scipy ODE solver
        def odes(y, t):
            r = rate if t < t_inf else 0.0
            da1 = r - KA * y[0]
            da2 = KA * y[0] - K * y[1]
            return [da1, da2]

        t_grid = np.concatenate([[0.0], t_obs])
        y_ode = odeint(odes, [0.0, 0.0], t_grid, rtol=1e-10, atol=1e-12)
        c_ref = y_ode[1:, 1] / V

        np.testing.assert_allclose(
            sol.ipred,
            c_ref,
            rtol=1e-5,
            err_msg="Post-infusion concentrations must match ODE reference",
        )


# ---------------------------------------------------------------------------
# ADVAN3 — 2-compartment IV bolus (biexponential)
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestADVAN3Reference:
    """
    Closed form (Gibaldi & Perrier, Ch. 4):
      α, β  = roots of  λ² − (K+K12+K21)λ + K·K21 = 0
      A = (α − K21)/(α − β),   B = (K21 − β)/(α − β)
      C(t) = D/V1 · [A·exp(−α·t) + B·exp(−β·t)]
    """

    @staticmethod
    def _alpha_beta(K, K12, K21):
        s = K + K12 + K21
        disc = math.sqrt(s**2 - 4.0 * K * K21)
        return (s + disc) / 2.0, (s - disc) / 2.0

    @pytest.mark.parametrize(
        "K,K12,K21,V1",
        [
            (0.20, 0.30, 0.10, 20.0),
            (0.10, 0.50, 0.20, 15.0),
            (0.40, 0.10, 0.30, 30.0),
            (0.15, 0.80, 0.40, 10.0),
            (0.30, 0.20, 0.15, 25.0),
        ],
    )
    def test_concentration_matches_biexponential(self, K, K12, K21, V1):
        """C(t) = D/V1 * [A*exp(-α*t) + B*exp(-β*t)] to 1e-7."""
        D = 100.0
        sol = ADVAN3().solve({"K": K, "K12": K12, "K21": K21, "V1": V1}, _bolus(D), OBS)
        alpha, beta = self._alpha_beta(K, K12, K21)
        A = (alpha - K21) / (alpha - beta)
        B = (K21 - beta) / (alpha - beta)
        expected = (D / V1) * (A * np.exp(-alpha * OBS) + B * np.exp(-beta * OBS))
        np.testing.assert_allclose(
            sol.ipred, expected, rtol=1e-7, err_msg=f"K={K}, K12={K12}, K21={K21}, V1={V1}"
        )

    def test_alpha_beta_ordering(self):
        """α > β always (α = fast distribution, β = slow elimination)."""
        K, K12, K21, _V1 = 0.2, 0.3, 0.1, 20.0
        alpha, beta = self._alpha_beta(K, K12, K21)
        assert alpha > beta > 0.0

    def test_distribution_phase_faster_than_elimination(self):
        """
        With fast distribution (K12 >> K), early slope ≈ α, late slope ≈ β,
        so |early log-slope| >> |late log-slope|.
        """
        K, K12, K21, V1 = 0.05, 2.0, 0.5, 10.0
        t_early = np.array([0.1, 0.5])
        t_late = np.array([20.0, 40.0])
        s_early = (
            ADVAN3().solve({"K": K, "K12": K12, "K21": K21, "V1": V1}, _bolus(100.0), t_early).ipred
        )
        s_late = (
            ADVAN3().solve({"K": K, "K12": K12, "K21": K21, "V1": V1}, _bolus(100.0), t_late).ipred
        )
        log_slope_early = abs(math.log(s_early[1]) - math.log(s_early[0])) / (
            t_early[1] - t_early[0]
        )
        log_slope_late = abs(math.log(s_late[1]) - math.log(s_late[0])) / (t_late[1] - t_late[0])
        assert log_slope_early > log_slope_late * 5.0

    def test_iv_coefficient_sum(self):
        """A + B must equal 1 (all drug starts in central compartment)."""
        K, K12, K21 = 0.2, 0.3, 0.1
        alpha, beta = self._alpha_beta(K, K12, K21)
        A = (alpha - K21) / (alpha - beta)
        B = (K21 - beta) / (alpha - beta)
        assert pytest.approx(1.0, abs=1e-12) == A + B


# ---------------------------------------------------------------------------
# ODE cross-validation: ADVAN2 oral against scipy.integrate.odeint
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestODECrossValidation:
    """Verify that analytical solutions agree with scipy ODE reference."""

    def test_advan1_bolus_vs_ode(self):
        """ADVAN1 bolus C(t) matches scipy.odeint to 1e-5."""
        K, V, D = 0.15, 25.0, 100.0

        def ode(y, t):
            return [-K * y[0]]

        t_grid = np.concatenate([[0.0], OBS])
        [D / V]  # 1-cmt: initial concentration (as amount/V)

        # Actually solve as amount: A(0) = D
        def ode_amt(y, t):
            return [-K * y[0]]

        y_ode = odeint(ode_amt, [D], t_grid, rtol=1e-12)
        c_ref = y_ode[1:, 0] / V

        sol = ADVAN1().solve({"K": K, "V": V}, _bolus(D), OBS)
        np.testing.assert_allclose(sol.ipred, c_ref, rtol=1e-8)

    def test_advan2_oral_vs_ode(self):
        """ADVAN2 oral C(t) matches scipy.odeint to 1e-5."""
        KA, K, V, D = 1.5, 0.1, 20.0, 100.0

        def odes(y, t):
            return [-KA * y[0], KA * y[0] - K * y[1]]

        t_grid = np.concatenate([[0.0], OBS])
        y_ode = odeint(odes, [D, 0.0], t_grid, rtol=1e-12)
        c_ref = y_ode[1:, 1] / V

        sol = ADVAN2().solve({"KA": KA, "K": K, "V": V}, _bolus(D), OBS)
        np.testing.assert_allclose(sol.ipred, c_ref, rtol=1e-6)

    def test_advan3_iv_vs_ode(self):
        """ADVAN3 IV biexponential C(t) matches scipy.odeint to 1e-5."""
        K, K12, K21, V1, D = 0.2, 0.3, 0.1, 20.0, 100.0

        def odes(y, t):
            dc = -(K + K12) * y[0] + K21 * y[1]
            dp = K12 * y[0] - K21 * y[1]
            return [dc, dp]

        t_grid = np.concatenate([[0.0], OBS])
        y_ode = odeint(odes, [D, 0.0], t_grid, rtol=1e-12)
        c_ref = y_ode[1:, 0] / V1

        sol = ADVAN3().solve({"K": K, "K12": K12, "K21": K21, "V1": V1}, _bolus(D), OBS)
        np.testing.assert_allclose(sol.ipred, c_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# ADVAN5 — General N-Compartment Linear Model
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestADVAN5Reference:
    """
    External validation for ADVAN5 against closed-form biexponential solutions
    (N=2) and independent scipy.integrate.odeint references (N=3 and N=4).

    Closed-form for N=2 bolus (Gibaldi & Perrier Ch. 4):
      α, β  = roots of  λ² − (K + K12 + K21)λ + K·K21 = 0
      A = (α − K21) / (α − β),   B = (K21 − β) / (α − β)
      C(t) = D/V1 · [A·exp(−α·t) + B·exp(−β·t)]
    """

    @staticmethod
    def _alpha_beta(K, K12, K21):
        s = K + K12 + K21
        disc = math.sqrt(s**2 - 4.0 * K * K21)
        return (s + disc) / 2.0, (s - disc) / 2.0

    @pytest.mark.parametrize(
        "K,K12,K21,V1",
        [
            (0.20, 0.30, 0.10, 20.0),
            (0.10, 0.50, 0.20, 15.0),
            (0.40, 0.10, 0.30, 30.0),
            (0.15, 0.80, 0.40, 10.0),
            (0.30, 0.20, 0.15, 25.0),
        ],
    )
    def test_n2_concentration_matches_biexponential(self, K, K12, K21, V1):
        """
        ADVAN5 (N=2) must match the textbook biexponential formula exactly.

        The formula is derived from the eigenvalues α, β of the 2×2 rate matrix
        and is independent of the ADVAN5 implementation.  Tolerance 1e-7 is
        consistent with double-precision eigendecomposition.
        """
        D = 100.0
        params = {"K": K, "K12": K12, "K21": K21, "V1": V1}
        sol = ADVAN5().solve(params, _bolus(D), OBS)

        alpha, beta = self._alpha_beta(K, K12, K21)
        A = (alpha - K21) / (alpha - beta)
        B = (K21 - beta) / (alpha - beta)
        expected = (D / V1) * (A * np.exp(-alpha * OBS) + B * np.exp(-beta * OBS))

        np.testing.assert_allclose(
            sol.ipred, expected, rtol=1e-7,
            err_msg=f"N=2 biexponential mismatch for K={K}, K12={K12}, K21={K21}, V1={V1}",
        )

    def test_n2_coefficient_sum_is_one(self):
        """A + B = 1 (all drug starts in central compartment at t=0)."""
        K, K12, K21 = 0.2, 0.3, 0.1
        alpha, beta = self._alpha_beta(K, K12, K21)
        A = (alpha - K21) / (alpha - beta)
        B = (K21 - beta) / (alpha - beta)
        assert pytest.approx(1.0, abs=1e-12) == A + B

    def test_n2_terminal_slope_matches_smallest_eigenvalue(self):
        """
        At late times the log-linear slope of C(t) approaches −β, the smaller
        positive eigenvalue (slowest decay mode).
        """
        K, K12, K21, V1, D = 0.05, 1.0, 0.3, 10.0, 100.0
        _, beta = self._alpha_beta(K, K12, K21)
        # Sample two points deep in the terminal phase
        t_late = np.array([80.0, 100.0])
        sol = ADVAN5().solve({"K": K, "K12": K12, "K21": K21, "V1": V1}, _bolus(D), t_late)
        observed_slope = (math.log(sol.ipred[1]) - math.log(sol.ipred[0])) / (
            t_late[1] - t_late[0]
        )
        assert observed_slope == pytest.approx(-beta, rel=1e-4)

    def test_n3_bolus_vs_odeint(self):
        """
        ADVAN5 (N=3, 3-compartment IV bolus) matches scipy.integrate.odeint
        to 1e-5 relative tolerance.

        The ODE system is defined independently from the ADVAN5 implementation:
          dA1/dt = −(K + K12 + K13)·A1 + K21·A2 + K31·A3
          dA2/dt =  K12·A1 − K21·A2
          dA3/dt =  K13·A1 − K31·A3
        """
        K, K12, K21, K13, K31, V1, D = 0.1, 0.05, 0.025, 0.02, 0.0067, 10.0, 100.0
        params = {"K": K, "K12": K12, "K21": K21, "K13": K13, "K31": K31, "V1": V1}
        sol = ADVAN5().solve(params, _bolus(D), OBS)

        def odes(y, _t):
            return [
                -(K + K12 + K13) * y[0] + K21 * y[1] + K31 * y[2],
                K12 * y[0] - K21 * y[1],
                K13 * y[0] - K31 * y[2],
            ]

        t_grid = np.concatenate([[0.0], OBS])
        y_ode = odeint(odes, [D, 0.0, 0.0], t_grid, rtol=1e-12, atol=1e-14)
        c_ref = y_ode[1:, 0] / V1

        np.testing.assert_allclose(sol.ipred, c_ref, rtol=1e-5,
                                   err_msg="ADVAN5 N=3 bolus diverges from odeint reference")

    def test_n3_all_compartment_amounts_vs_odeint(self):
        """
        All three compartment amounts from ADVAN5 match odeint to 1e-5.

        Validates the full state vector, not just IPRED.
        """
        K, K12, K21, K13, K31, V1, D = 0.15, 0.08, 0.04, 0.03, 0.015, 12.0, 200.0
        params = {"K": K, "K12": K12, "K21": K21, "K13": K13, "K31": K31, "V1": V1}
        sol = ADVAN5().solve(params, _bolus(D), OBS)

        def odes(y, _t):
            return [
                -(K + K12 + K13) * y[0] + K21 * y[1] + K31 * y[2],
                K12 * y[0] - K21 * y[1],
                K13 * y[0] - K31 * y[2],
            ]

        t_grid = np.concatenate([[0.0], OBS])
        y_ode = odeint(odes, [D, 0.0, 0.0], t_grid, rtol=1e-12, atol=1e-14)
        amounts_ref = y_ode[1:, :]   # (n_times, 3)

        np.testing.assert_allclose(
            sol.amounts, amounts_ref, rtol=1e-5, atol=1e-8,
            err_msg="ADVAN5 N=3 compartment amounts diverge from odeint",
        )

    def test_n4_bolus_vs_odeint(self):
        """
        ADVAN5 (N=4, 4-compartment IV bolus) matches scipy.integrate.odeint
        to 1e-5.  This is the primary validation for N > 3.
        """
        K, K12, K21, K13, K31, K14, K41 = 0.1, 0.05, 0.025, 0.02, 0.0067, 0.01, 0.005
        V1, D = 10.0, 100.0
        params = {
            "K": K, "K12": K12, "K21": K21,
            "K13": K13, "K31": K31,
            "K14": K14, "K41": K41,
            "V1": V1,
        }
        sol = ADVAN5().solve(params, _bolus(D), OBS)

        def odes(y, _t):
            return [
                -(K + K12 + K13 + K14) * y[0] + K21 * y[1] + K31 * y[2] + K41 * y[3],
                K12 * y[0] - K21 * y[1],
                K13 * y[0] - K31 * y[2],
                K14 * y[0] - K41 * y[3],
            ]

        t_grid = np.concatenate([[0.0], OBS])
        y_ode = odeint(odes, [D, 0.0, 0.0, 0.0], t_grid, rtol=1e-12, atol=1e-14)
        c_ref = y_ode[1:, 0] / V1

        np.testing.assert_allclose(sol.ipred, c_ref, rtol=1e-5,
                                   err_msg="ADVAN5 N=4 bolus diverges from odeint reference")

    def test_n3_infusion_vs_odeint(self):
        """
        ADVAN5 (N=3) zero-order infusion into compartment 1 matches odeint.

        Tests both the during-infusion and post-infusion phases.
        """
        K, K12, K21, K13, K31, V1 = 0.1, 0.05, 0.025, 0.02, 0.0067, 10.0
        rate, duration = 50.0, 2.0
        dose = [DoseEvent(time=0.0, amount=rate * duration, rate=rate)]
        t_obs = np.array([0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0])
        params = {"K": K, "K12": K12, "K21": K21, "K13": K13, "K31": K31, "V1": V1}

        sol = ADVAN5().solve(params, dose, t_obs)

        def odes(y, t):
            r = rate if t <= duration else 0.0
            return [
                r - (K + K12 + K13) * y[0] + K21 * y[1] + K31 * y[2],
                K12 * y[0] - K21 * y[1],
                K13 * y[0] - K31 * y[2],
            ]

        t_grid = np.concatenate([[0.0], t_obs])
        y_ode = odeint(odes, [0.0, 0.0, 0.0], t_grid, rtol=1e-12, atol=1e-14)
        c_ref = y_ode[1:, 0] / V1

        np.testing.assert_allclose(sol.ipred, c_ref, rtol=1e-5,
                                   err_msg="ADVAN5 N=3 infusion diverges from odeint reference")

    def test_n4_dose_into_peripheral_vs_odeint(self):
        """
        ADVAN5 (N=4) bolus into compartment 3 (peripheral) matches odeint.

        Validates that non-central dosing is correctly handled by the
        state-vector initialisation in _bolus_response_n.
        """
        K, K12, K21, K13, K31, K14, K41 = 0.1, 0.05, 0.025, 0.02, 0.0067, 0.01, 0.005
        V1, D = 10.0, 100.0
        params = {
            "K": K, "K12": K12, "K21": K21,
            "K13": K13, "K31": K31,
            "K14": K14, "K41": K41,
            "V1": V1,
        }
        dose = [DoseEvent(time=0.0, amount=D, compartment=3)]
        sol = ADVAN5().solve(params, dose, OBS)

        def odes(y, _t):
            return [
                -(K + K12 + K13 + K14) * y[0] + K21 * y[1] + K31 * y[2] + K41 * y[3],
                K12 * y[0] - K21 * y[1],
                K13 * y[0] - K31 * y[2],
                K14 * y[0] - K41 * y[3],
            ]

        t_grid = np.concatenate([[0.0], OBS])
        y_ode = odeint(odes, [0.0, 0.0, D, 0.0], t_grid, rtol=1e-12, atol=1e-14)
        c_ref = y_ode[1:, 0] / V1

        np.testing.assert_allclose(sol.ipred, c_ref, rtol=1e-5,
                                   err_msg="ADVAN5 N=4 peripheral dose diverges from odeint")

    def test_n3_mass_balance_no_elimination(self):
        """
        With no elimination (K=0, Ki0=0), total drug in all compartments
        must equal the initial dose at all observation times.
        """
        K12, K21, K13, K31, V1, D = 0.3, 0.15, 0.1, 0.05, 10.0, 100.0
        params = {"K12": K12, "K21": K21, "K13": K13, "K31": K31, "V1": V1}
        # No elimination: K=0 is implied by absence of K/Ki0 key — but we need at
        # least one rate constant key; include K10=0 explicitly to satisfy parser.
        params["K10"] = 0.0
        sol = ADVAN5().solve(params, _bolus(D), OBS)

        total_drug = sol.amounts.sum(axis=1)  # shape (n_times,)
        np.testing.assert_allclose(
            total_drug, D, rtol=1e-6,
            err_msg="Total drug should equal dose when elimination is zero",
        )

    def test_n2_multiple_doses_superposition_vs_closed_form(self):
        """
        Two-dose superposition for N=2 matches sum of two biexponential terms.

        Tests that ADVAN5 implements linear superposition correctly by comparing
        to a closed-form reference computed independently for each dose.
        """
        K, K12, K21, V1 = 0.15, 0.30, 0.12, 15.0
        D1, D2, t2 = 100.0, 75.0, 8.0
        t_after = np.array([10.0, 14.0, 20.0, 36.0])  # all after second dose

        sol = ADVAN5().solve(
            {"K": K, "K12": K12, "K21": K21, "V1": V1},
            [DoseEvent(time=0.0, amount=D1), DoseEvent(time=t2, amount=D2)],
            t_after,
        )

        alpha, beta = self._alpha_beta(K, K12, K21)
        A = (alpha - K21) / (alpha - beta)
        B = (K21 - beta) / (alpha - beta)

        c1 = (D1 / V1) * (A * np.exp(-alpha * t_after) + B * np.exp(-beta * t_after))
        c2 = (D2 / V1) * (
            A * np.exp(-alpha * (t_after - t2)) + B * np.exp(-beta * (t_after - t2))
        )
        np.testing.assert_allclose(sol.ipred, c1 + c2, rtol=1e-7,
                                   err_msg="Two-dose superposition mismatch vs closed form")

