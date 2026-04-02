"""
Numerical accuracy and external-validation tests for OpenPKPD core algorithms.

Each test section anchors results to an independent authoritative source:

  PK analytical  — closed-form expressions from Gibaldi & Perrier (1982),
                   Rowland & Tozer (2011), or direct matrix-exponential
                   evaluation via scipy.linalg.expm.
  NCA            — exact integrals of monoexponential profiles; FDA guidance
                   formulas; Gabrielsson & Weiner (2006) reference formulas.
  Estimation     — log-likelihood from scipy.stats; AIC/BIC from first
                   principles; shrinkage from NONMEM definition (Karlsson
                   & Sheiner, 1993).
  Residuals      — Cholesky-based WRES verified against numpy.linalg.lstsq;
                   CWRES formula verified against the documented approximation.

A failure in any test below indicates a potential numerical discrepancy that
should be investigated before relying on model output for decision-making.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from scipy.linalg import expm

from openpkpd.data.event_processor import DoseEvent
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.residuals import (
    compute_iwres,
    compute_residual_variance,
    compute_wres,
    log_likelihood_normal,
)
from openpkpd.nca.nca import NCAEngine
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan2 import ADVAN2, _infusion_central_limit
from openpkpd.pk.analytical.advan3 import ADVAN3, _biexp_central, _eigenvalues, _propagate_2cmt
from openpkpd.pk.analytical.advan4 import ADVAN4
from openpkpd.utils.constants import LOG2PI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bolus(t: float = 0.0, amt: float = 100.0, cmt: int = 1) -> list[DoseEvent]:
    return [DoseEvent(time=t, amount=amt, compartment=cmt)]


def _infusion(
    t: float = 0.0, amt: float = 100.0, rate: float = 10.0, cmt: int = 1
) -> list[DoseEvent]:
    return [DoseEvent(time=t, amount=amt, rate=rate, compartment=cmt)]


# ============================================================================
# Section 1 — ADVAN1: 1-compartment IV bolus / infusion (exact formulas)
# ============================================================================


@pytest.mark.unit
class TestADVAN1ExactFormulas:
    """
    Verify ADVAN1 against the analytical formula C(t) = (DOSE/V) * exp(-K*t).
    Reference: Gibaldi & Perrier (1982), equation for one-compartment model.
    """

    def test_bolus_single_time_point_exact(self):
        """A(t) = DOSE * exp(-K*t), so C(t) = DOSE/V * exp(-K*t) exactly."""
        K, V, D = 0.2, 15.0, 120.0
        times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
        sol = ADVAN1().solve({"K": K, "V": V}, _bolus(amt=D), times)
        expected = D / V * np.exp(-K * times)
        np.testing.assert_allclose(
            sol.ipred, expected, rtol=1e-12, err_msg="ADVAN1 bolus deviates from C0*exp(-K*t)"
        )

    def test_infusion_during_formula(self):
        """
        During infusion (0 < t <= D): A(t) = R/K * (1 - exp(-K*t)).
        Reference: Rowland & Tozer (2011), Chapter 4.
        """
        K, V = 0.15, 20.0
        rate, amt = 50.0, 200.0
        amt / rate  # 4 h
        times_during = np.array([0.5, 1.0, 2.0, 3.5, 4.0])
        sol = ADVAN1().solve({"K": K, "V": V}, _infusion(amt=amt, rate=rate), times_during)
        expected = rate / K * (1.0 - np.exp(-K * times_during)) / V
        np.testing.assert_allclose(
            sol.ipred,
            expected,
            rtol=1e-12,
            err_msg="During-infusion ADVAN1 deviates from R/K*(1-exp(-K*t))/V",
        )

    def test_infusion_after_formula(self):
        """
        After infusion (t > D): A(t) = R/K * (1 - exp(-K*D)) * exp(-K*(t-D)).
        Reference: Rowland & Tozer (2011), Chapter 4.
        """
        K, V = 0.15, 20.0
        rate, amt = 50.0, 200.0
        duration = amt / rate  # 4 h
        times_after = np.array([5.0, 6.0, 8.0, 12.0, 24.0])
        sol = ADVAN1().solve({"K": K, "V": V}, _infusion(amt=amt, rate=rate), times_after)
        a_end = rate / K * (1.0 - np.exp(-K * duration))
        expected = a_end * np.exp(-K * (times_after - duration)) / V
        np.testing.assert_allclose(
            sol.ipred,
            expected,
            rtol=1e-12,
            err_msg="Post-infusion ADVAN1 deviates from R/K*(1-exp(-K*D))*exp(-K*(t-D))/V",
        )

    def test_infusion_mass_balance(self):
        """
        At any time, A(t) + eliminated(t) = total input(t).
        Rate of elimination = K * A(t); integrate numerically over [0, t].
        """
        K, V = 0.1, 10.0
        rate, amt = 20.0, 100.0
        duration = amt / rate  # 5 h
        times = np.concatenate(
            [np.linspace(0.01, duration, 50), np.linspace(duration + 0.01, 20.0, 30)]
        )
        sol = ADVAN1().solve({"K": K, "V": V}, _infusion(amt=amt, rate=rate), times)
        amounts = sol.amounts[:, 0]
        # Total input at each time
        total_input = np.where(times <= duration, rate * times, amt)
        # Eliminated = total_input - amount_in_compartment
        eliminated = total_input - amounts
        assert np.all(eliminated >= -1e-9), "Negative eliminated amount (mass violation)"
        assert np.all(amounts <= total_input + 1e-9), "Amount exceeds total input"

    def test_multiple_doses_superposition(self):
        """
        Superposition principle: C(t; D1+D2) = C(t; D1) + C(t; D2) at the same
        times when the two dosing events are at different times.
        """
        K, V = 0.1, 20.0
        doses = [DoseEvent(time=0.0, amount=80.0), DoseEvent(time=6.0, amount=40.0)]
        times = np.array([3.0, 6.0, 9.0, 12.0, 24.0])

        sol_combined = ADVAN1().solve({"K": K, "V": V}, doses, times)

        # Individual contributions
        sol_d1 = ADVAN1().solve({"K": K, "V": V}, _bolus(t=0.0, amt=80.0), times)
        sol_d2 = ADVAN1().solve({"K": K, "V": V}, _bolus(t=6.0, amt=40.0), times)

        expected = sol_d1.ipred + sol_d2.ipred
        np.testing.assert_allclose(
            sol_combined.ipred,
            expected,
            rtol=1e-12,
            err_msg="Superposition fails for multi-dose ADVAN1",
        )


# ============================================================================
# Section 2 — ADVAN2: Bateman function exact values
# ============================================================================


@pytest.mark.unit
class TestADVAN2ExactBatemanFormula:
    """
    Verify ADVAN2 against the Bateman function:
        C(t) = F*D*KA / (V*(KA-K)) * (exp(-K*t) - exp(-KA*t))

    Reference: Bateman (1910); Gibaldi & Perrier (1982), p. 21.
    """

    # Typical oral PK parameters
    KA, K, V, F, D = 1.2, 0.08, 25.0, 1.0, 300.0

    @property
    def _times(self) -> np.ndarray:
        return np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])

    def _bateman(self, t: np.ndarray) -> np.ndarray:
        ka, k, v, f, d = self.KA, self.K, self.V, self.F, self.D
        return f * d * ka / (v * (ka - k)) * (np.exp(-k * t) - np.exp(-ka * t))

    def test_exact_concentration_values(self):
        """ADVAN2 central compartment must match the Bateman function exactly."""
        sol = ADVAN2().solve(
            {"KA": self.KA, "K": self.K, "V": self.V, "F1": self.F},
            _bolus(amt=self.D),
            self._times,
        )
        expected = self._bateman(self._times)
        np.testing.assert_allclose(
            sol.ipred, expected, rtol=1e-12, err_msg="ADVAN2 IPRED deviates from Bateman function"
        )

    def test_peak_concentration_at_tmax(self):
        """
        Cmax should occur at Tmax = ln(KA/K) / (KA - K) and equal the
        Bateman function evaluated there.
        Reference: Gibaldi & Perrier (1982).
        """
        ka, k = self.KA, self.K
        tmax = math.log(ka / k) / (ka - k)
        cmax_analytical = self._bateman(np.array([tmax]))[0]

        times_dense = np.linspace(0.01, 3 * tmax, 2000)
        sol = ADVAN2().solve(
            {"KA": ka, "K": k, "V": self.V, "F1": self.F},
            _bolus(amt=self.D),
            times_dense,
        )
        cmax_numerical = float(np.max(sol.ipred))
        # Cmax from dense grid should be within 0.1% of analytical value
        assert cmax_numerical == pytest.approx(cmax_analytical, rel=1e-3), (
            f"Cmax numerical {cmax_numerical:.4f} vs analytical {cmax_analytical:.4f}"
        )

    def test_depot_follows_exact_exponential_decay(self):
        """
        Depot compartment (A1) = F*D * exp(-KA*t) for oral bolus.
        Reference: 1-compartment absorption.
        """
        times = self._times
        sol = ADVAN2().solve(
            {"KA": self.KA, "K": self.K, "V": self.V, "F1": self.F},
            _bolus(amt=self.D),
            times,
        )
        expected_a1 = self.F * self.D * np.exp(-self.KA * times)
        np.testing.assert_allclose(
            sol.amounts[:, 0],
            expected_a1,
            rtol=1e-12,
            err_msg="Depot A1 deviates from F*D*exp(-KA*t)",
        )

    def test_mass_balance_oral_bolus(self):
        """
        At any time t, A1(t) + A2(t) + eliminated(t) = F * DOSE.
        Eliminated drug = F*DOSE - A1(t) - A2(t), which must be non-negative.
        """
        times = np.linspace(0.1, 72.0, 300)
        sol = ADVAN2().solve(
            {"KA": self.KA, "K": self.K, "V": self.V, "F1": self.F},
            _bolus(amt=self.D),
            times,
        )
        total_remaining = sol.amounts[:, 0] + sol.amounts[:, 1]
        eliminated = self.F * self.D - total_remaining
        assert np.all(eliminated >= -1e-8), f"Negative eliminated (worst = {eliminated.min():.2e})"
        # At 72 h (≈8 half-lives for K=0.08) >99% should be eliminated.
        # t½ = ln(2)/K = ln(2)/0.08 ≈ 8.66 h; 72/8.66 ≈ 8.3 half-lives → ~99.7%
        assert eliminated[-1] > 0.99 * self.F * self.D, (
            f"Expected >99% elimination at 72 h but got {eliminated[-1] / (self.F * self.D):.2%}"
        )

    def test_limit_form_ka_approx_k_continuity(self):
        """
        When KA → K the limit form A2(t) = F*D*KA*t*exp(-K*t) is used.
        Just outside the tolerance (|KA-K| > 1e-6), the general Bateman
        formula should give values continuous with the limit form.
        Rationale: verify no discontinuity at the threshold.
        """
        K = 0.1
        times = np.array([1.0, 3.0, 6.0])

        # Use KA just above tolerance: smooth general form
        ka_outside = K + 5e-5  # well outside _KA_K_TOL = 1e-6
        sol_outside = ADVAN2().solve(
            {"KA": ka_outside, "K": K, "V": 10.0},
            _bolus(amt=100.0),
            times,
        )

        # Analytical limit: A2(t) = D*KA*t*exp(-K*t)/V
        limit_central = K * times * np.exp(-K * times) * 100.0 / 10.0

        # Should be close (within ~5% since KA differs from K by ~5e-5/K ≈ 0.05%)
        rel_error = np.abs(sol_outside.ipred - limit_central) / (limit_central + 1e-30)
        # The Bateman value just outside tolerance should be within 1% of limit form
        assert np.all(rel_error < 0.02), f"Large discontinuity near KA≈K: rel_errors = {rel_error}"

    def test_infusion_central_limit_formula_correction(self):
        """
        Regression test for the corrected _infusion_central_limit implementation.
        The correct limit form (KA → K) for the central compartment during
        infusion is:
            A2(t) = r * [(1 - exp(-K*t))/K - t*exp(-K*t)]
        """
        K = 0.1
        rate = 10.0
        times = np.array([1.0, 3.0, 5.0])

        # Correct limit formula (analytic derivation via L'Hôpital):
        correct_limit = rate * ((1.0 - np.exp(-K * times)) / K - times * np.exp(-K * times))

        np.testing.assert_allclose(
            _infusion_central_limit(rate, K, times),
            correct_limit,
            rtol=1e-10,
            atol=1e-12,
            err_msg="ADVAN2 infusion KA≈K limit form regressed from the analytic solution",
        )

    def test_infusion_regular_formula_outside_limit_zone(self):
        """
        For KA significantly different from K, the regular infusion formula
        should give A2 matching the convolution integral (via ADVAN6 ODE).
        Validates that the common code path (not the limit form) is correct.

        Note: ADVAN6 defaults to output_compartment=1 (depot), so we must
        set output_compartment=2 to get the central compartment (A2).
        """
        from openpkpd.parser.code_compiler import NMTRANCompiler
        from openpkpd.pk.ode.advan6 import ADVAN6

        KA, K, V = 1.5, 0.1, 10.0
        rate, amt = 20.0, 60.0  # 3 h infusion
        times = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 10.0])

        # ADVAN2 analytical — central compartment (A2/V)
        sol_analytical = ADVAN2().solve(
            {"KA": KA, "K": K, "V": V},
            _infusion(amt=amt, rate=rate),
            times,
        )

        # ADVAN6 numerical ODE — set output_compartment=2 to get A(2)/V
        des_oral = """
DADT(1) = -KA * A(1)
DADT(2) = KA * A(1) - K10 * A(2)
"""
        compiler = NMTRANCompiler()
        des = compiler.compile_des(des_oral, n_compartments=2)
        pk_params = {"KA": KA, "K10": K, "V": V}
        advan6 = ADVAN6(n_compartments=2)
        advan6.output_compartment = 2  # observe central, not depot
        sol_ode = advan6.solve(
            pk_params,
            _infusion(amt=amt, rate=rate),
            times,
            des_callable=des,
        )

        np.testing.assert_allclose(
            sol_analytical.ipred,
            sol_ode.ipred,
            rtol=1e-4,
            err_msg="ADVAN2 infusion deviates from ADVAN6 ODE by more than 1e-4",
        )


# ============================================================================
# Section 3 — ADVAN3 eigenvalue and propagation correctness
# ============================================================================


@pytest.mark.unit
class TestADVAN3ExactEigendecomposition:
    """
    Verify the ADVAN3 biexponential formula against independent matrix-exponential
    evaluations via scipy.linalg.expm.  The matrix-exponential approach is
    mathematically equivalent but derived from a completely different code path.
    """

    # Clinically plausible 2-cmt IV parameters
    K, K12, K21, V1, DOSE = 0.2, 0.12, 0.05, 8.0, 150.0

    def _rate_matrix(self) -> np.ndarray:
        k, k12, k21 = self.K, self.K12, self.K21
        return np.array([[-(k + k12), k21], [k12, -k21]])

    def _matrix_exp_amounts(self, t: np.ndarray) -> np.ndarray:
        """Compute A1, A2 via matrix exponential (reference implementation)."""
        m = self._rate_matrix()
        y0 = np.array([self.DOSE, 0.0])
        out = np.zeros((len(t), 2))
        for i, ti in enumerate(t):
            out[i] = expm(m * ti) @ y0
        return out

    def test_bolus_amounts_match_matrix_exponential(self):
        """
        A1(t), A2(t) from ADVAN3 analytical formula must match scipy.linalg.expm.
        Reference: matrix-exponential solution to linear ODE (standard linear algebra).
        """
        times = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        sol = ADVAN3().solve(
            {"K": self.K, "K12": self.K12, "K21": self.K21, "V1": self.V1},
            _bolus(amt=self.DOSE),
            times,
        )
        ref = self._matrix_exp_amounts(times)
        np.testing.assert_allclose(
            sol.amounts,
            ref,
            rtol=1e-10,
            atol=1e-12,
            err_msg="ADVAN3 amounts diverge from matrix expm",
        )

    def test_ipred_equals_a1_over_v1(self):
        """IPRED must equal A1 / V1 throughout."""
        times = np.array([1.0, 4.0, 10.0])
        sol = ADVAN3().solve(
            {"K": self.K, "K12": self.K12, "K21": self.K21, "V1": self.V1},
            _bolus(amt=self.DOSE),
            times,
        )
        np.testing.assert_allclose(sol.ipred, sol.amounts[:, 0] / self.V1, rtol=1e-14)

    def test_mass_balance_bolus(self):
        """
        A1(t) + A2(t) must be ≤ DOSE at all t > 0 (some drug is eliminated).
        This checks conservation of mass.

        With K=0.2, K12=0.12, K21=0.05 the slow eigenvalue is λ1≈0.029 h⁻¹
        (t½≈24 h), so at t=200 h (>8 half-lives) less than 1% of dose remains.
        """
        times = np.linspace(0.1, 200.0, 200)
        sol = ADVAN3().solve(
            {"K": self.K, "K12": self.K12, "K21": self.K21, "V1": self.V1},
            _bolus(amt=self.DOSE),
            times,
        )
        total = sol.amounts.sum(axis=1)
        assert np.all(total <= self.DOSE + 1e-8), (
            f"Mass exceeded dose: max = {total.max():.4f}, dose = {self.DOSE}"
        )
        # At t=200h (>8 slow-phase half-lives), less than 1% should remain
        assert total[-1] < 0.01 * self.DOSE, (
            f"Expected <1% drug remaining at t=200h; got {total[-1] / self.DOSE:.1%}"
        )

    def test_eigenvalues_satisfy_vieta_formulas(self):
        """
        The eigenvalues λ1, λ2 must satisfy:
            λ1 + λ2 = K + K12 + K21
            λ1 * λ2 = K * K21
        Reference: Vieta's formulas for the characteristic polynomial.
        """
        k, k12, k21 = 0.15, 0.08, 0.04
        lam1, lam2 = _eigenvalues(k, k12, k21)
        assert lam1 + lam2 == pytest.approx(k + k12 + k21, rel=1e-12)
        assert lam1 * lam2 == pytest.approx(k * k21, rel=1e-12)

    def test_biexp_central_initial_condition(self):
        """
        At t=0+, A1(0) = DOSE and A2(0) = 0 (pre-dose convention uses t > 0).
        For t → 0, A1 → DOSE.
        """
        k, k12, k21 = 0.2, 0.1, 0.05
        lam1, lam2 = _eigenvalues(k, k12, k21)
        dt = np.array([1e-10])
        a1, a2 = _biexp_central(100.0, k, k12, k21, lam1, lam2, dt)
        assert float(a1[0]) == pytest.approx(100.0, rel=1e-6)
        assert float(a2[0]) == pytest.approx(0.0, abs=1e-6)

    def test_propagate_2cmt_matches_matrix_exponential(self):
        """
        _propagate_2cmt(a1_0, a2_0, ..., dt) should match expm(M*dt) @ [a1_0, a2_0].
        This propagation function is used for post-infusion free decay.
        """
        k, k12, k21 = 0.15, 0.10, 0.06
        lam1, lam2 = _eigenvalues(k, k12, k21)
        a1_0, a2_0 = 40.0, 25.0
        times = np.array([1.0, 3.0, 7.0, 15.0])
        m = np.array([[-(k + k12), k21], [k12, -k21]])
        y0 = np.array([a1_0, a2_0])

        a1_prop, a2_prop = _propagate_2cmt(a1_0, a2_0, k, k12, k21, lam1, lam2, times)

        for i, ti in enumerate(times):
            ref = expm(m * ti) @ y0
            assert a1_prop[i] == pytest.approx(ref[0], rel=1e-10)
            assert a2_prop[i] == pytest.approx(ref[1], rel=1e-10)

    def test_biexp_central_equal_eigenvalues_limit(self):
        """
        When λ1 ≈ λ2 (near-degenerate case), the limit form is used.
        Verify: (a) no NaN/Inf, (b) A1+A2 conserves mass, (c) continuous
        with the general form just outside the degenerate threshold.
        """
        # Construct parameters that give λ1 ≈ λ2
        # K21 ≈ 0 reduces the discriminant to (K+K12)^2, making λ1 ≈ 0, λ2 ≈ K+K12
        # Instead, choose K and K21 close so D ≈ 0
        # S^2 - 4*K*K21 ≈ 0 → K21 ≈ S^2/4K where S = K+K12+K21
        # Use numerical search: set K21 = (K+K12)^2/(4K) approximately
        K, K12 = 0.2, 0.001
        K21 = (K + K12) ** 2 / (4 * K) - 1e-12  # near-degenerate
        lam1, lam2 = _eigenvalues(K, K12, K21)
        DOSE = 100.0
        dt = np.array([0.5, 2.0, 10.0])

        a1, a2 = _biexp_central(DOSE, K, K12, K21, lam1, lam2, dt)
        assert np.all(np.isfinite(a1)), "NaN/Inf in A1 for near-degenerate eigenvalues"
        assert np.all(np.isfinite(a2)), "NaN/Inf in A2 for near-degenerate eigenvalues"
        assert np.all(a1 >= 0), "Negative A1 for near-degenerate eigenvalues"
        assert np.all(a2 >= 0), "Negative A2 for near-degenerate eigenvalues"
        # Mass conservation
        assert np.all(a1 + a2 <= DOSE + 1e-8)

    def test_propagate_2cmt_degenerate_matches_expm(self):
        """
        _propagate_2cmt degenerate branch (λ1=λ2) must match scipy.linalg.expm.

        Exact degenerate case: K = K21, K12 = 0 → D = sqrt(S²-4K*K21) = 0.
        Rate matrix becomes upper-triangular: M = [[-K, K],[0, -K]].
        expm(M*t) = exp(-K*t) * [[1, K*t],[0, 1]].
        """
        from scipy.linalg import expm

        K, K12, K21 = 0.3, 0.0, 0.3  # exact degenerate: D = 0, lam1 = lam2 = K
        lam1, lam2 = _eigenvalues(K, K12, K21)
        assert (lam2 - lam1) < 1e-10, "Test setup requires degenerate eigenvalues"

        a1_0, a2_0 = 40.0, 25.0
        times = np.array([0.5, 2.0, 8.0])
        m = np.array([[-(K + K12), K21], [K12, -K21]])
        y0 = np.array([a1_0, a2_0])

        a1_prop, a2_prop = _propagate_2cmt(a1_0, a2_0, K, K12, K21, lam1, lam2, times)

        for i, ti in enumerate(times):
            ref = expm(m * ti) @ y0
            assert a1_prop[i] == pytest.approx(ref[0], rel=1e-10, abs=1e-12), (
                f"A1 degenerate mismatch at t={ti}: got {a1_prop[i]:.6f}, expected {ref[0]:.6f}"
            )
            assert a2_prop[i] == pytest.approx(ref[1], rel=1e-10, abs=1e-12), (
                f"A2 degenerate mismatch at t={ti}: got {a2_prop[i]:.6f}, expected {ref[1]:.6f}"
            )


# ============================================================================
# Section 4 — ADVAN4 cross-validation against ADVAN6 ODE
# ============================================================================


@pytest.mark.unit
class TestADVAN4VsODE:
    """
    Cross-validate ADVAN4 (2-cmt oral analytical) against ADVAN6 (generic ODE).

    Both solve the same system:
        dA1/dt = -KA * A1
        dA2/dt =  KA * A1 - (K + K12) * A2 + K21 * A3
        dA3/dt =  K12 * A2 - K21 * A3

    The analytical solution (ADVAN4) and the ODE solver (ADVAN6+RK45) must
    agree on the central compartment to within the ODE numerical tolerance.
    Agreement on the peripheral compartment (A3) is tested separately because
    the current ADVAN4 implementation uses an approximate formula for A3.
    """

    PARAMS_ANAL = {"KA": 1.5, "K": 0.2, "K12": 0.1, "K21": 0.05, "V2": 10.0}
    PARAMS_ODE = {"KA": 1.5, "K10": 0.2, "K12": 0.1, "K21": 0.05, "V": 10.0, "V2": 10.0}
    DOSE_AMT = 100.0
    TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])

    @staticmethod
    def _build_des():
        from openpkpd.parser.code_compiler import NMTRANCompiler

        des = """
DADT(1) = -KA * A(1)
DADT(2) =  KA * A(1) - (K10 + K12) * A(2) + K21 * A(3)
DADT(3) =  K12 * A(2) - K21 * A(3)
"""
        return NMTRANCompiler().compile_des(des, n_compartments=3)

    def test_central_compartment_matches_ode(self):
        """
        IPRED from ADVAN4 must match ADVAN6 to within 1e-4 relative tolerance.
        This validates the central-compartment triexponential formula.

        Note: ADVAN6 defaults to output_compartment=1 (depot) so we must
        set output_compartment=2 (central) for a fair comparison.
        """
        from openpkpd.pk.ode.advan6 import ADVAN6

        des_callable = self._build_des()
        dose = _bolus(amt=self.DOSE_AMT)

        sol_anal = ADVAN4().solve(self.PARAMS_ANAL, dose, self.TIMES)

        advan6 = ADVAN6(n_compartments=3)
        advan6.output_compartment = 2  # central, not depot
        sol_ode = advan6.solve(self.PARAMS_ODE, dose, self.TIMES, des_callable=des_callable)

        np.testing.assert_allclose(
            sol_anal.ipred,
            sol_ode.ipred,
            rtol=1e-4,
            atol=1e-8,
            err_msg="ADVAN4 central compartment deviates from ADVAN6 ODE",
        )

    def test_central_compartment_multidose_matches_ode(self):
        """
        Multi-dose superposition for ADVAN4 central must also match ADVAN6.
        """
        from openpkpd.pk.ode.advan6 import ADVAN6

        doses = [DoseEvent(time=0.0, amount=100.0), DoseEvent(time=12.0, amount=100.0)]
        times = np.array([6.0, 12.0, 14.0, 18.0, 24.0, 36.0])
        des_callable = self._build_des()

        sol_anal = ADVAN4().solve(self.PARAMS_ANAL, doses, times)

        advan6 = ADVAN6(n_compartments=3)
        advan6.output_compartment = 2  # central, not depot
        sol_ode = advan6.solve(self.PARAMS_ODE, doses, times, des_callable=des_callable)

        np.testing.assert_allclose(
            sol_anal.ipred,
            sol_ode.ipred,
            rtol=1e-4,
            atol=1e-8,
            err_msg="ADVAN4 multi-dose central deviates from ADVAN6",
        )

    def test_peripheral_compartment_matches_ode(self):
        """
        ADVAN4 A3 must match the ADVAN6 ODE reference to numerical tolerance.

        This guards the exact depot→central→peripheral convolution formula used
        by the analytical solver.
        """
        from openpkpd.pk.ode.advan6 import ADVAN6

        des_callable = self._build_des()
        dose = _bolus(amt=self.DOSE_AMT)

        sol_anal = ADVAN4().solve(self.PARAMS_ANAL, dose, self.TIMES)
        advan6 = ADVAN6(n_compartments=3)
        advan6.output_compartment = 2
        sol_ode = advan6.solve(self.PARAMS_ODE, dose, self.TIMES, des_callable=des_callable)

        a3_anal = sol_anal.amounts[:, 2]
        a3_ode = sol_ode.amounts[:, 2]

        assert np.all(a3_ode >= -1e-8), "ODE A3 should be non-negative"
        np.testing.assert_allclose(
            a3_anal,
            a3_ode,
            rtol=1e-4,
            atol=1e-8,
            err_msg="ADVAN4 A3 deviates from the ADVAN6 ODE reference",
        )

    def test_mass_balance_central_plus_depot(self):
        """
        A1(t) + A2(t) must not exceed DOSE at any time, regardless of A3 accuracy.
        Mass balance: total drug ≤ DOSE.
        """
        times = np.linspace(0.1, 48.0, 100)
        sol = ADVAN4().solve(self.PARAMS_ANAL, _bolus(amt=self.DOSE_AMT), times)
        a1 = sol.amounts[:, 0]
        a2 = sol.amounts[:, 1]
        assert np.all(a1 >= -1e-8), "Depot A1 negative"
        assert np.all(a2 >= -1e-8), "Central A2 negative"
        assert np.all(a1 + a2 <= self.DOSE_AMT + 1e-6), "A1+A2 exceeds dose"

    def test_degenerate_eigenvalue_infusion_matches_ode(self):
        """
        A-2: Verify the ADVAN4 degenerate-eigenvalue infusion formula (λ1≈λ2)
        against the ADVAN6 numerical ODE.

        Degenerate eigenvalues occur when S² = 4*K*K21, i.e.
        (K+K12+K21)² = 4*K*K21.  With K=0.3, K21=0.3, K12=0.0 → S=0.6, D=0.
        This triggers the `abs(dl) < 1e-10` branch in _infusion_triexp.
        """
        from openpkpd.pk.ode.advan6 import ADVAN6

        # Degenerate: K = K21 = 0.3, K12 = 0  → D = sqrt((0.6)^2 - 4*0.3*0.3) = 0
        params_anal = {"KA": 1.0, "K": 0.3, "K12": 0.0, "K21": 0.3, "V2": 10.0}
        params_ode = {"KA": 1.0, "K10": 0.3, "K12": 0.0, "K21": 0.3, "V": 10.0, "V2": 10.0}

        # Infusion: 100 mg over 2 h (rate = 50 mg/h)
        infusion_dose = [DoseEvent(time=0.0, amount=100.0, rate=50.0, compartment=1)]
        times = np.array([0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0])

        des_callable = self._build_des()
        sol_anal = ADVAN4().solve(params_anal, infusion_dose, times)
        advan6 = ADVAN6(n_compartments=3)
        advan6.output_compartment = 2
        sol_ode = advan6.solve(params_ode, infusion_dose, times, des_callable=des_callable)

        np.testing.assert_allclose(
            sol_anal.ipred,
            sol_ode.ipred,
            rtol=1e-3,
            atol=1e-7,
            err_msg="ADVAN4 degenerate-eigenvalue infusion central deviates from ADVAN6 ODE",
        )


# ============================================================================
# Section 5 — NCA: exact integral validation
# ============================================================================


@pytest.mark.unit
class TestNCAExactIntegrals:
    """
    For a monoexponential profile C(t) = C0 * exp(-K*t) all NCA integrals
    have closed-form expressions that serve as exact reference values.

    AUC_last  = C0/K * (1 - exp(-K*t_last))
    AUMC_last = C0/K^2 * (1 - (K*t_last + 1)*exp(-K*t_last))
    AUC_inf   = C0/K
    AUMC_inf  = C0/K^2
    MRT       = AUMC_inf / AUC_inf = 1/K

    Reference: Gibaldi & Perrier (1982); Gabrielsson & Weiner (2006) Section 1.3.
    """

    K = 0.1
    C0 = 8.0
    DOSE = 80.0
    T_LAST = 24.0

    @property
    def _auc_last_exact(self) -> float:
        return self.C0 / self.K * (1.0 - math.exp(-self.K * self.T_LAST))

    @property
    def _auc_inf_exact(self) -> float:
        return self.C0 / self.K

    @property
    def _aumc_last_exact(self) -> float:
        return (
            self.C0
            / self.K**2
            * (1.0 - (self.K * self.T_LAST + 1.0) * math.exp(-self.K * self.T_LAST))
        )

    @property
    def _aumc_inf_exact(self) -> float:
        return self.C0 / self.K**2

    @property
    def _mrt_exact(self) -> float:
        return 1.0 / self.K

    def _make_profile(self, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Dense monoexponential profile from 0 to T_LAST."""
        t = np.linspace(0.0, self.T_LAST, n)
        c = self.C0 * np.exp(-self.K * t)
        return t, c

    def test_auc_last_log_trapezoidal_exact(self):
        """
        Log-trapezoidal AUC_last should be exact for a monoexponential profile
        (within floating-point precision) because each interval is perfectly
        log-linear.
        """
        t, c = self._make_profile(n=25)
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_subject(t, c, dose=self.DOSE, route="IV")
        assert result.auc_last == pytest.approx(self._auc_last_exact, rel=1e-10), (
            f"AUC_last {result.auc_last:.6f} != exact {self._auc_last_exact:.6f}"
        )

    def test_aumc_last_dense_grid(self):
        """
        AUMC_last = integral t*C(t) dt from 0 to t_last.
        On a dense grid the linear-log approximation on t*C(t) should be
        accurate to within 0.1% of the exact integral.
        Reference formula: C0/K^2 * (1 - (K*t_last + 1)*exp(-K*t_last))
        """
        t, c = self._make_profile(n=200)
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_subject(t, c, dose=self.DOSE, route="IV")
        assert result.aumc_last == pytest.approx(self._aumc_last_exact, rel=1e-3), (
            f"AUMC_last {result.aumc_last:.4f} != exact {self._aumc_last_exact:.4f}"
        )

    def test_aumc_inf_extrapolation_formula(self):
        """
        AUMC_inf = AUMC_last + C_last * t_last / lambda_z + C_last / lambda_z^2.
        For monoexponential: AUMC_inf = C0/K^2 exactly.
        Reference: Gibaldi & Perrier (1982), eq. 11-7.
        """
        t, c = self._make_profile(n=200)
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_subject(t, c, dose=self.DOSE, route="IV")
        assert np.isfinite(result.aumc_inf), "aumc_inf should be finite"
        assert result.aumc_inf == pytest.approx(self._aumc_inf_exact, rel=1e-3), (
            f"AUMC_inf {result.aumc_inf:.4f} != exact {self._aumc_inf_exact:.4f}"
        )

    def test_mrt_equals_one_over_k(self):
        """
        MRT = AUMC_inf / AUC_inf = 1/K for IV monoexponential.
        Reference: Gibaldi & Perrier (1982).
        """
        t, c = self._make_profile(n=200)
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_subject(t, c, dose=self.DOSE, route="IV")
        assert result.mrt == pytest.approx(self._mrt_exact, rel=1e-3), (
            f"MRT {result.mrt:.4f} != 1/K = {self._mrt_exact:.4f}"
        )

    def test_cl_f_equals_k_times_v(self):
        """CL/F = Dose / AUC_inf = K * V (when Dose = C0 * V)."""
        # C0 = Dose/V  → Dose = C0 * V, V = Dose/C0
        V = self.DOSE / self.C0
        t, c = self._make_profile(n=100)
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_subject(t, c, dose=self.DOSE, route="IV")
        expected_cl = self.K * V
        assert result.cl_f == pytest.approx(expected_cl, rel=1e-3)

    def test_vz_f_equals_v(self):
        """Vz/F = CL/F / lambda_z = V (exactly, by definition)."""
        V = self.DOSE / self.C0
        t, c = self._make_profile(n=100)
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_subject(t, c, dose=self.DOSE, route="IV")
        assert result.vz_f == pytest.approx(V, rel=1e-3)

    def test_auc_extrap_pct_formula(self):
        """
        AUC_extrap% = (AUC_inf - AUC_last) / AUC_inf * 100.
        For t_last = 24 h, K = 0.1: AUC_extrap% ≈ exp(-0.1*24)*100% ≈ 9.1%.
        """
        t, c = self._make_profile(n=50)
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_subject(t, c, dose=self.DOSE, route="IV")
        expected_pct = math.exp(-self.K * self.T_LAST) * 100.0
        assert result.auc_extrap_pct == pytest.approx(expected_pct, rel=0.02), (
            f"AUC_extrap% {result.auc_extrap_pct:.2f}% != expected {expected_pct:.2f}%"
        )

    def test_linear_trapezoidal_bias_quantified(self):
        """
        Linear trapezoidal should overestimate AUC for monoexponential profiles.
        This quantifies the bias so it can be monitored.
        """
        t, c = self._make_profile(n=25)
        engine_log = NCAEngine(auc_method="linear-log")
        engine_lin = NCAEngine(auc_method="linear-trapezoidal")

        r_log = engine_log.compute_subject(t, c, dose=self.DOSE, route="IV")
        r_lin = engine_lin.compute_subject(t, c, dose=self.DOSE, route="IV")

        # Log should be more accurate
        err_log = abs(r_log.auc_last - self._auc_last_exact)
        err_lin = abs(r_lin.auc_last - self._auc_last_exact)
        assert err_log < err_lin, (
            "Expected log trapezoidal to be more accurate for monoexponential decline"
        )
        # Confirm linear over-estimates (positive bias for decreasing profile)
        assert r_lin.auc_last > self._auc_last_exact


@pytest.mark.unit
class TestNCABLQHandling:
    """
    Verify BLQ handling rules produce the correct output concentrations.
    Reference: FDA Guidance for Industry (2003); PhRMA working group (2010).
    """

    LLOQ = 0.5
    TIMES = np.array([0.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    CONC = np.array([10.0, 4.5, 2.0, 0.8, 0.3, 0.1])  # last two below LLOQ

    def _engine(self) -> NCAEngine:
        return NCAEngine()

    def test_zero_rule(self):
        """'zero': BLQ values set to 0.0."""
        conc_out = self._engine().apply_predose_blq_rule(
            self.TIMES, self.CONC, self.LLOQ, rule="zero"
        )
        blq_mask = self.CONC < self.LLOQ
        assert np.all(conc_out[blq_mask] == 0.0)
        assert np.all(conc_out[~blq_mask] == self.CONC[~blq_mask])

    def test_lloq_half_rule(self):
        """'lloq_half': BLQ values set to LLOQ/2."""
        conc_out = self._engine().apply_predose_blq_rule(
            self.TIMES, self.CONC, self.LLOQ, rule="lloq_half"
        )
        blq_mask = self.CONC < self.LLOQ
        np.testing.assert_allclose(conc_out[blq_mask], self.LLOQ / 2.0, rtol=1e-12)

    def test_exclude_rule(self):
        """'exclude': BLQ values set to NaN."""
        conc_out = self._engine().apply_predose_blq_rule(
            self.TIMES, self.CONC, self.LLOQ, rule="exclude"
        )
        blq_mask = self.CONC < self.LLOQ
        assert np.all(np.isnan(conc_out[blq_mask]))

    def test_above_lloq_unchanged_by_all_rules(self):
        """Values above LLOQ must be unchanged regardless of rule."""
        for rule in ("zero", "lloq_half", "exclude"):
            conc_out = self._engine().apply_predose_blq_rule(
                self.TIMES, self.CONC, self.LLOQ, rule=rule
            )
            above_lloq = self.CONC >= self.LLOQ
            np.testing.assert_allclose(conc_out[above_lloq], self.CONC[above_lloq])


@pytest.mark.unit
class TestNCAMultidoseParameters:
    """
    Verify multidose NCA parameters: AUC_tau, C_min, C_avg, fluctuation, swing.

    Reference profile: monoexponential multiple-dose steady state.
    At steady state, C(t) within dosing interval [0, τ] is:
        C(t) = C_inf * (exp(-K*t) / (1 - exp(-K*tau)))
    where C_inf = F*D/V is the single-dose C0.

    Reference: Rowland & Tozer (2011), Chapter 17.
    """

    K = 0.1
    TAU = 12.0  # dosing interval (h)
    DOSE = 100.0
    V = 10.0
    DOSE_SS = 4  # number of doses before measuring SS interval

    def _make_ss_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate steady-state concentration-time profile within last tau interval
        using superposition of multiple oral bolus doses (approximation for test purposes).
        """
        C0_single = self.DOSE / self.V
        # Steady-state peak: C_peak = C0 / (1 - exp(-K*tau))
        c_peak_ss = C0_single / (1.0 - math.exp(-self.K * self.TAU))
        # Build SS profile within [0, TAU] using exact formula
        times = np.linspace(0.0, self.TAU, 49)  # 49 points including endpoints
        c_ss = c_peak_ss * np.exp(-self.K * times)
        return times, c_ss

    def test_auc_tau_exact(self):
        """
        AUC_tau at steady state = AUC_inf of single dose (for IV):
            = C0 / K = DOSE/V/K
        Reference: Rowland & Tozer (2011), Table 17-1.
        """
        times, c_ss = self._make_ss_profile()
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_multidose_subject(times, c_ss, dose=self.DOSE, tau=self.TAU)
        auc_tau_exact = self.DOSE / self.V / self.K
        assert result.auc_tau == pytest.approx(auc_tau_exact, rel=1e-10), (
            f"AUC_tau {result.auc_tau:.4f} != exact {auc_tau_exact:.4f}"
        )

    def test_c_avg_equals_auc_tau_over_tau(self):
        """C_avg = AUC_tau / tau (definitional identity)."""
        times, c_ss = self._make_ss_profile()
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_multidose_subject(times, c_ss, dose=self.DOSE, tau=self.TAU)
        if np.isfinite(result.auc_tau) and np.isfinite(result.c_avg):
            assert result.c_avg == pytest.approx(result.auc_tau / self.TAU, rel=1e-10)

    def test_fluctuation_formula(self):
        """
        Fluctuation = (Cmax - Cmin) / Cavg * 100.
        Reference: Rowland & Tozer (2011).
        """
        times, c_ss = self._make_ss_profile()
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_multidose_subject(times, c_ss, dose=self.DOSE, tau=self.TAU)
        if np.isfinite(result.fluctuation) and np.isfinite(result.c_avg) and result.c_avg > 0:
            expected = (result.cmax - result.c_min) / result.c_avg * 100.0
            assert result.fluctuation == pytest.approx(expected, rel=1e-10)

    def test_swing_formula(self):
        """Swing = (Cmax - Cmin) / Cmin when Cmin > 0."""
        times, c_ss = self._make_ss_profile()
        engine = NCAEngine()
        result = engine.compute_multidose_subject(times, c_ss, dose=self.DOSE, tau=self.TAU)
        if np.isfinite(result.swing) and result.c_min > 0:
            expected = (result.cmax - result.c_min) / result.c_min
            assert result.swing == pytest.approx(expected, rel=1e-10)

    def test_dose_normalised_parameters(self):
        """
        norm_cmax = cmax / dose; norm_auc_last = auc_last / dose, etc.
        Reference: FDA guidance on dose-normalisation.
        """
        times, c_ss = self._make_ss_profile()
        engine = NCAEngine(auc_method="linear-log")
        result = engine.compute_multidose_subject(times, c_ss, dose=self.DOSE, tau=self.TAU)
        if np.isfinite(result.norm_cmax):
            assert result.norm_cmax == pytest.approx(result.cmax / self.DOSE, rel=1e-12)
        if np.isfinite(result.norm_auc_last):
            assert result.norm_auc_last == pytest.approx(result.auc_last / self.DOSE, rel=1e-12)


# ============================================================================
# Section 6 — Log-likelihood formula vs scipy.stats cross-check
# ============================================================================


@pytest.mark.unit
class TestLogLikelihoodVsScipy:
    """
    Verify log_likelihood_normal against scipy.stats.norm.logpdf.
    Both compute log p(y | mu, sigma^2) = log N(y; mu, sigma^2).
    Reference: standard Gaussian log-likelihood; scipy.stats documentation.
    """

    @pytest.mark.parametrize(
        "y, mu, sigma2",
        [
            (2.5, 2.0, 4.0),
            (0.0, 0.0, 1.0),
            (10.0, 8.5, 0.25),
            (-3.0, -3.0, 0.01),
            (100.0, 95.0, 100.0),
        ],
    )
    def test_matches_scipy_logpdf(self, y: float, mu: float, sigma2: float):
        """
        log_likelihood_normal(y, mu, sigma2) should equal scipy.stats.norm.logpdf(y, mu, sqrt(sigma2)).
        """
        sigma = math.sqrt(sigma2)
        expected = float(stats.norm.logpdf(y, loc=mu, scale=sigma))
        result = log_likelihood_normal(y, mu, sigma2)
        assert result == pytest.approx(expected, rel=1e-10, abs=1e-14), (
            f"log_likelihood_normal({y}, {mu}, {sigma2}) = {result:.8f}, scipy gives {expected:.8f}"
        )

    def test_sum_equals_2ll_ofv(self):
        """
        OFV = -2 * sum(log p(y_i | mu_i, sigma2_i)) must equal the standard
        NONMEM OFV.  Verify on a 5-observation sample.
        """
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu = np.array([1.1, 1.9, 3.2, 4.1, 4.8])
        sigma2 = np.array([0.1, 0.2, 0.3, 0.1, 0.5])

        ll_openpkpd = sum(
            log_likelihood_normal(yi, mi, s2) for yi, mi, s2 in zip(y, mu, sigma2, strict=False)
        )
        ll_scipy = float(
            sum(
                stats.norm.logpdf(yi, loc=mi, scale=math.sqrt(s2))
                for yi, mi, s2 in zip(y, mu, sigma2, strict=False)
            )
        )
        assert ll_openpkpd == pytest.approx(ll_scipy, rel=1e-10)

    def test_log2pi_constant(self):
        """LOG2PI = log(2*pi) must equal the standard mathematical value."""
        assert pytest.approx(math.log(2.0 * math.pi), rel=1e-15) == LOG2PI


# ============================================================================
# Section 7 — Residuals: WRES via direct Cholesky vs numpy reference
# ============================================================================


@pytest.mark.unit
class TestWRESExactValues:
    """
    WRES = L^{-1} * (DV - PRED) where L is lower Cholesky factor of C_i.
    Verify against: (a) direct algebraic calculation, (b) numpy lstsq solution.
    Reference: Hooker et al. (2007), NONMEM technical guide.
    """

    def test_diagonal_covariance_exact_value(self):
        """
        For diagonal C_i, WRES_i = (DV_i - PRED_i) / sqrt(C_ii).
        Direct algebraic formula — independent implementation.
        """
        dv = np.array([3.0, 5.0, 7.0])
        pred = np.array([2.5, 4.8, 6.5])
        variances = np.array([0.25, 0.64, 1.44])
        c_i = np.diag(variances)

        wres = compute_wres(dv, pred, c_i)
        expected = (dv - pred) / np.sqrt(variances)  # algebraic formula

        np.testing.assert_allclose(
            wres, expected, rtol=1e-12, err_msg="WRES deviates from (DV-PRED)/sqrt(Var)"
        )

    def test_correlated_covariance_matches_lstsq(self):
        """
        For a full positive-definite C_i, verify WRES against numpy.linalg.lstsq.
        WRES = L^{-1}(DV-PRED) is equivalent to solving L*w = DV-PRED for w.
        """
        from scipy.linalg import cholesky

        rng = np.random.default_rng(12345)
        n = 4
        # Build random positive-definite C_i
        A = rng.normal(size=(n, n))
        c_i = A @ A.T + np.eye(n) * 0.5

        dv = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([0.8, 1.9, 3.1, 4.2])

        wres = compute_wres(dv, pred, c_i)

        # Reference: solve L*w = (DV-PRED) where L = lower Cholesky
        L = cholesky(c_i, lower=True)
        wres_ref = np.linalg.lstsq(L, dv - pred, rcond=None)[0]

        np.testing.assert_allclose(
            wres, wres_ref, rtol=1e-10, err_msg="WRES deviates from Cholesky back-substitution"
        )

    def test_wres_whitening_property(self):
        """
        If DV - PRED ~ N(0, C_i), then WRES = L^{-1}(DV-PRED) ~ N(0, I).
        Empirically: sample-covariance of WRES should approach identity.
        Uses Monte Carlo simulation.
        """
        from scipy.linalg import cholesky

        rng = np.random.default_rng(999)
        n = 3
        A = rng.normal(size=(n, n))
        c_i = A @ A.T + np.eye(n) * 0.1
        L = cholesky(c_i, lower=True)
        pred = np.zeros(n)

        # Simulate 5000 draws
        N_DRAWS = 5000
        wres_list = []
        for _ in range(N_DRAWS):
            eps = L @ rng.standard_normal(n)
            dv = pred + eps
            wres_list.append(compute_wres(dv, pred, c_i))

        wres_arr = np.array(wres_list)  # (N_DRAWS, n)
        sample_cov = np.cov(wres_arr.T)
        # Should be close to identity
        np.testing.assert_allclose(
            sample_cov,
            np.eye(n),
            atol=0.07,
            err_msg="WRES covariance deviates from identity matrix",
        )

    def test_iwres_exact_value(self):
        """IWRES = (DV - IPRED) / W with exact known values."""
        dv = np.array([5.0, 8.0, 3.0])
        ipred = np.array([4.5, 7.5, 3.2])
        w = np.array([0.5, 1.0, 0.25])
        expected = (dv - ipred) / w

        result = compute_iwres(dv, ipred, w)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_residual_variance_proportional_exact(self):
        """
        Proportional residual model: Var(Y|f) = f^2 * sigma[0,0].
        Verify the exact formula at multiple f values.
        """
        sigma = np.array([[0.04]])  # 20% CV proportional
        for f in [0.5, 1.0, 5.0, 10.0]:
            expected = f**2 * 0.04
            result = compute_residual_variance(f, sigma, error_type="proportional")
            assert result == pytest.approx(expected, rel=1e-12)

    def test_residual_variance_combined_exact(self):
        """
        Combined model: Var(Y|f) = sigma_add + f^2 * sigma_prop.
        Verify exact numerical values.
        """
        sigma_add = 0.1
        sigma_prop = 0.04
        sigma = np.diag([sigma_add, sigma_prop])
        f = 4.0
        expected = sigma_add + f**2 * sigma_prop
        result = compute_residual_variance(f, sigma, error_type="combined")
        assert result == pytest.approx(expected, rel=1e-12)


# ============================================================================
# Section 8 — AIC / BIC against first-principles computation
# ============================================================================


@pytest.mark.unit
class TestAICBICFirstPrinciples:
    """
    AIC = OFV + 2*k; BIC = OFV + k*ln(n).
    These are verified against direct arithmetic, not against other
    software, because they follow from the definition in Burnham & Anderson (2002).
    """

    def _make_result(self, ofv: float, k: int, n: int) -> EstimationResult:
        r = EstimationResult(
            theta_final=np.zeros(k),
            omega_final=np.zeros((0, 0)),
            sigma_final=np.zeros((0, 0)),
            ofv=ofv,
            n_observations=n,
        )
        r._n_parameters = k
        return r

    @pytest.mark.parametrize(
        "ofv, k, n",
        [
            (100.0, 5, 200),
            (200.0, 10, 500),
            (50.0, 3, 100),
            (0.0, 1, 1000),
        ],
    )
    def test_aic_formula(self, ofv: float, k: int, n: int):
        r = self._make_result(ofv, k, n)
        assert r.aic == pytest.approx(ofv + 2.0 * k, rel=1e-12)

    @pytest.mark.parametrize(
        "ofv, k, n",
        [
            (100.0, 5, 200),
            (200.0, 10, 500),
            (50.0, 3, 100),
        ],
    )
    def test_bic_formula(self, ofv: float, k: int, n: int):
        r = self._make_result(ofv, k, n)
        assert r.bic == pytest.approx(ofv + math.log(n) * k, rel=1e-12)

    def test_bic_is_larger_than_aic_for_n_greater_than_8(self):
        """
        BIC > AIC when n > e^2 ≈ 7.4 (i.e., n >= 8 for integer n).
        Reference: Burnham & Anderson (2002), p. 66.
        """
        for n in [10, 50, 200, 1000]:
            r = self._make_result(ofv=100.0, k=5, n=n)
            assert r.bic > r.aic, f"BIC should exceed AIC for n={n}"

    def test_lrt_ofv_difference_chi_squared_consistency(self):
        """
        The likelihood ratio test (LRT) statistic = OFV_null - OFV_alt
        should follow chi-squared distribution with df = Δk under H0.
        Verify: for a 1 d.f. test, the critical value at p=0.05 is 3.84.
        Reference: Beal & Sheiner (1992) NONMEM Users Guide.
        """
        from scipy.stats import chi2

        alpha = 0.05
        df = 1
        critical_value = chi2.ppf(1 - alpha, df=df)
        assert critical_value == pytest.approx(3.8415, abs=0.001)

        # LRT: reject H0 if delta_OFV > critical_value
        delta_ofv_significant = 4.0
        delta_ofv_not = 3.0
        assert delta_ofv_significant > critical_value, "Expected to reject H0"
        assert delta_ofv_not < critical_value, "Expected to fail to reject H0"


# ============================================================================
# Section 9 — ETA / EPS shrinkage formula (Karlsson & Sheiner definition)
# ============================================================================


@pytest.mark.unit
class TestShrinkageFormulas:
    """
    ETA shrinkage_k = 1 - SD(EBE_ik) / sqrt(omega_kk)

    When EBEs are drawn from N(0, omega_kk), the expected shrinkage is 0.
    When all EBEs = 0, the shrinkage is 1.0 (100%).

    Reference: Karlsson & Sheiner (1993) J. Pharmacokinet. Biopharm. 21(6):735-750.
    """

    def _result_with_etas(
        self, eta_matrix: np.ndarray, omega_diag: list[float]
    ) -> EstimationResult:
        omega = np.diag(omega_diag)
        post_hoc = {i + 1: eta_matrix[i] for i in range(len(eta_matrix))}
        return EstimationResult(
            theta_final=np.array([1.0]),
            omega_final=omega,
            sigma_final=np.diag([0.05]),
            ofv=100.0,
            post_hoc_etas=post_hoc,
        )

    def test_zero_shrinkage_exact(self):
        """
        When EBEs have exactly the same SD as sqrt(omega_kk), shrinkage = 0.
        Use deterministic EBEs with SD exactly equal to sqrt(omega).
        """
        n = 50
        omega_kk = 0.16  # SD = 0.4
        # Construct EBEs with exactly SD = 0.4 using a known distribution
        etas = np.linspace(-0.4, 0.4, n + 1)[:-1]  # symmetric, near 0.4 SD
        etas = etas - etas.mean()  # center at 0
        etas = etas * 0.4 / float(np.std(etas, ddof=1))  # rescale to SD = 0.4 exactly

        r = self._result_with_etas(etas.reshape(-1, 1), [omega_kk])
        r.compute_shrinkage()
        assert r.eta_shrinkage[0] == pytest.approx(0.0, abs=1e-10)

    def test_full_shrinkage_exact(self):
        """When all EBEs = 0, SD(EBE) = 0, shrinkage = 1.0."""
        etas = np.zeros((20, 1))
        r = self._result_with_etas(etas, [0.25])
        r.compute_shrinkage()
        assert r.eta_shrinkage[0] == pytest.approx(1.0, abs=1e-12)

    def test_half_shrinkage_exact(self):
        """
        If SD(EBE) = 0.5 * sqrt(omega_kk), shrinkage = 0.5.
        Use deterministic EBEs.
        """
        omega_kk = 0.36  # sqrt = 0.6
        target_sd = 0.5 * math.sqrt(omega_kk)  # = 0.3

        n = 40
        etas = np.linspace(-target_sd, target_sd, n + 1)[:-1]
        etas = etas - etas.mean()
        etas = etas * target_sd / float(np.std(etas, ddof=1))

        r = self._result_with_etas(etas.reshape(-1, 1), [omega_kk])
        r.compute_shrinkage()
        assert r.eta_shrinkage[0] == pytest.approx(0.5, abs=1e-10)

    def test_eps_shrinkage_sd_formula(self):
        """
        EPS shrinkage = 1 - SD(IWRES, ddof=1).
        For IWRES all equal to r, SD = 0, shrinkage = 1.
        For IWRES ~ standard normal, shrinkage ≈ 0.
        """
        # All identical IWRES → SD = 0 → shrinkage = 1
        iwres_flat = np.ones(50) * 0.5
        r = self._result_with_etas(np.zeros((5, 1)), [0.1])
        r.compute_shrinkage(iwres=iwres_flat)
        assert r.eps_shrinkage[0] == pytest.approx(1.0, abs=1e-10)

    def test_eps_shrinkage_exact_known_sd(self):
        """
        Use IWRES with exactly known SD to verify the formula.
        """
        n = 100
        target_sd = 0.7
        iwres = np.linspace(-target_sd, target_sd, n + 1)[:-1]
        iwres = iwres - iwres.mean()
        iwres = iwres * target_sd / float(np.std(iwres, ddof=1))

        r = self._result_with_etas(np.zeros((5, 1)), [0.1])
        r.compute_shrinkage(iwres=iwres)
        expected = 1.0 - target_sd
        assert r.eps_shrinkage[0] == pytest.approx(expected, abs=1e-10)


# ============================================================================
# Section 10 — Partial AUC log-interpolation exact values
# ============================================================================


@pytest.mark.unit
class TestNCALogInterpolation:
    """
    Verify partial AUC using log interpolation against the exact integral
    for a monoexponential profile.

    For C(t) = C0 * exp(-K*t), the AUC from t1 to t2 is:
        AUC(t1, t2) = C0/K * (exp(-K*t1) - exp(-K*t2))

    Log interpolation at interior boundary should recover this exactly.
    Reference: Gabrielsson & Weiner (2006), Appendix on trapezoidal methods.
    """

    K = 0.15
    C0 = 20.0
    TIMES = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])

    @property
    def _conc(self) -> np.ndarray:
        return self.C0 * np.exp(-self.K * self.TIMES)

    def _exact_auc(self, t1: float, t2: float) -> float:
        return self.C0 / self.K * (math.exp(-self.K * t1) - math.exp(-self.K * t2))

    def test_partial_auc_at_observed_boundaries(self):
        """
        Partial AUC between two observed time points must equal the exact integral.
        """
        engine = NCAEngine(auc_method="linear-log")
        for t1, t2 in [(0.0, 4.0), (1.0, 8.0), (2.0, 12.0), (4.0, 24.0)]:
            partial = engine.compute_partial_auc(self.TIMES, self._conc, t1, t2, method="log")
            exact = self._exact_auc(t1, t2)
            assert partial == pytest.approx(exact, rel=1e-10), (
                f"Partial AUC({t1},{t2}) = {partial:.6f}, exact = {exact:.6f}"
            )

    def test_partial_auc_at_interpolated_boundaries(self):
        """
        When t1 or t2 falls between observed points, log interpolation is used.
        The partial AUC should still be very close to the exact integral.
        """
        engine = NCAEngine(auc_method="linear-log")
        # t1 = 1.5 (between 1 and 2), t2 = 6.0 (between 4 and 8)
        partial = engine.compute_partial_auc(self.TIMES, self._conc, 1.5, 6.0, method="log")
        exact = self._exact_auc(1.5, 6.0)
        # Should be exact because log interpolation preserves monoexponential shape
        assert partial == pytest.approx(exact, rel=1e-10)


# ============================================================================
# Section 9 — Multi-dose accumulation / steady-state (H-6)
# ============================================================================


@pytest.mark.unit
class TestMultiDoseSteadyState:
    """
    Verify that superposition of N equally-spaced IV bolus doses (ADVAN1) and
    oral doses (ADVAN2) converges to the analytical steady-state.

    At steady state with dosing interval τ:
      C_trough = (D/V) * exp(-K*τ) / (1 - exp(-K*τ))
      C_peak   = (D/V) / (1 - exp(-K*τ))

    Reference: Rowland & Tozer, "Clinical Pharmacokinetics and Pharmacodynamics",
    4th ed., Chapter 17.
    """

    K, V = 0.15, 10.0
    DOSE, TAU = 100.0, 12.0
    N_DOSES = 30  # 30 doses >> 5 half-lives → within 1% of SS

    @staticmethod
    def _advan1_peak_trough(k: float, v: float, dose: float, tau: float, n: int) -> tuple[float, float]:
        """Simulate ADVAN1 and return peak/trough after the n-th dose."""
        times_all = []
        for i in range(n):
            t0 = i * tau
            times_all.extend([t0 + 1e-9, t0 + tau - 1e-9])
        obs_times = np.array(times_all)
        doses = [DoseEvent(time=i * tau, amount=dose, compartment=1) for i in range(n)]
        sol = ADVAN1().solve({"K": k, "V": v}, doses, obs_times)
        ipred = sol.ipred
        peak = float(ipred[-2])   # just after last dose
        trough = float(ipred[-1])  # just before next dose
        return peak, trough

    def test_advan1_trough_converges_to_ss(self):
        """After 30 doses, ADVAN1 trough must be within 1% of the analytical SS trough."""
        c_trough_ss = (self.DOSE / self.V) * math.exp(-self.K * self.TAU) / (
            1.0 - math.exp(-self.K * self.TAU)
        )
        _, trough = self._advan1_peak_trough(self.K, self.V, self.DOSE, self.TAU, self.N_DOSES)
        assert trough == pytest.approx(c_trough_ss, rel=0.01), (
            f"ADVAN1 trough after {self.N_DOSES} doses = {trough:.4f}; "
            f"SS analytical = {c_trough_ss:.4f}"
        )

    def test_advan1_peak_converges_to_ss(self):
        """After 30 doses, ADVAN1 peak must be within 1% of the analytical SS peak."""
        c_peak_ss = (self.DOSE / self.V) / (1.0 - math.exp(-self.K * self.TAU))
        peak, _ = self._advan1_peak_trough(self.K, self.V, self.DOSE, self.TAU, self.N_DOSES)
        assert peak == pytest.approx(c_peak_ss, rel=0.01), (
            f"ADVAN1 peak after {self.N_DOSES} doses = {peak:.4f}; "
            f"SS analytical = {c_peak_ss:.4f}"
        )

    def test_advan1_single_dose_to_ss_ratio(self):
        """
        Accumulation ratio R = C_peak_ss / C_peak_single = 1 / (1 - exp(-K*tau)).
        ADVAN1 simulated ratio must match within 1%.
        Reference: Rowland & Tozer, Table 17-1.
        """
        r_ss_exact = 1.0 / (1.0 - math.exp(-self.K * self.TAU))
        # Single dose peak (1 dose at t=0, measure at t=ε)
        t_eps = np.array([1e-9])
        sol1 = ADVAN1().solve(
            {"K": self.K, "V": self.V},
            [DoseEvent(time=0.0, amount=self.DOSE, compartment=1)],
            t_eps,
        )
        c_peak_single = float(sol1.ipred[0])
        peak_ss, _ = self._advan1_peak_trough(self.K, self.V, self.DOSE, self.TAU, self.N_DOSES)
        r_simulated = peak_ss / c_peak_single
        assert r_simulated == pytest.approx(r_ss_exact, rel=0.01), (
            f"Accumulation ratio {r_simulated:.4f} vs exact {r_ss_exact:.4f}"
        )

    def test_advan2_trough_converges_to_ss(self):
        """
        ADVAN2 multi-dose oral: trough after 30 doses must match analytical SS.

        At SS for oral dosing with 1-cmt model:
          C_trough = (F*D/V) * KA/(KA-K) * [exp(-K*tau)/(1-exp(-K*tau))
                                              - exp(-KA*tau)/(1-exp(-KA*tau))]
        Reference: Rowland & Tozer, Chapter 17, Eq. 17-4.
        """
        KA, K, V = 1.2, 0.15, 10.0
        DOSE, TAU, N = 200.0, 12.0, 30

        doses = [DoseEvent(time=i * TAU, amount=DOSE, compartment=1) for i in range(N)]
        # Observe just before the (N+1)-th dose — trough after N-th dose
        t_trough = np.array([(N - 1) * TAU + TAU - 1e-9])
        sol = ADVAN2().solve({"KA": KA, "K": K, "V": V}, doses, t_trough)
        c_trough_sim = float(sol.ipred[0])

        # Analytical SS trough for oral 1-cmt
        factor = KA / (KA - K)
        c_trough_ss = (DOSE / V) * factor * (
            math.exp(-K * TAU) / (1.0 - math.exp(-K * TAU))
            - math.exp(-KA * TAU) / (1.0 - math.exp(-KA * TAU))
        )
        assert c_trough_sim == pytest.approx(c_trough_ss, rel=0.01), (
            f"ADVAN2 trough after {N} doses = {c_trough_sim:.4f}; "
            f"SS analytical = {c_trough_ss:.4f}"
        )
