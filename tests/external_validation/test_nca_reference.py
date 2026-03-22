"""
NCA reference validation against closed-form analytical solutions.

No external software required. All reference values are derived from
exact mathematical integrals.

Profiles tested
---------------
1. Monoexponential IV bolus
     C(t) = C0 * exp(-λ·t)
     AUC∞ = C0/λ  (exact)
     t½   = ln(2)/λ  (exact)
     CL   = Dose/AUC∞ = Dose*λ/C0  (exact)
     Vd   = CL/λ  (exact)

2. Biexponential IV bolus (2-compartment)
     C(t) = A·exp(-α·t) + B·exp(-β·t)
     AUC∞ = A/α + B/β  (exact)
     t½(β) = ln(2)/β  (terminal half-life)

3. One-compartment oral (Bateman function)
     C(t) = (F·D·KA)/(V·(KA-K)) · (exp(-K·t) - exp(-KA·t))
     Tmax = ln(KA/K) / (KA-K)  (exact)
     Cmax = C(Tmax)  (exact analytical)

Tolerances are tight (<0.5%) because these are pure numerical integrals
against exact formulas — any larger error indicates a bug.

References
----------
Gibaldi M & Perrier D. (1982). Pharmacokinetics. 2nd ed. Marcel Dekker.
Rowland M & Tozer TN. (1995). Clinical Pharmacokinetics. 3rd ed.
Gabrielsson J & Weiner D. (2006). PK and PD Data Analysis. 4th ed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from openpkpd.nca.nca import NCAEngine, NCAParameters

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dense_times(t_end: float, n: int = 500) -> np.ndarray:
    """Dense time grid for accurate numerical integration."""
    return np.concatenate([[0.0], np.linspace(0.01, t_end, n)])


def _run_nca(
    times: np.ndarray, conc: np.ndarray, dose: float = 1.0, route: str = "iv"
) -> NCAParameters:
    engine = NCAEngine()
    return engine.compute_subject(times=times, conc=conc, dose=dose, route=route)


# ---------------------------------------------------------------------------
# 1. Monoexponential IV bolus
# ---------------------------------------------------------------------------


class TestMonoexponentialIVBolus:
    """Monoexponential IV: C(t) = C0 * exp(-λ*t)."""

    LAMBDA = 0.15  # h⁻¹
    C0 = 10.0  # mg/L (= Dose / V)
    DOSE = 500.0  # mg
    V = DOSE / C0  # = 50 L
    CL_TRUE = LAMBDA * V  # = 7.5 L/h

    AUC_TRUE = C0 / LAMBDA  # = 66.67 h·mg/L
    THALF_TRUE = math.log(2) / LAMBDA  # = 4.621 h
    CL_NCA_TRUE = DOSE / AUC_TRUE  # = 7.5 L/h
    VD_NCA_TRUE = CL_NCA_TRUE / LAMBDA  # = 50 L

    @pytest.fixture(scope="class")
    def nca(self):
        times = _dense_times(48.0)
        conc = self.C0 * np.exp(-self.LAMBDA * times)
        return _run_nca(times, conc, dose=self.DOSE)

    def test_auc_inf(self, nca):
        """AUC∞ should equal C0/λ within 0.5%."""
        rel_err = abs(nca.auc_inf - self.AUC_TRUE) / self.AUC_TRUE
        assert rel_err < 0.005, (
            f"AUC∞={nca.auc_inf:.4f}, exact={self.AUC_TRUE:.4f} (err={rel_err:.1%})"
        )

    def test_thalf(self, nca):
        """t½ = ln(2)/λ within 0.5%."""
        rel_err = abs(nca.t_half - self.THALF_TRUE) / self.THALF_TRUE
        assert rel_err < 0.005, (
            f"t½={nca.t_half:.4f}, exact={self.THALF_TRUE:.4f} (err={rel_err:.1%})"
        )

    def test_clearance(self, nca):
        """CL = Dose/AUC∞ within 0.5%."""
        rel_err = abs(nca.cl_f - self.CL_NCA_TRUE) / self.CL_NCA_TRUE
        assert rel_err < 0.005, (
            f"CL={nca.cl_f:.4f}, exact={self.CL_NCA_TRUE:.4f} (err={rel_err:.1%})"
        )

    def test_volume_of_distribution(self, nca):
        """Vd = CL/λ within 1%."""
        rel_err = abs(nca.vz_f - self.VD_NCA_TRUE) / self.VD_NCA_TRUE
        assert rel_err < 0.01, (
            f"Vd={nca.vz_f:.4f}, exact={self.VD_NCA_TRUE:.4f} (err={rel_err:.1%})"
        )

    def test_lambda_z(self, nca):
        """Terminal λz should recover true elimination rate within 0.1%."""
        rel_err = abs(nca.lambda_z - self.LAMBDA) / self.LAMBDA
        assert rel_err < 0.001, (
            f"λz={nca.lambda_z:.5f}, exact={self.LAMBDA:.5f} (err={rel_err:.2%})"
        )

    def test_cmax_is_c0(self, nca):
        """Cmax for IV bolus should equal C0 (first observed concentration)."""
        rel_err = abs(nca.cmax - self.C0) / self.C0
        assert rel_err < 0.01, f"Cmax={nca.cmax:.4f}, C0={self.C0:.4f} (err={rel_err:.1%})"


# ---------------------------------------------------------------------------
# 2. Biexponential IV bolus
# ---------------------------------------------------------------------------


class TestBiexponentialIVBolus:
    """Biexponential IV: C(t) = A*exp(-α*t) + B*exp(-β*t).

    Rowland & Tozer (1995) Example: α=1.5 h⁻¹, β=0.15 h⁻¹.
    """

    ALPHA = 1.5  # h⁻¹ (distribution)
    BETA = 0.15  # h⁻¹ (elimination)
    A = 8.0  # mg/L (intercept, fast phase)
    B = 4.0  # mg/L (intercept, slow phase)
    DOSE = 500.0  # mg

    AUC_TRUE = A / ALPHA + B / BETA  # = 5.333 + 26.667 = 32.0 h·mg/L
    THALF_TRUE = math.log(2) / BETA  # = 4.621 h (terminal half-life)
    CL_TRUE = DOSE / AUC_TRUE  # = 15.625 L/h

    @pytest.fixture(scope="class")
    def nca(self):
        times = _dense_times(60.0, n=600)
        conc = self.A * np.exp(-self.ALPHA * times) + self.B * np.exp(-self.BETA * times)
        return _run_nca(times, conc, dose=self.DOSE)

    def test_auc_inf_biexponential(self, nca):
        """AUC∞ = A/α + B/β within 1%."""
        rel_err = abs(nca.auc_inf - self.AUC_TRUE) / self.AUC_TRUE
        assert rel_err < 0.01, (
            f"AUC∞={nca.auc_inf:.4f}, exact={self.AUC_TRUE:.4f} (err={rel_err:.1%})"
        )

    def test_terminal_thalf_biexponential(self, nca):
        """Terminal t½ = ln(2)/β within 1%."""
        rel_err = abs(nca.t_half - self.THALF_TRUE) / self.THALF_TRUE
        assert rel_err < 0.01, (
            f"t½={nca.t_half:.4f}, exact={self.THALF_TRUE:.4f} (err={rel_err:.1%})"
        )

    def test_terminal_lambda_z_biexponential(self, nca):
        """Terminal λz = β within 0.5%."""
        rel_err = abs(nca.lambda_z - self.BETA) / self.BETA
        assert rel_err < 0.005, f"λz={nca.lambda_z:.5f}, exact={self.BETA:.5f} (err={rel_err:.2%})"

    def test_clearance_biexponential(self, nca):
        """CL = Dose/AUC∞ within 1%."""
        rel_err = abs(nca.cl_f - self.CL_TRUE) / self.CL_TRUE
        assert rel_err < 0.01, f"CL={nca.cl_f:.4f}, exact={self.CL_TRUE:.4f} (err={rel_err:.1%})"


# ---------------------------------------------------------------------------
# 3. One-compartment oral (Bateman function)
# ---------------------------------------------------------------------------


class TestOneCmtOral:
    """1-cmt oral: C(t) = (F·D·KA)/(V·(KA-K)) * (exp(-K*t) - exp(-KA*t)).

    Analytical Tmax = ln(KA/K) / (KA - K).
    Analytical Cmax = C(Tmax).
    """

    KA = 1.5  # h⁻¹
    K = 0.10  # h⁻¹ (elimination)
    V = 30.0  # L
    F = 1.0  # bioavailability
    DOSE = 320.0  # mg

    TMAX_TRUE = math.log(KA / K) / (KA - K)  # ≈ 1.636 h

    @classmethod
    def _cmax_true(cls) -> float:
        return (
            cls.F
            * cls.DOSE
            * cls.KA
            / (cls.V * (cls.KA - cls.K))
            * (math.exp(-cls.K * cls.TMAX_TRUE) - math.exp(-cls.KA * cls.TMAX_TRUE))
        )

    @pytest.fixture(scope="class")
    def nca(self):
        times = _dense_times(36.0, n=720)
        coeff = self.F * self.DOSE * self.KA / (self.V * (self.KA - self.K))
        conc = coeff * (np.exp(-self.K * times) - np.exp(-self.KA * times))
        # t=0 has conc=0, shift to avoid log issues in lambda_z
        times_nonzero = times[times > 0]
        conc_nonzero = conc[times > 0]
        return _run_nca(times_nonzero, conc_nonzero, dose=self.DOSE, route="oral")

    def test_tmax_oral(self, nca):
        """Tmax should equal ln(KA/K)/(KA-K) within 5% (limited by time grid)."""
        rel_err = abs(nca.tmax - self.TMAX_TRUE) / self.TMAX_TRUE
        assert rel_err < 0.05, (
            f"Tmax={nca.tmax:.4f}, exact={self.TMAX_TRUE:.4f} (err={rel_err:.1%})"
        )

    def test_cmax_oral(self, nca):
        """Cmax should match analytical value within 1%."""
        cmax_true = self._cmax_true()
        rel_err = abs(nca.cmax - cmax_true) / cmax_true
        assert rel_err < 0.01, f"Cmax={nca.cmax:.4f}, exact={cmax_true:.4f} (err={rel_err:.1%})"

    def test_terminal_thalf_oral(self, nca):
        """Terminal t½ should recover elimination half-life ln(2)/K within 1%."""
        thalf_true = math.log(2) / self.K
        rel_err = abs(nca.t_half - thalf_true) / thalf_true
        assert rel_err < 0.01, f"t½={nca.t_half:.4f}, exact={thalf_true:.4f} (err={rel_err:.1%})"

    def test_auc_inf_oral(self, nca):
        """AUC∞ for oral 1-cmt = F*D / (V*K) = F*D*CL within 1%."""
        auc_true = self.F * self.DOSE / (self.V * self.K)  # = 106.67 h·mg/L
        rel_err = abs(nca.auc_inf - auc_true) / auc_true
        assert rel_err < 0.01, f"AUC∞={nca.auc_inf:.4f}, exact={auc_true:.4f} (err={rel_err:.1%})"


# ---------------------------------------------------------------------------
# 4. Dose-linearity invariant
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestDoseLinearity:
    """Dose-normalised NCA parameters should be invariant to dose.

    For linear PK, doubling the dose should double Cmax and AUC.
    """

    LAMBDA = 0.12
    C0_PER_UNIT_DOSE = 0.2  # mg/L per mg dose → C0 = dose * 0.2

    def _run_at_dose(self, dose: float) -> NCAParameters:
        times = _dense_times(40.0)
        c0 = dose * self.C0_PER_UNIT_DOSE
        conc = c0 * np.exp(-self.LAMBDA * times)
        return _run_nca(times, conc, dose=dose, route="iv")

    def test_auc_scales_linearly_with_dose(self):
        """AUC should double when dose doubles."""
        r1 = self._run_at_dose(100.0)
        r2 = self._run_at_dose(200.0)
        ratio = r2.auc_inf / r1.auc_inf
        assert abs(ratio - 2.0) < 0.02, f"AUC ratio={ratio:.4f} (expected 2.0 for dose-linear PK)"

    def test_cmax_scales_linearly_with_dose(self):
        """Cmax should double when dose doubles."""
        r1 = self._run_at_dose(100.0)
        r2 = self._run_at_dose(200.0)
        ratio = r2.cmax / r1.cmax
        assert abs(ratio - 2.0) < 0.02, f"Cmax ratio={ratio:.4f} (expected 2.0 for dose-linear PK)"

    def test_norm_auc_invariant_to_dose(self):
        """Dose-normalised AUC (auc_inf/dose) should be identical across doses."""
        r1 = self._run_at_dose(100.0)
        r2 = self._run_at_dose(300.0)
        norm1 = r1.auc_inf / 100.0
        norm2 = r2.auc_inf / 300.0
        rel_diff = abs(norm1 - norm2) / norm2
        assert rel_diff < 0.01, (
            f"Normalised AUC differs: dose=100→{norm1:.6f}/dose, dose=300→{norm2:.6f}/dose"
        )

    def test_thalf_invariant_to_dose(self):
        """t½ should not change with dose (concentration-independent kinetics)."""
        r1 = self._run_at_dose(50.0)
        r2 = self._run_at_dose(500.0)
        rel_diff = abs(r1.t_half - r2.t_half) / r2.t_half
        assert rel_diff < 0.005, f"t½ changed with dose: {r1.t_half:.4f} vs {r2.t_half:.4f}"


# ---------------------------------------------------------------------------
# 5. Log-linear vs linear trapezoidal accuracy
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestTrapezoidalAccuracy:
    """Trapezoidal rule accuracy vs exact integral.

    AUC error should be <1% for standard sampling intervals.
    The log-linear rule is more accurate in the declining phase.
    """

    LAMBDA = 0.10  # slow elimination, tests the terminal phase well
    C0 = 5.0

    def test_auc_last_accuracy(self):
        """AUC_last with 12 sampling timepoints should be within 0.5% of exact."""
        # Clinically realistic sampling: 0, 0.5, 1, 2, 4, 6, 8, 12, 18, 24, 36, 48
        times = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0])
        conc = self.C0 * np.exp(-self.LAMBDA * times)
        nca = _run_nca(times, conc, dose=100.0, route="iv")

        # Exact AUC0-48
        exact_auc_last = (self.C0 / self.LAMBDA) * (1 - math.exp(-self.LAMBDA * 48.0))
        rel_err = abs(nca.auc_last - exact_auc_last) / exact_auc_last
        assert rel_err < 0.005, (
            f"AUC_last={nca.auc_last:.4f}, exact={exact_auc_last:.4f} (err={rel_err:.1%})"
        )
