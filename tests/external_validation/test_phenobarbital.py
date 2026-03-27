"""
External validation: Phenobarbital neonatal population PK.

Reference: Grasela TH Jr, Donn SM (1985). Neonatal population pharmacokinetics of
phenobarbital derived from routine clinical data. Dev Pharmacol Ther, 8(6):374-83.

Published parameters (NONMEM FO, 59 neonates):
  CL = 0.0047 L/h/kg  (BSV ~19%)
  V  = 0.96  L/kg     (BSV ~16%)
  t½ ≈ 141 h

The OpenPKPD test uses a simulated dataset generated with these exact population
parameters (data/phenobarbital_simulated.csv, seed=42), so the estimated parameters
should recover the simulation truth within expected variability.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

DATA_DIR = pathlib.Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "phenobarbital_simulated.csv"
REFERENCE_FILE = pathlib.Path(__file__).parent / "reference" / "grasela1985_phenobarbital_fo.json"

# Published simulation-truth parameters
TRUE_CL_PER_KG = 0.0047  # L/h/kg
TRUE_V_PER_KG = 0.96  # L/kg
TRUE_OMEGA_CL = 0.0361  # variance (BSV 19%)
TRUE_OMEGA_V = 0.0256  # variance (BSV 16%)
TRUE_SIGMA = 0.04  # proportional residual variance


def _load_reference() -> dict:
    return json.loads(REFERENCE_FILE.read_text())


@pytest.fixture(scope="module")
def reference() -> dict:
    if not REFERENCE_FILE.exists():
        pytest.skip(f"Phenobarbital reference not found: {REFERENCE_FILE}")
    return _load_reference()


@pytest.fixture(scope="module")
def result():
    if not DATA_FILE.exists():
        pytest.skip(f"Phenobarbital data not found: {DATA_FILE}")
    try:
        from openpkpd import ModelBuilder
        from openpkpd.data.dataset import NONMEMDataset
    except ImportError:
        pytest.skip("openpkpd not importable")

    ds = NONMEMDataset.from_csv(str(DATA_FILE))

    pk_code = """
TVCL = THETA(1) * WT
TVV  = THETA(2) * WT
CL   = TVCL * EXP(ETA(1))
V    = TVV  * EXP(ETA(2))
K    = CL / V
S1   = V
"""
    error_code = """
IPRED = F
W     = IPRED * THETA(3)
IRES  = DV - IPRED
IWRES = IRES / W
Y     = IPRED + W * EPS(1)
"""
    model = (
        ModelBuilder()
        .problem("Phenobarbital neonatal population PK — Grasela 1985")
        .dataset(ds)
        .covariates(["WT"])  # weight-based allometric scaling
        .subroutines(advan=1, trans=1)  # TRANS1: supply K and V directly
        .pk(pk_code)
        .error(error_code)
        .theta(
            [
                (0.001, TRUE_CL_PER_KG, 0.05),  # THETA(1): CL/kg
                (0.1, TRUE_V_PER_KG, 5.0),  # THETA(2): V/kg
                (0.001, 0.1, 1.0),  # THETA(3): prop error SD
            ]
        )
        .omega([[TRUE_OMEGA_CL, 0], [0, TRUE_OMEGA_V]])
        .sigma([[1.0]])
        .estimation(method="FO", maxeval=500)
        .build()
    )
    return model.fit()


class TestPhenobarbitalParameterRecovery:
    """
    Verifies that FO estimation on the simulated phenobarbital dataset
    recovers the simulation-truth parameters within expected tolerances.
    """

    def test_ofv_is_finite(self, result):
        assert np.isfinite(result.ofv), "OFV must be finite"

    def test_ofv_reasonable(self, result):
        """OFV should be in a plausible range for 171 sparse observations."""
        assert result.ofv < 5000.0, f"OFV={result.ofv:.1f} is unexpectedly large (penalty value?)"

    def test_cl_per_kg_recovered(self, result):
        """CL/kg should recover within 30% of simulation truth."""
        est_cl = float(result.theta_final[0])
        pct = 100.0 * abs(est_cl - TRUE_CL_PER_KG) / TRUE_CL_PER_KG
        assert pct < 30.0, f"CL/kg={est_cl:.5f} is {pct:.1f}% from truth {TRUE_CL_PER_KG}"

    def test_v_per_kg_recovered(self, result):
        """V/kg should recover within 25% of simulation truth."""
        est_v = float(result.theta_final[1])
        pct = 100.0 * abs(est_v - TRUE_V_PER_KG) / TRUE_V_PER_KG
        assert pct < 25.0, f"V/kg={est_v:.4f} is {pct:.1f}% from truth {TRUE_V_PER_KG}"

    def test_halflife_plausible(self, result):
        """Derived half-life should be in the neonatal phenobarbital range (50–300 h)."""
        cl = float(result.theta_final[0])
        v = float(result.theta_final[1])
        if cl > 0 and v > 0:
            halflife = v * np.log(2) / cl
            assert 50.0 < halflife < 300.0, (
                f"Half-life={halflife:.1f} h outside expected range (50–300 h)"
            )

    def test_omega_cl_positive(self, result):
        assert result.omega_final[0, 0] > 0, "IIV for CL must be positive"

    def test_omega_v_positive(self, result):
        assert result.omega_final[1, 1] > 0, "IIV for V must be positive"

    def test_omega_psd(self, result):
        eigvals = np.linalg.eigvalsh(result.omega_final)
        assert np.all(eigvals >= -1e-8), f"OMEGA not PSD: {eigvals}"

    def test_sigma_positive(self, result):
        assert result.sigma_final[0, 0] > 0, "Residual variance must be positive"

    def test_converged(self, result):
        assert result.converged, f"FO did not converge: {result.message}"

    def test_sigma_tracks_simulation_reference(self, result, reference):
        sigma_ref = float(reference["openpkpd_simulation_parameters"]["sigma_proportional"])
        sigma_est = float(result.theta_final[2])
        ratio = sigma_est / sigma_ref
        assert 0.25 < ratio < 4.0, (
            f"Residual SD={sigma_est:.4f} should remain within the same order of magnitude as "
            f"the simulation reference {sigma_ref:.4f} (ratio={ratio:.2f})"
        )

    def test_dataset_subject_count_matches_reference(self, result, reference):
        expected_n = int(reference["openpkpd_simulation_parameters"]["n_subjects"])
        n_subjects = len(result.post_hoc_etas)
        assert n_subjects == expected_n, f"Expected {expected_n} subjects, got {n_subjects}"


class TestPhenobarbitalHallifeVsLiterature:
    """
    Compare derived half-life against Grasela & Donn (1985) published value.
    Published: t½ ≈ 141 h for a typical neonate.
    """

    def test_halflife_close_to_literature(self, result):
        cl = float(result.theta_final[0])  # CL/kg
        v = float(result.theta_final[1])  # V/kg
        if cl <= 0 or v <= 0:
            pytest.skip("Negative parameters — cannot compute half-life")
        halflife = v * np.log(2) / cl
        lit_halflife = 141.6  # h
        pct = 100.0 * abs(halflife - lit_halflife) / lit_halflife
        assert pct < 35.0, (
            f"Half-life={halflife:.1f} h is {pct:.1f}% from literature {lit_halflife:.1f} h "
            f"(Grasela & Donn 1985)"
        )

    def test_half_life_matches_reference_json(self, result, reference):
        cl = float(result.theta_final[0])
        v = float(result.theta_final[1])
        if cl <= 0 or v <= 0:
            pytest.skip("Negative parameters — cannot compute half-life")
        halflife = v * np.log(2) / cl
        expected = float(reference["derived"]["halflife_h"])
        assert halflife == pytest.approx(expected, rel=0.35)


class TestPhenobarbitalBSVVsLiterature:
    """Compare recovered log-normal IIV against the published BSV percentages."""

    @staticmethod
    def _omega_to_cv_pct(omega_var: float) -> float:
        return float(np.sqrt(np.exp(omega_var) - 1.0) * 100.0)

    def test_cl_bsv_tracks_literature(self, result, reference):
        observed = self._omega_to_cv_pct(float(result.omega_final[0, 0]))
        expected = float(reference["bsv_cv_pct"]["CL"])
        assert observed == pytest.approx(expected, abs=8.0)

    def test_v_bsv_tracks_literature(self, result, reference):
        observed = self._omega_to_cv_pct(float(result.omega_final[1, 1]))
        expected = float(reference["bsv_cv_pct"]["V"])
        assert observed == pytest.approx(expected, abs=8.0)
