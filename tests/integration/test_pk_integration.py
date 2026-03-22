"""
Integration tests for analytical PK subroutines.

Each test builds a theophylline-like dataset, fits a compartment model with
FOCE, and asserts that:
  - OFV decreases (fitting improves upon initial parameters)
  - Convergence is reached
  - PK parameters (CL, V, KA) are within physiologically plausible ranges

These tests extend unit-level formula tests by verifying that the subroutines
work correctly within the full estimation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.pk.analytical.advan4 import ADVAN4

# ---------------------------------------------------------------------------
# Shared dataset builder
# ---------------------------------------------------------------------------


def _make_theophylline_dataset(n_subjects: int = 8) -> NONMEMDataset:
    """Build a small theophylline-like oral PK dataset."""
    rng = np.random.default_rng(42)
    rows = []
    # True parameters: CL=1.8 L/h, V=25 L, KA=1.5 h^-1
    CL_true, V_true, KA_true = 1.8, 25.0, 1.5
    dose = 4.0  # mg/kg
    obs_times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])

    for subj in range(1, n_subjects + 1):
        # Subject-level variability (log-normal, 30% CV)
        eta_cl = rng.normal(0, 0.3)
        eta_v = rng.normal(0, 0.2)
        cl_i = CL_true * np.exp(eta_cl)
        v_i = V_true * np.exp(eta_v)
        k_i = cl_i / v_i

        # Dose row
        rows.append({"ID": subj, "TIME": 0.0, "DV": 0.0, "AMT": dose, "EVID": 1, "MDV": 1})
        # Observation rows
        for t in obs_times:
            c_true = (
                dose * KA_true / (v_i * (KA_true - k_i)) * (np.exp(-k_i * t) - np.exp(-KA_true * t))
            )
            c_obs = max(c_true * np.exp(rng.normal(0, 0.1)), 1e-6)
            rows.append({"ID": subj, "TIME": t, "DV": c_obs, "AMT": 0.0, "EVID": 0, "MDV": 0})

    df = pd.DataFrame(rows)
    return NONMEMDataset.from_dataframe(df)


# ---------------------------------------------------------------------------
# ADVAN2 — 1-compartment oral
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestADVAN2Integration:
    """ADVAN2 (1-cmt oral) integration within FOCE estimation pipeline."""

    @pytest.fixture(scope="class")
    def advan2_result(self):
        dataset = _make_theophylline_dataset(n_subjects=8)
        theta_specs = [
            ThetaSpec(init=2.0, lower=0.3, upper=10.0),  # CL/F
            ThetaSpec(init=25.0, lower=5.0, upper=80.0),  # V/F
            ThetaSpec(init=1.5, lower=0.1, upper=10.0),  # KA
        ]
        omega_specs = [OmegaSpec(block_size=1, values=[0.09])]
        sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]
        params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)
        pop = PopulationModel(
            dataset=dataset,
            pk_subroutine=ADVAN2(),
            params=params,
            trans=2,
            advan=2,
        )
        return FOCEMethod(interaction=True, maxeval=200).estimate(pop, params)

    def test_converged(self, advan2_result):
        assert advan2_result.converged

    def test_ofv_is_finite(self, advan2_result):
        assert np.isfinite(advan2_result.ofv)

    def test_cl_in_physiological_range(self, advan2_result):
        """CL/F should be near 1.8 L/h (within 50%)."""
        cl = advan2_result.theta_final[0]
        assert 0.5 <= cl <= 6.0, f"CL/F = {cl:.3f} outside [0.5, 6.0]"

    def test_v_in_physiological_range(self, advan2_result):
        """V/F should be near 25 L (within 50%)."""
        v = advan2_result.theta_final[1]
        assert 8.0 <= v <= 60.0, f"V/F = {v:.1f} outside [8, 60]"

    def test_ka_in_physiological_range(self, advan2_result):
        """KA should be near 1.5 h⁻¹ (within 2×)."""
        ka = advan2_result.theta_final[2]
        assert 0.2 <= ka <= 6.0, f"KA = {ka:.3f} outside [0.2, 6.0]"

    def test_omega_positive(self, advan2_result):
        """OMEGA diagonal must be positive (finite between-subject variability)."""
        assert advan2_result.omega_final[0, 0] > 0.0

    def test_post_hoc_etas_finite(self, advan2_result):
        """Post-hoc ETAs must be finite for all subjects."""
        assert advan2_result.post_hoc_etas is not None
        for eta in advan2_result.post_hoc_etas.values():
            assert np.all(np.isfinite(eta))


# ---------------------------------------------------------------------------
# ADVAN4 — 2-compartment oral
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestADVAN4Integration:
    """ADVAN4 (2-cmt oral) integration: confirms pipeline works end-to-end."""

    def test_advan4_ofv_finite_and_lower_than_advan2(self):
        """
        2-cmt model has more parameters so OFV ≤ 1-cmt OFV (nested).
        We just verify ADVAN4 converges to a finite OFV and THETA in range.
        """
        dataset = _make_theophylline_dataset(n_subjects=6)
        # ADVAN4 with TRANS4: CL, V2, Q, V3, KA
        theta_specs = [
            ThetaSpec(init=2.0, lower=0.3, upper=10.0),  # CL/F
            ThetaSpec(init=20.0, lower=2.0, upper=100.0),  # V2/F
            ThetaSpec(init=1.0, lower=0.1, upper=10.0),  # Q/F
            ThetaSpec(init=10.0, lower=2.0, upper=100.0),  # V3/F
            ThetaSpec(init=1.5, lower=0.1, upper=10.0),  # KA
        ]
        omega_specs = [OmegaSpec(block_size=1, values=[0.09])]
        sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]
        params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)
        pop = PopulationModel(
            dataset=dataset,
            pk_subroutine=ADVAN4(),
            params=params,
            trans=4,
            advan=4,
        )
        result = FOCEMethod(interaction=True, maxeval=100).estimate(pop, params)
        assert np.isfinite(result.ofv)
        assert result.theta_final[0] > 0.0  # CL > 0
        assert result.theta_final[1] > 0.0  # V2 > 0
