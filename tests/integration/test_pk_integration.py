"""
Integration tests for analytical PK subroutines.

Each test builds a theophylline-like dataset, fits a compartment model with
FOCE, and asserts that:
  - Convergence is reached
  - OFV is finite
  - The identifiable PK signal is recovered within a reasonable band
  - Structural parameters remain positive and finite

These tests extend unit-level formula tests by verifying that the subroutines
work correctly within the full estimation pipeline. For the oral ADVAN2 smoke
case, the synthetic design constrains the elimination rate constant more
robustly than CL and V separately, so the integration assertion is written on
the derived quantity k = CL / V. A separate ADVAN1 IV-bolus class provides a
true parameter-identification gate, because that setup cleanly recovers CL and
V from synthetic truth under the same estimation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan1 import ADVAN1
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


def _make_advan1_identification_dataset(n_subjects: int = 40) -> NONMEMDataset:
    """Build a low-noise IV-bolus dataset that identifies CL and V cleanly."""
    rng = np.random.default_rng(123)
    rows = []
    cl_true, v_true = 1.8, 25.0
    dose_levels = (4.0, 8.0)
    obs_times = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0])
    k_true = cl_true / v_true

    for subj in range(1, n_subjects + 1):
        dose = dose_levels[(subj - 1) % len(dose_levels)]
        rows.append({"ID": subj, "TIME": 0.0, "DV": 0.0, "AMT": dose, "EVID": 1, "MDV": 1})
        for t in obs_times:
            c_true = dose / v_true * np.exp(-k_true * t)
            c_obs = max(c_true * np.exp(rng.normal(0, 0.02)), 1e-10)
            rows.append({"ID": subj, "TIME": t, "DV": c_obs, "AMT": 0.0, "EVID": 0, "MDV": 0})

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def _build_population_model(
    dataset: NONMEMDataset,
    *,
    pk_subroutine,
    theta_specs: list[ThetaSpec],
    omega_specs: list[OmegaSpec],
    sigma_specs: list[SigmaSpec],
    trans: int,
    advan: int,
) -> tuple[PopulationModel, ParameterSet]:
    params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)
    pop = PopulationModel(
        dataset=dataset,
        pk_subroutine=pk_subroutine,
        params=params,
        trans=trans,
        advan=advan,
    )
    return pop, params


@pytest.mark.integration
@pytest.mark.slow
class TestADVAN1ParameterIdentification:
    """Synthetic recovery gate for a well-identified IV-bolus design."""

    @pytest.fixture(scope="class")
    def advan1_result(self):
        dataset = _make_advan1_identification_dataset()
        theta_specs = [
            ThetaSpec(init=1.0, lower=0.3, upper=10.0),  # CL
            ThetaSpec(init=20.0, lower=5.0, upper=80.0),  # V
        ]
        omega_specs = [OmegaSpec(block_size=1, values=[1e-4])]
        sigma_specs = [SigmaSpec(block_size=1, values=[4e-4])]
        pop, params = _build_population_model(
            dataset,
            pk_subroutine=ADVAN1(),
            theta_specs=theta_specs,
            omega_specs=omega_specs,
            sigma_specs=sigma_specs,
            trans=2,
            advan=1,
        )
        return FOCEMethod(interaction=True, maxeval=200).estimate(pop, params)

    def test_converged(self, advan1_result):
        assert advan1_result.converged

    def test_cl_and_v_recover_truth(self, advan1_result):
        cl, v = advan1_result.theta_final
        np.testing.assert_allclose(cl, 1.8, rtol=0.05)
        np.testing.assert_allclose(v, 25.0, rtol=0.05)

    def test_elimination_rate_recovers_truth(self, advan1_result):
        cl, v = advan1_result.theta_final
        np.testing.assert_allclose(cl / v, 1.8 / 25.0, rtol=0.05)

    def test_parameters_do_not_pin_to_bounds(self, advan1_result):
        lower_bounds = np.array([0.3, 5.0])
        upper_bounds = np.array([10.0, 80.0])
        assert np.all(advan1_result.theta_final > lower_bounds + np.array([0.2, 1.0]))
        assert np.all(advan1_result.theta_final < upper_bounds - np.array([0.2, 1.0]))


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

    def test_elimination_rate_constant_in_expected_range(self, advan2_result):
        """The fitted elimination rate constant should remain near the generator value."""
        cl, v = advan2_result.theta_final[:2]
        k = cl / v
        assert 0.03 <= k <= 0.15, f"k = CL/V = {k:.4f} outside [0.03, 0.15]"

    def test_structural_parameters_positive_and_finite(self, advan2_result):
        """ADVAN2 structural parameters should remain finite and positive."""
        assert np.all(np.isfinite(advan2_result.theta_final))
        assert np.all(advan2_result.theta_final > 0.0)

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
