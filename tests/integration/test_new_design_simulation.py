"""
Part 4a: Integration tests for SimulationEngine.simulate_new_design().

Tests:
  - ADVAN6 (ODE) model: simulate_new_design() correctly passes des_callable
    (verifies the engine.py fix from Part 1B).
  - ADVAN2 (analytical) control: same call with analytical model should also
    work (confirms fix didn't break anything).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.pk.ode.advan6 import ADVAN6
from openpkpd.simulation.engine import SimulationEngine

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _build_small_1cmt_dataset(n_subjects: int = 6, seed: int = 99) -> NONMEMDataset:
    """Synthetic 1-compartment IV bolus dataset for fitting."""
    rng = np.random.default_rng(seed)
    cl_pop, v_pop = 2.0, 30.0
    dose = 100.0
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])

    rows = []
    for sid in range(1, n_subjects + 1):
        cl = cl_pop * math.exp(rng.normal(0, 0.2))
        v = v_pop * math.exp(rng.normal(0, 0.15))
        k = cl / v

        rows.append(
            {
                "ID": sid,
                "TIME": 0.0,
                "AMT": dose,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            }
        )
        for t in obs_times:
            c = dose / v * math.exp(-k * t)
            eps = rng.normal(0, 0.08)
            dv = max(c * (1 + eps), 0.001)
            rows.append(
                {
                    "ID": sid,
                    "TIME": t,
                    "AMT": 0.0,
                    "DV": dv,
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                    "RATE": 0.0,
                    "ADDL": 0,
                    "II": 0,
                    "SS": 0,
                }
            )

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def _build_1cmt_oral_dataset(n_subjects: int = 6, seed: int = 11) -> NONMEMDataset:
    """Synthetic 1-compartment oral dataset for ADVAN2 control test."""
    rng = np.random.default_rng(seed)
    ka_pop, cl_pop, v_pop = 1.5, 2.0, 25.0
    dose = 200.0
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])

    rows = []
    for sid in range(1, n_subjects + 1):
        ka = ka_pop * math.exp(rng.normal(0, 0.3))
        cl = cl_pop * math.exp(rng.normal(0, 0.2))
        v = v_pop * math.exp(rng.normal(0, 0.15))
        k = cl / v

        rows.append(
            {
                "ID": sid,
                "TIME": 0.0,
                "AMT": dose,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            }
        )
        for t in obs_times:
            if abs(ka - k) < 1e-6:
                c = dose * ka / v * t * math.exp(-k * t)
            else:
                c = dose * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))
            eps = rng.normal(0, 0.1)
            dv = max(c * (1 + eps), 0.001)
            rows.append(
                {
                    "ID": sid,
                    "TIME": t,
                    "AMT": 0.0,
                    "DV": dv,
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                    "RATE": 0.0,
                    "ADDL": 0,
                    "II": 0,
                    "SS": 0,
                }
            )

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def advan6_engine():
    """
    Fit a 1-cmt IV ODE model (ADVAN6) on synthetic data and return a
    SimulationEngine.  Uses FO (fastest method) to keep test time short.
    """
    from openpkpd.estimation.fo import FOMethod

    dataset = _build_small_1cmt_dataset()
    theta_specs = [
        ThetaSpec(init=2.0, lower=0.5, upper=10.0),  # CL
        ThetaSpec(init=30.0, lower=5.0, upper=80.0),  # V
    ]
    omega_specs = [OmegaSpec(block_size=1, values=[0.04])]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]
    params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)

    # DES: 1-compartment ODE  dA(1)/dt = -CL/V * A(1)
    def des_callable(t, a, pk_params, theta=None, eta=None):
        cl = pk_params.get("CL", pk_params.get("K", 1.0))
        v = pk_params.get("V", 1.0)
        k = cl / v
        return [-k * a[0]]

    # pk_callable: maps theta → pk_params
    def pk_callable(theta, eta, t=0.0):
        cl = theta[0] * math.exp(eta[0] if eta else 0.0)
        v = theta[1]
        return {"CL": cl, "V": v, "K": cl / v}

    advan6 = ADVAN6(n_compartments=1)
    pop_model = PopulationModel(
        dataset=dataset,
        pk_subroutine=advan6,
        params=params,
        trans=1,
        advan=6,
        pk_callable=pk_callable,
        des_callable=des_callable,
    )

    fo = FOMethod(maxeval=200, print_interval=999)
    result = fo.estimate(pop_model, params)
    return SimulationEngine(pop_model, result, seed=7)


@pytest.fixture(scope="module")
def advan2_engine():
    """Fit a 1-cmt oral ADVAN2 model and return a SimulationEngine."""
    from openpkpd.estimation.fo import FOMethod

    dataset = _build_1cmt_oral_dataset()
    theta_specs = [
        ThetaSpec(init=1.5, lower=0.2, upper=6.0),  # KA
        ThetaSpec(init=2.0, lower=0.5, upper=10.0),  # CL
        ThetaSpec(init=25.0, lower=5.0, upper=60.0),  # V
    ]
    omega_specs = [OmegaSpec(block_size=1, values=[0.04])]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]
    params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)

    pop_model = PopulationModel(
        dataset=dataset,
        pk_subroutine=ADVAN2(),
        params=params,
        trans=2,
        advan=2,
    )

    fo = FOMethod(maxeval=200, print_interval=999)
    result = fo.estimate(pop_model, params)
    return SimulationEngine(pop_model, result, seed=7)


# ---------------------------------------------------------------------------
# Tests: ADVAN6 (ODE)
# ---------------------------------------------------------------------------


def test_advan6_new_design_no_exception(advan6_engine):
    """simulate_new_design() with ADVAN6 must not raise."""
    dosing_df = pd.DataFrame(
        [
            {"ID": 1, "TIME": 0.0, "AMT": 100.0, "EVID": 1, "MDV": 1, "CMT": 1, "RATE": 0.0},
        ]
    )
    obs_times = np.array([1.0, 4.0, 8.0, 12.0])
    result = advan6_engine.simulate_new_design(
        dosing_df=dosing_df,
        obs_times=obs_times,
        n_subjects=5,
    )
    assert result is not None


def test_advan6_new_design_ipred_finite_positive(advan6_engine):
    """IPRED values from ADVAN6 new-design simulation must be finite and non-negative."""
    dosing_df = pd.DataFrame(
        [
            {"ID": 1, "TIME": 0.0, "AMT": 100.0, "EVID": 1, "MDV": 1, "CMT": 1, "RATE": 0.0},
        ]
    )
    obs_times = np.array([1.0, 4.0, 8.0, 12.0])
    result = advan6_engine.simulate_new_design(
        dosing_df=dosing_df,
        obs_times=obs_times,
        n_subjects=5,
    )
    ipred = result.simulated_df["IPRED"].values
    assert np.all(np.isfinite(ipred)), f"Non-finite IPRED values: {ipred}"
    assert np.all(ipred >= 0.0), f"Negative IPRED values: {ipred}"


def test_advan6_new_design_shape(advan6_engine):
    """SimulationResult DataFrame must include REP=0 plus all simulated replicates."""
    dosing_df = pd.DataFrame(
        [
            {"ID": 1, "TIME": 0.0, "AMT": 100.0, "EVID": 1, "MDV": 1, "CMT": 1, "RATE": 0.0},
        ]
    )
    obs_times = np.array([1.0, 4.0, 8.0, 12.0])
    n_subjects = 5
    n_replicates = 2
    result = advan6_engine.simulate_new_design(
        dosing_df=dosing_df,
        obs_times=obs_times,
        n_subjects=n_subjects,
        n_replicates=n_replicates,
    )
    df = result.simulated_df
    expected_rows = n_subjects * len(obs_times) * (n_replicates + 1)
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    assert set(df["REP"].unique()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Tests: ADVAN2 (analytical, control)
# ---------------------------------------------------------------------------


def test_advan2_new_design_no_exception(advan2_engine):
    """simulate_new_design() with ADVAN2 must not raise (control test)."""
    dosing_df = pd.DataFrame(
        [
            {"ID": 1, "TIME": 0.0, "AMT": 200.0, "EVID": 1, "MDV": 1, "CMT": 1, "RATE": 0.0},
        ]
    )
    obs_times = np.array([1.0, 4.0, 8.0, 12.0])
    result = advan2_engine.simulate_new_design(
        dosing_df=dosing_df,
        obs_times=obs_times,
        n_subjects=5,
    )
    assert result is not None


def test_advan2_new_design_ipred_finite(advan2_engine):
    """ADVAN2 new-design IPRED must be finite."""
    dosing_df = pd.DataFrame(
        [
            {"ID": 1, "TIME": 0.0, "AMT": 200.0, "EVID": 1, "MDV": 1, "CMT": 1, "RATE": 0.0},
        ]
    )
    obs_times = np.array([1.0, 4.0, 8.0, 12.0])
    result = advan2_engine.simulate_new_design(
        dosing_df=dosing_df,
        obs_times=obs_times,
        n_subjects=5,
    )
    ipred = result.simulated_df["IPRED"].values
    assert np.all(np.isfinite(ipred)), f"Non-finite IPRED: {ipred}"
