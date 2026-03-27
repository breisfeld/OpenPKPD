"""
Shared pytest fixtures for openpkpd test suite.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.data.event_processor import DoseEvent
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec

ROOT = Path(__file__).resolve().parent.parent
R_LIB_DIR = ROOT / ".r-lib"

# Make the project-local R library visible to tests that spawn Rscript.
if R_LIB_DIR.exists():
    existing_r_libs = os.environ.get("R_LIBS_USER", "").strip()
    if existing_r_libs:
        os.environ["R_LIBS_USER"] = f"{R_LIB_DIR}{os.pathsep}{existing_r_libs}"
    else:
        os.environ["R_LIBS_USER"] = str(R_LIB_DIR)

# ── Theophylline dataset (12 subjects, 1-cmt oral, classic NONMEM example) ───

THEO_CSV = """\
ID,TIME,AMT,DV,EVID,MDV
1,0,4.02,0,1,1
1,0.27,0,0.74,0,0
1,0.57,0,1.72,0,0
1,1.02,0,7.91,0,0
1,1.92,0,8.31,0,0
1,3.5,0,8.33,0,0
1,5.02,0,6.85,0,0
1,7.03,0,6.08,0,0
1,9.0,0,5.4,0,0
1,12.05,0,4.55,0,0
1,24.37,0,1.25,0,0
2,0,4.4,0,1,1
2,0.35,0,0.96,0,0
2,0.6,0,2.33,0,0
2,1.07,0,4.71,0,0
2,2.13,0,8.33,0,0
2,3.5,0,9.02,0,0
2,5.02,0,7.14,0,0
2,7.02,0,5.68,0,0
2,9.1,0,4.55,0,0
2,12.1,0,3.01,0,0
2,25.0,0,0.9,0,0
3,0,4.53,0,1,1
3,0.27,0,1.29,0,0
3,0.58,0,3.08,0,0
3,1.02,0,6.44,0,0
3,2.02,0,6.28,0,0
3,3.62,0,7.09,0,0
3,5.08,0,7.56,0,0
3,7.07,0,6.59,0,0
3,9.0,0,5.87,0,0
3,12.15,0,4.57,0,0
3,24.17,0,1.17,0,0
"""


@pytest.fixture
def theophylline_df() -> pd.DataFrame:
    """Small Theophylline-like dataset as a DataFrame."""
    return pd.read_csv(io.StringIO(THEO_CSV))


@pytest.fixture
def theophylline_dataset(theophylline_df: pd.DataFrame) -> NONMEMDataset:
    """NONMEMDataset from the Theophylline-like data."""
    return NONMEMDataset.from_dataframe(theophylline_df)


@pytest.fixture
def simple_theta_specs() -> list[ThetaSpec]:
    """Simple 3-parameter THETA specs for 1-cmt oral (KA, CL, V)."""
    return [
        ThetaSpec(lower=0.01, init=1.5, upper=20.0, label="KA"),
        ThetaSpec(lower=0.001, init=0.08, upper=5.0, label="CL"),
        ThetaSpec(lower=0.1, init=30.0, upper=500.0, label="V"),
    ]


@pytest.fixture
def simple_omega_specs() -> list[OmegaSpec]:
    """Diagonal 3x3 OMEGA for 1-cmt oral."""
    return [
        OmegaSpec(block_size=1, values=[0.5]),
        OmegaSpec(block_size=1, values=[0.3]),
        OmegaSpec(block_size=1, values=[0.3]),
    ]


@pytest.fixture
def simple_sigma_specs() -> list[SigmaSpec]:
    """Single proportional residual error."""
    return [SigmaSpec(block_size=1, values=[0.1])]


@pytest.fixture
def simple_params(
    simple_theta_specs: list[ThetaSpec],
    simple_omega_specs: list[OmegaSpec],
    simple_sigma_specs: list[SigmaSpec],
) -> ParameterSet:
    """ParameterSet for simple 1-cmt oral model."""
    return ParameterSet.from_specs(simple_theta_specs, simple_omega_specs, simple_sigma_specs)


@pytest.fixture
def simple_dose_events() -> list[DoseEvent]:
    """Single bolus dose at t=0, 4 mg into compartment 1."""
    return [DoseEvent(time=0.0, amount=4.0, compartment=1)]


@pytest.fixture
def simple_obs_times() -> np.ndarray:
    """Standard observation times for a 24-hour PK study."""
    return np.array([0.27, 0.57, 1.02, 1.92, 3.5, 5.02, 7.03, 9.0, 12.05, 24.37])


@pytest.fixture
def advan2_pk_params() -> dict[str, float]:
    """Typical PK parameters for ADVAN2 (1-cmt oral): KA, K, V."""
    return {"KA": 1.5, "K": 0.08 / 30.0, "V": 30.0}
