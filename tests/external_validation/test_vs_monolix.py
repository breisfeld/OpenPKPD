"""External-validation benchmarks against public Monolix theophylline output."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.saem import SAEMMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan2 import ADVAN2

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "theophylline_boeckmann.csv")
REFERENCE_PATH = os.path.join(HERE, "reference", "monolix_theophylline_saem.json")


def _load_reference() -> dict:
    with open(REFERENCE_PATH) as f:
        return json.load(f)


def _load_monolix_scaled_dataset() -> NONMEMDataset:
    df = pd.read_csv(DATA_PATH)
    dose_rows = df["EVID"] == 1
    df.loc[dose_rows, "AMT"] = df.loc[dose_rows, "AMT"] / df.loc[dose_rows, "WT"]
    return NONMEMDataset.from_dataframe(df)


def _build_monolix_like_model(dataset: NONMEMDataset) -> tuple[PopulationModel, ParameterSet]:
    params = ParameterSet.from_specs(
        theta_specs=[
            ThetaSpec(init=1.5, lower=0.2, upper=8.0),
            ThetaSpec(init=0.04, lower=0.005, upper=0.2),
            ThetaSpec(init=0.5, lower=0.05, upper=5.0),
        ],
        omega_specs=[
            OmegaSpec(block_size=1, values=[0.09]),
            OmegaSpec(block_size=1, values=[0.06]),
            OmegaSpec(block_size=1, values=[0.04]),
        ],
        sigma_specs=[SigmaSpec(block_size=1, values=[0.02])],
    )
    pop = PopulationModel(
        dataset=dataset,
        pk_subroutine=ADVAN2(),
        params=params,
        trans=2,
        advan=2,
    )
    return pop, params


@pytest.fixture(scope="module")
def monolix_reference() -> dict:
    return _load_reference()


@pytest.fixture(scope="module")
def monolix_fit_result():
    dataset = _load_monolix_scaled_dataset()
    pop, params = _build_monolix_like_model(dataset)
    return SAEMMethod(n_iter_phase1=200, n_iter_phase2=100, n_chains=2, seed=42).estimate(
        pop, params
    )


@pytest.mark.external_validation
@pytest.mark.slow
class TestTheophyllineVsMonolix:
    def test_theta_matches_public_monolix_reference(self, monolix_fit_result, monolix_reference):
        expected = monolix_reference["theta_natural_scale"]
        ka, cl, v = monolix_fit_result.theta_final
        np.testing.assert_allclose(ka, expected["ka_pop"], rtol=0.12)
        np.testing.assert_allclose(cl, expected["Cl_pop"], rtol=0.12)
        np.testing.assert_allclose(v, expected["V_pop"], rtol=0.15)

    def test_fit_result_is_numerically_well_behaved(self, monolix_fit_result):
        assert np.isfinite(monolix_fit_result.ofv)
        assert monolix_fit_result.ofv > 0.0
        assert monolix_fit_result.ofv < 1000.0
