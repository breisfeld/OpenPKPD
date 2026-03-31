"""Tests for BuiltModel.fit metadata wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.api.model_builder import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.base import EstimationResult
from openpkpd.utils.errors import ModelError


@pytest.mark.unit
def test_built_fit_populates_result_metadata(monkeypatch):
    df = pd.DataFrame(
        {
            "ID": [1, 1, 2, 2],
            "TIME": [0.0, 1.0, 0.0, 1.0],
            "AMT": [1.0, 0.0, 1.0, 0.0],
            "DV": [0.0, 1.2, 0.0, 0.9],
            "EVID": [1, 0, 1, 0],
            "MDV": [1, 0, 1, 0],
        }
    )
    ds = NONMEMDataset.from_dataframe(df)
    built = (
        ModelBuilder()
        .problem("metadata demo")
        .dataset(ds)
        .subroutines(advan=1, trans=1)
        .pk("CL = THETA(1)")
        .error("Y = F + EPS(1)")
        .theta([1.0])
        .omega([0.1])
        .sigma([0.05])
        .estimation(method="FOCE", maxeval=1)
        .build()
    )

    class _FakeEstimator:
        def estimate(self, population_model, init_params):
            assert population_model.n_subjects() == 2
            return EstimationResult(
                theta_final=np.array([1.0]),
                omega_final=np.array([[0.1]]),
                sigma_final=np.array([[0.05]]),
                ofv=12.34,
                converged=True,
                method="FAKE",
            )

    monkeypatch.setattr(
        "openpkpd.api.model_builder.get_estimation_method", lambda *a, **k: _FakeEstimator()
    )

    result = built.fit()

    assert result.n_subjects == 2
    assert result.n_observations == 2
    assert result.n_parameters == 3
    assert np.isfinite(result.bic)


@pytest.mark.unit
def test_subroutines_forwards_supported_subroutine_kwargs():
    df = pd.DataFrame(
        {
            "ID": [1, 1],
            "TIME": [0.0, 1.0],
            "AMT": [1.0, 0.0],
            "DV": [0.0, 0.5],
            "EVID": [1, 0],
            "MDV": [1, 0],
        }
    )
    ds = NONMEMDataset.from_dataframe(df)

    built = (
        ModelBuilder()
        .problem("ode kwargs demo")
        .dataset(ds)
        .subroutines(advan=6, trans=1, jit="numpy", method="BDF")
        .pk("K = THETA(1)\nV = THETA(2)")
        .des("DADT(1) = -K*A(1)")
        .error("Y = F + EPS(1)")
        .theta([0.1, 10.0])
        .omega([0.1])
        .sigma([0.05])
        .build()
    )

    assert built.population_model.pk_subroutine.jit == "numpy"
    assert built.population_model.pk_subroutine.method == "BDF"


@pytest.mark.unit
def test_subroutines_rejects_unknown_subroutine_kwargs():
    df = pd.DataFrame(
        {
            "ID": [1, 1],
            "TIME": [0.0, 1.0],
            "AMT": [1.0, 0.0],
            "DV": [0.0, 0.5],
            "EVID": [1, 0],
            "MDV": [1, 0],
        }
    )
    ds = NONMEMDataset.from_dataframe(df)

    with pytest.raises(ModelError, match="does not support subroutine option"):
        (
            ModelBuilder()
            .problem("bad ode kwargs demo")
            .dataset(ds)
            .subroutines(advan=6, trans=1, does_not_exist=True)
            .pk("K = THETA(1)\nV = THETA(2)")
            .des("DADT(1) = -K*A(1)")
            .error("Y = F + EPS(1)")
            .theta([0.1, 10.0])
            .omega([0.1])
            .sigma([0.05])
            .build()
        )
