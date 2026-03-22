"""Tests for BuiltModel.fit metadata wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.api.model_builder import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.base import EstimationResult


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
