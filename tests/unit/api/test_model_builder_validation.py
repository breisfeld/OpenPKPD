"""
Tests for ADVAN/TRANS validation and covariate column validation in ModelBuilder
(MB2 and MB3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.api.model_builder import ConfigurationError, ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.base import EstimationResult


def _minimal_ds(extra_cols: dict | None = None) -> NONMEMDataset:
    data = {
        "ID": [1, 1],
        "TIME": [0.0, 1.0],
        "AMT": [100.0, 0.0],
        "DV": [0.0, 5.0],
        "EVID": [1, 0],
        "MDV": [1, 0],
    }
    if extra_cols:
        data.update(extra_cols)
    return NONMEMDataset.from_dataframe(pd.DataFrame(data))


def _base_builder(ds=None, advan=2, trans=2):
    b = ModelBuilder()
    if ds is None:
        ds = _minimal_ds()
    return (
        b.problem("test")
        .dataset(ds)
        .subroutines(advan=advan, trans=trans)
        .pk("CL = THETA(1)\nV = THETA(2)")
        .error("Y = F + EPS(1)")
        .theta([0.1, 10.0])
        .omega([0.1, 0.1])
        .sigma([0.05])
    )


# ---- ADVAN/TRANS validation (MB2) ----------------------------------------

@pytest.mark.parametrize("advan,trans", [
    (1, 2),
    (3, 4),
    (5, 1),
    (7, 1),
    (11, 6),
])
def test_valid_advan_trans_combinations(advan, trans):
    """Spot-check valid ADVAN/TRANS combos succeed."""
    b = ModelBuilder()
    b.subroutines(advan=advan, trans=trans)  # should not raise


def test_invalid_trans_for_advan3():
    """ADVAN=3, TRANS=2 -> ConfigurationError."""
    with pytest.raises(ConfigurationError, match="TRANS=2"):
        ModelBuilder().subroutines(advan=3, trans=2)


def test_invalid_advan_99():
    """ADVAN=99 -> ConfigurationError."""
    with pytest.raises(ConfigurationError, match="ADVAN=99"):
        ModelBuilder().subroutines(advan=99, trans=1)


def test_ode_advan_any_trans():
    """ADVAN=6, TRANS=99 succeeds (no TRANS restriction for ODE models)."""
    b = ModelBuilder()
    b.subroutines(advan=6, trans=99)  # should not raise


def test_advan2_trans2_builds(monkeypatch):
    """ADVAN=2/TRANS=2 must build end-to-end (regression guard)."""
    ds = _minimal_ds()
    built = _base_builder(ds=ds, advan=2, trans=2).build()
    assert built is not None
    assert built.population_model is not None


# ---- Covariate column validation (MB3) -----------------------------------

def test_valid_covariate_columns_no_error():
    """Valid covariate column in dataset -> no error."""
    ds = _minimal_ds(extra_cols={"WT": [70.0, 70.0]})
    b = _base_builder(ds=ds).covariates(["WT"])
    built = b.build()  # should not raise
    assert built is not None


def test_missing_covariate_column_raises():
    """One missing covariate column -> ConfigurationError naming the column."""
    ds = _minimal_ds()
    b = _base_builder(ds=ds).covariates(["WT"])
    with pytest.raises(ConfigurationError, match="WT"):
        b.build()


def test_multiple_missing_covariate_columns_raises():
    """Multiple missing columns -> error lists all of them."""
    ds = _minimal_ds()
    b = _base_builder(ds=ds).covariates(["WT", "AGE"])
    with pytest.raises(ConfigurationError) as exc_info:
        b.build()
    msg = str(exc_info.value)
    assert "WT" in msg
    assert "AGE" in msg


def test_empty_covariate_list_no_error():
    """Empty covariate list -> no error even if dataset has no extra columns."""
    ds = _minimal_ds()
    b = _base_builder(ds=ds).covariates([])
    built = b.build()  # should not raise
    assert built is not None
