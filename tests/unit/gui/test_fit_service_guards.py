"""Guard-behaviour tests for FitService data-integrity checks."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.services.fit_service import FitService


# ---------------------------------------------------------------------------
# FS2 — scalar LOQ injection warns when LLOQ column is non-uniform
# ---------------------------------------------------------------------------


def _make_dataset_with_lloq(lloq_values: list[float]) -> MagicMock:
    """Return a mock NONMEMDataset whose LLOQ column has the given values."""
    df = pd.DataFrame({"LLOQ": lloq_values})
    mock_ds = MagicMock()
    mock_ds.has_lloq = True
    mock_ds.df = df
    return mock_ds


def test_scalar_loq_warns_when_lloq_column_is_non_uniform() -> None:
    """When the dataset already has a non-uniform LLOQ column, a warning must be emitted
    and the column must NOT be overwritten with the scalar LOQ value."""
    dataset_asset = DatasetAsset(source_path="/fake/data.csv", loq=0.5)
    builder_mock = MagicMock()
    dataset_with_lloq = _make_dataset_with_lloq([0.1, 0.2, 0.3])  # non-uniform

    with patch(
        "openpkpd_gui.services.fit_service.NONMEMDataset.from_csv",
        return_value=dataset_with_lloq,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FitService._apply_dataset_asset_to_builder(builder_mock, dataset_asset)

    warning_messages = [str(w.message) for w in caught]
    assert any("non-uniform LLOQ" in msg for msg in warning_messages), (
        f"Expected a non-uniform LLOQ warning, got: {warning_messages}"
    )
    # The LLOQ column values should remain unchanged (not overwritten with 0.5).
    assert list(dataset_with_lloq.df["LLOQ"]) == [0.1, 0.2, 0.3]


def test_scalar_loq_not_applied_when_lloq_column_uniform() -> None:
    """When the dataset has a uniform LLOQ column, no warning is emitted and the column
    is not overwritten."""
    dataset_asset = DatasetAsset(source_path="/fake/data.csv", loq=0.5)
    builder_mock = MagicMock()
    dataset_with_uniform_lloq = _make_dataset_with_lloq([0.1, 0.1, 0.1])

    with patch(
        "openpkpd_gui.services.fit_service.NONMEMDataset.from_csv",
        return_value=dataset_with_uniform_lloq,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FitService._apply_dataset_asset_to_builder(builder_mock, dataset_with_uniform_lloq)

    warning_messages = [str(w.message) for w in caught]
    assert not any("non-uniform LLOQ" in msg for msg in warning_messages)
    # Column must remain as-is.
    assert list(dataset_with_uniform_lloq.df["LLOQ"]) == [0.1, 0.1, 0.1]


def test_scalar_loq_applied_when_no_lloq_column() -> None:
    """When the dataset has no LLOQ column, the scalar LOQ is injected normally."""
    dataset_asset = DatasetAsset(source_path="/fake/data.csv", loq=0.5)
    builder_mock = MagicMock()

    df = pd.DataFrame({"DV": [1.0, 2.0]})
    mock_ds = MagicMock()
    mock_ds.has_lloq = False
    mock_ds.df = df

    with patch(
        "openpkpd_gui.services.fit_service.NONMEMDataset.from_csv",
        return_value=mock_ds,
    ):
        FitService._apply_dataset_asset_to_builder(builder_mock, dataset_asset)

    assert list(mock_ds.df["LLOQ"]) == [0.5, 0.5]
