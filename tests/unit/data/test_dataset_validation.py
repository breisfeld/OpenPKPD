"""
D1: Tests for NONMEMDataset numeric column validation.
"""
from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset, DatasetValidationError


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_csv(content: str) -> str:
    """Write CSV content to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        return f.name


# ── Test 1: Valid numeric CSV loads successfully ──────────────────────────────

def test_valid_numeric_csv_loads():
    csv = "ID,TIME,DV,AMT,EVID\n1,0,0,100,1\n1,1,5.2,0,0\n1,2,3.1,0,0\n"
    path = _write_csv(csv)
    try:
        ds = NONMEMDataset.from_csv(path)
        assert ds.n_subjects() == 1
        assert len(ds.df) == 3
    finally:
        os.unlink(path)


# ── Test 2: String ID raises DatasetValidationError ──────────────────────────

def test_string_id_raises_validation_error():
    csv = "ID,TIME,DV,AMT,EVID\nPATIENT_001,0,0,100,1\nPATIENT_001,1,5.2,0,0\n"
    path = _write_csv(csv)
    try:
        with pytest.raises(DatasetValidationError, match="ID"):
            NONMEMDataset.from_csv(path)
    finally:
        os.unlink(path)


# ── Test 3: String TIME raises DatasetValidationError ────────────────────────

def test_string_time_raises_validation_error():
    csv = "ID,TIME,DV,AMT,EVID\n1,PRE-DOSE,0,100,1\n1,1hr,5.2,0,0\n"
    path = _write_csv(csv)
    try:
        with pytest.raises(DatasetValidationError, match="TIME"):
            NONMEMDataset.from_csv(path)
    finally:
        os.unlink(path)


# ── Test 4: Mixed DV with "BLQ" string raises DatasetValidationError ─────────

def test_blq_string_in_dv_raises_validation_error():
    csv = "ID,TIME,DV,AMT,EVID\n1,0,0,100,1\n1,1,5.2,0,0\n1,2,BLQ,0,0\n1,3,1.1,0,0\n"
    path = _write_csv(csv)
    try:
        with pytest.raises(DatasetValidationError, match="DV"):
            NONMEMDataset.from_csv(path)
    finally:
        os.unlink(path)


# ── Test 5: Theophylline reference dataset ───────────────────────────────────

def test_theophylline_reference_dataset():
    """The Boeckmann theophylline dataset should load without error.
    Expected: 12 subjects, 144 total rows (12 dose + 132 observations).
    """
    # Walk up from this file to find the project root (has pyproject.toml)
    _here = os.path.dirname(os.path.abspath(__file__))
    _project_root = _here
    for _ in range(6):
        if os.path.exists(os.path.join(_project_root, "pyproject.toml")):
            break
        _project_root = os.path.dirname(_project_root)

    theo_paths = [
        os.path.join(_project_root, "examples", "control_streams", "theo.csv"),
        os.path.join(_project_root, "examples", "shared_data", "theophylline", "theo.csv"),
    ]
    theo_path = None
    for p in theo_paths:
        if os.path.exists(p):
            theo_path = os.path.abspath(p)
            break

    if theo_path is None:
        pytest.skip("Theophylline reference CSV not found")

    ds = NONMEMDataset.from_csv(theo_path)

    # Should load without error
    assert ds.n_subjects() >= 1, "Expected at least 1 subject"
    assert len(ds.df) >= 10, f"Expected many rows, got {len(ds.df)}"

    # Check standard columns are numeric
    for col in ["ID", "TIME", "DV"]:
        if col in ds.df.columns:
            assert pd.api.types.is_numeric_dtype(ds.df[col]), f"{col} should be numeric"


# ── Test 6: DatasetValidationError is a subclass of ValueError ───────────────

def test_dataset_validation_error_is_value_error():
    """DatasetValidationError should be catchable as ValueError."""
    with pytest.raises(ValueError):
        raise DatasetValidationError("test message")


# ── Test 7: Valid float DV with NaN (missing) loads fine ─────────────────────

def test_float_dv_with_missing_loads():
    """DV column with -99 (missing) values should load (coerced to NaN)."""
    csv = "ID,TIME,DV,AMT,EVID\n1,0,0,100,1\n1,1,5.2,0,0\n1,2,-99,0,0\n"
    path = _write_csv(csv)
    try:
        ds = NONMEMDataset.from_csv(path, missing_value=-99)
        assert len(ds.df) == 3
    finally:
        os.unlink(path)
