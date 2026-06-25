"""
openpkpd.data — Dataset loading, column utilities, and BLQ handling.

Usage:
    from openpkpd.data import load_theophylline, load_warfarin
    from openpkpd.data.dataset import NONMEMDataset
"""

from __future__ import annotations

import os
import pathlib

from openpkpd.data.dataset import NONMEMDataset

_DATA_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "tests" / "external_validation" / "data"


def _bundled(name: str) -> "NONMEMDataset":
    path = _DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled dataset '{name}' not found at {path}. "
            "Ensure the repository is checked out with its test data."
        )
    return NONMEMDataset.from_csv(str(path))


def load_theophylline() -> "NONMEMDataset":
    """
    Load the 12-subject Boeckmann theophylline dataset.

    Returns:
        NONMEMDataset with columns ID, TIME, AMT, DV, EVID, MDV, WT.
        144 rows (12 dose events + 132 observations, 12 subjects × 11 timepoints).
    """
    return _bundled("theophylline_boeckmann.csv")


def load_warfarin() -> "NONMEMDataset":
    """Load the warfarin PK dataset."""
    return _bundled("warfarin_pk.csv")
