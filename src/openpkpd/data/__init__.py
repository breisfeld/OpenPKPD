"""
openpkpd.data — Dataset loading, column utilities, and BLQ handling.

Usage:
    from openpkpd.data import load_theophylline, load_warfarin
    from openpkpd.data.dataset import NONMEMDataset
"""

from __future__ import annotations

import importlib.resources as resources

from openpkpd.data.dataset import NONMEMDataset

# Bundled example datasets live inside the installed package
# (src/openpkpd/data/datasets/) so they are packaged into the wheel and
# resolve on a clean PyPI install, not just a source checkout.
_DATASETS_DIR = "datasets"


def _bundled(name: str) -> "NONMEMDataset":
    resource = resources.files(__package__).joinpath(_DATASETS_DIR, name)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Bundled dataset '{name}' not found at {resource}. "
            "The OpenPKPD installation may be incomplete; reinstall with "
            "'pip install --force-reinstall openpkpd'."
        )
    with resources.as_file(resource) as path:
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
