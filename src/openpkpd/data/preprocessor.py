"""Dataset preprocessor utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from openpkpd.data.columns import EVID, ID, MDV, TIME
from openpkpd.utils.constants import EVID_OBS


def auto_generate_mdv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-generate MDV column if absent.

    Rows with EVID != 0 get MDV=1 (missing/not used in OFV).
    Rows with EVID == 0 get MDV=0 (observed).
    """
    df = df.copy()
    if MDV not in df.columns:
        df[MDV] = (df[EVID] != EVID_OBS).astype(int)
    return df


def sort_by_id_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataset by ID then TIME (stable sort preserves tie order)."""
    return df.sort_values([ID, TIME], kind="stable").reset_index(drop=True)


def validate_monotone_time(df: pd.DataFrame) -> list[str]:
    """
    Check that TIME is monotonically non-decreasing within each subject.

    Returns a list of warning strings for subjects with time reversals.
    """
    warnings: list[str] = []
    for subj_id, grp in df.groupby(ID, sort=True):
        times = grp[TIME].values
        if np.any(np.diff(times) < 0):
            warnings.append(f"Subject {subj_id}: non-monotone TIME values")
    return warnings
