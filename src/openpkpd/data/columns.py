"""Column name constants and helpers for NONMEM datasets."""

from __future__ import annotations

from openpkpd.utils.constants import Columns as _C

# Re-export standard column names
ID = _C.ID
TIME = _C.TIME
DV = _C.DV
AMT = _C.AMT
RATE = _C.RATE
EVID = _C.EVID
MDV = _C.MDV
CMT = _C.CMT
ADDL = _C.ADDL
II = _C.II
SS = _C.SS

# All required columns for a minimal NONMEM dataset
REQUIRED_COLUMNS = {ID, TIME, DV}

# All standard columns that have defined meaning in NONMEM
STANDARD_COLUMNS = {
    ID,
    TIME,
    DV,
    AMT,
    RATE,
    EVID,
    MDV,
    CMT,
    ADDL,
    II,
    SS,
    _C.BLQ,
    _C.LLOQ,
    _C.DUR,
}

# Columns that trigger special event processing logic
EVENT_COLUMNS = {EVID, AMT, RATE, CMT, ADDL, II, SS}


def infer_column_aliases(
    input_record_columns: list[str], dataset_columns: list[str]
) -> dict[str, str]:
    """
    Map dataset column names to NONMEM standard names using $INPUT declarations.

    Returns a dict: {dataset_col → nonmem_standard_name}.
    Dropped columns are excluded.
    """
    mapping: dict[str, str] = {}
    for pos, nm_col in enumerate(input_record_columns):
        if nm_col.startswith("_DROP_"):
            continue
        if pos < len(dataset_columns):
            ds_col = dataset_columns[pos]
            mapping[ds_col] = nm_col
        else:
            # Named column not in dataset (may be okay if dataset has fewer columns)
            pass
    return mapping
