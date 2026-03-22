"""PK subroutine routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openpkpd.pk.absorption.parallel import ParallelAbsorption
from openpkpd.pk.absorption.transit import TransitAbsorption
from openpkpd.pk.analytical import ADVAN1, ADVAN2, ADVAN3, ADVAN4, ADVAN11, ADVAN12
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.pk.ode import ADVAN6, ADVAN8, ADVAN10, ADVAN13
from openpkpd.pk.ode.dde import DDESubroutine
from openpkpd.utils.errors import PKError

if TYPE_CHECKING:
    from openpkpd.parser.records.subroutines import SubroutinesRecord


_ADVAN_MAP: dict[int, type[PKSubroutine]] = {
    1: ADVAN1,
    2: ADVAN2,
    3: ADVAN3,
    4: ADVAN4,
    6: ADVAN6,
    8: ADVAN8,
    10: ADVAN10,
    11: ADVAN11,
    12: ADVAN12,
    13: ADVAN13,
    16: DDESubroutine,  # DDE extension (analogous to NONMEM ADVAN16/17)
}

# D1: TRANS codes that select non-standard absorption models
# TRANS=7 → TransitAbsorption (Savic transit compartment)
# TRANS=8 → ParallelAbsorption (zero-order + first-order)
_TRANS_ABSORPTION_MAP: dict[int, type[PKSubroutine]] = {
    7: TransitAbsorption,
    8: ParallelAbsorption,
}


def get_advan(advan: int) -> PKSubroutine:
    """Return an instantiated PKSubroutine for the given ADVAN number."""
    cls = _ADVAN_MAP.get(advan)
    if cls is None:
        raise PKError(
            f"ADVAN{advan} is not yet implemented. Supported: {sorted(_ADVAN_MAP.keys())}"
        )
    return cls()


def get_absorption_model(subr_rec: SubroutinesRecord | None) -> PKSubroutine | None:
    """
    D1: Check SubroutinesRecord for non-standard absorption models.

    Returns a specialized PKSubroutine for transit or parallel absorption,
    or None if standard ADVAN routing should be used.

    Convention:
        TRANS=7 → TransitAbsorption (Savic 2007 transit compartment model)
        TRANS=8 → ParallelAbsorption (zero-order + first-order parallel)
    """
    if subr_rec is None:
        return None
    trans = getattr(subr_rec, "trans", None)
    if trans is None:
        return None
    cls = _TRANS_ABSORPTION_MAP.get(int(trans))
    if cls is not None:
        return cls()
    return None


__all__ = [
    "PKSubroutine",
    "PKSolution",
    "get_advan",
    "get_absorption_model",
    "ADVAN1",
    "ADVAN2",
    "ADVAN3",
    "ADVAN4",
    "ADVAN6",
    "ADVAN8",
    "ADVAN10",
    "ADVAN11",
    "ADVAN12",
    "ADVAN13",
    "DDESubroutine",
    "TransitAbsorption",
    "ParallelAbsorption",
]
