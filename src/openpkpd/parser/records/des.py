"""$DES code block record (ODE system)."""

from __future__ import annotations

from .base import BaseRecord


class DESRecord(BaseRecord):
    """
    $DES <FORTRAN-style NM-TRAN code>

    Code block defining dA(n)/dt = DADT(n) = ... for ODE-based models (ADVAN6/8/13).
    Preserved verbatim and compiled by NMTRANCompiler.
    """

    record_name = "DES"

    def _parse(self, text: str) -> None:
        self.code: str = text
