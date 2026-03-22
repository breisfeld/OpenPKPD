"""$ERROR code block record."""

from __future__ import annotations

from .base import BaseRecord


class ErrorRecord(BaseRecord):
    """
    $ERROR <FORTRAN-style NM-TRAN code>

    Code block defining Y (and optionally F, IPRED, W) for the observation model.
    Preserved verbatim and compiled by NMTRANCompiler.
    """

    record_name = "ERROR"

    def _parse(self, text: str) -> None:
        self.code: str = text
