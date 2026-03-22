"""$PK code block record."""

from __future__ import annotations

from .base import BaseRecord


class PKRecord(BaseRecord):
    """
    $PK <FORTRAN-style NM-TRAN code>

    The entire body is preserved verbatim as a code string
    and later compiled by NMTRANCompiler.
    """

    record_name = "PK"

    def _parse(self, text: str) -> None:
        self.code: str = text
