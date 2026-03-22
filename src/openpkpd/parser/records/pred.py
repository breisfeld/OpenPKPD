"""$PRED code block record (combined PK+ERROR for user-defined models)."""

from __future__ import annotations

from .base import BaseRecord


class PredRecord(BaseRecord):
    """
    $PRED <FORTRAN-style NM-TRAN code>

    Combined PK + ERROR code block. Used when ADVAN is not specified
    and the user defines the complete prediction model manually.
    """

    record_name = "PRED"

    def _parse(self, text: str) -> None:
        self.code: str = text
