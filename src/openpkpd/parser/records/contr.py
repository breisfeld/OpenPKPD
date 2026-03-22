"""$CONTR record parser (user-supplied contrast subroutine)."""

from __future__ import annotations

from .base import BaseRecord


class ContrRecord(BaseRecord):
    """$CONTR — specifies user-supplied CONTR subroutine options."""

    record_name = "CONTR"

    def _parse(self, text: str) -> None:
        self.text: str = text.strip()
