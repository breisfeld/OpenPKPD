"""$ABBREVIATED record parser."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseRecord


class AbbreviatedRecord(BaseRecord):
    """
    $ABBREVIATED DERIV2=NO COMRES=n

    Controls NONMEM computational options (second derivative evaluation, etc.).
    """

    record_name = "ABBREVIATED"

    def _parse(self, text: str) -> None:
        flat = re.sub(r";.*", "", " ".join(text.splitlines()))
        self.deriv2: str | None = None
        self.comres: int | None = None

        m = re.search(r"\bDERIV2\s*=\s*(\S+)", flat, re.IGNORECASE)
        if m:
            self.deriv2 = m.group(1).upper()

        m = re.search(r"\bCOMRES\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.comres = int(m.group(1))

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"deriv2": self.deriv2, "comres": self.comres})
        return d
