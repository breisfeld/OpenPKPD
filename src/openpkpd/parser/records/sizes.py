"""$SIZES record parser."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseRecord


class SizesRecord(BaseRecord):
    """
    $SIZES LTH=20 LVR=20 LNP4=10 ...

    Sets compile-time size parameters (mostly legacy NONMEM FORTRAN limits).
    """

    record_name = "SIZES"

    def _parse(self, text: str) -> None:
        flat = re.sub(r";.*", "", " ".join(text.splitlines()))
        self.sizes: dict[str, int] = {}
        for m in re.finditer(r"\b([A-Z0-9_]+)\s*=\s*(\d+)", flat, re.IGNORECASE):
            self.sizes[m.group(1).upper()] = int(m.group(2))

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["sizes"] = self.sizes
        return d
