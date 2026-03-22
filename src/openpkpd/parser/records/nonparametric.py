"""$NONPARAMETRIC record parser."""

from __future__ import annotations

import re

from .base import BaseRecord


class NonparametricRecord(BaseRecord):
    """
    $NONPARAMETRIC [NPSUPP=n] [MCETA=n]
    """

    record_name = "NONPARAMETRIC"

    def _parse(self, text: str) -> None:
        flat = re.sub(r";.*", "", " ".join(text.splitlines()))
        self.npsupp: int | None = None
        self.mceta: int | None = None

        m = re.search(r"\bNPSUPP\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.npsupp = int(m.group(1))

        m = re.search(r"\bMCETA\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.mceta = int(m.group(1))
