"""$COVARIANCE record parser."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseRecord


class CovarianceRecord(BaseRecord):
    """
    $COVARIANCE [MATRIX=S|R|SR] [PRINT=E] [UNCONDITIONAL] [ONLY] [SIGL=n]
    """

    record_name = "COVARIANCE"

    def _parse(self, text: str) -> None:
        flat = re.sub(r";.*", "", " ".join(text.splitlines()))

        self.matrix: str = "SR"  # default: sandwich estimator
        self.unconditional: bool = False
        self.only: bool = False
        self.sigl: int | None = None
        self.print_e: bool = False

        m = re.search(r"\bMATRIX\s*=\s*([A-Z]+)", flat, re.IGNORECASE)
        if m:
            self.matrix = m.group(1).upper()

        self.unconditional = bool(re.search(r"\bUNCONDITIONAL\b", flat, re.IGNORECASE))
        self.only = bool(re.search(r"\bONLY\b", flat, re.IGNORECASE))
        self.print_e = bool(re.search(r"\bPRINT\s*=\s*E\b", flat, re.IGNORECASE))

        m = re.search(r"\bSIGL\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.sigl = int(m.group(1))

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"matrix": self.matrix, "unconditional": self.unconditional})
        return d
