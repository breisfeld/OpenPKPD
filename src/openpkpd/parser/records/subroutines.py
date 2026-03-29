"""$SUBROUTINES record parser."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseRecord


class SubroutinesRecord(BaseRecord):
    """
    $SUBROUTINES ADVAN1 TRANS2 [TOL=6] [INFN=routine] [PRED=routine]

    Routes to the correct PK subroutine and parameter transformation.
    """

    record_name = "SUBROUTINES"

    def _parse(self, text: str) -> None:
        lines = [
            ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(";")
        ]
        flat = re.sub(r";.*", "", " ".join(lines))

        self.advan: int | None = None
        self.trans: int | None = None
        self.tol: int = 6
        self.infn: str | None = None
        self.pred: str | None = None

        # ADVAN<n>
        m = re.search(r"\bADVAN(\d+)\b", flat, re.IGNORECASE)
        if m:
            self.advan = int(m.group(1))

        # TRANS<n>
        m = re.search(r"\bTRANS(\d+)\b", flat, re.IGNORECASE)
        if m:
            self.trans = int(m.group(1))

        # TOL=n
        m = re.search(r"\bTOL\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.tol = int(m.group(1))

        # INFN=routine
        m = re.search(r"\bINFN\s*=\s*(\S+)", flat, re.IGNORECASE)
        if m:
            self.infn = m.group(1)

        # PRED=routine (for user-supplied PRED)
        m = re.search(r"\bPRED\s*=\s*(\S+)", flat, re.IGNORECASE)
        if m:
            self.pred = m.group(1)

    def to_string(self) -> str:
        """Serialize SUBROUTINES record from parsed fields."""
        parts: list[str] = []
        if self.advan is not None:
            parts.append(f"ADVAN{self.advan}")
        if self.trans is not None:
            parts.append(f"TRANS{self.trans}")
        if self.tol != 6:
            parts.append(f"TOL={self.tol}")
        if self.infn:
            parts.append(f"INFN={self.infn}")
        if self.pred:
            parts.append(f"PRED={self.pred}")
        return f"$SUBROUTINES {' '.join(parts)}\n"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"advan": self.advan, "trans": self.trans, "tol": self.tol})
        return d
