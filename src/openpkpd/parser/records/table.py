"""$TABLE record parser."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseRecord


class TableRecord(BaseRecord):
    """
    $TABLE ID TIME DV PRED IPRED CWRES NOPRINT ONEHEADER FILE=sdtab001

    Specifies columns to output and output file options.
    """

    record_name = "TABLE"

    def _parse(self, text: str) -> None:
        flat = re.sub(r";.*", "", " ".join(ln.strip() for ln in text.splitlines()))

        self.columns: list[str] = []
        self.file: str | None = None
        self.noprint: bool = False
        self.print: bool = True
        self.noappend: bool = False
        self.append: bool = False
        self.oneheader: bool = False
        self.firstonly: bool = False
        self.lastonly: bool = False
        self.notitle: bool = False
        self.nolabel: bool = False
        self.format: str | None = None
        self.esample: int | None = None

        # Extract FILE= option
        m = re.search(r"\bFILE\s*=\s*(\S+)", flat, re.IGNORECASE)
        if m:
            self.file = m.group(1)
            flat = flat[: m.start()] + flat[m.end() :]

        m = re.search(r"\bFORMAT\s*=\s*(\S+)", flat, re.IGNORECASE)
        if m:
            self.format = m.group(1)
            flat = flat[: m.start()] + flat[m.end() :]

        m = re.search(r"\bESAMPLE\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.esample = int(m.group(1))
            flat = flat[: m.start()] + flat[m.end() :]

        # Boolean flags
        self.noprint = bool(re.search(r"\bNOPRINT\b", flat, re.IGNORECASE))
        self.noappend = bool(re.search(r"\bNOAPPEND\b", flat, re.IGNORECASE))
        self.append = bool(re.search(r"\bAPPEND\b", flat, re.IGNORECASE))
        self.oneheader = bool(re.search(r"\bONEHEADER\b", flat, re.IGNORECASE))
        self.firstonly = bool(re.search(r"\bFIRSTONLY\b", flat, re.IGNORECASE))
        self.lastonly = bool(re.search(r"\bLASTONLY\b", flat, re.IGNORECASE))
        self.notitle = bool(re.search(r"\bNOTITLE\b", flat, re.IGNORECASE))
        self.nolabel = bool(re.search(r"\bNOLABEL\b", flat, re.IGNORECASE))

        # Extract column names (remaining words)
        reserved = {
            "NOPRINT",
            "PRINT",
            "NOAPPEND",
            "APPEND",
            "ONEHEADER",
            "FIRSTONLY",
            "LASTONLY",
            "NOTITLE",
            "NOLABEL",
            "BY",
        }
        for tok in re.split(r"[\s,]+", flat.strip()):
            if tok and tok.upper() not in reserved:
                self.columns.append(tok.upper())

    def to_string(self) -> str:
        """Serialize TABLE record from parsed fields."""
        parts: list[str] = list(self.columns)
        if self.noprint:
            parts.append("NOPRINT")
        if self.noappend:
            parts.append("NOAPPEND")
        if self.append:
            parts.append("APPEND")
        if self.oneheader:
            parts.append("ONEHEADER")
        if self.firstonly:
            parts.append("FIRSTONLY")
        if self.lastonly:
            parts.append("LASTONLY")
        if self.notitle:
            parts.append("NOTITLE")
        if self.nolabel:
            parts.append("NOLABEL")
        if self.format:
            parts.append(f"FORMAT={self.format}")
        if self.esample is not None:
            parts.append(f"ESAMPLE={self.esample}")
        if self.file:
            parts.append(f"FILE={self.file}")
        return f"$TABLE {' '.join(parts)}\n"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"columns": self.columns, "file": self.file, "noprint": self.noprint})
        return d
