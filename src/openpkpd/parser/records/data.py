"""$DATA record parser."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseRecord


class DataRecord(BaseRecord):
    """
    $DATA <filename> [IGNORE=@] [IGNORE=(EVID.EQ.3)] [ACCEPT=(...)]
          [NOWIDE] [WIDE] [CHECKOUT] [RECORDS=n] [LRECL=n]
    """

    record_name = "DATA"

    def _parse(self, text: str) -> None:
        lines = [
            ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(";")
        ]
        full = " ".join(lines)

        # Extract filename (first token, may be quoted)
        m = re.match(r"""^["']?([^"'\s]+)["']?\s*(.*)$""", full.strip(), re.DOTALL)
        self.filename: str = m.group(1) if m else ""
        rest = m.group(2) if m else full

        self.ignore_char: str | None = None
        self.ignore_list: list[str] = []
        self.accept_list: list[str] = []
        self.records: int | None = None
        self.lrecl: int | None = None
        self.nowide: bool = False
        self.wide: bool = False

        # IGNORE=@ or IGNORE=(expr)
        m_ignore_char = re.search(r"\bIGNORE\s*=\s*([^()\s,]+)", rest, re.IGNORECASE)
        if m_ignore_char and not m_ignore_char.group(1).startswith("("):
            self.ignore_char = m_ignore_char.group(1).strip("'\"")

        m_ignore_list = re.search(r"\bIGNORE\s*=\s*\(([^)]+)\)", rest, re.IGNORECASE)
        if m_ignore_list:
            self.ignore_list = [s.strip() for s in m_ignore_list.group(1).split(",")]

        m_accept = re.search(r"\bACCEPT\s*=\s*\(([^)]+)\)", rest, re.IGNORECASE)
        if m_accept:
            self.accept_list = [s.strip() for s in m_accept.group(1).split(",")]

        m_records = re.search(r"\bRECORDS?\s*=\s*(\d+)", rest, re.IGNORECASE)
        if m_records:
            self.records = int(m_records.group(1))

        m_lrecl = re.search(r"\bLRECL\s*=\s*(\d+)", rest, re.IGNORECASE)
        if m_lrecl:
            self.lrecl = int(m_lrecl.group(1))

        self.nowide = bool(re.search(r"\bNOWIDE\b", rest, re.IGNORECASE))
        self.wide = bool(re.search(r"\bWIDE\b", rest, re.IGNORECASE))

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "filename": self.filename,
                "ignore_char": self.ignore_char,
                "ignore_list": self.ignore_list,
                "accept_list": self.accept_list,
            }
        )
        return d
