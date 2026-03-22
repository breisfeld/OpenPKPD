"""$INPUT record parser."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseRecord


class InputRecord(BaseRecord):
    """
    $INPUT ID TIME AMT DV EVID WT=WEIGHT DROP ...

    Maps dataset column positions to names and optional aliases.
    DROP / SKIP columns are recorded but excluded from the active column list.
    """

    record_name = "INPUT"

    def _parse(self, text: str) -> None:
        # Flatten multi-line text
        lines = [
            ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(";")
        ]
        flat = " ".join(lines)
        # Remove inline comments
        flat = re.sub(r";.*", "", flat)

        self.columns: list[str] = []  # Active column names in order
        self.aliases: dict[str, str] = {}  # alias → original  (COL=ALIAS)
        self.dropped: list[int] = []  # 1-based column positions that are DROP/SKIP

        tokens = re.split(r"[\s,]+", flat.strip())
        pos = 1
        for tok in tokens:
            if not tok:
                continue
            upper = tok.upper()
            if "=" in tok:
                parts = tok.split("=", 1)
                col_name = parts[0].strip().upper()
                alias = parts[1].strip().upper()
                if alias in ("DROP", "SKIP"):
                    self.dropped.append(pos)
                    self.columns.append(f"_DROP_{pos}")
                else:
                    self.columns.append(col_name)
                    self.aliases[alias] = col_name
            elif upper in ("DROP", "SKIP"):
                self.dropped.append(pos)
                self.columns.append(f"_DROP_{pos}")
            else:
                self.columns.append(upper)
            pos += 1

    def active_columns(self) -> list[str]:
        """Return only non-dropped column names."""
        return [c for c in self.columns if not c.startswith("_DROP_")]

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["columns"] = self.columns
        d["aliases"] = self.aliases
        d["dropped"] = self.dropped
        return d
