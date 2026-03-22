"""$DESIGN record parser (optimal design)."""

from __future__ import annotations

import re

from .base import BaseRecord


class DesignRecord(BaseRecord):
    """$DESIGN — optimal design parameters."""

    record_name = "DESIGN"

    def _parse(self, text: str) -> None:
        flat = re.sub(r";.*", "", " ".join(text.splitlines()))
        self.options: dict[str, str] = {}
        for m in re.finditer(r"\b([A-Z_]+)\s*=\s*(\S+)", flat, re.IGNORECASE):
            self.options[m.group(1).upper()] = m.group(2)
