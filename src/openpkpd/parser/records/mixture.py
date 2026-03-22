"""$MIXTURE record parser."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseRecord


class MixtureRecord(BaseRecord):
    """
    $MIXTURE NSPOP=2 [PMIX=THETA(n)]

    Defines finite mixture sub-populations.
    """

    record_name = "MIXTURE"

    def _parse(self, text: str) -> None:
        flat = re.sub(r";.*", "", " ".join(text.splitlines()))
        m = re.search(r"\bNSPOP\s*=\s*(\d+)", flat, re.IGNORECASE)
        self.nspop: int = int(m.group(1)) if m else 2
        m = re.search(r"\bPMIX\s*=\s*THETA\((\d+)\)", flat, re.IGNORECASE)
        self.pmix_theta_index: int | None = int(m.group(1)) if m else None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["nspop"] = self.nspop
        d["pmix_theta_index"] = self.pmix_theta_index
        return d
