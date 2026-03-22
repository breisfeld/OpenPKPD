"""$SIGMA record parser (identical structure to $OMEGA)."""

from __future__ import annotations

from typing import Any

from openpkpd.model.parameters import SigmaSpec

from .base import BaseRecord
from .omega import OmegaRecord


class SigmaRecord(BaseRecord):
    """
    $SIGMA — residual error variance matrix.

    Syntax identical to $OMEGA. Produces list of SigmaSpec.
    """

    record_name = "SIGMA"

    def _parse(self, text: str) -> None:
        # Reuse OmegaRecord parsing logic
        _omega_proxy = OmegaRecord.__new__(OmegaRecord)
        _omega_proxy.raw_text = text
        _omega_proxy.header_line = self.header_line
        _omega_proxy.specs = []  # type: ignore[attr-defined]
        _omega_proxy._parse(text)  # type: ignore[attr-defined]

        # Convert OmegaSpec → SigmaSpec
        self.specs: list[SigmaSpec] = [
            SigmaSpec(
                block_size=s.block_size,
                values=s.values,
                fixed=s.fixed,
            )
            for s in _omega_proxy.specs  # type: ignore[attr-defined]
            if not s.same  # SAME not valid for SIGMA
        ]

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["specs"] = [
            {"block_size": s.block_size, "values": s.values, "fixed": s.fixed} for s in self.specs
        ]
        return d
