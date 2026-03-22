"""$SIMULATION record parser."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseRecord


class SimulationRecord(BaseRecord):
    """
    $SIMULATION (seed) [ONLYSIMULATION] [SUBPROBLEMS=n] [TRUE=FINAL]
    """

    record_name = "SIMULATION"

    def _parse(self, text: str) -> None:
        flat = re.sub(r";.*", "", " ".join(text.splitlines()))

        self.seeds: list[int] = []
        self.subproblems: int = 1
        self.onlysimulation: bool = False
        self.true_final: bool = False
        self.new_seed: int | None = None

        # Extract (seed) groups
        for m in re.finditer(r"\((\d+)(?:\s+(\d+))?\)", flat):
            self.seeds.append(int(m.group(1)))
            if m.group(2):
                self.seeds.append(int(m.group(2)))

        m_sub = re.search(r"\bSUBPROBLEMS?\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m_sub:
            self.subproblems = int(m_sub.group(1))

        self.onlysimulation = bool(re.search(r"\bONLYSIMULATION\b", flat, re.IGNORECASE))
        self.true_final = bool(re.search(r"\bTRUE\s*=\s*FINAL\b", flat, re.IGNORECASE))

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "seeds": self.seeds,
                "subproblems": self.subproblems,
                "onlysimulation": self.onlysimulation,
                "true_final": self.true_final,
            }
        )
        return d
