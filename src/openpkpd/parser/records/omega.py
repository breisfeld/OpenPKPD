"""$OMEGA record parser."""

from __future__ import annotations

import re
from typing import Any

from openpkpd.model.parameters import OmegaSpec
from openpkpd.utils.errors import ParseError

from .base import BaseRecord


class OmegaRecord(BaseRecord):
    """
    $OMEGA
      0.1                 ; diagonal scalar
      BLOCK(2)            ; 2x2 block follows
        0.1
        0.01 0.2
      SAME                ; IOV block (same as previous)
      FIXED 0.1
    """

    record_name = "OMEGA"

    _NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?"

    def _parse(self, text: str) -> None:
        self.specs: list[OmegaSpec] = []
        # Strip comments, flatten lines
        lines = text.splitlines()
        clean: list[str] = []
        for line in lines:
            pos = line.find(";")
            clean.append(line[:pos] if pos >= 0 else line)
        flat = " ".join(clean)

        self._parse_flat(flat)

    def _parse_flat(self, flat: str) -> None:
        remaining = flat.strip()
        while remaining:
            remaining = remaining.strip()
            if not remaining:
                break

            # SAME keyword
            m_same = re.match(r"\bSAME\b", remaining, re.IGNORECASE)
            if m_same:
                self.specs.append(OmegaSpec(block_size=1, values=[], same=True))
                remaining = remaining[m_same.end() :]
                continue

            # BLOCK(n)
            m_block = re.match(r"\bBLOCK\s*\(\s*(\d+)\s*\)", remaining, re.IGNORECASE)
            if m_block:
                n = int(m_block.group(1))
                remaining = remaining[m_block.end() :].strip()
                # Check for SAME after BLOCK(n)
                m_same2 = re.match(r"\bSAME\b", remaining, re.IGNORECASE)
                if m_same2:
                    self.specs.append(OmegaSpec(block_size=n, values=[], same=True))
                    remaining = remaining[m_same2.end() :]
                    continue
                # Check FIXED
                fixed = False
                m_fix = re.match(r"\bFIX(?:ED)?\b", remaining, re.IGNORECASE)
                if m_fix:
                    fixed = True
                    remaining = remaining[m_fix.end() :].strip()
                # Read n*(n+1)/2 values
                n_vals = n * (n + 1) // 2
                vals: list[float] = []
                for _ in range(n_vals):
                    remaining = remaining.strip()
                    m_num = re.match(self._NUM, remaining)
                    if not m_num:
                        raise ParseError(
                            f"Expected number in $OMEGA BLOCK({n}), got: {remaining[:20]!r}"
                        )
                    vals.append(float(m_num.group()))
                    remaining = remaining[m_num.end() :]
                self.specs.append(OmegaSpec(block_size=n, values=vals, fixed=fixed))
                continue

            # FIXED keyword before value
            fixed = False
            m_fix = re.match(r"\bFIX(?:ED)?\b", remaining, re.IGNORECASE)
            if m_fix:
                fixed = True
                remaining = remaining[m_fix.end() :].strip()

            # Bare number (diagonal element)
            m_num = re.match(self._NUM, remaining)
            if m_num:
                val = float(m_num.group())
                remaining = remaining[m_num.end() :]
                # Check for FIXED after number
                remaining2 = remaining.strip()
                m_fix2 = re.match(r"\bFIX(?:ED)?\b", remaining2, re.IGNORECASE)
                if m_fix2:
                    fixed = True
                    remaining = remaining2[m_fix2.end() :]
                self.specs.append(OmegaSpec(block_size=1, values=[val], fixed=fixed))
                continue

            # Skip unrecognised character
            remaining = remaining[1:]

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["specs"] = [
            {
                "block_size": s.block_size,
                "values": s.values,
                "fixed": s.fixed,
                "same": s.same,
            }
            for s in self.specs
        ]
        return d
