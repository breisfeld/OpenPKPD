"""$THETA record parser."""

from __future__ import annotations

import math
import re
from typing import Any

from openpkpd.model.parameters import ThetaSpec
from openpkpd.utils.errors import ParseError

from .base import BaseRecord


class ThetaRecord(BaseRecord):
    """
    $THETA
      (low, init, high)   ; labelled bounds
      init                ; fixed bounds inferred
      (0, 1)              ; lower=0, init=1, upper=inf
      FIX init            ; fixed parameter
      init FIX
      LABEL = "CL"

    Produces a list of ThetaSpec instances.
    """

    record_name = "THETA"

    # Regexes for number parsing
    _NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?"

    def _parse(self, text: str) -> None:
        self.specs: list[ThetaSpec] = []
        # Remove comments
        lines = text.splitlines()
        clean_lines: list[str] = []
        for line in lines:
            pos = line.find(";")
            if pos >= 0:
                # preserve potential label before ;
                clean_lines.append(line[:pos])
            else:
                clean_lines.append(line)
        clean = " ".join(clean_lines)

        # Tokenise: extract parenthesised groups and bare tokens
        # Strategy: find (...) groups, then handle remaining tokens
        remaining = clean.strip()
        label_map: dict[int, str] = {}  # index → label

        while remaining:
            remaining = remaining.strip()
            if not remaining:
                break

            # Check for LABEL assignment like LABEL(n) = "name"
            m_label = re.match(
                r"LABEL\s*\(\s*(\d+)\s*\)\s*=\s*[\"']?([^\"',\s]+)[\"']?", remaining, re.IGNORECASE
            )
            if m_label:
                label_map[int(m_label.group(1))] = m_label.group(2)
                remaining = remaining[m_label.end() :].strip()
                continue

            # Parenthesised triplet: (low, init, high) or (low, init) or (init,)
            if remaining.startswith("("):
                end = remaining.find(")")
                if end < 0:
                    raise ParseError("Unclosed '(' in $THETA")
                inner = remaining[1:end]
                remaining = remaining[end + 1 :]
                spec = self._parse_parenthesised(inner)
                self.specs.append(spec)
                continue

            # Bare number, optionally followed by FIXED
            m_num = re.match(self._NUM, remaining)
            if m_num:
                val = float(m_num.group())
                remaining = remaining[m_num.end() :].strip()
                fixed = False
                m_fix = re.match(r"FIX(?:ED)?\b", remaining, re.IGNORECASE)
                if m_fix:
                    fixed = True
                    remaining = remaining[m_fix.end() :].strip()
                self.specs.append(ThetaSpec(init=val, fixed=fixed))
                continue

            # FIXED before a number: FIX 1.5
            m_fix_first = re.match(r"FIX(?:ED)?\s+(" + self._NUM + r")", remaining, re.IGNORECASE)
            if m_fix_first:
                val = float(m_fix_first.group(1))
                remaining = remaining[m_fix_first.end() :].strip()
                self.specs.append(ThetaSpec(init=val, fixed=True))
                continue

            # Skip unrecognised tokens
            remaining = remaining[1:]

        # Apply labels
        for i, label in label_map.items():
            if 1 <= i <= len(self.specs):
                self.specs[i - 1] = ThetaSpec(
                    init=self.specs[i - 1].init,
                    lower=self.specs[i - 1].lower,
                    upper=self.specs[i - 1].upper,
                    fixed=self.specs[i - 1].fixed,
                    label=label,
                )

    def _parse_parenthesised(self, inner: str) -> ThetaSpec:
        """Parse the content of a (...) theta specification."""
        parts = [p.strip() for p in inner.split(",")]
        fixed = False

        # Check for FIXED keyword anywhere
        clean_parts = []
        for p in parts:
            m = re.match(r"(.*?)\s*FIX(?:ED)?\s*$", p, re.IGNORECASE)
            if m:
                fixed = True
                clean_parts.append(m.group(1).strip())
            else:
                m2 = re.match(r"FIX(?:ED)?\s+(.*)", p, re.IGNORECASE)
                if m2:
                    fixed = True
                    clean_parts.append(m2.group(1).strip())
                else:
                    clean_parts.append(p)

        parts = clean_parts

        def to_float(s: str) -> float:
            s = s.strip()
            if not s:
                return 0.0
            # Allow -INF / INF variants
            if re.match(r"-?INF(INITY)?", s, re.IGNORECASE):
                return -float("inf") if s.startswith("-") else float("inf")
            return float(s)

        if len(parts) == 1:
            return ThetaSpec(init=to_float(parts[0]), fixed=fixed)
        elif len(parts) == 2:
            lo = to_float(parts[0]) if parts[0] else -float("inf")
            init = to_float(parts[1])
            return ThetaSpec(lower=lo, init=init, fixed=fixed)
        elif len(parts) == 3:
            lo = to_float(parts[0]) if parts[0] else -float("inf")
            init = to_float(parts[1])
            hi_str = parts[2]
            hi = float("inf") if not hi_str else to_float(hi_str)
            return ThetaSpec(lower=lo, init=init, upper=hi, fixed=fixed)
        else:
            raise ParseError(f"Cannot parse $THETA entry: ({inner})")

    def to_string(self) -> str:
        """Serialize THETA record from parsed specs."""
        lines = ["$THETA"]
        for spec in self.specs:
            has_lower = not math.isinf(spec.lower)
            has_upper = not math.isinf(spec.upper)
            if has_lower or has_upper:
                lo_s = "" if math.isinf(spec.lower) else str(spec.lower)
                hi_s = "" if math.isinf(spec.upper) else str(spec.upper)
                token = f"({lo_s},{spec.init},{hi_s})"
            else:
                token = str(spec.init)
            fix_s = " FIXED" if spec.fixed else ""
            label_s = f"  ; {spec.label}" if spec.label else ""
            lines.append(f"  {token}{fix_s}{label_s}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["specs"] = [
            {
                "init": s.init,
                "lower": s.lower,
                "upper": s.upper,
                "fixed": s.fixed,
                "label": s.label,
            }
            for s in self.specs
        ]
        return d
