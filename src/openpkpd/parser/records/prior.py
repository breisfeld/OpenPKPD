"""$PRIOR, $THETAP, $THETAPV, $OMEGAP, $OMEGAPD, $SIGMAP, $SIGMAPD record parsers."""

from __future__ import annotations

import re

from .base import BaseRecord


class PriorRecord(BaseRecord):
    """$PRIOR NWPRI [NTHETA=n] [NETA=n] [NEPS=n]"""

    record_name = "PRIOR"

    def _parse(self, text: str) -> None:
        flat = re.sub(r";.*", "", " ".join(text.splitlines()))
        self.type: str = "NWPRI"
        m = re.search(r"\b(NWPRI|TNPRI)\b", flat, re.IGNORECASE)
        if m:
            self.type = m.group(1).upper()
        self.ntheta: int | None = None
        self.neta: int | None = None
        self.neps: int | None = None
        for kw in ("NTHETA", "NETA", "NEPS"):
            mm = re.search(rf"\b{kw}\s*=\s*(\d+)", flat, re.IGNORECASE)
            if mm:
                setattr(self, kw.lower(), int(mm.group(1)))


class ThetaPRecord(BaseRecord):
    """$THETAP — prior mean for THETA (same syntax as $THETA)."""

    record_name = "THETAP"

    def _parse(self, text: str) -> None:
        self.values: list[float] = []
        for m in re.finditer(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?", text):
            self.values.append(float(m.group()))


class ThetaPVRecord(BaseRecord):
    """$THETAPV — prior variance matrix for THETA."""

    record_name = "THETAPV"

    def _parse(self, text: str) -> None:
        self.values: list[float] = []
        for m in re.finditer(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?", text):
            self.values.append(float(m.group()))


class OmegaPRecord(BaseRecord):
    """$OMEGAP — prior mean for OMEGA (lower triangular)."""

    record_name = "OMEGAP"

    def _parse(self, text: str) -> None:
        self.values: list[float] = []
        for m in re.finditer(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?", text):
            self.values.append(float(m.group()))


class OmegaPDRecord(BaseRecord):
    """$OMEGAPD — degrees of freedom for Wishart prior on OMEGA."""

    record_name = "OMEGAPD"

    def _parse(self, text: str) -> None:
        self.values: list[float] = []
        for m in re.finditer(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?", text):
            self.values.append(float(m.group()))


class SigmaPRecord(BaseRecord):
    """$SIGMAP — prior mean for SIGMA."""

    record_name = "SIGMAP"

    def _parse(self, text: str) -> None:
        self.values: list[float] = []
        for m in re.finditer(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?", text):
            self.values.append(float(m.group()))


class SigmaPDRecord(BaseRecord):
    """$SIGMAPD — degrees of freedom for Wishart prior on SIGMA."""

    record_name = "SIGMAPD"

    def _parse(self, text: str) -> None:
        self.values: list[float] = []
        for m in re.finditer(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?", text):
            self.values.append(float(m.group()))
