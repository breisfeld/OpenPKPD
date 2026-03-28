"""$ESTIMATION record parser."""

from __future__ import annotations

import contextlib
import re
from typing import Any

from openpkpd.utils.constants import Method

from .base import BaseRecord

# Map NONMEM method names / abbreviations to canonical names
_METHOD_MAP: dict[str, str] = {
    "0": Method.FO,
    "ZERO": Method.FO,
    "FO": Method.FO,
    "1": Method.FOCE,
    "COND": Method.FOCE,
    "CONDITIONAL": Method.FOCE,
    "FOCE": Method.FOCE,
    "LAPLACE": Method.LAPLACIAN,
    "LAPLACIAN": Method.LAPLACIAN,
    "SAEM": Method.SAEM,
    "IMP": Method.IMP,
    "IMPMAP": Method.IMPMAP,
    "BAYES": Method.BAYES,
    "NUTS": Method.NUTS,
    "CHAIN": "CHAIN",
    "MAP": "MAP",
}


class EstimationRecord(BaseRecord):
    """
    $ESTIMATION METHOD=COND INTER MAXEVAL=9999 SIGDIG=3 PRINT=5 NOABORT
                 POSTHOC MSFO=file NOTHETABOUNDTEST NOOMEGABOUNDTEST
    """

    record_name = "ESTIMATION"

    def _parse(self, text: str) -> None:
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(";")]
        flat = re.sub(r";.*", "", " ".join(lines))

        self.method: str = Method.FOCE
        self.interaction: bool = False
        self.maxeval: int = 9999
        self.sigdig: int = 3
        self.sigl: int = 6
        self.print_interval: int = 5
        self.noabort: bool = False
        self.posthoc: bool = False
        self.msfo: str | None = None
        self.nooabort: bool = False
        self.laplace: bool = False
        self.auto: bool = False
        self.isample: int | None = None  # IMP samples
        self.niter: int | None = None  # SAEM iterations
        self.seed: int | None = None
        self.cinterval: float | None = None
        self.nothetaboundtest: bool = False
        self.noomegaboundtest: bool = False
        self.nsigmas: int | None = None
        self.format: str | None = None
        self.file: str | None = None
        self.unbounded: bool = False
        self.numerical: bool = False
        self.gradient: bool = False

        # METHOD= or METHOD 0 or METHOD=COND
        m = re.search(r"\bMETHOD\s*=\s*(\S+)", flat, re.IGNORECASE)
        if not m:
            m = re.search(r"\bMETHOD\s+(\S+)", flat, re.IGNORECASE)
        if m:
            raw_method = m.group(1).upper().rstrip(",")
            self.method = _METHOD_MAP.get(raw_method, raw_method)

        self.interaction = bool(re.search(r"\bINTER(?:ACTION)?\b", flat, re.IGNORECASE))
        self.laplace = bool(re.search(r"\bLAPLACE?\b", flat, re.IGNORECASE))
        if self.laplace:
            self.method = Method.LAPLACIAN

        m = re.search(r"\bMAXEVAL\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.maxeval = int(m.group(1))

        m = re.search(r"\bSIGDIG\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.sigdig = int(m.group(1))

        m = re.search(r"\bSIGL\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.sigl = int(m.group(1))

        m = re.search(r"\bPRINT\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.print_interval = int(m.group(1))

        self.noabort = bool(re.search(r"\bNOABORT\b", flat, re.IGNORECASE))
        self.posthoc = bool(re.search(r"\bPOSTHOC\b", flat, re.IGNORECASE))

        m = re.search(r"\bMSFO\s*=\s*(\S+)", flat, re.IGNORECASE)
        if m:
            self.msfo = m.group(1)

        m = re.search(r"\bISAMPLE\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.isample = int(m.group(1))

        m = re.search(r"\bNITER\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.niter = int(m.group(1))

        m = re.search(r"\bSEED\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.seed = int(m.group(1))

        self.nothetaboundtest = bool(re.search(r"\bNOTHETABOUNDTEST\b", flat, re.IGNORECASE))
        self.noomegaboundtest = bool(re.search(r"\bNOOMEGABOUNDTEST\b", flat, re.IGNORECASE))
        self.numerical = bool(re.search(r"\bNUMERICAL\b", flat, re.IGNORECASE))
        self.gradient = bool(re.search(r"\bGRADIENT\b", flat, re.IGNORECASE))

        # OpenPKPD extensions (not in standard NONMEM)
        self.n_starts: int = 1
        m = re.search(r"\bNSTARTS?\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.n_starts = int(m.group(1))

        self.gtol: float = 1e-5
        m = re.search(r"\bGTOL\s*=\s*([\d.eE+\-]+)", flat, re.IGNORECASE)
        if m:
            with contextlib.suppress(ValueError):
                self.gtol = float(m.group(1))

        self.perturbation_scale: float = 1.0
        m = re.search(r"\bPERTURB\s*=\s*([\d.eE+\-]+)", flat, re.IGNORECASE)
        if m:
            with contextlib.suppress(ValueError):
                self.perturbation_scale = float(m.group(1))

        self.outer_optimizer: str | None = None
        m = re.search(r"\bOUTEROPT\s*=\s*(\S+)", flat, re.IGNORECASE)
        if m:
            self.outer_optimizer = m.group(1).rstrip(",")

        self.outer_fallback_optimizer: str | None = None
        m = re.search(r"\bFALLBACKOPT\s*=\s*(\S+)", flat, re.IGNORECASE)
        if m:
            self.outer_fallback_optimizer = m.group(1).rstrip(",")

        self.outer_fallback_maxeval: int | None = None
        m = re.search(r"\bFALLBACKMAXEVAL\s*=\s*(\d+)", flat, re.IGNORECASE)
        if m:
            self.outer_fallback_maxeval = int(m.group(1))

        self.retain_best_iterate: bool | None = None
        if re.search(r"\bRETAINBEST\b", flat, re.IGNORECASE):
            self.retain_best_iterate = True
        elif re.search(r"\bNORETAINBEST\b", flat, re.IGNORECASE):
            self.retain_best_iterate = False

        self.retry_on_abnormal: bool | None = None
        if re.search(r"\bRETRYONABNORMAL\b", flat, re.IGNORECASE):
            self.retry_on_abnormal = True
        elif re.search(r"\bNORETRYONABNORMAL\b", flat, re.IGNORECASE):
            self.retry_on_abnormal = False

        self.retry_omega_scales: tuple[float, ...] = ()
        m = re.search(r"\bRETRYOMEGASCALE\s*=\s*([^\s]+)", flat, re.IGNORECASE)
        if m:
            parts = [p for p in m.group(1).split(",") if p]
            vals: list[float] = []
            for part in parts:
                with contextlib.suppress(ValueError):
                    vals.append(float(part))
            self.retry_omega_scales = tuple(vals)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "method": self.method,
                "interaction": self.interaction,
                "maxeval": self.maxeval,
                "sigdig": self.sigdig,
                "laplace": self.laplace,
            }
        )
        return d
