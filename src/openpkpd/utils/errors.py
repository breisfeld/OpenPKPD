"""Custom exception hierarchy and structured warning types for openpkpd."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


# ── Warning infrastructure ─────────────────────────────────────────────────────


class WarningSeverity(enum.Enum):
    """Severity level for an EstimationWarning."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class WarningCode(enum.Enum):
    """
    Structured codes for estimation diagnostics.

    WARN_001  Condition number > 1 000 (moderate ill-conditioning).
    WARN_002  Condition number > 10 000 (severe ill-conditioning).
    WARN_003  Gradient norm at convergence exceeds tolerance (questionable convergence).
    WARN_004  Near-singular Omega: smallest eigenvalue below threshold.
    WARN_005  High ETA shrinkage > 30 % (parameter identifiability concern).
    WARN_006  Low IMP effective sample size (poor importance-sampling coverage).
    WARN_007  SAEM phase-2 parameter stability criterion not met.
    WARN_008  IOV ETA gradient may be unreliable (occasion structure incomplete).
    """

    WARN_001 = "WARN_001"
    WARN_002 = "WARN_002"
    WARN_003 = "WARN_003"
    WARN_004 = "WARN_004"
    WARN_005 = "WARN_005"
    WARN_006 = "WARN_006"
    WARN_007 = "WARN_007"
    WARN_008 = "WARN_008"


@dataclass
class EstimationWarning:
    """
    A structured diagnostic message attached to an EstimationResult.

    Attributes:
        code:     Unique WarningCode identifying the category.
        message:  Human-readable description, including numeric values.
        severity: INFO, WARNING, or ERROR.
    """

    code: WarningCode
    message: str
    severity: WarningSeverity = field(default=WarningSeverity.WARNING)

    def __str__(self) -> str:  # noqa: D105
        return f"[{self.code.value}] {self.message}"

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"EstimationWarning(code={self.code!r}, "
            f"severity={self.severity!r}, message={self.message!r})"
        )


# ── Exception hierarchy ────────────────────────────────────────────────────────


class PyNONMEMError(Exception):
    """Base exception for all openpkpd errors."""


class ParseError(PyNONMEMError):
    """Raised when parsing a NONMEM control stream or data file fails."""

    def __init__(self, message: str, line: int | None = None, context: str | None = None) -> None:
        self.line = line
        self.context = context
        detail = f" (line {line})" if line is not None else ""
        ctx = f"\n  Context: {context}" if context else ""
        super().__init__(f"{message}{detail}{ctx}")


class DataError(PyNONMEMError):
    """Raised when dataset validation or preprocessing fails."""


class ModelError(PyNONMEMError):
    """Raised when model assembly or evaluation fails."""


class EstimationError(PyNONMEMError):
    """Raised when an estimation method encounters an unrecoverable error."""


class ConvergenceError(EstimationError):
    """Raised when an optimization fails to converge within tolerances."""

    def __init__(
        self,
        message: str,
        iterations: int | None = None,
        final_ofv: float | None = None,
    ) -> None:
        self.iterations = iterations
        self.final_ofv = final_ofv
        parts = [message]
        if iterations is not None:
            parts.append(f"iterations={iterations}")
        if final_ofv is not None:
            parts.append(f"final_ofv={final_ofv:.6g}")
        super().__init__(", ".join(parts))


class NumericalError(PyNONMEMError):
    """Raised for numerical issues (singular matrix, non-PD matrix, etc.)."""


class PKError(PyNONMEMError):
    """Raised when PK model evaluation fails."""


class CompilerError(PyNONMEMError):
    """Raised when the NMTRANCompiler cannot translate a code block."""


class OutputError(PyNONMEMError):
    """Raised when writing output files fails."""


class ParseWarning(UserWarning):
    """Emitted when a control stream record contains a suspicious but parseable construct."""
