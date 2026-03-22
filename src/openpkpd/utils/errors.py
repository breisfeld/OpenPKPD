"""Custom exception hierarchy for openpkpd."""

from __future__ import annotations


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
