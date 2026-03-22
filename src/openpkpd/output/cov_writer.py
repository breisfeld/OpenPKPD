"""
.cov output file writer — parameter covariance matrix.
"""

from __future__ import annotations

import numpy as np

from openpkpd.covariance.sandwich import CovarianceResult
from openpkpd.utils.errors import OutputError


def write_cov(
    path: str,
    cov_result: CovarianceResult,
    problem_no: int = 1,
) -> None:
    """Write NONMEM-compatible .cov file."""
    _write_matrix_file(
        path,
        cov_result.cov_matrix,
        cov_result.param_names,
        f"TABLE NO.     {problem_no}: COVARIANCE MATRIX",
    )


def write_cor(
    path: str,
    cov_result: CovarianceResult,
    problem_no: int = 1,
) -> None:
    """Write NONMEM-compatible .cor file (correlation matrix + SE diagonal)."""
    _write_matrix_file(
        path,
        cov_result.cor_matrix,
        cov_result.param_names,
        f"TABLE NO.     {problem_no}: CORRELATION MATRIX",
        se_row=cov_result.se,
    )


def _write_matrix_file(
    path: str,
    matrix: np.ndarray,
    param_names: list[str],
    header: str,
    se_row: np.ndarray | None = None,
) -> None:
    """Write a NONMEM-style symmetric matrix output file."""
    try:
        len(param_names)
        with open(path, "w") as fh:
            fh.write(header + "\n")
            # Header row
            fh.write(f"{'NAME':>15}" + "".join(f"{nm:>15}" for nm in param_names) + "\n")
            # Optional SE row (for .cor files)
            if se_row is not None:
                fh.write(f"{'SE':>15}" + "".join(f"{v:>15.6E}" for v in se_row) + "\n")
            # Matrix rows
            for i, name in enumerate(param_names):
                row_vals = matrix[i, :]
                fh.write(f"{name:>15}" + "".join(f"{v:>15.6E}" for v in row_vals) + "\n")
    except OSError as exc:
        raise OutputError(f"Failed to write matrix file {path!r}: {exc}") from exc
