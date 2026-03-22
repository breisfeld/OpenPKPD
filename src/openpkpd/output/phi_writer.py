"""
.phi output file writer — individual EBE (Empirical Bayes Estimates) output.

NONMEM .phi format:
  TABLE NO.     1: FIRST ORDER CONDITIONAL ESTIMATION
  SUBJECT_NO   ID   ETA1   ETA2   ...   ETC1,1  ETC2,1  ETC2,2  ...   OBJ
"""

from __future__ import annotations

import numpy as np

from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.errors import OutputError


def write_phi(
    path: str,
    result: EstimationResult,
    params: ParameterSet,
    subject_ids: list[int],
    method: str = "FOCE",
    problem_no: int = 1,
) -> None:
    """
    Write a NONMEM-compatible .phi file with individual EBEs.

    Args:
        path:        Output file path.
        result:      EstimationResult with post_hoc_etas.
        params:      Final ParameterSet.
        subject_ids: Ordered list of subject IDs.
        method:      Estimation method for header.
        problem_no:  Problem number.
    """
    n_eta = params.n_eta()

    try:
        with open(path, "w") as fh:
            fh.write(f"TABLE NO.     {problem_no}: {method}\n")

            # Header
            eta_cols = [f"ETA({k + 1})" for k in range(n_eta)]
            etc_cols = []
            for r in range(n_eta):
                for c in range(r + 1):
                    etc_cols.append(f"ETC({r + 1},{c + 1})")

            header = ["SUBJECT_NO", "ID"] + eta_cols + etc_cols + ["OBJ"]
            fh.write(" ".join(f"{h:>15}" for h in header) + "\n")

            # Default ETC (identity, no individual covariance computed)
            etc_identity = []
            for r in range(n_eta):
                for c in range(r + 1):
                    etc_identity.append(1.0 if r == c else 0.0)

            for idx, sid in enumerate(subject_ids):
                eta_i = result.post_hoc_etas.get(sid, np.zeros(n_eta))
                # Individual OFV (not separately tracked, use 0)
                row = [idx + 1, sid] + list(eta_i) + etc_identity + [0.0]
                fh.write(" ".join(f"{v:>15.6E}" for v in row) + "\n")

    except OSError as exc:
        raise OutputError(f"Failed to write .phi file {path!r}: {exc}") from exc
