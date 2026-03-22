"""
.ext output file writer — iteration-by-iteration parameter estimates.

NONMEM .ext format:
  TABLE NO. 1: ESTIMATION METHOD: FOCE
  ITERATION   THETA1   THETA2 ... SIGMA(1,1)  OBJ
  0           init...                         init_OFV
  1           ...
  ...
  -1000000000 final...                        final_OFV
"""

from __future__ import annotations

from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.errors import OutputError


def write_ext(
    path: str,
    result: EstimationResult,
    params: ParameterSet,
    method: str = "FOCE",
    problem_no: int = 1,
) -> None:
    """
    Write a NONMEM-compatible .ext file.

    Args:
        path:       Output file path.
        result:     EstimationResult from estimation run.
        params:     Final ParameterSet.
        method:     Estimation method name for header.
        problem_no: Problem number (for TABLE NO. header).
    """
    try:
        with open(path, "w") as fh:
            fh.write(f"TABLE NO.     {problem_no}: ESTIMATION METHOD: {method}\n")
            # Build header
            header = _build_ext_header(params)
            fh.write(f"{'ITERATION':>15}" + "".join(f"{h:>15}" for h in header) + f"{'OBJ':>20}\n")

            # Write OFV history
            params.to_vector()
            if result.ofv_history:
                fh.write(
                    f"{'0':>15}"
                    + "".join(f"{v:>15.6E}" for v in _ext_params(params))
                    + f"{result.ofv_history[0]:>20.6f}\n"
                )
                for i, ofv in enumerate(result.ofv_history[1:], start=1):
                    fh.write(
                        f"{i:>15}"
                        + "".join(f"{v:>15.6E}" for v in _ext_params(params))
                        + f"{ofv:>20.6f}\n"
                    )

            # Final estimates row: iteration = -1000000000
            fh.write(
                f"{-1000000000:>15}"
                + "".join(f"{v:>15.6E}" for v in _ext_params_final(result, params))
                + f"{result.ofv:>20.6f}\n"
            )

    except OSError as exc:
        raise OutputError(f"Failed to write .ext file {path!r}: {exc}") from exc


def _build_ext_header(params: ParameterSet) -> list[str]:
    """Build column names for .ext file."""
    names: list[str] = []
    for i, _spec in enumerate(params.theta_specs):
        names.append(f"THETA{i + 1}")
    n = params.omega.shape[0]
    for r in range(n):
        for c in range(r + 1):
            names.append(f"OMEGA({r + 1},{c + 1})")
    n_s = params.sigma.shape[0]
    for r in range(n_s):
        for c in range(r + 1):
            names.append(f"SIGMA({r + 1},{c + 1})")
    return names


def _ext_params(params: ParameterSet) -> list[float]:
    """Extract parameter values in .ext column order."""
    vals: list[float] = list(params.theta)
    n = params.omega.shape[0]
    for r in range(n):
        for c in range(r + 1):
            vals.append(float(params.omega[r, c]))
    n_s = params.sigma.shape[0]
    for r in range(n_s):
        for c in range(r + 1):
            vals.append(float(params.sigma[r, c]))
    return vals


def _ext_params_final(result: EstimationResult, template: ParameterSet) -> list[float]:
    """Extract final parameter values from EstimationResult."""
    from openpkpd.model.parameters import ParameterSet

    final = ParameterSet(
        theta=result.theta_final,
        omega=result.omega_final,
        sigma=result.sigma_final,
        theta_specs=template.theta_specs,
        omega_specs=template.omega_specs,
        sigma_specs=template.sigma_specs,
    )
    return _ext_params(final)
