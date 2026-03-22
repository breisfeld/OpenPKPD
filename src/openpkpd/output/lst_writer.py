"""
.lst output file writer — main NONMEM report listing.

The .lst file is the primary human-readable output showing:
  - Problem title
  - Data summary
  - Estimation results
  - Parameter estimates with standard errors (if covariance step done)
  - ETA and EPS shrinkage
  - Gradient at final estimates
"""

from __future__ import annotations

import datetime
from typing import Any

from openpkpd.covariance.sandwich import CovarianceResult
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.errors import OutputError


def write_lst(
    path: str,
    result: EstimationResult,
    params: ParameterSet,
    title: str = "",
    cov_result: CovarianceResult | None = None,
    n_subjects: int = 0,
    n_obs: int = 0,
    method: str = "FOCE",
    problem_no: int = 1,
) -> None:
    """
    Write a NONMEM-compatible .lst listing file.
    """
    try:
        with open(path, "w") as fh:
            _write_header(fh, title, problem_no)
            _write_data_info(fh, n_subjects, n_obs)
            _write_estimation_results(fh, result, params, method)
            if cov_result is not None:
                _write_covariance_results(fh, cov_result, params)
            _write_footer(fh)
    except OSError as exc:
        raise OutputError(f"Failed to write .lst file {path!r}: {exc}") from exc


def _write_header(fh: Any, title: str, problem_no: int) -> None:
    fh.write(f"\n\nPROBLEM NO.: {problem_no}\n")
    fh.write(f"PROBLEM TITLE: {title}\n")
    fh.write(f"\nDate: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    fh.write("Computed with openpkpd\n\n")


def _write_data_info(fh: Any, n_subjects: int, n_obs: int) -> None:
    fh.write(f"NO. OF DATA RECS: {n_subjects + n_obs}\n")
    fh.write(f"NO. OF INDIVIDUALS: {n_subjects}\n")
    fh.write(f"NO. OF OBS RECS: {n_obs}\n\n")


def _write_estimation_results(
    fh: Any,
    result: EstimationResult,
    params: ParameterSet,
    method: str,
) -> None:
    fh.write(f"ESTIMATION METHOD: {method}\n")
    fh.write(f"CONVERGENCE: {'YES' if result.converged else 'NO'}\n")
    fh.write("MINIMIZATION SUCCESSFUL\n" if result.converged else "MINIMIZATION TERMINATED\n")
    fh.write("\nFINAL PARAMETER ESTIMATES\n\n")

    # THETA
    fh.write("THETA - VECTOR OF FIXED EFFECTS PARAMETERS:\n")
    for i, (val, spec) in enumerate(zip(result.theta_final, params.theta_specs, strict=False)):
        label = spec.label or f"TH {i + 1}"
        fh.write(f"  TH {i + 1:2d}  {val:>15.4E}   {label}\n")

    fh.write("\nOMEGA - COV MATRIX FOR RANDOM EFFECTS:\n")
    n = result.omega_final.shape[0]
    for r in range(n):
        for c in range(r + 1):
            fh.write(f"  OMEGA({r + 1},{c + 1}) = {result.omega_final[r, c]:>15.4E}\n")

    fh.write("\nSIGMA - COV MATRIX FOR RESIDUAL EFFECTS:\n")
    n_s = result.sigma_final.shape[0]
    for r in range(n_s):
        for c in range(r + 1):
            fh.write(f"  SIGMA({r + 1},{c + 1}) = {result.sigma_final[r, c]:>15.4E}\n")

    fh.write(f"\nOBJ FUNC VAL:  {result.ofv:>20.6f}\n")
    fh.write(f"AIC:           {result.aic:>20.4f}\n")
    fh.write(f"BIC:           {result.bic:>20.4f}   (n_obs = {result.n_observations})\n")

    if result.eta_shrinkage is not None and len(result.eta_shrinkage) > 0:
        fh.write("\nETA SHRINKAGE IN %:\n")
        for k, sh in enumerate(result.eta_shrinkage):
            fh.write(f"  ETA{k + 1}: {sh * 100:8.2f}%\n")

    if result.eps_shrinkage is not None and len(result.eps_shrinkage) > 0:
        fh.write("\nEPS SHRINKAGE IN %:\n")
        for k, sh in enumerate(result.eps_shrinkage):
            fh.write(f"  EPS{k + 1}: {sh * 100:8.2f}%\n")

    all_warnings = list(result.warnings)
    if hasattr(result, "shrinkage_warnings"):
        all_warnings.extend(result.shrinkage_warnings)
    if all_warnings:
        fh.write("\nWARNINGS:\n")
        for w in all_warnings:
            fh.write(f"  {w}\n")


def _write_covariance_results(
    fh: Any,
    cov_result: CovarianceResult,
    params: ParameterSet,
) -> None:
    fh.write("\n\nCOVARIANCE STEP COMPLETED\n\n")
    fh.write("STANDARD ERRORS OF FINAL ESTIMATES:\n\n")
    fh.write("THETA:\n")
    free_theta = [i for i, s in enumerate(params.theta_specs) if not s.fixed]
    for j, i in enumerate(free_theta):
        spec = params.theta_specs[i]
        label = spec.label or f"TH {i + 1}"
        if j < len(cov_result.se):
            fh.write(f"  TH {i + 1:2d}  SE={cov_result.se[j]:>12.4E}   {label}\n")

    if cov_result.condition_number:
        fh.write(f"\nCONDITION NUMBER OF COVARIANCE MATRIX: {cov_result.condition_number:.2E}\n")

    if (
        hasattr(cov_result, "eigenvalues")
        and cov_result.eigenvalues is not None
        and len(cov_result.eigenvalues) > 0
    ):
        fh.write("\nEIGENVALUES OF COVARIANCE MATRIX:\n")
        for i, ev in enumerate(cov_result.eigenvalues):
            fh.write(f"  {i + 1:3d}: {ev:>15.4E}\n")


def _write_footer(fh: Any) -> None:
    fh.write("\n\nStop Time:\n")
    fh.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
