"""
Sandwich (R/S) covariance estimator.

The sandwich covariance matrix for the FOCE estimator is:

    Cov(θ) = R^{-1} S R^{-1}

where:
    R = -d²/dθ² [OFV(θ)]  (Hessian of OFV at convergence, ≈ Fisher information)
    S = Σ_i g_i g_i^T       (outer product of per-subject gradients)
    g_i = d/dθ [OFV_i(θ)]

For NONMEM compatibility:
  .cov file: Cov(θ) = R^{-1} S R^{-1}
  .cor file: correlation matrix from Cov(θ)
  Standard errors: sqrt(diag(Cov(θ)))

Reference: Beal (1992); White (1982) sandwich estimator.
"""

from __future__ import annotations

import warnings as warnings_module
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpkpd.math.matrix import numerical_gradient, numerical_hessian, repair_pd
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.logging import get_logger

logger = get_logger("covariance.sandwich")


class CovarianceEstimationWarning(UserWarning):
    pass


@dataclass
class CovarianceResult:
    """
    Result of covariance step estimation.

    Attributes:
        cov_matrix:    Full parameter covariance matrix.
        cor_matrix:    Correlation matrix.
        se:            Standard errors for all free parameters.
        r_matrix:      R matrix (negative Hessian).
        s_matrix:      S matrix (sum of outer products of gradients).
        condition_number: Condition number of R matrix.
        converged:     True if covariance step completed successfully.
        warnings:      Warning messages.
        param_names:   Names of parameters in order.
    """

    cov_matrix: np.ndarray
    cor_matrix: np.ndarray
    se: np.ndarray
    r_matrix: np.ndarray
    s_matrix: np.ndarray
    condition_number: float = float("nan")
    converged: bool = False
    warnings: list[str] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)


class SandwichCovariance:
    """
    Sandwich (R/S) covariance estimator.

    Computes the sandwich covariance matrix at the final parameter estimates
    using numerical derivatives of the OFV.
    """

    def __init__(
        self,
        eps: float = 1e-4,
        matrix: str = "SR",
    ) -> None:
        """
        Args:
            eps:    Step size for numerical differentiation.
            matrix: Which matrix to use: 'S' (outer product only),
                    'R' (Hessian only), or 'SR' (sandwich).
        """
        self.eps = eps
        self.matrix = matrix.upper()

    def compute(
        self,
        population_model: Any,
        params: ParameterSet,
        eta_hat: dict[int, np.ndarray],
    ) -> CovarianceResult:
        """
        Compute the sandwich covariance matrix.

        Args:
            population_model: Assembled PopulationModel.
            params:           Final ParameterSet at convergence.
            eta_hat:          Post-hoc ETA estimates {subj_id: eta_vector}.

        Returns:
            CovarianceResult with cov/cor matrices and standard errors.
        """
        x_final = params.to_vector()
        n_params = len(x_final)

        logger.info(f"Computing {self.matrix} covariance matrix, {n_params} free parameters")

        # Build per-subject OFV functions
        def make_indiv_ofv(subj_id: int) -> Callable:
            indiv = population_model.individual_model(subj_id)
            eta_i = eta_hat.get(subj_id, np.zeros(params.n_eta()))

            def ofv_i(x: np.ndarray) -> float:
                p = ParameterSet.from_vector(x, params).apply_bounds()
                try:
                    return float(
                        indiv.obj_eta(
                            eta_i,
                            p.theta,
                            p.omega,
                            p.sigma,
                            trans=population_model.trans,
                        )
                    )
                except Exception as e:
                    logger.warning("Subject %s individual OFV failed during covariance: %s", subj_id, e)
                    return 1e10

            return ofv_i

        # Total OFV function
        def total_ofv(x: np.ndarray) -> float:
            return sum(make_indiv_ofv(sid)(x) for sid in population_model.subject_ids())

        # Compute R matrix (Hessian of total OFV)
        logger.debug("Computing R matrix (Hessian)...")
        try:
            R = numerical_hessian(total_ofv, x_final, eps=self.eps)
            R = (R + R.T) / 2  # Symmetrize
        except Exception as exc:
            logger.warning(f"Hessian computation failed: {exc}")
            R = np.eye(n_params)

        # Compute S matrix (outer product of per-subject gradients)
        logger.debug("Computing S matrix (gradient outer products)...")
        S = np.zeros((n_params, n_params))
        n_skipped = 0
        skipped_ids: list[int] = []
        for subj_id in population_model.subject_ids():
            ofv_i = make_indiv_ofv(subj_id)
            try:
                g_i = numerical_gradient(ofv_i, x_final, eps=self.eps)
                S += np.outer(g_i, g_i)
            except Exception as e:
                n_skipped += 1
                skipped_ids.append(subj_id)
                logger.warning("Sandwich: subject %s failed during S-matrix: %s", subj_id, e)

        if n_skipped > 0:
            warnings_module.warn(
                f"Sandwich covariance: {n_skipped} subject(s) failed during S-matrix accumulation "
                f"(IDs: {skipped_ids}). SE may be underestimated.",
                CovarianceEstimationWarning,
                stacklevel=2,
            )

        # Sandwich estimator
        warnings: list[str] = []
        cov_success = True
        try:
            R_inv = np.linalg.inv(repair_pd(R, epsilon=1e-10))
            cond = float(np.linalg.cond(R))
            if cond > 1e10:
                warnings.append(f"Condition number of R matrix is very large: {cond:.2e}")
                warnings_module.warn(
                    f"Covariance matrix condition number {cond:.2e} exceeds 1e10. "
                    "Standard errors may be unreliable.",
                    CovarianceEstimationWarning,
                    stacklevel=2,
                )
                cov_success = False
            elif cond > 1e6:
                warnings.append(f"Condition number of R matrix is large: {cond:.2e}")
        except np.linalg.LinAlgError as exc:
            warnings.append(f"R matrix is singular: {exc}")
            warnings_module.warn(
                "Hessian singular; covariance SE unreliable",
                CovarianceEstimationWarning,
                stacklevel=3,
            )
            R_inv = np.eye(n_params)
            cond = float("inf")
            cov_success = False

        if self.matrix == "S":
            cov = S
        elif self.matrix == "R":
            cov = R_inv
        else:  # SR (sandwich)
            cov = R_inv @ S @ R_inv

        cov = repair_pd(cov, epsilon=1e-12)

        # Standard errors
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))

        # Correlation matrix
        se_outer = np.outer(se, se)
        with np.errstate(divide="ignore", invalid="ignore"):
            cor = np.where(se_outer > 0, cov / se_outer, 0.0)
        np.fill_diagonal(cor, 1.0)

        # Parameter names
        param_names = _build_param_names(params)

        result = CovarianceResult(
            cov_matrix=cov,
            cor_matrix=cor,
            se=se,
            r_matrix=R,
            s_matrix=S,
            condition_number=cond,
            converged=cov_success and len(warnings) == 0,
            warnings=warnings,
            param_names=param_names,
        )
        result.skipped_subject_ids = skipped_ids  # type: ignore[attr-defined]
        if warnings:
            logger.warning(f"Covariance step warnings: {warnings}")
        logger.info(f"Covariance step completed, condition number={cond:.2e}")
        return result


def _build_param_names(params: ParameterSet) -> list[str]:
    """Build ordered list of free parameter names for the covariance matrix."""
    names: list[str] = []
    # THETAs (free only)
    for i, spec in enumerate(params.theta_specs):
        if not spec.fixed:
            label = spec.label or f"THETA{i + 1}"
            names.append(label)
    # OMEGA elements — free blocks only, lower-triangular within each block
    offset = 0
    for ospec in params.omega_specs:
        if not ospec.fixed:
            for dc in range(ospec.block_size):
                for dr in range(dc, ospec.block_size):
                    names.append(f"OMEGA({offset + dr + 1},{offset + dc + 1})")
        offset += ospec.block_size
    # SIGMA elements — free blocks only, lower-triangular within each block
    offset = 0
    for sspec in params.sigma_specs:
        if not sspec.fixed:
            for dc in range(sspec.block_size):
                for dr in range(dc, sspec.block_size):
                    names.append(f"SIGMA({offset + dr + 1},{offset + dc + 1})")
        offset += sspec.block_size
    return names
