"""
Abstract base class and result dataclass for estimation methods.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpkpd.utils.errors import EstimationWarning, WarningCode, WarningSeverity


@dataclass
class EstimationResult:
    """
    Result from a completed estimation run.
    """

    theta_final: np.ndarray
    omega_final: np.ndarray
    sigma_final: np.ndarray
    ofv: float
    converged: bool = False
    condition_number: float | None = None
    eta_shrinkage: np.ndarray = field(default_factory=lambda: np.array([]))
    eps_shrinkage: np.ndarray = field(default_factory=lambda: np.array([]))
    post_hoc_etas: dict[int, np.ndarray] = field(default_factory=dict)
    ofv_history: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    structured_warnings: list[EstimationWarning] = field(default_factory=list)
    shrinkage_warnings: list[str] = field(default_factory=list)
    n_function_evals: int = 0
    elapsed_time: float = 0.0
    method: str = ""
    message: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    n_observations: int = 0
    n_subjects: int = 0
    _n_parameters: int = 0

    # ── Information criteria ──────────────────────────────────────────────

    @property
    def n_parameters(self) -> int:
        """
        Number of free parameters in the model.

        Counts free THETA elements plus unique free OMEGA and SIGMA
        elements (lower-triangular, excluding fixed zeros). If
        ``_n_parameters`` has been set explicitly (e.g. via
        ``compute_n_parameters``), that value is returned directly.
        Otherwise the count is inferred from the matrix shapes.
        """
        if self._n_parameters > 0:
            return self._n_parameters
        # Fallback: infer from matrix shapes
        n_theta = int(self.theta_final.size)
        # Count lower-triangular elements (including diagonal) for OMEGA
        n_omega = self.omega_final.shape[0]
        n_omega_params = n_omega * (n_omega + 1) // 2
        n_sigma = self.sigma_final.shape[0]
        n_sigma_params = n_sigma * (n_sigma + 1) // 2
        return n_theta + n_omega_params + n_sigma_params

    @property
    def aic(self) -> float:
        """
        Akaike Information Criterion.

        AIC = OFV + 2 * n_parameters

        where OFV = -2 * log-likelihood.
        """
        return self.ofv + 2.0 * self.n_parameters

    @property
    def bic(self) -> float:
        """
        Bayesian Information Criterion.

        BIC = OFV + ln(n_observations) * n_parameters

        Returns inf when n_observations <= 0.
        """
        if self.n_observations <= 0:
            return float("inf")
        return self.ofv + math.log(self.n_observations) * self.n_parameters

    def compute_n_parameters(
        self,
        theta_specs: list[Any] | None = None,
        omega_specs: list[Any] | None = None,
        sigma_specs: list[Any] | None = None,
    ) -> None:
        """
        Compute and store n_parameters from specs or infer from matrix shapes.

        If spec objects are provided they must expose a ``fixed`` boolean
        attribute. Elements with ``fixed=True`` do not contribute to the
        parameter count. When specs are absent, all elements are treated as
        free.

        Args:
            theta_specs: List of ThetaSpec-like objects with a ``fixed``
                         attribute, one per THETA element.
            omega_specs: List of OmegaSpec-like objects with a ``fixed``
                         attribute, one per lower-triangular OMEGA element.
            sigma_specs: List of SigmaSpec-like objects with a ``fixed``
                         attribute, one per lower-triangular SIGMA element.
        """
        n: int = 0

        # Count free THETA
        if theta_specs is not None:
            n += sum(1 for s in theta_specs if not getattr(s, "fixed", False))
        else:
            n += int(self.theta_final.size)

        # Count free OMEGA (lower triangular)
        if omega_specs is not None:
            n += sum(1 for s in omega_specs if not getattr(s, "fixed", False))
        else:
            n_eta = self.omega_final.shape[0]
            n += n_eta * (n_eta + 1) // 2

        # Count free SIGMA (lower triangular)
        if sigma_specs is not None:
            n += sum(1 for s in sigma_specs if not getattr(s, "fixed", False))
        else:
            n_eps = self.sigma_final.shape[0]
            n += n_eps * (n_eps + 1) // 2

        self._n_parameters = n

    # ── Structured warning helpers ─────────────────────────────────────────

    def add_structured_warning(
        self,
        code: WarningCode,
        message: str,
        severity: WarningSeverity = WarningSeverity.WARNING,
    ) -> None:
        """
        Append a structured EstimationWarning and its string representation.

        The string form is added to ``warnings`` for backward compatibility;
        the typed form is added to ``structured_warnings``.
        """
        w = EstimationWarning(code=code, message=message, severity=severity)
        self.structured_warnings.append(w)
        self.warnings.append(str(w))

    def check_omega_conditioning(self) -> None:
        """
        Check Omega eigenvalues and condition number; add WARN_001/002/004.

        Intended to be called after the M-step or covariance step.
        A 0×0 or empty Omega (sigma-only models) is silently skipped.
        """
        if self.omega_final is None or self.omega_final.size == 0:
            return
        try:
            eigenvalues = np.linalg.eigvalsh(self.omega_final)
            if eigenvalues.size == 0:
                return
            lmin = float(np.min(eigenvalues))
            lmax = float(np.max(eigenvalues))
        except np.linalg.LinAlgError:
            return

        if lmin < 1e-6:
            self.add_structured_warning(
                WarningCode.WARN_004,
                f"Near-singular Omega: smallest eigenvalue = {lmin:.3e} "
                "(consider re-parameterisation or fixing off-diagonal elements).",
                WarningSeverity.ERROR,
            )
            return  # condition number undefined

        cond = lmax / lmin
        if cond > 1e4:
            self.add_structured_warning(
                WarningCode.WARN_002,
                f"Omega condition number = {cond:.2e} (> 10 000). "
                "Standard errors and correlations may be unreliable.",
                WarningSeverity.ERROR,
            )
        elif cond > 1e3:
            self.add_structured_warning(
                WarningCode.WARN_001,
                f"Omega condition number = {cond:.2e} (> 1 000). "
                "Mild ill-conditioning; inspect parameter correlations.",
                WarningSeverity.WARNING,
            )

    def summary(self) -> str:
        """Return a text summary of the estimation result."""
        lines = [
            f"Method: {self.method}",
            f"OFV: {self.ofv:.4f}",
            f"AIC: {self.aic:.4f}",
            f"BIC: {self.bic:.4f}   (n_obs = {self.n_observations})",
            f"n_parameters: {self.n_parameters}",
            f"Converged: {self.converged}",
            f"THETA: {self.theta_final}",
            f"OMEGA (diagonal): {np.diag(self.omega_final)}",
            f"SIGMA (diagonal): {np.diag(self.sigma_final)}",
        ]
        if len(self.eta_shrinkage) > 0:
            sh_pct = [f"{s * 100:.1f}%" for s in self.eta_shrinkage]
            lines.append(f"ETA shrinkage: {sh_pct}")
        if len(self.eps_shrinkage) > 0:
            sh_pct = [f"{s * 100:.1f}%" for s in self.eps_shrinkage]
            lines.append(f"EPS shrinkage: {sh_pct}")
        if self.shrinkage_warnings:
            lines.append("Shrinkage warnings:")
            for w in self.shrinkage_warnings:
                lines.append(f"  {w}")
        if self.structured_warnings:
            lines.append("Estimation warnings:")
            for w in self.structured_warnings:
                lines.append(f"  {w}")
        elif self.warnings:
            lines.append(f"Warnings: {self.warnings}")
        return "\n".join(lines)

    def to_html(
        self,
        path: str,
        params: Any | None = None,
        title: str = "OpenPKPD Estimation Report",
        cov_result: Any | None = None,
        provenance: dict[str, object] | None = None,
    ) -> None:
        """
        Write a self-contained HTML report to *path*.

        Requires a ParameterSet for parameter labels.  If *params* is None,
        a minimal report (no labels) is generated.

        Args:
            path:       Output file path (e.g. ``"results.html"``).
            params:     ParameterSet (optional, for THETA labels).
            title:      Report heading.
            cov_result: Optional CovarianceResult for SE / condition number.
        """
        from openpkpd.output.report import write_html_report

        if params is None:
            from openpkpd.model.parameters import ParameterSet, ThetaSpec

            params = ParameterSet.from_specs(
                [ThetaSpec(init=v) for v in self.theta_final],
                [],
                [],
            )
        write_html_report(
            path, self, params, title=title, cov_result=cov_result, provenance=provenance
        )

    def to_pdf(
        self,
        path: str,
        params: Any | None = None,
        title: str = "OpenPKPD Estimation Report",
        cov_result: Any | None = None,
        provenance: dict[str, object] | None = None,
    ) -> None:
        """Write a PDF report to *path* using the optional GUI dependencies."""
        from openpkpd.output.report import write_pdf_report

        if params is None:
            from openpkpd.model.parameters import ParameterSet, ThetaSpec

            params = ParameterSet.from_specs(
                [ThetaSpec(init=v) for v in self.theta_final],
                [],
                [],
            )
        write_pdf_report(
            path, self, params, title=title, cov_result=cov_result, provenance=provenance
        )

    def compute_shrinkage(
        self,
        params_at_convergence: Any | None = None,
        iwres: np.ndarray | None = None,
    ) -> None:
        """
        Compute ETA and EPS shrinkage with per-ETA breakdown.

        ETA shrinkage for random effect k:
            shrinkage_k = 1 - SD(EBE_ik, i = 1..N) / sqrt(omega_kk)

        EPS shrinkage (if IWRES are available):
            eps_shrinkage = 1 - SD(IWRES)

        A warning is emitted and stored in ``shrinkage_warnings`` for any
        ETA whose shrinkage exceeds 30%, as high ETA shrinkage can inflate
        the apparent precision of parameter estimates and invalidates
        post-hoc ETA-based analyses.

        Args:
            params_at_convergence: Unused; retained for API compatibility.
            iwres: Individual weighted residuals array for EPS shrinkage
                   computation. If None, EPS shrinkage is not updated.
        """
        if not self.post_hoc_etas:
            return

        n_eta = self.omega_final.shape[0]
        eta_matrix = np.array(list(self.post_hoc_etas.values()))  # (N, n_eta)
        shrinkage = np.zeros(n_eta)
        self.shrinkage_warnings = []

        for k in range(n_eta):
            sd_ebe = float(np.std(eta_matrix[:, k], ddof=1)) if eta_matrix.shape[0] > 1 else 0.0
            omega_kk = float(self.omega_final[k, k])
            if omega_kk > 0:
                shrinkage[k] = 1.0 - sd_ebe / math.sqrt(omega_kk)
            else:
                shrinkage[k] = 0.0

            if shrinkage[k] > 0.30:
                msg = (
                    f"ETA{k + 1} shrinkage is {shrinkage[k] * 100:.1f}% (>30%). "
                    f"EBE-based analyses for this parameter may be unreliable."
                )
                self.shrinkage_warnings.append(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                self.add_structured_warning(
                    WarningCode.WARN_005,
                    msg,
                    WarningSeverity.WARNING,
                )

        self.eta_shrinkage = shrinkage

        # EPS shrinkage from IWRES if provided
        if iwres is not None and len(iwres) > 0:
            valid = iwres[np.isfinite(iwres)]
            eps_sh = 1.0 - float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
            self.eps_shrinkage = np.array([eps_sh])

    def compute_deshrinkage_etas(self) -> dict[int, np.ndarray]:
        """
        Return de-shrunken EBEs using the Combes (2013) rescaling correction.

        EBEs from FOCE (and to a lesser extent SAEM) are biased toward zero
        because the posterior mode is shrunk relative to the true individual
        values.  This method rescales each subject's ETA vector so that the
        marginal variance matches omega_kk exactly:

            eta_adj_ik = eta_ik / (1 − shrinkage_k)

        which is equivalent to:

            eta_adj_ik = eta_ik × sqrt(omega_kk) / SD(EBEs_k)

        The relative ranking of subjects is preserved.  Subjects with the
        same sign as the population mean shift in proportion to their original
        EBE magnitude.

        Returns an empty dict when post-hoc ETAs or shrinkage have not been
        computed, or when a random effect is fully shrunk (SD(EBEs) ≈ 0).

        Reference
        ---------
        Combes F-P, Retout S, Frey N, Mentré F (2013).  Prediction of shrinkage
        of individual parameters using the Bayesian information matrix in
        nonlinear mixed-effects models with evaluation in pharmacokinetics.
        Pharm Res 30:2355–2367.  https://doi.org/10.1007/s11095-013-1079-3
        """
        if not self.post_hoc_etas or len(self.eta_shrinkage) == 0:
            return {}

        subjects = list(self.post_hoc_etas.keys())
        eta_matrix = np.array([self.post_hoc_etas[sid] for sid in subjects], dtype=float)
        n_eta = self.omega_final.shape[0]

        for k in range(min(n_eta, eta_matrix.shape[1])):
            shrinkage_k = float(self.eta_shrinkage[k])
            denom = 1.0 - shrinkage_k
            if abs(denom) < 1e-8:
                # Fully shrunk: de-shrinkage is undefined, leave as zero
                continue
            eta_matrix[:, k] /= denom

        return {sid: eta_matrix[i] for i, sid in enumerate(subjects)}


class EstimationMethod(ABC):
    """
    Abstract base class for estimation methods.

    Each method implements estimate() which takes a PopulationModel
    and initial parameters and returns an EstimationResult.
    """

    method_name: str = "BASE"

    @abstractmethod
    def estimate(
        self,
        population_model: Any,  # PopulationModel
        init_params: Any,  # ParameterSet
        **kwargs: Any,
    ) -> EstimationResult:
        """
        Run the estimation algorithm.

        Args:
            population_model: Assembled PopulationModel with dataset and PK callable.
            init_params:       Initial ParameterSet.
            **kwargs:          Method-specific options.

        Returns:
            EstimationResult with final parameter estimates.
        """

    def _make_result(
        self,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        ofv: float,
        converged: bool = False,
        **kwargs: Any,
    ) -> EstimationResult:
        """Helper to construct EstimationResult."""
        return EstimationResult(
            theta_final=theta,
            omega_final=omega,
            sigma_final=sigma,
            ofv=ofv,
            converged=converged,
            method=self.method_name,
            **kwargs,
        )
