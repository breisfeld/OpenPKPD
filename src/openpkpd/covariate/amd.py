"""
Automated Model Development (AMD) pipeline.

AMD automates the structural model selection and covariate screening steps that
are normally performed manually:

1. **Structural model search** — fit a set of candidate structural models
   (e.g. ADVAN1–4), rank by AIC/BIC, and select the best.
2. **Absorption model selection** (oral data only) — test 0-order vs. 1-order
   absorption, transit compartment models, etc.
3. **Covariate screening** — run SCM or FREM on the best structural model.
4. Optionally test IOV, error models, etc.

Usage::

    from openpkpd.covariate.amd import AMDPipeline, AMDResult

    pipeline = AMDPipeline(
        dataset=ds,
        administration="oral",
        covariate_columns=["WT", "AGE", "SEX"],
        structural_candidates=["advan2", "advan4"],
        covariate_method="scm",
    )
    result = pipeline.run()
    print(result.summary())
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship
from openpkpd.covariate.scm import SCMEngine, SCMResult
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import OmegaSpec, SigmaSpec, ThetaSpec

logger = logging.getLogger("openpkpd.covariate.amd")

# ── Structural model templates ────────────────────────────────────────────────

_ORAL_STRUCTURAL_TEMPLATES: dict[str, dict[str, Any]] = {
    "advan2": {
        "advan": 2, "trans": 2,
        "pk": (
            "KA = THETA(1) * EXP(ETA(1))\n"
            "CL = THETA(2) * EXP(ETA(2))\n"
            "V  = THETA(3) * EXP(ETA(3))"
        ),
        "theta": [(0.01, 1.0, 30), (0.001, 0.1, 10), (0.1, 20.0, 500)],
        "omega": [0.5, 0.3, 0.3],
        "n_eta": 3,
        "eta_labels": ["ETA_KA", "ETA_CL", "ETA_V"],
        "label": "1-cmt oral (ADVAN2)",
    },
    "advan4": {
        "advan": 4, "trans": 4,
        "pk": (
            "KA = THETA(1) * EXP(ETA(1))\n"
            "CL = THETA(2) * EXP(ETA(2))\n"
            "V2 = THETA(3) * EXP(ETA(3))\n"
            "Q  = THETA(4)\n"
            "V3 = THETA(5)"
        ),
        "theta": [(0.01, 1.0, 30), (0.001, 0.1, 10), (0.1, 20.0, 500), (0.01, 0.5, 50), (0.1, 50.0, 1000)],
        "omega": [0.5, 0.3, 0.3],
        "n_eta": 3,
        "eta_labels": ["ETA_KA", "ETA_CL", "ETA_V2"],
        "label": "2-cmt oral (ADVAN4)",
    },
}

_IV_STRUCTURAL_TEMPLATES: dict[str, dict[str, Any]] = {
    "advan1": {
        "advan": 1, "trans": 2,
        "pk": (
            "CL = THETA(1) * EXP(ETA(1))\n"
            "V  = THETA(2) * EXP(ETA(2))"
        ),
        "theta": [(0.001, 0.1, 10), (0.1, 20.0, 500)],
        "omega": [0.3, 0.3],
        "n_eta": 2,
        "eta_labels": ["ETA_CL", "ETA_V"],
        "label": "1-cmt IV (ADVAN1)",
    },
    "advan3": {
        "advan": 3, "trans": 4,
        "pk": (
            "CL = THETA(1) * EXP(ETA(1))\n"
            "V1 = THETA(2) * EXP(ETA(2))\n"
            "Q  = THETA(3)\n"
            "V2 = THETA(4)"
        ),
        "theta": [(0.001, 0.1, 10), (0.1, 20.0, 500), (0.01, 0.5, 50), (0.1, 50.0, 1000)],
        "omega": [0.3, 0.3],
        "n_eta": 2,
        "eta_labels": ["ETA_CL", "ETA_V1"],
        "label": "2-cmt IV (ADVAN3)",
    },
}

# ── Result containers ─────────────────────────────────────────────────────────


@dataclass
class AMDStructuralResult:
    """Result for a single structural model candidate."""
    label: str
    advan: int
    ofv: float
    aic: float
    bic: float
    n_params: int
    estimation_result: EstimationResult


@dataclass
class AMDResult:
    """Full result of the AMD pipeline."""
    administration: str
    structural_results: list[AMDStructuralResult]
    best_structural: AMDStructuralResult
    scm_result: SCMResult | None = None
    frem_result: Any | None = None   # FREMResult | None
    covariate_method: str = "scm"

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "Automated Model Development (AMD) Summary",
            "=" * 70,
            f"Administration: {self.administration}",
            "",
            "Structural model selection:",
            f"  {'Model':<25}{'OFV':>10}{'AIC':>10}{'BIC':>10}{'n_p':>5}",
            "  " + "-" * 52,
        ]
        for r in self.structural_results:
            flag = " *" if r.label == self.best_structural.label else ""
            lines.append(
                f"  {r.label:<25}{r.ofv:>10.2f}{r.aic:>10.2f}{r.bic:>10.2f}{r.n_params:>5}{flag}"
            )
        lines += ["", f"Best structural model: {self.best_structural.label}"]
        if self.scm_result is not None:
            lines += ["", self.scm_result.summary()]
        if self.frem_result is not None:
            lines += ["", self.frem_result.summary()]
        lines += ["", "=" * 70]
        return "\n".join(lines)


# ── Pipeline ──────────────────────────────────────────────────────────────────


class AMDPipeline:
    """
    Automated Model Development pipeline.

    Args:
        dataset:               NONMEMDataset to analyse.
        administration:        ``'oral'`` or ``'iv'``.
        covariate_columns:     Covariate column names to screen.
        structural_candidates: List of structural model keys to test.
            Oral candidates: ``'advan2'``, ``'advan4'``.
            IV candidates:   ``'advan1'``, ``'advan3'``.
        covariate_method:      ``'scm'`` (default), ``'frem'``, or ``'none'``.
        error_model:           ``'proportional'`` (default) or ``'additive'``.
        estimation_method:     Estimation method (default ``'FOCE'``).
        estimation_kwargs:     Extra kwargs for estimation.
        forward_pvalue:        Forward selection p-value for SCM.
        backward_pvalue:       Backward elimination p-value for SCM.
        n_subjects_bic:        Override for BIC subject count (default = dataset n_subjects).
    """

    def __init__(
        self,
        dataset: Any,
        administration: str = "oral",
        covariate_columns: list[str] | None = None,
        structural_candidates: list[str] | None = None,
        covariate_method: str = "scm",
        error_model: str = "proportional",
        estimation_method: str = "FOCE",
        estimation_kwargs: dict[str, Any] | None = None,
        forward_pvalue: float = 0.05,
        backward_pvalue: float = 0.001,
        n_subjects_bic: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.administration = administration.lower()
        self.covariate_columns = list(covariate_columns or [])
        self.covariate_method = covariate_method.lower()
        self.error_model = error_model.lower()
        self.estimation_method = estimation_method
        self.estimation_kwargs: dict[str, Any] = estimation_kwargs or {}
        self.forward_pvalue = forward_pvalue
        self.backward_pvalue = backward_pvalue
        self.n_subjects_bic = n_subjects_bic

        templates = (
            _ORAL_STRUCTURAL_TEMPLATES if self.administration == "oral"
            else _IV_STRUCTURAL_TEMPLATES
        )
        if structural_candidates:
            self.candidates = [k for k in structural_candidates if k in templates]
        else:
            self.candidates = list(templates.keys())

        self._templates = templates

    def run(self) -> AMDResult:
        """Execute the full AMD pipeline."""
        logger.info("AMD: starting structural model search (%s)", self.administration)
        structural_results = self._run_structural_search()
        if not structural_results:
            raise RuntimeError("AMD: no structural model converged successfully.")

        best = min(structural_results, key=lambda r: r.aic)
        logger.info("AMD: best structural model = %s (AIC=%.2f)", best.label, best.aic)

        scm_result = None
        frem_result = None

        if self.covariate_columns and self.covariate_method != "none":
            scm_result, frem_result = self._run_covariate_screening(best)

        return AMDResult(
            administration=self.administration,
            structural_results=structural_results,
            best_structural=best,
            scm_result=scm_result,
            frem_result=frem_result,
            covariate_method=self.covariate_method,
        )

    def _run_structural_search(self) -> list[AMDStructuralResult]:
        from openpkpd.api.model_builder import ModelBuilder

        results: list[AMDStructuralResult] = []
        for key in self.candidates:
            tmpl = self._templates[key]
            logger.info("AMD: fitting %s …", tmpl["label"])
            try:
                error_code = self._error_code()
                sigma_init = [0.1] if self.error_model == "proportional" else [1.0]
                theta_specs = tmpl["theta"]
                if self.error_model == "combined":
                    theta_specs = list(theta_specs) + [(0.01, 0.1, 2)]
                builder = (
                    ModelBuilder()
                    .dataset(self.dataset)
                    .subroutines(advan=tmpl["advan"], trans=tmpl["trans"])
                    .pk(tmpl["pk"])
                    .error(error_code)
                    .theta(theta_specs)
                    .omega(tmpl["omega"])
                    .sigma(sigma_init)
                    .estimation(method=self.estimation_method, **self.estimation_kwargs)
                )
                result = builder.build().fit()
                n_theta = len(tmpl["theta"]) + (1 if self.error_model == "combined" else 0)
                n_omega = tmpl["n_eta"]
                n_sigma = 1
                n_p = n_theta + n_omega + n_sigma
                n_obs = int(self.dataset.df.shape[0] - self.dataset.df.shape[0] * 0)
                n_subj = self.n_subjects_bic or len(self.dataset.df["ID"].unique())
                aic = result.ofv + 2 * n_p
                bic = result.ofv + n_p * np.log(n_subj)
                results.append(AMDStructuralResult(
                    label=tmpl["label"],
                    advan=tmpl["advan"],
                    ofv=result.ofv,
                    aic=aic,
                    bic=bic,
                    n_params=n_p,
                    estimation_result=result,
                ))
                logger.info("AMD: %s OFV=%.4f AIC=%.2f", tmpl["label"], result.ofv, aic)
            except Exception as exc:
                logger.warning("AMD: %s failed: %s", tmpl["label"], exc)
        return results

    def _error_code(self) -> str:
        if self.error_model == "additive":
            return "IPRED = F\nW = THETA(LAST)\nY = IPRED + W * EPS(1)"
        elif self.error_model == "combined":
            return "IPRED = F\nW = SQRT(THETA(LAST-1)**2*IPRED**2 + THETA(LAST)**2)\nY = IPRED + W * EPS(1)"
        else:  # proportional
            return "IPRED = F\nW = IPRED * THETA(LAST)\nY = IPRED + W * EPS(1)"

    def _run_covariate_screening(
        self, best: AMDStructuralResult
    ) -> tuple[SCMResult | None, Any | None]:
        tmpl = next(
            t for t in self._templates.values() if t["label"] == best.label
        )
        pk_code = tmpl["pk"]

        if self.covariate_method in ("frem",):
            from openpkpd.covariate.frem import FREMEngine
            from openpkpd.api.model_builder import ModelBuilder
            error_code = self._error_code()
            sigma_init = [0.1]
            builder = (
                ModelBuilder()
                .dataset(self.dataset)
                .subroutines(advan=tmpl["advan"], trans=tmpl["trans"])
                .pk(pk_code)
                .error(error_code)
                .theta(tmpl["theta"])
                .omega(tmpl["omega"])
                .sigma(sigma_init)
                .estimation(method=self.estimation_method, **self.estimation_kwargs)
                .covariates(self.covariate_columns)
            )
            frem_engine = FREMEngine(
                base_model_builder=builder,
                covariate_columns=self.covariate_columns,
                eta_labels=tmpl.get("eta_labels"),
                alpha=self.forward_pvalue,
            )
            frem_result = frem_engine.run()
            return None, frem_result
        else:
            # Default: SCM
            from openpkpd.api.model_builder import ModelBuilder
            from openpkpd.covariate.effects import CovariateEffect

            n_theta = len(tmpl["theta"])
            # Auto-generate power effect candidates for all numeric covariates
            candidates = []
            df = self.dataset.df
            for col in self.covariate_columns:
                if col in df.columns:
                    for param_idx, label in enumerate(tmpl.get("eta_labels", [])):
                        param = label.replace("ETA_", "")
                        ref = float(df[col].median())
                        candidates.append(CovariateRelationship(param, col, CovariateEffect.POWER, reference=ref))

            if not candidates:
                return None, None

            error_code = self._error_code()
            sigma_init = [0.1]
            builder = (
                ModelBuilder()
                .dataset(self.dataset)
                .subroutines(advan=tmpl["advan"], trans=tmpl["trans"])
                .pk(pk_code)
                .error(error_code)
                .theta(tmpl["theta"])
                .omega(tmpl["omega"])
                .sigma(sigma_init)
                .estimation(method=self.estimation_method, **self.estimation_kwargs)
                .covariates(self.covariate_columns)
            )
            scm_engine = SCMEngine(
                base_model_builder=builder,
                base_pk_code=pk_code,
                candidates=candidates,
                forward_pvalue=self.forward_pvalue,
                backward_pvalue=self.backward_pvalue,
                estimation_method=self.estimation_method,
                estimation_kwargs=self.estimation_kwargs,
            )
            scm_result = scm_engine.run()
            return scm_result, None
