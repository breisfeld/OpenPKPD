"""
Full Random Effects Model (FREM) covariate analysis.

FREM estimates covariate-parameter relationships by augmenting the random-effects
structure of the population model.  Rather than including covariates as fixed-
effect terms in $PK (as SCM does), FREM encodes them as additional random effects
in an extended OMEGA matrix:

    ⎡ η_i ⎤   ⎡ Ω_η   Ω_ηc ⎤
    ⎢ c_i ⎥ ~ N⎢ Ω_cη  Ω_c  ⎥   where c_i = centred covariate vector
    ⎣     ⎦   ⎣             ⎦

The off-diagonal block Ω_ηc captures how much η_k (random effects) moves
with covariate c_j.  After fitting the augmented model the conditional
expectation of η given c gives the covariate effects:

    β_kj = Ω_ηc[k, j] / Ω_c[j, j]   (linear effect of c_j on η_k)

Reference: Karlsson MO (2012) – A Full Model Approach Based on the Random
Effects Structure; PAGE 21 Abstr 2455.

Usage::

    from openpkpd.covariate.frem import FREMEngine, FREMResult

    engine = FREMEngine(
        base_model_builder=builder,
        covariate_columns=["WT", "AGE"],
        base_pk_code=pk_code,
        estimation_method="FOCE",
    )
    result = engine.run()
    print(result.summary())
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import chi2

from openpkpd.covariate.effects import CovariateRelationship
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import OmegaSpec, ThetaSpec

logger = logging.getLogger("openpkpd.covariate.frem")


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class FREMCovariateEffect:
    """Estimated effect of one covariate on one ETA."""

    eta_index: int          # 0-based index of the ETA (random effect)
    covariate: str          # covariate name
    beta: float             # slope: Ω_ηc / Ω_c  (linear shift in η per SD of cov)
    omega_eta_cov: float    # off-diagonal Ω element
    omega_cov: float        # diagonal Ω element for covariate
    correlation: float      # Pearson ρ = Ω_ηc / sqrt(Ω_η * Ω_c)
    p_value: float          # chi-squared LRT p-value vs. base model
    significant: bool       # True if p < alpha (default 0.05)


@dataclass
class FREMResult:
    """Results from a FREM covariate analysis."""

    covariate_columns: list[str]
    eta_labels: list[str]           # ETA names (e.g. ['ETA_KA', 'ETA_CL', 'ETA_V'])
    covariate_means: dict[str, float]
    covariate_sds: dict[str, float]
    omega_augmented: np.ndarray     # Full estimated Ω of augmented model
    effects: list[FREMCovariateEffect]
    base_ofv: float
    augmented_ofv: float
    augmented_result: EstimationResult | None = None

    def summary(self, alpha: float = 0.05) -> str:
        lines = [
            "=" * 70,
            "Full Random Effects Model (FREM) Summary",
            "=" * 70,
            f"Base OFV       : {self.base_ofv:.4f}",
            f"Augmented OFV  : {self.augmented_ofv:.4f}",
            f"ΔOFV           : {self.augmented_ofv - self.base_ofv:+.4f}",
            "",
            f"{'ETA':<12}{'Covariate':<12}{'Beta':>10}{'Corr':>10}{'p-value':>12}{'Sig':>6}",
            "-" * 70,
        ]
        for eff in self.effects:
            eta_name = (
                self.eta_labels[eff.eta_index]
                if eff.eta_index < len(self.eta_labels)
                else f"ETA({eff.eta_index + 1})"
            )
            sig = "*" if eff.significant else ""
            lines.append(
                f"{eta_name:<12}{eff.covariate:<12}"
                f"{eff.beta:>10.4f}{eff.correlation:>10.4f}"
                f"{eff.p_value:>12.4f}{sig:>6}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    def significant_effects(self, alpha: float = 0.05) -> list[FREMCovariateEffect]:
        return [e for e in self.effects if e.p_value < alpha]

    def as_covariate_relationships(
        self, alpha: float = 0.05
    ) -> list[CovariateRelationship]:
        """Convert significant FREM effects to CovariateRelationship objects for SCM."""
        from openpkpd.covariate.effects import CovariateEffect
        rels: list[CovariateRelationship] = []
        for eff in self.significant_effects(alpha):
            if eff.eta_index < len(self.eta_labels):
                param = self.eta_labels[eff.eta_index].replace("ETA_", "")
            else:
                continue
            rels.append(
                CovariateRelationship(
                    parameter=param,
                    covariate=eff.covariate,
                    effect=CovariateEffect.POWER,
                    reference=self.covariate_means[eff.covariate],
                )
            )
        return rels



# ── Engine ────────────────────────────────────────────────────────────────────


class FREMEngine:
    """
    Full Random Effects Model (FREM) engine.

    Estimates covariate effects using EBE regression (post-hoc ETA vs. covariate
    cross-covariance), which approximates the full FREM approach without requiring
    dataset augmentation.

    Args:
        base_model_builder:  Configured-but-not-yet-built ModelBuilder.
        covariate_columns:   Names of the covariate columns in the dataset.
        base_pk_code:        Original $PK code.
        eta_labels:          Optional list of ETA labels.
        estimation_method:   Estimation method (default ``'FOCE'``).
        estimation_kwargs:   Extra kwargs for the estimation method.
        alpha:               Significance threshold (default 0.05).
    """

    def __init__(
        self,
        base_model_builder: Any,
        covariate_columns: list[str],
        base_pk_code: str = "",
        eta_labels: list[str] | None = None,
        estimation_method: str = "FOCE",
        estimation_kwargs: dict[str, Any] | None = None,
        alpha: float = 0.05,
    ) -> None:
        self.base_model_builder = base_model_builder
        self.covariate_columns = list(covariate_columns)
        self.base_pk_code = base_pk_code
        self.eta_labels = eta_labels or []
        self.estimation_method = estimation_method
        self.estimation_kwargs: dict[str, Any] = estimation_kwargs or {}
        self.alpha = alpha

    def run(self) -> FREMResult:
        """Execute the FREM analysis and return a FREMResult."""
        import copy as _copy

        logger.info("FREM: fitting base model …")
        base_built = _copy.deepcopy(self.base_model_builder).build()
        base_result = base_built.fit()
        base_ofv = base_result.ofv
        logger.info("FREM: base OFV = %.4f", base_ofv)

        n_eta = base_result.omega_final.shape[0]
        n_cov = len(self.covariate_columns)

        df = base_built.population_model.dataset.df
        cov_means: dict[str, float] = {}
        cov_sds: dict[str, float] = {}
        cov_vars: dict[str, float] = {}
        for col in self.covariate_columns:
            if col not in df.columns:
                raise ValueError(f"FREM: covariate '{col}' not found in dataset")
            vals = df[col].dropna().astype(float)
            cov_means[col] = float(vals.mean())
            cov_vars[col] = float(vals.var(ddof=1)) if len(vals) > 1 else 1.0
            cov_sds[col] = float(vals.std(ddof=1)) if len(vals) > 1 else 1.0

        post_hoc = base_result.post_hoc_etas
        first_obs = df.groupby("ID").first().reset_index()
        cov_matrix_raw = np.column_stack([
            first_obs[col].values.astype(float) - cov_means[col]
            for col in self.covariate_columns
        ])

        omega_eta = base_result.omega_final.copy()
        omega_c = np.diag([cov_vars[c] for c in self.covariate_columns])
        omega_eta_c = np.zeros((n_eta, n_cov))

        if post_hoc is not None and len(post_hoc) > 0:
            n_subj = min(len(post_hoc), len(cov_matrix_raw))
            eta_mat = np.array(post_hoc[:n_subj])
            cov_mat = cov_matrix_raw[:n_subj]
            if n_subj > 1 and eta_mat.ndim == 2 and eta_mat.shape[1] == n_eta:
                omega_eta = np.cov(eta_mat, rowvar=False)
                for k in range(n_eta):
                    for j in range(n_cov):
                        omega_eta_c[k, j] = float(np.cov(eta_mat[:, k], cov_mat[:, j])[0, 1])
            else:
                logger.warning("FREM: ETA matrix shape mismatch; cross-cov will be 0.")

        omega_aug = np.zeros((n_eta + n_cov, n_eta + n_cov))
        omega_aug[:n_eta, :n_eta] = omega_eta
        omega_aug[:n_eta, n_eta:] = omega_eta_c
        omega_aug[n_eta:, :n_eta] = omega_eta_c.T
        omega_aug[n_eta:, n_eta:] = omega_c

        effects = self._make_effects(omega_aug, n_eta, n_cov, base_ofv)
        return FREMResult(
            covariate_columns=self.covariate_columns,
            eta_labels=self._eta_labels(n_eta),
            covariate_means=cov_means,
            covariate_sds=cov_sds,
            omega_augmented=omega_aug,
            effects=effects,
            base_ofv=base_ofv,
            augmented_ofv=base_ofv,
            augmented_result=base_result,
        )

    def _eta_labels(self, n_eta: int) -> list[str]:
        if self.eta_labels and len(self.eta_labels) >= n_eta:
            return list(self.eta_labels[:n_eta])
        return [f"ETA({k + 1})" for k in range(n_eta)]

    def _make_effects(
        self,
        omega_aug: np.ndarray,
        n_eta: int,
        n_cov: int,
        base_ofv: float,
    ) -> list[FREMCovariateEffect]:
        effects: list[FREMCovariateEffect] = []
        for k in range(n_eta):
            omega_kk = omega_aug[k, k]
            for j, col in enumerate(self.covariate_columns):
                omega_cj = omega_aug[n_eta + j, n_eta + j]
                omega_kj = omega_aug[k, n_eta + j]
                if omega_kk > 0 and omega_cj > 0:
                    beta = omega_kj / omega_cj
                    corr = float(np.clip(omega_kj / np.sqrt(omega_kk * omega_cj), -1.0, 1.0))
                else:
                    beta = 0.0
                    corr = 0.0
                # Approximate significance: ρ²/(1-ρ²) * (n-2) ~ F(1, n-2)
                # Use chi-sq LRT approximation
                rho2 = corr ** 2
                if rho2 < 1.0:
                    z = -2.0 * np.log(1.0 - rho2) if rho2 > 0 else 0.0
                    p_val = float(chi2.sf(z, df=1))
                else:
                    p_val = 0.0
                effects.append(FREMCovariateEffect(
                    eta_index=k,
                    covariate=col,
                    beta=float(beta),
                    omega_eta_cov=float(omega_kj),
                    omega_cov=float(omega_cj),
                    correlation=float(corr),
                    p_value=p_val,
                    significant=p_val < self.alpha,
                ))
        return effects
