"""
Stepwise Covariate Modeling (SCM).

Automated forward-backward search for covariate-parameter relationships.

The SCM algorithm:
  1. **Forward selection** — start from the base model and, at each step, try
     adding every remaining candidate relationship.  Accept the one that gives
     the largest ΔOFV improvement *and* meets the forward p-value criterion
     (chi-squared LRT with 1 df).
  2. **Backward elimination** — starting from the final forward model, try
     removing each accepted relationship one at a time.  Remove a relationship
     only if its removal does *not* cause a significant OFV worsening (i.e.
     ΔOFV < backward threshold).

The engine uses the ModelBuilder fluent API: the caller supplies the
*original* (base) ModelBuilder instance so that it can rebuild the model
with modified $PK code and additional THETA parameters for each candidate.

Usage::

    from openpkpd.covariate.effects import CovariateRelationship, CovariateEffect
    from openpkpd.covariate.scm import SCMEngine

    engine = SCMEngine(
        base_model_builder=builder,      # ModelBuilder (not yet built)
        base_pk_code=pk_code_str,        # original $PK code string
        candidates=[
            CovariateRelationship('CL', 'WT',  CovariateEffect.POWER,  reference=70.0),
            CovariateRelationship('V',  'WT',  CovariateEffect.POWER,  reference=70.0),
            CovariateRelationship('CL', 'AGE', CovariateEffect.LINEAR, reference=40.0),
        ],
        forward_pvalue=0.05,
        backward_pvalue=0.001,
    )
    result = engine.run()
    print(result.summary())
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from scipy.stats import chi2

from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import ThetaSpec

logger = logging.getLogger("openpkpd.covariate.scm")


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class SCMStep:
    """Records one step in the SCM procedure."""

    step_type: str  # 'forward' or 'backward'
    relationship: CovariateRelationship
    ofv_base: float
    ofv_new: float
    delta_ofv: float  # ofv_new - ofv_base  (negative = improvement)
    df: int  # degrees of freedom (1 for scalar covariate effect)
    p_value: float
    accepted: bool

    def __str__(self) -> str:
        direction = "+" if self.step_type == "forward" else "-"
        status = "ACCEPTED" if self.accepted else "rejected"
        return (
            f"[{self.step_type.upper()}] {direction}{self.relationship.parameter}"
            f"~{self.relationship.covariate}({self.relationship.effect.value}): "
            f"ΔOFV={self.delta_ofv:+.3f}  p={self.p_value:.4f}  {status}"
        )


@dataclass
class SCMResult:
    """Full result of the SCM procedure."""

    base_ofv: float
    final_ofv: float
    accepted_relationships: list[CovariateRelationship]
    steps: list[SCMStep]
    model_history: list[EstimationResult]

    def summary(self) -> str:
        """Return a text summary of the SCM procedure."""
        lines: list[str] = [
            "=" * 60,
            "Stepwise Covariate Modeling (SCM) Summary",
            "=" * 60,
            f"Base OFV  : {self.base_ofv:.4f}",
            f"Final OFV : {self.final_ofv:.4f}",
            f"ΔOFV      : {self.final_ofv - self.base_ofv:+.4f}",
            "",
            "Steps:",
        ]
        for i, step in enumerate(self.steps, start=1):
            lines.append(f"  {i:2d}. {step}")

        lines.append("")
        if self.accepted_relationships:
            lines.append("Accepted relationships:")
            for rel in self.accepted_relationships:
                lines.append(
                    f"  {rel.parameter} ~ {rel.covariate} [{rel.effect.value}, ref={rel.reference}]"
                )
        else:
            lines.append("No covariate relationships were accepted.")

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Engine ─────────────────────────────────────────────────────────────────────


class SCMEngine:
    """
    Stepwise covariate model building engine.

    Args:
        base_model_builder:   A *configured but not yet built* ModelBuilder
                              instance representing the base model.  SCMEngine
                              will call ``.build()`` and ``.fit()`` on copies of
                              this builder with modified $PK code and extra
                              THETA parameters.
        base_pk_code:         The original $PK code string (before any covariate
                              additions).  Covariate lines will be appended.
        candidates:           List of CovariateRelationship candidates to test.
        forward_pvalue:       Chi-squared p-value threshold for forward selection
                              (default 0.05; conventional NONMEM default is 0.05).
        backward_pvalue:      Chi-squared p-value threshold for backward
                              elimination (default 0.001; more conservative to
                              avoid removal of informative covariates).
        max_forward_steps:    Safety cap on forward selection iterations.
        n_jobs:               Number of parallel workers for candidate evaluation
                              in the forward-selection inner loop.  ``1`` =
                              sequential (default).  ``-1`` = use all CPUs.
                              Uses ``ThreadPoolExecutor`` internally so model
                              builders do not need to be picklable.
        estimation_method:    Estimation method forwarded to the re-built models
                              (default ``'FOCE'``).
        estimation_kwargs:    Additional keyword arguments forwarded to
                              ``ModelBuilder.estimation()``.
    """

    def __init__(
        self,
        base_model_builder: Any,
        base_pk_code: str,
        candidates: list[CovariateRelationship],
        forward_pvalue: float = 0.05,
        backward_pvalue: float = 0.001,
        max_forward_steps: int = 20,
        n_jobs: int = 1,
        estimation_method: str = "FOCE",
        estimation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_model_builder = base_model_builder
        self.base_pk_code = base_pk_code
        self.candidates = list(candidates)
        self.forward_pvalue = forward_pvalue
        self.backward_pvalue = backward_pvalue
        self.max_forward_steps = max_forward_steps
        self.n_jobs = n_jobs
        self.estimation_method = estimation_method
        self.estimation_kwargs: dict[str, Any] = estimation_kwargs or {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> SCMResult:
        """
        Execute the full forward-backward SCM procedure.

        Returns:
            SCMResult containing all steps, accepted relationships and OFV history.
        """
        logger.info(
            "Starting SCM with %d candidates (forward p=%.3f, backward p=%.4f)",
            len(self.candidates),
            self.forward_pvalue,
            self.backward_pvalue,
        )

        # Fit base model to get starting OFV
        base_result = self._fit_current([], [])
        base_ofv = base_result.ofv
        logger.info("Base OFV = %.4f", base_ofv)

        steps: list[SCMStep] = []
        model_history: list[EstimationResult] = [base_result]
        accepted: list[CovariateRelationship] = []
        remaining = list(self.candidates)
        current_result = base_result

        # ── Forward selection ─────────────────────────────────────────────────
        for fwd_iter in range(self.max_forward_steps):
            if not remaining:
                logger.info("No remaining candidates; forward selection complete.")
                break

            step = self._forward_step(current_result, remaining, accepted)
            if step is None:
                logger.info(
                    "Forward step %d: no candidate met criterion — stopping forward.",
                    fwd_iter + 1,
                )
                break

            steps.append(step)
            logger.info("Forward step %d: %s", fwd_iter + 1, step)

            if step.accepted:
                accepted.append(step.relationship)
                remaining.remove(step.relationship)
                # Re-fit with the newly accepted relationship to get current_result
                current_result = self._fit_current(
                    [r.pk_code_suffix for r in _covariate_suffixes(accepted)],
                    accepted,
                )
                model_history.append(current_result)
            else:
                break

        # ── Backward elimination ──────────────────────────────────────────────
        if accepted:
            changed = True
            while changed:
                changed = False
                step = self._backward_step(current_result, accepted)
                if step is not None:
                    steps.append(step)
                    logger.info("Backward step: %s", step)
                    if step.accepted:
                        accepted.remove(step.relationship)
                        current_result = self._fit_current(
                            [r.pk_code_suffix for r in _covariate_suffixes(accepted)],
                            accepted,
                        )
                        model_history.append(current_result)
                        changed = True

        final_ofv = current_result.ofv
        logger.info(
            "SCM complete. Final OFV=%.4f, %d relationships accepted.",
            final_ofv,
            len(accepted),
        )

        return SCMResult(
            base_ofv=base_ofv,
            final_ofv=final_ofv,
            accepted_relationships=accepted,
            steps=steps,
            model_history=model_history,
        )

    # ── Forward step ───────────────────────────────────────────────────────────

    def _forward_step(
        self,
        base_result: EstimationResult,
        remaining: list[CovariateRelationship],
        accepted: list[CovariateRelationship],
    ) -> SCMStep | None:
        """
        Test each remaining candidate relationship, optionally in parallel.

        When ``self.n_jobs != 1``, all candidates are evaluated concurrently
        using ``ThreadPoolExecutor`` (n_jobs=-1 → all available CPUs).

        Returns the best (most improved OFV) step if it meets the forward
        p-value criterion, otherwise returns None.
        """
        n_workers = os.cpu_count() or 1 if self.n_jobs < 1 else self.n_jobs

        def _run(
            candidate: CovariateRelationship,
        ) -> tuple[CovariateRelationship, Any, Exception | None]:
            try:
                return candidate, self._fit_with_addition(accepted, candidate), None
            except Exception as exc:
                return candidate, None, exc

        if n_workers == 1:
            raw_results = [_run(c) for c in remaining]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_run, c): c for c in remaining}
                raw_results = [f.result() for f in as_completed(futures)]

        best_step: SCMStep | None = None
        best_delta: float = 0.0  # we want the most negative delta (largest drop)

        for candidate, trial_result, exc in raw_results:
            if exc is not None:
                logger.warning(
                    "Failed to fit candidate %s~%s: %s",
                    candidate.parameter,
                    candidate.covariate,
                    exc,
                )
                continue

            delta = trial_result.ofv - base_result.ofv  # negative = improvement
            df = 1  # one extra THETA per continuous covariate relationship
            # Chi-squared p-value: large improvement → small p
            p_value = _lrt_pvalue(-delta, df) if delta < 0 else 1.0
            accepted_flag = p_value < self.forward_pvalue

            step = SCMStep(
                step_type="forward",
                relationship=candidate,
                ofv_base=base_result.ofv,
                ofv_new=trial_result.ofv,
                delta_ofv=delta,
                df=df,
                p_value=p_value,
                accepted=accepted_flag,
            )

            # Track best improvement (whether or not it meets threshold)
            if delta < best_delta:
                best_delta = delta
                best_step = step

        if best_step is None:
            return None

        # Only propagate if the best step actually meets the criterion
        best_step.accepted = best_step.p_value < self.forward_pvalue
        return best_step

    # ── Backward step ──────────────────────────────────────────────────────────

    def _backward_step(
        self,
        current_result: EstimationResult,
        accepted: list[CovariateRelationship],
    ) -> SCMStep | None:
        """
        Test removing each accepted relationship.

        Returns the step for the relationship whose removal causes the *least*
        significant worsening, if that worsening is below the backward threshold.
        Returns None if no relationship can be safely removed.
        """
        best_step: SCMStep | None = None
        best_delta: float = float("inf")  # smallest (least harmful) increase

        for candidate in accepted:
            remaining_accepted = [r for r in accepted if r is not candidate]
            try:
                trial_result = self._fit_current(
                    [r.pk_code_suffix for r in _covariate_suffixes(remaining_accepted)],
                    remaining_accepted,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to fit without %s~%s: %s",
                    candidate.parameter,
                    candidate.covariate,
                    exc,
                )
                continue

            delta = trial_result.ofv - current_result.ofv  # positive = worsening
            df = 1
            # p_value: small p means significant worsening (don't remove)
            p_value = _lrt_pvalue(delta, df) if delta > 0 else 0.0
            # Remove if removal does NOT significantly worsen fit
            accepted_flag = p_value > self.backward_pvalue

            step = SCMStep(
                step_type="backward",
                relationship=candidate,
                ofv_base=current_result.ofv,
                ofv_new=trial_result.ofv,
                delta_ofv=delta,
                df=df,
                p_value=p_value,
                accepted=accepted_flag,
            )

            # Prefer the removal that worsens OFV the least
            if accepted_flag and delta < best_delta:
                best_delta = delta
                best_step = step

        return best_step

    # ── Model building helpers ─────────────────────────────────────────────────

    def _fit_with_addition(
        self,
        already_accepted: list[CovariateRelationship],
        new_candidate: CovariateRelationship,
    ) -> EstimationResult:
        """
        Fit a model that includes all already-accepted relationships plus the
        new candidate.
        """
        trial_accepted = already_accepted + [new_candidate]
        return self._fit_current(
            [r.pk_code_suffix for r in _covariate_suffixes(trial_accepted)],
            trial_accepted,
        )

    def _fit_current(
        self,
        _pk_suffixes: list[str],  # retained for API symmetry; generated internally
        accepted_rels: list[CovariateRelationship],
    ) -> EstimationResult:
        """
        Build and fit a model that includes the given accepted relationships.

        New THETA parameters are appended (one per continuous relationship).
        The $PK code has covariate lines appended after the base code.
        """
        import copy as _copy

        builder = _copy.deepcopy(self.base_model_builder)

        # Build augmented $PK code
        pk_lines = [self.base_pk_code.rstrip()]
        n_base_theta = len(builder._theta_specs)

        for i, rel in enumerate(accepted_rels):
            theta_idx = n_base_theta + i + 1  # 1-based NM-TRAN index
            pk_lines.append("")
            pk_lines.append(rel.generate_pk_code(theta_idx))

        augmented_pk = "\n".join(pk_lines)
        builder.pk(augmented_pk)

        # Append THETA specs for each new covariate relationship
        for rel in accepted_rels:
            init, lower, upper = _default_covariate_theta(rel)
            builder._theta_specs = list(builder._theta_specs) + [
                ThetaSpec(
                    init=init,
                    lower=lower,
                    upper=upper,
                    label=f"theta_{rel.covariate.lower()}_{rel.parameter.lower()}",
                )
            ]

        # Set estimation method
        builder.estimation(method=self.estimation_method, **self.estimation_kwargs)

        built = builder.build()
        return built.fit()

    def _add_relationship(
        self,
        base_model: Any,
        relationship: CovariateRelationship,
    ) -> tuple[Any, EstimationResult]:
        """
        Create a new model with the given relationship added to $PK code.

        Returns (new_built_model, estimation_result).

        .. note::
            This method exists for API compatibility.  The primary workflow
            uses :meth:`_fit_current` / :meth:`_fit_with_addition` directly.
        """
        result = self._fit_with_addition([], relationship)
        return base_model, result

    def _remove_relationship(
        self,
        current_model: Any,
        relationship: CovariateRelationship,
    ) -> tuple[Any, EstimationResult]:
        """
        Remove a relationship from $PK code and re-fit.

        Returns (new_built_model, estimation_result).
        """
        result = self._fit_current([], [])
        return current_model, result


# ── Helpers ───────────────────────────────────────────────────────────────────


class _CovSuffix:
    """Internal wrapper to satisfy the pk_code_suffix protocol."""

    def __init__(self, rel: CovariateRelationship, theta_index: int) -> None:
        self.rel = rel
        self.pk_code_suffix = rel.generate_pk_code(theta_index)


def _covariate_suffixes(
    accepted: list[CovariateRelationship],
    n_base_theta: int = 0,
) -> list[_CovSuffix]:
    """Generate pk_code_suffix wrappers for all accepted relationships."""
    return [_CovSuffix(rel, n_base_theta + i + 1) for i, rel in enumerate(accepted)]


def _lrt_pvalue(delta_ofv: float, df: int) -> float:
    """
    Compute the likelihood-ratio test p-value.

    The test statistic is ΔOFV (which is -2 * Δlog-likelihood), distributed
    as chi-squared(df) under H0.

    Args:
        delta_ofv:  Improvement in OFV (positive = improvement).
        df:         Degrees of freedom.

    Returns:
        p-value (probability of observing this or a larger improvement by chance).
    """
    if delta_ofv <= 0:
        return 1.0
    return float(chi2.sf(delta_ofv, df))


def _default_covariate_theta(rel: CovariateRelationship) -> tuple[float, float, float]:
    """
    Return (init, lower, upper) for a new covariate THETA.

    Convention:
      - Power / exponential effects: init=0 (no effect at null), unbounded.
      - Linear effects: init=0, unbounded.
      - Categorical: init=1, lower=0.001 (multiplicative, must be positive).
    """
    if rel.effect == CovariateEffect.CATEGORICAL:
        return (1.0, 0.001, float("inf"))
    else:
        return (0.0, -float("inf"), float("inf"))
