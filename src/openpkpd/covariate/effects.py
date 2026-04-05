"""
Standard covariate-parameter effect parameterizations.

These represent the relationship between a covariate (e.g., weight, age, sex)
and a PK/PD parameter (e.g., CL, V).

Supported effect types:
  - LINEAR:       P = theta_P * (1 + theta_cov * (COV - ref))
  - POWER:        P = theta_P * (COV / ref) ** theta_cov
  - EXPONENTIAL:  P = theta_P * exp(theta_cov * (COV - ref))
  - CATEGORICAL:  P = theta_P * theta_cat[category]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np


class CovariateEffect(Enum):
    """Supported covariate-parameter relationship types."""

    LINEAR = "linear"
    POWER = "power"
    EXPONENTIAL = "exp"
    CATEGORICAL = "categorical"


def _validate_integer_categories(categories: list) -> None:
    """Validate that categories are integers forming a contiguous 1..K range.

    NONMEM categorical covariates must be encoded as integers 1, 2, ..., K
    because NM-TRAN code uses ``IF (COV == 1) ...`` etc.
    """
    if not categories:
        raise ValueError("categories must be a non-empty list")
    for i, cat in enumerate(categories):
        if not isinstance(cat, int):
            raise ValueError(
                f"categories[{i}]={cat!r} is not an integer. "
                "NONMEM categorical covariates must be integer-coded (1, 2, ..., K). "
                "String categories like 'MALE'/'FEMALE' must be re-encoded before use."
            )
    sorted_cats = sorted(categories)
    expected = list(range(1, len(categories) + 1))
    if sorted_cats != expected:
        raise ValueError(
            f"categories {sorted_cats} do not form a contiguous range starting at 1. "
            f"Expected {expected}. "
            "NONMEM integer-coding requires categories 1, 2, ..., K with no gaps and no 0-based indexing."
        )


@dataclass
class CovariateRelationship:
    """
    Defines one covariate-parameter relationship.

    Attributes:
        parameter:   PK/PD parameter name (e.g., 'CL', 'V', 'KA').
        covariate:   Covariate column name in the dataset (e.g., 'WT', 'AGE', 'SEX').
        effect:      The functional form linking covariate to parameter.
        reference:   Reference value for the covariate (used for centering).
                     For categorical covariates this is the reference category code.
        categories:  For CATEGORICAL effects: ordered list of integer category codes.
                     The first category is the reference (multiplier = 1.0).
    """

    parameter: str
    covariate: str
    effect: CovariateEffect
    reference: float = 70.0
    categories: list[int] | None = None

    def __post_init__(self) -> None:
        if self.effect == CovariateEffect.CATEGORICAL and self.categories:
            _validate_integer_categories(self.categories)

    def apply(
        self,
        base_value: float,
        cov_value: float,
        theta_cov: float,
    ) -> float:
        """
        Apply this relationship to get the adjusted parameter value.

        For continuous effects (LINEAR, POWER, EXPONENTIAL) the covariate
        multiplier is applied to *base_value*.  For CATEGORICAL effects the
        caller is responsible for supplying the appropriate per-category theta.

        Args:
            base_value:  The un-adjusted parameter value (e.g., THETA(2)).
            cov_value:   The individual's covariate value.
            theta_cov:   The covariate effect coefficient estimated by the model.

        Returns:
            Adjusted parameter value.
        """
        if self.effect == CovariateEffect.POWER:
            if self.reference <= 0:
                raise ValueError(f"Power covariate reference must be > 0, got {self.reference}")
            ratio = cov_value / self.reference
            # Guard against non-positive ratios (e.g., WT=0 would blow up)
            if ratio <= 0:
                warnings.warn(
                    f"Power covariate value {cov_value!r} produces a non-positive ratio "
                    f"({ratio:.4g}); clamping to 1e-10. Check input data for data quality issues "
                    f"(e.g., negative body weight or zero clearance).",
                    UserWarning,
                    stacklevel=2,
                )
            ratio = max(ratio, 1e-10)
            return base_value * (ratio**theta_cov)

        elif self.effect == CovariateEffect.LINEAR:
            return base_value * (1.0 + theta_cov * (cov_value - self.reference))

        elif self.effect == CovariateEffect.EXPONENTIAL:
            return base_value * np.exp(theta_cov * (cov_value - self.reference))

        else:
            # CATEGORICAL: handled externally; return base unchanged as fallback
            warnings.warn(
                "CovariateEffect.apply() called on CATEGORICAL effect; use apply_categorical() instead",
                UserWarning,
                stacklevel=2,
            )
            return base_value

    def apply_categorical(
        self,
        base_value: float,
        category: int,
        theta_per_category: dict[int, float],
    ) -> float:
        """
        Apply a categorical effect using per-category theta multipliers.

        The reference category has a multiplier of 1.0 (no extra theta needed).
        All other categories get a free theta estimated by the model.

        Args:
            base_value:          The un-adjusted parameter value.
            category:            The individual's integer-coded category.
            theta_per_category:  Dict mapping category code → theta multiplier.
                                 The reference category need not be present.

        Returns:
            Adjusted parameter value.
        """
        if self.effect != CovariateEffect.CATEGORICAL:
            raise ValueError(f"apply_categorical called on a {self.effect.value} relationship")
        multiplier = theta_per_category.get(category, 1.0)
        return base_value * multiplier

    def generate_pk_code(self, theta_index: int) -> str:
        """
        Generate an NM-TRAN $PK code snippet for this relationship.

        The generated line multiplies the base parameter by the appropriate
        covariate function.  It assumes the base parameter has already been
        assigned (e.g., ``CL = THETA(2)*EXP(ETA(2))``) and appends a
        multiplicative correction.

        Args:
            theta_index:  1-based THETA index for the covariate coefficient.

        Returns:
            NM-TRAN code string (one or more lines).
        """
        par = self.parameter.upper()
        cov = self.covariate.upper()
        ti = theta_index  # 1-based

        if self.effect == CovariateEffect.POWER:
            return (
                f"; Power effect of {cov} on {par} (reference = {self.reference})\n"
                f"{par} = {par} * ({cov}/{self.reference})**THETA({ti})"
            )

        elif self.effect == CovariateEffect.LINEAR:
            return (
                f"; Linear effect of {cov} on {par} (reference = {self.reference})\n"
                f"{par} = {par} * (1 + THETA({ti}) * ({cov} - {self.reference}))"
            )

        elif self.effect == CovariateEffect.EXPONENTIAL:
            return (
                f"; Exponential effect of {cov} on {par} (reference = {self.reference})\n"
                f"{par} = {par} * EXP(THETA({ti}) * ({cov} - {self.reference}))"
            )

        elif self.effect == CovariateEffect.CATEGORICAL:
            if not self.categories:
                raise ValueError("categories must be specified for CATEGORICAL covariate effect")
            # Validate again at code-generation time (catches bypassed construction)
            _validate_integer_categories(self.categories)
            lines = [f"; Categorical effect of {cov} on {par}"]
            for i, _ in enumerate(self.categories[1:], start=0):
                # Each non-reference category gets its own theta
                lines.append(f"IF ({cov} == {i + 1}) {par} = {par} * THETA({ti + i})")
            return "\n".join(lines)

        else:
            raise ValueError(f"Unknown CovariateEffect: {self.effect}")  # pragma: no cover

    def __repr__(self) -> str:
        return (
            f"CovariateRelationship("
            f"parameter={self.parameter!r}, "
            f"covariate={self.covariate!r}, "
            f"effect={self.effect.value!r}, "
            f"reference={self.reference})"
        )


# ── Convenience constructors ──────────────────────────────────────────────────


def power_effect(
    parameter: str,
    covariate: str,
    reference: float = 70.0,
) -> CovariateRelationship:
    """Create a power-law covariate relationship."""
    return CovariateRelationship(
        parameter=parameter,
        covariate=covariate,
        effect=CovariateEffect.POWER,
        reference=reference,
    )


def linear_effect(
    parameter: str,
    covariate: str,
    reference: float = 0.0,
) -> CovariateRelationship:
    """Create a linear covariate relationship."""
    return CovariateRelationship(
        parameter=parameter,
        covariate=covariate,
        effect=CovariateEffect.LINEAR,
        reference=reference,
    )


def exponential_effect(
    parameter: str,
    covariate: str,
    reference: float = 0.0,
) -> CovariateRelationship:
    """Create an exponential covariate relationship."""
    return CovariateRelationship(
        parameter=parameter,
        covariate=covariate,
        effect=CovariateEffect.EXPONENTIAL,
        reference=reference,
    )


def categorical_effect(
    parameter: str,
    covariate: str,
    categories: list[str],
) -> CovariateRelationship:
    """Create a categorical covariate relationship."""
    return CovariateRelationship(
        parameter=parameter,
        covariate=covariate,
        effect=CovariateEffect.CATEGORICAL,
        categories=categories,
    )
