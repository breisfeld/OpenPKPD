"""Tests for covariate effects warnings and construction-time validation — CV3 + CV4."""

from __future__ import annotations

import warnings

import pytest

from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship


# ---------------------------------------------------------------------------
# CV3: Power effect warning for non-positive covariate values
# ---------------------------------------------------------------------------


class TestPowerEffectWarning:
    def test_positive_value_no_warning(self):
        """Positive covariate value → no warning, correct power effect."""
        rel = CovariateRelationship(
            parameter="CL",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = rel.apply(base_value=1.0, cov_value=70.0, theta_cov=0.75)

        assert result == pytest.approx(1.0)

    def test_zero_value_emits_warning(self):
        """Zero covariate value → UserWarning emitted, result is finite (clamped)."""
        rel = CovariateRelationship(
            parameter="CL",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        )
        with pytest.warns(UserWarning, match="non-positive"):
            result = rel.apply(base_value=1.0, cov_value=0.0, theta_cov=0.75)

        assert result > 0
        import math
        assert math.isfinite(result)

    def test_negative_value_emits_warning(self):
        """Negative covariate value → UserWarning emitted."""
        rel = CovariateRelationship(
            parameter="CL",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        )
        with pytest.warns(UserWarning, match="non-positive"):
            result = rel.apply(base_value=1.0, cov_value=-5.0, theta_cov=0.75)

        assert result > 0  # clamped to 1e-10

    def test_power_identity_at_reference(self):
        """Power effect with cov == reference is exactly 1.0 × base_value."""
        rel = CovariateRelationship(
            parameter="CL",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        )
        result = rel.apply(base_value=1.0, cov_value=70.0, theta_cov=0.75)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_power_double_reference(self):
        """Power effect with cov=140, ref=70, theta=0.75 → 2^0.75 ≈ 1.6818."""
        rel = CovariateRelationship(
            parameter="CL",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        )
        result = rel.apply(base_value=1.0, cov_value=140.0, theta_cov=0.75)
        expected = 2.0 ** 0.75  # ≈ 1.6818
        assert result == pytest.approx(expected, rel=1e-6)

    def test_warning_clamping_produces_finite_result(self):
        """Clamped value (1e-10) for zero input produces a finite result."""
        rel = CovariateRelationship(
            parameter="CL",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result = rel.apply(base_value=2.0, cov_value=0.0, theta_cov=1.0)

        import math
        assert math.isfinite(result)
        assert result > 0


# ---------------------------------------------------------------------------
# CV4: Construction-time category validation
# ---------------------------------------------------------------------------


class TestCategoricalConstructionValidation:
    def test_valid_integer_categories_no_error(self):
        """Constructing with valid integer categories [1, 2, 3] → no error."""
        rel = CovariateRelationship(
            parameter="CL",
            covariate="SEX",
            effect=CovariateEffect.CATEGORICAL,
            categories=[1, 2],
        )
        assert rel.categories == [1, 2]

    def test_string_categories_raise_at_construction(self):
        """Constructing with string categories raises ValueError at construction time."""
        with pytest.raises(ValueError, match="not an integer"):
            CovariateRelationship(
                parameter="CL",
                covariate="SEX",
                effect=CovariateEffect.CATEGORICAL,
                categories=["MALE", "FEMALE"],
            )

    def test_zero_based_categories_raise_at_construction(self):
        """Constructing with [0, 1, 2] (0-based) raises ValueError at construction time."""
        with pytest.raises(ValueError, match="starting at 1"):
            CovariateRelationship(
                parameter="CL",
                covariate="RACE",
                effect=CovariateEffect.CATEGORICAL,
                categories=[0, 1, 2],
            )

    def test_error_at_construction_not_deferred(self):
        """The ValueError is raised immediately at construction, not at generate_pk_code()."""
        raised = False
        try:
            CovariateRelationship(
                parameter="CL",
                covariate="SEX",
                effect=CovariateEffect.CATEGORICAL,
                categories=["M", "F"],
            )
        except ValueError:
            raised = True

        assert raised, "Expected ValueError at construction time, not deferred to code gen"

    def test_categories_none_no_validation(self):
        """Constructing a CATEGORICAL relationship with categories=None does not raise."""
        rel = CovariateRelationship(
            parameter="CL",
            covariate="SEX",
            effect=CovariateEffect.CATEGORICAL,
            categories=None,
        )
        assert rel.categories is None

    def test_non_categorical_with_no_categories_no_error(self):
        """Constructing a POWER relationship with no categories never raises."""
        rel = CovariateRelationship(
            parameter="CL",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        )
        assert rel.categories is None

    def test_discontinuous_categories_raise(self):
        """Categories with gaps (e.g., [1, 3]) raise ValueError at construction."""
        with pytest.raises(ValueError, match="contiguous"):
            CovariateRelationship(
                parameter="CL",
                covariate="RACE",
                effect=CovariateEffect.CATEGORICAL,
                categories=[1, 3],
            )
