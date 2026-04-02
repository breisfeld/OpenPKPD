"""
CV1: Tests for CovariateRelationship categorical integer encoding validation.
"""
from __future__ import annotations

import pytest

from openpkpd.covariate.effects import (
    CovariateEffect,
    CovariateRelationship,
    categorical_effect,
    _validate_integer_categories,
)


# ── Test 1: Valid integer 1..K construction ──────────────────────────────────

def test_valid_single_category():
    """Single category [1] is valid (only reference, no non-ref categories)."""
    rel = CovariateRelationship(
        parameter="CL",
        covariate="SEX",
        effect=CovariateEffect.CATEGORICAL,
        categories=[1],
    )
    assert rel.categories == [1]


def test_valid_two_categories():
    rel = CovariateRelationship(
        parameter="CL",
        covariate="SEX",
        effect=CovariateEffect.CATEGORICAL,
        categories=[1, 2],
    )
    assert rel.categories == [1, 2]


def test_valid_three_categories():
    rel = CovariateRelationship(
        parameter="V",
        covariate="RACE",
        effect=CovariateEffect.CATEGORICAL,
        categories=[1, 2, 3],
    )
    assert rel.categories == [1, 2, 3]


def test_valid_out_of_order_integers():
    """Categories don't need to be provided in order — they just need to sort to 1..K."""
    rel = CovariateRelationship(
        parameter="V",
        covariate="RACE",
        effect=CovariateEffect.CATEGORICAL,
        categories=[3, 1, 2],
    )
    assert rel.categories == [3, 1, 2]


# ── Test 2: String categories raise ValueError at construction time ────────────
# CV4: validation was moved from generate_pk_code() to __post_init__.
# Both construction-time and code-gen errors are tested here.

def test_string_categories_raise_at_construction():
    """String categories raise ValueError at construction (CV4 behaviour)."""
    with pytest.raises(ValueError, match="not an integer"):
        CovariateRelationship(
            parameter="CL",
            covariate="SEX",
            effect=CovariateEffect.CATEGORICAL,
            categories=["MALE", "FEMALE"],
        )


def test_string_categories_validator_direct():
    """_validate_integer_categories rejects strings directly."""
    with pytest.raises(ValueError, match="not an integer"):
        _validate_integer_categories(["MALE", "FEMALE"])


def test_mixed_string_int_categories_raise_at_construction():
    """Mixed string/int categories raise ValueError at construction (CV4 behaviour)."""
    with pytest.raises(ValueError, match="not an integer"):
        CovariateRelationship(
            parameter="CL",
            covariate="SEX",
            effect=CovariateEffect.CATEGORICAL,
            categories=[1, "TWO"],
        )


# ── Test 3: Non-contiguous integers raise ValueError ─────────────────────────

def test_noncontiguous_categories_raise():
    """Non-contiguous integers are caught by the validator and at code-gen."""
    with pytest.raises(ValueError, match="contiguous range"):
        _validate_integer_categories([1, 3])


def test_gap_in_range_raise():
    with pytest.raises(ValueError, match="contiguous range"):
        _validate_integer_categories([1, 2, 4])


# ── Test 4: 0-based integers raise ValueError ────────────────────────────────

def test_zero_based_categories_raise():
    """0-based categories are rejected by the validator."""
    with pytest.raises(ValueError, match="starting at 1"):
        _validate_integer_categories([0, 1, 2])


def test_zero_alone_raises():
    with pytest.raises(ValueError, match="starting at 1"):
        _validate_integer_categories([0])


# ── Test 5: Numerical — code generation matches expected theta assignments ────

def test_categorical_code_generation_3_categories():
    """For a 3-category effect, verify the correct IF blocks are generated.

    For categories [1, 2, 3]:
      - Category 1 is the reference (no IF block — remains unchanged)
      - Categories 2 and 3 get their own THETA multipliers
    The loop iterates over categories[1:] so:
      i=0 → IF (SEX == 1) CL = CL * THETA(ti+0)   (category 2 in 1-based space)
      i=1 → IF (SEX == 2) CL = CL * THETA(ti+1)   (category 3 in 1-based space)
    """
    rel = CovariateRelationship(
        parameter="CL",
        covariate="SEX",
        effect=CovariateEffect.CATEGORICAL,
        categories=[1, 2, 3],
    )
    code = rel.generate_pk_code(theta_index=5)

    # Two non-reference categories, two IF lines
    assert "IF (SEX == 1) CL = CL * THETA(5)" in code
    assert "IF (SEX == 2) CL = CL * THETA(6)" in code
    # No IF line for category 3 (it's actually i=1 which gives SEX==2)
    assert "IF (SEX == 3)" not in code


def test_categorical_numerical_eval():
    """Verify per-category theta application via the compiled callable."""
    from openpkpd.parser.code_compiler import CompiledPKCallable

    # Build a $PK snippet: CL = THETA(1) * EXP(ETA(1))
    # then apply a 3-category sex effect starting at THETA(2)
    rel = CovariateRelationship(
        parameter="CL",
        covariate="SEX",
        effect=CovariateEffect.CATEGORICAL,
        categories=[1, 2, 3],
    )
    cov_code = rel.generate_pk_code(theta_index=2)

    # The generated NM-TRAN code uses IF (SEX == 1) CL = CL * THETA(2) for category 2
    # and IF (SEX == 2) CL = CL * THETA(3) for category 3.
    # We build a full $PK snippet and compile it.
    pk_snippet = f"CL = THETA(1)\n{cov_code}"
    from openpkpd.parser.code_compiler import NMTRANCompiler
    compiler = NMTRANCompiler()
    pk_fn = compiler.compile_pk(pk_snippet)

    theta = [10.0, 2.0, 0.5]  # CL_base=10, cat2_mult=2.0, cat3_mult=0.5
    eta = [0.0]

    # SEX not matching any IF (cov_value=0 or any other value) → CL = theta[0] = 10.0
    result_ref = pk_fn(theta, eta, covariates={"SEX": 0.0})
    assert abs(result_ref.get("CL", result_ref.get("cl", float("nan"))) - 10.0) < 1e-10

    # SEX == 1 → IF (SEX == 1) fires → CL = 10.0 * THETA(2) = 10 * 2.0 = 20.0
    result_cat2 = pk_fn(theta, eta, covariates={"SEX": 1.0})
    assert abs(result_cat2.get("CL", result_cat2.get("cl", float("nan"))) - 20.0) < 1e-10, (
        f"Category 2: expected 20.0, got {result_cat2}"
    )

    # SEX == 2 → IF (SEX == 2) fires → CL = 10.0 * THETA(3) = 10 * 0.5 = 5.0
    result_cat3 = pk_fn(theta, eta, covariates={"SEX": 2.0})
    assert abs(result_cat3.get("CL", result_cat3.get("cl", float("nan"))) - 5.0) < 1e-10, (
        f"Category 3: expected 5.0, got {result_cat3}"
    )


def test_validate_at_code_gen_bypassed_construction():
    """Validation runs again at code-gen even if construction was bypassed."""
    # Bypass __post_init__ by building the object manually
    rel = object.__new__(CovariateRelationship)
    rel.parameter = "CL"
    rel.covariate = "SEX"
    rel.effect = CovariateEffect.CATEGORICAL
    rel.reference = 1
    rel.categories = ["BAD", "CATS"]  # invalid — bypass __post_init__

    with pytest.raises(ValueError, match="not an integer"):
        rel.generate_pk_code(theta_index=1)
