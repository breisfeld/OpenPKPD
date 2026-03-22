"""
External validation: Covariate effect parameterisations vs allometric scaling
literature and published pharmacometric standards.

Validates openpkpd covariate effect functions against:
  - Anderson & Holford (2008) allometric scaling theory
  - Standard NONMEM/nlmixr2 covariate parameterisations
  - Closed-form mathematical identities

All tests are purely mathematical (no fitting) and run in < 0.1 seconds.

Effect parameterisations
------------------------
POWER:       P = theta_P · (COV / ref)^theta_cov
LINEAR:      P = theta_P · (1 + theta_cov · (COV - ref))
EXPONENTIAL: P = theta_P · exp(theta_cov · (COV - ref))
CATEGORICAL: P = theta_P · theta_cat[category]

Centering property: at COV = ref, all continuous effects give P = theta_P.

References
----------
Anderson BJ & Holford NHG (2008). Mechanism-based concepts of size and
    maturity in pharmacokinetics. Annu Rev Pharmacol Toxicol 48:303–332.
West GB, Brown JH & Enquist BJ (1997). A general model for the origin of
    allometric scaling laws in biology. Science 276(5309):122–126.
Karlsson MO & Savic RM (2007). Diagnosing model diagnostics. Clin Pharmacol
    Ther 82(1):17–20.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from openpkpd.covariate.effects import CovariateRelationship

# ---------------------------------------------------------------------------
# Helper: build CovariateRelationship instances
# ---------------------------------------------------------------------------


def _power(parameter: str, covariate: str, reference: float = 70.0) -> CovariateRelationship:
    from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship

    return CovariateRelationship(
        parameter=parameter, covariate=covariate, effect=CovariateEffect.POWER, reference=reference
    )


def _linear(parameter: str, covariate: str, reference: float = 70.0) -> CovariateRelationship:
    from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship

    return CovariateRelationship(
        parameter=parameter, covariate=covariate, effect=CovariateEffect.LINEAR, reference=reference
    )


def _exponential(parameter: str, covariate: str, reference: float = 70.0) -> CovariateRelationship:
    from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship

    return CovariateRelationship(
        parameter=parameter,
        covariate=covariate,
        effect=CovariateEffect.EXPONENTIAL,
        reference=reference,
    )


def _categorical(parameter: str, covariate: str, categories: list) -> CovariateRelationship:
    from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship

    return CovariateRelationship(
        parameter=parameter,
        covariate=covariate,
        effect=CovariateEffect.CATEGORICAL,
        categories=categories,
    )


# ---------------------------------------------------------------------------
# Layer A: Centering property (universal, all effect types)
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestCenteringProperty:
    """
    At COV = reference, all continuous effect functions must return base_value.

    This is the fundamental centering property: individuals with the reference
    covariate value have the population-typical parameter (no covariate effect).
    """

    def test_power_at_reference_unchanged(self):
        """Power effect at COV=ref → P = theta_P for any theta_cov."""
        rel = _power("CL", "WT", reference=70.0)
        base = 2.5
        for theta_cov in [-0.5, 0.0, 0.75, 1.0, 2.0]:
            result = rel.apply(base_value=base, cov_value=70.0, theta_cov=theta_cov)
            np.testing.assert_allclose(
                result, base, rtol=1e-12, err_msg=f"Power at ref: theta_cov={theta_cov}"
            )

    def test_linear_at_reference_unchanged(self):
        """Linear effect at COV=ref → P = theta_P for any theta_cov."""
        rel = _linear("CL", "WT", reference=70.0)
        base = 2.5
        for theta_cov in [-1.0, 0.0, 0.5, 1.0]:
            result = rel.apply(base_value=base, cov_value=70.0, theta_cov=theta_cov)
            np.testing.assert_allclose(
                result, base, rtol=1e-12, err_msg=f"Linear at ref: theta_cov={theta_cov}"
            )

    def test_exponential_at_reference_unchanged(self):
        """Exponential effect at COV=ref → P = theta_P for any theta_cov."""
        rel = _exponential("V", "WT", reference=70.0)
        base = 30.0
        for theta_cov in [-1.0, 0.0, 0.5, 2.0]:
            result = rel.apply(base_value=base, cov_value=70.0, theta_cov=theta_cov)
            np.testing.assert_allclose(
                result, base, rtol=1e-12, err_msg=f"Exponential at ref: theta_cov={theta_cov}"
            )

    def test_categorical_reference_category_unchanged(self):
        """Reference category (default multiplier=1.0) must not change base_value."""
        rel = _categorical("CL", "SEX", categories=["female", "male"])
        base = 2.5
        result = rel.apply_categorical(base, category="female", theta_per_category={"male": 1.30})
        np.testing.assert_allclose(
            result, base, rtol=1e-12, err_msg="Reference category should leave P unchanged"
        )


# ---------------------------------------------------------------------------
# Layer B: Power effect — allometric scaling (Anderson & Holford 2008)
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestPowerEffect:
    """
    Validate power covariate effect against allometric scaling theory.

    The allometric scaling law for pharmacokinetics (Anderson & Holford 2008):
        CL ∝ WT^0.75   (blood flow / metabolic rate)
        V  ∝ WT^1.00   (isometric volume)

    Formula: P = theta_P · (WT / ref_WT)^theta_cov
    """

    def test_allometric_cl_scaling_075(self):
        """
        CL allometric scaling: CL(70 kg) = 3.0 L/h → CL(140 kg)?

        With theta_cov=0.75 (standard allometric exponent for CL):
            CL(140) = CL(70) · (140/70)^0.75 = 3.0 · 2^0.75 ≈ 5.040
        """
        rel = _power("CL", "WT", reference=70.0)
        cl_ref = 3.0  # L/h at reference weight 70 kg
        wt_new = 140.0  # kg
        expected = cl_ref * (wt_new / 70.0) ** 0.75

        result = rel.apply(base_value=cl_ref, cov_value=wt_new, theta_cov=0.75)
        np.testing.assert_allclose(
            result, expected, rtol=1e-12, err_msg="Allometric CL scaling (exponent 0.75)"
        )

    def test_allometric_v_scaling_10(self):
        """
        V allometric (isometric) scaling: V(70 kg) = 32 L → V(140 kg)?

        With theta_cov=1.0 (isometric):
            V(140) = V(70) · (140/70)^1.0 = 32 · 2 = 64 L
        """
        rel = _power("V", "WT", reference=70.0)
        v_ref = 32.0
        wt_new = 140.0
        expected = v_ref * (wt_new / 70.0) ** 1.0

        result = rel.apply(base_value=v_ref, cov_value=wt_new, theta_cov=1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-12)
        np.testing.assert_allclose(
            result, 64.0, rtol=1e-12, err_msg="Isometric V doubles when WT doubles"
        )

    def test_power_effect_formula(self):
        """P = theta_P · (COV/ref)^theta_cov — algebraic identity."""
        base = 2.5
        cov_val = 50.0
        reference = 70.0
        for theta_cov in [-0.5, 0.0, 0.75, 1.0, 1.5]:
            expected = base * (cov_val / reference) ** theta_cov
            rel = _power("CL", "WT", reference=reference)
            result = rel.apply(base, cov_val, theta_cov)
            np.testing.assert_allclose(
                result, expected, rtol=1e-12, err_msg=f"Power formula at theta_cov={theta_cov}"
            )

    def test_power_with_theta_zero_is_identity(self):
        """theta_cov = 0 → (COV/ref)^0 = 1 → P = base for any COV."""
        rel = _power("CL", "WT", reference=70.0)
        base = 2.5
        for wt in [20.0, 50.0, 70.0, 100.0, 200.0]:
            result = rel.apply(base_value=base, cov_value=wt, theta_cov=0.0)
            np.testing.assert_allclose(
                result, base, rtol=1e-12, err_msg=f"theta_cov=0 should give identity at WT={wt}"
            )

    def test_power_inverse_symmetry(self):
        """P(COV=2*ref, theta=0.75) · P(COV=0.5*ref, theta=0.75) = base^2 · 1."""
        rel = _power("CL", "WT", reference=70.0)
        base = 2.5
        p_up = rel.apply(base, 140.0, 0.75)  # 2× ref
        p_down = rel.apply(base, 35.0, 0.75)  # 0.5× ref
        # (2)^0.75 · (0.5)^0.75 = 2^0.75 · 2^-0.75 = 1
        product = (p_up * p_down) / (base**2)
        np.testing.assert_allclose(
            product, 1.0, rtol=1e-10, err_msg="Power effect inverse symmetry"
        )


# ---------------------------------------------------------------------------
# Layer C: Linear effect — NONMEM standard parameterisation
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestLinearEffect:
    """
    Validate linear covariate effect.

    Formula: P = theta_P · (1 + theta_cov · (COV - ref))

    This is the standard linear covariate model used in NONMEM and nlmixr2.
    The slope theta_cov represents fractional change per unit covariate.
    """

    def test_linear_formula(self):
        """P = theta_P · (1 + theta_cov · (COV - ref)) — exact."""
        base, ref = 2.5, 70.0
        for cov_val in [50.0, 60.0, 70.0, 80.0, 100.0]:
            for theta_cov in [-0.02, 0.0, 0.01, 0.05]:
                expected = base * (1.0 + theta_cov * (cov_val - ref))
                rel = _linear("CL", "WT", reference=ref)
                result = rel.apply(base, cov_val, theta_cov)
                np.testing.assert_allclose(
                    result,
                    expected,
                    rtol=1e-12,
                    err_msg=f"Linear at cov={cov_val}, theta={theta_cov}",
                )

    def test_linear_slope_interpretation(self):
        """
        A unit increase above reference gives proportional increase theta_cov.

        If theta_cov = 0.01 and COV - ref = 10:
            P = base · (1 + 0.01 · 10) = base · 1.10  (+10%)
        """
        rel = _linear("CL", "AGE", reference=40.0)
        base = 2.5
        result = rel.apply(base, cov_value=50.0, theta_cov=0.01)
        expected = base * (1.0 + 0.01 * (50.0 - 40.0))
        np.testing.assert_allclose(result, expected, rtol=1e-12)
        np.testing.assert_allclose(
            result, base * 1.10, rtol=1e-12, err_msg="10% above ref should give 10% higher P"
        )

    def test_linear_negative_theta_decreases_above_ref(self):
        """Negative theta_cov → P decreases when COV > ref (inverse relationship)."""
        rel = _linear("CL", "WT", reference=70.0)
        base = 2.5
        result = rel.apply(base, cov_value=100.0, theta_cov=-0.01)
        assert result < base, "Negative theta_cov should decrease P above reference"


# ---------------------------------------------------------------------------
# Layer D: Exponential effect
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestExponentialEffect:
    """
    Validate exponential covariate effect.

    Formula: P = theta_P · exp(theta_cov · (COV - ref))

    The exponential model maintains P > 0 for any theta_cov and is favoured
    for parameters that must stay positive (e.g., CL, V in log-normal models).
    """

    def test_exponential_formula(self):
        """P = theta_P · exp(theta_cov · (COV - ref)) — exact."""
        base, ref = 2.5, 70.0
        for cov_val in [40.0, 70.0, 100.0]:
            for theta_cov in [-0.05, 0.0, 0.02, 0.10]:
                expected = base * math.exp(theta_cov * (cov_val - ref))
                rel = _exponential("CL", "WT", reference=ref)
                result = rel.apply(base, cov_val, theta_cov)
                np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_exponential_always_positive(self):
        """P = theta_P · exp(...) must remain positive for any theta_cov."""
        rel = _exponential("CL", "WT", reference=70.0)
        base = 2.5
        for theta_cov in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            result = rel.apply(base, cov_value=100.0, theta_cov=theta_cov)
            assert result > 0, (
                f"Exponential effect must give positive P; got {result} at theta={theta_cov}"
            )

    def test_exponential_vs_linear_approximation(self):
        """
        For small theta_cov · (COV - ref), exponential ≈ linear (first-order Taylor).

        exp(x) ≈ 1 + x for |x| << 1, so both models agree to O(x²).
        """
        base, ref = 2.5, 70.0
        cov_val = 71.0  # COV - ref = 1 (small perturbation)
        theta_cov = 0.001  # very small effect

        rel_lin = _linear("CL", "WT", reference=ref)
        rel_exp = _exponential("CL", "WT", reference=ref)

        p_lin = rel_lin.apply(base, cov_val, theta_cov)
        p_exp = rel_exp.apply(base, cov_val, theta_cov)

        np.testing.assert_allclose(
            p_lin,
            p_exp,
            rtol=1e-4,
            err_msg="Linear and exponential should agree for tiny perturbations",
        )


# ---------------------------------------------------------------------------
# Layer E: Categorical effect
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestCategoricalEffect:
    """
    Validate categorical covariate effect (sex, race, formulation group).

    Formula: P = theta_P · theta_cat[category]
             (reference category: multiplier = 1.0)
    """

    def test_reference_category_multiplier_one(self):
        """Reference category must leave P unchanged (implicit multiplier=1.0)."""
        rel = _categorical("CL", "SEX", categories=["female", "male"])
        base = 2.5
        result = rel.apply_categorical(base, "female", theta_per_category={"male": 1.3})
        np.testing.assert_allclose(result, base, rtol=1e-12)

    def test_nonreference_category_multiplied(self):
        """Non-reference category applies the theta multiplier."""
        rel = _categorical("CL", "SEX", categories=["female", "male"])
        base = 2.5
        result = rel.apply_categorical(base, "male", theta_per_category={"male": 1.3})
        np.testing.assert_allclose(
            result,
            base * 1.3,
            rtol=1e-12,
            err_msg="Non-reference category should multiply by theta",
        )

    def test_three_category_effect(self):
        """Three-category covariate: multipliers for each non-reference category."""
        rel = _categorical("CL", "RACE", categories=["white", "black", "asian"])
        base = 3.0
        theta = {"black": 1.20, "asian": 0.85}

        np.testing.assert_allclose(rel.apply_categorical(base, "white", theta), base, rtol=1e-12)
        np.testing.assert_allclose(
            rel.apply_categorical(base, "black", theta), base * 1.20, rtol=1e-12
        )
        np.testing.assert_allclose(
            rel.apply_categorical(base, "asian", theta), base * 0.85, rtol=1e-12
        )

    def test_unknown_category_defaults_to_one(self):
        """Categories not in theta_per_category default to multiplier=1.0."""
        rel = _categorical("CL", "SEX", categories=["female", "male"])
        base = 2.5
        result = rel.apply_categorical(base, "other", theta_per_category={"male": 1.3})
        np.testing.assert_allclose(
            result, base, rtol=1e-12, err_msg="Unknown category should default to multiplier=1.0"
        )


# ---------------------------------------------------------------------------
# Layer F: Population-level invariants
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestPopulationLevelInvariants:
    """
    Validate that covariate effect functions preserve population-level invariants.

    For continuous effects: the geometric mean of P over a lognormal population
    of COV should approximately equal base_value when the population mean equals
    the reference covariate.
    """

    def test_power_geometric_mean_at_reference_population(self):
        """
        For a log-normal WT distribution centred on 70 kg:
        geometric mean of CL = base_value when theta_cov = 0.75.

        This holds because E[log(WT/70)] = 0 for a symmetric log-normal.
        """
        rng = np.random.default_rng(0)
        n = 5000
        ref = 70.0
        base = 2.5
        theta_cov = 0.75
        # Lognormal centred at ref (median = 70 kg)
        wts = ref * np.exp(rng.normal(0, 0.15, n))

        rel = _power("CL", "WT", reference=ref)
        cls_i = np.array([rel.apply(base, wt, theta_cov) for wt in wts])

        # Geometric mean of CL_i = base (exactly in the limit, ≈ for finite sample)
        geo_mean = np.exp(np.mean(np.log(cls_i)))
        np.testing.assert_allclose(
            geo_mean,
            base,
            rtol=0.01,
            err_msg="Geometric mean CL over lognormal WT pop should ≈ base",
        )

    def test_linear_arithmetic_mean_at_reference_population(self):
        """
        For a normal(ref, σ) covariate distribution:
        arithmetic mean of P ≈ base_value (linearity of expectation).

        E[P] = E[base · (1 + theta · (COV - ref))] = base · (1 + theta · E[COV - ref]) = base.
        """
        rng = np.random.default_rng(1)
        n = 5000
        ref, base, theta_cov = 70.0, 2.5, 0.01
        covs = rng.normal(ref, 10.0, n)

        rel = _linear("CL", "WT", reference=ref)
        ps = np.array([rel.apply(base, cov, theta_cov) for cov in covs])

        arith_mean = float(np.mean(ps))
        np.testing.assert_allclose(
            arith_mean,
            base,
            rtol=0.01,
            err_msg="Arithmetic mean P ≈ base for centred linear effect",
        )
