"""
Unit tests for covariate effect parameterizations and SCM utilities.

Tests cover:
  - CovariateRelationship.apply() for each supported effect type
  - CovariateRelationship.generate_pk_code() output format
  - SCMStep and SCMResult construction
  - LRT p-value helper
  - Default covariate THETA initialization
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from openpkpd.covariate.effects import (
    CovariateEffect,
    CovariateRelationship,
    categorical_effect,
    exponential_effect,
    linear_effect,
    power_effect,
)
from openpkpd.covariate.scm import SCMResult, SCMStep, _default_covariate_theta, _lrt_pvalue
from openpkpd.estimation.base import EstimationResult

# ── Power effect ──────────────────────────────────────────────────────────────


class TestPowerEffect:
    """Tests for CovariateEffect.POWER."""

    def test_power_at_reference_no_change(self) -> None:
        """At COV = reference, the power effect multiplier is 1."""
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=70.0)
        result = rel.apply(10.0, 70.0, theta_cov=0.75)
        assert result == pytest.approx(10.0)

    def test_power_above_reference_increases(self) -> None:
        """When COV > reference and theta_cov > 0, parameter increases."""
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=70.0)
        result = rel.apply(10.0, 140.0, theta_cov=0.75)
        assert result > 10.0

    def test_power_below_reference_decreases(self) -> None:
        """When COV < reference and theta_cov > 0, parameter decreases."""
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=70.0)
        result = rel.apply(10.0, 35.0, theta_cov=0.75)
        assert result < 10.0

    def test_power_exact_formula(self) -> None:
        """Verify exact power formula: base * (cov/ref)^theta."""
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=70.0)
        # WT = 140 = 2 * 70; theta_cov = 1.0 → CL doubles
        result = rel.apply(5.0, 140.0, theta_cov=1.0)
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_power_theta_zero_is_identity(self) -> None:
        """When theta_cov = 0, power effect is neutral (multiplier = 1)."""
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=70.0)
        result = rel.apply(10.0, 200.0, theta_cov=0.0)
        assert result == pytest.approx(10.0)

    def test_power_convenience_constructor(self) -> None:
        """power_effect() convenience constructor."""
        rel = power_effect("CL", "WT", reference=70.0)
        assert rel.effect == CovariateEffect.POWER
        assert rel.parameter == "CL"
        assert rel.covariate == "WT"

    def test_power_invalid_reference(self) -> None:
        """Reference <= 0 raises ValueError."""
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=0.0)
        with pytest.raises(ValueError, match="reference"):
            rel.apply(10.0, 70.0, theta_cov=0.5)


# ── Linear effect ─────────────────────────────────────────────────────────────


class TestLinearEffect:
    """Tests for CovariateEffect.LINEAR."""

    def test_linear_at_reference_no_change(self) -> None:
        """At COV = reference, linear effect multiplier is 1."""
        rel = CovariateRelationship("CL", "AGE", CovariateEffect.LINEAR, reference=40.0)
        result = rel.apply(5.0, 40.0, theta_cov=0.1)
        assert result == pytest.approx(5.0)

    def test_linear_above_reference_with_positive_theta(self) -> None:
        """When COV > reference and theta_cov > 0, parameter increases."""
        rel = CovariateRelationship("CL", "AGE", CovariateEffect.LINEAR, reference=40.0)
        result = rel.apply(5.0, 60.0, theta_cov=0.1)
        expected = 5.0 * (1.0 + 0.1 * (60.0 - 40.0))
        assert result == pytest.approx(expected)

    def test_linear_below_reference_with_positive_theta(self) -> None:
        """When COV < reference and theta_cov > 0, parameter decreases."""
        rel = CovariateRelationship("CL", "AGE", CovariateEffect.LINEAR, reference=40.0)
        result = rel.apply(5.0, 20.0, theta_cov=0.1)
        expected = 5.0 * (1.0 + 0.1 * (20.0 - 40.0))
        assert result == pytest.approx(expected)

    def test_linear_convenience_constructor(self) -> None:
        """linear_effect() convenience constructor."""
        rel = linear_effect("V", "AGE", reference=40.0)
        assert rel.effect == CovariateEffect.LINEAR
        assert rel.reference == 40.0


# ── Exponential effect ────────────────────────────────────────────────────────


class TestExponentialEffect:
    """Tests for CovariateEffect.EXPONENTIAL."""

    def test_exponential_at_reference_no_change(self) -> None:
        """At COV = reference, exp(theta * 0) = 1 → no change."""
        rel = CovariateRelationship("V", "WT", CovariateEffect.EXPONENTIAL, reference=70.0)
        result = rel.apply(10.0, 70.0, theta_cov=0.02)
        assert result == pytest.approx(10.0)

    def test_exponential_above_reference_increases(self) -> None:
        """When COV > reference and theta_cov > 0, parameter increases."""
        rel = CovariateRelationship("V", "WT", CovariateEffect.EXPONENTIAL, reference=70.0)
        result = rel.apply(10.0, 100.0, theta_cov=0.02)
        assert result > 10.0

    def test_exponential_exact_formula(self) -> None:
        """Verify exact exp formula: base * exp(theta * (cov - ref))."""
        rel = CovariateRelationship("V", "WT", CovariateEffect.EXPONENTIAL, reference=70.0)
        result = rel.apply(10.0, 100.0, theta_cov=0.02)
        expected = 10.0 * math.exp(0.02 * (100.0 - 70.0))
        assert result == pytest.approx(expected)

    def test_exponential_theta_zero_is_identity(self) -> None:
        """theta_cov = 0: exp(0) = 1 → no change."""
        rel = CovariateRelationship("V", "WT", CovariateEffect.EXPONENTIAL, reference=70.0)
        result = rel.apply(10.0, 200.0, theta_cov=0.0)
        assert result == pytest.approx(10.0)

    def test_exponential_convenience_constructor(self) -> None:
        """exponential_effect() convenience constructor."""
        rel = exponential_effect("KA", "AGE")
        assert rel.effect == CovariateEffect.EXPONENTIAL


# ── Categorical effect ────────────────────────────────────────────────────────


class TestCategoricalEffect:
    """Tests for CovariateEffect.CATEGORICAL."""

    def test_categorical_apply_returns_base_unchanged(self) -> None:
        """apply() for CATEGORICAL returns base_value (handled by apply_categorical)."""
        rel = CovariateRelationship(
            "CL",
            "SEX",
            CovariateEffect.CATEGORICAL,
            categories=["male", "female"],
        )
        result = rel.apply(10.0, 0.0, theta_cov=1.5)
        assert result == pytest.approx(10.0)

    def test_categorical_apply_categorical_reference_unchanged(self) -> None:
        """Reference category has multiplier 1.0."""
        rel = CovariateRelationship(
            "CL",
            "SEX",
            CovariateEffect.CATEGORICAL,
            categories=["male", "female"],
        )
        result = rel.apply_categorical(10.0, "male", {"female": 1.3})
        assert result == pytest.approx(10.0)

    def test_categorical_apply_categorical_non_reference(self) -> None:
        """Non-reference category applies the given multiplier."""
        rel = CovariateRelationship(
            "CL",
            "SEX",
            CovariateEffect.CATEGORICAL,
            categories=["male", "female"],
        )
        result = rel.apply_categorical(10.0, "female", {"female": 1.3})
        assert result == pytest.approx(13.0)

    def test_categorical_convenience_constructor(self) -> None:
        """categorical_effect() convenience constructor."""
        rel = categorical_effect("CL", "SEX", categories=["male", "female"])
        assert rel.effect == CovariateEffect.CATEGORICAL
        assert rel.categories == ["male", "female"]

    def test_apply_categorical_wrong_effect_type(self) -> None:
        """apply_categorical on a non-categorical relationship raises ValueError."""
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER)
        with pytest.raises(ValueError, match="apply_categorical"):
            rel.apply_categorical(10.0, "A", {"A": 1.0})


# ── generate_pk_code ──────────────────────────────────────────────────────────


class TestGeneratePKCode:
    """Tests for CovariateRelationship.generate_pk_code()."""

    def test_power_code_contains_theta(self) -> None:
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=70.0)
        code = rel.generate_pk_code(theta_index=4)
        assert "THETA(4)" in code
        assert "WT" in code
        assert "CL" in code

    def test_linear_code_contains_correct_structure(self) -> None:
        rel = CovariateRelationship("V", "AGE", CovariateEffect.LINEAR, reference=40.0)
        code = rel.generate_pk_code(theta_index=5)
        assert "THETA(5)" in code
        assert "AGE" in code
        assert "V" in code

    def test_exponential_code_contains_exp(self) -> None:
        rel = CovariateRelationship("CL", "WT", CovariateEffect.EXPONENTIAL, reference=70.0)
        code = rel.generate_pk_code(theta_index=3)
        assert "EXP" in code.upper()
        assert "THETA(3)" in code

    def test_categorical_code_requires_categories(self) -> None:
        rel = CovariateRelationship(
            "CL",
            "SEX",
            CovariateEffect.CATEGORICAL,
            categories=["male", "female"],
        )
        code = rel.generate_pk_code(theta_index=4)
        assert "IF" in code.upper() or "THETA" in code


# ── SCMStep and SCMResult ─────────────────────────────────────────────────────


class TestSCMDataClasses:
    """Tests for SCMStep and SCMResult structure."""

    def _make_step(self, step_type: str = "forward", accepted: bool = True) -> SCMStep:
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER)
        return SCMStep(
            step_type=step_type,
            relationship=rel,
            ofv_base=100.0,
            ofv_new=93.5,
            delta_ofv=-6.5,
            df=1,
            p_value=0.011,
            accepted=accepted,
        )

    def test_scm_step_forward_creation(self) -> None:
        step = self._make_step()
        assert step.step_type == "forward"
        assert step.accepted is True
        assert step.delta_ofv == pytest.approx(-6.5)

    def test_scm_step_backward_creation(self) -> None:
        step = self._make_step("backward", accepted=False)
        assert step.step_type == "backward"
        assert step.accepted is False

    def test_scm_step_str_forward(self) -> None:
        step = self._make_step()
        s = str(step)
        assert "FORWARD" in s
        assert "ACCEPTED" in s

    def test_scm_step_str_backward_rejected(self) -> None:
        step = self._make_step("backward", accepted=False)
        s = str(step)
        assert "BACKWARD" in s
        assert "rejected" in s

    def test_scm_result_summary(self) -> None:
        step = self._make_step()
        from openpkpd.estimation.base import EstimationResult

        est_res = EstimationResult(
            theta_final=np.array([1.5, 0.08, 30.0]),
            omega_final=np.eye(3) * 0.3,
            sigma_final=np.eye(1) * 0.1,
            ofv=93.5,
        )
        result = SCMResult(
            base_ofv=100.0,
            final_ofv=93.5,
            accepted_relationships=[step.relationship],
            steps=[step],
            model_history=[est_res],
        )
        summary = result.summary()
        assert "Base OFV" in summary
        assert "Final OFV" in summary
        assert "CL" in summary

    def test_scm_result_no_accepted(self) -> None:
        result = SCMResult(
            base_ofv=100.0,
            final_ofv=100.0,
            accepted_relationships=[],
            steps=[],
            model_history=[],
        )
        summary = result.summary()
        assert "No covariate" in summary


# ── LRT p-value ───────────────────────────────────────────────────────────────


class TestLRTPValue:
    """Tests for _lrt_pvalue()."""

    def test_zero_delta_ofv(self) -> None:
        """ΔOFV = 0 → p-value of 1.0 (no improvement)."""
        assert _lrt_pvalue(0.0, df=1) == pytest.approx(1.0)

    def test_negative_delta_ofv(self) -> None:
        """Negative ΔOFV (worsening from caller's sign convention) → p=1."""
        assert _lrt_pvalue(-5.0, df=1) == pytest.approx(1.0)

    def test_large_delta_ofv_small_pvalue(self) -> None:
        """Large improvement → very small p-value."""
        p = _lrt_pvalue(20.0, df=1)
        assert p < 0.001

    def test_chi2_critical_value(self) -> None:
        """ΔOFV ≈ chi2(1, 0.05) critical value → p ≈ 0.05."""
        # chi2(1) critical value at 5% significance ≈ 3.841
        p = _lrt_pvalue(3.841, df=1)
        assert abs(p - 0.05) < 0.01

    def test_df_2(self) -> None:
        """Works for df=2 (e.g., categorical covariate with 2 categories)."""
        p = _lrt_pvalue(5.991, df=2)  # chi2(2) 5% critical value
        assert abs(p - 0.05) < 0.02


# ── Default THETA initialisation ──────────────────────────────────────────────


class TestDefaultCovariateTheta:
    """Tests for _default_covariate_theta()."""

    def test_power_starts_at_zero(self) -> None:
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER)
        init, lower, upper = _default_covariate_theta(rel)
        assert init == pytest.approx(0.0)

    def test_linear_starts_at_zero(self) -> None:
        rel = CovariateRelationship("CL", "AGE", CovariateEffect.LINEAR)
        init, lower, upper = _default_covariate_theta(rel)
        assert init == pytest.approx(0.0)

    def test_exponential_starts_at_zero(self) -> None:
        rel = CovariateRelationship("V", "WT", CovariateEffect.EXPONENTIAL)
        init, lower, upper = _default_covariate_theta(rel)
        assert init == pytest.approx(0.0)

    def test_categorical_starts_at_one(self) -> None:
        rel = CovariateRelationship(
            "CL",
            "SEX",
            CovariateEffect.CATEGORICAL,
            categories=["M", "F"],
        )
        init, lower, upper = _default_covariate_theta(rel)
        assert init == pytest.approx(1.0)
        assert lower > 0.0  # must be positive (multiplicative)


# ── Parallel _forward_step ────────────────────────────────────────────────────


class _MockEstResult:
    """Minimal EstimationResult stand-in."""

    def __init__(self, ofv: float) -> None:
        self.ofv = ofv


def _make_scm_engine(n_jobs: int = 1):
    """Create a minimal SCMEngine with no real model builder."""
    from unittest.mock import MagicMock

    from openpkpd.covariate.scm import SCMEngine

    engine = SCMEngine(
        base_model_builder=MagicMock(),
        base_pk_code="CL = THETA(1)",
        candidates=[],
        n_jobs=n_jobs,
    )
    return engine


class TestSCMParallelForwardStep:
    """Tests for parallel candidate evaluation in _forward_step."""

    def _candidates(self) -> list[CovariateRelationship]:
        return [
            CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=70.0),
            CovariateRelationship("V", "WT", CovariateEffect.POWER, reference=70.0),
            CovariateRelationship("CL", "AGE", CovariateEffect.LINEAR, reference=40.0),
        ]

    def _mock_fit(self, engine, ofv_by_candidate: dict[str, float]):
        """Patch _fit_with_addition to return mock results keyed by covariate name."""
        from unittest.mock import patch

        def _fit_with_addition(accepted, candidate):
            key = candidate.covariate
            if key not in ofv_by_candidate:
                raise ValueError(f"unexpected candidate {key}")
            return _MockEstResult(ofv_by_candidate[key])

        return patch.object(engine, "_fit_with_addition", side_effect=_fit_with_addition)

    def test_sequential_returns_best_candidate(self) -> None:
        """n_jobs=1: _forward_step selects the candidate with the largest OFV drop."""
        engine = _make_scm_engine(n_jobs=1)
        engine.forward_pvalue = 0.05
        candidates = self._candidates()
        base_result = _MockEstResult(ofv=100.0)

        ofv_map = {"WT": 90.0, "AGE": 95.0}  # WT gives bigger drop
        with self._mock_fit(engine, ofv_map):
            step = engine._forward_step(base_result, candidates[:2], [])

        assert step is not None
        assert step.relationship.covariate == "WT"
        assert step.delta_ofv == pytest.approx(-10.0)

    def test_parallel_n_jobs_2_same_result_as_sequential(self) -> None:
        """n_jobs=2 produces the same best step as n_jobs=1."""
        candidates = self._candidates()[:2]
        base_result = _MockEstResult(ofv=100.0)
        ofv_map = {"WT": 92.0, "AGE": 88.0}  # AGE gives bigger drop

        engine_seq = _make_scm_engine(n_jobs=1)
        engine_seq.forward_pvalue = 0.05
        engine_par = _make_scm_engine(n_jobs=2)
        engine_par.forward_pvalue = 0.05

        with self._mock_fit(engine_seq, ofv_map):
            step_seq = engine_seq._forward_step(base_result, candidates, [])
        with self._mock_fit(engine_par, ofv_map):
            step_par = engine_par._forward_step(base_result, candidates, [])

        assert step_seq is not None and step_par is not None
        assert step_seq.relationship.covariate == step_par.relationship.covariate
        assert step_seq.delta_ofv == pytest.approx(step_par.delta_ofv)

    def test_parallel_n_jobs_minus_one_returns_step(self) -> None:
        """n_jobs=-1 (all CPUs) still returns a valid step."""
        engine = _make_scm_engine(n_jobs=-1)
        engine.forward_pvalue = 0.05
        candidates = self._candidates()[:2]
        base_result = _MockEstResult(ofv=100.0)
        ofv_map = {"WT": 85.0, "AGE": 93.0}

        with self._mock_fit(engine, ofv_map):
            step = engine._forward_step(base_result, candidates, [])

        assert step is not None
        assert step.relationship.covariate == "WT"

    def test_failed_candidate_skipped_other_returned(self) -> None:
        """If one candidate raises, the other is still evaluated."""
        from unittest.mock import patch

        engine = _make_scm_engine(n_jobs=1)
        engine.forward_pvalue = 0.05
        all_cands = self._candidates()
        candidates = [all_cands[0], all_cands[2]]  # CL~WT, CL~AGE
        base_result = _MockEstResult(ofv=100.0)

        def _fit_with_addition(accepted, candidate):
            if candidate.covariate == "WT":
                raise RuntimeError("fit failed")
            return _MockEstResult(ofv=90.0)

        with patch.object(engine, "_fit_with_addition", side_effect=_fit_with_addition):
            step = engine._forward_step(base_result, candidates, [])

        assert step is not None
        assert step.relationship.covariate == "AGE"

    def test_all_candidates_fail_returns_none(self) -> None:
        """If every candidate raises, _forward_step returns None."""
        from unittest.mock import patch

        engine = _make_scm_engine(n_jobs=1)
        candidates = self._candidates()[:2]
        base_result = _MockEstResult(ofv=100.0)

        with patch.object(engine, "_fit_with_addition", side_effect=RuntimeError("fail")):
            step = engine._forward_step(base_result, candidates, [])

        assert step is None

    def test_no_improvement_returns_none_when_threshold_high(self) -> None:
        """If no candidate improves OFV, best_step has delta=0 → not selected."""
        engine = _make_scm_engine(n_jobs=1)
        engine.forward_pvalue = 0.05
        candidates = self._candidates()[:2]
        base_result = _MockEstResult(ofv=100.0)
        ofv_map = {"WT": 101.0, "AGE": 102.0}  # all worsen the fit

        with self._mock_fit(engine, ofv_map):
            step = engine._forward_step(base_result, candidates, [])

        assert step is None

    def test_parallel_preserves_accepted_flag(self) -> None:
        """Parallel result sets accepted=True when p-value < forward_pvalue."""
        engine = _make_scm_engine(n_jobs=2)
        engine.forward_pvalue = 0.05
        candidates = self._candidates()[:1]
        base_result = _MockEstResult(ofv=100.0)
        ofv_map = {"WT": 90.0}  # ΔOFV=10 → p << 0.05

        with self._mock_fit(engine, ofv_map):
            step = engine._forward_step(base_result, candidates, [])

        assert step is not None
        assert step.accepted is True


class _DeterministicSCMEngine:
    """SCM test helper with deterministic OFVs keyed by accepted relationships."""

    def __init__(self, candidates, ofv_map, **kwargs):
        from openpkpd.covariate.scm import SCMEngine

        class _Engine(SCMEngine):
            pass

        self._engine = _Engine(
            base_model_builder=MagicMock(),
            base_pk_code="CL = THETA(1)",
            candidates=candidates,
            **kwargs,
        )
        self._ofv_map = ofv_map
        self.fit_history = []

        def _fit_current(_pk_suffixes, accepted_rels):
            key = self._key(accepted_rels)
            self.fit_history.append(key)
            return EstimationResult(
                theta_final=np.array([1.0]),
                omega_final=np.array([[0.1]]),
                sigma_final=np.array([[0.1]]),
                ofv=self._ofv_map[key],
            )

        self._engine._fit_current = _fit_current  # type: ignore[method-assign]

    @staticmethod
    def _key(relationships):
        return tuple(sorted((r.parameter, r.covariate, r.effect.value) for r in relationships))

    def __getattr__(self, name):
        return getattr(self._engine, name)


class TestSCMRunSemantics:
    """Focused tests for SCMEngine.run() forward/backward bookkeeping."""

    def test_run_allows_backward_removal_after_single_forward_acceptance(self) -> None:
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=70.0)
        engine = _DeterministicSCMEngine(
            candidates=[rel],
            ofv_map={
                (): 100.0,
                (("CL", "WT", "power"),): 95.0,
            },
            forward_pvalue=0.05,
            backward_pvalue=0.001,
        )

        result = engine.run()

        assert result.base_ofv == pytest.approx(100.0)
        assert result.final_ofv == pytest.approx(100.0)
        assert result.accepted_relationships == []
        assert [(step.step_type, step.accepted) for step in result.steps] == [
            ("forward", True),
            ("backward", True),
        ]
        assert [res.ofv for res in result.model_history] == pytest.approx([100.0, 95.0, 100.0])

    def test_run_retains_single_relationship_when_backward_worsening_is_significant(self) -> None:
        rel = CovariateRelationship("CL", "WT", CovariateEffect.POWER, reference=70.0)
        engine = _DeterministicSCMEngine(
            candidates=[rel],
            ofv_map={
                (): 100.0,
                (("CL", "WT", "power"),): 88.0,
            },
            forward_pvalue=0.05,
            backward_pvalue=0.001,
        )

        result = engine.run()

        assert result.final_ofv == pytest.approx(88.0)
        assert result.accepted_relationships == [rel]
        assert [(step.step_type, step.accepted) for step in result.steps] == [("forward", True)]
        assert [res.ofv for res in result.model_history] == pytest.approx([100.0, 88.0])
