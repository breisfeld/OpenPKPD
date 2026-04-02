"""
Tests for SCM Bonferroni correction option (SC4).

Tests:
  1. correction=None → same results as before (no change in behaviour)
  2. correction='bonferroni' with 20 candidates → effective alpha = forward_pvalue/20
  3. Candidate that passes uncorrected but fails Bonferroni → not selected
  4. Candidate that passes Bonferroni → selected regardless of correction setting
  5. correction='invalid' → ValueError
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship
from openpkpd.covariate.scm import SCMEngine, SCMStep, _lrt_pvalue
from openpkpd.estimation.base import EstimationResult
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rel(param: str = "CL", cov: str = "WT") -> CovariateRelationship:
    return CovariateRelationship(
        parameter=param,
        covariate=cov,
        effect=CovariateEffect.POWER,
        reference=70.0,
    )


def _make_est_result(ofv: float, converged: bool = True) -> EstimationResult:
    return EstimationResult(
        theta_final=np.array([1.0]),
        omega_final=np.array([[0.1]]),
        sigma_final=np.array([[0.05]]),
        ofv=ofv,
        converged=converged,
        method="FOCE",
        message="ok",
    )


def _make_engine(
    candidates: list[CovariateRelationship],
    forward_pvalue: float = 0.05,
    correction=None,
) -> SCMEngine:
    """Build a minimal SCMEngine with mocked builder."""
    builder = MagicMock()
    builder.clone.return_value = builder
    builder._theta_specs = []
    builder.pk.return_value = builder
    builder.estimation.return_value = builder

    engine = SCMEngine(
        base_model_builder=builder,
        base_pk_code="CL = THETA(1)",
        candidates=candidates,
        forward_pvalue=forward_pvalue,
        correction=correction,
        n_jobs=1,
    )
    return engine


# ---------------------------------------------------------------------------
# Test 1: correction=None behaves same as before
# ---------------------------------------------------------------------------


def test_no_correction_same_as_uncorrected():
    """correction=None: acceptance threshold is forward_pvalue directly."""
    rel = _make_rel()
    engine = _make_engine([rel], forward_pvalue=0.05, correction=None)
    # Verify the effective alpha in _forward_step is just forward_pvalue
    # We check by confirming a candidate with p < 0.05 is accepted
    base_result = _make_est_result(ofv=500.0)
    candidate_result = _make_est_result(ofv=494.0)  # delta = -6 → p << 0.05

    with patch.object(engine, "_fit_with_addition", return_value=candidate_result):
        step = engine._forward_step(base_result, [rel], [])

    assert step is not None
    assert step.accepted is True


def test_correction_none_is_default():
    """Default correction is None."""
    rel = _make_rel()
    engine = _make_engine([rel])
    assert engine.correction is None


# ---------------------------------------------------------------------------
# Test 2: correction='bonferroni' with 20 candidates → effective alpha = fwd/20
# ---------------------------------------------------------------------------


def test_bonferroni_effective_alpha_20_candidates():
    """With 20 candidates and Bonferroni, effective alpha = forward_pvalue / 20."""
    rels = [_make_rel(param="CL", cov=f"COV{i}") for i in range(20)]
    engine = _make_engine(rels, forward_pvalue=0.05, correction="bonferroni")

    # p-value that passes uncorrected (0.05) but fails Bonferroni (0.05/20 = 0.0025)
    # delta_ofv = 5.0 → chi2(1) p ≈ 0.025
    base_result = _make_est_result(ofv=500.0)
    # OFV improvement of 5 gives p≈0.025 < 0.05 but > 0.0025
    candidate_result = _make_est_result(ofv=495.0)

    # All candidates return the same result
    with patch.object(engine, "_fit_with_addition", return_value=candidate_result):
        step = engine._forward_step(base_result, rels, [])

    assert step is not None
    # p ≈ 0.025 > 0.0025, so Bonferroni should reject
    assert step.accepted is False


# ---------------------------------------------------------------------------
# Test 3: Candidate passes uncorrected but fails Bonferroni → not selected
# ---------------------------------------------------------------------------


def test_bonferroni_rejects_borderline_candidate():
    """A candidate with p=0.03 passes uncorrected (0.05) but fails Bonferroni/5."""
    rels = [_make_rel(param="CL", cov=f"COV{i}") for i in range(5)]
    # forward_pvalue=0.05, Bonferroni alpha = 0.05/5 = 0.01
    engine_uncorr = _make_engine(rels, forward_pvalue=0.05, correction=None)
    engine_bonf = _make_engine(rels, forward_pvalue=0.05, correction="bonferroni")

    base_result = _make_est_result(ofv=500.0)
    # delta_ofv = 4.7 → p ≈ 0.030 < 0.05 but > 0.01
    candidate_result = _make_est_result(ofv=495.3)

    with patch.object(engine_uncorr, "_fit_with_addition", return_value=candidate_result):
        step_uncorr = engine_uncorr._forward_step(base_result, rels, [])

    with patch.object(engine_bonf, "_fit_with_addition", return_value=candidate_result):
        step_bonf = engine_bonf._forward_step(base_result, rels, [])

    # Uncorrected accepts; Bonferroni rejects
    assert step_uncorr is not None and step_uncorr.accepted is True
    assert step_bonf is not None and step_bonf.accepted is False


# ---------------------------------------------------------------------------
# Test 4: Candidate that passes Bonferroni → selected in both settings
# ---------------------------------------------------------------------------


def test_strong_candidate_passes_both():
    """A very significant candidate (p << 0.001) is selected with or without correction."""
    rels = [_make_rel(param="CL", cov=f"COV{i}") for i in range(10)]

    engine_uncorr = _make_engine(rels, forward_pvalue=0.05, correction=None)
    engine_bonf = _make_engine(rels, forward_pvalue=0.05, correction="bonferroni")

    base_result = _make_est_result(ofv=500.0)
    # delta_ofv = 30 → p extremely small
    candidate_result = _make_est_result(ofv=470.0)

    with patch.object(engine_uncorr, "_fit_with_addition", return_value=candidate_result):
        step_uncorr = engine_uncorr._forward_step(base_result, rels, [])

    with patch.object(engine_bonf, "_fit_with_addition", return_value=candidate_result):
        step_bonf = engine_bonf._forward_step(base_result, rels, [])

    assert step_uncorr is not None and step_uncorr.accepted is True
    assert step_bonf is not None and step_bonf.accepted is True


# ---------------------------------------------------------------------------
# Test 5: correction='invalid' → ValueError
# ---------------------------------------------------------------------------


def test_invalid_correction_raises_value_error():
    """correction='invalid' → ValueError at construction time."""
    with pytest.raises(ValueError, match="correction="):
        _make_engine([_make_rel()], correction="invalid")
