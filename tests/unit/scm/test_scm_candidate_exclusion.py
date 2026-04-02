"""Tests for SCM persistent failure tracking (SC5)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship
from openpkpd.covariate.scm import SCMEngine


def _make_relationship(param: str = "CL", covariate: str = "WT") -> CovariateRelationship:
    return CovariateRelationship(
        parameter=param,
        covariate=covariate,
        effect=CovariateEffect.POWER,
        reference=70.0,
    )


def _make_engine(*candidates: CovariateRelationship) -> SCMEngine:
    """Build a minimal SCMEngine with mocked model builder."""
    builder = MagicMock()
    return SCMEngine(
        base_model_builder=builder,
        base_pk_code="CL=theta[0]*math.exp(eta[0])",
        candidates=list(candidates),
        forward_pvalue=0.05,
    )


class TestSCMCandidateExclusion:
    """Tests for SCM permanent candidate exclusion after repeated failures."""

    def _make_base_result(self, ofv: float = 100.0) -> MagicMock:
        r = MagicMock()
        r.ofv = ofv
        r.converged = True
        return r

    def test_candidate_fails_once_not_excluded(self):
        """A candidate that fails once should not be permanently excluded."""
        engine = _make_engine(_make_relationship("CL", "WT"))
        assert ("CL", "WT") not in engine._permanently_excluded
        assert engine._failed_counts.get(("CL", "WT"), 0) == 0

        base_result = self._make_base_result()
        remaining = [_make_relationship("CL", "WT")]

        # Patch _fit_with_addition to raise once
        with patch.object(engine, "_fit_with_addition", side_effect=RuntimeError("fit failed")):
            engine._forward_step(base_result, remaining, [])

        # After one failure, failure count = 1, not permanently excluded
        assert engine._failed_counts.get(("CL", "WT"), 0) == 1
        assert ("CL", "WT") not in engine._permanently_excluded

    def test_candidate_fails_twice_permanently_excluded(self, caplog):
        """A candidate that fails twice should be permanently excluded with a WARNING."""
        engine = _make_engine(_make_relationship("CL", "WT"))
        base_result = self._make_base_result()
        remaining = [_make_relationship("CL", "WT")]

        with patch.object(engine, "_fit_with_addition", side_effect=RuntimeError("fit failed")):
            with caplog.at_level(logging.WARNING, logger="openpkpd.covariate.scm"):
                # First failure
                engine._forward_step(base_result, remaining, [])
                # Second failure — should trigger permanent exclusion
                engine._forward_step(base_result, remaining, [])

        assert ("CL", "WT") in engine._permanently_excluded
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("permanently excluded" in msg for msg in warning_messages), (
            f"Expected 'permanently excluded' warning, got: {warning_messages}"
        )

    def test_candidate_fails_then_succeeds_counter_reset(self):
        """A candidate that fails once then succeeds should have its counter reset."""
        engine = _make_engine(_make_relationship("CL", "WT"))
        base_result = self._make_base_result(ofv=100.0)
        remaining = [_make_relationship("CL", "WT")]

        # First call: fail
        with patch.object(engine, "_fit_with_addition", side_effect=RuntimeError("fail")):
            engine._forward_step(base_result, remaining, [])

        assert engine._failed_counts.get(("CL", "WT"), 0) == 1

        # Second call: succeed
        success_result = self._make_base_result(ofv=80.0)  # big improvement
        with patch.object(engine, "_fit_with_addition", return_value=success_result):
            engine._forward_step(base_result, remaining, [])

        # Counter should be reset
        assert engine._failed_counts.get(("CL", "WT"), 0) == 0
        assert ("CL", "WT") not in engine._permanently_excluded

    def test_permanently_excluded_never_tried_again(self):
        """Once permanently excluded, candidate should not be attempted in subsequent steps."""
        engine = _make_engine(_make_relationship("CL", "WT"))
        base_result = self._make_base_result()
        remaining = [_make_relationship("CL", "WT")]

        # Mark as permanently excluded manually
        engine._permanently_excluded.add(("CL", "WT"))

        call_count = {"n": 0}

        def _track_calls(*args, **kwargs):
            call_count["n"] += 1
            return self._make_base_result(ofv=90.0)

        with patch.object(engine, "_fit_with_addition", side_effect=_track_calls):
            engine._forward_step(base_result, remaining, [])
            engine._forward_step(base_result, remaining, [])

        # The excluded candidate should never be tried
        assert call_count["n"] == 0, (
            f"Permanently excluded candidate was attempted {call_count['n']} time(s)"
        )

    def test_permanently_excluded_multiple_iterations(self, caplog):
        """Over multiple forward steps, permanently excluded candidates are never retried."""
        cand1 = _make_relationship("CL", "WT")
        cand2 = _make_relationship("V", "AGE")
        engine = _make_engine(cand1, cand2)
        base_result = self._make_base_result(ofv=100.0)
        remaining = [cand1, cand2]

        # Make cand1 always fail, cand2 always succeed (but not significantly)
        success_result = self._make_base_result(ofv=99.0)

        call_log = {"CL": 0, "V": 0}

        def _side_effect(accepted, candidate):
            if candidate.parameter == "CL":
                call_log["CL"] += 1
                raise RuntimeError("CL fit failed")
            call_log["V"] += 1
            return success_result

        with patch.object(engine, "_fit_with_addition", side_effect=_side_effect):
            with caplog.at_level(logging.WARNING, logger="openpkpd.covariate.scm"):
                # 3 iterations
                for _ in range(3):
                    engine._forward_step(base_result, remaining, [])

        # CL should have been tried at most 2 times (excluded after 2nd failure)
        assert call_log["CL"] <= 2, (
            f"CL candidate was tried {call_log['CL']} times; should stop after 2"
        )
        assert ("CL", "WT") in engine._permanently_excluded

    def test_failed_counts_initialized_empty(self):
        """SCMEngine should start with empty failure tracking structures."""
        engine = _make_engine(_make_relationship())
        assert engine._failed_counts == {}
        assert engine._permanently_excluded == set()
