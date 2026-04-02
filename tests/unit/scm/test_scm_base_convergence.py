"""Tests for SCM base model convergence check — SC3."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship
from openpkpd.covariate.scm import SCMEngine, SCMError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_result(converged: bool, ofv: float = 100.0, **extra):
    """Create a minimal EstimationResult-like mock."""
    result = SimpleNamespace(
        converged=converged,
        ofv=ofv,
        theta_final=[1.0],
        omega_final=[[0.1]],
        sigma_final=[[0.05]],
        post_hoc_etas={},
        **extra,
    )
    return result


def _make_mock_builder():
    """Create a minimal ModelBuilder mock that returns the expected interface."""
    builder = MagicMock()
    builder._theta_specs = []
    builder.clone.return_value = builder
    return builder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSCMBaseConvergence:
    def test_converged_base_model_proceeds(self):
        """SCM with a converged base model proceeds normally (does not raise)."""
        converged_result = _make_mock_result(converged=True, ofv=100.0)

        engine = SCMEngine(
            base_model_builder=_make_mock_builder(),
            base_pk_code="CL = THETA(1)",
            candidates=[],
        )

        with patch.object(engine, "_fit_current", return_value=converged_result):
            result = engine.run()

        assert result.base_ofv == pytest.approx(100.0)

    def test_non_converged_base_model_raises_scm_error(self):
        """SCM with base_result.converged=False raises SCMError with OFV in message."""
        non_converged_result = _make_mock_result(converged=False, ofv=999.1234)

        engine = SCMEngine(
            base_model_builder=_make_mock_builder(),
            base_pk_code="CL = THETA(1)",
            candidates=[],
        )

        with patch.object(engine, "_fit_current", return_value=non_converged_result):
            with pytest.raises(SCMError, match="999.1234"):
                engine.run()

    def test_non_converged_error_message_mentions_base_model(self):
        """SCMError message says 'Base model did not converge'."""
        non_converged_result = _make_mock_result(converged=False, ofv=500.0)

        engine = SCMEngine(
            base_model_builder=_make_mock_builder(),
            base_pk_code="CL = THETA(1)",
            candidates=[],
        )

        with patch.object(engine, "_fit_current", return_value=non_converged_result):
            with pytest.raises(SCMError, match="Base model did not converge"):
                engine.run()

    def test_missing_converged_attribute_defaults_to_ok(self):
        """SCM with a base_result lacking 'converged' attribute proceeds (getattr default=True)."""
        # SimpleNamespace without converged attribute
        result_no_converged = SimpleNamespace(
            ofv=100.0,
            theta_final=[1.0],
            omega_final=[[0.1]],
            sigma_final=[[0.05]],
            post_hoc_etas={},
        )
        # Should NOT have a 'converged' attribute
        assert not hasattr(result_no_converged, "converged")

        engine = SCMEngine(
            base_model_builder=_make_mock_builder(),
            base_pk_code="CL = THETA(1)",
            candidates=[],
        )

        with patch.object(engine, "_fit_current", return_value=result_no_converged):
            # Should not raise — getattr(..., True) defaults to converged=True
            result = engine.run()

        assert result.base_ofv == pytest.approx(100.0)

    def test_scm_error_is_runtime_error(self):
        """SCMError is a subclass of RuntimeError."""
        assert issubclass(SCMError, RuntimeError)
