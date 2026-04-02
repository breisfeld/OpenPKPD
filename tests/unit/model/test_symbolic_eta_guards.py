"""Tests for _common_symbolic_build_guards logging (SY2)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("sympy", reason="sympy required for symbolic gradient tests")

from openpkpd.model.symbolic_eta import _common_symbolic_build_guards
from openpkpd.utils.constants import BLQMethod


def _make_callable_with_source(source: str) -> MagicMock:
    """Create a mock callable with a ._source attribute."""
    m = MagicMock()
    m._source = source
    return m


def _valid_indiv(**overrides) -> SimpleNamespace:
    """Build a minimal valid IndividualModel-like namespace that passes all guards."""
    obj = SimpleNamespace(
        pk_callable=_make_callable_with_source("KA=theta[0]*math.exp(eta[0])\nCL=theta[1]*math.exp(eta[1])\nV=theta[2]*math.exp(eta[2])"),
        error_callable=_make_callable_with_source("w=f*theta[3]"),
        occasion_indices=None,
        blq_method=BLQMethod.M1,
        lloq=None,
        des_callable=None,
        _error_requires_amounts=False,
        _base_covariates={},
        _observation_covariates=(),
        subject_events=None,
        n_eps=1,
    )
    for k, v in overrides.items():
        setattr(obj, k, v)
    return obj


class TestCommonSymbolicBuildGuards:
    """Tests for _common_symbolic_build_guards debug logging."""

    def test_pk_callable_none_returns_false_and_logs(self, caplog):
        """Guard returns False and logs 'pk_callable' when pk_callable is None."""
        indiv = _valid_indiv(pk_callable=None)
        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.symbolic_eta"):
            result = _common_symbolic_build_guards(indiv)
        assert result is False
        assert any("pk_callable" in r.message for r in caplog.records), (
            f"Expected 'pk_callable' in debug log, got: {[r.message for r in caplog.records]}"
        )

    def test_error_callable_none_returns_false_and_logs(self, caplog):
        """Guard returns False and logs 'error_callable' when error_callable is None."""
        indiv = _valid_indiv(error_callable=None)
        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.symbolic_eta"):
            result = _common_symbolic_build_guards(indiv)
        assert result is False
        assert any("error_callable" in r.message for r in caplog.records)

    def test_non_m1_blq_returns_false_and_logs(self, caplog):
        """Guard returns False and logs 'blq_method' for non-M1 BLQ."""
        indiv = _valid_indiv(blq_method=BLQMethod.M3)
        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.symbolic_eta"):
            result = _common_symbolic_build_guards(indiv)
        assert result is False
        assert any("blq_method" in r.message for r in caplog.records), (
            f"Expected 'blq_method' in debug log, got: {[r.message for r in caplog.records]}"
        )

    def test_iov_active_returns_false_and_logs(self, caplog):
        """Guard returns False and logs 'IOV' when occasion_indices is set."""
        indiv = _valid_indiv(occasion_indices=[0, 2])
        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.symbolic_eta"):
            result = _common_symbolic_build_guards(indiv)
        assert result is False
        assert any("IOV" in r.message or "occasion" in r.message for r in caplog.records), (
            f"Expected IOV/occasion mention in debug log, got: {[r.message for r in caplog.records]}"
        )

    def test_lloq_set_returns_false_and_logs(self, caplog):
        """Guard returns False and logs when lloq is set (BLQ-related)."""
        indiv = _valid_indiv(lloq=0.1)
        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.symbolic_eta"):
            result = _common_symbolic_build_guards(indiv)
        assert result is False
        # lloq triggers the same IOV/BLQ branch
        assert any(r.levelno == logging.DEBUG for r in caplog.records)

    def test_des_callable_set_returns_false_and_logs(self, caplog):
        """Guard returns False and logs when des_callable is set."""
        indiv = _valid_indiv(des_callable=MagicMock())
        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.symbolic_eta"):
            result = _common_symbolic_build_guards(indiv)
        assert result is False
        assert any("des_callable" in r.message for r in caplog.records)

    def test_valid_model_returns_true_no_guard_log(self, caplog):
        """Valid model (all conditions met) returns True; no DEBUG guard log emitted."""
        indiv = _valid_indiv()
        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.symbolic_eta"):
            result = _common_symbolic_build_guards(indiv)
        assert result is True
        # No guard-related debug messages should appear
        guard_messages = [
            r.message
            for r in caplog.records
            if "Symbolic gradient unavailable" in r.message
        ]
        assert len(guard_messages) == 0, (
            f"Unexpected guard messages for valid model: {guard_messages}"
        )

    def test_no_source_on_callable_returns_false_and_logs(self, caplog):
        """Guard returns False and logs when callable has no _source attribute."""
        pk = MagicMock()
        del pk._source  # Remove _source so getattr returns None
        pk._source = None  # Actually set to None so isinstance check fails
        indiv = _valid_indiv(pk_callable=pk)
        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.symbolic_eta"):
            result = _common_symbolic_build_guards(indiv)
        assert result is False
