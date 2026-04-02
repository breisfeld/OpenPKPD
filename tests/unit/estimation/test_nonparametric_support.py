"""
Tests for nonparametric support point truncation logging (NNP3).

Tests:
  1. n_support <= n_subjects → no INFO log about truncation
  2. n_support > n_subjects → INFO log with requested and actual counts
  3. After truncation, the number of support points equals n_subjects
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.estimation.nonparametric import NonparametricMethod, NonparametricResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base_result(n_subjects: int) -> EstimationResult:
    """Return a fake EstimationResult with post-hoc ETAs for n_subjects."""
    n_eta = 2
    return EstimationResult(
        theta_final=np.array([1.0, 2.0]),
        omega_final=np.eye(n_eta) * 0.1,
        sigma_final=np.array([[0.05]]),
        ofv=500.0,
        converged=True,
        method="FOCE",
        message="ok",
        post_hoc_etas={
            sid: np.random.default_rng(sid).standard_normal(n_eta)
            for sid in range(1, n_subjects + 1)
        },
    )


def _make_pop_model(n_subjects: int) -> MagicMock:
    """Return a minimal mock PopulationModel."""
    pop = MagicMock()
    pop.subject_ids.return_value = list(range(1, n_subjects + 1))
    pop.n_subjects.return_value = n_subjects

    # Individual models for weight optimisation
    indiv = MagicMock()
    # obj_eta returns a constant so the EM weights stay uniform
    indiv.obj_eta.return_value = 0.0
    pop.individual_model.return_value = indiv

    pop.trans = 1
    return pop


def _make_params(n_eta: int = 2) -> MagicMock:
    params = MagicMock()
    params.omega = np.eye(n_eta) * 0.1
    params.omega.shape = (n_eta, n_eta)
    return params


# ---------------------------------------------------------------------------
# Test 1: n_support <= n_subjects → no truncation log
# ---------------------------------------------------------------------------


def test_no_log_when_support_within_subjects(caplog):
    """n_support=5 with n_subjects=10 → no truncation INFO log."""
    n_subjects = 10
    n_support = 5

    method = NonparametricMethod(
        base_method="FOCE",
        n_support_points=n_support,
        max_iter=2,
    )

    base_result = _make_base_result(n_subjects)
    pop = _make_pop_model(n_subjects)
    params = _make_params()

    with patch("openpkpd.estimation.get_estimation_method") as mock_get:
        mock_get.return_value.estimate.return_value = base_result
        with caplog.at_level(logging.INFO, logger="openpkpd.estimation.nonparametric"):
            result = method.estimate(pop, params)

    truncation_logs = [
        r for r in caplog.records
        if "truncating" in r.message.lower()
    ]
    assert len(truncation_logs) == 0


def test_no_log_when_support_equals_subjects(caplog):
    """n_support=10 with n_subjects=10 → no truncation log."""
    n_subjects = 10
    n_support = 10

    method = NonparametricMethod(
        base_method="FOCE",
        n_support_points=n_support,
        max_iter=2,
    )

    base_result = _make_base_result(n_subjects)
    pop = _make_pop_model(n_subjects)
    params = _make_params()

    with patch("openpkpd.estimation.get_estimation_method") as mock_get:
        mock_get.return_value.estimate.return_value = base_result
        with caplog.at_level(logging.INFO, logger="openpkpd.estimation.nonparametric"):
            result = method.estimate(pop, params)

    truncation_logs = [
        r for r in caplog.records
        if "truncating" in r.message.lower()
    ]
    assert len(truncation_logs) == 0


# ---------------------------------------------------------------------------
# Test 2: n_support > n_subjects → INFO log with both counts
# ---------------------------------------------------------------------------


def test_info_log_when_support_exceeds_subjects(caplog):
    """n_support=20 with n_subjects=10 → INFO log mentioning both 20 and 10."""
    n_subjects = 10
    n_support = 20

    method = NonparametricMethod(
        base_method="FOCE",
        n_support_points=n_support,
        max_iter=2,
    )

    base_result = _make_base_result(n_subjects)
    pop = _make_pop_model(n_subjects)
    params = _make_params()

    with patch("openpkpd.estimation.get_estimation_method") as mock_get:
        mock_get.return_value.estimate.return_value = base_result
        with caplog.at_level(logging.INFO, logger="openpkpd.estimation.nonparametric"):
            result = method.estimate(pop, params)

    truncation_logs = [
        r for r in caplog.records
        if "truncating" in r.getMessage().lower() or (
            "support" in r.getMessage().lower() and "10" in r.getMessage()
        )
    ]
    assert len(truncation_logs) >= 1
    # The log message should mention the requested count (20) and actual (10)
    msg = truncation_logs[0].getMessage()
    assert "20" in msg
    assert "10" in msg


# ---------------------------------------------------------------------------
# Test 3: After truncation, n_support_points in result equals n_subjects
# ---------------------------------------------------------------------------


def test_support_points_capped_at_n_subjects(caplog):
    """After truncation, the result has exactly n_subjects support points."""
    n_subjects = 8
    n_support = 50  # way over

    method = NonparametricMethod(
        base_method="FOCE",
        n_support_points=n_support,
        max_iter=2,
    )

    base_result = _make_base_result(n_subjects)
    pop = _make_pop_model(n_subjects)
    params = _make_params()

    with patch("openpkpd.estimation.get_estimation_method") as mock_get:
        mock_get.return_value.estimate.return_value = base_result
        with caplog.at_level(logging.INFO, logger="openpkpd.estimation.nonparametric"):
            result = method.estimate(pop, params)

    assert len(result.support_weights) == n_subjects
    assert result.support_points.shape[0] == n_subjects
