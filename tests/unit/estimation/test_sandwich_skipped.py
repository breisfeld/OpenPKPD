"""
Tests for Sandwich S-matrix skipped subject tracking (C3) and
high condition number cov_success=False (C4).
"""
from __future__ import annotations

import warnings
from unittest.mock import patch

import numpy as np
import pytest

from openpkpd.covariance.sandwich import (
    CovarianceEstimationWarning,
    CovarianceResult,
    SandwichCovariance,
)
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(n_theta=1):
    theta_specs = [ThetaSpec(init=1.0) for _ in range(n_theta)]
    omega_specs = [OmegaSpec(block_size=1, values=[0.1])]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.05])]
    return ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)


class _DummyIndividualModel:
    def __init__(self, sid, should_fail=False):
        self.sid = sid
        self.should_fail = should_fail

    def obj_eta(self, eta, theta, omega, sigma, trans=2):
        # Simple quadratic; 'should_fail' is used at gradient level via population model
        return float(0.5 * np.sum(eta ** 2) + float(theta[0]))


class _DummyPopulationModel:
    trans = 2

    def __init__(self, subject_infos: list[tuple[int, bool]]):
        """subject_infos: list of (subject_id, should_fail)."""
        self._models = {sid: _DummyIndividualModel(sid, fail) for sid, fail in subject_infos}
        self._failing_sids = {sid for sid, fail in subject_infos if fail}

    def individual_model(self, sid):
        return self._models[sid]

    def subject_ids(self):
        return list(self._models.keys())


# ---------------------------------------------------------------------------
# C3 Tests: Skipped subjects — patch numerical_gradient to raise for specific subjects
# ---------------------------------------------------------------------------

def test_all_subjects_succeed_no_warning():
    """All subjects succeed -> skipped_subject_ids is empty, no S-matrix warning."""
    pop = _DummyPopulationModel([(1, False), (2, False), (3, False)])
    params = _make_params(n_theta=1)
    eta_hat = {1: np.zeros(1), 2: np.zeros(1), 3: np.zeros(1)}

    cov_est = SandwichCovariance(eps=1e-4)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = cov_est.compute(pop, params, eta_hat)

    cov_warnings = [x for x in w if issubclass(x.category, CovarianceEstimationWarning)
                    and "S-matrix" in str(x.message)]
    assert len(cov_warnings) == 0
    assert hasattr(result, "skipped_subject_ids")
    assert result.skipped_subject_ids == []


def test_one_subject_fails_warning_emitted():
    """numerical_gradient raises for subject 2 -> S-matrix warning, ID in skipped_subject_ids."""
    pop = _DummyPopulationModel([(1, False), (2, True), (3, False)])
    params = _make_params(n_theta=1)
    eta_hat = {1: np.zeros(1), 2: np.zeros(1), 3: np.zeros(1)}

    # We'll track call count to make numerical_gradient raise only for subject 2
    call_count = [0]
    original_gradient = None

    def patched_gradient(f, x, eps):
        # Raise on the 2nd unique gradient call (which corresponds to subject 2)
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("Simulated gradient failure for subject 2")
        return np.zeros(len(x))

    cov_est = SandwichCovariance(eps=1e-4)

    with patch("openpkpd.covariance.sandwich.numerical_gradient", side_effect=patched_gradient):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cov_est.compute(pop, params, eta_hat)

    cov_warnings = [x for x in w if issubclass(x.category, CovarianceEstimationWarning)
                    and "S-matrix" in str(x.message)]
    assert len(cov_warnings) >= 1, (
        f"Expected S-matrix CovarianceEstimationWarning, got: {[str(x.message) for x in w]}"
    )
    assert hasattr(result, "skipped_subject_ids")
    assert len(result.skipped_subject_ids) >= 1


def test_skipped_subjects_se_finite():
    """With N=3, 1 failing, SE is still finite and warning was emitted."""
    pop = _DummyPopulationModel([(1, False), (2, False), (3, False)])
    params = _make_params(n_theta=1)
    eta_hat = {1: np.zeros(1), 2: np.zeros(1), 3: np.zeros(1)}

    call_count = [0]

    def patched_gradient(f, x, eps):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Simulated failure for first subject")
        return np.ones(len(x)) * 0.1

    cov_est = SandwichCovariance(eps=1e-4)

    with patch("openpkpd.covariance.sandwich.numerical_gradient", side_effect=patched_gradient):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cov_est.compute(pop, params, eta_hat)

    cov_warnings = [x for x in w if issubclass(x.category, CovarianceEstimationWarning)
                    and "S-matrix" in str(x.message)]
    assert len(cov_warnings) >= 1
    assert np.all(np.isfinite(result.se)), f"SE not finite: {result.se}"
    assert len(result.skipped_subject_ids) >= 1


# ---------------------------------------------------------------------------
# C4 Tests: High condition number -> cov_success=False
# ---------------------------------------------------------------------------

def test_well_conditioned_no_condition_warning():
    """Well-conditioned R (cond ~ 10) -> no condition number CovarianceEstimationWarning."""
    pop = _DummyPopulationModel([(1, False), (2, False)])
    params = _make_params(n_theta=1)
    eta_hat = {1: np.zeros(1), 2: np.zeros(1)}

    cov_est = SandwichCovariance(eps=1e-4)

    # Patch cond to return a small value
    with patch("openpkpd.covariance.sandwich.np.linalg.cond", return_value=10.0):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cov_est.compute(pop, params, eta_hat)

    cond_warnings = [x for x in w if issubclass(x.category, CovarianceEstimationWarning)
                     and "condition number" in str(x.message).lower()]
    assert len(cond_warnings) == 0, (
        f"Unexpected condition number warning: {[str(x.message) for x in cond_warnings]}"
    )
    assert result.converged is True


def test_ill_conditioned_warning_and_converged_false():
    """Ill-conditioned R (cond > 1e10) -> CovarianceEstimationWarning, converged=False."""
    pop = _DummyPopulationModel([(1, False), (2, False)])
    params = _make_params(n_theta=1)
    eta_hat = {1: np.zeros(1), 2: np.zeros(1)}

    cov_est = SandwichCovariance(eps=1e-4)

    with patch("openpkpd.covariance.sandwich.np.linalg.cond", return_value=1e15):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cov_est.compute(pop, params, eta_hat)

    cond_warnings = [x for x in w if issubclass(x.category, CovarianceEstimationWarning)
                     and "condition number" in str(x.message).lower()]
    assert len(cond_warnings) >= 1, (
        f"Expected CovarianceEstimationWarning about condition number, got: {[str(x.message) for x in w]}"
    )
    assert result.converged is False


def test_near_singular_matrix_condition_number():
    """Near-singular 2x2 matrix has condition number > 1e10."""
    eps = 1e-12
    cov_matrix = np.array([[1.0, 1 - eps], [1 - eps, 1.0]])
    cond = np.linalg.cond(cov_matrix)
    assert cond > 1e10, f"Expected cond > 1e10, got {cond:.2e}"
