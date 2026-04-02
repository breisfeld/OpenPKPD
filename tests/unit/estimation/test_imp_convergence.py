"""
Tests for IMPMAP MAP convergence warning (I3).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openpkpd.estimation.imp import IMPMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummySubjectEvents:
    def __init__(self):
        self.obs_dv = np.array([2.0])

    def observation_mask(self):
        return np.array([True])


class _DummyIndividualModel:
    def __init__(self, dv=2.0, obs_var=0.09, prior_var=1.0, prior_mean=0.0):
        self.dv = dv
        self.obs_var = obs_var
        self.prior_var = prior_var
        self.prior_mean = prior_mean
        self.subject_events = _DummySubjectEvents()

    def obj_eta(self, eta, theta, omega, sigma, trans=None):
        eta_val = float(eta[0])
        # negative log posterior (Gaussian likelihood + Gaussian prior)
        obs_nll = 0.5 * (self.dv - eta_val) ** 2 / self.obs_var
        prior_nll = 0.5 * (eta_val - self.prior_mean) ** 2 / self.prior_var
        return 2.0 * (obs_nll + prior_nll)

    def log_likelihood(self, theta, eta, sigma, trans=None):
        eta_val = float(eta[0])
        return -0.5 * (self.dv - eta_val) ** 2 / self.obs_var - 0.5 * np.log(2 * np.pi * self.obs_var)


class _DummyPopulationModel:
    trans = 2

    def __init__(self, indiv):
        self._indiv = indiv
        self._sid = 42

    def individual_model(self, subj_id):
        return self._indiv

    def subject_ids(self):
        return [self._sid]

    def n_subjects(self):
        return 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_map_failure_logs_warning(caplog):
    """Mock minimize result with success=False -> WARNING logged (via _map_etas)."""
    import logging

    imp = IMPMethod(isample=10, maxeval=1, is_map=True)
    indiv = _DummyIndividualModel()
    pop = _DummyPopulationModel(indiv)

    from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
    params = ParameterSet.from_specs(
        [ThetaSpec(init=1.0)],
        [OmegaSpec(block_size=1, values=[1.0])],
        [SigmaSpec(block_size=1, values=[0.09])],
    )

    failed_result = SimpleNamespace(
        x=np.array([0.0]),
        success=False,
        message="maxiter exceeded",
        fun=1.0,
    )

    with patch("openpkpd.estimation.imp.minimize", return_value=failed_result):
        with caplog.at_level(logging.WARNING, logger="openpkpd.estimation.imp"):
            try:
                imp._map_etas(pop, params)
            except Exception:
                pass  # we only care about the warning

    assert any("MAP did not converge" in r.message for r in caplog.records), (
        f"Expected 'MAP did not converge' warning, got: {[r.message for r in caplog.records]}"
    )


def test_map_success_no_warning(caplog):
    """Mock minimize result with success=True -> no warning."""
    import logging

    imp = IMPMethod(isample=10, maxeval=1, is_map=True)
    indiv = _DummyIndividualModel()
    pop = _DummyPopulationModel(indiv)

    from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
    params = ParameterSet.from_specs(
        [ThetaSpec(init=1.0)],
        [OmegaSpec(block_size=1, values=[1.0])],
        [SigmaSpec(block_size=1, values=[0.09])],
    )

    success_result = SimpleNamespace(
        x=np.array([1.0]),
        success=True,
        message="Converged",
        fun=0.5,
    )

    with patch("openpkpd.estimation.imp.minimize", return_value=success_result):
        with caplog.at_level(logging.WARNING, logger="openpkpd.estimation.imp"):
            try:
                imp._map_etas(pop, params)
            except Exception:
                pass

    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("MAP did not converge" in m for m in warning_msgs), (
        f"Unexpected 'MAP did not converge' warning: {warning_msgs}"
    )


def test_map_posterior_mean_numerical():
    """MAP estimate should equal posterior mean for known Gaussian problem."""
    # Analytical posterior mean: (prior_mean/prior_var + obs/obs_var) / (1/prior_var + 1/obs_var)
    obs = 3.0
    obs_var = 0.25    # sigma2 = 0.25
    prior_mean = 0.0
    prior_var = 1.0

    posterior_mean = (prior_mean / prior_var + obs / obs_var) / (1 / prior_var + 1 / obs_var)

    from scipy.optimize import minimize

    def neg_log_posterior(eta):
        eta_val = float(eta[0])
        return 0.5 * (eta_val - obs) ** 2 / obs_var + 0.5 * (eta_val - prior_mean) ** 2 / prior_var

    result = minimize(neg_log_posterior, [0.0], method="L-BFGS-B")
    map_estimate = float(result.x[0])

    assert abs(map_estimate - posterior_mean) < 1e-4, (
        f"MAP estimate {map_estimate:.6f} vs analytical posterior mean {posterior_mean:.6f}"
    )
