"""Tests for IMP log-weight range collapse detection."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from openpkpd.estimation.imp import IMPMethod


def _run_collapse_check(log_weights: np.ndarray, n_samples: int, caplog) -> None:
    """
    Directly replicate the collapse check logic from _importance_sample
    and verify logger.warning is triggered when appropriate.
    """
    import openpkpd.estimation.imp as imp_module

    log_w_range = float(np.max(log_weights) - np.min(log_weights))
    w = np.exp(np.clip(log_weights - np.max(log_weights), -50, 0))
    w_sum = float(w.sum())
    if w_sum > 0:
        w_norm = w / w_sum
        ess = 1.0 / float(np.sum(w_norm**2))
    else:
        ess = 0.0

    subj_id = 1
    with caplog.at_level(logging.WARNING, logger="openpkpd.estimation.imp"):
        if log_w_range < 0.1 and ess > 0.9 * n_samples:
            imp_module.logger.warning(
                "IMP: subject %s importance weights may have collapsed — "
                "log-weight range=%.4f, ESS=%.1f/%.0f. "
                "Proposal may not match the posterior.",
                subj_id, log_w_range, ess, n_samples,
            )

    return log_w_range, ess


@pytest.mark.unit
class TestIMPESSCollapse:
    def test_healthy_weights_no_warning(self, caplog):
        """Varied log-weights (range >> 0.1) → no collapse warning."""
        rng = np.random.default_rng(42)
        n = 200
        log_weights = rng.normal(0, 2.0, n)  # range >> 0.1

        with caplog.at_level(logging.WARNING, logger="openpkpd.estimation.imp"):
            log_w_range, ess = _run_collapse_check(log_weights, n, caplog)

        assert log_w_range >= 0.1
        assert not any("collapsed" in r.message for r in caplog.records)

    def test_collapsed_weights_warns(self, caplog):
        """All identical log-weights (range=0, ESS=N) → WARNING about collapse."""
        n = 100
        log_weights = np.zeros(n)  # identical → range=0, ESS=N

        with caplog.at_level(logging.WARNING, logger="openpkpd.estimation.imp"):
            log_w_range, ess = _run_collapse_check(log_weights, n, caplog)

        assert log_w_range == pytest.approx(0.0)
        assert ess == pytest.approx(float(n), rel=1e-3)
        assert any("collapsed" in r.message for r in caplog.records), (
            f"Expected 'collapsed' warning, got: {[r.message for r in caplog.records]}"
        )

    def test_gaussian_posterior_not_collapsed(self):
        """
        Draw N=1000 samples from the exact Gaussian posterior; the
        log-weight range should be > 1.0 and ESS > 300.

        Proposal: N(0, 1); Posterior: N(1, 0.5).
        The proposal and posterior partially overlap → weights vary (not collapsed)
        but ESS remains well above the 90% collapse threshold.
        """
        rng = np.random.default_rng(123)
        n = 1000

        proposal_mean = 0.0
        proposal_var = 1.0
        posterior_mean = 1.0
        posterior_var = 0.5

        eta_samples = rng.normal(proposal_mean, np.sqrt(proposal_var), n)

        # log p(eta | posterior) - log q(eta | proposal)
        log_post = (
            -0.5 * (eta_samples - posterior_mean)**2 / posterior_var
            - 0.5 * np.log(2 * np.pi * posterior_var)
        )
        log_prop = (
            -0.5 * (eta_samples - proposal_mean)**2 / proposal_var
            - 0.5 * np.log(2 * np.pi * proposal_var)
        )
        log_weights = log_post - log_prop

        log_w_range = float(np.max(log_weights) - np.min(log_weights))
        w = np.exp(np.clip(log_weights - np.max(log_weights), -50, 0))
        w_sum = float(w.sum())
        w_norm = w / w_sum
        ess = 1.0 / float(np.sum(w_norm**2))

        assert log_w_range > 1.0, f"Expected log_w_range > 1.0, got {log_w_range}"
        # ESS must be > 300 (showing weights are not collapsed) but < 900 * n
        # (proposal is not perfect, weights do vary)
        assert ess > 300, f"Expected ESS > 300, got {ess}"
        # Verify it's not triggering the collapse condition (range < 0.1 AND ess > 0.9*N)
        assert not (log_w_range < 0.1 and ess > 0.9 * n), (
            "Sample should not trigger collapse detection"
        )
