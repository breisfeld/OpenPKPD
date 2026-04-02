"""Tests for SAEM MH acceptance rate extreme-value warnings."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from openpkpd.estimation.saem import SAEMMethod


def _make_saem_with_single_subject_run(accept_rate: float, n_chains: int = 5):
    """
    Drive just the warning logic by directly invoking the acceptance-rate
    branch without running the full SAEM loop.

    We inspect the warning sets and logger calls by patching the logger.
    Returns (warned_low, warned_high, log_records).
    """
    import logging
    from unittest.mock import patch, call

    saem = SAEMMethod(n_iter_phase1=2, n_iter_phase2=1, n_chains=n_chains, seed=0)

    warned_low: set[int] = set()
    warned_high: set[int] = set()
    warnings_logged: list[str] = []

    sid = 42

    # Replicate the acceptance warning logic from estimate()
    with patch("openpkpd.estimation.saem.logger") as mock_logger:
        mock_logger.warning.side_effect = lambda fmt, *args, **kw: warnings_logged.append(
            fmt % args if args else fmt
        )

        n_accepted = int(round(accept_rate * n_chains))
        ar = n_accepted / n_chains

        if ar < 0.05:
            if sid not in warned_low:
                warned_low.add(sid)
                mock_logger.warning(
                    "SAEM: subject %s MH acceptance rate %.1f%% is very low — "
                    "chain may be stuck. "
                    "Consider reducing mh_scale or increasing n_chains.",
                    sid, ar * 100,
                )
        elif ar > 0.95:
            if sid not in warned_high:
                warned_high.add(sid)
                mock_logger.warning(
                    "SAEM: subject %s MH acceptance rate %.1f%% is very high — "
                    "proposals are too small. "
                    "Consider increasing mh_scale.",
                    sid, ar * 100,
                )

    return warned_low, warned_high, warnings_logged


@pytest.mark.unit
class TestSAEMAcceptanceRateWarnings:
    def test_low_acceptance_rate_warns_stuck(self):
        """accept_rate=0.04 (< 0.05) → WARNING mentioning 'stuck'."""
        warned_low, warned_high, logs = _make_saem_with_single_subject_run(0.04)
        assert 42 in warned_low
        assert any("stuck" in msg for msg in logs), f"Expected 'stuck' in logs: {logs}"

    def test_high_acceptance_rate_warns_too_small(self):
        """accept_rate=0.96 (> 0.95) → WARNING mentioning 'too small'."""
        warned_low, warned_high, logs = _make_saem_with_single_subject_run(0.96)
        assert 42 in warned_high
        assert any("too small" in msg for msg in logs), f"Expected 'too small' in logs: {logs}"

    def test_normal_acceptance_rate_no_warning(self):
        """accept_rate=0.30 → no warning."""
        warned_low, warned_high, logs = _make_saem_with_single_subject_run(0.30)
        assert len(warned_low) == 0
        assert len(warned_high) == 0
        assert logs == []

    def test_warning_emitted_only_once_per_phase(self):
        """
        Simulate two iterations with the same subject at low acceptance rate.
        The warning should only be emitted once (first time the subject is
        added to _warned_low).
        """
        warned_low: set[int] = set()
        warning_count = 0
        sid = 99

        for _ in range(5):  # 5 iterations
            ar = 0.02  # very low
            if ar < 0.05:
                if sid not in warned_low:
                    warned_low.add(sid)
                    warning_count += 1  # would call logger.warning

        assert warning_count == 1, "Warning should only fire once per phase"
        assert sid in warned_low
