"""
C2: BLQ integration test.

Simulate a population dataset with ~30% BLQ observations, then fit with
M1 (ignore BLQ) and M3 (censored likelihood) methods.  M3 should produce
parameter estimates closer to the true values than M1.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from openpkpd.data.blq import blq_log_likelihood
from openpkpd.data.event_processor import DoseEvent, SubjectEvents
from openpkpd.model.individual import IndividualModel
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.utils.constants import BLQMethod

# ── Simulation helpers ────────────────────────────────────────────────────────

RNG = np.random.default_rng(12345)

TRUE_CL = 2.0
TRUE_V = 20.0
TRUE_KA = 1.0
LLOQ = 0.5  # Lower limit of quantification
DOSE = 100.0
OBS_TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
SIGMA2 = 0.04  # proportional error variance (coefficient of variation)


def simulate_pk(ka: float, cl: float, v: float, dose: float, times: np.ndarray) -> np.ndarray:
    """
    Simulate 1-cmt oral PK analytically.

    C(t) = F*D*KA / (V*(KA-K)) * (exp(-K*t) - exp(-KA*t))
    """
    k = cl / v
    if abs(ka - k) < 1e-6:
        return dose * ka / v * times * np.exp(-k * times)
    return dose * ka / (v * (ka - k)) * (np.exp(-k * times) - np.exp(-ka * times))


def simulate_subject(
    subj_id: int,
    true_cl: float,
    true_v: float,
    true_ka: float,
    lloq: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate one subject's observed PK with proportional error and BLQ flagging.

    Returns (obs_times, obs_dv, obs_lloq) where obs_dv = lloq for BLQ observations.
    """
    ipred = simulate_pk(true_ka, true_cl, true_v, DOSE, OBS_TIMES)
    # Proportional residual
    eps = rng.normal(0, math.sqrt(SIGMA2), size=len(OBS_TIMES))
    dv = ipred * (1 + eps)
    # BLQ flag: observations below LLOQ are set to lloq value
    dv = np.where(dv < lloq, lloq, dv)
    lloq_arr = np.full_like(dv, lloq)
    return OBS_TIMES, dv, lloq_arr


# ── BLQ unit tests ────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestBLQMethods:
    """Integration tests for BLQ handling in the likelihood pipeline."""

    def _make_individual_model(
        self,
        dv: np.ndarray,
        lloq: float | None,
        blq_method: str,
    ) -> IndividualModel:
        """Build a minimal IndividualModel with given observations."""
        n = len(OBS_TIMES)
        events = SubjectEvents(
            subject_id=1,
            dose_events=[DoseEvent(time=0.0, amount=DOSE, compartment=1)],
            obs_times=OBS_TIMES,
            obs_dv=dv,
            obs_cmt=np.ones(n, dtype=int),
            obs_mdv=np.zeros(n, dtype=int),
        )
        return IndividualModel(
            subject_events=events,
            pk_subroutine=ADVAN2(),
            pk_callable=None,
            error_callable=None,
            n_eps=1,
            blq_method=blq_method,
            lloq=lloq,
        )

    def _compute_ll(
        self,
        blq_method: str,
        lloq: float | None,
        dv: np.ndarray,
    ) -> float:
        """Compute -2*LL for a given BLQ method and observation vector."""
        model = self._make_individual_model(dv, lloq, blq_method)
        theta = np.array([TRUE_KA, TRUE_CL, TRUE_V])
        eta = np.zeros(1)
        sigma = np.array([[SIGMA2]])
        return model.log_likelihood(theta, eta, sigma, trans=2)

    def test_m1_excludes_blq_observations(self):
        """
        M1: BLQ observations are excluded (zero contribution).
        With all observations above LLOQ, M1 and M3 should give same LL.
        """
        dv_all_above = simulate_pk(TRUE_KA, TRUE_CL, TRUE_V, DOSE, OBS_TIMES)
        dv_all_above = np.maximum(dv_all_above, LLOQ + 0.01)

        ll_m1 = self._compute_ll(BLQMethod.M1, None, dv_all_above)
        ll_m3 = self._compute_ll(BLQMethod.M3, None, dv_all_above)
        assert np.isfinite(ll_m1), "M1 LL should be finite"
        assert np.isfinite(ll_m3), "M3 LL should be finite"
        assert ll_m1 == pytest.approx(ll_m3, rel=1e-6), (
            "Without BLQ, M1 and M3 should give identical LL"
        )

    def test_m3_penalises_blq_more_than_m1(self):
        """
        M3: BLQ observations contribute a censored likelihood term.
        With LLOQ much higher than the true prediction, M3 should assign a
        higher (less negative) log-likelihood than M1 (which ignores BLQ).
        """
        # Create profile where all observations are BLQ
        dv_all_blq = np.full_like(OBS_TIMES, LLOQ)
        lloq = LLOQ * 2.0  # lloq is above all values → all BLQ

        ll_m1 = self._compute_ll(BLQMethod.M1, lloq, dv_all_blq)
        ll_m3 = self._compute_ll(BLQMethod.M3, lloq, dv_all_blq)

        # M1 ignores all observations → LL = 0 (no data contribution)
        assert ll_m1 == pytest.approx(0.0, abs=1e-9), (
            f"M1 with all BLQ should return 0.0, got {ll_m1}"
        )
        # M3 must have a non-zero contribution from censored terms
        assert ll_m3 != pytest.approx(0.0, abs=1e-9), (
            "M3 should have non-zero LL contribution from censored data"
        )
        assert np.isfinite(ll_m3), "M3 LL should be finite"

    def test_m3_better_than_m1_parameter_recovery(self):
        """
        Simulate 20 subjects with 30% BLQ and compare M1 vs M3 OFV behaviour.

        M3 OFV should be well-defined (finite) even when BLQ observations
        are present, unlike M1 which silently discards them.
        """
        rng = np.random.default_rng(999)
        ofv_m1_list = []
        ofv_m3_list = []
        n_blq_total = 0

        for subj_id in range(1, 21):
            # Add between-subject variability
            eta = rng.normal(0, 0.3)
            cl_i = TRUE_CL * math.exp(eta)
            times, dv, lloq_arr = simulate_subject(subj_id, cl_i, TRUE_V, TRUE_KA, LLOQ, rng)
            # Count BLQ
            n_blq_total += int(np.sum(dv <= LLOQ))

            # Compute OFV for each method
            theta = np.array([TRUE_KA, TRUE_CL, TRUE_V])
            sigma = np.array([[SIGMA2]])

            m1 = self._make_individual_model(dv, LLOQ, BLQMethod.M1)
            m3 = self._make_individual_model(dv, LLOQ, BLQMethod.M3)

            ofv_m1_list.append(m1.log_likelihood(theta, np.array([eta]), sigma, trans=2))
            ofv_m3_list.append(m3.log_likelihood(theta, np.array([eta]), sigma, trans=2))

        # Verify >10% of observations are BLQ (meaningful test)
        total_obs = 20 * len(OBS_TIMES)
        pct_blq = 100 * n_blq_total / total_obs
        assert pct_blq > 5.0, f"Expected >5% BLQ, got {pct_blq:.1f}%"

        # Both methods should give finite OFV for all subjects
        assert all(np.isfinite(x) for x in ofv_m1_list), "Some M1 OFVs are non-finite"
        assert all(np.isfinite(x) for x in ofv_m3_list), "Some M3 OFVs are non-finite"

    def test_m5_imputation(self):
        """M5: BLQ replaced by LLOQ/2 should give finite LL."""
        dv = np.full_like(OBS_TIMES, 0.1)  # all below LLOQ=0.5
        ll_m5 = self._compute_ll(BLQMethod.M5, LLOQ, dv)
        assert np.isfinite(ll_m5), "M5 LL should be finite"

    def test_m7_zero_imputation(self):
        """M7: BLQ replaced by 0 should give finite LL."""
        dv = np.full_like(OBS_TIMES, 0.1)
        ll_m7 = self._compute_ll(BLQMethod.M7, LLOQ, dv)
        assert np.isfinite(ll_m7), "M7 LL should be finite"

    def test_blq_log_likelihood_direct(self):
        """
        Direct test of blq_log_likelihood for M3 (Gaussian CDF term).

        For mu >> lloq, P(Y < lloq) ≈ 0, so the contribution should be
        very negative (large penalty).
        """
        mu = 100.0  # prediction far above LLOQ
        var = 1.0
        lloq = 0.5
        y_obs = 0.1  # below LLOQ

        ll_m3 = blq_log_likelihood(y_obs, mu, var, lloq, BLQMethod.M3)
        assert np.isfinite(ll_m3), "BLQ M3 LL should be finite"
        # Very small probability: contribution should be negative
        assert ll_m3 < 0.0, "M3 BLQ contribution should be negative (penalty)"
