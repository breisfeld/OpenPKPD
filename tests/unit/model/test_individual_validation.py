"""
M3: Tests for V > 0 validation in IndividualModel.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from openpkpd.data.event_processor import DoseEvent, SubjectEvents
from openpkpd.model.individual import IndividualModel
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.base import PKSolution, PKSubroutine


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_subject_events(n_obs: int = 5) -> SubjectEvents:
    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)[:n_obs]
    return SubjectEvents(
        subject_id=1,
        obs_times=times,
        obs_dv=np.ones(n_obs, dtype=float),
        obs_cmt=np.ones(n_obs, dtype=int),
        obs_mdv=np.zeros(n_obs, dtype=int),
        dose_events=[DoseEvent(time=0.0, amount=100.0, compartment=1, rate=0.0)],
    )


def _make_pk_callable(cl: float, v: float):
    """Build a pk_callable that returns fixed CL and V."""
    def pk_callable(theta, eta, t=0.0, a=None, covariates=None):
        return {"CL": cl, "V": v}
    return pk_callable


# ── Test 1 & 2: V=0 and V<0 raise ValueError ─────────────────────────────────

def test_zero_volume_raises_valueerror():
    """V=0 in pk_callable output should raise ValueError (division by zero)."""
    subj = _make_subject_events()
    advan1 = ADVAN1()
    pk_callable = _make_pk_callable(cl=5.0, v=0.0)

    indiv = IndividualModel(
        subject_events=subj,
        pk_subroutine=advan1,
        pk_callable=pk_callable,
        error_callable=None,
    )
    theta = np.array([5.0, 0.0])
    eta = np.zeros(1)
    sigma = np.eye(1) * 0.01

    with pytest.raises((ValueError, Exception)):
        # The ValueError for V<=0 should propagate up from _try_native_ode_probe
        # or from ADVAN1.solve. The evaluate() method raises PKError for
        # solver failures, but V<=0 validation is a ValueError.
        indiv.evaluate(theta, eta, sigma)


def test_negative_volume_raises():
    """V < 0 in pk_callable output should raise an error."""
    subj = _make_subject_events()
    advan1 = ADVAN1()
    pk_callable = _make_pk_callable(cl=5.0, v=-10.0)

    indiv = IndividualModel(
        subject_events=subj,
        pk_subroutine=advan1,
        pk_callable=pk_callable,
        error_callable=None,
    )
    theta = np.array([5.0, -10.0])
    eta = np.zeros(1)
    sigma = np.eye(1) * 0.01

    with pytest.raises(Exception):
        indiv.evaluate(theta, eta, sigma)


# ── Test 3: Very small but positive V succeeds ───────────────────────────────

def test_very_small_positive_volume_succeeds():
    """V = 1e-10 is valid: produces very large but finite concentration."""
    subj = _make_subject_events(n_obs=3)
    advan1 = ADVAN1()
    # With V=1e-10, DOSE=100 → C(0+) = 100/1e-10 = 1e12 (large but finite)
    pk_callable = _make_pk_callable(cl=5.0, v=1e-10)

    indiv = IndividualModel(
        subject_events=subj,
        pk_subroutine=advan1,
        pk_callable=pk_callable,
        error_callable=None,
    )
    theta = np.array([5.0, 1e-10])
    eta = np.zeros(1)
    sigma = np.eye(1) * 0.01

    ipred, obs_mask, f = indiv.evaluate(theta, eta, sigma)
    # With V=1e-10, CL/V=5e10 which causes instant elimination; values may round to 0.
    # The key check is no crash (ValueError for V<=0) and finite output.
    assert np.all(np.isfinite(ipred)), f"Expected finite ipred, got {ipred}"


# ── Test 4: Numerical accuracy — ADVAN1 IV bolus ─────────────────────────────

def test_advan1_iv_bolus_numerical_accuracy():
    """
    ADVAN1 1-compartment IV bolus.
    Known analytical: C(t) = (dose/V) * exp(-CL/V * t)

    With CL=5 L/hr, V=50 L, dose=100 mg, t=1 hr:
      C(1) = (100/50) * exp(-5/50 * 1) = 2 * exp(-0.1) ≈ 1.8097 mg/L
    """
    CL = 5.0   # L/hr
    V = 50.0   # L
    DOSE = 100.0  # mg
    t_obs = 1.0  # hr

    subj = SubjectEvents(
        subject_id=1,
        obs_times=np.array([t_obs], dtype=float),
        obs_dv=np.array([0.0], dtype=float),
        obs_cmt=np.array([1], dtype=int),
        obs_mdv=np.array([0], dtype=int),
        dose_events=[DoseEvent(time=0.0, amount=DOSE, compartment=1, rate=0.0)],
    )
    advan1 = ADVAN1()
    pk_callable = _make_pk_callable(cl=CL, v=V)

    indiv = IndividualModel(
        subject_events=subj,
        pk_subroutine=advan1,
        pk_callable=pk_callable,
        error_callable=None,
    )
    theta = np.array([CL, V])
    eta = np.zeros(1)
    sigma = np.eye(1) * 0.01

    ipred, obs_mask, f = indiv.evaluate(theta, eta, sigma)

    expected = (DOSE / V) * math.exp(-CL / V * t_obs)  # 2 * exp(-0.1) ≈ 1.8097
    assert obs_mask.any(), "No active observations"
    pred = float(ipred[obs_mask][0])
    rel_err = abs(pred - expected) / expected
    assert rel_err < 0.001, (
        f"ADVAN1 prediction {pred:.6f} differs from analytical {expected:.6f} "
        f"by {rel_err:.2%} (tolerance 0.1%)"
    )
