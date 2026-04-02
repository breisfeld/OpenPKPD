"""
PA1 & PA4: Tests for ParameterSet log-transform and logit round-trips.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from openpkpd.model.parameters import ParameterSet, ThetaSpec, OmegaSpec, SigmaSpec


def _make_param_set(theta_specs, theta_vals):
    """Build a ParameterSet with the given theta specs and values."""
    omega = np.eye(1) * 0.04
    sigma = np.eye(1) * 0.01
    omega_specs = [OmegaSpec(block_size=1, values=[0.04])]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]
    ps = ParameterSet(
        theta=np.array(theta_vals, dtype=float),
        omega=omega,
        sigma=sigma,
        theta_specs=theta_specs,
        omega_specs=omega_specs,
        sigma_specs=sigma_specs,
    )
    return ps


# ── PA1: Log-transform round-trip ─────────────────────────────────────────────

@pytest.mark.parametrize("val", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
def test_log_transform_round_trip(val):
    """to_vector then from_vector should recover original theta to < 1e-10."""
    specs = [ThetaSpec(init=val, lower=0.0, label="CL")]
    ps = _make_param_set(specs, [val])

    vec = ps.to_vector()
    ps2 = ParameterSet.from_vector(vec, ps)

    recovered = float(ps2.theta[0])
    assert abs(recovered - val) / val < 1e-10, (
        f"Round-trip failed for val={val}: got {recovered}"
    )


def test_log_transform_no_neg_inf():
    """to_vector with val near zero should not produce -inf."""
    specs = [ThetaSpec(init=1e-8, lower=0.0, label="CL")]
    ps = _make_param_set(specs, [1e-8])

    vec = ps.to_vector()
    assert np.all(np.isfinite(vec)), f"to_vector produced non-finite values: {vec}"


def test_log_transform_log1():
    """to_vector([1.0]) for a log-transform theta should equal [0.0] = log(1)."""
    specs = [ThetaSpec(init=1.0, lower=0.0, label="V")]
    ps = _make_param_set(specs, [1.0])

    vec = ps.to_vector()
    theta_part = vec[0]  # first element is log(theta[0])
    assert abs(theta_part - 0.0) < 1e-12, f"log(1.0) should be 0.0, got {theta_part}"


def test_log_transform_log_e():
    """to_vector([e]) for a log-transform theta should equal [1.0] = log(e)."""
    e = math.e
    specs = [ThetaSpec(init=e, lower=0.0, label="V")]
    ps = _make_param_set(specs, [e])

    vec = ps.to_vector()
    theta_part = vec[0]
    assert abs(theta_part - 1.0) < 1e-12, f"log(e) should be 1.0, got {theta_part}"


# ── PA4: Logit round-trip ─────────────────────────────────────────────────────

@pytest.mark.parametrize("val", np.linspace(0.12, 0.88, 20).tolist())
def test_logit_round_trip_bounded(val):
    """to_vector then from_vector for a bounded theta should recover value to < 1e-10."""
    lower, upper = 0.1, 0.9
    specs = [ThetaSpec(init=val, lower=lower, upper=upper, label="F1")]
    ps = _make_param_set(specs, [val])

    vec = ps.to_vector()
    ps2 = ParameterSet.from_vector(vec, ps)
    recovered = float(ps2.theta[0])

    assert abs(recovered - val) < 1e-10, (
        f"Logit round-trip failed for val={val:.4f}: got {recovered:.15f}"
    )


def test_logit_at_lower_bound_stays_in_range():
    """Val exactly at lower bound after round-trip must be within [lower, upper]."""
    lower, upper = 0.1, 0.9
    val = lower + 1e-9  # just above lower (lower itself would raise on ThetaSpec)
    specs = [ThetaSpec(init=val, lower=lower, upper=upper, label="F1")]
    ps = _make_param_set(specs, [val])

    vec = ps.to_vector()
    ps2 = ParameterSet.from_vector(vec, ps)
    recovered = float(ps2.theta[0])

    assert lower <= recovered <= upper, (
        f"Round-trip value {recovered} is outside [{lower}, {upper}]"
    )


def test_logit_at_upper_bound_stays_in_range():
    """Val near upper bound after round-trip must be <= upper."""
    lower, upper = 0.1, 0.9
    val = upper - 1e-9  # just below upper
    specs = [ThetaSpec(init=val, lower=lower, upper=upper, label="F1")]
    ps = _make_param_set(specs, [val])

    vec = ps.to_vector()
    ps2 = ParameterSet.from_vector(vec, ps)
    recovered = float(ps2.theta[0])

    assert recovered <= upper, f"Round-trip value {recovered} exceeds upper bound {upper}"
    assert recovered >= lower, f"Round-trip value {recovered} is below lower bound {lower}"


def test_logit_midpoint_value():
    """For lower=0.1, upper=0.9, val=0.5 (midpoint):
    to_vector should give logit((0.5-0.1)/(0.9-0.1)) = logit(0.5) = 0.0.
    from_vector([0.0]) should give 0.5 to within 1e-12.
    """
    lower, upper = 0.1, 0.9
    val = 0.5
    specs = [ThetaSpec(init=val, lower=lower, upper=upper, label="F1")]
    ps = _make_param_set(specs, [val])

    vec = ps.to_vector()
    # logit((0.5-0.1)/(0.9-0.1)) = logit(0.5) = log(0.5/0.5) = log(1) = 0.0
    expected_transformed = 0.0
    assert abs(vec[0] - expected_transformed) < 1e-12, (
        f"to_vector midpoint: got {vec[0]:.6e}, expected {expected_transformed}"
    )

    ps2 = ParameterSet.from_vector(np.array([0.0, vec[1], vec[2]]), ps)
    recovered = float(ps2.theta[0])
    assert abs(recovered - 0.5) < 1e-12, (
        f"from_vector([0.0]) should give 0.5, got {recovered:.15f}"
    )


def test_logit_near_upper_bound_stays_bounded():
    """Val = upper - 1e-12 must round-trip to <= upper."""
    lower, upper = 0.1, 0.9
    val = upper - 1e-12
    # ThetaSpec validates lower <= init <= upper, so use val that satisfies this
    val = upper - 1e-9
    specs = [ThetaSpec(init=val, lower=lower, upper=upper, label="F1")]
    ps = _make_param_set(specs, [val])

    vec = ps.to_vector()
    ps2 = ParameterSet.from_vector(vec, ps)
    recovered = float(ps2.theta[0])

    assert recovered <= upper, f"Near-upper round-trip {recovered} exceeds {upper}"
