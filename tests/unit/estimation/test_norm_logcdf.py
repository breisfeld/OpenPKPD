"""
R1: Tests for norm_logcdf accuracy in the deep tail (asymptotic expansion fix).

Tests use scipy.stats.norm.logcdf as reference values.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from scipy.stats import norm

# Try to import the Rust norm_logcdf via the BLQ M3 path
try:
    from openpkpd._native import import_core_symbol
    _neg2ll_loop = import_core_symbol("neg2ll_obs_loop")
    _RUST_AVAILABLE = True
except (ImportError, Exception):
    _RUST_AVAILABLE = False


def _logcdf_via_blq(z: float) -> float:
    """Compute log Φ(z) via the Rust neg2ll BLQ M3 path.

    For a single BLQ observation:
      M3 contrib = log Φ((lloq - mu) / sigma)
    We want z = (lloq - mu) / sigma.

    Strategy: set sigma=1, lloq = z + 1.0, mu = 1.0
      → (lloq - mu) / sigma = (z+1 - 1) / 1 = z

    The BLQ condition requires dv < lloq. Set dv = z (which is < z+1 = lloq
    for all finite z, since z < z+1).
    The neg2ll = -2 * ll, so ll = -neg2ll/2.
    """
    lloq_val = z + 1.0    # lloq
    mu = 1.0              # so (lloq - mu) / sigma = z
    sigma_sq = 1.0
    dv_val = z - 1.0      # guaranteed < lloq

    dv = np.array([dv_val], dtype=np.float64)
    pred = np.array([mu], dtype=np.float64)
    var = np.array([sigma_sq], dtype=np.float64)
    obs_mask = np.array([True])
    lloq = np.array([lloq_val], dtype=np.float64)
    blq_method = np.uint8(3)  # M3

    neg2ll = _neg2ll_loop(dv, pred, var, obs_mask, lloq, blq_method)
    return -neg2ll / 2.0


# Skip all tests if Rust extension not available
pytestmark = pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust extension not available")


# ── Test 1: Reference values from scipy ──────────────────────────────────────

@pytest.mark.parametrize("z, expected, atol", [
    (-1.0,  norm.logcdf(-1.0),  1e-4),    # -1.8411
    (-3.0,  norm.logcdf(-3.0),  1e-4),    # -5.6341
    (-5.0,  norm.logcdf(-5.0),  1e-4),    # -9.7961
    (-10.0, norm.logcdf(-10.0), 0.01),    # -53.268
])
def test_norm_logcdf_reference_values(z, expected, atol):
    """norm_logcdf should match scipy.stats.norm.logcdf to specified tolerance."""
    result = _logcdf_via_blq(z)
    assert abs(result - expected) < atol, (
        f"norm_logcdf({z}): got {result:.4f}, expected {expected:.4f}, "
        f"abs error = {abs(result - expected):.4e}"
    )


def test_norm_logcdf_deep_tail_z_neg30():
    """norm_logcdf(-30) should match scipy to within 1.0 (old clamp was -690 vs correct ~-453)."""
    z = -30.0
    expected = norm.logcdf(z)  # ≈ -453.49
    result = _logcdf_via_blq(z)
    assert abs(result - expected) < 1.0, (
        f"norm_logcdf(-30): got {result:.2f}, expected {expected:.2f} "
        f"(old clamp of 1e-300 would give {math.log(1e-300):.1f})"
    )


def test_norm_logcdf_deep_tail_z_neg50():
    """norm_logcdf(-50) should be finite and within 0.1 relative error of scipy."""
    z = -50.0
    expected = norm.logcdf(z)
    result = _logcdf_via_blq(z)
    assert math.isfinite(result), "norm_logcdf(-50) should be finite"
    assert expected != 0
    rel_err = abs(result - expected) / abs(expected)
    assert rel_err < 0.1, (
        f"norm_logcdf(-50): got {result:.2f}, expected {expected:.2f}, "
        f"relative error = {rel_err:.3f}"
    )


# ── Test 2: Boundary ──────────────────────────────────────────────────────────

def test_norm_logcdf_zero():
    """norm_logcdf(0) = ln(0.5) ≈ -0.6931."""
    result = _logcdf_via_blq(0.0)
    expected = math.log(0.5)
    assert abs(result - expected) < 1e-4, f"norm_logcdf(0): got {result:.4f}, expected {expected:.4f}"


# ── Test 3: Deep tail is finite ───────────────────────────────────────────────

def test_norm_logcdf_z_neg100_is_finite():
    """norm_logcdf(-100) must be finite (old clamp of 1e-300 gives ln(1e-300) = -690, wrong value)."""
    result = _logcdf_via_blq(-100.0)
    assert math.isfinite(result), f"norm_logcdf(-100) should be finite, got {result}"
    # The asymptotic value is approximately -0.5*100^2 - 0.5*ln(2π) - ln(100) ≈ -5005
    assert result < -1000, f"norm_logcdf(-100) should be very negative, got {result}"


# ── Test 4: BLQ M3 path numerical check ──────────────────────────────────────

def test_blq_m3_matches_scipy_for_moderate_lloq():
    """BLQ M3 log-likelihood should match scipy.stats.norm.logcdf formula."""
    # Observation far below LLOQ: lloq=0.1, ipred=1.0, sigma=sqrt(0.04)=0.2
    # z = (lloq - ipred) / sigma = (0.1 - 1.0) / 0.2 = -4.5
    lloq = 0.1
    ipred = 1.0
    sigma_sq = 0.04  # variance
    sigma = math.sqrt(sigma_sq)
    z = (lloq - ipred) / sigma  # -4.5

    expected_logcdf = norm.logcdf(z)  # scipy reference

    dv = np.array([0.0], dtype=np.float64)
    pred = np.array([ipred], dtype=np.float64)
    var = np.array([sigma_sq], dtype=np.float64)
    obs_mask = np.array([True])
    lloq_arr = np.array([lloq], dtype=np.float64)
    blq_code = np.uint8(3)

    neg2ll = _neg2ll_loop(dv, pred, var, obs_mask, lloq_arr, blq_code)
    ll = -neg2ll / 2.0

    assert abs(ll - expected_logcdf) < 1e-4, (
        f"BLQ M3 LL: got {ll:.6f}, expected {expected_logcdf:.6f}"
    )
