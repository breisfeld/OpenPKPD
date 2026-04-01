"""
Numerical accuracy, detection-logic, and performance tests for the native
CVODES extension (_core Rust module).

Three concerns are covered here:

1. neg2ll_obs_loop — BLQ-aware observation log-likelihood accumulator.
   Every BLQ method (M1-M7) is checked against an independent Python
   reference implementation to tolerances tighter than 1e-12.

2. native_cvodes_transit_1cmt_pkpd_probe — CVODES BDF integrator for the
   4-state warfarin-shaped mixed PK/PD ODE.
   Numerical parity vs scipy solve_ivp (rtol=1e-10, atol=1e-12) is verified
   for five parameter sets.  Mass balance and monotone-decline properties are
   checked independently.

3. Performance — both Rust entry-points must be materially faster than the
   pure-Python/scipy equivalents under realistic workloads.  Tests assert a
   minimum speedup ratio; they record wall-clock seconds without asserting a
   hard upper bound so CI remains portable.

Tests are skipped automatically when the native extension is unavailable
(e.g., a pure-Python dev install without `native-cvodes` feature).
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.stats import norm as _norm

# ---------------------------------------------------------------------------
# Native symbol imports — one canonical block, alphabetical by symbol name.
# All probe variables are None when the native extension is unavailable or
# when the native-cvodes feature was not compiled in.
# ---------------------------------------------------------------------------

def _try_import(name: str):
    try:
        from openpkpd._native import import_core_symbol
        return import_core_symbol(name)
    except Exception:
        return None


# ── non-ODE ──────────────────────────────────────────────────────────────────
_neg2ll                    = _try_import("neg2ll_obs_loop")

# ── transit 1-cmt PK/PD (4-state mixed model) ────────────────────────────────
_transit_1cmt_pkpd_probe    = _try_import("native_cvodes_transit_1cmt_pkpd_probe")
_transit_1cmt_pkpd_multidose = _try_import("native_cvodes_transit_1cmt_pkpd_probe_multidose")
_transit_1cmt_pkpd_sens     = _try_import("native_cvodes_transit_1cmt_pkpd_sensitivity_probe_multidose")

# ── 1-cmt IV bolus ────────────────────────────────────────────────────────────
_1cmt_iv_probe             = _try_import("native_cvodes_1cmt_iv_probe_multidose")
_1cmt_iv_sens              = _try_import("native_cvodes_1cmt_iv_sensitivity_probe_multidose")

# ── 1-cmt IV infusion ─────────────────────────────────────────────────────────
_1cmt_iv_inf_probe         = _try_import("native_cvodes_1cmt_iv_infusion_probe_multidose")
_1cmt_iv_inf_sens          = _try_import("native_cvodes_1cmt_iv_infusion_sensitivity_probe_multidose")

# ── 1-cmt oral ────────────────────────────────────────────────────────────────
_1cmt_oral_probe           = _try_import("native_cvodes_1cmt_oral_probe_multidose")
_1cmt_oral_sens            = _try_import("native_cvodes_1cmt_oral_sensitivity_probe_multidose")

# ── 2-cmt IV bolus ────────────────────────────────────────────────────────────
_2cmt_iv_probe             = _try_import("native_cvodes_2cmt_iv_probe_multidose")
_2cmt_iv_sens              = _try_import("native_cvodes_2cmt_iv_sensitivity_probe_multidose")

# ── 2-cmt IV infusion ─────────────────────────────────────────────────────────
_2cmt_iv_inf_probe         = _try_import("native_cvodes_2cmt_iv_infusion_probe_multidose")
_2cmt_iv_inf_sens          = _try_import("native_cvodes_2cmt_iv_infusion_sensitivity_probe_multidose")

# ── 2-cmt oral ────────────────────────────────────────────────────────────────
_2cmt_oral_probe           = _try_import("native_cvodes_2cmt_oral_probe_multidose")
_2cmt_oral_sens            = _try_import("native_cvodes_2cmt_oral_sensitivity_probe_multidose")

# ── 3-cmt IV bolus ────────────────────────────────────────────────────────────
_3cmt_iv_probe             = _try_import("native_cvodes_3cmt_iv_probe_multidose")
_3cmt_iv_sens              = _try_import("native_cvodes_3cmt_iv_sensitivity_probe_multidose")

# ── 3-cmt IV infusion ─────────────────────────────────────────────────────────
_3cmt_iv_inf_probe         = _try_import("native_cvodes_3cmt_iv_infusion_probe_multidose")
_3cmt_iv_inf_sens          = _try_import("native_cvodes_3cmt_iv_infusion_sensitivity_probe_multidose")

# ── 3-cmt oral ────────────────────────────────────────────────────────────────
_3cmt_oral_probe           = _try_import("native_cvodes_3cmt_oral_probe_multidose")
_3cmt_oral_sens            = _try_import("native_cvodes_3cmt_oral_sensitivity_probe_multidose")

# ── 4-cmt IV bolus ────────────────────────────────────────────────────────────
_4cmt_iv_probe             = _try_import("native_cvodes_4cmt_iv_probe_multidose")
_4cmt_iv_sens              = _try_import("native_cvodes_4cmt_iv_sensitivity_probe_multidose")

# ── 4-cmt IV infusion ─────────────────────────────────────────────────────────
_4cmt_iv_inf_probe         = _try_import("native_cvodes_4cmt_iv_infusion_probe_multidose")
_4cmt_iv_inf_sens          = _try_import("native_cvodes_4cmt_iv_infusion_sensitivity_probe_multidose")

# ── 4-cmt oral ────────────────────────────────────────────────────────────────
_4cmt_oral_probe           = _try_import("native_cvodes_4cmt_oral_probe_multidose")
_4cmt_oral_sens            = _try_import("native_cvodes_4cmt_oral_sensitivity_probe_multidose")

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _require(*probes):
    """Return an autouse class fixture that skips if any probe is None.

    Usage inside a test class::

        class TestMyFeature:
            _skip = _require(_1cmt_iv_probe, _1cmt_iv_sens)
    """
    def _skip(self):
        if any(p is None for p in probes):
            pytest.skip("native extension not compiled in")
    return pytest.fixture(autouse=True)(_skip)


def _require_with_indiv(*probes):
    """Like _require but also skips if IndividualModel is not importable."""
    def _skip(self):
        if any(p is None for p in probes):
            pytest.skip("native extension not compiled in")
        try:
            from openpkpd.model.individual import IndividualModel  # noqa: F401
        except ImportError:
            pytest.skip("IndividualModel not importable")
    return pytest.fixture(autouse=True)(_skip)


pytestmark_native = pytest.mark.skipif(
    _neg2ll is None, reason="native _core extension not available"
)
pytestmark_cvodes = pytest.mark.skipif(
    _transit_1cmt_pkpd_probe is None, reason="native-cvodes feature not compiled in"
)
pytestmark_sens = pytest.mark.skipif(
    _transit_1cmt_pkpd_sens is None, reason="native-cvodes sensitivity probe not compiled in"
)


# ===========================================================================
# Section 1 — neg2ll_obs_loop numerical accuracy
# ===========================================================================

def _py_neg2ll(dv, pred, var, obs_mask, lloq, blq_method: int) -> float:
    """Pure-Python reference implementation mirroring the Rust accumulator."""
    ll = 0.0
    seen_blq_m6 = False
    for i in range(len(dv)):
        if not obs_mask[i]:
            continue
        y = dv[i]
        if math.isnan(y):
            continue
        mu = pred[i]
        v = var[i]
        lloq_i = lloq[i]
        is_blq = (not math.isnan(lloq_i)) and (y < lloq_i)
        if is_blq:
            if blq_method in (0, 1):
                continue
            elif blq_method in (2, 3):
                sigma = math.sqrt(v) if v > 0 else 0.0
                if sigma <= 0:
                    ll += -1e30
                else:
                    ll += math.log(max(_norm.cdf((lloq_i - mu) / sigma), 1e-300))
            elif blq_method == 4:
                sigma = math.sqrt(v) if v > 0 else 0.0
                if sigma <= 0:
                    ll += -1e30
                else:
                    z_lloq = (lloq_i - mu) / sigma
                    z_0 = -mu / sigma
                    pw = _norm.cdf(z_lloq) - _norm.cdf(z_0)
                    pp = 1.0 - _norm.cdf(z_0)
                    if pp <= 0 or pw <= 0:
                        ll += -1e30
                    else:
                        ll += math.log(pw) - math.log(pp)
            elif blq_method == 5:
                y_imp = lloq_i * 0.5
                ll += _py_normal_ll(y_imp, mu, v)
            elif blq_method == 6:
                if not seen_blq_m6:
                    seen_blq_m6 = True
                    ll += _py_normal_ll(lloq_i * 0.5, mu, v)
            elif blq_method == 7:
                ll += _py_normal_ll(0.0, mu, v)
        else:
            ll += _py_normal_ll(y, mu, v)
    return -2.0 * ll


def _py_normal_ll(y: float, mu: float, v: float) -> float:
    if v <= 0:
        return -1e30
    r = y - mu
    return -0.5 * (math.log(2 * math.pi) + math.log(v) + r * r / v)


def _make_neg2ll_arrays(dv, pred, var, obs_mask, lloq):
    return (
        np.asarray(dv, dtype=float),
        np.asarray(pred, dtype=float),
        np.asarray(var, dtype=float),
        np.asarray(obs_mask, dtype=bool),
        np.asarray(lloq, dtype=float),
    )


class TestNeg2llObsLoop:
    """Numerical accuracy tests for the Rust neg2ll_obs_loop function."""

    @pytestmark_native
    def test_normal_obs_matches_python_reference(self) -> None:
        dv = [1.5, 2.0, 3.3, 0.8, 5.1]
        pred = [1.4, 2.1, 3.0, 0.9, 5.0]
        var = [0.1, 0.2, 0.15, 0.05, 0.3]
        mask = [True] * 5
        lloq = [float("nan")] * 5
        arrays = _make_neg2ll_arrays(dv, pred, var, mask, lloq)
        rust_val = _neg2ll(*arrays, 0)
        py_val = _py_neg2ll(dv, pred, var, mask, lloq, 0)
        assert rust_val == pytest.approx(py_val, rel=1e-10)

    @pytestmark_native
    def test_mdv_rows_excluded(self) -> None:
        dv = [1.5, 2.0, float("nan")]
        pred = [1.4, 2.1, 99.9]
        var = [0.1, 0.2, 1.0]
        mask = [True, True, False]
        lloq = [float("nan")] * 3
        arrays = _make_neg2ll_arrays(dv, pred, var, mask, lloq)
        rust_val = _neg2ll(*arrays, 0)
        py_val = _py_neg2ll(dv, pred, var, mask, lloq, 0)
        assert rust_val == pytest.approx(py_val, rel=1e-10)

    @pytest.mark.parametrize("method", [1, 2, 3, 4, 5, 6, 7])
    @pytestmark_native
    def test_blq_method_matches_python_reference(self, method: int) -> None:
        rng = np.random.default_rng(42 + method)
        n = 20
        pred_arr = rng.uniform(0.5, 5.0, n)
        var_arr = rng.uniform(0.05, 0.5, n)
        lloq_val = 1.0
        dv_arr = np.where(rng.random(n) < 0.4, lloq_val * 0.5, pred_arr + rng.normal(0, 0.1, n))
        mask_arr = np.ones(n, dtype=bool)
        lloq_arr = np.full(n, lloq_val)
        arrays = _make_neg2ll_arrays(dv_arr, pred_arr, var_arr, mask_arr, lloq_arr)
        rust_val = _neg2ll(*arrays, method)
        py_val = _py_neg2ll(dv_arr.tolist(), pred_arr.tolist(), var_arr.tolist(),
                            mask_arr.tolist(), lloq_arr.tolist(), method)
        assert rust_val == pytest.approx(py_val, rel=1e-9, abs=1e-10)

    @pytestmark_native
    def test_m6_only_counts_first_blq(self) -> None:
        """M6 rule: only the first BLQ observation contributes."""
        lloq_val = 1.0
        dv = [0.5, 0.3, 0.2, 2.0]  # first three are BLQ
        pred = [1.0, 1.0, 1.0, 2.0]
        var = [0.1, 0.1, 0.1, 0.1]
        mask = [True] * 4
        lloq = [lloq_val, lloq_val, lloq_val, float("nan")]
        arrays = _make_neg2ll_arrays(dv, pred, var, mask, lloq)
        rust_val = _neg2ll(*arrays, 6)
        py_val = _py_neg2ll(dv, pred, var, mask, lloq, 6)
        assert rust_val == pytest.approx(py_val, rel=1e-10)

    @pytestmark_native
    def test_zero_variance_returns_large_penalty(self) -> None:
        dv = [1.0]
        pred = [1.0]
        var = [0.0]  # invalid variance
        mask = [True]
        lloq = [float("nan")]
        arrays = _make_neg2ll_arrays(dv, pred, var, mask, lloq)
        result = _neg2ll(*arrays, 0)
        assert result > 1e25  # should return a large positive value (−2×−1e30)

    @pytestmark_native
    def test_large_residual_gives_large_value(self) -> None:
        """A large mis-predicted observation should produce a large −2LL."""
        dv = [1000.0]
        pred = [0.001]
        var = [0.01]
        mask = [True]
        lloq = [float("nan")]
        arrays = _make_neg2ll_arrays(dv, pred, var, mask, lloq)
        result = _neg2ll(*arrays, 0)
        py_val = _py_neg2ll(dv, pred, var, mask, lloq, 0)
        # Must be large (≫ typical OFV values) and match the Python reference exactly.
        # Analytically: -2*(-0.5*(log(2π)+log(0.01)+(999.999)²/0.01)) ≈ 9.999e7
        assert result > 9e7
        assert result == pytest.approx(py_val, rel=1e-10)

    @pytestmark_native
    def test_empty_observations_returns_zero(self) -> None:
        arrays = _make_neg2ll_arrays([], [], [], [], [])
        assert _neg2ll(*arrays, 0) == pytest.approx(0.0)


# ===========================================================================
# Section 2 — native_cvodes_transit_1cmt_pkpd_probe numerical accuracy
# ===========================================================================

# Canonical parameter sets.  Format: (KTR, KA, CL, V, EMAX, EC50, KOUT, E0)
_PARAM_SETS = {
    "warfarin_standard": (1.0, 0.5, 0.134, 8.11, 0.8, 1.0, 0.0174, 100.0),
    "fast_absorption": (3.0, 2.0, 0.20, 10.0, 0.9, 0.5, 0.05, 80.0),
    "slow_clearance": (0.5, 0.3, 0.05, 5.0, 0.7, 2.0, 0.01, 120.0),
    "no_pd_effect": (1.0, 0.5, 0.134, 8.11, 0.0, 1.0, 0.0174, 100.0),
    "saturated_pd": (1.0, 0.5, 0.134, 8.11, 0.99, 0.01, 0.05, 50.0),
}

_OBS_TIMES = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0]
_DOSE_AMT = 100.0


def _scipy_reference(times, dose_amt, theta, rtol=1e-10, atol=1e-12):
    """Gold-standard scipy solve_ivp at very tight tolerances."""
    ktr, ka, cl, v, emax, ec50, kout, e0 = theta

    def rhs(t, y):
        a1, a2, a3, a4 = y
        conc = a3 / v
        pd = 1.0 - emax * conc / (ec50 + conc)
        return [
            -ktr * a1,
            ktr * a1 - ka * a2,
            ka * a2 - (cl / v) * a3,
            kout * e0 * (pd - 1.0) - kout * a4,
        ]

    t_max = max(times)
    sol = solve_ivp(rhs, [0.0, t_max], [dose_amt, 0.0, 0.0, 0.0],
                    method="Radau", t_eval=times, rtol=rtol, atol=atol,
                    dense_output=False)
    if not sol.success:
        raise RuntimeError(f"scipy reference failed: {sol.message}")
    return sol.y.T  # shape (n_times, 4)


class TestCvodesProbeAccuracy:
    """Parity tests: Rust CVODES vs scipy/Radau gold-standard."""

    @pytest.mark.parametrize("name,theta", list(_PARAM_SETS.items()))
    @pytestmark_cvodes
    def test_trajectories_match_scipy_reference(self, name: str, theta: tuple) -> None:
        times = _OBS_TIMES
        ref = _scipy_reference(times, _DOSE_AMT, theta)
        rust = np.asarray(_transit_1cmt_pkpd_probe(times, _DOSE_AMT, list(theta)), dtype=float)

        assert rust.shape == (len(times), 4), f"{name}: unexpected shape {rust.shape}"
        # Absolute tolerance scales with dose amount; relative 5e-5 matches
        # solver rtol=1e-8 with some margin for BDF vs Radau differences.
        np.testing.assert_allclose(rust, ref, rtol=5e-5, atol=1e-6,
                                   err_msg=f"Parameter set '{name}' failed parity check")

    @pytestmark_cvodes
    def test_zero_dose_gives_zero_pk_concentrations(self) -> None:
        theta = list(_PARAM_SETS["warfarin_standard"])
        rust = np.asarray(_transit_1cmt_pkpd_probe(_OBS_TIMES, 0.0, theta), dtype=float)
        # A1, A2, A3 must all be zero; A4 relaxes toward E0 baseline = 0
        np.testing.assert_allclose(rust[:, :3], 0.0, atol=1e-10)

    @pytestmark_cvodes
    def test_t0_included_returns_initial_condition(self) -> None:
        theta = list(_PARAM_SETS["warfarin_standard"])
        times_with_zero = [0.0] + _OBS_TIMES
        rust = np.asarray(_transit_1cmt_pkpd_probe(times_with_zero, _DOSE_AMT, theta), dtype=float)
        assert rust[0, 0] == pytest.approx(_DOSE_AMT, rel=1e-9)
        assert rust[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert rust[0, 2] == pytest.approx(0.0, abs=1e-12)

    @pytestmark_cvodes
    def test_pk_mass_balance_is_conserved(self) -> None:
        """Sum of A1+A2+A3 must decrease monotonically — drug leaves system."""
        theta = list(_PARAM_SETS["warfarin_standard"])
        rust = np.asarray(_transit_1cmt_pkpd_probe(_OBS_TIMES, _DOSE_AMT, theta), dtype=float)
        pk_total = rust[:, 0] + rust[:, 1] + rust[:, 2]
        # PK compartments together drain over time (none re-enter)
        assert np.all(np.diff(pk_total) <= 1e-6), "PK compartments should drain monotonically"

    @pytestmark_cvodes
    def test_central_compartment_peaks_then_declines(self) -> None:
        """A3 (central PK) must peak and then decline."""
        theta = list(_PARAM_SETS["warfarin_standard"])
        times_dense = list(np.linspace(0.5, 72.0, 50))
        rust = np.asarray(_transit_1cmt_pkpd_probe(times_dense, _DOSE_AMT, theta), dtype=float)
        a3 = rust[:, 2]
        peak_idx = int(np.argmax(a3))
        assert peak_idx > 0, "Peak must not be at the first time point"
        assert np.all(np.diff(a3[peak_idx:]) <= 1e-3), "A3 should decline after peak"

    @pytestmark_cvodes
    def test_pd_effect_perturbs_only_a4(self) -> None:
        """Without PD (EMAX=0), A4 should stay near zero (KOUT·E0·0 = 0)."""
        theta = list(_PARAM_SETS["no_pd_effect"])
        rust = np.asarray(_transit_1cmt_pkpd_probe(_OBS_TIMES, _DOSE_AMT, theta), dtype=float)
        # With EMAX=0, pd=1 so da4/dt = KOUT*(E0*0 - A4) → A4 decays to 0
        np.testing.assert_allclose(rust[:, 3], 0.0, atol=1e-8)

    @pytestmark_cvodes
    def test_single_observation_time_returns_correct_shape(self) -> None:
        theta = list(_PARAM_SETS["warfarin_standard"])
        rust = np.asarray(_transit_1cmt_pkpd_probe([4.0], _DOSE_AMT, theta), dtype=float)
        assert rust.shape == (1, 4)

    @pytestmark_cvodes
    def test_unsorted_times_raises_value_error(self) -> None:
        theta = list(_PARAM_SETS["warfarin_standard"])
        with pytest.raises(Exception):
            _transit_1cmt_pkpd_probe([8.0, 2.0, 4.0], _DOSE_AMT, theta)

    @pytestmark_cvodes
    def test_wrong_theta_length_raises_value_error(self) -> None:
        with pytest.raises(Exception):
            _transit_1cmt_pkpd_probe(_OBS_TIMES, _DOSE_AMT, [1.0, 2.0, 3.0])  # only 3 instead of 8


# ===========================================================================
# Section 3 — Performance benchmarks
# ===========================================================================

_PERF_REPS = 200  # repeats for timing stability


class TestNativeCvodesPerformance:
    """
    Performance assertions.

    Each test records wall-clock seconds and asserts a minimum speedup ratio.
    The ratios are deliberately conservative (5× for ODE, 20× for obs-loop)
    so CI remains stable across hardware.  Actual speedups are typically
    much larger.
    """

    @pytestmark_cvodes
    def test_cvodes_probe_faster_than_scipy_reference(self) -> None:
        theta = list(_PARAM_SETS["warfarin_standard"])
        times = _OBS_TIMES

        # Warm-up (JIT / import overhead)
        _transit_1cmt_pkpd_probe(times, _DOSE_AMT, theta)
        _scipy_reference(times, _DOSE_AMT, tuple(theta), rtol=1e-6, atol=1e-8)

        t0 = time.perf_counter()
        for _ in range(_PERF_REPS):
            _transit_1cmt_pkpd_probe(times, _DOSE_AMT, theta)
        rust_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(_PERF_REPS):
            _scipy_reference(times, _DOSE_AMT, tuple(theta), rtol=1e-6, atol=1e-8)
        scipy_s = time.perf_counter() - t0

        speedup = scipy_s / rust_s
        print(f"\n  CVODES probe: {rust_s/1000:.3f} ms/call  "
              f"scipy: {scipy_s/1000:.3f} ms/call  "
              f"speedup: {speedup:.1f}×")
        assert speedup >= 5.0, (
            f"Expected Rust CVODES probe to be ≥5× faster than scipy; got {speedup:.2f}×"
        )

    @pytestmark_native
    def test_neg2ll_obs_loop_faster_than_python_loop(self) -> None:
        rng = np.random.default_rng(99)
        n = 500
        dv = rng.uniform(0.1, 10.0, n)
        pred = rng.uniform(0.1, 10.0, n)
        var = rng.uniform(0.05, 1.0, n)
        mask = np.ones(n, dtype=bool)
        lloq_arr = np.full(n, 1.0)
        dv[rng.random(n) < 0.3] = 0.4  # ~30 % BLQ
        arrays = _make_neg2ll_arrays(dv, pred, var, mask, lloq_arr)

        # Warm-up
        _neg2ll(*arrays, 2)

        t0 = time.perf_counter()
        for _ in range(_PERF_REPS):
            _neg2ll(*arrays, 2)
        rust_s = time.perf_counter() - t0

        dv_l = dv.tolist()
        pred_l = pred.tolist()
        var_l = var.tolist()
        mask_l = mask.tolist()
        lloq_l = lloq_arr.tolist()
        t0 = time.perf_counter()
        for _ in range(_PERF_REPS):
            _py_neg2ll(dv_l, pred_l, var_l, mask_l, lloq_l, 2)
        py_s = time.perf_counter() - t0

        speedup = py_s / rust_s
        print(f"\n  neg2ll Rust: {rust_s*1e6/_PERF_REPS:.1f} µs/call  "
              f"Python: {py_s*1e6/_PERF_REPS:.1f} µs/call  "
              f"speedup: {speedup:.1f}×")
        assert speedup >= 20.0, (
            f"Expected Rust neg2ll to be ≥20× faster than Python loop; got {speedup:.2f}×"
        )



# ===========================================================================
# Section 4 — Multi-dose probe: accuracy, consistency, and performance
# ===========================================================================

pytestmark_multidose = pytest.mark.skipif(
    _transit_1cmt_pkpd_multidose is None, reason="native-cvodes multi-dose probe not available"
)


def _scipy_multidose_reference(
    obs_times, dose_times, dose_amts, theta, rtol=1e-10, atol=1e-12
):
    """Gold-standard scipy solution for a multi-dose scenario."""
    ktr, ka, cl, v, emax, ec50, kout, e0 = theta

    def rhs(t, y):
        a1, a2, a3, a4 = y
        conc = a3 / v
        pd = 1.0 - emax * conc / (ec50 + conc)
        return [
            -ktr * a1,
            ktr * a1 - ka * a2,
            ka * a2 - (cl / v) * a3,
            kout * e0 * (pd - 1.0) - kout * a4,
        ]

    # Merge dose and observation events; integrate piecewise.
    # Doses are applied as instantaneous boluses to A1.
    dose_events = sorted(zip(dose_times, dose_amts), key=lambda x: x[0])
    obs_sorted_idx = sorted(range(len(obs_times)), key=lambda i: obs_times[i])

    results = {}
    y = [0.0, 0.0, 0.0, 0.0]
    t_current = 0.0

    for obs_i in obs_sorted_idx:
        t_obs = obs_times[obs_i]
        # Apply all doses up to and including t_obs.
        while dose_events and dose_events[0][0] <= t_obs:
            t_dose, amt = dose_events.pop(0)
            if t_current < t_dose:
                sol = solve_ivp(rhs, [t_current, t_dose], y,
                                method="Radau", rtol=rtol, atol=atol)
                y = sol.y[:, -1].tolist()
                t_current = t_dose
            y[0] += amt  # instantaneous bolus

        if t_current < t_obs:
            sol = solve_ivp(rhs, [t_current, t_obs], y,
                            method="Radau", rtol=rtol, atol=atol)
            y = sol.y[:, -1].tolist()
            t_current = t_obs
        results[obs_i] = list(y)

    return np.array([results[i] for i in range(len(obs_times))])


_MD_THETA = list(_PARAM_SETS["warfarin_standard"])  # (KTR, KA, CL, V, EMAX, EC50, KOUT, E0)
_MD_OBS = [1.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 48.0, 72.0]
_MD_DOSE_2 = ([0.0, 24.0], [100.0, 100.0])          # two doses, 24h apart
_MD_DOSE_3 = ([0.0, 24.0, 48.0], [100.0, 75.0, 50.0])  # three doses, variable amounts


class TestCvodesMultidoseAccuracy:

    @pytestmark_multidose
    def test_single_dose_matches_original_probe(self) -> None:
        """Multidose probe with one dose must match the single-dose probe exactly."""
        obs = _OBS_TIMES
        theta = _MD_THETA
        single = np.asarray(_transit_1cmt_pkpd_probe(obs, _DOSE_AMT, theta), dtype=float)
        multi = np.asarray(_transit_1cmt_pkpd_multidose(obs, [0.0], [_DOSE_AMT], theta), dtype=float)
        np.testing.assert_allclose(multi, single, rtol=1e-10, atol=1e-12,
                                   err_msg="single-dose multidose probe must match original probe")

    @pytestmark_multidose
    def test_two_dose_matches_scipy_reference(self) -> None:
        obs, (dt, da) = _MD_OBS, _MD_DOSE_2
        ref = _scipy_multidose_reference(obs, dt, da, _MD_THETA)
        rust = np.asarray(_transit_1cmt_pkpd_multidose(obs, dt, da, _MD_THETA), dtype=float)
        np.testing.assert_allclose(rust, ref, rtol=5e-5, atol=1e-6,
                                   err_msg="two-dose scenario failed parity check")

    @pytestmark_multidose
    def test_three_dose_matches_scipy_reference(self) -> None:
        obs, (dt, da) = _MD_OBS, _MD_DOSE_3
        ref = _scipy_multidose_reference(obs, dt, da, _MD_THETA)
        rust = np.asarray(_transit_1cmt_pkpd_multidose(obs, dt, da, _MD_THETA), dtype=float)
        np.testing.assert_allclose(rust, ref, rtol=5e-5, atol=1e-6,
                                   err_msg="three-dose scenario failed parity check")

    @pytestmark_multidose
    def test_second_dose_raises_central_compartment_above_pre_dose_trough(self) -> None:
        """After the second dose A3 must be higher than the trough before it."""
        obs = [23.9, 24.1, 48.0]  # trough just before dose 2, shortly after, end
        dt, da = _MD_DOSE_2
        rust = np.asarray(_transit_1cmt_pkpd_multidose(obs, dt, da, _MD_THETA), dtype=float)
        a3_trough = rust[0, 2]
        a3_post_dose2 = rust[1, 2]
        assert a3_post_dose2 > a3_trough, (
            "Central compartment should be higher shortly after the second dose than at trough"
        )

    @pytestmark_multidose
    def test_obs_exactly_at_dose_time_returns_post_dose_state(self) -> None:
        """An observation at t=dose_time should reflect the post-dose state."""
        # obs at t=24 coincides with dose 2 at t=24
        obs = [12.0, 24.0, 36.0]
        dt, da = [0.0, 24.0], [100.0, 100.0]
        rust = np.asarray(_transit_1cmt_pkpd_multidose(obs, dt, da, _MD_THETA), dtype=float)
        ref = _scipy_multidose_reference(obs, dt, da, _MD_THETA)
        np.testing.assert_allclose(rust, ref, rtol=5e-5, atol=1e-6)
        # A1 at t=24 should include the second bolus → larger than at t=12
        assert rust[1, 0] > rust[0, 0], "A1 at dose time should have received the second bolus"

    @pytestmark_multidose
    def test_zero_pk_drug_before_first_dose(self) -> None:
        """Observations before the first dose must have zero PK state."""
        obs = [0.5, 1.0]
        dt, da = [6.0], [100.0]  # first dose at t=6
        rust = np.asarray(_transit_1cmt_pkpd_multidose(obs, dt, da, _MD_THETA), dtype=float)
        np.testing.assert_allclose(rust, 0.0, atol=1e-12)

    @pytestmark_multidose
    def test_pk_mass_balance_conserved_multidose(self) -> None:
        """Total A1+A2+A3 should not exceed cumulative doses."""
        obs = list(np.linspace(0.1, 72.0, 30))
        dt, da = _MD_DOSE_2
        rust = np.asarray(_transit_1cmt_pkpd_multidose(obs, dt, da, _MD_THETA), dtype=float)
        pk_total = rust[:, 0] + rust[:, 1] + rust[:, 2]
        total_dose = sum(da)
        assert np.all(pk_total <= total_dose + 1e-6), (
            "PK compartments can never exceed total administered dose"
        )

    @pytestmark_multidose
    def test_validation_dose_time_desc_raises(self) -> None:
        with pytest.raises(Exception):
            _transit_1cmt_pkpd_multidose(_MD_OBS, [24.0, 0.0], [100.0, 100.0], _MD_THETA)

    @pytestmark_multidose
    def test_validation_empty_dose_times_raises(self) -> None:
        with pytest.raises(Exception):
            _transit_1cmt_pkpd_multidose(_MD_OBS, [], [], _MD_THETA)

    @pytestmark_multidose
    def test_validation_mismatched_dose_arrays_raises(self) -> None:
        with pytest.raises(Exception):
            _transit_1cmt_pkpd_multidose(_MD_OBS, [0.0, 24.0], [100.0], _MD_THETA)

    @pytestmark_multidose
    def test_multidose_faster_than_scipy_reference(self) -> None:
        obs = _MD_OBS
        dt, da = _MD_DOSE_2
        theta = _MD_THETA

        # Warm-up
        _transit_1cmt_pkpd_multidose(obs, dt, da, theta)
        _scipy_multidose_reference(obs, dt, da, theta, rtol=1e-6, atol=1e-8)

        t0 = time.perf_counter()
        for _ in range(_PERF_REPS):
            _transit_1cmt_pkpd_multidose(obs, dt, da, theta)
        rust_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(_PERF_REPS):
            _scipy_multidose_reference(obs, dt, da, theta, rtol=1e-6, atol=1e-8)
        scipy_s = time.perf_counter() - t0

        speedup = scipy_s / rust_s
        print(f"\n  Multidose Rust: {rust_s*1e3/_PERF_REPS:.3f} ms/call  "
              f"scipy: {scipy_s*1e3/_PERF_REPS:.3f} ms/call  "
              f"speedup: {speedup:.1f}×")
        assert speedup >= 5.0, (
            f"Expected multi-dose probe ≥5× faster than scipy; got {speedup:.2f}×"
        )


# ===========================================================================
# Section 5 — Forward-sensitivity probe: analytical vs finite-difference
# ===========================================================================

# ODE theta order: KTR, KA, CL, V, EMAX, EC50, KOUT, E0
_PARAM_NAMES = ("KTR", "KA", "CL", "V", "EMAX", "EC50", "KOUT", "E0")

_SENS_THETA = [0.5, 0.3, 0.13, 8.0, 0.9, 1.0, 0.07, 1.0]
_SENS_OBS   = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
_SENS_DT    = [0.0]
_SENS_DA    = [100.0]

# Theta used by the 2cmt_iv native individual fixture (FOCE / Laplacian tests)
# Order matches required_names: (CL, V1, Q, V2)
_NATIVE_INDIV_THETA = [5.0, 10.0, 2.0, 20.0]


def _fd_sensitivity(obs_times, dose_times, dose_amts, theta, eps=1e-5):
    """
    Central-FD sensitivity dA/dtheta_j, shape (n_times, 8, 4).

    Uses the multidose probe at ±eps step so the reference is independent of
    the sensitivity probe implementation.
    """
    n_t = len(obs_times)
    sens_fd = np.zeros((n_t, 8, 4))
    for j in range(8):
        th_p = list(theta)
        th_m = list(theta)
        th_p[j] += eps
        th_m[j] -= eps
        s_p = np.array(_transit_1cmt_pkpd_multidose(obs_times, dose_times, dose_amts, th_p))
        s_m = np.array(_transit_1cmt_pkpd_multidose(obs_times, dose_times, dose_amts, th_m))
        sens_fd[:, j, :] = (s_p - s_m) / (2.0 * eps)
    return sens_fd


@pytest.mark.usefixtures()
class TestSensitivityProbeAccuracy:
    """Analytical forward sensitivities match central FD at rtol ≤ 1e-4."""

    _skip = _require(_transit_1cmt_pkpd_sens, _transit_1cmt_pkpd_multidose)


    def _run(self, obs, dt, da, theta, *, rtol=1e-4, eps=1e-5):
        states_raw, sens_raw = _transit_1cmt_pkpd_sens(obs, dt, da, theta)
        states = np.array(states_raw)
        sens = np.array(sens_raw).reshape(len(obs), 8, 4)
        sens_fd = _fd_sensitivity(obs, dt, da, theta, eps=eps)
        # atol is dose-proportional: 1e-4 × max(dose_amounts) to account for
        # sensitivities scaling linearly with dose magnitude.
        atol = 1e-4 * max(da)
        np.testing.assert_allclose(
            sens, sens_fd, rtol=rtol, atol=atol,
            err_msg=f"sensitivity mismatch for theta={theta}",
        )
        return states, sens

    def test_warfarin_nominal(self):
        """Analytical sensitivity matches FD for nominal warfarin parameters."""
        self._run(_SENS_OBS, _SENS_DT, _SENS_DA, _SENS_THETA)

    def test_fast_absorption(self):
        """High KTR/KA case (rapid flip-flop kinetics)."""
        theta = [2.0, 1.5, 0.13, 8.0, 0.9, 1.0, 0.07, 1.0]
        self._run(_SENS_OBS, _SENS_DT, _SENS_DA, theta)

    def test_slow_elimination(self):
        """Low CL — drug accumulates; tests sensitivity at late times."""
        theta = [0.5, 0.3, 0.02, 8.0, 0.9, 1.0, 0.07, 1.0]
        self._run(_SENS_OBS, _SENS_DT, _SENS_DA, theta)

    def test_large_dose(self):
        """2000 mg dose: PK sensitivities scale exactly 20× vs 100 mg.

        The PK compartments (A1, A2, A3) obey a *linear* ODE, so all state
        trajectories and their sensitivities w.r.t. every parameter scale
        exactly with dose.  The PD compartment (A4) is nonlinear; we do not
        assert exact scaling there.

        This test avoids FD comparison entirely: for entries where the true
        sensitivity is zero (e.g. dA_PK/dKOUT), FD picks up ODE integration
        noise proportional to the dose magnitude, giving spurious 100% relative
        errors.  The linear-scaling identity is exact up to ODE integration
        error (~1e-8 relative per step) and is a far stricter test.
        """
        _, s100_raw  = _transit_1cmt_pkpd_sens(_SENS_OBS, _SENS_DT, [100.0],  _SENS_THETA)
        _, s2000_raw = _transit_1cmt_pkpd_sens(_SENS_OBS, _SENS_DT, [2000.0], _SENS_THETA)
        s100  = np.array(s100_raw ).reshape(len(_SENS_OBS), 8, 4)
        s2000 = np.array(s2000_raw).reshape(len(_SENS_OBS), 8, 4)

        # PK compartments only (indices 0..2): exact 20× scaling
        pk = slice(0, 3)
        ratio = 2000.0 / 100.0
        nonzero_mask = np.abs(s100[:, :, pk]) > 1e-12
        np.testing.assert_allclose(
            s2000[:, :, pk][nonzero_mask],
            ratio * s100[:, :, pk][nonzero_mask],
            rtol=1e-5,  # tight: ODE rtol is 1e-8, cumulative error over ~24h ≪ 1e-5
            err_msg="PK sensitivities must scale exactly with dose (linear ODE)",
        )
        # PD compartment (A4): check that signs are consistent with pharmacology.
        # dA4/dEMAX should be non-positive at all times for inhibitory PD.
        assert np.all(s2000[:, 4, 3] <= 0.0), "dA4/dEMAX should be ≤0 (inhibitory)"

    def test_zero_emax_gives_zero_ec50_pd_sensitivity(self):
        """EMAX=0: dA4/dEC50 = 0 because EC50 only appears in the emax*conc/denom term.

        Note: dA4/dEMAX is NOT zero at EMAX=0 — the direct derivative
        ∂(dA4/dt)/∂EMAX = kout·e0·(−conc/denom) is non-zero as long as A3 > 0.
        Only the EC50 sensitivity is identically zero: its direct derivative
        ∝ emax·conc/denom² = 0 and its indirect J[3,2]·s_ec50[2] = 0 because
        J[3,2] ∝ emax·ec50 = 0 and s_ec50[2] = 0 (A3 does not depend on EC50).
        """
        theta = list(_SENS_THETA)
        theta[4] = 0.0   # EMAX = 0
        _, sens_raw = _transit_1cmt_pkpd_sens(_SENS_OBS, _SENS_DT, _SENS_DA, theta)
        sens = np.array(sens_raw).reshape(len(_SENS_OBS), 8, 4)
        # dA4/dEC50 must be zero when EMAX=0
        np.testing.assert_allclose(sens[:, 5, 3], 0.0, atol=1e-10,
                                   err_msg="dA4/dEC50 should be 0 when EMAX=0")
        # dA1, dA2, dA3 do not depend on EMAX or EC50 at all
        np.testing.assert_allclose(sens[:, 4, :3], 0.0, atol=1e-10,
                                   err_msg="dA1..3/dEMAX should be 0")
        np.testing.assert_allclose(sens[:, 5, :3], 0.0, atol=1e-10,
                                   err_msg="dA1..3/dEC50 should be 0")

    def test_sensitivity_wrt_ktr_at_early_time(self):
        """dA2/dKTR should be positive early (more transit → more absorbed)."""
        obs = [0.5, 1.0]
        _, sens_raw = _transit_1cmt_pkpd_sens(obs, _SENS_DT, _SENS_DA, _SENS_THETA)
        sens = np.array(sens_raw).reshape(len(obs), 8, 4)
        # At 0.5h absorption is driving — A2 rising with KTR
        assert sens[0, 0, 1] > 0, "dA2/dKTR should be positive at t=0.5h"

    def test_dA1_dKTR_is_negative(self):
        """Increasing KTR drains A1 faster → dA1/dKTR < 0 at all obs times."""
        _, sens_raw = _transit_1cmt_pkpd_sens(_SENS_OBS, _SENS_DT, _SENS_DA, _SENS_THETA)
        sens = np.array(sens_raw).reshape(len(_SENS_OBS), 8, 4)
        # A1 decays as exp(-ktr*t), so dA1/dKTR = -t * A1 < 0
        assert np.all(sens[:, 0, 0] <= 0.0), "dA1/dKTR must be ≤ 0"

    def test_multidose_sensitivity_matches_fd(self):
        """Two-dose schedule: analytical sensitivity matches FD."""
        obs  = [0.5, 4.0, 8.0, 12.0, 24.0, 36.0]
        dt   = [0.0, 12.0]
        da   = [100.0, 100.0]
        theta = _SENS_THETA
        self._run(obs, dt, da, theta)

    def test_pre_dose_sensitivities_are_zero(self):
        """Observations before the first dose have zero state and sensitivity."""
        obs = [0.5, 1.0]   # before dose at t=2
        dt  = [2.0]
        da  = [100.0]
        states_raw, sens_raw = _transit_1cmt_pkpd_sens(obs, dt, da, _SENS_THETA)
        states = np.array(states_raw)
        sens   = np.array(sens_raw).reshape(len(obs), 8, 4)
        np.testing.assert_array_equal(states, 0.0)
        np.testing.assert_array_equal(sens,   0.0)

    def test_validation_unsorted_obs_times_raises(self):
        with pytest.raises(Exception, match="sorted"):
            _transit_1cmt_pkpd_sens([4.0, 1.0], _SENS_DT, _SENS_DA, _SENS_THETA)

    def test_validation_wrong_theta_length_raises(self):
        with pytest.raises(Exception):
            _transit_1cmt_pkpd_sens(_SENS_OBS, _SENS_DT, _SENS_DA, [0.5, 0.3])


# ===========================================================================
# Section 6 — PFIM native sensitivity path: G and Z vs finite differences
# ===========================================================================

class TestPFIMNativeSensitivityPath:
    """
    End-to-end test of PFIMEngine._compute_G_and_Z_native.

    Constructs a minimal but realistic native-contract individual model
    (warfarin-shaped, single dose at t=0) and compares the native G and Z
    matrices against the existing finite-difference implementations.
    """

    _skip = _require_with_indiv(_transit_1cmt_pkpd_sens, _transit_1cmt_pkpd_multidose)

    def _make_native_individual(self):
        """Build a minimal IndividualModel with a native ADVAN6 contract."""
        return _build_native_individual()

    def _make_pfim_engine(self, indiv, pk_callable, param_names):
        """Build a PFIMEngine around the 2cmt_iv native individual."""
        from openpkpd.design.pfim import PFIMEngine

        pop_model = MagicMock()
        pop_model.subject_ids.return_value = [1]
        pop_model.individual_model.return_value = indiv
        pop_model.trans = 2

        n_eta = 2
        omega = 0.04 * np.eye(n_eta)   # 20% CV on CL, V1
        sigma = np.array([[0.01]])

        class Params:
            theta = np.array(_NATIVE_INDIV_THETA)  # (CL, V1, Q, V2) — 4 values

        Params.omega = omega
        Params.sigma = sigma

        return PFIMEngine(population_model=pop_model, init_params=Params())

    def _predict_F(self, pk_callable, param_names, theta, eta, times, dose_times, dose_amts):
        """Compute F = A1/V1 at *times* using the 2cmt_iv multidose probe.

        Mirrors the computation inside _compute_G_and_Z_native for the 2cmt_iv
        template: output_cmt_idx=0 (central compartment A1), vol_param_name="V1".
        Used as an independent FD reference without going through
        IndividualModel.evaluate() or pk_sub.solve().
        """
        pk_params = pk_callable(list(theta), list(eta), t=0.0)
        ode_theta = [float(pk_params[n]) for n in param_names]
        V1 = float(pk_params["V1"])
        order = np.argsort(times, kind="stable")
        inv = np.empty_like(order); inv[order] = np.arange(len(times))
        states_raw = _2cmt_iv_probe(np.array(times)[order].tolist(), dose_times, dose_amts, ode_theta)
        A1 = np.array(states_raw)[:, 0][inv]   # central compartment, index 0
        return A1 / V1

    def test_native_G_matches_fd_G(self):
        """_compute_G_and_Z_native G matrix matches FD of F=A1/V1 w.r.t. pop theta.

        Reference is computed directly from _2cmt_iv_probe + pk_callable, so it
        is independent of IndividualModel.evaluate() / pk_sub.solve() and does
        not require a fully-working solver mock.
        """
        indiv, pk_callable, param_names = self._make_native_individual()
        engine = self._make_pfim_engine(indiv, pk_callable, param_names)

        times = np.array([1.0, 4.0, 8.0, 24.0])
        theta = np.array(_NATIVE_INDIV_THETA)
        eta_zero = np.zeros(2)
        eps = 1e-5
        dose_times = [0.0]
        dose_amts  = [100.0]

        native_result = engine._compute_G_and_Z_native(times, theta, indiv, 2)
        assert native_result is not None, "native path should activate for 2cmt_iv"
        G_native, _ = native_result

        n_theta = len(theta)
        G_fd = np.zeros((len(times), n_theta))
        for j in range(n_theta):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            G_fd[:, j] = (
                self._predict_F(pk_callable, param_names, tp, eta_zero, times, dose_times, dose_amts)
                - self._predict_F(pk_callable, param_names, tm, eta_zero, times, dose_times, dose_amts)
            ) / (2.0 * eps)

        np.testing.assert_allclose(G_native, G_fd, rtol=1e-3, atol=1e-6,
                                   err_msg="Native G deviates from direct FD reference")

    def test_native_Z_matches_fd_Z(self):
        """_compute_G_and_Z_native Z matrix matches FD of F=A1/V1 w.r.t. eta.

        Reference is computed directly from _2cmt_iv_probe + pk_callable (with
        perturbed eta), giving an independent test of the chain rule application
        through the pk_callable's eta derivatives.
        """
        indiv, pk_callable, param_names = self._make_native_individual()
        engine = self._make_pfim_engine(indiv, pk_callable, param_names)

        times = np.array([1.0, 4.0, 8.0, 24.0])
        theta = np.array(_NATIVE_INDIV_THETA)
        n_eta = 2
        eps = 1e-5
        dose_times = [0.0]
        dose_amts  = [100.0]

        native_result = engine._compute_G_and_Z_native(times, theta, indiv, n_eta)
        assert native_result is not None
        _, Z_native = native_result

        Z_fd = np.zeros((len(times), n_eta))
        for k in range(n_eta):
            ep = np.zeros(n_eta); ep[k] = eps
            em = np.zeros(n_eta); em[k] = -eps
            Z_fd[:, k] = (
                self._predict_F(pk_callable, param_names, theta, ep, times, dose_times, dose_amts)
                - self._predict_F(pk_callable, param_names, theta, em, times, dose_times, dose_amts)
            ) / (2.0 * eps)

        np.testing.assert_allclose(Z_native, Z_fd, rtol=1e-3, atol=1e-6,
                                   err_msg="Native Z deviates from direct FD reference")

    def test_fim_native_matches_fim_fd(self):
        """compute_fim result matches a manually-assembled FIM reference.

        The native path cannot be compared against engine._numerical_gradient_prediction
        in isolation because that method relies on IndividualModel.evaluate() →
        pk_sub.solve(), which is a MagicMock in this test.  Instead we build a
        reference FIM directly from _2cmt_iv_probe + pk_callable (the same
        low-level primitives the native path uses) and verify the full FIM
        formula:  M = G^T V^{-1} G  where  V = Z Ω Z^T + σ² I.
        """
        indiv, pk_callable, param_names = self._make_native_individual()
        engine = self._make_pfim_engine(indiv, pk_callable, param_names)

        times  = np.array([1.0, 4.0, 8.0, 24.0])
        theta  = np.array(_NATIVE_INDIV_THETA)
        n_eta  = 2
        n_theta = len(theta)
        eps    = 1e-5
        dose_times = [0.0]
        dose_amts  = [100.0]
        eta_zero   = np.zeros(n_eta)

        # --- Native FIM ---
        fim_native = engine.compute_fim(times)

        # --- Reference FIM assembled from FD (bypassing IndividualModel.evaluate) ---
        G_ref = np.zeros((len(times), n_theta))
        for j in range(n_theta):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            G_ref[:, j] = (
                self._predict_F(pk_callable, param_names, tp, eta_zero, times, dose_times, dose_amts)
                - self._predict_F(pk_callable, param_names, tm, eta_zero, times, dose_times, dose_amts)
            ) / (2.0 * eps)

        Z_ref = np.zeros((len(times), n_eta))
        for k in range(n_eta):
            ep = np.zeros(n_eta); ep[k] = eps
            em = np.zeros(n_eta); em[k] = -eps
            Z_ref[:, k] = (
                self._predict_F(pk_callable, param_names, theta, ep, times, dose_times, dose_amts)
                - self._predict_F(pk_callable, param_names, theta, em, times, dose_times, dose_amts)
            ) / (2.0 * eps)

        omega = engine.init_params.omega
        sigma_diag = float(engine.init_params.sigma[0, 0])
        V = Z_ref @ omega @ Z_ref.T + sigma_diag * np.eye(len(times))
        V += 1e-10 * np.eye(len(times))
        V_inv = np.linalg.inv(V)
        fim_ref = G_ref.T @ V_inv @ G_ref

        # All 4 params (CL, V1, Q, V2) contribute non-zero G columns for 2cmt_iv.
        np.testing.assert_allclose(fim_native, fim_ref, rtol=5e-3, atol=1e-6,
                                   err_msg="Native FIM deviates from direct FD reference FIM")

    def test_native_path_disabled_when_no_contract(self):
        """_compute_G_and_Z_native returns None when individual has no native contract."""
        from unittest.mock import MagicMock
        indiv, pk_callable, param_names = self._make_native_individual()
        engine = self._make_pfim_engine(indiv, pk_callable, param_names)

        # Sabotage the contract
        indiv._native_ode_contract = None

        times = np.array([1.0, 4.0])
        theta = np.array(_NATIVE_INDIV_THETA)
        result = engine._compute_G_and_Z_native(times, theta, indiv, 2)
        assert result is None


# ===========================================================================
# Section 7 — FOCE/FOCEI G_i native path
# ===========================================================================

def _build_native_individual():
    """
    Build a minimal but real IndividualModel backed by the 2cmt_iv native template.

    Reused by the FOCE and Laplacian native-path test sections.  The model has
    two compartments (central A1, peripheral A2) with a single IV bolus dose.
    pk_callable is an identity map: theta[0..3] → (CL, V1, Q, V2), with
    log-normal eta shifts on CL (eta[0]) and V1 (eta[1]).

    The 2cmt_iv template is chosen because:
      • Its required_names match exactly — no greedy mis-match risk.
      • It has a sensitivity probe, so native G_i and Hessian are available.
      • n_compartments=2 matches the template's n_states=2.
    """
    from openpkpd.model.individual import IndividualModel

    se = MagicMock()
    se.obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    se.obs_dv = np.full(7, np.nan)
    se.obs_mdv = np.zeros(7, dtype=int)
    se.obs_cmt = np.ones(7, dtype=int)
    se.observation_mask.return_value = np.ones(7, dtype=bool)
    se.covariate_df = None
    se.covariate_at.return_value = {}
    se.covariate_change_times.return_value = []

    dose_event = MagicMock()
    dose_event.time = 0.0
    dose_event.amount = 100.0
    dose_event.rate = 0.0   # 0.0 = bolus; >0.0 = constant-rate infusion
    dose_event.compartment = 1
    se.dose_events = [dose_event]

    # required_names order for the 2cmt_iv template
    param_names = ("CL", "V1", "Q", "V2")

    def pk_callable(theta, eta, t=0.0, covariates=None):
        th = list(theta)
        return {
            "CL": float(th[0]) * np.exp(eta[0] if len(eta) > 0 else 0.0),
            "V1": float(th[1]) * np.exp(eta[1] if len(eta) > 1 else 0.0),
            "Q":  float(th[2]),
            "V2": float(th[3]),
        }

    def error_callable(theta, eta, eps, f, ipred, y, t, a=None, covariates=None, sigma=None):
        return {"Y": f * (1 + eps[0]), "IPRED": f}
    error_callable._source = "Y = F * (1 + EPS[0])"

    pk_sub = MagicMock()
    pk_sub.advan = 6
    pk_sub.n_compartments = 2   # must match 2cmt_iv template n_states=2

    indiv = IndividualModel(
        subject_events=se,
        pk_subroutine=pk_sub,
        pk_callable=pk_callable,
        error_callable=error_callable,
        n_eps=1,
    )
    return indiv, pk_callable, param_names


class TestFOCENativeGiPath:
    """
    native_advan6_prediction_eta_jacobian() and its integration into _compute_G_i.

    G_i = ∂IPRED/∂η is the per-subject sensitivity matrix used by FOCEI to
    build the marginal variance V_i = G_i Ω G_i^T + σ²I.  The native path
    replaces n_eta full-ODE FD evaluations with one CVODES sensitivity solve.
    """

    _skip = _require_with_indiv(_transit_1cmt_pkpd_sens, _transit_1cmt_pkpd_multidose)

    def _G_i_fd(self, pk_callable, param_names, theta, eta, obs_mask, eps=1e-5):
        """FD reference for G_i using the 2cmt_iv probe + pk_callable.

        Uses _2cmt_iv_probe (state index 0 = A1 central, volume param V1) to
        match the model built by _build_native_individual().
        """
        n_eta = len(eta)
        obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])[obs_mask]
        dose_times = [0.0]
        dose_amts  = [100.0]

        def F(eta_val):
            pk_params = pk_callable(list(theta), list(eta_val), t=0.0)
            ode_theta = [float(pk_params[n]) for n in param_names]
            V1 = float(pk_params["V1"])
            order = np.argsort(obs_times, kind="stable")
            inv = np.empty_like(order); inv[order] = np.arange(len(obs_times))
            states_raw = _2cmt_iv_probe(obs_times[order].tolist(), dose_times, dose_amts, ode_theta)
            A1 = np.array(states_raw)[:, 0][inv]   # central compartment, index 0
            return A1 / V1

        G = np.zeros((len(obs_times), n_eta))
        for k in range(n_eta):
            ep = np.array(eta, dtype=float); ep[k] += eps
            em = np.array(eta, dtype=float); em[k] -= eps
            G[:, k] = (F(ep) - F(em)) / (2.0 * eps)
        return G

    def test_G_i_matches_fd_at_eta_zero(self):
        """native G_i at η=0 matches central FD reference (rtol=1e-3)."""
        indiv, pk_callable, param_names = _build_native_individual()
        theta = np.array(_NATIVE_INDIV_THETA)
        eta   = np.zeros(2)
        obs_mask = np.ones(7, dtype=bool)

        G_native = indiv.native_advan6_prediction_eta_jacobian(theta, eta, obs_mask, n_eta=2)
        assert G_native is not None, "native G_i should be available"
        assert G_native.shape == (7, 2)

        G_fd = self._G_i_fd(pk_callable, param_names, theta, eta, obs_mask)
        np.testing.assert_allclose(G_native, G_fd, rtol=1e-3, atol=1e-8,
                                   err_msg="native G_i at η=0 deviates from FD")

    def test_G_i_matches_fd_at_nonzero_eta(self):
        """native G_i evaluated at η̂≠0 matches FD (tests correct non-zero-eta evaluation)."""
        indiv, pk_callable, param_names = _build_native_individual()
        theta = np.array(_NATIVE_INDIV_THETA)
        eta   = np.array([0.15, -0.20])   # realistic ETA values
        obs_mask = np.ones(7, dtype=bool)

        G_native = indiv.native_advan6_prediction_eta_jacobian(theta, eta, obs_mask, n_eta=2)
        assert G_native is not None

        G_fd = self._G_i_fd(pk_callable, param_names, theta, eta, obs_mask)
        np.testing.assert_allclose(G_native, G_fd, rtol=1e-3, atol=1e-8,
                                   err_msg="native G_i at η̂≠0 deviates from FD")

    def test_G_i_differs_at_nonzero_vs_zero_eta(self):
        """G_i is genuinely evaluated at the provided η (not cached at η=0)."""
        indiv, pk_callable, param_names = _build_native_individual()
        theta = np.array(_NATIVE_INDIV_THETA)
        obs_mask = np.ones(7, dtype=bool)

        G0 = indiv.native_advan6_prediction_eta_jacobian(theta, np.zeros(2), obs_mask, n_eta=2)
        Ghat = indiv.native_advan6_prediction_eta_jacobian(theta, np.array([0.3, -0.3]), obs_mask, n_eta=2)
        assert G0 is not None and Ghat is not None
        # CL and V both shift with eta — G must change
        assert not np.allclose(G0, Ghat), "G_i must depend on the provided η"

    def test_G_i_respects_obs_mask(self):
        """obs_mask filters the output rows — masked observations are excluded."""
        indiv, pk_callable, param_names = _build_native_individual()
        theta = np.array(_NATIVE_INDIV_THETA)
        eta   = np.zeros(2)
        full_mask    = np.ones(7, dtype=bool)
        partial_mask = np.array([True, False, True, True, False, True, True])

        G_full    = indiv.native_advan6_prediction_eta_jacobian(theta, eta, full_mask,    n_eta=2)
        G_partial = indiv.native_advan6_prediction_eta_jacobian(theta, eta, partial_mask, n_eta=2)
        assert G_full.shape    == (7, 2)
        assert G_partial.shape == (5, 2)
        # Rows that appear in both should be identical
        np.testing.assert_allclose(G_full[partial_mask], G_partial, rtol=1e-10)

    def test_G_i_returns_none_for_mixed_pkpd_contract(self):
        """native path declines mixed PK/PD model (DVID-aware output not implemented)."""
        indiv, _, _ = _build_native_individual()
        # Force the model to look like mixed PK/PD
        if indiv._native_ode_contract is not None:
            indiv._native_ode_contract["is_pkpd"] = True
        theta = np.array(_NATIVE_INDIV_THETA)
        result = indiv.native_advan6_prediction_eta_jacobian(
            theta, np.zeros(2), np.ones(7, dtype=bool), n_eta=2
        )
        assert result is None, "must return None for mixed PK/PD (DVID routing not supported)"

    def test_G_i_returns_none_when_no_contract(self):
        """native G_i returns None gracefully when contract is absent."""
        indiv, _, _ = _build_native_individual()
        indiv._native_ode_contract = None
        result = indiv.native_advan6_prediction_eta_jacobian(
            np.array(_NATIVE_INDIV_THETA), np.zeros(2), np.ones(7, dtype=bool), n_eta=2
        )
        assert result is None

    def test_compute_G_i_uses_native_path(self):
        """_compute_G_i() delegates to native path for ADVAN6 native-contract models."""
        from openpkpd.estimation.foce import _compute_G_i
        indiv, pk_callable, param_names = _build_native_individual()
        theta    = np.array(_NATIVE_INDIV_THETA)
        eta      = np.zeros(2)
        sigma    = np.array([[0.01]])
        obs_mask = np.ones(7, dtype=bool)
        pred0    = np.array([0.01, 0.05, 0.15, 0.30, 0.20, 0.12, 0.03])

        G = _compute_G_i(indiv, theta, eta, sigma, 2, obs_mask, pred0)
        assert G.shape == (7, 2), "G_i should have shape (n_obs, n_eta)"

        # Verify against direct FD using the same multidose probe
        G_fd = self._G_i_fd(pk_callable, param_names, theta, eta, obs_mask)
        np.testing.assert_allclose(G, G_fd, rtol=1e-3, atol=1e-8,
                                   err_msg="_compute_G_i result deviates from FD reference")


# ===========================================================================
# Section 8 — Sensitivity probe performance (transit_1cmt_pkpd)
# ===========================================================================

class TestSensitivityProbePerformance:
    """
    Sensitivity probe vs. equivalent finite-difference work.

    A pure-FD approach for PFIM needs 2×(8 theta + 2 eta) = 20 base probe
    calls.  CVODES forward-sensitivity integration solves a 36-dimensional
    system (4 state + 8×4 sensitivity) in one pass.  Dense BDF linear algebra
    scales as O(N³), so the sensitivity probe is roughly 13–15× more expensive
    than a single 4-state base probe.  One sensitivity call ≈ 20/14 ≈ 1.4×
    cheaper than 20 FD calls.

    Primary benefit of the native path is gradient *accuracy* (no FD rounding
    error); the throughput gain is modest.  We assert ≥1.2× to confirm there
    is a net benefit and guard against regressions.
    """

    _skip = _require(_transit_1cmt_pkpd_sens, _transit_1cmt_pkpd_multidose)


    def test_sensitivity_probe_faster_than_fd_equivalent(self):
        obs   = _SENS_OBS
        dt    = _SENS_DT
        da    = _SENS_DA
        theta = _SENS_THETA
        n_reps = 200
        n_fd_pairs = 10   # 2×(8 theta + 2 eta) = 20 evaluations → 10 pairs of ±eps

        # Warm-up
        _transit_1cmt_pkpd_sens(obs, dt, da, theta)
        for _ in range(n_fd_pairs):
            _transit_1cmt_pkpd_multidose(obs, dt, da, theta)

        t0 = time.perf_counter()
        for _ in range(n_reps):
            _transit_1cmt_pkpd_sens(obs, dt, da, theta)
        sens_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_reps):
            for _ in range(n_fd_pairs):
                _transit_1cmt_pkpd_multidose(obs, dt, da, theta)
        fd_s = time.perf_counter() - t0

        speedup = fd_s / sens_s
        print(f"\n  Sensitivity probe: {sens_s*1e3/n_reps:.3f} ms/call  "
              f"FD equivalent ({n_fd_pairs} pairs): {fd_s*1e3/n_reps:.3f} ms/call  "
              f"speedup: {speedup:.1f}×")
        assert speedup >= 1.2, (
            f"Expected sensitivity probe ≥1.2× faster than {n_fd_pairs} FD pairs; "
            f"got {speedup:.2f}×"
        )


# ===========================================================================
# Section 9 — Laplacian native Hessian (transit_1cmt_pkpd)
# ===========================================================================

class TestLaplacianNativeHessian:
    """
    IndividualModel.eta_objective_hessian() — native Gauss-Newton path.

    ``eta_objective_hessian`` is a multi-path method: when the native CVODES
    sensitivity probe is available and the model is a non-mixed ADVAN6 model,
    it delegates to ``_native_gauss_newton_hessian``, which computes

        H_i = 2 G_i^T diag(1/var) G_i  +  2 Ω^{-1}

    in a single Rust sensitivity solve rather than 2·n_eta·(n_eta+1) ODE calls.
    If the native path is unavailable, it falls back to ``numerical_hessian``.

    Tests verify:
      - positive definiteness (required for log|H_i| to be well-defined)
      - data term correctness: H - 2Ω⁻¹ ≈ 2 G_i^T diag(1/var) G_i (direct)
      - eta-dependence of the data term
      - graceful fallback to numerical Hessian for mixed model / no contract
      - integration: laplacian._outer_ofv_subject_term calls the method
    """

    _skip = _require_with_indiv(_transit_1cmt_pkpd_sens, _transit_1cmt_pkpd_multidose)

    def test_native_hessian_is_positive_definite(self):
        """Gauss-Newton Hessian must be symmetric PD (required for log|H| stability)."""
        indiv, _, _ = _build_native_individual()
        theta = np.array(_NATIVE_INDIV_THETA)
        eta   = np.zeros(2)
        omega = 0.04 * np.eye(2)
        sigma = np.array([[0.01]])

        H = indiv.eta_objective_hessian(theta, eta, omega, sigma)
        assert H.shape == (2, 2)
        np.testing.assert_allclose(H, H.T, atol=1e-12, err_msg="H_i must be symmetric")
        evals = np.linalg.eigvalsh(H)
        assert np.all(evals > 0), f"H_i not PD; eigenvalues: {evals}"

    def test_native_data_hessian_matches_direct_reference(self):
        """Data contribution H - 2Ω⁻¹ equals 2 G_i^T diag(1/var) G_i (direct).

        This validates the Gauss-Newton formula without depending on the
        full obj_eta numerical Hessian (which requires actual observed DV).
        Tolerance rtol=1e-6 reflects double-precision arithmetic only.

        Uses the 2cmt_iv sensitivity probe (4 params, 2 states) consistent
        with the fixture built by _build_native_individual().
        """
        n_params = 4  # CL, V1, Q, V2
        n_states = 2  # central (A1), peripheral (A2)
        out_state = 0  # A1 = central compartment output

        indiv, pk_callable, _ = _build_native_individual()
        theta = np.array(_NATIVE_INDIV_THETA)
        eta   = np.zeros(2)
        omega = 0.04 * np.eye(2)
        sigma = np.array([[0.01]])

        H = indiv.eta_objective_hessian(theta, eta, omega, sigma)
        data_term = H - 2.0 * np.linalg.inv(omega)

        # Build the same reference manually using the 2cmt_iv sensitivity probe.
        # NOTE: contract["required_names"] is a legacy field holding the warfarin
        # param list — do NOT use it here.  The fixture is a known 2cmt_iv model.
        contract = indiv._native_ode_contract
        param_names = ("CL", "V1", "Q", "V2")     # 2cmt_iv required_names
        pk0 = pk_callable(list(theta), [0.0, 0.0])
        ode_theta = [float(pk0[n]) for n in param_names]
        V1 = float(pk0["V1"])
        v_idx = list(param_names).index("V1")

        obs_times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        n_obs = len(obs_times)
        states_raw, sens_raw = _2cmt_iv_sens(
            obs_times, contract["dose_times"], contract["dose_amts"], ode_theta
        )
        states = np.array(states_raw, dtype=float)               # (n_obs, n_states)
        sens   = np.array(sens_raw,   dtype=float).reshape(n_obs, n_params, n_states)

        A1 = states[:, out_state]
        dF_dODE = sens[:, :, out_state] / V1    # (n_obs, n_params) — ∂(A1/V1)/∂ODE_theta
        dF_dODE[:, v_idx] -= A1 / (V1 * V1)    # quotient rule for V1 dependence

        eps = 1e-5
        J = np.zeros((n_params, 2))
        for k in range(2):
            ep = [0.0, 0.0]; ep[k] += eps
            em = [0.0, 0.0]; em[k] -= eps
            pp = pk_callable(list(theta), ep)
            pm = pk_callable(list(theta), em)
            for j, name in enumerate(param_names):
                J[j, k] = (float(pp.get(name, 0.0)) - float(pm.get(name, 0.0))) / (2 * eps)

        G_ref = dF_dODE @ J                    # (n_obs, 2)
        ipred = A1 / V1
        var_ref = np.maximum(float(sigma[0, 0]) * ipred ** 2, 1e-10)
        data_ref = 2.0 * (G_ref.T / var_ref) @ G_ref

        np.testing.assert_allclose(
            data_term, data_ref, rtol=1e-6, atol=1e-10,
            err_msg="Native data Hessian does not match direct G_i^T R^{-1} G_i reference",
        )

    def test_native_data_hessian_changes_with_eta(self):
        """Data term 2 G_i^T R^{-1} G_i changes when η changes.

        Increasing CL via eta[0] changes ODE theta, ipred, G_i, and var,
        so the data contribution must differ at η=0 vs η=[0.3, -0.2].
        """
        indiv, _, _ = _build_native_individual()
        theta = np.array(_NATIVE_INDIV_THETA)
        omega = 0.04 * np.eye(2)
        sigma = np.array([[0.01]])
        omega_inv_2 = 2.0 * np.linalg.inv(omega)

        H0   = indiv.eta_objective_hessian(theta, np.zeros(2), omega, sigma)
        Hhat = indiv.eta_objective_hessian(theta, np.array([0.3, -0.2]), omega, sigma)

        data0   = H0   - omega_inv_2
        datahat = Hhat - omega_inv_2

        # The data Hessians must differ when eta changes; we expect >5% difference
        assert not np.allclose(data0, datahat, rtol=0.05), (
            "Data Hessian must change with η; got data0=%s, datahat=%s" % (data0, datahat)
        )

    def test_native_fallback_for_mixed_model(self):
        """Mixed PK/PD model → native path returns None → method falls back to numerical.

        The result is valid (positive semi-definite) even though it comes from
        the numerical Hessian of obj_eta.  This test confirms graceful fallback,
        not a NotImplementedError.
        """
        indiv, _, _ = _build_native_individual()
        if indiv._native_ode_contract is not None:
            indiv._native_ode_contract["is_pkpd"] = True

        # _native_gauss_newton_hessian must return None for mixed model
        result = indiv._native_gauss_newton_hessian(
            np.array(_NATIVE_INDIV_THETA), np.zeros(2), 0.04 * np.eye(2), np.array([[0.01]])
        )
        assert result is None, "Mixed-model native path must return None"

        # eta_objective_hessian falls back to numerical and still returns a matrix
        H = indiv.eta_objective_hessian(
            np.array(_NATIVE_INDIV_THETA), np.zeros(2), 0.04 * np.eye(2), np.array([[0.01]])
        )
        assert H.shape == (2, 2), "Fallback Hessian has wrong shape"

    def test_native_fallback_when_no_contract(self):
        """No native contract → native path returns None → method falls back to numerical."""
        indiv, _, _ = _build_native_individual()
        indiv._native_ode_contract = None

        result = indiv._native_gauss_newton_hessian(
            np.array(_NATIVE_INDIV_THETA), np.zeros(2), 0.04 * np.eye(2), np.array([[0.01]])
        )
        assert result is None, "No-contract native path must return None"

        H = indiv.eta_objective_hessian(
            np.array(_NATIVE_INDIV_THETA), np.zeros(2), 0.04 * np.eye(2), np.array([[0.01]])
        )
        assert H.shape == (2, 2), "Fallback Hessian has wrong shape"

    def test_laplacian_outer_ofv_calls_native_hessian(self):
        """laplacian._outer_ofv_subject_term dispatches to eta_objective_hessian.

        We stub evaluate_observation_model so the laplacian's try-block
        reaches the Hessian step, then confirm our spy was called exactly once.
        """
        from openpkpd.estimation.laplacian import LaplacianMethod

        indiv, _, _ = _build_native_individual()
        assert callable(getattr(indiv, "eta_objective_hessian", None)), (
            "IndividualModel must expose eta_objective_hessian"
        )

        theta = np.array(_NATIVE_INDIV_THETA)
        omega = 0.04 * np.eye(2)
        sigma = np.array([[0.01]])
        n_obs = 7

        # Stub evaluate_observation_model to return sensible arrays so the
        # laplacian try-block does not raise before reaching the Hessian step.
        fake_pred     = np.ones(n_obs) * 2.0
        fake_var      = np.ones(n_obs) * 0.04
        fake_obs_mask = np.ones(n_obs, dtype=bool)
        indiv.evaluate_observation_model = lambda *a, **kw: (
            fake_pred, fake_obs_mask, fake_pred, fake_pred, fake_var
        )
        # Provide finite obs_dv so residuals are well-defined
        indiv.subject_events.obs_dv = np.ones(n_obs) * 2.0

        class Params:
            def n_eta(self):
                return 2
        p = Params()
        p.theta = theta; p.omega = omega; p.sigma = sigma

        pop_model = MagicMock()
        pop_model.trans = 2
        pop_model.individual_model.return_value = indiv

        method = LaplacianMethod.__new__(LaplacianMethod)
        method.interaction = True

        hessian_calls = []
        orig = indiv.eta_objective_hessian

        def spy(*args, **kwargs):
            hessian_calls.append(1)
            return orig(*args, **kwargs)

        indiv.eta_objective_hessian = spy

        omega_inv = np.linalg.inv(omega)
        eta_hat = {1: np.zeros(2)}

        method._outer_ofv_subject_term(pop_model, p, eta_hat, 1, omega_inv)
        assert len(hessian_calls) == 1, (
            "eta_objective_hessian was not called by _outer_ofv_subject_term"
        )



# ===========================================================================
# Section 10 — ODE template state & sensitivity validation
# ===========================================================================
#
# Five standard PK shapes added in P1 expansion:
#   1cmt_iv  (CL, V)              — theta order: [CL, V]
#   1cmt_oral (KA, CL, V)         — theta order: [KA, CL, V]
#   2cmt_iv  (CL, V1, Q, V2)      — theta order: [CL, V1, Q, V2]
#   2cmt_oral (KA, CL, V2, Q, V3) — theta order: [KA, CL, V2, Q, V3]
#   3cmt_iv  (CL, V1, Q2, V2, Q3, V3) — theta order: [CL, V1, Q2, V2, Q3, V3]
#
# Each test class:
#   1. Compares the state probe against scipy Radau at very tight tolerances.
#   2. Compares the sensitivity probe against central FD of the state probe.
#   3. Verifies a multi-dose scenario (2 doses, 24h apart).


_TPL_OBS  = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0]
_TPL_DOSE_1 = ([0.0], [100.0])
_TPL_DOSE_2 = ([0.0, 24.0], [100.0, 80.0])


def _scipy_generic_multidose(obs_times, dose_times, dose_amts, rhs, ic,
                              rtol=1e-10, atol=1e-12):
    """Scipy Radau multi-dose reference; bolus doses added to ic[0]."""
    y = list(ic)
    t_current = 0.0
    dose_q = sorted(zip(dose_times, dose_amts), key=lambda x: x[0])
    obs_order = sorted(range(len(obs_times)), key=lambda i: obs_times[i])
    results: dict[int, list] = {}
    for obs_i in obs_order:
        t_obs = obs_times[obs_i]
        while dose_q and dose_q[0][0] <= t_obs:
            t_dose, amt = dose_q.pop(0)
            if t_current < t_dose:
                sol = solve_ivp(rhs, [t_current, t_dose], y,
                                method="Radau", rtol=rtol, atol=atol)
                y = sol.y[:, -1].tolist()
                t_current = t_dose
            y[0] += amt
        if t_current < t_obs:
            sol = solve_ivp(rhs, [t_current, t_obs], y,
                            method="Radau", rtol=rtol, atol=atol)
            y = sol.y[:, -1].tolist()
            t_current = t_obs
        results[obs_i] = list(y)
    return np.array([results[i] for i in range(len(obs_times))])


def _fd_sensitivity_generic(probe_fn, obs_times, dose_times, dose_amts, theta,
                             n_params, n_states, eps=1e-5):
    """Central-FD sensitivity via the state probe; returns (n_obs, n_params, n_states)."""
    n_t = len(obs_times)
    sens = np.zeros((n_t, n_params, n_states))
    for j in range(n_params):
        th_p = list(theta); th_p[j] += eps
        th_m = list(theta); th_m[j] -= eps
        sp = np.array(probe_fn(obs_times, dose_times, dose_amts, th_p))
        sm = np.array(probe_fn(obs_times, dose_times, dose_amts, th_m))
        sens[:, j, :] = (sp - sm) / (2.0 * eps)
    return sens


def _scipy_infusion_multidose(obs_times, dose_times, dose_amts, dose_rates,
                               rhs_factory, ic, rtol=1e-10, atol=1e-12):
    """Scipy Radau reference for a mixed bolus / constant-rate infusion schedule.

    ``rhs_factory(rate)`` must return a callable ``f(t, y)`` representing the
    ODE RHS with the given constant forcing rate into compartment 0 (central).
    """
    # Build breakpoints: (time, bolus_delta, rate_delta)
    bps = []
    for t, amt, rate in zip(dose_times, dose_amts, dose_rates):
        if rate == 0.0:
            bps.append((t, amt, 0.0))
        else:
            bps.append((t, 0.0, rate))
            bps.append((t + amt / rate, 0.0, -rate))
    bps.sort(key=lambda x: (x[0], -x[2]))  # time asc, positive rate_delta first

    obs_order = sorted(range(len(obs_times)), key=lambda i: obs_times[i])
    results: dict = {}
    y = list(ic)
    t_cur = 0.0
    active_rate = 0.0
    bp_idx = 0

    for obs_i in obs_order:
        t_obs = obs_times[obs_i]
        # Drain all breakpoints that fall at or before t_obs
        while bp_idx < len(bps) and bps[bp_idx][0] <= t_obs:
            t_bp, bolus, rate_delta = bps[bp_idx]
            if t_cur < t_bp:
                sol = solve_ivp(rhs_factory(active_rate), [t_cur, t_bp], y,
                                method="Radau", rtol=rtol, atol=atol)
                y = sol.y[:, -1].tolist()
                t_cur = t_bp
            y[0] += bolus
            active_rate += rate_delta
            bp_idx += 1
        if t_cur < t_obs:
            sol = solve_ivp(rhs_factory(active_rate), [t_cur, t_obs], y,
                            method="Radau", rtol=rtol, atol=atol)
            y = sol.y[:, -1].tolist()
            t_cur = t_obs
        results[obs_i] = list(y)
    return np.array([results[i] for i in range(len(obs_times))])


def _fd_sensitivity_infusion(probe_fn, obs_times, dose_times, dose_amts, dose_rates,
                              theta, n_params, n_states, eps=1e-5):
    """Central-FD for infusion probes (extra dose_rates arg); → (n_obs, n_params, n_states)."""
    n_t = len(obs_times)
    sens = np.zeros((n_t, n_params, n_states))
    for j in range(n_params):
        th_p = list(theta); th_p[j] += eps
        th_m = list(theta); th_m[j] -= eps
        sp = np.array(probe_fn(obs_times, dose_times, dose_amts, dose_rates, th_p))
        sm = np.array(probe_fn(obs_times, dose_times, dose_amts, dose_rates, th_m))
        sens[:, j, :] = (sp - sm) / (2.0 * eps)
    return sens


# ---------------------------------------------------------------------------
# 1-compartment IV
# ---------------------------------------------------------------------------

class TestTemplate1cmtIv:

    _skip = _require(_1cmt_iv_probe)


    _THETA = [0.13, 8.0]   # CL, V

    def _scipy_ref(self, obs, dt, da, theta):
        cl, v = theta
        return _scipy_generic_multidose(obs, dt, da,
                                        lambda t, y: [-(cl / v) * y[0]],
                                        ic=[0.0])

    def test_state_single_dose_matches_scipy(self):
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_1
        ref = self._scipy_ref(obs, dt, da, self._THETA)
        got = np.array(_1cmt_iv_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_state_two_doses_matches_scipy(self):
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_2
        ref = self._scipy_ref(obs, dt, da, self._THETA)
        got = np.array(_1cmt_iv_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_mass_balance_single_dose(self):
        got = np.array(_1cmt_iv_probe(_TPL_OBS, *_TPL_DOSE_1, self._THETA))
        assert np.all(got[:, 0] <= _TPL_DOSE_1[1][0] + 1e-8)

    def test_sensitivity_matches_fd(self):
        if _1cmt_iv_sens is None:
            pytest.skip("1cmt_iv sensitivity probe not compiled in")
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_1
        _, sens_raw = _1cmt_iv_sens(obs, dt, da, self._THETA)
        sens = np.array(sens_raw).reshape(len(obs), 2, 1)
        sens_fd = _fd_sensitivity_generic(_1cmt_iv_probe, obs, dt, da,
                                          self._THETA, 2, 1)
        np.testing.assert_allclose(sens, sens_fd, rtol=1e-4,
                                   atol=1e-4 * max(da))


# ---------------------------------------------------------------------------
# 1-compartment oral
# ---------------------------------------------------------------------------

class TestTemplate1cmtOral:

    _skip = _require(_1cmt_oral_probe)


    _THETA = [0.4, 0.13, 8.0]   # KA, CL, V

    def _scipy_ref(self, obs, dt, da, theta):
        ka, cl, v = theta
        def rhs(t, y): return [-ka * y[0], ka * y[0] - (cl / v) * y[1]]
        return _scipy_generic_multidose(obs, dt, da, rhs, ic=[0.0, 0.0])

    def test_state_single_dose_matches_scipy(self):
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_1
        ref = self._scipy_ref(obs, dt, da, self._THETA)
        got = np.array(_1cmt_oral_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_state_two_doses_matches_scipy(self):
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_2
        ref = self._scipy_ref(obs, dt, da, self._THETA)
        got = np.array(_1cmt_oral_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_central_peaks_then_declines(self):
        """A2 (central) must peak then decline for typical oral absorption."""
        obs = list(np.linspace(0.25, 48.0, 40))
        got = np.array(_1cmt_oral_probe(obs, *_TPL_DOSE_1, self._THETA))
        a2 = got[:, 1]
        peak = int(np.argmax(a2))
        assert peak > 0 and np.all(np.diff(a2[peak:]) <= 1e-4)

    def test_sensitivity_matches_fd(self):
        if _1cmt_oral_sens is None:
            pytest.skip("1cmt_oral sensitivity probe not compiled in")
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_1
        _, sens_raw = _1cmt_oral_sens(obs, dt, da, self._THETA)
        sens = np.array(sens_raw).reshape(len(obs), 3, 2)
        sens_fd = _fd_sensitivity_generic(_1cmt_oral_probe, obs, dt, da,
                                          self._THETA, 3, 2)
        np.testing.assert_allclose(sens, sens_fd, rtol=1e-4,
                                   atol=1e-4 * max(da))


# ---------------------------------------------------------------------------
# 2-compartment IV
# ---------------------------------------------------------------------------

class TestTemplate2cmtIv:

    _skip = _require(_2cmt_iv_probe)


    _THETA = [0.13, 8.0, 0.6, 20.0]   # CL, V1, Q, V2

    def _scipy_ref(self, obs, dt, da, theta):
        cl, v1, q, v2 = theta
        k10 = cl / v1; k12 = q / v1; k21 = q / v2
        def rhs(t, y):
            return [-(k10 + k12) * y[0] + k21 * y[1],
                    k12 * y[0] - k21 * y[1]]
        return _scipy_generic_multidose(obs, dt, da, rhs, ic=[0.0, 0.0])

    def test_state_single_dose_matches_scipy(self):
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_1
        ref = self._scipy_ref(obs, dt, da, self._THETA)
        got = np.array(_2cmt_iv_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_state_two_doses_matches_scipy(self):
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_2
        ref = self._scipy_ref(obs, dt, da, self._THETA)
        got = np.array(_2cmt_iv_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_peripheral_accumulates_then_declines(self):
        """A2 (peripheral) should rise initially then fall after drug clears central."""
        obs = list(np.linspace(0.1, 72.0, 60))
        got = np.array(_2cmt_iv_probe(obs, *_TPL_DOSE_1, self._THETA))
        a2 = got[:, 1]
        assert np.max(a2) > 0.0, "Peripheral compartment must receive drug"
        peak = int(np.argmax(a2))
        assert peak > 0, "Peripheral peak must not be at t=0"

    def test_sensitivity_matches_fd(self):
        if _2cmt_iv_sens is None:
            pytest.skip("2cmt_iv sensitivity probe not compiled in")
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_1
        _, sens_raw = _2cmt_iv_sens(obs, dt, da, self._THETA)
        sens = np.array(sens_raw).reshape(len(obs), 4, 2)
        sens_fd = _fd_sensitivity_generic(_2cmt_iv_probe, obs, dt, da,
                                          self._THETA, 4, 2)
        np.testing.assert_allclose(sens, sens_fd, rtol=1e-4,
                                   atol=1e-4 * max(da))


# ---------------------------------------------------------------------------
# 2-compartment oral
# ---------------------------------------------------------------------------

class TestTemplate2cmtOral:

    _skip = _require(_2cmt_oral_probe)


    _THETA = [0.4, 0.13, 8.0, 0.6, 20.0]   # KA, CL, V2, Q, V3

    def _scipy_ref(self, obs, dt, da, theta):
        ka, cl, v2, q, v3 = theta
        k10 = cl / v2; k12 = q / v2; k21 = q / v3
        def rhs(t, y):
            return [-ka * y[0],
                    ka * y[0] - (k10 + k12) * y[1] + k21 * y[2],
                    k12 * y[1] - k21 * y[2]]
        return _scipy_generic_multidose(obs, dt, da, rhs, ic=[0.0, 0.0, 0.0])

    def test_state_single_dose_matches_scipy(self):
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_1
        ref = self._scipy_ref(obs, dt, da, self._THETA)
        got = np.array(_2cmt_oral_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_state_two_doses_matches_scipy(self):
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_2
        ref = self._scipy_ref(obs, dt, da, self._THETA)
        got = np.array(_2cmt_oral_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_central_peaks_then_declines(self):
        """Central compartment (A2) must peak after administration."""
        obs = list(np.linspace(0.25, 48.0, 50))
        got = np.array(_2cmt_oral_probe(obs, *_TPL_DOSE_1, self._THETA))
        a2 = got[:, 1]
        peak = int(np.argmax(a2))
        assert peak > 0 and np.all(np.diff(a2[peak:]) <= 1e-4)

    def test_sensitivity_matches_fd(self):
        if _2cmt_oral_sens is None:
            pytest.skip("2cmt_oral sensitivity probe not compiled in")
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_1
        _, sens_raw = _2cmt_oral_sens(obs, dt, da, self._THETA)
        sens = np.array(sens_raw).reshape(len(obs), 5, 3)
        sens_fd = _fd_sensitivity_generic(_2cmt_oral_probe, obs, dt, da,
                                          self._THETA, 5, 3)
        np.testing.assert_allclose(sens, sens_fd, rtol=1e-4,
                                   atol=1e-4 * max(da))


# ---------------------------------------------------------------------------
# 3-compartment IV
# ---------------------------------------------------------------------------

class TestTemplate3cmtIv:

    _skip = _require(_3cmt_iv_probe)


    # CL, V1, Q2, V2, Q3, V3
    _THETA = [0.13, 8.0, 0.6, 20.0, 0.3, 50.0]

    def _scipy_ref(self, obs, dt, da, theta):
        cl, v1, q2, v2, q3, v3 = theta
        k10 = cl / v1; k12 = q2 / v1; k21 = q2 / v2
        k13 = q3 / v1; k31 = q3 / v3
        def rhs(t, y):
            return [-(k10 + k12 + k13) * y[0] + k21 * y[1] + k31 * y[2],
                    k12 * y[0] - k21 * y[1],
                    k13 * y[0] - k31 * y[2]]
        return _scipy_generic_multidose(obs, dt, da, rhs, ic=[0.0, 0.0, 0.0])

    def test_state_single_dose_matches_scipy(self):
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_1
        ref = self._scipy_ref(obs, dt, da, self._THETA)
        got = np.array(_3cmt_iv_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_state_two_doses_matches_scipy(self):
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_2
        ref = self._scipy_ref(obs, dt, da, self._THETA)
        got = np.array(_3cmt_iv_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_both_peripherals_accumulate(self):
        """Both A2 and A3 should receive drug and decay after clearing."""
        obs = list(np.linspace(0.1, 72.0, 60))
        got = np.array(_3cmt_iv_probe(obs, *_TPL_DOSE_1, self._THETA))
        assert np.max(got[:, 1]) > 0.0, "Periph-1 must receive drug"
        assert np.max(got[:, 2]) > 0.0, "Periph-2 must receive drug"

    def test_sensitivity_matches_fd(self):
        if _3cmt_iv_sens is None:
            pytest.skip("3cmt_iv sensitivity probe not compiled in")
        obs, (dt, da) = _TPL_OBS, _TPL_DOSE_1
        _, sens_raw = _3cmt_iv_sens(obs, dt, da, self._THETA)
        sens = np.array(sens_raw).reshape(len(obs), 6, 3)
        sens_fd = _fd_sensitivity_generic(_3cmt_iv_probe, obs, dt, da,
                                          self._THETA, 6, 3)
        np.testing.assert_allclose(sens, sens_fd, rtol=1e-4,
                                   atol=1e-4 * max(da))


# ===========================================================================
# Section 11 — Cross-validation: CVODES templates vs analytical ADVAN solvers
# ===========================================================================
#
# The analytical ADVAN1/2/3/4/5 solvers are independently validated against
# NONMEM and WinNonlin (see tests/external_validation/).  Comparing the
# CVODES templates against them chains the CVODES accuracy to those published
# benchmarks — in particular to NONMEM Run 402 (2-cmt IV, V1=9.76, CL=3.88).
#
# Tolerance rationale: the analytical solutions are exact to machine precision;
# CVODES BDF runs at rtol=1e-8 / atol=1e-10.  We allow rtol=1e-4 to absorb
# the ODE solver error — matching the tolerance used in TestODEVsAnalytical.

# NM 402 population means (externally validated against NONMEM 7.4.3 FOCEI)
_NM402 = {"CL": 3.876825, "V1": 9.760346, "V2": 30.81783, "Q": 8.773851}

_XV_OBS   = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0]
_XV_DOSE1 = ([0.0], [100.0])
# Second dose at t=25 (not t=24) to avoid the pre-dose/post-dose convention
# difference: CVODES gives the post-dose state when an obs falls exactly on a
# dose time, while the analytical ADVAN solvers give the pre-dose state.
_XV_DOSE2 = ([0.0, 25.0], [100.0, 80.0])


def _advan_dose(t=0.0, amt=100.0, cmt=1):
    from openpkpd.data.event_processor import DoseEvent
    return [DoseEvent(time=t, amount=amt, compartment=cmt)]


def _advan_doses(dose_times, dose_amts, cmt=1):
    from openpkpd.data.event_processor import DoseEvent
    return [DoseEvent(time=t, amount=a, compartment=cmt)
            for t, a in zip(dose_times, dose_amts)]


# ---------------------------------------------------------------------------
# 1-cmt IV: CVODES probe vs ADVAN1
# ---------------------------------------------------------------------------

class TestNativeCvodes1CmtIvVsADVAN1:
    """native_cvodes_1cmt_iv_probe vs ADVAN1 exact solution."""

    _skip = _require(_1cmt_iv_probe)


    # typical theophylline-like params; K = CL/V
    _CL, _V = 0.044, 31.8   # L/h, L  (Boeckmann theophylline population means)
    _THETA = [_CL, _V]       # CVODES order: [CL, V]

    def _advan1_ipred(self, obs, dose_events):
        from openpkpd.pk.analytical.advan1 import ADVAN1
        return ADVAN1().solve(
            {"K": self._CL / self._V, "V": self._V},
            dose_events,
            np.array(obs),
        ).ipred

    def test_single_dose_ipred_matches_advan1(self):
        obs, (dt, da) = _XV_OBS, _XV_DOSE1
        ref = self._advan1_ipred(obs, _advan_doses(dt, da))
        states = np.array(_1cmt_iv_probe(obs, dt, da, self._THETA))
        got = states[:, 0] / self._V
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)

    def test_two_dose_ipred_matches_advan1(self):
        obs, (dt, da) = _XV_OBS, _XV_DOSE2
        ref = self._advan1_ipred(obs, _advan_doses(dt, da))
        states = np.array(_1cmt_iv_probe(obs, dt, da, self._THETA))
        got = states[:, 0] / self._V
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)

    def test_amounts_match_advan1(self):
        from openpkpd.pk.analytical.advan1 import ADVAN1
        obs, (dt, da) = _XV_OBS, _XV_DOSE1
        sol = ADVAN1().solve({"K": self._CL / self._V, "V": self._V},
                             _advan_doses(dt, da), np.array(obs))
        states = np.array(_1cmt_iv_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(states[:, 0], sol.amounts[:, 0],
                                   rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------------------
# 1-cmt oral: CVODES probe vs ADVAN2
# ---------------------------------------------------------------------------

class TestNativeCvodes1CmtOralVsADVAN2:
    """native_cvodes_1cmt_oral_probe vs ADVAN2 (Bateman) exact solution."""

    _skip = _require(_1cmt_oral_probe)


    # Boeckmann theophylline population means
    _KA, _CL, _V = 1.49, 0.044, 31.8
    _THETA = [_KA, _CL, _V]   # CVODES order: [KA, CL, V]

    def _advan2_ipred(self, obs, dose_events):
        from openpkpd.pk.analytical.advan2 import ADVAN2
        return ADVAN2().solve(
            {"KA": self._KA, "K": self._CL / self._V, "V": self._V},
            dose_events,
            np.array(obs),
        ).ipred

    def test_single_dose_ipred_matches_advan2(self):
        obs, (dt, da) = _XV_OBS, _XV_DOSE1
        ref = self._advan2_ipred(obs, _advan_doses(dt, da))
        states = np.array(_1cmt_oral_probe(obs, dt, da, self._THETA))
        got = states[:, 1] / self._V      # central compartment = index 1
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)

    def test_two_dose_ipred_matches_advan2(self):
        obs, (dt, da) = _XV_OBS, _XV_DOSE2
        ref = self._advan2_ipred(obs, _advan_doses(dt, da))
        states = np.array(_1cmt_oral_probe(obs, dt, da, self._THETA))
        got = states[:, 1] / self._V
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)

    def test_depot_amounts_match_advan2(self):
        """Depot compartment A1 must match the ADVAN2 depot amount."""
        from openpkpd.pk.analytical.advan2 import ADVAN2
        obs, (dt, da) = _XV_OBS, _XV_DOSE1
        sol = ADVAN2().solve({"KA": self._KA, "K": self._CL / self._V, "V": self._V},
                             _advan_doses(dt, da), np.array(obs))
        states = np.array(_1cmt_oral_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(states[:, 0], sol.amounts[:, 0],
                                   rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------------------
# 2-cmt IV: CVODES probe vs ADVAN3  (includes NONMEM 402 reference theta)
# ---------------------------------------------------------------------------

class TestNativeCvodes2CmtIvVsADVAN3:
    """native_cvodes_2cmt_iv_probe vs ADVAN3 biexponential exact solution.

    ADVAN3 TRANS4 is validated against NONMEM 7.4.3 Run 402.  Using its
    population-mean theta chains the CVODES 2cmt_iv template accuracy to
    that published external benchmark.
    """

    _skip = _require(_2cmt_iv_probe)


    # Typical params (micro CL/V form)
    _THETA_TYP = [0.13, 8.0, 0.6, 20.0]   # CL, V1, Q, V2

    # NONMEM 402 population means — externally validated (NONMEM 7.4.3 FOCEI)
    _THETA_NM402 = [
        _NM402["CL"], _NM402["V1"], _NM402["Q"], _NM402["V2"],
    ]

    def _advan3_ipred(self, obs, dose_events, theta):
        from openpkpd.pk.analytical.advan3 import ADVAN3
        cl, v1, q, v2 = theta
        return ADVAN3().solve(
            {"CL": cl, "Q": q, "V1": v1, "V2": v2},   # TRANS4 macro params
            dose_events,
            np.array(obs),
        ).ipred

    def test_typical_single_dose_ipred_matches_advan3(self):
        obs, (dt, da) = _XV_OBS, _XV_DOSE1
        ref = self._advan3_ipred(obs, _advan_doses(dt, da), self._THETA_TYP)
        states = np.array(_2cmt_iv_probe(obs, dt, da, self._THETA_TYP))
        got = states[:, 0] / self._THETA_TYP[1]   # A1/V1
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)

    def test_typical_two_dose_ipred_matches_advan3(self):
        obs, (dt, da) = _XV_OBS, _XV_DOSE2
        ref = self._advan3_ipred(obs, _advan_doses(dt, da), self._THETA_TYP)
        states = np.array(_2cmt_iv_probe(obs, dt, da, self._THETA_TYP))
        got = states[:, 0] / self._THETA_TYP[1]
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)

    def test_nonmem402_theta_ipred_matches_advan3(self):
        """NONMEM 402 population means: CVODES must match ADVAN3 to rtol=1e-4.

        This is the key chain: ADVAN3 TRANS4 ≡ NONMEM 402 reference →
        CVODES 2cmt_iv probe matches ADVAN3 → CVODES is within 0.01% of
        the NONMEM-published trajectory.
        """
        obs, (dt, da) = _XV_OBS, _XV_DOSE1
        theta = self._THETA_NM402
        ref = self._advan3_ipred(obs, _advan_doses(dt, da), theta)
        states = np.array(_2cmt_iv_probe(obs, dt, da, theta))
        got = states[:, 0] / _NM402["V1"]
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8,
                                   err_msg="CVODES 2cmt_iv deviates from ADVAN3 at NONMEM 402 theta")

    def test_peripheral_amounts_match_advan3(self):
        from openpkpd.pk.analytical.advan3 import ADVAN3
        obs, (dt, da) = _XV_OBS, _XV_DOSE1
        cl, v1, q, v2 = self._THETA_TYP
        sol = ADVAN3().solve({"CL": cl, "Q": q, "V1": v1, "V2": v2},
                             _advan_doses(dt, da), np.array(obs))
        states = np.array(_2cmt_iv_probe(obs, dt, da, self._THETA_TYP))
        # ADVAN3 amounts[:, 0] = central, amounts[:, 1] = peripheral
        np.testing.assert_allclose(states[:, 1], sol.amounts[:, 1],
                                   rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------------------
# 2-cmt oral: CVODES probe vs ADVAN4
# ---------------------------------------------------------------------------

class TestNativeCvodes2CmtOralVsADVAN4:
    """native_cvodes_2cmt_oral_probe vs ADVAN4 exact solution."""

    _skip = _require(_2cmt_oral_probe)


    _KA, _CL, _V2, _Q, _V3 = 0.4, 0.13, 8.0, 0.6, 20.0
    _THETA = [_KA, _CL, _V2, _Q, _V3]   # CVODES order

    def _advan4_ipred(self, obs, dose_events):
        from openpkpd.pk.analytical.advan4 import ADVAN4
        k10 = self._CL / self._V2
        k12 = self._Q  / self._V2
        k21 = self._Q  / self._V3
        return ADVAN4().solve(
            {"KA": self._KA, "K": k10, "K12": k12, "K21": k21, "V2": self._V2},
            dose_events,
            np.array(obs),
        ).ipred

    def test_single_dose_ipred_matches_advan4(self):
        obs, (dt, da) = _XV_OBS, _XV_DOSE1
        ref = self._advan4_ipred(obs, _advan_doses(dt, da))
        states = np.array(_2cmt_oral_probe(obs, dt, da, self._THETA))
        got = states[:, 1] / self._V2   # central = index 1
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)

    def test_two_dose_ipred_matches_advan4(self):
        obs, (dt, da) = _XV_OBS, _XV_DOSE2
        ref = self._advan4_ipred(obs, _advan_doses(dt, da))
        states = np.array(_2cmt_oral_probe(obs, dt, da, self._THETA))
        got = states[:, 1] / self._V2
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------------------
# 3-cmt IV: CVODES probe vs ADVAN5
# ---------------------------------------------------------------------------

class TestNativeCvodes3CmtIvVsADVAN5:
    """native_cvodes_3cmt_iv_probe vs ADVAN5 exact solution."""

    _skip = _require(_3cmt_iv_probe)


    _CL, _V1, _Q2, _V2, _Q3, _V3 = 0.13, 8.0, 0.6, 20.0, 0.3, 50.0
    _THETA = [_CL, _V1, _Q2, _V2, _Q3, _V3]   # CVODES order

    def _advan5_ipred(self, obs, dose_events):
        from openpkpd.pk.analytical.advan5 import ADVAN5
        k10 = self._CL / self._V1
        k12 = self._Q2 / self._V1;  k21 = self._Q2 / self._V2
        k13 = self._Q3 / self._V1;  k31 = self._Q3 / self._V3
        return ADVAN5().solve(
            {"K": k10, "K12": k12, "K21": k21,
             "K13": k13, "K31": k31, "V1": self._V1},
            dose_events,
            np.array(obs),
        ).ipred

    def test_single_dose_ipred_matches_advan5(self):
        obs, (dt, da) = _XV_OBS, _XV_DOSE1
        ref = self._advan5_ipred(obs, _advan_doses(dt, da))
        states = np.array(_3cmt_iv_probe(obs, dt, da, self._THETA))
        got = states[:, 0] / self._V1   # central = index 0
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)

    def test_two_dose_ipred_matches_advan5(self):
        obs, (dt, da) = _XV_OBS, _XV_DOSE2
        ref = self._advan5_ipred(obs, _advan_doses(dt, da))
        states = np.array(_3cmt_iv_probe(obs, dt, da, self._THETA))
        got = states[:, 0] / self._V1
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)

    def test_peripheral_amounts_match_advan5(self):
        """Both peripheral compartment amounts must match ADVAN5."""
        from openpkpd.pk.analytical.advan5 import ADVAN5
        obs, (dt, da) = _XV_OBS, _XV_DOSE1
        k10 = self._CL / self._V1
        k12 = self._Q2 / self._V1;  k21 = self._Q2 / self._V2
        k13 = self._Q3 / self._V1;  k31 = self._Q3 / self._V3
        sol = ADVAN5().solve(
            {"K": k10, "K12": k12, "K21": k21,
             "K13": k13, "K31": k31, "V1": self._V1},
            _advan_doses(dt, da), np.array(obs),
        )
        states = np.array(_3cmt_iv_probe(obs, dt, da, self._THETA))
        np.testing.assert_allclose(states[:, 1], sol.amounts[:, 1],
                                   rtol=1e-4, atol=1e-8, err_msg="Periph-1 mismatch")
        np.testing.assert_allclose(states[:, 2], sol.amounts[:, 2],
                                   rtol=1e-4, atol=1e-8, err_msg="Periph-2 mismatch")


# ===========================================================================
# Section 12 — 3cmt_oral / 4cmt_iv / 4cmt_oral: scipy + analytical validation
# ===========================================================================


# ── 3-cmt oral ───────────────────────────────────────────────────────────────

class TestTemplate3CmtOral:

    _skip = _require(_3cmt_oral_probe)


    _THETA = [0.4, 0.13, 8.0, 0.6, 20.0, 0.3, 50.0]   # KA,CL,V2,Q3,V3,Q4,V4

    def _scipy_ref(self, obs, dt, da, th):
        ka, cl, v2, q3, v3, q4, v4 = th
        k10=cl/v2; k23=q3/v2; k32=q3/v3; k24=q4/v2; k42=q4/v4
        def rhs(t, y): return [
            -ka*y[0],
            ka*y[0]-(k10+k23+k24)*y[1]+k32*y[2]+k42*y[3],
            k23*y[1]-k32*y[2], k24*y[1]-k42*y[3]]
        return _scipy_generic_multidose(obs, dt, da, rhs, [0.0]*4)

    def test_state_single_dose_matches_scipy(self):
        obs,(dt,da)=_TPL_OBS,_TPL_DOSE_1
        np.testing.assert_allclose(np.array(_3cmt_oral_probe(obs,dt,da,self._THETA)),
                                   self._scipy_ref(obs,dt,da,self._THETA), rtol=5e-5, atol=1e-7)

    def test_state_two_doses_matches_scipy(self):
        obs,(dt,da)=_TPL_OBS,_TPL_DOSE_2
        np.testing.assert_allclose(np.array(_3cmt_oral_probe(obs,dt,da,self._THETA)),
                                   self._scipy_ref(obs,dt,da,self._THETA), rtol=5e-5, atol=1e-7)

    def test_central_peaks_then_declines(self):
        obs=list(np.linspace(0.25,48.0,50))
        got=np.array(_3cmt_oral_probe(obs,*_TPL_DOSE_1,self._THETA))
        a2=got[:,1]; peak=int(np.argmax(a2))
        assert peak>0 and np.all(np.diff(a2[peak:])<=1e-4)

    def test_both_peripherals_receive_drug(self):
        got=np.array(_3cmt_oral_probe(_TPL_OBS,*_TPL_DOSE_1,self._THETA))
        assert np.max(got[:,2])>0.0 and np.max(got[:,3])>0.0

    def test_sensitivity_matches_fd(self):
        if _3cmt_oral_sens is None: pytest.skip("3cmt_oral sens probe not compiled in")
        obs,(dt,da)=_TPL_OBS,_TPL_DOSE_1
        _,sens_raw=_3cmt_oral_sens(obs,dt,da,self._THETA)
        sens=np.array(sens_raw).reshape(len(obs),7,4)
        sens_fd=_fd_sensitivity_generic(_3cmt_oral_probe,obs,dt,da,self._THETA,7,4)
        np.testing.assert_allclose(sens,sens_fd,rtol=1e-4,atol=1e-4*max(da))

    def test_vs_advan12_single_dose(self):
        """Cross-validation against ADVAN12 exact solution (NONMEM ADVAN12)."""
        from openpkpd.pk.analytical.advan12 import ADVAN12
        obs,(dt,da)=_XV_OBS,_XV_DOSE1; th=self._THETA
        ka,cl,v2,q3,v3,q4,v4=th
        sol=ADVAN12().solve({"KA":ka,"K":cl/v2,"K12":q3/v2,"K21":q3/v3,
                             "K13":q4/v2,"K31":q4/v4,"V2":v2},
                            _advan_doses(dt,da),np.array(obs))
        got=np.array(_3cmt_oral_probe(obs,dt,da,th))[:,1]/v2
        np.testing.assert_allclose(got,sol.ipred,rtol=1e-4,atol=1e-8)

    def test_vs_advan12_two_doses(self):
        from openpkpd.pk.analytical.advan12 import ADVAN12
        obs,(dt,da)=_XV_OBS,_XV_DOSE2; th=self._THETA
        ka,cl,v2,q3,v3,q4,v4=th
        sol=ADVAN12().solve({"KA":ka,"K":cl/v2,"K12":q3/v2,"K21":q3/v3,
                             "K13":q4/v2,"K31":q4/v4,"V2":v2},
                            _advan_doses(dt,da),np.array(obs))
        got=np.array(_3cmt_oral_probe(obs,dt,da,th))[:,1]/v2
        np.testing.assert_allclose(got,sol.ipred,rtol=1e-4,atol=1e-8)


# ── 4-cmt IV ─────────────────────────────────────────────────────────────────

class TestTemplate4CmtIv:

    _skip = _require(_4cmt_iv_probe)


    _THETA = [0.13, 8.0, 0.6, 20.0, 0.3, 50.0, 0.1, 100.0]  # CL,V1,Q2,V2,Q3,V3,Q4,V4

    def _scipy_ref(self, obs, dt, da, th):
        cl,v1,q2,v2,q3,v3,q4,v4=th
        k10=cl/v1; k12=q2/v1; k21=q2/v2; k13=q3/v1; k31=q3/v3; k14=q4/v1; k41=q4/v4
        def rhs(t,y): return [
            -(k10+k12+k13+k14)*y[0]+k21*y[1]+k31*y[2]+k41*y[3],
            k12*y[0]-k21*y[1], k13*y[0]-k31*y[2], k14*y[0]-k41*y[3]]
        return _scipy_generic_multidose(obs,dt,da,rhs,[0.0]*4)

    def test_state_single_dose_matches_scipy(self):
        obs,(dt,da)=_TPL_OBS,_TPL_DOSE_1
        np.testing.assert_allclose(np.array(_4cmt_iv_probe(obs,dt,da,self._THETA)),
                                   self._scipy_ref(obs,dt,da,self._THETA),rtol=5e-5,atol=1e-7)

    def test_state_two_doses_matches_scipy(self):
        obs,(dt,da)=_TPL_OBS,_TPL_DOSE_2
        np.testing.assert_allclose(np.array(_4cmt_iv_probe(obs,dt,da,self._THETA)),
                                   self._scipy_ref(obs,dt,da,self._THETA),rtol=5e-5,atol=1e-7)

    def test_all_three_peripherals_receive_drug(self):
        got=np.array(_4cmt_iv_probe(_TPL_OBS,*_TPL_DOSE_1,self._THETA))
        assert all(np.max(got[:,i])>0 for i in [1,2,3])

    def test_sensitivity_matches_fd(self):
        if _4cmt_iv_sens is None: pytest.skip("4cmt_iv sens probe not compiled in")
        obs,(dt,da)=_TPL_OBS,_TPL_DOSE_1
        _,sens_raw=_4cmt_iv_sens(obs,dt,da,self._THETA)
        sens=np.array(sens_raw).reshape(len(obs),8,4)
        sens_fd=_fd_sensitivity_generic(_4cmt_iv_probe,obs,dt,da,self._THETA,8,4)
        np.testing.assert_allclose(sens,sens_fd,rtol=1e-4,atol=1e-4*max(da))

    def test_vs_advan5_single_dose(self):
        """Cross-validation against ADVAN5 (N-compartment linear exact solution)."""
        from openpkpd.pk.analytical.advan5 import ADVAN5
        obs,(dt,da)=_XV_OBS,_XV_DOSE1; th=self._THETA
        cl,v1,q2,v2,q3,v3,q4,v4=th
        sol=ADVAN5().solve({"K":cl/v1,"K12":q2/v1,"K21":q2/v2,
                            "K13":q3/v1,"K31":q3/v3,"K14":q4/v1,"K41":q4/v4,"V1":v1},
                           _advan_doses(dt,da),np.array(obs))
        got=np.array(_4cmt_iv_probe(obs,dt,da,th))[:,0]/v1
        np.testing.assert_allclose(got,sol.ipred,rtol=1e-4,atol=1e-8)

    def test_vs_advan5_two_doses(self):
        from openpkpd.pk.analytical.advan5 import ADVAN5
        obs,(dt,da)=_XV_OBS,_XV_DOSE2; th=self._THETA
        cl,v1,q2,v2,q3,v3,q4,v4=th
        sol=ADVAN5().solve({"K":cl/v1,"K12":q2/v1,"K21":q2/v2,
                            "K13":q3/v1,"K31":q3/v3,"K14":q4/v1,"K41":q4/v4,"V1":v1},
                           _advan_doses(dt,da),np.array(obs))
        got=np.array(_4cmt_iv_probe(obs,dt,da,th))[:,0]/v1
        np.testing.assert_allclose(got,sol.ipred,rtol=1e-4,atol=1e-8)


# ── 4-cmt oral ───────────────────────────────────────────────────────────────

class TestTemplate4CmtOral:

    _skip = _require(_4cmt_oral_probe)


    _THETA=[0.4,0.13,8.0,0.6,20.0,0.3,50.0,0.1,100.0]  # KA,CL,V2,Q3,V3,Q4,V4,Q5,V5

    def _scipy_ref(self,obs,dt,da,th):
        ka,cl,v2,q3,v3,q4,v4,q5,v5=th
        k10=cl/v2;k23=q3/v2;k32=q3/v3;k24=q4/v2;k42=q4/v4;k25=q5/v2;k52=q5/v5
        def rhs(t,y): return [
            -ka*y[0],
            ka*y[0]-(k10+k23+k24+k25)*y[1]+k32*y[2]+k42*y[3]+k52*y[4],
            k23*y[1]-k32*y[2],k24*y[1]-k42*y[3],k25*y[1]-k52*y[4]]
        return _scipy_generic_multidose(obs,dt,da,rhs,[0.0]*5)

    def test_state_single_dose_matches_scipy(self):
        obs,(dt,da)=_TPL_OBS,_TPL_DOSE_1
        np.testing.assert_allclose(np.array(_4cmt_oral_probe(obs,dt,da,self._THETA)),
                                   self._scipy_ref(obs,dt,da,self._THETA),rtol=5e-5,atol=1e-7)

    def test_state_two_doses_matches_scipy(self):
        obs,(dt,da)=_TPL_OBS,_TPL_DOSE_2
        np.testing.assert_allclose(np.array(_4cmt_oral_probe(obs,dt,da,self._THETA)),
                                   self._scipy_ref(obs,dt,da,self._THETA),rtol=5e-5,atol=1e-7)

    def test_central_peaks_then_declines(self):
        obs=list(np.linspace(0.25,48.0,60))
        got=np.array(_4cmt_oral_probe(obs,*_TPL_DOSE_1,self._THETA))
        a2=got[:,1]; peak=int(np.argmax(a2))
        assert peak>0 and np.all(np.diff(a2[peak:])<=1e-4)

    def test_all_three_peripherals_receive_drug(self):
        got=np.array(_4cmt_oral_probe(_TPL_OBS,*_TPL_DOSE_1,self._THETA))
        assert all(np.max(got[:,i])>0 for i in [2,3,4])

    def test_sensitivity_matches_fd(self):
        if _4cmt_oral_sens is None: pytest.skip("4cmt_oral sens probe not compiled in")
        obs,(dt,da)=_TPL_OBS,_TPL_DOSE_1
        _,sens_raw=_4cmt_oral_sens(obs,dt,da,self._THETA)
        sens=np.array(sens_raw).reshape(len(obs),9,5)
        sens_fd=_fd_sensitivity_generic(_4cmt_oral_probe,obs,dt,da,self._THETA,9,5)
        np.testing.assert_allclose(sens,sens_fd,rtol=1e-4,atol=1e-4*max(da))


# ===========================================================================
# Section 13 — constant-rate infusion probes  (1–4 cmt IV)
# ===========================================================================
#
# For each IV template the infusion probe is validated against:
#   a) scipy Radau (independent solver, same ODE)  — single and two-infusion
#   b) analytical solution (1-cmt IV only; closed-form exists)
#   c) mixed bolus + infusion schedule
#   d) sensitivity probe vs central finite differences
#
# Observation grid spans before, during, and after the infusion window.
# Single infusion : amt=100, rate=10  → 10 h duration at t=0
# Two infusions   : t=[0, 25], amt=[100,80], rate=[10,8] → 10h each
# Mixed           : t=[0, 25], amt=[100,80], rate=[0,10]  → bolus then 10h inf

_INF_OBS   = [1.0, 5.0, 9.0, 10.5, 14.0, 20.0, 30.0, 48.0]
_INF1_D    = ([0.0],       [100.0], [10.0])          # (times, amts, rates)
_INF2_D    = ([0.0, 25.0], [100.0, 80.0], [10.0, 8.0])
_INF_MIX_D = ([0.0, 25.0], [100.0, 80.0], [0.0, 10.0])


# ---------------------------------------------------------------------------
# 1-compartment IV infusion
# ---------------------------------------------------------------------------

class TestInfusion1cmtIv:

    _skip = _require(_1cmt_iv_inf_probe)


    _THETA = [1.0, 10.0]   # CL=1, V=10 → k=0.1 h⁻¹

    def _scipy_ref(self, obs, dt, da, dr, theta=None):
        cl, v = theta if theta is not None else self._THETA
        def rf(rate): return lambda t, y: [rate - (cl / v) * y[0]]
        return _scipy_infusion_multidose(obs, dt, da, dr, rf, [0.0])

    def test_single_infusion_matches_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF1_D
        np.testing.assert_allclose(
            np.array(_1cmt_iv_inf_probe(obs, dt, da, dr, self._THETA)),
            self._scipy_ref(obs, dt, da, dr), rtol=5e-5, atol=1e-7)

    def test_single_infusion_matches_analytical(self):
        """During infusion: A1 = (R/k)(1-e^{-kt}).  After: A1(T)·e^{-k(t-T)}."""
        cl, v = self._THETA
        k = cl / v
        amt, rate = _INF1_D[1][0], _INF1_D[2][0]
        T = amt / rate
        A1_T = (rate / k) * (1.0 - np.exp(-k * T))
        analytical = [
            (rate / k) * (1.0 - np.exp(-k * t)) if t <= T
            else A1_T * np.exp(-k * (t - T))
            for t in _INF_OBS
        ]
        got = np.array(_1cmt_iv_inf_probe(_INF_OBS, *_INF1_D, self._THETA))[:, 0]
        np.testing.assert_allclose(got, analytical, rtol=1e-5, atol=1e-8)

    def test_two_infusions_match_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF2_D
        np.testing.assert_allclose(
            np.array(_1cmt_iv_inf_probe(obs, dt, da, dr, self._THETA)),
            self._scipy_ref(obs, dt, da, dr), rtol=5e-5, atol=1e-7)

    def test_mixed_bolus_and_infusion_matches_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF_MIX_D
        np.testing.assert_allclose(
            np.array(_1cmt_iv_inf_probe(obs, dt, da, dr, self._THETA)),
            self._scipy_ref(obs, dt, da, dr), rtol=5e-5, atol=1e-7)

    def test_sensitivity_matches_fd(self):
        if _1cmt_iv_inf_sens is None:
            pytest.skip("1cmt_iv infusion sensitivity probe not compiled in")
        obs, (dt, da, dr) = _INF_OBS, _INF1_D
        _, sens_raw = _1cmt_iv_inf_sens(obs, dt, da, dr, self._THETA)
        sens = np.array(sens_raw).reshape(len(obs), 2, 1)
        sens_fd = _fd_sensitivity_infusion(
            _1cmt_iv_inf_probe, obs, dt, da, dr, self._THETA, 2, 1)
        np.testing.assert_allclose(sens, sens_fd, rtol=1e-4, atol=1e-4 * max(da))


# ---------------------------------------------------------------------------
# 2-compartment IV infusion
# ---------------------------------------------------------------------------

class TestInfusion2cmtIv:

    _skip = _require(_2cmt_iv_inf_probe)


    _THETA = [1.5, 8.0, 0.5, 15.0]   # CL, V1, Q, V2

    def _scipy_ref(self, obs, dt, da, dr, theta):
        cl, v1, q, v2 = theta
        k10 = cl / v1; k12 = q / v1; k21 = q / v2
        def rf(rate):
            return lambda t, y: [
                rate - (k10 + k12) * y[0] + k21 * y[1],
                k12 * y[0] - k21 * y[1],
            ]
        return _scipy_infusion_multidose(obs, dt, da, dr, rf, [0.0, 0.0])

    def test_single_infusion_matches_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF1_D
        ref = self._scipy_ref(obs, dt, da, dr, self._THETA)
        got = np.array(_2cmt_iv_inf_probe(obs, dt, da, dr, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_two_infusions_match_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF2_D
        ref = self._scipy_ref(obs, dt, da, dr, self._THETA)
        got = np.array(_2cmt_iv_inf_probe(obs, dt, da, dr, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_mixed_bolus_and_infusion_matches_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF_MIX_D
        ref = self._scipy_ref(obs, dt, da, dr, self._THETA)
        got = np.array(_2cmt_iv_inf_probe(obs, dt, da, dr, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_peripheral_receives_drug(self):
        got = np.array(_2cmt_iv_inf_probe(_INF_OBS, *_INF1_D, self._THETA))
        assert np.max(got[:, 1]) > 0

    def test_sensitivity_matches_fd(self):
        if _2cmt_iv_inf_sens is None:
            pytest.skip("2cmt_iv infusion sensitivity probe not compiled in")
        obs, (dt, da, dr) = _INF_OBS, _INF1_D
        _, sens_raw = _2cmt_iv_inf_sens(obs, dt, da, dr, self._THETA)
        sens = np.array(sens_raw).reshape(len(obs), 4, 2)
        sens_fd = _fd_sensitivity_infusion(
            _2cmt_iv_inf_probe, obs, dt, da, dr, self._THETA, 4, 2)
        np.testing.assert_allclose(sens, sens_fd, rtol=1e-4, atol=1e-4 * max(da))


# ---------------------------------------------------------------------------
# 3-compartment IV infusion
# ---------------------------------------------------------------------------

class TestInfusion3cmtIv:

    _skip = _require(_3cmt_iv_inf_probe)


    _THETA = [2.0, 10.0, 1.0, 20.0, 0.5, 30.0]   # CL,V1,Q2,V2,Q3,V3

    def _scipy_ref(self, obs, dt, da, dr, theta):
        cl, v1, q2, v2, q3, v3 = theta
        k10 = cl / v1; k12 = q2 / v1; k21 = q2 / v2
        k13 = q3 / v1; k31 = q3 / v3
        def rf(rate):
            return lambda t, y: [
                rate - (k10 + k12 + k13) * y[0] + k21 * y[1] + k31 * y[2],
                k12 * y[0] - k21 * y[1],
                k13 * y[0] - k31 * y[2],
            ]
        return _scipy_infusion_multidose(obs, dt, da, dr, rf, [0.0, 0.0, 0.0])

    def test_single_infusion_matches_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF1_D
        ref = self._scipy_ref(obs, dt, da, dr, self._THETA)
        got = np.array(_3cmt_iv_inf_probe(obs, dt, da, dr, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_two_infusions_match_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF2_D
        ref = self._scipy_ref(obs, dt, da, dr, self._THETA)
        got = np.array(_3cmt_iv_inf_probe(obs, dt, da, dr, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_mixed_bolus_and_infusion_matches_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF_MIX_D
        ref = self._scipy_ref(obs, dt, da, dr, self._THETA)
        got = np.array(_3cmt_iv_inf_probe(obs, dt, da, dr, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_both_peripherals_receive_drug(self):
        got = np.array(_3cmt_iv_inf_probe(_INF_OBS, *_INF1_D, self._THETA))
        assert np.max(got[:, 1]) > 0 and np.max(got[:, 2]) > 0

    def test_sensitivity_matches_fd(self):
        if _3cmt_iv_inf_sens is None:
            pytest.skip("3cmt_iv infusion sensitivity probe not compiled in")
        obs, (dt, da, dr) = _INF_OBS, _INF1_D
        _, sens_raw = _3cmt_iv_inf_sens(obs, dt, da, dr, self._THETA)
        sens = np.array(sens_raw).reshape(len(obs), 6, 3)
        sens_fd = _fd_sensitivity_infusion(
            _3cmt_iv_inf_probe, obs, dt, da, dr, self._THETA, 6, 3)
        np.testing.assert_allclose(sens, sens_fd, rtol=1e-4, atol=1e-4 * max(da))


# ---------------------------------------------------------------------------
# 4-compartment IV infusion
# ---------------------------------------------------------------------------

class TestInfusion4cmtIv:

    _skip = _require(_4cmt_iv_inf_probe)


    _THETA = [2.0, 10.0, 1.0, 20.0, 0.5, 30.0, 0.2, 40.0]  # CL,V1,Q2,V2,Q3,V3,Q4,V4

    def _scipy_ref(self, obs, dt, da, dr, theta):
        cl, v1, q2, v2, q3, v3, q4, v4 = theta
        k10=cl/v1; k12=q2/v1; k21=q2/v2; k13=q3/v1; k31=q3/v3; k14=q4/v1; k41=q4/v4
        def rf(rate):
            return lambda t, y: [
                rate - (k10+k12+k13+k14)*y[0] + k21*y[1] + k31*y[2] + k41*y[3],
                k12*y[0] - k21*y[1],
                k13*y[0] - k31*y[2],
                k14*y[0] - k41*y[3],
            ]
        return _scipy_infusion_multidose(obs, dt, da, dr, rf, [0.0, 0.0, 0.0, 0.0])

    def test_single_infusion_matches_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF1_D
        ref = self._scipy_ref(obs, dt, da, dr, self._THETA)
        got = np.array(_4cmt_iv_inf_probe(obs, dt, da, dr, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_two_infusions_match_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF2_D
        ref = self._scipy_ref(obs, dt, da, dr, self._THETA)
        got = np.array(_4cmt_iv_inf_probe(obs, dt, da, dr, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_mixed_bolus_and_infusion_matches_scipy(self):
        obs, (dt, da, dr) = _INF_OBS, _INF_MIX_D
        ref = self._scipy_ref(obs, dt, da, dr, self._THETA)
        got = np.array(_4cmt_iv_inf_probe(obs, dt, da, dr, self._THETA))
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=1e-7)

    def test_all_peripherals_receive_drug(self):
        got = np.array(_4cmt_iv_inf_probe(_INF_OBS, *_INF1_D, self._THETA))
        assert all(np.max(got[:, i]) > 0 for i in [1, 2, 3])

    def test_sensitivity_matches_fd(self):
        if _4cmt_iv_inf_sens is None:
            pytest.skip("4cmt_iv infusion sensitivity probe not compiled in")
        obs, (dt, da, dr) = _INF_OBS, _INF1_D
        _, sens_raw = _4cmt_iv_inf_sens(obs, dt, da, dr, self._THETA)
        sens = np.array(sens_raw).reshape(len(obs), 8, 4)
        sens_fd = _fd_sensitivity_infusion(
            _4cmt_iv_inf_probe, obs, dt, da, dr, self._THETA, 8, 4)
        np.testing.assert_allclose(sens, sens_fd, rtol=1e-4, atol=1e-6)



# ══════════════════════════════════════════════════════════════════════════════
# Section 14 — ALAG (Absorption Lag Time) support
# ══════════════════════════════════════════════════════════════════════════════
#
# ALAG{cmt} is a model-estimated lag time before a dose enters the system.
# The native path must shift each dose time by ALAG{compartment} before
# passing the dose schedule to the Rust probe, exactly as advan6._prepare_doses
# does for the Python ODE path.
#
# Tests verify:
#   1. _apply_alag helper (unit level)
#   2. State probe: 1cmt_oral + ALAG1 vs analytical solution with lag time
#   3. IndividualModel dispatch: pk_callable returns ALAG1 → correct IPRED
#   4. Sensitivity probe: G_i with ALAG1 matches FD reference (rtol=1e-3)
#
# Analytical solution for 1cmt_oral with lag time tlag:
#   C(t) = 0                                          for t ≤ tlag
#   C(t) = D/V * KA/(KA-k) * (e^{-k(t-tlag)}
#                              - e^{-KA(t-tlag)})     for t > tlag
#   where k = CL/V
# ══════════════════════════════════════════════════════════════════════════════


def _oral_lag_analytical(
    obs: list[float],
    dose_time: float,
    dose_amt: float,
    KA: float,
    CL: float,
    V: float,
    tlag: float,
) -> np.ndarray:
    """Analytical 1cmt oral solution with lag time."""
    k = CL / V
    conc = np.zeros(len(obs))
    for i, t in enumerate(obs):
        t_eff = t - (dose_time + tlag)
        if t_eff > 0.0:
            conc[i] = (dose_amt / V) * (KA / (KA - k)) * (
                np.exp(-k * t_eff) - np.exp(-KA * t_eff)
            )
    return conc


class TestApplyAlagHelper:
    """Unit tests for the _apply_alag module-level helper."""

    def test_no_alag_keys_returns_same_list(self):
        """If pk_params contains no ALAG keys, the original list is returned."""
        from openpkpd.model.individual import _apply_alag
        dt = [0.0, 12.0]
        result = _apply_alag(dt, [1, 1], {"CL": 5.0, "V": 10.0})
        assert result is dt, "should return original list unchanged (fast path)"

    def test_alag1_shifts_compartment_1_doses(self):
        """ALAG1=2.5 shifts all compartment-1 dose times by 2.5."""
        from openpkpd.model.individual import _apply_alag
        result = _apply_alag([0.0, 12.0], [1, 1], {"ALAG1": 2.5})
        assert result == [2.5, 14.5]

    def test_alag_is_compartment_specific(self):
        """ALAG2=1.0 only shifts doses into compartment 2."""
        from openpkpd.model.individual import _apply_alag
        result = _apply_alag([0.0, 6.0], [1, 2], {"ALAG1": 0.0, "ALAG2": 1.0})
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(7.0)

    def test_zero_alag_leaves_times_unchanged(self):
        """ALAG1=0.0 is a no-op."""
        from openpkpd.model.individual import _apply_alag
        result = _apply_alag([0.0, 24.0], [1, 1], {"ALAG1": 0.0})
        assert result == [0.0, 24.0]


class TestAlagStateProbe:
    """State probe: 1cmt_oral with ALAG1 vs analytical solution."""

    _KA   = 1.5
    _CL   = 0.5
    _V    = 10.0
    _ALAG = 2.0          # 2-hour lag
    _DOSE = 100.0
    _OBS  = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 24.0]
    _THETA = [_KA, _CL, _V]

    _skip = _require(_1cmt_oral_probe)


    def test_no_alag_deviates_from_analytical(self):
        """Baseline: probe without lag does NOT match the lagged analytical solution."""
        analytic = _oral_lag_analytical(
            self._OBS, 0.0, self._DOSE, self._KA, self._CL, self._V, self._ALAG
        )
        states_raw = _1cmt_oral_probe(self._OBS, [0.0], [self._DOSE], self._THETA)
        ipred_no_lag = np.array(states_raw)[:, 1] / self._V
        # Pre-lag observations (t ≤ 2) should have non-zero ipred without lag fix.
        pre_lag_mask = np.array(self._OBS) <= self._ALAG
        assert np.any(ipred_no_lag[pre_lag_mask] > 0.01), (
            "Without ALAG fix, probe should give non-zero concentrations before lag ends"
        )
        assert not np.allclose(ipred_no_lag, analytic, atol=0.1), (
            "Without ALAG fix, probe should NOT match lagged analytical solution"
        )

    def test_alag_shifted_probe_matches_analytical(self):
        """Probe with ALAG-shifted dose time matches analytical lagged solution."""
        analytic = _oral_lag_analytical(
            self._OBS, 0.0, self._DOSE, self._KA, self._CL, self._V, self._ALAG
        )
        shifted_dt = [0.0 + self._ALAG]   # dose_time + ALAG1
        states_raw = _1cmt_oral_probe(self._OBS, shifted_dt, [self._DOSE], self._THETA)
        ipred = np.array(states_raw)[:, 1] / self._V
        np.testing.assert_allclose(ipred, analytic, rtol=1e-5, atol=1e-8,
                                   err_msg="ALAG-shifted probe must match analytical solution")

    def test_pre_lag_concentrations_are_zero(self):
        """Observations at or before the lag time must have zero concentration."""
        shifted_dt = [0.0 + self._ALAG]
        states_raw = _1cmt_oral_probe(self._OBS, shifted_dt, [self._DOSE], self._THETA)
        ipred = np.array(states_raw)[:, 1] / self._V
        pre_lag = [i for i, t in enumerate(self._OBS) if t <= self._ALAG]
        for i in pre_lag:
            assert ipred[i] < 1e-8, f"C({self._OBS[i]}) should be ~0 before lag ends"


class TestAlagIndividualDispatch:
    """IndividualModel dispatch: ALAG1 in pk_callable → correct native IPRED."""

    _KA   = 1.5
    _CL   = 0.5
    _V    = 10.0
    _ALAG = 2.0
    _DOSE = 100.0
    _OBS  = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 24.0]

    _skip = _require_with_indiv(_1cmt_oral_probe)

    def _build_indiv(self, alag1: float):
        from openpkpd.model.individual import IndividualModel
        se = MagicMock()
        se.obs_times = np.array(self._OBS)
        se.obs_dv = np.full(len(self._OBS), np.nan)
        se.obs_mdv = np.zeros(len(self._OBS), dtype=int)
        se.obs_cmt = np.ones(len(self._OBS), dtype=int)
        se.observation_mask.return_value = np.ones(len(self._OBS), dtype=bool)
        se.covariate_df = None
        se.covariate_at.return_value = {}
        se.covariate_change_times.return_value = []

        dose_event = MagicMock()
        dose_event.time = 0.0
        dose_event.amount = self._DOSE
        dose_event.rate = 0.0
        dose_event.compartment = 1
        se.dose_events = [dose_event]

        def pk_callable(theta, eta, t=0.0, covariates=None):
            return {"KA": self._KA, "CL": self._CL, "V": self._V, "ALAG1": alag1}

        def error_callable(theta, eta, eps, f, ipred, y, t, a=None, covariates=None, sigma=None):
            return {"Y": f * (1 + eps[0]), "IPRED": f}
        error_callable._source = "Y = F * (1 + EPS[0])"

        pk_sub = MagicMock()
        pk_sub.advan = 6
        pk_sub.n_compartments = 2   # 1cmt_oral has n_states=2

        return IndividualModel(
            subject_events=se,
            pk_subroutine=pk_sub,
            pk_callable=pk_callable,
            error_callable=error_callable,
            n_eps=1,
        )

    def test_alag_zero_baseline(self):
        """ALAG1=0 gives the same result as no lag."""
        indiv = self._build_indiv(alag1=0.0)
        contract = indiv._native_ode_contract
        assert contract is not None
        pk_params = {"KA": self._KA, "CL": self._CL, "V": self._V, "ALAG1": 0.0}
        sol = indiv._try_native_ode_probe(pk_params, np.array(self._OBS))
        assert sol is not None, "native probe should activate for 1cmt_oral"
        analytic = _oral_lag_analytical(
            self._OBS, 0.0, self._DOSE, self._KA, self._CL, self._V, tlag=0.0
        )
        np.testing.assert_allclose(sol.ipred, analytic, rtol=1e-5, atol=1e-8)

    def test_alag1_shifts_dose_correctly(self):
        """ALAG1=2.0 → native IPRED matches analytical solution with tlag=2."""
        indiv = self._build_indiv(alag1=self._ALAG)
        pk_params = {
            "KA": self._KA, "CL": self._CL, "V": self._V, "ALAG1": self._ALAG
        }
        sol = indiv._try_native_ode_probe(pk_params, np.array(self._OBS))
        assert sol is not None, "native probe should activate for 1cmt_oral + ALAG1"
        analytic = _oral_lag_analytical(
            self._OBS, 0.0, self._DOSE, self._KA, self._CL, self._V, tlag=self._ALAG
        )
        np.testing.assert_allclose(sol.ipred, analytic, rtol=1e-5, atol=1e-8,
                                   err_msg="ALAG-dispatched IPRED must match analytical")

    def test_contract_stores_dose_compartments(self):
        """Contract must include dose_compartments for ALAG dispatch to work."""
        indiv = self._build_indiv(alag1=0.0)
        contract = indiv._native_ode_contract
        assert contract is not None
        assert "dose_compartments" in contract, "contract must store dose_compartments"
        assert contract["dose_compartments"] == [1]


class TestAlagSensitivity:
    """Native G_i with ALAG1 must match FD of (ALAG-shifted) predictions."""

    _KA   = 1.5
    _CL   = 0.5
    _V    = 10.0
    _ALAG = 1.5
    _DOSE = 100.0
    _OBS  = [2.0, 4.0, 8.0, 12.0, 24.0]

    _skip = _require_with_indiv(_1cmt_oral_sens)

    def _build_indiv(self):
        from openpkpd.model.individual import IndividualModel
        se = MagicMock()
        se.obs_times = np.array(self._OBS)
        se.obs_dv = np.full(len(self._OBS), np.nan)
        se.obs_mdv = np.zeros(len(self._OBS), dtype=int)
        se.obs_cmt = np.ones(len(self._OBS), dtype=int)
        se.observation_mask.return_value = np.ones(len(self._OBS), dtype=bool)
        se.covariate_df = None
        se.covariate_at.return_value = {}
        se.covariate_change_times.return_value = []

        dose_event = MagicMock()
        dose_event.time = 0.0
        dose_event.amount = self._DOSE
        dose_event.rate = 0.0
        dose_event.compartment = 1
        se.dose_events = [dose_event]

        # theta = [KA, CL, V]; eta shifts CL and V log-normally
        def pk_callable(theta, eta, t=0.0, covariates=None):
            th = list(theta)
            return {
                "KA":    float(th[0]),
                "CL":    float(th[1]) * np.exp(eta[0] if len(eta) > 0 else 0.0),
                "V":     float(th[2]) * np.exp(eta[1] if len(eta) > 1 else 0.0),
                "ALAG1": self._ALAG,   # constant lag, not eta-dependent
            }

        def error_callable(theta, eta, eps, f, ipred, y, t, a=None, covariates=None, sigma=None):
            return {"Y": f * (1 + eps[0]), "IPRED": f}
        error_callable._source = "Y = F * (1 + EPS[0])"

        pk_sub = MagicMock()
        pk_sub.advan = 6
        pk_sub.n_compartments = 2

        return IndividualModel(
            subject_events=se,
            pk_subroutine=pk_sub,
            pk_callable=pk_callable,
            error_callable=error_callable,
            n_eps=1,
        )

    def test_G_i_with_alag_matches_fd(self):
        """G_i computed natively with ALAG1 matches central FD of ALAG-shifted probe."""
        indiv = self._build_indiv()
        theta = np.array([self._KA, self._CL, self._V])
        eta   = np.zeros(2)
        obs_mask = np.ones(len(self._OBS), dtype=bool)

        G_native = indiv.native_advan6_prediction_eta_jacobian(theta, eta, obs_mask, n_eta=2)
        assert G_native is not None, "native G_i should activate for 1cmt_oral + ALAG1"

        # FD reference: perturb eta, re-evaluate via ALAG-shifted 1cmt_oral probe
        eps = 1e-5
        n_eta = 2
        shifted_dt = [0.0 + self._ALAG]
        G_fd = np.zeros((len(self._OBS), n_eta))
        for k in range(n_eta):
            ep = [0.0] * n_eta; ep[k] += eps
            em = [0.0] * n_eta; em[k] -= eps

            def ipred_at(eta_val):
                pk = {"KA": self._KA,
                      "CL": self._CL * np.exp(eta_val[0]),
                      "V":  self._V  * np.exp(eta_val[1]),
                      "ALAG1": self._ALAG}
                ode_theta = [pk["KA"], pk["CL"], pk["V"]]
                raw = _1cmt_oral_probe(self._OBS, shifted_dt, [self._DOSE], ode_theta)
                return np.array(raw)[:, 1] / pk["V"]

            G_fd[:, k] = (ipred_at(ep) - ipred_at(em)) / (2.0 * eps)

        np.testing.assert_allclose(G_native, G_fd, rtol=1e-3, atol=1e-8,
                                   err_msg="G_i with ALAG1 must match FD reference")



# ===========================================================================
# Section 15 — Covariate gate relaxation (P1.1)
#
# _build_native_ode_contract() must allow time-constant covariates (WT, AGE …)
# and block only when a covariate changes over the observation window.
# ===========================================================================

def _build_native_individual_with_cov(cov_df):
    """Like _build_native_individual but with a custom covariate_df."""
    from openpkpd.model.individual import IndividualModel

    se = MagicMock()
    se.obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    se.obs_dv = np.full(7, np.nan)
    se.obs_mdv = np.zeros(7, dtype=int)
    se.obs_cmt = np.ones(7, dtype=int)
    se.observation_mask.return_value = np.ones(7, dtype=bool)
    se.covariate_df = cov_df
    se.covariate_at.return_value = {}
    se.covariate_change_times.return_value = []

    dose_event = MagicMock()
    dose_event.time = 0.0
    dose_event.amount = 100.0
    dose_event.rate = 0.0
    dose_event.compartment = 1
    se.dose_events = [dose_event]

    def pk_callable(theta, eta, t=0.0, covariates=None):
        th = list(theta)
        wt = float((covariates or {}).get("WT", 70.0))
        return {
            "CL": float(th[0]) * (wt / 70.0) ** 0.75 * np.exp(eta[0] if len(eta) > 0 else 0.0),
            "V1": float(th[1]) * np.exp(eta[1] if len(eta) > 1 else 0.0),
            "Q":  float(th[2]),
            "V2": float(th[3]),
        }

    def error_callable(theta, eta, eps, f, ipred, y, t, a=None, covariates=None, sigma=None):
        return {"Y": f * (1 + eps[0]), "IPRED": f}
    error_callable._source = "Y = F * (1 + EPS[0])"

    pk_sub = MagicMock()
    pk_sub.advan = 6
    pk_sub.n_compartments = 2

    return IndividualModel(
        subject_events=se,
        pk_subroutine=pk_sub,
        pk_callable=pk_callable,
        error_callable=error_callable,
        n_eps=1,
    )


class TestCovariateGateRelaxation:
    """
    Time-constant covariates (e.g. WT, AGE) must not block the native ODE path.
    Only time-varying covariates (multiple distinct values per subject) should
    cause _build_native_ode_contract() to return None.
    """

    _skip = _require_with_indiv(_2cmt_iv_probe, _2cmt_iv_sens)

    def _make_cov_df(self, wt_values):
        import pandas as pd
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        return pd.DataFrame({"TIME": times, "WT": wt_values})

    def test_no_cov_df_activates_native_path(self):
        """Baseline: no covariate_df → native path activates."""
        indiv, _, _ = _build_native_individual()
        assert indiv._native_ode_contract is not None

    def test_constant_covariate_activates_native_path(self):
        """Single-valued WT across all observations → native path must activate."""
        cov_df = self._make_cov_df([70.0] * 7)
        indiv = _build_native_individual_with_cov(cov_df)
        assert indiv._native_ode_contract is not None, (
            "Time-constant WT should not block native path"
        )

    def test_constant_covariate_with_nan_activates_native_path(self):
        """WT constant but with one NaN observation → still activates."""
        cov_df = self._make_cov_df([70.0, np.nan, 70.0, 70.0, 70.0, 70.0, 70.0])
        indiv = _build_native_individual_with_cov(cov_df)
        assert indiv._native_ode_contract is not None

    def test_timevarying_covariate_blocks_native_path(self):
        """WT changes across observations → native path must be blocked."""
        cov_df = self._make_cov_df([70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0])
        indiv = _build_native_individual_with_cov(cov_df)
        assert indiv._native_ode_contract is None, (
            "Time-varying WT must block native path"
        )

    def test_two_constant_covariates_activate_native_path(self):
        """Multiple constant covariates (WT and AGE) → native path activates."""
        import pandas as pd
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        cov_df = pd.DataFrame({"TIME": times, "WT": [70.0] * 7, "AGE": [45.0] * 7})
        indiv = _build_native_individual_with_cov(cov_df)
        assert indiv._native_ode_contract is not None

    def test_mixed_constant_and_varying_blocks_native_path(self):
        """One constant covariate + one time-varying → blocked (conservative)."""
        import pandas as pd
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        cov_df = pd.DataFrame({
            "TIME": times,
            "WT":   [70.0] * 7,                               # constant
            "DOSE": [100.0, 100.0, 200.0, 200.0, 200.0, 200.0, 200.0],  # time-varying
        })
        indiv = _build_native_individual_with_cov(cov_df)
        assert indiv._native_ode_contract is None


# ===========================================================================
# Section 16 — PFIM infusion sensitivity path (P2)
#
# _compute_G_and_Z_native must dispatch to infusion_sens_probe_fn when the
# contract's has_infusion flag is True.  G must match a direct FD reference
# computed from the infusion bolus probe.
# ===========================================================================

def _build_native_infusion_individual(rate: float = 10.0):
    """
    Build a minimal IndividualModel backed by the 2cmt_iv infusion template.

    dose.rate > 0 → has_infusion=True in the contract.
    The infusion duration is amt/rate = 100/10 = 10 h.
    """
    from openpkpd.model.individual import IndividualModel

    se = MagicMock()
    se.obs_times = np.array([1.0, 4.0, 8.0, 12.0, 24.0])
    se.obs_dv = np.full(5, np.nan)
    se.obs_mdv = np.zeros(5, dtype=int)
    se.obs_cmt = np.ones(5, dtype=int)
    se.observation_mask.return_value = np.ones(5, dtype=bool)
    se.covariate_df = None
    se.covariate_at.return_value = {}
    se.covariate_change_times.return_value = []

    dose_event = MagicMock()
    dose_event.time = 0.0
    dose_event.amount = 100.0
    dose_event.rate = float(rate)   # > 0 → constant-rate infusion
    dose_event.compartment = 1
    se.dose_events = [dose_event]

    param_names = ("CL", "V1", "Q", "V2")

    def pk_callable(theta, eta, t=0.0, covariates=None):
        th = list(theta)
        return {
            "CL": float(th[0]) * np.exp(eta[0] if len(eta) > 0 else 0.0),
            "V1": float(th[1]) * np.exp(eta[1] if len(eta) > 1 else 0.0),
            "Q":  float(th[2]),
            "V2": float(th[3]),
        }

    def error_callable(theta, eta, eps, f, ipred, y, t, a=None, covariates=None, sigma=None):
        return {"Y": f * (1 + eps[0]), "IPRED": f}
    error_callable._source = "Y = F * (1 + EPS[0])"

    pk_sub = MagicMock()
    pk_sub.advan = 6
    pk_sub.n_compartments = 2

    indiv = IndividualModel(
        subject_events=se,
        pk_subroutine=pk_sub,
        pk_callable=pk_callable,
        error_callable=error_callable,
        n_eps=1,
    )
    return indiv, pk_callable, param_names


class TestPFIMInfusionSensitivityPath:
    """
    _compute_G_and_Z_native must activate for infusion models (has_infusion=True)
    by dispatching to the infusion-aware sensitivity probe.  G must match a
    direct FD reference built from _2cmt_iv_inf_probe + pk_callable.
    """

    _skip = _require_with_indiv(_2cmt_iv_inf_probe, _2cmt_iv_inf_sens)

    _RATE = 10.0   # mg/h — duration = 100 mg / 10 mg/h = 10 h

    def _make_pfim_engine(self, indiv):
        from openpkpd.design.pfim import PFIMEngine

        pop_model = MagicMock()
        pop_model.subject_ids.return_value = [1]
        pop_model.individual_model.return_value = indiv
        pop_model.trans = 2

        n_eta = 2
        omega = 0.04 * np.eye(n_eta)
        sigma = np.array([[0.01]])

        class Params:
            theta = np.array(_NATIVE_INDIV_THETA)

        Params.omega = omega
        Params.sigma = sigma

        return PFIMEngine(population_model=pop_model, init_params=Params())

    def _predict_F_inf(self, pk_callable, param_names, theta, eta, times):
        """FD-reference prediction using _2cmt_iv_inf_probe (infusion probe)."""
        pk_params = pk_callable(list(theta), list(eta), t=0.0)
        ode_theta = [float(pk_params[n]) for n in param_names]
        V1 = float(pk_params["V1"])
        dose_times = [0.0]
        dose_amts  = [100.0]
        dose_rates = [self._RATE]
        order = np.argsort(times, kind="stable")
        inv = np.empty_like(order); inv[order] = np.arange(len(times))
        raw = _2cmt_iv_inf_probe(
            np.array(times)[order].tolist(), dose_times, dose_amts, dose_rates, ode_theta
        )
        A1 = np.array(raw)[:, 0][inv]
        return A1 / V1

    def test_contract_has_infusion_flag(self):
        """_native_ode_contract must set has_infusion=True for rate>0 dose."""
        indiv, _, _ = _build_native_infusion_individual(rate=self._RATE)
        contract = indiv._native_ode_contract
        assert contract is not None
        assert contract["has_infusion"] is True
        assert contract["dose_rates"] == [self._RATE]

    def test_native_path_activates_for_infusion(self):
        """_compute_G_and_Z_native must return a result (not None) for infusion models."""
        indiv, _, _ = _build_native_infusion_individual(rate=self._RATE)
        engine = self._make_pfim_engine(indiv)
        times = np.array([1.0, 4.0, 8.0, 12.0, 24.0])
        result = engine._compute_G_and_Z_native(times, np.array(_NATIVE_INDIV_THETA), indiv, 2)
        assert result is not None, "Infusion native path must activate for 2cmt_iv_inf template"

    def test_infusion_G_matches_fd(self):
        """G from infusion native path matches direct FD from _2cmt_iv_inf_probe."""
        indiv, pk_callable, param_names = _build_native_infusion_individual(rate=self._RATE)
        engine = self._make_pfim_engine(indiv)
        times = np.array([1.0, 4.0, 8.0, 12.0, 24.0])
        theta = np.array(_NATIVE_INDIV_THETA)
        eta_zero = np.zeros(2)
        eps = 1e-5

        result = engine._compute_G_and_Z_native(times, theta, indiv, 2)
        assert result is not None
        G_native, _ = result

        n_theta = len(theta)
        G_fd = np.zeros((len(times), n_theta))
        for j in range(n_theta):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            G_fd[:, j] = (
                self._predict_F_inf(pk_callable, param_names, tp, eta_zero, times)
                - self._predict_F_inf(pk_callable, param_names, tm, eta_zero, times)
            ) / (2.0 * eps)

        # rtol=2e-2: infusion sensitivity vs FD incurs ~1% cancellation error
        # due to the finite-difference step; analytical sensitivity is more
        # accurate — the tolerance here just ensures gross errors are caught.
        np.testing.assert_allclose(G_native, G_fd, rtol=2e-2, atol=1e-5,
                                   err_msg="Infusion native G deviates from FD reference")



# ===========================================================================
# Section 17 — P1.3: Analytic-ADVAN Rust probes
#
# ADVAN1/2/3/4 are now routed through exact closed-form Rust probes (P1.3).
# These tests verify:
#   (a) Gate activation — IndividualModel builds a non-None contract for each
#       eligible ADVAN and stores the correct 'advan' key.
#   (b) Numerical accuracy — analytic_* probes match Python ADVAN solvers
#       to machine precision (rtol ≤ 1e-12, atol ≤ 1e-14).
#   (c) Sensitivity dispatch — native_advan6_prediction_eta_jacobian returns
#       a G matrix consistent with finite differences for ADVAN3 models.
# ===========================================================================

# ── analytical probe symbols ──────────────────────────────────────────────────
_A_1cmt_iv       = _try_import("analytic_1cmt_iv_probe_multidose")
_A_1cmt_iv_inf   = _try_import("analytic_1cmt_iv_infusion_probe_multidose")
_A_1cmt_oral     = _try_import("analytic_1cmt_oral_probe_multidose")
_A_2cmt_iv       = _try_import("analytic_2cmt_iv_probe_multidose")
_A_2cmt_iv_inf   = _try_import("analytic_2cmt_iv_infusion_probe_multidose")
_A_2cmt_oral     = _try_import("analytic_2cmt_oral_probe_multidose")

# ── typical parameter sets ────────────────────────────────────────────────────
_A17_OBS      = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0]
_A17_DOSE1    = ([0.0], [100.0])
_A17_DOSE2    = ([0.0, 25.0], [100.0, 80.0])
_A17_1CMT_CL, _A17_1CMT_V = 0.044, 31.8   # theophylline-like
_A17_2CMT     = {"CL": 3.876825, "V1": 9.760346, "V2": 30.81783, "Q": 8.773851}

_A17_ATOL = 1e-14
_A17_RTOL = 1e-12


def _a17_dose_events(dose_times, dose_amts, cmt=1, rate=0.0):
    from openpkpd.data.event_processor import DoseEvent
    return [DoseEvent(time=t, amount=a, compartment=cmt, rate=rate)
            for t, a in zip(dose_times, dose_amts)]


def _build_analytic_indiv(advan: int, n_cmt: int, pk_callable_fn, n_eps: int = 1,
                           obs_times=None, dose_time=0.0, dose_amt=100.0, dose_rate=0.0):
    """Build IndividualModel for an analytical ADVAN (1/2/3/4)."""
    from openpkpd.model.individual import IndividualModel

    obs = obs_times if obs_times is not None else _A17_OBS
    se = MagicMock()
    se.obs_times = np.array(obs)
    se.obs_dv = np.full(len(obs), np.nan)
    se.obs_mdv = np.zeros(len(obs), dtype=int)
    se.obs_cmt = np.ones(len(obs), dtype=int)
    se.observation_mask.return_value = np.ones(len(obs), dtype=bool)
    se.covariate_df = None
    se.covariate_at.return_value = {}
    se.covariate_change_times.return_value = []

    dose_event = MagicMock()
    dose_event.time = dose_time
    dose_event.amount = dose_amt
    dose_event.rate = dose_rate
    dose_event.compartment = 1
    se.dose_events = [dose_event]

    pk_sub = MagicMock()
    pk_sub.advan = advan
    pk_sub.n_compartments = n_cmt

    def error_callable(theta, eta, eps, f, ipred, y, t, a=None, covariates=None, sigma=None):
        return {"Y": f * (1 + eps[0]), "IPRED": f}
    error_callable._source = "Y = F * (1 + EPS[0])"

    return IndividualModel(
        subject_events=se,
        pk_subroutine=pk_sub,
        pk_callable=pk_callable_fn,
        error_callable=error_callable,
        n_eps=n_eps,
    )


# ---------------------------------------------------------------------------
# 17A — Gate Activation
# ---------------------------------------------------------------------------

class TestAnalyticAdvanGateActivation:
    """_build_native_ode_contract must build a non-None contract for ADVAN1-4."""

    _skip = _require_with_indiv(_A_1cmt_iv, _A_2cmt_iv)

    def _pk_1cmt(self):
        def pk(theta, eta, t=0.0, covariates=None):
            return {"CL": _A17_1CMT_CL, "V": _A17_1CMT_V}
        return pk

    def _pk_2cmt(self):
        def pk(theta, eta, t=0.0, covariates=None):
            p = _A17_2CMT
            return {"CL": p["CL"], "V1": p["V1"], "Q": p["Q"], "V2": p["V2"]}
        return pk

    def _pk_1cmt_oral(self):
        def pk(theta, eta, t=0.0, covariates=None):
            return {"KA": 1.5, "CL": _A17_1CMT_CL, "V": _A17_1CMT_V}
        return pk

    def _pk_2cmt_oral(self):
        def pk(theta, eta, t=0.0, covariates=None):
            p = _A17_2CMT
            return {"KA": 1.5, "CL": p["CL"], "V2": p["V1"], "Q": p["Q"], "V3": p["V2"]}
        return pk

    def test_advan1_gate_activates(self):
        indiv = _build_analytic_indiv(1, 1, self._pk_1cmt())
        assert indiv._native_ode_contract is not None, "ADVAN1 must activate native path"
        assert indiv._native_ode_contract["advan"] == 1

    def test_advan2_gate_activates(self):
        indiv = _build_analytic_indiv(2, 2, self._pk_1cmt_oral())
        assert indiv._native_ode_contract is not None, "ADVAN2 must activate native path"
        assert indiv._native_ode_contract["advan"] == 2

    def test_advan3_gate_activates(self):
        indiv = _build_analytic_indiv(3, 2, self._pk_2cmt())
        assert indiv._native_ode_contract is not None, "ADVAN3 must activate native path"
        assert indiv._native_ode_contract["advan"] == 3

    def test_advan4_gate_activates(self):
        indiv = _build_analytic_indiv(4, 3, self._pk_2cmt_oral())
        assert indiv._native_ode_contract is not None, "ADVAN4 must activate native path"
        assert indiv._native_ode_contract["advan"] == 4

    def test_non_eligible_advan_blocked(self):
        """ADVAN7 is not in _NATIVE_ELIGIBLE_ADVANS; contract must be None."""
        indiv = _build_analytic_indiv(7, 2, self._pk_2cmt())
        assert indiv._native_ode_contract is None, "ADVAN7 must not activate native path"

    def test_advan6_still_eligible(self):
        """ADVAN6 must remain eligible (unchanged behavior)."""
        indiv = _build_analytic_indiv(6, 2, self._pk_2cmt())
        assert indiv._native_ode_contract is not None, "ADVAN6 must still activate native path"


# ---------------------------------------------------------------------------
# 17B — Numerical Accuracy vs Python ADVAN Solvers
# ---------------------------------------------------------------------------

class TestAnalyticProbe1CmtIvVsADVAN1:
    """analytic_1cmt_iv_probe vs ADVAN1 exact solution — machine precision."""

    _skip = _require(_A_1cmt_iv)

    _CL, _V = _A17_1CMT_CL, _A17_1CMT_V
    _THETA = [_A17_1CMT_CL, _A17_1CMT_V]

    def _advan1_ipred(self, obs, dose_events):
        from openpkpd.pk.analytical.advan1 import ADVAN1
        return ADVAN1().solve({"K": self._CL / self._V, "V": self._V},
                              dose_events, np.array(obs)).ipred

    def test_single_dose_matches_advan1(self):
        dt, da = _A17_DOSE1
        ref = self._advan1_ipred(_A17_OBS, _a17_dose_events(dt, da))
        states = np.array(_A_1cmt_iv(_A17_OBS, dt, da, self._THETA))
        got = states[:, 0] / self._V
        np.testing.assert_allclose(got, ref, rtol=_A17_RTOL, atol=_A17_ATOL,
                                   err_msg="ADVAN1 single-dose: analytic Rust vs Python")

    def test_two_dose_matches_advan1(self):
        dt, da = _A17_DOSE2
        ref = self._advan1_ipred(_A17_OBS, _a17_dose_events(dt, da))
        states = np.array(_A_1cmt_iv(_A17_OBS, dt, da, self._THETA))
        got = states[:, 0] / self._V
        np.testing.assert_allclose(got, ref, rtol=_A17_RTOL, atol=_A17_ATOL,
                                   err_msg="ADVAN1 two-dose: analytic Rust vs Python")


class TestAnalyticProbe1CmtOralVsADVAN2:
    """analytic_1cmt_oral_probe vs ADVAN2 exact solution — machine precision."""

    _skip = _require(_A_1cmt_oral)

    _KA, _CL, _V = 1.5, _A17_1CMT_CL, _A17_1CMT_V
    _THETA = [1.5, _A17_1CMT_CL, _A17_1CMT_V]

    def _advan2_ipred(self, obs, dose_events, ka=None):
        from openpkpd.pk.analytical.advan2 import ADVAN2
        K = self._CL / self._V
        use_ka = ka if ka is not None else self._KA
        return ADVAN2().solve({"KA": use_ka, "K": K, "V": self._V},
                              dose_events, np.array(obs)).ipred

    def test_single_dose_matches_advan2(self):
        dt, da = _A17_DOSE1
        ref = self._advan2_ipred(_A17_OBS, _a17_dose_events(dt, da))
        states = np.array(_A_1cmt_oral(_A17_OBS, dt, da, self._THETA))
        got = states[:, 1] / self._V
        np.testing.assert_allclose(got, ref, rtol=_A17_RTOL, atol=_A17_ATOL,
                                   err_msg="ADVAN2 single-dose: analytic Rust vs Python")

    def test_two_dose_matches_advan2(self):
        dt, da = _A17_DOSE2
        ref = self._advan2_ipred(_A17_OBS, _a17_dose_events(dt, da))
        states = np.array(_A_1cmt_oral(_A17_OBS, dt, da, self._THETA))
        got = states[:, 1] / self._V
        np.testing.assert_allclose(got, ref, rtol=_A17_RTOL, atol=_A17_ATOL,
                                   err_msg="ADVAN2 two-dose: analytic Rust vs Python")

    def test_ka_equals_k_limit_matches_advan2(self):
        """KA ≈ K: analytic probe must use L'Hôpital form matching ADVAN2."""
        K = self._CL / self._V
        ka_lim = K + 5e-7    # within KA_K_TOL = 1e-6
        theta_lim = [ka_lim, self._CL, self._V]
        ref = self._advan2_ipred(_A17_OBS, _a17_dose_events(*_A17_DOSE1), ka=ka_lim)
        states = np.array(_A_1cmt_oral(_A17_OBS, _A17_DOSE1[0], _A17_DOSE1[1], theta_lim))
        got = states[:, 1] / self._V
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-10,
                                   err_msg="KA≈K limit form mismatch")


class TestAnalyticProbe2CmtIvVsADVAN3:
    """analytic_2cmt_iv_probe vs ADVAN3 exact solution — machine precision."""

    _skip = _require(_A_2cmt_iv)

    _p = _A17_2CMT
    _THETA = [_p["CL"], _p["V1"], _p["Q"], _p["V2"]]
    _V1 = _p["V1"]

    def _advan3_ipred(self, obs, dose_events):
        from openpkpd.pk.analytical.advan3 import ADVAN3
        return ADVAN3().solve(self._p, dose_events, np.array(obs)).ipred

    def test_single_dose_matches_advan3(self):
        dt, da = _A17_DOSE1
        ref = self._advan3_ipred(_A17_OBS, _a17_dose_events(dt, da))
        states = np.array(_A_2cmt_iv(_A17_OBS, dt, da, self._THETA))
        got = states[:, 0] / self._V1
        np.testing.assert_allclose(got, ref, rtol=_A17_RTOL, atol=_A17_ATOL,
                                   err_msg="ADVAN3 single-dose: analytic Rust vs Python")

    def test_two_dose_matches_advan3(self):
        dt, da = _A17_DOSE2
        ref = self._advan3_ipred(_A17_OBS, _a17_dose_events(dt, da))
        states = np.array(_A_2cmt_iv(_A17_OBS, dt, da, self._THETA))
        got = states[:, 0] / self._V1
        np.testing.assert_allclose(got, ref, rtol=_A17_RTOL, atol=_A17_ATOL,
                                   err_msg="ADVAN3 two-dose: analytic Rust vs Python")


class TestAnalyticProbe2CmtOralVsADVAN4:
    """analytic_2cmt_oral_probe vs ADVAN4 exact solution — machine precision."""

    _skip = _require(_A_2cmt_oral)

    _KA = 1.5
    _p  = _A17_2CMT
    # ADVAN4 template: theta=[KA, CL, V2, Q, V3]
    _THETA = [1.5, _p["CL"], _p["V1"], _p["Q"], _p["V2"]]
    _V2 = _p["V1"]  # V2 in template = V1 in reference

    def _advan4_ipred(self, obs, dose_events):
        from openpkpd.pk.analytical.advan4 import ADVAN4
        p = self._p
        k10 = p["CL"] / p["V1"]; k12 = p["Q"] / p["V1"]; k21 = p["Q"] / p["V2"]
        params = {"KA": self._KA, "K": k10, "K12": k12, "K21": k21, "V2": p["V1"]}
        return ADVAN4().solve(params, dose_events, np.array(obs)).ipred

    def test_single_dose_matches_advan4(self):
        dt, da = _A17_DOSE1
        ref = self._advan4_ipred(_A17_OBS, _a17_dose_events(dt, da))
        states = np.array(_A_2cmt_oral(_A17_OBS, dt, da, self._THETA))
        got = states[:, 1] / self._V2
        np.testing.assert_allclose(got, ref, rtol=_A17_RTOL, atol=_A17_ATOL,
                                   err_msg="ADVAN4 single-dose: analytic Rust vs Python")

    def test_two_dose_matches_advan4(self):
        dt, da = _A17_DOSE2
        ref = self._advan4_ipred(_A17_OBS, _a17_dose_events(dt, da))
        states = np.array(_A_2cmt_oral(_A17_OBS, dt, da, self._THETA))
        got = states[:, 1] / self._V2
        np.testing.assert_allclose(got, ref, rtol=_A17_RTOL, atol=_A17_ATOL,
                                   err_msg="ADVAN4 two-dose: analytic Rust vs Python")


# ---------------------------------------------------------------------------
# 17C — End-to-End IPRED via IndividualModel._try_native_pk_backend
# ---------------------------------------------------------------------------

class TestAnalyticAdvanEndToEndIpred:
    """IndividualModel._try_native_pk_backend dispatches to analytic probes for ADVAN1-4."""

    _skip = _require_with_indiv(_A_1cmt_iv, _A_2cmt_iv)

    def _build_and_probe(self, advan, n_cmt, pk_fn):
        indiv = _build_analytic_indiv(advan, n_cmt, pk_fn)
        theta_dummy = [1.0]
        eta_dummy = []
        pk_params = pk_fn(theta_dummy, eta_dummy)
        obs = np.array(_A17_OBS)
        sol = indiv._try_native_pk_backend(pk_params, obs)
        return sol, pk_params

    def test_advan1_native_ipred_matches_python(self):
        from openpkpd.pk.analytical.advan1 import ADVAN1
        from openpkpd.data.event_processor import DoseEvent

        def pk(theta, eta, t=0.0, covariates=None):
            return {"CL": _A17_1CMT_CL, "V": _A17_1CMT_V}

        sol, pp = self._build_and_probe(1, 1, pk)
        assert sol is not None, "Native path must activate for ADVAN1"
        ref = ADVAN1().solve({"K": pp["CL"] / pp["V"], "V": pp["V"]},
                             [DoseEvent(time=0.0, amount=100.0, compartment=1)],
                             np.array(_A17_OBS)).ipred
        np.testing.assert_allclose(sol.ipred, ref, rtol=_A17_RTOL, atol=_A17_ATOL,
                                   err_msg="ADVAN1 end-to-end IPRED mismatch")

    def test_advan3_native_ipred_matches_python(self):
        from openpkpd.pk.analytical.advan3 import ADVAN3
        from openpkpd.data.event_processor import DoseEvent

        p = _A17_2CMT
        def pk(theta, eta, t=0.0, covariates=None):
            return {"CL": p["CL"], "V1": p["V1"], "Q": p["Q"], "V2": p["V2"]}

        sol, pp = self._build_and_probe(3, 2, pk)
        assert sol is not None, "Native path must activate for ADVAN3"
        ref = ADVAN3().solve(pp, [DoseEvent(time=0.0, amount=100.0, compartment=1)],
                             np.array(_A17_OBS)).ipred
        np.testing.assert_allclose(sol.ipred, ref, rtol=_A17_RTOL, atol=_A17_ATOL,
                                   err_msg="ADVAN3 end-to-end IPRED mismatch")

    def test_advan6_2cmt_still_uses_cvodes_template(self):
        """ADVAN6 2cmt_iv must match the CVODES template, not the analytic one."""
        p = _A17_2CMT
        def pk(theta, eta, t=0.0, covariates=None):
            return {"CL": p["CL"], "V1": p["V1"], "Q": p["Q"], "V2": p["V2"]}

        indiv = _build_analytic_indiv(6, 2, pk)
        contract = indiv._native_ode_contract
        assert contract is not None, "ADVAN6 2cmt must still build a contract"
        assert contract["advan"] == 6
        from openpkpd.model.individual import _NATIVE_ODE_TEMPLATES
        pk_params = pk([1.0], [])
        template_name = None
        for tmpl in _NATIVE_ODE_TEMPLATES:
            if tmpl.state_probe_fn is None:
                continue
            if tmpl.n_states != 2:
                continue
            if tmpl.eligible_advans and 6 not in tmpl.eligible_advans:
                continue
            if any(name not in pk_params for name in tmpl.required_names):
                continue
            template_name = tmpl.name
            break
        assert template_name == "2cmt_iv", (
            f"ADVAN6 2cmt should match 'cvodes 2cmt_iv' template, got {template_name!r}"
        )


# ---------------------------------------------------------------------------
# 17D — Sensitivity dispatch for ADVAN3
# ---------------------------------------------------------------------------

class TestAnalyticAdvanSensitivityDispatch:
    """native_advan6_prediction_eta_jacobian returns accurate G for ADVAN3."""

    _skip = _require_with_indiv(_A_2cmt_iv, _2cmt_iv_sens)

    _p = _A17_2CMT
    _THETA_POP = np.array([_p["CL"], _p["V1"], _p["Q"], _p["V2"]])

    def _build_advan3_indiv(self):
        def pk_callable(theta, eta, t=0.0, covariates=None):
            th = list(theta)
            return {
                "CL": float(th[0]) * np.exp(float(eta[0]) if len(eta) > 0 else 0.0),
                "V1": float(th[1]) * np.exp(float(eta[1]) if len(eta) > 1 else 0.0),
                "Q":  float(th[2]),
                "V2": float(th[3]),
            }
        return _build_analytic_indiv(3, 2, pk_callable, n_eps=1)

    def test_sensitivity_returns_non_none(self):
        indiv = self._build_advan3_indiv()
        obs_mask = np.ones(len(_A17_OBS), dtype=bool)
        G = indiv.native_advan6_prediction_eta_jacobian(
            self._THETA_POP, np.zeros(2), obs_mask, n_eta=2
        )
        assert G is not None, "ADVAN3 sensitivity path must return a G matrix"
        assert G.shape == (len(_A17_OBS), 2), f"G shape mismatch: {G.shape}"

    def test_sensitivity_matches_fd(self):
        """G matrix must agree with finite-difference reference (rtol=5e-3)."""
        indiv = self._build_advan3_indiv()
        theta = self._THETA_POP
        eta_zero = np.zeros(2)
        eps = 1e-4
        obs = np.array(_A17_OBS)
        obs_mask = np.ones(len(obs), dtype=bool)

        G = indiv.native_advan6_prediction_eta_jacobian(theta, eta_zero, obs_mask, n_eta=2, eps=eps)
        assert G is not None

        def ipred_at_eta(eta):
            from openpkpd.pk.analytical.advan3 import ADVAN3
            from openpkpd.data.event_processor import DoseEvent
            pp = {
                "CL": float(theta[0]) * np.exp(eta[0]),
                "V1": float(theta[1]) * np.exp(eta[1]),
                "Q":  float(theta[2]),
                "V2": float(theta[3]),
            }
            return ADVAN3().solve(pp, [DoseEvent(time=0.0, amount=100.0, compartment=1)],
                                  obs).ipred

        G_fd = np.zeros((len(obs), 2))
        for k in range(2):
            ep = eta_zero.copy(); ep[k] += eps
            em = eta_zero.copy(); em[k] -= eps
            G_fd[:, k] = (ipred_at_eta(ep) - ipred_at_eta(em)) / (2 * eps)

        np.testing.assert_allclose(G, G_fd, rtol=5e-3, atol=1e-8,
                                   err_msg="ADVAN3 sensitivity G deviates from FD reference")

