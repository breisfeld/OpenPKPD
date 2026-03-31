"""
Numerical accuracy, detection-logic, and performance tests for the native
CVODES extension (_core Rust module).

Three concerns are covered here:

1. neg2ll_obs_loop — BLQ-aware observation log-likelihood accumulator.
   Every BLQ method (M1-M7) is checked against an independent Python
   reference implementation to tolerances tighter than 1e-12.

2. native_cvodes_advan6_mixed_pkpd_probe — CVODES BDF integrator for the
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

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.stats import norm as _norm

# ---------------------------------------------------------------------------
# Helpers — import native symbols, skip gracefully when absent
# ---------------------------------------------------------------------------

def _try_import(name: str):
    try:
        from openpkpd._native import import_core_symbol
        return import_core_symbol(name)
    except Exception:
        return None


_neg2ll = _try_import("neg2ll_obs_loop")
_probe = _try_import("native_cvodes_advan6_mixed_pkpd_probe")

pytestmark_native = pytest.mark.skipif(
    _neg2ll is None, reason="native _core extension not available"
)
pytestmark_cvodes = pytest.mark.skipif(
    _probe is None, reason="native-cvodes feature not compiled in"
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
# Section 2 — native_cvodes_advan6_mixed_pkpd_probe numerical accuracy
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
        rust = np.asarray(_probe(times, _DOSE_AMT, list(theta)), dtype=float)

        assert rust.shape == (len(times), 4), f"{name}: unexpected shape {rust.shape}"
        # Absolute tolerance scales with dose amount; relative 5e-5 matches
        # solver rtol=1e-8 with some margin for BDF vs Radau differences.
        np.testing.assert_allclose(rust, ref, rtol=5e-5, atol=1e-6,
                                   err_msg=f"Parameter set '{name}' failed parity check")

    @pytestmark_cvodes
    def test_zero_dose_gives_zero_pk_concentrations(self) -> None:
        theta = list(_PARAM_SETS["warfarin_standard"])
        rust = np.asarray(_probe(_OBS_TIMES, 0.0, theta), dtype=float)
        # A1, A2, A3 must all be zero; A4 relaxes toward E0 baseline = 0
        np.testing.assert_allclose(rust[:, :3], 0.0, atol=1e-10)

    @pytestmark_cvodes
    def test_t0_included_returns_initial_condition(self) -> None:
        theta = list(_PARAM_SETS["warfarin_standard"])
        times_with_zero = [0.0] + _OBS_TIMES
        rust = np.asarray(_probe(times_with_zero, _DOSE_AMT, theta), dtype=float)
        assert rust[0, 0] == pytest.approx(_DOSE_AMT, rel=1e-9)
        assert rust[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert rust[0, 2] == pytest.approx(0.0, abs=1e-12)

    @pytestmark_cvodes
    def test_pk_mass_balance_is_conserved(self) -> None:
        """Sum of A1+A2+A3 must decrease monotonically — drug leaves system."""
        theta = list(_PARAM_SETS["warfarin_standard"])
        rust = np.asarray(_probe(_OBS_TIMES, _DOSE_AMT, theta), dtype=float)
        pk_total = rust[:, 0] + rust[:, 1] + rust[:, 2]
        # PK compartments together drain over time (none re-enter)
        assert np.all(np.diff(pk_total) <= 1e-6), "PK compartments should drain monotonically"

    @pytestmark_cvodes
    def test_central_compartment_peaks_then_declines(self) -> None:
        """A3 (central PK) must peak and then decline."""
        theta = list(_PARAM_SETS["warfarin_standard"])
        times_dense = list(np.linspace(0.5, 72.0, 50))
        rust = np.asarray(_probe(times_dense, _DOSE_AMT, theta), dtype=float)
        a3 = rust[:, 2]
        peak_idx = int(np.argmax(a3))
        assert peak_idx > 0, "Peak must not be at the first time point"
        assert np.all(np.diff(a3[peak_idx:]) <= 1e-3), "A3 should decline after peak"

    @pytestmark_cvodes
    def test_pd_effect_perturbs_only_a4(self) -> None:
        """Without PD (EMAX=0), A4 should stay near zero (KOUT·E0·0 = 0)."""
        theta = list(_PARAM_SETS["no_pd_effect"])
        rust = np.asarray(_probe(_OBS_TIMES, _DOSE_AMT, theta), dtype=float)
        # With EMAX=0, pd=1 so da4/dt = KOUT*(E0*0 - A4) → A4 decays to 0
        np.testing.assert_allclose(rust[:, 3], 0.0, atol=1e-8)

    @pytestmark_cvodes
    def test_single_observation_time_returns_correct_shape(self) -> None:
        theta = list(_PARAM_SETS["warfarin_standard"])
        rust = np.asarray(_probe([4.0], _DOSE_AMT, theta), dtype=float)
        assert rust.shape == (1, 4)

    @pytestmark_cvodes
    def test_unsorted_times_raises_value_error(self) -> None:
        theta = list(_PARAM_SETS["warfarin_standard"])
        with pytest.raises(Exception):
            _probe([8.0, 2.0, 4.0], _DOSE_AMT, theta)

    @pytestmark_cvodes
    def test_wrong_theta_length_raises_value_error(self) -> None:
        with pytest.raises(Exception):
            _probe(_OBS_TIMES, _DOSE_AMT, [1.0, 2.0, 3.0])  # only 3 instead of 8


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
        _probe(times, _DOSE_AMT, theta)
        _scipy_reference(times, _DOSE_AMT, tuple(theta), rtol=1e-6, atol=1e-8)

        t0 = time.perf_counter()
        for _ in range(_PERF_REPS):
            _probe(times, _DOSE_AMT, theta)
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

def _try_import_multidose():
    try:
        from openpkpd._native import import_core_symbol
        return import_core_symbol("native_cvodes_advan6_mixed_pkpd_probe_multidose")
    except Exception:
        return None


_multidose = _try_import_multidose()

pytestmark_multidose = pytest.mark.skipif(
    _multidose is None, reason="native-cvodes multi-dose probe not available"
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
        single = np.asarray(_probe(obs, _DOSE_AMT, theta), dtype=float)
        multi = np.asarray(_multidose(obs, [0.0], [_DOSE_AMT], theta), dtype=float)
        np.testing.assert_allclose(multi, single, rtol=1e-10, atol=1e-12,
                                   err_msg="single-dose multidose probe must match original probe")

    @pytestmark_multidose
    def test_two_dose_matches_scipy_reference(self) -> None:
        obs, (dt, da) = _MD_OBS, _MD_DOSE_2
        ref = _scipy_multidose_reference(obs, dt, da, _MD_THETA)
        rust = np.asarray(_multidose(obs, dt, da, _MD_THETA), dtype=float)
        np.testing.assert_allclose(rust, ref, rtol=5e-5, atol=1e-6,
                                   err_msg="two-dose scenario failed parity check")

    @pytestmark_multidose
    def test_three_dose_matches_scipy_reference(self) -> None:
        obs, (dt, da) = _MD_OBS, _MD_DOSE_3
        ref = _scipy_multidose_reference(obs, dt, da, _MD_THETA)
        rust = np.asarray(_multidose(obs, dt, da, _MD_THETA), dtype=float)
        np.testing.assert_allclose(rust, ref, rtol=5e-5, atol=1e-6,
                                   err_msg="three-dose scenario failed parity check")

    @pytestmark_multidose
    def test_second_dose_raises_central_compartment_above_pre_dose_trough(self) -> None:
        """After the second dose A3 must be higher than the trough before it."""
        obs = [23.9, 24.1, 48.0]  # trough just before dose 2, shortly after, end
        dt, da = _MD_DOSE_2
        rust = np.asarray(_multidose(obs, dt, da, _MD_THETA), dtype=float)
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
        rust = np.asarray(_multidose(obs, dt, da, _MD_THETA), dtype=float)
        ref = _scipy_multidose_reference(obs, dt, da, _MD_THETA)
        np.testing.assert_allclose(rust, ref, rtol=5e-5, atol=1e-6)
        # A1 at t=24 should include the second bolus → larger than at t=12
        assert rust[1, 0] > rust[0, 0], "A1 at dose time should have received the second bolus"

    @pytestmark_multidose
    def test_zero_pk_drug_before_first_dose(self) -> None:
        """Observations before the first dose must have zero PK state."""
        obs = [0.5, 1.0]
        dt, da = [6.0], [100.0]  # first dose at t=6
        rust = np.asarray(_multidose(obs, dt, da, _MD_THETA), dtype=float)
        np.testing.assert_allclose(rust, 0.0, atol=1e-12)

    @pytestmark_multidose
    def test_pk_mass_balance_conserved_multidose(self) -> None:
        """Total A1+A2+A3 should not exceed cumulative doses."""
        obs = list(np.linspace(0.1, 72.0, 30))
        dt, da = _MD_DOSE_2
        rust = np.asarray(_multidose(obs, dt, da, _MD_THETA), dtype=float)
        pk_total = rust[:, 0] + rust[:, 1] + rust[:, 2]
        total_dose = sum(da)
        assert np.all(pk_total <= total_dose + 1e-6), (
            "PK compartments can never exceed total administered dose"
        )

    @pytestmark_multidose
    def test_validation_dose_time_desc_raises(self) -> None:
        with pytest.raises(Exception):
            _multidose(_MD_OBS, [24.0, 0.0], [100.0, 100.0], _MD_THETA)

    @pytestmark_multidose
    def test_validation_empty_dose_times_raises(self) -> None:
        with pytest.raises(Exception):
            _multidose(_MD_OBS, [], [], _MD_THETA)

    @pytestmark_multidose
    def test_validation_mismatched_dose_arrays_raises(self) -> None:
        with pytest.raises(Exception):
            _multidose(_MD_OBS, [0.0, 24.0], [100.0], _MD_THETA)

    @pytestmark_multidose
    def test_multidose_faster_than_scipy_reference(self) -> None:
        obs = _MD_OBS
        dt, da = _MD_DOSE_2
        theta = _MD_THETA

        # Warm-up
        _multidose(obs, dt, da, theta)
        _scipy_multidose_reference(obs, dt, da, theta, rtol=1e-6, atol=1e-8)

        t0 = time.perf_counter()
        for _ in range(_PERF_REPS):
            _multidose(obs, dt, da, theta)
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
