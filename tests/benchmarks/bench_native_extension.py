"""
Native CVODES extension benchmark — probe speed and sensitivity speed.

Compares each native template against the scipy Radau reference and against
central finite-differences for the sensitivity (Jacobian) path.

Run with:
    OPENPKPD_NATIVE_DEV=1 uv run python tests/benchmarks/bench_native_extension.py

or from pytest (tagged @benchmark, not collected by default):
    OPENPKPD_NATIVE_DEV=1 uv run pytest tests/benchmarks/bench_native_extension.py -v -s -m benchmark
"""

from __future__ import annotations

import time
import numpy as np
from scipy.integrate import solve_ivp

try:
    from openpkpd.model.individual import _try_import
    _HAS_INDIVIDUAL = True
except ImportError:
    _try_import = lambda _: None  # noqa: E731
    _HAS_INDIVIDUAL = False

# ---------------------------------------------------------------------------
# Load native probes
# ---------------------------------------------------------------------------

_P = _try_import
_1iv      = _P("native_cvodes_1cmt_iv_probe_multidose")
_1iv_s    = _P("native_cvodes_1cmt_iv_sensitivity_probe_multidose")
_1oral    = _P("native_cvodes_1cmt_oral_probe_multidose")
_1oral_s  = _P("native_cvodes_1cmt_oral_sensitivity_probe_multidose")
_2iv      = _P("native_cvodes_2cmt_iv_probe_multidose")
_2iv_s    = _P("native_cvodes_2cmt_iv_sensitivity_probe_multidose")
_2oral    = _P("native_cvodes_2cmt_oral_probe_multidose")
_2oral_s  = _P("native_cvodes_2cmt_oral_sensitivity_probe_multidose")
_3iv      = _P("native_cvodes_3cmt_iv_probe_multidose")
_3iv_s    = _P("native_cvodes_3cmt_iv_sensitivity_probe_multidose")
_warf     = _P("native_cvodes_advan6_mixed_pkpd_probe_multidose")
_warf_s   = _P("native_cvodes_advan6_mixed_pkpd_sensitivity_probe_multidose")

OBS  = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0]
DT1  = [0.0]; DA1 = [100.0]
DT2  = [0.0, 25.0]; DA2 = [100.0, 80.0]
N    = 500   # repetitions for timing


# ---------------------------------------------------------------------------
# Scipy RHS reference functions
# ---------------------------------------------------------------------------

def _rhs_1iv(cl, v):
    def f(t, y): return [-(cl/v)*y[0]]
    return f

def _rhs_1oral(ka, cl, v):
    def f(t, y): return [-ka*y[0], ka*y[0] - (cl/v)*y[1]]
    return f

def _rhs_2iv(cl, v1, q, v2):
    k10=cl/v1; k12=q/v1; k21=q/v2
    def f(t, y): return [-(k10+k12)*y[0]+k21*y[1], k12*y[0]-k21*y[1]]
    return f

def _rhs_2oral(ka, cl, v2, q, v3):
    k10=cl/v2; k12=q/v2; k21=q/v3
    def f(t, y): return [-ka*y[0], ka*y[0]-(k10+k12)*y[1]+k21*y[2], k12*y[1]-k21*y[2]]
    return f

def _rhs_3iv(cl, v1, q2, v2, q3, v3):
    k10=cl/v1; k12=q2/v1; k21=q2/v2; k13=q3/v1; k31=q3/v3
    def f(t, y): return [-(k10+k12+k13)*y[0]+k21*y[1]+k31*y[2], k12*y[0]-k21*y[1], k13*y[0]-k31*y[2]]
    return f

def _rhs_warf(ktr, ka, cl, v, emax, ec50, kout, e0):
    def f(t, y):
        conc=y[2]/v; pd=1-emax*conc/(ec50+conc)
        return [-ktr*y[0], ktr*y[0]-ka*y[1], ka*y[1]-(cl/v)*y[2], kout*e0*(pd-1)-kout*y[3]]
    return f


def _scipy_single(rhs, ic, obs=OBS):
    sol = solve_ivp(rhs, [0.0, max(obs)], ic, method="Radau",
                    t_eval=obs, rtol=1e-8, atol=1e-10)
    return sol.y.T


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _timeit(fn, n=N):
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1e3   # ms per call


def _fd_sens(probe, theta, n_params, dt=DT1, da=DA1, eps=1e-5):
    """Central FD over the state probe — n_params×2 extra solves."""
    for j in range(n_params):
        tp = list(theta); tp[j] += eps
        tm = list(theta); tm[j] -= eps
        probe(OBS, dt, da, tp)
        probe(OBS, dt, da, tm)


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

CASES = []   # (label, native_probe, scipy_rhs, scipy_ic, theta, sens_probe, n_params)

if _1iv:
    th = [0.044, 31.8]
    CASES.append(("1cmt_iv",   _1iv, _rhs_1iv(*th),   [0.0],         th, _1iv_s,   2))
if _1oral:
    th = [1.49, 0.044, 31.8]
    CASES.append(("1cmt_oral", _1oral, _rhs_1oral(*th), [0.0, 0.0],   th, _1oral_s, 3))
if _2iv:
    th = [3.877, 9.760, 8.774, 30.818]   # NONMEM 402 pop means
    CASES.append(("2cmt_iv",   _2iv, _rhs_2iv(*th),   [0.0, 0.0],    th, _2iv_s,   4))
if _2oral:
    th = [0.4, 0.13, 8.0, 0.6, 20.0]
    CASES.append(("2cmt_oral", _2oral, _rhs_2oral(*th), [0.0,0.0,0.0], th, _2oral_s, 5))
if _3iv:
    th = [0.13, 8.0, 0.6, 20.0, 0.3, 50.0]
    CASES.append(("3cmt_iv",   _3iv, _rhs_3iv(*th),   [0.0,0.0,0.0], th, _3iv_s,   6))
if _warf:
    th = [0.3, 0.3, 0.1, 8.0, 0.5, 0.3, 0.05, 100.0]
    CASES.append(("warfarin_pkpd", _warf, _rhs_warf(*th), [DA1[0],0,0,0], th, _warf_s, 8))


def run_benchmarks(n=N):
    col_sens = any(c[5] is not None for c in CASES)
    hdr = f"{'Model':<16} {'Native probe':>14} {'Scipy probe':>13} {'Probe ×':>9}"
    if col_sens:
        hdr += f"  {'Native sens':>13} {'FD sens':>11} {'Sens ×':>8}"
    sep = "=" * len(hdr)
    print(); print(sep)
    print("  OpenPKPD native CVODES extension — speed vs scipy Radau")
    print(f"  {n} repetitions, {len(OBS)} obs, single IV bolus")
    print(sep); print(hdr); print("-" * len(hdr))

    for label, probe, rhs, ic, theta, sens_probe, n_params in CASES:
        ic_boosted = list(ic); ic_boosted[0] = DA1[0]

        t_native = _timeit(lambda p=probe, th=theta: p(OBS, DT1, DA1, th), n)
        t_scipy  = _timeit(lambda r=rhs, i=ic_boosted: _scipy_single(r, i), n)
        sp = t_scipy / t_native if t_native > 0 else float("nan")

        row = (f"  {label:<14} {t_native:>12.3f} ms {t_scipy:>11.3f} ms {sp:>7.1f}×")

        if col_sens:
            if sens_probe is not None:
                t_ns = _timeit(lambda sp=sens_probe, th=theta: sp(OBS, DT1, DA1, th), n)
                t_fd = _timeit(lambda p=probe, th=theta, np_=n_params:
                               _fd_sens(p, th, np_), n)
                ss = t_fd / t_ns if t_ns > 0 else float("nan")
                row += f"  {t_ns:>11.3f} ms {t_fd:>9.3f} ms {ss:>6.1f}×"
            else:
                row += "  (sens probe unavailable)"
        print(row)

    print("-" * len(hdr))
    print(f"  Sens FD cost = n_params×2 extra state probe calls (central differences).")
    print()


if __name__ == "__main__":
    if not CASES:
        print("No native probes found. Build with: maturin develop --features native-cvodes")
    else:
        run_benchmarks()
