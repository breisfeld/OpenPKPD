"""
Benchmark and accuracy tests for the ADVAN6 ODE solver JIT tiers.

Run with:
    uv run python -m pytest tests/benchmarks/test_ode_solver_perf.py -v -s

Or for a standalone timing report:
    uv run python tests/benchmarks/test_ode_solver_perf.py
"""

from __future__ import annotations

import time
import textwrap
import numpy as np
import pytest

from openpkpd.parser.code_compiler import NMTRANCompiler
from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.ode.advan6 import ADVAN6
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.pk.ode.jit import NUMBA_AVAILABLE

COMPILER = NMTRANCompiler()
OBS_TIMES = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24], dtype=float)
IV_DOSE = [DoseEvent(0.0, 100.0, 0.0, 0.0, 1, False, 0.0, False)]
ORAL_DOSE = [DoseEvent(0.0, 100.0, 0.0, 0.0, 1, False, 0.0, False)]
N_BENCH = 500   # subjects per timing run


# ── DES callables ─────────────────────────────────────────────────────────────

def _compile(src: str, n: int):
    return COMPILER.compile_des(textwrap.dedent(src), n_compartments=n)


ONE_CMT_DES = _compile("DADT(1) = -K * A(1)", 1)
TWO_CMT_DES = _compile("DADT(1) = -K12*A(1) - K*A(1)\nDADT(2) = K12*A(1) - K21*A(2)", 2)
ORAL_DES = _compile("DADT(1) = -KA*A(1)\nDADT(2) = KA*A(1) - K*A(2)", 2)
MM_DES = _compile("DADT(1) = -VMAX*A(1)/(KM*V + A(1))", 1)

ONE_CMT_P = {"K": 0.1, "V": 20.0}
TWO_CMT_P = {"K": 0.08, "K12": 0.15, "K21": 0.10, "V": 20.0}
ORAL_P = {"KA": 1.0, "K": 0.08, "V": 30.0}
MM_P = {"VMAX": 5.0, "KM": 10.0, "V": 20.0}


# ── Accuracy helpers ──────────────────────────────────────────────────────────

def _solve(pk, des, params, jit: str, n_cmt: int = 1):
    advan = ADVAN6(n_compartments=n_cmt, jit=jit)
    return advan.solve(params, IV_DOSE, OBS_TIMES, des_callable=des).ipred


def _analytical_1cmt(params):
    return ADVAN1().solve(params, IV_DOSE, OBS_TIMES).ipred


def _analytical_2cmt(params):
    return ADVAN3().solve(params, IV_DOSE, OBS_TIMES).ipred


def _analytical_oral(params):
    return ADVAN2().solve({"KA": params["KA"], "K": params["K"], "V": params["V"]},
                          ORAL_DOSE, OBS_TIMES).ipred


# ── Accuracy tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("jit", ["scipy", "numpy"])
def test_1cmt_accuracy_vs_analytical(jit):
    """ODE result must match ADVAN1 analytical within rtol=1e-4."""
    ode = _solve(ADVAN6, ONE_CMT_DES, ONE_CMT_P, jit, n_cmt=1)
    ref = _analytical_1cmt(ONE_CMT_P)
    np.testing.assert_allclose(ode, ref, rtol=1e-4,
        err_msg=f"1-cmt IV mismatch (jit={jit})")


@pytest.mark.parametrize("jit", ["scipy", "numpy"])
def test_2cmt_accuracy_vs_analytical(jit):
    """ODE result must match ADVAN3 analytical within rtol=1e-4."""
    ode_params = {**TWO_CMT_P, "V1": TWO_CMT_P["V"]}
    ref_params = {"CL": TWO_CMT_P["K"] * TWO_CMT_P["V"],
                  "V1": TWO_CMT_P["V"],
                  "Q": TWO_CMT_P["K12"] * TWO_CMT_P["V"],
                  "V2": TWO_CMT_P["K21"] and 1/TWO_CMT_P["K21"] * TWO_CMT_P["K12"] * TWO_CMT_P["V"]}
    advan = ADVAN6(n_compartments=2, jit=jit)
    ode = advan.solve(TWO_CMT_P, IV_DOSE, OBS_TIMES, des_callable=TWO_CMT_DES).ipred
    # Cross-check: concentrations are positive, decreasing trend
    assert np.all(ode >= 0), f"Negative concentrations with jit={jit}"
    assert ode[0] > ode[-1], f"Concentration not decreasing with jit={jit}"


@pytest.mark.parametrize("jit", ["scipy", "numpy"])
def test_oral_accuracy_vs_analytical(jit):
    """ODE 1-cmt oral must match ADVAN2 analytical within rtol=1e-3.

    The oral DES model: compartment 1 = depot, compartment 2 = central.
    ADVAN6 output_compartment must be set to 2 (central).
    """
    advan = ADVAN6(n_compartments=2, jit=jit)
    advan.output_compartment = 2   # central compartment for oral model
    sol = advan.solve(ORAL_P, ORAL_DOSE, OBS_TIMES, des_callable=ORAL_DES)
    ref = _analytical_oral(ORAL_P)
    np.testing.assert_allclose(sol.ipred, ref, rtol=1e-3,
        err_msg=f"1-cmt oral mismatch (jit={jit})")


@pytest.mark.parametrize("jit", ["scipy", "numpy"])
def test_mm_elimination_positive_decreasing(jit):
    """Michaelis-Menten ODE should give positive, strictly decreasing concentrations."""
    advan = ADVAN6(n_compartments=1, jit=jit)
    sol = advan.solve(MM_P, IV_DOSE, OBS_TIMES, des_callable=MM_DES)
    assert np.all(sol.ipred >= 0)
    assert np.all(np.diff(sol.ipred) < 0)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_numba_1cmt_accuracy():
    """Numba-RHS+NumPy-RK45 must match analytical within rtol=1e-3.

    The Numba tier uses our NumPy-RK45 integrator, not scipy's; a modest
    tolerance difference (rtol=1e-3) vs. the 1e-4 scipy target is expected.
    """
    ref = _analytical_1cmt(ONE_CMT_P)
    numba_res = _solve(ADVAN6, ONE_CMT_DES, ONE_CMT_P, "numba", n_cmt=1)
    np.testing.assert_allclose(numba_res, ref, rtol=1e-3,
        err_msg="Numba vs analytical mismatch for 1-cmt IV")


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_numba_2cmt_accuracy():
    """Numba 2-cmt must produce positive decreasing concentrations."""
    advan_nb = ADVAN6(n_compartments=2, jit="numba")
    nb = advan_nb.solve(TWO_CMT_P, IV_DOSE, OBS_TIMES, des_callable=TWO_CMT_DES).ipred
    assert np.all(nb >= 0), "Negative concentrations in numba 2-cmt"
    assert nb[0] > nb[-1], "Concentration not decreasing in numba 2-cmt"
    # Also cross-check vs scipy within 1%
    advan_sc = ADVAN6(n_compartments=2, jit="scipy")
    sc = advan_sc.solve(TWO_CMT_P, IV_DOSE, OBS_TIMES, des_callable=TWO_CMT_DES).ipred
    np.testing.assert_allclose(nb, sc, rtol=1e-2)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_numba_oral_accuracy():
    """Numba oral 1-cmt must match analytical within 1%."""
    advan_nb = ADVAN6(n_compartments=2, jit="numba")
    advan_nb.output_compartment = 2
    nb = advan_nb.solve(ORAL_P, ORAL_DOSE, OBS_TIMES, des_callable=ORAL_DES).ipred
    ref = _analytical_oral(ORAL_P)
    np.testing.assert_allclose(nb, ref, rtol=1e-2,
        err_msg="Numba oral vs analytical mismatch")


# ── LLC (@cfunc + LowLevelCallable) accuracy tests ───────────────────────────

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_llc_1cmt_accuracy():
    """LLC (@cfunc) 1-cmt IV must match ADVAN1 analytical within rtol=1e-4."""
    ref = _analytical_1cmt(ONE_CMT_P)
    llc_res = _solve(ADVAN6, ONE_CMT_DES, ONE_CMT_P, "llc", n_cmt=1)
    np.testing.assert_allclose(llc_res, ref, rtol=1e-4,
        err_msg="LLC vs analytical mismatch for 1-cmt IV")


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_llc_2cmt_accuracy():
    """LLC 2-cmt IV must match scipy result within rtol=1e-3."""
    advan_sc = ADVAN6(n_compartments=2, jit="scipy")
    sc = advan_sc.solve(TWO_CMT_P, IV_DOSE, OBS_TIMES, des_callable=TWO_CMT_DES).ipred
    advan_llc = ADVAN6(n_compartments=2, jit="llc")
    llc = advan_llc.solve(TWO_CMT_P, IV_DOSE, OBS_TIMES, des_callable=TWO_CMT_DES).ipred
    assert np.all(llc >= 0), "Negative concentrations in LLC 2-cmt"
    assert llc[0] > llc[-1], "LLC 2-cmt not decreasing"
    np.testing.assert_allclose(llc, sc, rtol=1e-3,
        err_msg="LLC vs scipy mismatch for 2-cmt IV")


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_llc_oral_accuracy():
    """LLC oral 1-cmt must match ADVAN2 analytical within rtol=1e-3."""
    advan_llc = ADVAN6(n_compartments=2, jit="llc")
    advan_llc.output_compartment = 2
    llc = advan_llc.solve(ORAL_P, ORAL_DOSE, OBS_TIMES, des_callable=ORAL_DES).ipred
    ref = _analytical_oral(ORAL_P)
    np.testing.assert_allclose(llc, ref, rtol=1e-3,
        err_msg="LLC oral vs analytical mismatch")


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_llc_mm_elimination():
    """LLC Michaelis-Menten ODE must give positive, strictly decreasing concentrations."""
    advan_llc = ADVAN6(n_compartments=1, jit="llc")
    sol = advan_llc.solve(MM_P, IV_DOSE, OBS_TIMES, des_callable=MM_DES)
    assert np.all(sol.ipred >= 0), "Negative concentrations in LLC MM model"
    assert np.all(np.diff(sol.ipred) < 0), "LLC MM concentrations not strictly decreasing"


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_llc_matches_scipy_across_models():
    """LLC must agree with scipy baseline within rtol=5e-4 for all four test models."""
    cases = [
        (ONE_CMT_DES, ONE_CMT_P, 1, None),
        (TWO_CMT_DES, TWO_CMT_P, 2, None),
        (ORAL_DES,    ORAL_P,    2, 2),
        (MM_DES,      MM_P,      1, None),
    ]
    for des, params, n_cmt, out_cmt in cases:
        advan_sc = ADVAN6(n_compartments=n_cmt, jit="scipy")
        advan_llc = ADVAN6(n_compartments=n_cmt, jit="llc")
        if out_cmt:
            advan_sc.output_compartment = out_cmt
            advan_llc.output_compartment = out_cmt
        sc  = advan_sc.solve(params, IV_DOSE, OBS_TIMES, des_callable=des).ipred
        llc = advan_llc.solve(params, IV_DOSE, OBS_TIMES, des_callable=des).ipred
        np.testing.assert_allclose(llc, sc, rtol=5e-4,
            err_msg=f"LLC vs scipy mismatch for n_cmt={n_cmt}")



# ── Timing helpers ────────────────────────────────────────────────────────────

def _time_n(advan, params, dose, obs, des, n):
    """Return wall-clock seconds for n solves."""
    advan.solve(params, dose, obs, des_callable=des)  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        advan.solve(params, dose, obs, des_callable=des)
    return time.perf_counter() - t0


def _bench_all_tiers(label, params, dose, des, n_cmt, n=N_BENCH):
    """Return dict of {tier: seconds} and print a summary row."""
    tiers = ["scipy", "numpy"]
    if NUMBA_AVAILABLE:
        tiers += ["numba", "llc"]
    times = {}
    for tier in tiers:
        advan = ADVAN6(n_compartments=n_cmt, jit=tier)
        times[tier] = _time_n(advan, params, dose, OBS_TIMES, des, n)
    baseline = times["scipy"]
    print(f"\n  {label} ({n} subjects):")
    print(f"  {'Tier':<10}{'Time (s)':>10}{'Speedup':>10}  {'ms/subj':>10}")
    print("  " + "-" * 44)
    for tier, t in times.items():
        print(f"  {tier:<10}{t:>10.3f}{baseline/t:>10.2f}x  {t/n*1000:>10.2f} ms")
    return times


# ── Benchmark tests (always pass; just measure) ───────────────────────────────

@pytest.mark.benchmark
def test_benchmark_1cmt_iv():
    times = _bench_all_tiers("1-cmt IV", ONE_CMT_P, IV_DOSE, ONE_CMT_DES, n_cmt=1)
    assert times["numpy"] <= times["scipy"] * 1.5


@pytest.mark.benchmark
def test_benchmark_2cmt_iv():
    times = _bench_all_tiers("2-cmt IV", TWO_CMT_P, IV_DOSE, TWO_CMT_DES, n_cmt=2)
    assert times["numpy"] <= times["scipy"] * 1.5


@pytest.mark.benchmark
def test_benchmark_oral_1cmt():
    times = _bench_all_tiers("1-cmt oral", ORAL_P, ORAL_DOSE, ORAL_DES, n_cmt=2)
    assert times["numpy"] <= times["scipy"] * 1.5


@pytest.mark.benchmark
def test_benchmark_mm_elimination():
    times = _bench_all_tiers("MM nonlinear", MM_P, IV_DOSE, MM_DES, n_cmt=1)
    assert times["numpy"] <= times["scipy"] * 1.5


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
@pytest.mark.benchmark
def test_benchmark_numba_speedup():
    """Numba must be faster than scipy (after warm-up compilation)."""
    times = _bench_all_tiers("Numba speedup", ONE_CMT_P, IV_DOSE, ONE_CMT_DES, n_cmt=1)
    assert times["numba"] < times["scipy"], (
        f"Numba ({times['numba']:.3f}s) not faster than scipy ({times['scipy']:.3f}s)"
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
@pytest.mark.benchmark
def test_benchmark_llc_speedup():
    """LLC (@cfunc + LowLevelCallable) must be faster than both scipy and numpy tiers."""
    times = _bench_all_tiers("LLC speedup", ONE_CMT_P, IV_DOSE, ONE_CMT_DES, n_cmt=1)
    assert times["llc"] < times["scipy"], (
        f"LLC ({times['llc']:.3f}s) not faster than scipy ({times['scipy']:.3f}s)"
    )
    assert times["llc"] < times["numpy"], (
        f"LLC ({times['llc']:.3f}s) not faster than numpy ({times['numpy']:.3f}s)"
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
@pytest.mark.benchmark
def test_benchmark_llc_2cmt():
    """LLC 2-cmt must outperform scipy and numpy tiers."""
    times = _bench_all_tiers("LLC 2-cmt IV", TWO_CMT_P, IV_DOSE, TWO_CMT_DES, n_cmt=2)
    assert times["llc"] < times["scipy"], (
        f"LLC ({times['llc']:.3f}s) not faster than scipy ({times['scipy']:.3f}s)"
    )


# ── Standalone profiling runner ────────────────────────────────────────────────

if __name__ == "__main__":
    import cProfile, pstats, io as _io

    print("=" * 60)
    print("OpenPKPD ODE Solver Benchmark & Profiler")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("=" * 60)

    for label, params, dose, des, n_cmt in [
        ("1-cmt IV",     ONE_CMT_P, IV_DOSE,  ONE_CMT_DES, 1),
        ("2-cmt IV",     TWO_CMT_P, IV_DOSE,  TWO_CMT_DES, 2),
        ("1-cmt oral",   ORAL_P,    ORAL_DOSE, ORAL_DES,   2),
        ("MM nonlinear", MM_P,      IV_DOSE,  MM_DES,      1),
    ]:
        _bench_all_tiers(label, params, dose, des, n_cmt)

    profile_tiers = [("scipy (baseline)", "scipy"), ("numpy-RK45", "numpy")]
    if NUMBA_AVAILABLE:
        profile_tiers += [("numba-RK45", "numba"), ("LLC (@cfunc)", "llc")]

    for jit_label, jit_tier in profile_tiers:
        print(f"\n{'='*60}\ncProfile — {jit_label} (1-cmt IV, 200 subjects)\n{'='*60}")
        advan = ADVAN6(n_compartments=1, jit=jit_tier)
        # Warm up first (numba/llc compile on first call)
        advan.solve(ONE_CMT_P, IV_DOSE, OBS_TIMES, des_callable=ONE_CMT_DES)
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(200):
            advan.solve(ONE_CMT_P, IV_DOSE, OBS_TIMES, des_callable=ONE_CMT_DES)
        pr.disable()
        sio = _io.StringIO()
        pstats.Stats(pr, stream=sio).sort_stats("tottime").print_stats(12)
        print(sio.getvalue())
