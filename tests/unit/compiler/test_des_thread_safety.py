"""Tests for CM2: Thread-safe DADT in generated ODE callable.

Verifies that CompiledDESCallable allocates a fresh dadt list on each call
so that concurrent calls from different threads do not share mutable state.
"""

from __future__ import annotations

import concurrent.futures
import threading

import numpy as np
import pytest

from openpkpd.parser.code_compiler import NMTRANCompiler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def one_cmt_des_fn():
    """Compile a simple 1-compartment DES block: DADT(1) = -K10 * A(1)."""
    compiler = NMTRANCompiler()
    des_code = "DADT(1) = -K10 * A(1)"
    des_fn = compiler.compile_des(des_code, n_compartments=1)
    return des_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDESThreadSafety:
    """CompiledDESCallable is thread-safe — no shared mutable state in dadt."""

    def test_concurrent_calls_return_correct_results(self, one_cmt_des_fn):
        """10 concurrent threads, each with different a values, all return correct dadt."""
        n_threads = 10
        k10 = 0.1
        errors: list[str] = []
        lock = threading.Lock()

        def call_and_check(a_value: float) -> float:
            result = one_cmt_des_fn(
                t=0.0,
                a=[a_value],
                pk_params={"K10": k10},
                theta=[],
                eta=[],
            )
            return result[0]  # dadt[0] = -K10 * a_value

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            a_values = [float(i + 1) * 10.0 for i in range(n_threads)]
            futures = {executor.submit(call_and_check, av): av for av in a_values}
            for fut, av in futures.items():
                got = fut.result()
                expected = -k10 * av
                if abs(got - expected) > 1e-12:
                    with lock:
                        errors.append(f"a={av}: expected {expected}, got {got}")

        assert not errors, "Thread-safety failures:\n" + "\n".join(errors)

    def test_returned_list_is_independent_across_calls(self, one_cmt_des_fn):
        """Each call returns a distinct list object (no aliasing)."""
        result1 = one_cmt_des_fn(
            t=0.0, a=[100.0], pk_params={"K10": 0.1}, theta=[], eta=[]
        )
        result2 = one_cmt_des_fn(
            t=0.0, a=[200.0], pk_params={"K10": 0.1}, theta=[], eta=[]
        )
        assert result1 is not result2, "Calls must return different list objects"
        # Mutating result1 must not affect result2
        if isinstance(result1, list):
            result1[0] = 9999.0
            assert result2[0] != 9999.0, (
                "Mutating result1 affected result2 — shared mutable state detected"
            )

    def test_numerical_correctness_concurrent_100_calls(self):
        """100 concurrent calls with A(1)=100, K10=0.1 all return dadt[0]=-10.0."""
        compiler = NMTRANCompiler()
        des_fn = compiler.compile_des("DADT(1) = -K10 * A(1)", n_compartments=1)

        n = 100
        results: list[float] = [float("nan")] * n

        def worker(i: int) -> None:
            r = des_fn(
                t=0.0,
                a=[100.0],
                pk_params={"K10": 0.1},
                theta=[],
                eta=[],
            )
            results[i] = r[0]

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, val in enumerate(results):
            assert val == pytest.approx(-10.0, abs=1e-12), (
                f"Thread {i}: expected -10.0, got {val}"
            )

    def test_two_compartment_concurrent(self):
        """2-compartment ODE called concurrently returns correct dadt vectors."""
        compiler = NMTRANCompiler()
        des_code = "DADT(1) = -K12 * A(1)\nDADT(2) = K12 * A(1) - K20 * A(2)"
        des_fn = compiler.compile_des(des_code, n_compartments=2)

        pk = {"K12": 0.3, "K20": 0.1}
        errors: list[str] = []
        lock = threading.Lock()

        def worker(a1: float, a2: float) -> list:
            return des_fn(t=0.0, a=[a1, a2], pk_params=pk, theta=[], eta=[])

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            inputs = [(float(i), float(i * 0.5)) for i in range(1, 21)]
            futures = {executor.submit(worker, a1, a2): (a1, a2) for a1, a2 in inputs}
            for fut, (a1, a2) in futures.items():
                got = fut.result()
                exp0 = -pk["K12"] * a1
                exp1 = pk["K12"] * a1 - pk["K20"] * a2
                if abs(got[0] - exp0) > 1e-12 or abs(got[1] - exp1) > 1e-12:
                    with lock:
                        errors.append(
                            f"a=({a1},{a2}): expected ({exp0},{exp1}), got ({got[0]},{got[1]})"
                        )

        assert not errors, "Thread-safety failures:\n" + "\n".join(errors)
