"""
CM1: Tests for $DES block compiler — A(0) and DADT(0) zero-index guard.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from openpkpd.parser.code_compiler import CompiledDESCallable, NMTRANCompiler, _translate_line, _INTRINSICS
from openpkpd.utils.errors import CompilerError


# ── Test 1: Valid DES block compiles ─────────────────────────────────────────

def test_valid_des_compiles_1cmt():
    """$DES with DADT(1) = -K * A(1) should compile without error."""
    code = "DADT(1) = -K * A(1)"
    compiler = NMTRANCompiler()
    des_fn = compiler.compile_des(code, n_compartments=1)
    assert des_fn is not None


def test_valid_des_compiles_2cmt():
    """$DES with 2-compartment kinetics should compile without error."""
    code = "DADT(1) = -K10*A(1) - K12*A(1) + K21*A(2)\nDADT(2) = K12*A(1) - K21*A(2)"
    compiler = NMTRANCompiler()
    des_fn = compiler.compile_des(code, n_compartments=2)
    assert des_fn is not None


# ── Test 2: A(0) raises CompilerError ────────────────────────────────────────

def test_a_zero_raises_compiler_error():
    """A(0) in $DES should raise CompilerError with helpful message."""
    line = "DADT(1) = -K * A(0)"
    with pytest.raises(CompilerError, match="1-based"):
        _translate_line(line, _INTRINSICS)


def test_a_zero_via_compiler():
    """Compiling a $DES block with A(0) should raise CompilerError."""
    code = "DADT(1) = -K * A(0)"
    compiler = NMTRANCompiler()
    with pytest.raises(CompilerError, match="1-based"):
        compiler.compile_des(code, n_compartments=1)


# ── Test 3: DADT(0) raises CompilerError ─────────────────────────────────────

def test_dadt_zero_raises_compiler_error():
    """DADT(0) in $DES should raise CompilerError with helpful message."""
    line = "DADT(0) = -K * A(1)"
    with pytest.raises(CompilerError, match="1-based"):
        _translate_line(line, _INTRINSICS)


def test_dadt_zero_via_compiler():
    """Compiling a $DES block with DADT(0) should raise CompilerError."""
    code = "DADT(0) = -K * A(1)"
    compiler = NMTRANCompiler()
    with pytest.raises(CompilerError, match="1-based"):
        compiler.compile_des(code, n_compartments=1)


# ── Test 4: Numerical — 1-compartment ODE solution ───────────────────────────

def test_1cmt_ode_numerical_accuracy():
    """
    Compile DADT(1) = -K10 * A(1) with K10=0.1, A(1)_0=100.
    Solve to t=10: analytical A(1) = 100 * exp(-0.1 * 10) = 100*exp(-1) ≈ 36.788.
    Verify within 0.01%.
    """
    from scipy.integrate import solve_ivp

    code = "DADT(1) = -K10 * A(1)"
    compiler = NMTRANCompiler()
    des_fn = compiler.compile_des(code, n_compartments=1)

    K10 = 0.1
    A0 = 100.0
    t_end = 10.0

    def rhs(t, y):
        return des_fn(t, list(y), {"K10": K10}, [], [])

    sol = solve_ivp(rhs, [0.0, t_end], [A0], dense_output=True, rtol=1e-8, atol=1e-10)
    a_t10 = float(sol.y[0, -1])

    expected = A0 * math.exp(-K10 * t_end)  # ≈ 36.7879
    rel_err = abs(a_t10 - expected) / expected
    assert rel_err < 0.0001, (
        f"1-cmt ODE at t=10: got {a_t10:.5f}, expected {expected:.5f}, "
        f"relative error = {rel_err:.4%}"
    )


# ── Test 5: Numerical — 2-compartment steady-state ratio ─────────────────────

def test_2cmt_conservation():
    """
    2-compartment model: verify conservation of mass when K10=0 (no elimination).
    DADT(1) = -K12*A(1) + K21*A(2)
    DADT(2) = K12*A(1) - K21*A(2)

    Total A(1)+A(2) must be conserved (= initial dose = 100).
    At equilibrium (t→∞): A(1)*K12 = A(2)*K21  →  A(2)/A(1) = K12/K21 = 2.0.
    """
    from scipy.integrate import solve_ivp

    # No elimination: K10=0
    code = (
        "DADT(1) = -K12*A(1) + K21*A(2)\n"
        "DADT(2) = K12*A(1) - K21*A(2)"
    )
    compiler = NMTRANCompiler()
    des_fn = compiler.compile_des(code, n_compartments=2)

    K12 = 0.1
    K21 = 0.05
    expected_ratio = K12 / K21  # = 2.0
    initial_total = 100.0

    def rhs(t, y):
        return des_fn(t, list(y), {"K12": K12, "K21": K21}, [], [])

    sol = solve_ivp(rhs, [0.0, 200.0], [initial_total, 0.0], rtol=1e-10, atol=1e-12)
    a1 = float(sol.y[0, -1])
    a2 = float(sol.y[1, -1])

    # Conservation: A1 + A2 should remain = 100
    total = a1 + a2
    assert abs(total - initial_total) < 0.01, (
        f"Mass not conserved: A(1)+A(2) = {total:.4f}, expected {initial_total}"
    )

    # Equilibrium ratio: A2/A1 = K12/K21
    if a1 > 1e-6:
        ratio = a2 / a1
        assert abs(ratio - expected_ratio) < 0.05, (
            f"Equilibrium A(2)/A(1) = {ratio:.4f}, expected {expected_ratio:.4f}"
        )
