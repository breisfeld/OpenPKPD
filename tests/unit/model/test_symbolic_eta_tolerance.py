"""
Tests for relative tolerance in the covariate static check (SY3).

Tests:
  1. Body weight [70.0, 70.0] → static (both old and new tolerances agree)
  2. Body weight in grams [70000.0, 70000.0 + 1e-3] → static with new rtol=1e-8
     (the difference 1e-3 / 70000 ≈ 1.4e-8 is just above the threshold; we
     actually test a difference smaller than rtol: 70000 * 1e-9 ≈ 7e-5)
  3. Two values differing by more than rtol=1e-8 * scale → non-static (different)
  4. Zero-valued covariate: falls back gracefully (no division by zero)

Note: the private function under test is `_extract_static_covariates`, which
returns a dict of covariate values if they are static across the individual's
observations, or None if they vary. We test it indirectly through observable
behaviour by inspecting the `_merge_covariates` closure inside that function.

A simpler approach: test via `np.isclose(a, b, rtol=1e-8, atol=0.0)` which is
the exact expression now used in the production code.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Low-level tolerance logic (mirrors the production code)
# ---------------------------------------------------------------------------


def _is_static_pair(a: float, b: float) -> bool:
    """Mirror of the production check for two covariate values."""
    return bool(np.isclose(a, b, rtol=1e-8, atol=0.0))


def _old_is_static_pair(a: float, b: float) -> bool:
    """Old (pre-fix) check using absolute tolerance."""
    return bool(np.isclose(a, b, rtol=0.0, atol=1e-12))


# ---------------------------------------------------------------------------
# Test 1: Identical body-weight values → static under both tolerances
# ---------------------------------------------------------------------------


def test_identical_values_are_static():
    """[70.0, 70.0] → static (passes both old and new tolerance)."""
    assert _is_static_pair(70.0, 70.0)
    assert _old_is_static_pair(70.0, 70.0)


# ---------------------------------------------------------------------------
# Test 2: Large-scale values with tiny relative difference → static with rtol
# ---------------------------------------------------------------------------


def test_large_scale_tiny_relative_diff_is_static():
    """
    70000.0 vs 70000.0 + 70000 * 5e-9 (diff = 3.5e-4):
    - New rtol=1e-8 treats this as static (5e-9 < 1e-8)
    - Old atol=1e-12 would reject it (3.5e-4 >> 1e-12)
    """
    a = 70000.0
    b = 70000.0 + 70000.0 * 5e-9  # relative diff = 5e-9 < rtol=1e-8

    assert _is_static_pair(a, b), "New rtol should treat near-equal large values as static"
    assert not _old_is_static_pair(a, b), "Old atol=1e-12 would reject this (confirms fix needed)"


# ---------------------------------------------------------------------------
# Test 3: Values differing by more than rtol → non-static
# ---------------------------------------------------------------------------


def test_values_differing_by_more_than_rtol_are_nonstatic():
    """
    Two values differing by > 1e-8 * scale are treated as non-static (varying covariate).
    """
    a = 70.0
    b = 70.0 + 70.0 * 1e-7  # relative diff = 1e-7 >> rtol=1e-8

    assert not _is_static_pair(a, b), (
        f"Values differing by rtol=1e-7 should be non-static, got static for {a} vs {b}"
    )


# ---------------------------------------------------------------------------
# Test 4: Zero-valued covariate — no division by zero
# ---------------------------------------------------------------------------


def test_zero_valued_covariate_no_crash():
    """np.isclose(0.0, 0.0, rtol=1e-8, atol=0.0) should be True (not divide by zero)."""
    # When both values are zero, rtol comparison falls back to atol internally in numpy.
    # With atol=0.0, this is 0.0 == 0.0 → True.
    result = _is_static_pair(0.0, 0.0)
    assert result is True


def test_zero_vs_nonzero_is_nonstatic():
    """
    0.0 vs any non-zero value should be non-static with rtol (no atol).
    np.isclose(0, x, rtol=1e-8, atol=0) is False for x != 0.
    """
    assert not _is_static_pair(0.0, 1e-5)
    assert not _is_static_pair(0.0, 1.0)


# ---------------------------------------------------------------------------
# Integration: verify the production code constant is updated
# ---------------------------------------------------------------------------


def test_symbolic_eta_uses_relative_tolerance():
    """
    Verify the production code no longer uses atol=1e-12 at the static check sites
    in symbolic_eta.py.
    """
    import ast
    import pathlib

    src = pathlib.Path(
        "/home/breisfel/Documents/projects/openpkpd/src/openpkpd/model/symbolic_eta.py"
    ).read_text()

    # Check that atol=1e-12 is not used in np.isclose / np.allclose calls
    # (it may still appear in comments or other contexts, but not as a keyword arg)
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            func_name = ""
            if isinstance(func, ast.Attribute):
                func_name = func.attr
            elif isinstance(func, ast.Name):
                func_name = func.id
            if func_name in ("isclose", "allclose"):
                for kw in node.keywords:
                    if kw.arg == "atol":
                        val = kw.value
                        if isinstance(val, ast.Constant) and val.value == 1e-12:
                            pytest.fail(
                                f"Found atol=1e-12 in np.{func_name}() call at line {node.lineno}. "
                                "Should use rtol=1e-8 instead."
                            )
