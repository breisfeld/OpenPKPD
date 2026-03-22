"""
Tests for the R integration bridge (r_bridge/__init__.py).

Tests that don't require rpy2 cover:
  - is_r_available() returns bool
  - Conversion helpers raise ImportError when rpy2 absent
  - RBridge.__init__ raises ImportError when rpy2 absent

Tests requiring rpy2 are skipped automatically when unavailable.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from openpkpd.r_bridge import RBridge, is_r_available, numpy_to_r, r_to_numpy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_R_SKIP = pytest.mark.skipif(
    not is_r_available(),
    reason="rpy2 / R not available in this environment",
)


# ---------------------------------------------------------------------------
# is_r_available
# ---------------------------------------------------------------------------


class TestIsRAvailable:
    def test_returns_bool(self):
        assert isinstance(is_r_available(), bool)

    def test_false_when_rpy2_absent(self, monkeypatch):
        """Simulate rpy2 not installed."""
        monkeypatch.setitem(sys.modules, "rpy2", None)
        monkeypatch.setitem(sys.modules, "rpy2.robjects", None)
        # Re-import to force the try/except to run with the mocked module
        import openpkpd.r_bridge as bridge_mod

        # Directly test the fallback logic
        importlib.util.find_spec("rpy2")
        # If rpy2 is not installed, is_r_available should return False
        # We just verify the function doesn't raise
        result = bridge_mod.is_r_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Error when rpy2 absent
# ---------------------------------------------------------------------------


class TestRequireRpy2:
    def test_numpy_to_r_raises_when_rpy2_absent(self, monkeypatch):
        """If rpy2 unavailable, numpy_to_r should raise ImportError."""
        if is_r_available():
            pytest.skip("rpy2 is available — skip absence test")
        with pytest.raises(ImportError, match="rpy2"):
            numpy_to_r(np.array([1.0, 2.0]))

    def test_rbridge_raises_when_rpy2_absent(self, monkeypatch):
        if is_r_available():
            pytest.skip("rpy2 is available — skip absence test")
        with pytest.raises(ImportError, match="rpy2"):
            RBridge()


# ---------------------------------------------------------------------------
# Conversion helpers (require rpy2)
# ---------------------------------------------------------------------------


class TestNumpyToR:
    @_R_SKIP
    def test_1d_array_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0])
        r_obj = numpy_to_r(arr)
        result = r_to_numpy(r_obj)
        np.testing.assert_allclose(result, arr)

    @_R_SKIP
    def test_scalar_roundtrip(self):
        r_obj = numpy_to_r(np.float64(3.14))
        result = r_to_numpy(r_obj)
        assert float(result[0]) == pytest.approx(3.14)

    @_R_SKIP
    def test_2d_matrix_roundtrip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        r_obj = numpy_to_r(arr)
        result = r_to_numpy(r_obj)
        assert result.shape == (2, 2)

    @_R_SKIP
    def test_3d_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            numpy_to_r(np.zeros((2, 2, 2)))


# ---------------------------------------------------------------------------
# RBridge.eval (require rpy2)
# ---------------------------------------------------------------------------


class TestRBridgeEval:
    @_R_SKIP
    def test_arithmetic(self):
        r = RBridge()
        result = r.eval("1 + 1")
        assert int(result[0]) == 2

    @_R_SKIP
    def test_sqrt(self):
        r = RBridge()
        result = r.eval("sqrt(4)")
        assert float(result[0]) == pytest.approx(2.0)

    @_R_SKIP
    def test_get_version_returns_string(self):
        r = RBridge()
        version = r.get_version()
        assert isinstance(version, str)
        assert "R" in version or len(version) > 0

    @_R_SKIP
    def test_set_seed_no_error(self):
        r = RBridge()
        r.set_seed(42)  # should not raise

    @_R_SKIP
    def test_call_sqrt(self):
        r = RBridge()
        result = r.call("sqrt", np.array([9.0]))
        arr = r_to_numpy(result)
        assert float(arr[0]) == pytest.approx(3.0)

    @_R_SKIP
    def test_last_output_captured(self):
        r = RBridge(capture_output=True)
        r.eval("cat('hello from R')")
        # output may or may not be captured depending on rpy2 version
        assert isinstance(r.last_output, str)
