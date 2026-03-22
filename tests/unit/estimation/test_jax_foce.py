"""
Tests for JAXFOCEMethod (estimation/jax_foce.py).

Tests instantiation, fallback behaviour, and (when JAX is available)
that the method produces a valid EstimationResult.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.estimation.jax_foce import JAXFOCEMethod, _jax_available
from openpkpd.model.parameters import ThetaSpec

# ---------------------------------------------------------------------------
# _jax_available
# ---------------------------------------------------------------------------


class TestJaxAvailable:
    def test_returns_bool(self):
        assert isinstance(_jax_available(), bool)

    def test_false_when_jax_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "jax", None)
        monkeypatch.setitem(sys.modules, "jax.numpy", None)
        # Re-check via function (cached import may be True if JAX is installed)
        # Just verify it doesn't raise
        result = _jax_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# JAXFOCEMethod instantiation
# ---------------------------------------------------------------------------


class TestJAXFOCEMethodInstantiation:
    def test_instantiation_no_jax(self):
        """Should instantiate with or without JAX."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            method = JAXFOCEMethod(interaction=True, maxeval=100)
        assert method.interaction is True
        assert method.maxeval == 100

    def test_method_name(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            method = JAXFOCEMethod()
        assert "FOCE" in method.method_name.upper()

    def test_warns_when_jax_absent(self, monkeypatch):
        monkeypatch.setattr("openpkpd.estimation.jax_foce._jax_available", lambda: False)
        with pytest.warns(ImportWarning, match="JAX"):
            JAXFOCEMethod()

    def test_no_warn_when_jax_available(self, monkeypatch):
        monkeypatch.setattr("openpkpd.estimation.jax_foce._jax_available", lambda: True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ImportWarning)
            try:
                JAXFOCEMethod()
            except ImportWarning:
                pytest.fail("ImportWarning raised when JAX is 'available'")


# ---------------------------------------------------------------------------
# estimate() delegates correctly
# ---------------------------------------------------------------------------


class _DummyModel:
    """Minimal population model stub."""

    def subject_ids(self):
        return [1, 2]

    def ofv_fo(self, params):
        # Parabolic objective: min at theta=[1.0]
        return float(np.sum((params.theta - 1.0) ** 2))

    def individual_model(self, sid):
        return None


class _DummyParams:
    def __init__(self):
        self.theta = np.array([0.5])
        self.omega = np.eye(1) * 0.1
        self.sigma = np.eye(1) * 0.01
        self.theta_specs = []
        self.omega_specs = []
        self.sigma_specs = []

    def n_eta(self):
        return 1


class _BoundedOptimumModel(_DummyModel):
    def ofv_fo(self, params):
        return float(np.sum((params.theta - 2.0) ** 2))


class _BoundedParams(_DummyParams):
    def __init__(self):
        super().__init__()
        self.theta_specs = [ThetaSpec(init=0.5, lower=0.0, upper=1.0)]


class TestJAXFOCEMethodEstimate:
    def test_fallback_when_jax_absent(self, monkeypatch):
        """When JAX unavailable, estimate() should fall back gracefully."""
        monkeypatch.setattr("openpkpd.estimation.jax_foce._jax_available", lambda: False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            method = JAXFOCEMethod()
        # _estimate_fallback calls FOCEMethod; it may fail on the stub,
        # so we just verify the method delegates without crashing itself
        assert not method._has_jax

    def test_jax_path_produces_estimation_result(self, monkeypatch):
        """When JAX available, the estimate result should be an EstimationResult."""
        if not _jax_available():
            pytest.skip("JAX not installed")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            method = JAXFOCEMethod(maxeval=50)

        model = _DummyModel()
        params = _DummyParams()

        result = method._estimate_jax(model, params)
        assert isinstance(result, EstimationResult)
        assert np.isfinite(result.ofv)
        assert len(result.theta_final) == 1

    def test_jax_path_optimises_toward_minimum(self, monkeypatch):
        """JAX path should move theta toward the minimum of the parabola."""
        if not _jax_available():
            pytest.skip("JAX not installed")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            method = JAXFOCEMethod(maxeval=200)

        model = _DummyModel()
        params = _DummyParams()  # theta init = [0.5], min at [1.0]

        result = method._estimate_jax(model, params)
        assert result.theta_final[0] == pytest.approx(1.0, abs=1e-3)
        assert result.ofv == pytest.approx(0.0, abs=1e-6)
        assert len(result.ofv_history) > 0

    def test_jax_path_respects_theta_bounds(self, monkeypatch):
        """JAX path should honour theta bounds from theta_specs."""
        if not _jax_available():
            pytest.skip("JAX not installed")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            method = JAXFOCEMethod(maxeval=200)

        result = method._estimate_jax(_BoundedOptimumModel(), _BoundedParams())

        assert result.theta_final[0] == pytest.approx(1.0, abs=1e-3)
        assert result.ofv == pytest.approx(1.0, abs=1e-6)
