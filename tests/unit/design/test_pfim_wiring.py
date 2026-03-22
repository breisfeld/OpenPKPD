"""
Tests for PFIM wiring into the model pipeline (P4.4).

Verifies that BuiltModel.design() returns a PFIMEngine and that it
integrates correctly with the existing PFIMEngine API.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.api.model_builder import BuiltModel
from openpkpd.design.pfim import DesignResult, PFIMEngine
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubPopulationModel:
    """Minimal stub for tests that don't exercise the PK model."""

    pass


def _make_params(n_theta=2):
    theta_specs = [ThetaSpec(lower=0.0, init=1.0, upper=10.0) for _ in range(n_theta)]
    omega_specs = [OmegaSpec(block_size=1, values=[0.1])]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.05])]
    return ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)


def _make_built_model(n_theta=2):
    pop_model = _StubPopulationModel()
    params = _make_params(n_theta)
    built = BuiltModel(
        population_model=pop_model,
        params=params,
        estimation_kwargs={"method": "FOCE"},
    )
    return built


# ---------------------------------------------------------------------------
# BuiltModel.design()
# ---------------------------------------------------------------------------


class TestBuiltModelDesign:
    def test_returns_pfim_engine(self):
        built = _make_built_model()
        engine = built.design()
        assert isinstance(engine, PFIMEngine)

    def test_engine_has_population_model(self):
        built = _make_built_model()
        engine = built.design()
        assert engine.population_model is built.population_model

    def test_engine_has_init_params(self):
        built = _make_built_model()
        engine = built.design()
        assert engine.init_params is built.params

    def test_default_sampling_times_set(self):
        built = _make_built_model()
        engine = built.design()
        # Default is np.arange(1, 25, 1) — 24 points
        assert len(engine.sampling_times) == 24

    def test_custom_sampling_times(self):
        built = _make_built_model()
        times = [0.5, 1.0, 2.0, 4.0, 8.0]
        engine = built.design(sampling_times=times)
        np.testing.assert_array_equal(engine.sampling_times, times)

    def test_custom_times_as_array(self):
        built = _make_built_model()
        times = np.array([1.0, 3.0, 6.0, 12.0])
        engine = built.design(sampling_times=times)
        np.testing.assert_array_equal(engine.sampling_times, times)


# ---------------------------------------------------------------------------
# PFIMEngine standalone (existing API, no model needed for basic tests)
# ---------------------------------------------------------------------------


class TestPFIMEngineBasic:
    def test_instantiation_without_model(self):
        engine = PFIMEngine(
            population_model=None,
            init_params=None,
        )
        assert engine.population_model is None

    def test_compute_fim_raises_without_model(self):
        engine = PFIMEngine(population_model=None, init_params=None)
        with pytest.raises(RuntimeError, match="population_model"):
            engine.compute_fim(np.array([1.0, 2.0]))

    def test_optimize_design_raises_without_model(self):
        engine = PFIMEngine(population_model=None, init_params=None)
        with pytest.raises(RuntimeError, match="population_model"):
            engine.optimize_design(n_samples=3)

    def test_design_result_summary(self):
        dr = DesignResult(
            sampling_times=np.array([1.0, 4.0, 12.0]),
            information_matrix=np.eye(2),
            d_efficiency=1.0,
            a_efficiency=2.0,
            condition_number=1.0,
            se_theta=np.array([0.1, 0.2]),
        )
        summary = dr.summary()
        assert "D-efficiency" in summary
        assert "A-criterion" in summary

    def test_design_result_stores_times(self):
        times = np.array([0.5, 2.0, 8.0])
        dr = DesignResult(
            sampling_times=times,
            information_matrix=np.eye(1),
            d_efficiency=0.5,
            a_efficiency=3.0,
            condition_number=10.0,
            se_theta=np.array([0.3]),
        )
        np.testing.assert_array_equal(dr.sampling_times, times)
