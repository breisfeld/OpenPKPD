"""Tests for SSE (Stochastic Simulation & Re-estimation) infrastructure."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import openpkpd.estimation as estimation_module
import openpkpd.estimation.fo as fo_module
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.base import EstimationResult
from openpkpd.simulation.engine import SimulationResult
from openpkpd.simulation.sse import SSEEngine, SSEResult, _empirical_coverage


class TestSSEResult:
    def test_summary_string(self):
        """SSEResult.summary() returns non-empty string."""
        result = SSEResult(
            n_replicates=100,
            parameter_names=["THETA1", "OMEGA(1,1)"],
            true_values={"THETA1": 1.0, "OMEGA(1,1)": 0.04},
            estimates=pd.DataFrame(
                {
                    "THETA1": [1.02, 0.98, 1.05],
                    "OMEGA(1,1)": [0.041, 0.039, 0.042],
                }
            ),
            bias={"THETA1": 0.02, "OMEGA(1,1)": 0.01},
            rmse={"THETA1": 0.05, "OMEGA(1,1)": 0.003},
            coverage_95={"THETA1": 0.92, "OMEGA(1,1)": 0.94},
            convergence_rate=0.96,
        )
        s = result.summary()
        assert len(s) > 0
        assert "THETA1" in s
        assert "96.0%" in s or "96" in s

    def test_dataclass_fields(self):
        """SSEResult has all expected fields."""
        result = SSEResult(
            n_replicates=50,
            parameter_names=["THETA1"],
            true_values={"THETA1": 2.0},
            estimates=pd.DataFrame({"THETA1": [2.1, 1.9]}),
            bias={"THETA1": 0.0},
            rmse={"THETA1": 0.1},
            coverage_95={"THETA1": 0.95},
            convergence_rate=1.0,
        )
        assert result.n_replicates == 50
        assert result.convergence_rate == 1.0
        assert "THETA1" in result.true_values


class TestEmpiricalCoverage:
    """Tests for the _empirical_coverage helper."""

    def test_returns_float(self):
        rng = np.random.default_rng(0)
        estimates = rng.normal(5.0, 0.5, 100)
        cov = _empirical_coverage(estimates, true_value=5.0)
        assert isinstance(cov, float)

    def test_range_zero_to_one(self):
        rng = np.random.default_rng(1)
        estimates = rng.normal(5.0, 0.5, 200)
        cov = _empirical_coverage(estimates, true_value=5.0)
        assert 0.0 <= cov <= 1.0

    def test_true_at_mean_gives_near_full_coverage(self):
        """When true value equals the mean, most estimates should be within 1.96 SD."""
        rng = np.random.default_rng(2)
        estimates = rng.normal(5.0, 1.0, 1000)
        cov = _empirical_coverage(estimates, true_value=5.0)
        # Almost all estimates are within 1.96 SD of 5.0 when true=mean
        assert cov > 0.90

    def test_true_far_outside_gives_low_coverage(self):
        """When true value is far from all estimates, coverage should be near 0."""
        rng = np.random.default_rng(3)
        estimates = rng.normal(5.0, 0.1, 200)  # tight cluster
        cov = _empirical_coverage(estimates, true_value=100.0)  # far outside
        assert cov == pytest.approx(0.0, abs=0.01)

    def test_too_few_estimates_returns_nan(self):
        cov = _empirical_coverage(np.array([5.0]), true_value=5.0)
        assert np.isnan(cov)

    def test_all_identical_true_in_range(self):
        """All estimates identical and equal to true → coverage = 1."""
        estimates = np.full(10, 3.0)
        cov = _empirical_coverage(estimates, true_value=3.0)
        assert cov == pytest.approx(1.0)

    def test_all_identical_true_outside(self):
        """All estimates identical but true value differs → coverage = 0."""
        estimates = np.full(10, 3.0)
        cov = _empirical_coverage(estimates, true_value=10.0)
        assert cov == pytest.approx(0.0)

    def test_nan_estimates_ignored(self):
        """NaN estimates should be filtered out."""
        estimates = np.array([5.0, np.nan, 5.2, np.nan, 4.9])
        cov = _empirical_coverage(estimates, true_value=5.0)
        assert np.isfinite(cov)


class TestSSEEngine:
    def test_run_reestimates_on_simulated_dataset_not_original_observations(self, monkeypatch):
        original_df = pd.DataFrame(
            [
                {"ID": 1, "TIME": 0.0, "AMT": 100.0, "DV": 0.0, "EVID": 1, "MDV": 1},
                {"ID": 1, "TIME": 1.0, "AMT": 0.0, "DV": 1.0, "EVID": 0, "MDV": 0},
            ]
        )
        pop_model = SimpleNamespace(
            dataset=NONMEMDataset.from_dataframe(original_df), params=object()
        )
        true_result = EstimationResult(
            theta_final=np.array([1.0]),
            omega_final=np.zeros((0, 0)),
            sigma_final=np.zeros((0, 0)),
            ofv=0.0,
            converged=True,
        )

        class FakeSimulationEngine:
            def __init__(self, population_model, result, seed=42):
                self.seed = seed

            def simulate(self, n_replicates=1):
                df = pd.DataFrame(
                    [
                        {"ID": 1, "TIME": 1.0, "DV": 1.0, "REP": 0},
                        {"ID": 1, "TIME": 1.0, "DV": 7.0, "REP": 1},
                    ]
                )
                return SimulationResult(simulated_df=df, seed=self.seed, n_replicates=n_replicates)

        class LegacyEstimator:
            def __init__(self, population_model=None):
                self.population_model = population_model

            def estimate(self, population_model=None, init_params=None):
                model = population_model or self.population_model
                obs = model.dataset.df[
                    (model.dataset.df["EVID"] == 0) & (model.dataset.df["MDV"] == 0)
                ]
                value = float(obs["DV"].mean())
                return EstimationResult(
                    theta_final=np.array([value]),
                    omega_final=np.zeros((0, 0)),
                    sigma_final=np.zeros((0, 0)),
                    ofv=0.0,
                    converged=True,
                )

        class RoutedMethod:
            def estimate(self, population_model, init_params):
                obs = population_model.dataset.df[
                    (population_model.dataset.df["EVID"] == 0)
                    & (population_model.dataset.df["MDV"] == 0)
                ]
                value = float(obs["DV"].mean())
                return EstimationResult(
                    theta_final=np.array([value]),
                    omega_final=np.zeros((0, 0)),
                    sigma_final=np.zeros((0, 0)),
                    ofv=0.0,
                    converged=True,
                )

        monkeypatch.setattr("openpkpd.simulation.engine.SimulationEngine", FakeSimulationEngine)
        monkeypatch.setattr(fo_module, "FOEstimator", LegacyEstimator, raising=False)
        monkeypatch.setattr(
            estimation_module, "get_estimation_method", lambda *args, **kwargs: RoutedMethod()
        )

        result = SSEEngine(pop_model, true_result, estimation_method="FO").run(
            n_replicates=1, seed=123
        )

        assert result.convergence_rate == pytest.approx(1.0)
        assert result.estimates["THETA1"].tolist() == pytest.approx([7.0])
        assert result.bias["THETA1"] == pytest.approx(6.0)
        assert result.rmse["THETA1"] == pytest.approx(6.0)

    def test_run_computes_exact_bias_and_rmse_from_replicate_estimates(self, monkeypatch):
        original_df = pd.DataFrame(
            [
                {"ID": 1, "TIME": 0.0, "AMT": 100.0, "DV": 0.0, "EVID": 1, "MDV": 1},
                {"ID": 1, "TIME": 1.0, "AMT": 0.0, "DV": 1.0, "EVID": 0, "MDV": 0},
            ]
        )
        pop_model = SimpleNamespace(
            dataset=NONMEMDataset.from_dataframe(original_df), params=object()
        )
        true_result = EstimationResult(
            theta_final=np.array([1.0]),
            omega_final=np.zeros((0, 0)),
            sigma_final=np.zeros((0, 0)),
            ofv=0.0,
            converged=True,
        )
        simulated_dvs = iter([1.0, 3.0])

        class FakeSimulationEngine:
            def __init__(self, population_model, result, seed=42):
                self.seed = seed

            def simulate(self, n_replicates=1):
                sim_dv = next(simulated_dvs)
                df = pd.DataFrame(
                    [
                        {"ID": 1, "TIME": 1.0, "DV": 1.0, "REP": 0},
                        {"ID": 1, "TIME": 1.0, "DV": sim_dv, "REP": 1},
                    ]
                )
                return SimulationResult(simulated_df=df, seed=self.seed, n_replicates=n_replicates)

        class RoutedMethod:
            def estimate(self, population_model, init_params):
                obs = population_model.dataset.df[
                    (population_model.dataset.df["EVID"] == 0)
                    & (population_model.dataset.df["MDV"] == 0)
                ]
                value = float(obs["DV"].mean())
                return EstimationResult(
                    theta_final=np.array([value]),
                    omega_final=np.zeros((0, 0)),
                    sigma_final=np.zeros((0, 0)),
                    ofv=0.0,
                    converged=True,
                )

        monkeypatch.setattr("openpkpd.simulation.engine.SimulationEngine", FakeSimulationEngine)
        monkeypatch.setattr(
            estimation_module, "get_estimation_method", lambda *args, **kwargs: RoutedMethod()
        )

        result = SSEEngine(pop_model, true_result, estimation_method="FO").run(
            n_replicates=2, seed=123
        )

        assert result.estimates["THETA1"].tolist() == pytest.approx([1.0, 3.0])
        assert result.bias["THETA1"] == pytest.approx(1.0)
        assert result.rmse["THETA1"] == pytest.approx(np.sqrt(2.0))
        assert result.coverage_95["THETA1"] == pytest.approx(1.0)
