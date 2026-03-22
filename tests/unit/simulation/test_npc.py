"""Tests for NPC (Numerical Predictive Check) engine."""

import numpy as np
import pandas as pd
import pytest

from openpkpd.simulation.npc import NPCEngine, NPCResult


def make_sim_result(n_subjects=10, n_obs_per_subj=5, n_replicates=100, seed=42):
    """Create a synthetic SimulationResult-like object."""
    rng = np.random.default_rng(seed)
    records = []

    # REP=0: observed data
    for subj in range(1, n_subjects + 1):
        for t in range(1, n_obs_per_subj + 1):
            true_val = 10.0 * np.exp(-0.1 * t) + rng.normal(0, 0.5)
            records.append({"ID": subj, "TIME": float(t), "DV": true_val, "REP": 0})

    # REP=1..n_replicates: simulated data
    for rep in range(1, n_replicates + 1):
        for subj in range(1, n_subjects + 1):
            for t in range(1, n_obs_per_subj + 1):
                sim_val = 10.0 * np.exp(-0.1 * t) + rng.normal(0, 0.5)
                records.append({"ID": subj, "TIME": float(t), "DV": sim_val, "REP": rep})

    df = pd.DataFrame(records)

    class MockSimResult:
        pass

    mock = MockSimResult()
    mock.simulated_df = df
    mock.seed = seed
    mock.n_replicates = n_replicates

    return mock


class TestNPCResult:
    def test_summary_string(self):
        result = NPCResult(
            pi_lower=0.05,
            pi_upper=0.95,
            obs_below_lower=0.04,
            obs_above_upper=0.05,
            obs_within=0.91,
            expected_within=0.90,
            n_observations=50,
        )
        s = result.summary()
        assert "5%" in s or "0.05" in s or "5.0" in s
        assert "n observations" in s

    def test_dataclass_fields(self):
        result = NPCResult(
            pi_lower=0.05,
            pi_upper=0.95,
            obs_below_lower=0.05,
            obs_above_upper=0.05,
            obs_within=0.90,
            expected_within=0.90,
            n_observations=100,
        )
        assert result.pi_lower == 0.05
        assert result.expected_within == 0.90
        assert result.n_observations == 100


class TestNPCEngine:
    def test_exact_empirical_pvalue_classification(self):
        """Hand-crafted simulated values should give exact below/within/above fractions."""
        df = pd.DataFrame(
            [
                {"ID": 1, "TIME": 1.0, "DV": 5.0, "REP": 0},
                {"ID": 1, "TIME": 2.0, "DV": 5.0, "REP": 0},
                {"ID": 1, "TIME": 3.0, "DV": 5.0, "REP": 0},
                {"ID": 1, "TIME": 1.0, "DV": 6.0, "REP": 1},
                {"ID": 1, "TIME": 1.0, "DV": 7.0, "REP": 2},
                {"ID": 1, "TIME": 2.0, "DV": 4.0, "REP": 1},
                {"ID": 1, "TIME": 2.0, "DV": 6.0, "REP": 2},
                {"ID": 1, "TIME": 3.0, "DV": 3.0, "REP": 1},
                {"ID": 1, "TIME": 3.0, "DV": 4.0, "REP": 2},
            ]
        )

        class MockSim:
            simulated_df = df
            n_replicates = 2

        result = NPCEngine(MockSim()).compute(pi_lower=0.25, pi_upper=0.75)

        assert result.n_observations == 3
        assert result.obs_below_lower == pytest.approx(1.0 / 3.0)
        assert result.obs_within == pytest.approx(1.0 / 3.0)
        assert result.obs_above_upper == pytest.approx(1.0 / 3.0)

    def test_binned_output_stays_aligned_when_some_observations_have_no_matching_simulations(self):
        """Rows without matching simulated values should not shift later time-bin assignments."""
        df = pd.DataFrame(
            [
                {"ID": 1, "TIME": 1.0, "DV": 5.0, "REP": 0},
                {"ID": 1, "TIME": 2.0, "DV": 5.0, "REP": 0},
                {"ID": 1, "TIME": 3.0, "DV": 5.0, "REP": 0},
                {"ID": 1, "TIME": 2.0, "DV": 4.0, "REP": 1},
                {"ID": 1, "TIME": 2.0, "DV": 6.0, "REP": 2},
                {"ID": 1, "TIME": 3.0, "DV": 3.0, "REP": 1},
                {"ID": 1, "TIME": 3.0, "DV": 4.0, "REP": 2},
            ]
        )

        class MockSim:
            simulated_df = df
            n_replicates = 2

        result = NPCEngine(MockSim()).compute(pi_lower=0.25, pi_upper=0.75, n_bins=3)

        assert result.n_observations == 2
        assert result.binned is not None
        assert len(result.binned) == 2
        assert result.binned["t_lo"].min() > 1.0

        last_bin = result.binned.sort_values("t_lo").iloc[-1]
        assert last_bin["t_lo"] > 2.0
        assert last_bin["obs_above_upper"] == pytest.approx(1.0)

    def test_basic_npc(self):
        """NPC produces sensible results for well-specified model."""
        sim_result = make_sim_result(n_subjects=10, n_obs_per_subj=5, n_replicates=50)
        engine = NPCEngine(sim_result)
        result = engine.compute(pi_lower=0.05, pi_upper=0.95)

        assert isinstance(result, NPCResult)
        assert result.n_observations > 0
        assert 0.0 <= result.obs_below_lower <= 1.0
        assert 0.0 <= result.obs_above_upper <= 1.0
        assert 0.0 <= result.obs_within <= 1.0
        assert abs(result.obs_below_lower + result.obs_above_upper + result.obs_within - 1.0) < 0.01

    def test_well_specified_model_within_pi(self):
        """Well-specified model: observed fraction within PI ≈ expected."""
        sim_result = make_sim_result(n_subjects=20, n_obs_per_subj=8, n_replicates=200, seed=7)
        engine = NPCEngine(sim_result)
        result = engine.compute(pi_lower=0.05, pi_upper=0.95)

        # With correct model, obs_within should be near 0.90
        # Allow generous tolerance for randomness
        assert result.obs_within >= 0.50

    def test_expected_within(self):
        """expected_within = pi_upper - pi_lower."""
        sim_result = make_sim_result()
        engine = NPCEngine(sim_result)
        result = engine.compute(pi_lower=0.10, pi_upper=0.90)
        assert result.expected_within == pytest.approx(0.80)

    def test_binned_output(self):
        """n_bins > 0 produces a binned DataFrame."""
        sim_result = make_sim_result()
        engine = NPCEngine(sim_result)
        result = engine.compute(pi_lower=0.05, pi_upper=0.95, n_bins=3)
        assert result.binned is not None
        assert len(result.binned) > 0
        assert "t_mid" in result.binned.columns

    def test_no_observed_data_raises(self):
        """Raises ValueError when no REP=0 rows exist."""
        df = pd.DataFrame({"ID": [1], "TIME": [1.0], "DV": [5.0], "REP": [1]})

        class MockSim:
            simulated_df = df
            n_replicates = 1

        engine = NPCEngine(MockSim())
        with pytest.raises(ValueError, match="observed data"):
            engine.compute()

    def test_missing_rep_column_raises(self):
        """Raises ValueError when REP column is absent."""
        df = pd.DataFrame({"ID": [1], "TIME": [1.0], "DV": [5.0]})

        class MockSim:
            simulated_df = df

        engine = NPCEngine(MockSim())
        with pytest.raises(ValueError, match="REP"):
            engine.compute()
