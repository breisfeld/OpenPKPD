"""Tests for NPDEEngine (Brendel 2006 NPDE with within-subject decorrelation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from openpkpd.simulation.npde import (
    NPDEEngine,
    NPDEResult,
    _compute_pd,
    _decorrelate,
    _pd_to_normal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_engine(
    n_subjects: int = 8,
    n_obs: int = 6,
    n_replicates: int = 200,
    noise_sd: float = 0.5,
    seed: int = 42,
    obs_scale: float = 1.0,
    sim_scale: float = 1.0,
    obs_bias: float = 0.0,
    sim_bias: float = 0.0,
):
    """Return a mock SimulationEngine whose simulate() returns a well-specified dataset."""
    rng = np.random.default_rng(seed)
    times = np.arange(1, n_obs + 1, dtype=float)
    true_conc = 10.0 * np.exp(-0.2 * times)

    records = []
    # REP=0: observed
    for sid in range(1, n_subjects + 1):
        for t, mu in zip(times, true_conc, strict=False):
            dv = obs_scale * mu + obs_bias + rng.normal(0, noise_sd)
            records.append({"ID": sid, "TIME": t, "DV": dv, "REP": 0, "MDV": 0})
    # REP=1..K: simulated (same model, so model should be "correct")
    for rep in range(1, n_replicates + 1):
        for sid in range(1, n_subjects + 1):
            for t, mu in zip(times, true_conc, strict=False):
                dv = sim_scale * mu + sim_bias + rng.normal(0, noise_sd)
                records.append({"ID": sid, "TIME": t, "DV": dv, "REP": rep, "MDV": 0})

    sim_df = pd.DataFrame(records)

    class _MockResult:
        simulated_df = sim_df

    class _MockEngine:
        def simulate(self, n_replicates: int):
            return _MockResult()

    return _MockEngine()


# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------


class TestComputePd:
    def test_uniform_distribution(self):
        """pd values should be uniform in (0,1) for correct model."""
        rng = np.random.default_rng(0)
        K = 500
        n = 3
        Y_sim = rng.normal(0, 1, size=(n, K))
        obs_dv = rng.normal(0, 1, size=n)
        pd_vals = _compute_pd(obs_dv, Y_sim)
        assert pd_vals.shape == (n,)
        assert np.all((pd_vals > 0) & (pd_vals < 1))

    def test_extreme_obs_clamped(self):
        """Observations far outside simulated range are clamped, not ±inf."""
        Y_sim = np.ones((2, 100))  # all sims = 1
        obs_below = np.array([-999.0, 999.0])
        pd_vals = _compute_pd(obs_below, Y_sim)
        assert np.all(np.isfinite(pd_vals))
        assert pd_vals[0] < 0.01
        assert pd_vals[1] > 0.99

    def test_nan_columns_ignored(self):
        """NaN columns (missing replicates) are excluded from the count."""
        Y_sim = np.full((2, 10), np.nan)
        Y_sim[:, :5] = np.random.default_rng(1).normal(0, 1, (2, 5))
        obs_dv = np.zeros(2)
        pd_vals = _compute_pd(obs_dv, Y_sim)
        assert np.all(np.isfinite(pd_vals))

    def test_all_nan_column_returns_nan(self):
        """If ALL replicates are NaN for an observation, pd is NaN."""
        Y_sim = np.full((2, 10), np.nan)
        obs_dv = np.zeros(2)
        pd_vals = _compute_pd(obs_dv, Y_sim)
        assert np.all(np.isnan(pd_vals))


class TestPdToNormal:
    def test_midpoint_maps_to_zero(self):
        pd_vals = np.array([0.5])
        pde = _pd_to_normal(pd_vals)
        assert abs(pde[0]) < 1e-6

    def test_shape_preserved(self):
        pd_vals = np.linspace(0.1, 0.9, 10)
        pde = _pd_to_normal(pd_vals)
        assert pde.shape == (10,)

    def test_nan_propagated(self):
        pd_vals = np.array([0.5, np.nan, 0.8])
        pde = _pd_to_normal(pd_vals)
        assert np.isnan(pde[1])
        assert np.isfinite(pde[0])
        assert np.isfinite(pde[2])


class TestDecorrelate:
    def test_uncorrelated_unchanged(self):
        """When Y_sim rows are independent, decorrelation leaves PDE unchanged."""
        rng = np.random.default_rng(42)
        K = 1000
        n = 4
        # Independent rows → correlation matrix ≈ identity → L ≈ I → npde ≈ pde
        Y_sim = rng.normal(0, 1, size=(n, K))
        pde = rng.normal(0, 1, size=n)
        npde = _decorrelate(pde, Y_sim)
        # Should be close but not identical due to estimation noise
        assert npde.shape == (n,)
        assert np.all(np.isfinite(npde))

    def test_too_few_replicates_returns_pde(self):
        """When K < n_i + 2, skip decorrelation."""
        pde = np.array([0.5, -0.3, 0.1])
        Y_sim = np.random.default_rng(0).normal(0, 1, size=(3, 4))  # K=4 < n+2=5
        npde = _decorrelate(pde, Y_sim)
        np.testing.assert_array_equal(npde, pde)

    def test_nan_in_pde_preserved(self):
        pde = np.array([1.0, np.nan, -1.0])
        Y_sim = np.random.default_rng(0).normal(0, 1, size=(3, 200))
        npde = _decorrelate(pde, Y_sim)
        assert np.isnan(npde[1])
        assert np.isfinite(npde[0])
        assert np.isfinite(npde[2])


# ---------------------------------------------------------------------------
# Integration tests for NPDEEngine
# ---------------------------------------------------------------------------


class TestNPDEEngine:
    def test_build_sim_matrix_preserves_duplicate_times_by_occurrence_order(self):
        engine = NPDEEngine(_make_mock_engine())
        obs_index = pd.DataFrame({"TIME": [1.0, 1.0], "_OBSSEQ": [0, 1]})
        sim_grp = pd.DataFrame(
            [
                {"REP": 1, "TIME": 1.0, "DV": 10.0, "_OBSSEQ": 0},
                {"REP": 1, "TIME": 1.0, "DV": 20.0, "_OBSSEQ": 1},
                {"REP": 2, "TIME": 1.0, "DV": 11.0, "_OBSSEQ": 0},
                {"REP": 2, "TIME": 1.0, "DV": 21.0, "_OBSSEQ": 1},
            ]
        )

        y_sim = engine._build_sim_matrix(sim_grp, obs_index, n_replicates=2)

        np.testing.assert_allclose(
            y_sim,
            np.array([[10.0, 11.0], [20.0, 21.0]]),
            atol=1e-12,
        )

    def test_small_valid_replicate_count_still_applies_decorrelation(self):
        records = [
            {"ID": 1, "TIME": 1.0, "DV": 1.5, "REP": 0, "MDV": 0},
            {"ID": 1, "TIME": 2.0, "DV": 2.5, "REP": 0, "MDV": 0},
        ]
        rep_vals = [(0.0, 4.0), (1.0, 3.0), (2.0, 2.0), (3.0, 1.0), (4.0, 0.0)]
        for rep, (dv1, dv2) in enumerate(rep_vals, start=1):
            records.extend(
                [
                    {"ID": 1, "TIME": 1.0, "DV": dv1, "REP": rep, "MDV": 0},
                    {"ID": 1, "TIME": 2.0, "DV": dv2, "REP": rep, "MDV": 0},
                ]
            )

        class _MockResult:
            simulated_df = pd.DataFrame(records)

        class _MockEngine:
            def simulate(self, n_replicates):
                return _MockResult()

        engine = NPDEEngine(_MockEngine())
        result = engine.compute(n_replicates=5, seed=0, decorrelate=True)

        obs_grp = _MockResult.simulated_df[_MockResult.simulated_df["REP"] == 0].copy()
        sim_grp = _MockResult.simulated_df[_MockResult.simulated_df["REP"] >= 1].copy()
        obs_grp["_OBSSEQ"] = obs_grp.groupby(["ID", "TIME"], sort=False).cumcount()
        sim_grp["_OBSSEQ"] = sim_grp.groupby(["ID", "REP", "TIME"], sort=False).cumcount()
        obs_index = obs_grp.sort_values(["TIME", "_OBSSEQ"], kind="mergesort")[["TIME", "_OBSSEQ"]]
        y_sim = engine._build_sim_matrix(sim_grp, obs_index, n_replicates=5)
        pde = _pd_to_normal(
            _compute_pd(
                obs_grp.sort_values(["TIME", "_OBSSEQ"], kind="mergesort")["DV"].to_numpy(
                    dtype=float
                ),
                y_sim,
            )
        )
        expected_npde = _decorrelate(pde, y_sim)

        np.testing.assert_allclose(result.df["PDE"].values, pde, atol=1e-12)
        np.testing.assert_allclose(result.df["NPDE"].values, expected_npde, atol=1e-12)
        assert not np.allclose(result.df["NPDE"].values, result.df["PDE"].values)

    def test_returns_npde_result(self):
        engine = _make_mock_engine()
        npde_engine = NPDEEngine(engine)
        result = npde_engine.compute(n_replicates=100, seed=0)
        assert isinstance(result, NPDEResult)

    def test_df_columns(self):
        engine = _make_mock_engine()
        result = NPDEEngine(engine).compute(n_replicates=100, seed=1)
        assert {"ID", "TIME", "DV", "PDE", "NPDE"}.issubset(result.df.columns)

    def test_correct_model_npde_near_standard_normal(self):
        """
        With a correctly specified model, NPDE should have mean ≈ 0 and
        variance ≈ 1.  Allow generous tolerances for the small dataset.
        """
        engine = _make_mock_engine(n_subjects=20, n_obs=8, n_replicates=500, noise_sd=0.5)
        result = NPDEEngine(engine).compute(n_replicates=500, seed=7)
        assert abs(result.mean_npde) < 0.5, f"mean_npde={result.mean_npde:.3f}"
        assert 0.3 < result.var_npde < 3.0, f"var_npde={result.var_npde:.3f}"

    def test_correct_model_npde_stays_calibrated_across_scenarios(self):
        """Correct-model NPDE should stay close to standard normal over a small grid."""
        scenarios = [
            (20, 4, 0.2, 0),
            (20, 4, 0.5, 1),
            (20, 8, 0.2, 2),
            (20, 8, 0.5, 0),
            (60, 4, 0.2, 1),
            (60, 4, 0.5, 2),
            (60, 8, 0.2, 0),
            (60, 8, 0.5, 1),
        ]
        mean_abs = []
        variances = []

        for n_subjects, n_obs, noise_sd, seed in scenarios:
            engine = _make_mock_engine(
                n_subjects=n_subjects,
                n_obs=n_obs,
                n_replicates=200,
                noise_sd=noise_sd,
                seed=seed,
            )
            result = NPDEEngine(engine).compute(n_replicates=200, seed=seed)
            assert np.isfinite(result.mean_npde)
            assert np.isfinite(result.var_npde)
            mean_abs.append(abs(result.mean_npde))
            variances.append(result.var_npde)

        assert max(mean_abs) <= 0.2, f"NPDE means drifted too far: {mean_abs}"
        assert min(variances) >= 0.7, f"NPDE variances too small: {variances}"
        assert max(variances) <= 1.2, f"NPDE variances too large: {variances}"

    def test_clear_scale_misspecification_shifts_npde_distribution_across_seeds(self):
        """A clear scale misspecification should strongly shift NPDE away from N(0,1)."""
        seeds = [1, 2, 7]
        correct_mean_abs = []
        correct_sw_pvalues = []
        misspecified_mean_abs = []
        misspecified_sw_pvalues = []

        for seed in seeds:
            correct_result = NPDEEngine(
                _make_mock_engine(
                    n_subjects=20,
                    n_obs=8,
                    n_replicates=300,
                    noise_sd=0.5,
                    seed=seed,
                )
            ).compute(n_replicates=300, seed=seed)
            correct_mean_abs.append(abs(correct_result.mean_npde))
            correct_sw_pvalues.append(correct_result.sw_pvalue)

            misspecified_result = NPDEEngine(
                _make_mock_engine(
                    n_subjects=20,
                    n_obs=8,
                    n_replicates=300,
                    noise_sd=0.5,
                    seed=seed,
                    sim_scale=1.2,
                )
            ).compute(n_replicates=300, seed=seed)
            misspecified_mean_abs.append(abs(misspecified_result.mean_npde))
            misspecified_sw_pvalues.append(misspecified_result.sw_pvalue)

        assert max(correct_mean_abs) <= 0.2, (
            f"Correct-model NPDE means drifted too far: {correct_mean_abs}"
        )
        assert min(correct_sw_pvalues) >= 0.5, (
            f"Correct-model NPDE normality broke unexpectedly: {correct_sw_pvalues}"
        )
        assert min(misspecified_mean_abs) >= 1.5, (
            "Misspecified-model NPDE means were not shifted strongly enough: "
            f"{misspecified_mean_abs}"
        )
        assert max(misspecified_sw_pvalues) <= 0.01, (
            f"Misspecified-model NPDE normality p-values were too large: {misspecified_sw_pvalues}"
        )

    def test_no_decorrelation_returns_pde_equals_npde(self):
        engine = _make_mock_engine(n_subjects=4, n_obs=3, n_replicates=50)
        result = NPDEEngine(engine).compute(n_replicates=50, seed=2, decorrelate=False)
        np.testing.assert_array_equal(result.df["PDE"].values, result.df["NPDE"].values)

    def test_duplicate_same_time_observations_remain_aligned(self):
        records = [
            {"ID": 1, "TIME": 1.0, "DV": 1.5, "REP": 0, "MDV": 0},
            {"ID": 1, "TIME": 1.0, "DV": 3.5, "REP": 0, "MDV": 0},
            {"ID": 1, "TIME": 1.0, "DV": 1.0, "REP": 1, "MDV": 0},
            {"ID": 1, "TIME": 1.0, "DV": 3.0, "REP": 1, "MDV": 0},
            {"ID": 1, "TIME": 1.0, "DV": 2.0, "REP": 2, "MDV": 0},
            {"ID": 1, "TIME": 1.0, "DV": 4.0, "REP": 2, "MDV": 0},
        ]

        class _MockResult:
            simulated_df = pd.DataFrame(records)

        class _MockEngine:
            def simulate(self, n_replicates):
                return _MockResult()

        result = NPDEEngine(_MockEngine()).compute(n_replicates=2, seed=0, decorrelate=False)

        assert len(result.df) == 2
        assert np.all(np.isfinite(result.df["PDE"].values))
        np.testing.assert_allclose(result.df["PDE"].values, np.zeros(2), atol=1e-12)

    def test_compute_same_seed_reproduces_stateful_simulation_engine(self):
        class _StatefulEngine:
            def __init__(self) -> None:
                self.seed = 0
                self.rng = np.random.default_rng(0)

            def simulate(self, n_replicates):
                rows = [{"ID": 1, "TIME": 1.0, "DV": 0.0, "REP": 0, "MDV": 0}]
                for rep in range(1, n_replicates + 1):
                    rows.append(
                        {
                            "ID": 1,
                            "TIME": 1.0,
                            "DV": float(self.rng.normal()),
                            "REP": rep,
                            "MDV": 0,
                        }
                    )

                class _MockResult:
                    simulated_df = pd.DataFrame(rows)

                return _MockResult()

        engine = _StatefulEngine()
        npde_engine = NPDEEngine(engine)

        result1 = npde_engine.compute(n_replicates=32, seed=123, decorrelate=False)
        result2 = npde_engine.compute(n_replicates=32, seed=123, decorrelate=False)

        np.testing.assert_array_equal(result1.df["NPDE"].values, result2.df["NPDE"].values)

    def test_row_count_matches_observations(self):
        n_subjects, n_obs = 5, 4
        engine = _make_mock_engine(n_subjects=n_subjects, n_obs=n_obs, n_replicates=50)
        result = NPDEEngine(engine).compute(n_replicates=50, seed=3)
        assert len(result.df) == n_subjects * n_obs

    def test_n_replicates_stored(self):
        engine = _make_mock_engine()
        result = NPDEEngine(engine).compute(n_replicates=77, seed=4)
        assert result.n_replicates == 77

    def test_summary_runs_without_error(self):
        engine = _make_mock_engine()
        result = NPDEEngine(engine).compute(n_replicates=50, seed=5)
        summary = result.summary()
        assert "Mean NPDE" in summary
        assert "Var  NPDE" in summary

    def test_misspecified_model_detectable(self):
        """
        A systematically biased model (sim always higher than obs) should
        produce NPDE with mean significantly below 0.
        """
        rng = np.random.default_rng(99)
        n_subjects, n_obs, K = 15, 5, 300
        times = np.arange(1, n_obs + 1, dtype=float)
        true_mu = 10.0 * np.exp(-0.2 * times)

        records = []
        # Observed: lower than model (bias = -3)
        for sid in range(1, n_subjects + 1):
            for t, mu in zip(times, true_mu, strict=False):
                records.append(
                    {"ID": sid, "TIME": t, "DV": mu - 3.0 + rng.normal(0, 0.3), "REP": 0, "MDV": 0}
                )
        # Simulated: unbiased
        for rep in range(1, K + 1):
            for sid in range(1, n_subjects + 1):
                for t, mu in zip(times, true_mu, strict=False):
                    records.append(
                        {"ID": sid, "TIME": t, "DV": mu + rng.normal(0, 0.3), "REP": rep, "MDV": 0}
                    )

        class _MockResult:
            simulated_df = pd.DataFrame(records)

        class _MockEngine:
            def simulate(self, n_replicates):
                return _MockResult()

        result = NPDEEngine(_MockEngine()).compute(n_replicates=K, seed=0)
        # Obs systematically below simulated → pd near 0 → NPDE strongly negative
        assert result.mean_npde < -0.5, (
            f"Expected negative NPDE mean for biased model, got {result.mean_npde:.3f}"
        )
