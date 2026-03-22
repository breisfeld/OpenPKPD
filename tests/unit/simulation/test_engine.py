"""
Unit tests for SimulationEngine and VPCEngine.

Tests verify:
  - SimulationEngine produces a SimulationResult with correct structure
  - REP=0 rows contain observed data
  - REP>0 rows contain simulated data with plausible DV values
  - Multiple replicates produce different simulated values (stochastic)
  - SimulationResult DataFrame has required columns
  - VPCEngine produces VPCResult with expected structure
  - Seed reproducibility
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.parser.code_compiler import NMTRANCompiler
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.simulation.engine import SimulationEngine, SimulationResult, _draw_mvn
from openpkpd.simulation.vpc import VPCEngine, VPCResult

# ── Test fixture helpers ───────────────────────────────────────────────────────


def _advan1_superposition(
    times: np.ndarray,
    dose_events: list[tuple[float, float]],
    k: float,
    v: float,
) -> np.ndarray:
    """Closed-form ADVAN1 concentration using bolus-dose superposition."""
    times = np.asarray(times, dtype=float)
    conc = np.zeros_like(times, dtype=float)
    for dose_time, amount in dose_events:
        dt = times - float(dose_time)
        mask = dt > 0.0  # pre-dose convention at exact dose time
        conc[mask] += float(amount) * np.exp(-k * dt[mask]) / v
    return conc


def _make_simple_dataset(n_subj: int = 4, seed: int = 0) -> NONMEMDataset:
    """
    Build a minimal NONMEM-format dataset (1-cmt IV, single dose per subject).
    """
    rng = np.random.default_rng(seed)
    obs_times = np.array([1.0, 4.0, 8.0, 12.0, 24.0])
    K, V = 0.1, 20.0

    rows = []
    for i in range(1, n_subj + 1):
        # Dose row
        rows.append({"ID": i, "TIME": 0.0, "AMT": 100.0, "DV": 0.0, "EVID": 1, "MDV": 1})
        dose_amt = 100.0 * np.exp(rng.normal(0, 0.1))
        for t in obs_times:
            conc = dose_amt / V * np.exp(-K * t) * (1 + rng.normal(0, 0.05))
            rows.append(
                {"ID": i, "TIME": t, "AMT": 0.0, "DV": max(conc, 0.01), "EVID": 0, "MDV": 0}
            )

    df = pd.DataFrame(rows)
    return NONMEMDataset.from_dataframe(df)


def _make_params(
    theta: list[float],
    omega_diag: list[float],
    sigma_diag: list[float],
) -> ParameterSet:
    """Build a simple ParameterSet from lists of initial values."""
    theta_specs = [ThetaSpec(init=v, lower=0.0) for v in theta]
    omega_specs = [OmegaSpec(block_size=1, values=[v]) for v in omega_diag]
    sigma_specs = [SigmaSpec(block_size=1, values=[v]) for v in sigma_diag]
    return ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)


def _make_estimation_result(
    theta: list[float],
    omega_diag: list[float],
    sigma_diag: list[float],
    n_subj: int = 4,
) -> EstimationResult:
    """Build a mock EstimationResult at given parameter values."""
    n_eta = len(omega_diag)
    omega = np.diag(omega_diag)
    sigma = np.diag(sigma_diag)
    post_hoc = {i: np.zeros(n_eta) for i in range(1, n_subj + 1)}
    return EstimationResult(
        theta_final=np.array(theta),
        omega_final=omega,
        sigma_final=sigma,
        ofv=0.0,
        converged=True,
        post_hoc_etas=post_hoc,
    )


def _make_exact_advan1_setup(n_subj: int = 2) -> tuple[PopulationModel, EstimationResult]:
    """Return an ADVAN1 setup with zero ETA/EPS variance for exact checks."""
    ds = _make_simple_dataset(n_subj=n_subj, seed=123)
    params = _make_params(theta=[0.1, 20.0], omega_diag=[0.0], sigma_diag=[0.0])
    compiler = NMTRANCompiler()
    pk_callable = compiler.compile_pk("K = THETA(1)*EXP(ETA(1))\nV = THETA(2)")
    error_callable = compiler.compile_error("Y = F*(1 + EPS(1))")

    pop_model = PopulationModel(
        dataset=ds,
        pk_subroutine=ADVAN1(),
        params=params,
        pk_callable=pk_callable,
        error_callable=error_callable,
        trans=1,
        advan=1,
    )
    result = _make_estimation_result(
        theta=[0.1, 20.0],
        omega_diag=[0.0],
        sigma_diag=[0.0],
        n_subj=n_subj,
    )
    return pop_model, result


def _make_default_advan2_setup(n_subj: int = 6) -> tuple[PopulationModel, EstimationResult]:
    """Return a default ADVAN2/TRANS2 setup without custom $PK/$ERROR callables."""
    rng = np.random.default_rng(777)
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0], dtype=float)
    dose = 200.0
    ka, cl, v = 1.2, 5.0, 50.0
    k = cl / v

    rows = []
    for sid in range(1, n_subj + 1):
        rows.append({"ID": sid, "TIME": 0.0, "AMT": dose, "DV": 0.0, "EVID": 1, "MDV": 1, "CMT": 1})
        for t in obs_times:
            conc = dose * ka / (v * (ka - k)) * (np.exp(-k * t) - np.exp(-ka * t))
            dv = max(conc * (1 + rng.normal(0.0, 0.05)), 0.01)
            rows.append({"ID": sid, "TIME": t, "AMT": 0.0, "DV": dv, "EVID": 0, "MDV": 0, "CMT": 1})

    ds = NONMEMDataset.from_dataframe(pd.DataFrame(rows))
    params = _make_params(theta=[ka, cl, v], omega_diag=[0.09, 0.04, 0.01], sigma_diag=[0.04])
    pop_model = PopulationModel(
        dataset=ds,
        pk_subroutine=ADVAN2(),
        params=params,
        trans=2,
        advan=2,
    )
    result = _make_estimation_result(
        theta=[ka, cl, v],
        omega_diag=[0.09, 0.04, 0.01],
        sigma_diag=[0.04],
        n_subj=n_subj,
    )
    return pop_model, result


def _make_error_model_setup(
    error_callable,
    *,
    sigma_diag: list[float],
    n_subj: int = 3,
) -> tuple[PopulationModel, EstimationResult]:
    ds = _make_simple_dataset(n_subj=n_subj, seed=321)
    params = _make_params(theta=[0.1, 20.0, 0.25, 0.1], omega_diag=[0.0], sigma_diag=sigma_diag)
    compiler = NMTRANCompiler()
    pk_callable = compiler.compile_pk("K = THETA(1)*EXP(ETA(1))\nV = THETA(2)")
    pop_model = PopulationModel(
        dataset=ds,
        pk_subroutine=ADVAN1(),
        params=params,
        pk_callable=pk_callable,
        error_callable=error_callable,
        trans=1,
        advan=1,
    )
    result = _make_estimation_result(
        theta=[0.1, 20.0, 0.25, 0.1],
        omega_diag=[0.0],
        sigma_diag=sigma_diag,
        n_subj=n_subj,
    )
    return pop_model, result


@pytest.fixture
def simple_setup():
    """
    Return (population_model, estimation_result) for a 1-cmt IV ADVAN1 model.
    """
    ds = _make_simple_dataset(n_subj=4)
    params = _make_params(
        theta=[0.1, 20.0],  # K, V
        omega_diag=[0.04],  # 1 ETA on K
        sigma_diag=[0.01],  # proportional error variance
    )
    compiler = NMTRANCompiler()
    pk_callable = compiler.compile_pk("K = THETA(1)*EXP(ETA(1))\nV = THETA(2)")
    error_callable = compiler.compile_error("Y = F*(1 + EPS(1))")

    pop_model = PopulationModel(
        dataset=ds,
        pk_subroutine=ADVAN1(),
        params=params,
        pk_callable=pk_callable,
        error_callable=error_callable,
        trans=1,  # TRANS1: identity (pass K, V through)
        advan=1,
    )

    result = _make_estimation_result(
        theta=[0.1, 20.0],
        omega_diag=[0.04],
        sigma_diag=[0.01],
        n_subj=4,
    )

    return pop_model, result


# ── _draw_mvn ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDrawMVN:
    """Tests for the _draw_mvn helper."""

    def test_zero_dim(self):
        """Zero-dimension draw should return empty array."""
        rng = np.random.default_rng(42)
        sample = _draw_mvn(rng, np.array([]).reshape(0, 0), n=0)
        assert sample.shape == (0,)

    def test_1d_normal(self):
        """1-D draw should be a scalar-array."""
        rng = np.random.default_rng(42)
        sample = _draw_mvn(rng, np.array([[0.1]]), n=1)
        assert sample.shape == (1,)
        assert np.isfinite(sample[0])

    def test_2d_mvn(self):
        """2-D draw should return 2-element vector."""
        rng = np.random.default_rng(42)
        cov = np.array([[0.1, 0.0], [0.0, 0.2]])
        sample = _draw_mvn(rng, cov, n=2)
        assert sample.shape == (2,)
        assert np.all(np.isfinite(sample))

    def test_reproducibility(self):
        """Same seed should produce same draw."""
        cov = np.array([[0.1]])
        s1 = _draw_mvn(np.random.default_rng(99), cov, 1)
        s2 = _draw_mvn(np.random.default_rng(99), cov, 1)
        assert s1[0] == s2[0]


# ── SimulationEngine ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSimulationEngine:
    """Tests for SimulationEngine.simulate()."""

    def test_returns_simulation_result(self, simple_setup):
        """simulate() should return a SimulationResult."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate(n_replicates=1)
        assert isinstance(sim, SimulationResult)

    def test_result_has_required_columns(self, simple_setup):
        """SimulationResult.simulated_df must have ID, TIME, DV, IPRED, PRED, REP."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate(n_replicates=1)
        required = {"ID", "TIME", "DV", "IPRED", "PRED", "REP"}
        assert required.issubset(set(sim.simulated_df.columns)), (
            f"Missing columns: {required - set(sim.simulated_df.columns)}"
        )

    def test_rep0_is_observed(self, simple_setup):
        """REP=0 rows should come from the observed dataset."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate(n_replicates=1)
        rep0 = sim.simulated_df[sim.simulated_df["REP"] == 0]
        assert len(rep0) > 0, "REP=0 (observed) rows should exist"

    def test_rep_gt0_is_simulated(self, simple_setup):
        """REP>0 rows should exist for simulated replicates."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate(n_replicates=2)
        reps = sim.simulated_df["REP"].unique()
        assert 1 in reps, "REP=1 should exist"
        assert 2 in reps, "REP=2 should exist"

    def test_n_replicates_correct(self, simple_setup):
        """n_replicates attribute should match request."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate(n_replicates=5)
        assert sim.n_replicates == 5
        max_rep = sim.simulated_df["REP"].max()
        assert max_rep == 5, f"Expected max REP=5, got {max_rep}"

    def test_simulated_dv_positive(self, simple_setup):
        """Simulated DV for non-missing obs should be non-negative."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate(n_replicates=3)
        rep_gt0 = sim.simulated_df[sim.simulated_df["REP"] > 0]
        obs_rows = rep_gt0[rep_gt0.get("MDV", pd.Series(0, index=rep_gt0.index)) == 0]
        assert np.all(obs_rows["DV"] >= 0), "Simulated DV should be non-negative"

    def test_seed_reproducibility(self, simple_setup):
        """Same seed should produce identical simulated datasets."""
        pop_model, result = simple_setup
        engine1 = SimulationEngine(pop_model, result, seed=42)
        engine2 = SimulationEngine(pop_model, result, seed=42)
        sim1 = engine1.simulate(n_replicates=2)
        sim2 = engine2.simulate(n_replicates=2)
        rep1_df1 = sim1.simulated_df[sim1.simulated_df["REP"] == 1]["DV"].values
        rep1_df2 = sim2.simulated_df[sim2.simulated_df["REP"] == 1]["DV"].values
        np.testing.assert_array_equal(rep1_df1, rep1_df2)

    def test_different_seeds_different_results(self, simple_setup):
        """Different seeds should produce different simulated values."""
        pop_model, result = simple_setup
        engine1 = SimulationEngine(pop_model, result, seed=1)
        engine2 = SimulationEngine(pop_model, result, seed=999)
        sim1 = engine1.simulate(n_replicates=1)
        sim2 = engine2.simulate(n_replicates=1)
        rep1_dv1 = sim1.simulated_df[sim1.simulated_df["REP"] == 1]["DV"].values
        rep1_dv2 = sim2.simulated_df[sim2.simulated_df["REP"] == 1]["DV"].values
        assert not np.allclose(rep1_dv1, rep1_dv2), (
            "Different seeds should produce different simulated DV"
        )

    def test_seed_stored(self, simple_setup):
        """SimulationResult should record the seed used."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=123)
        sim = engine.simulate(n_replicates=1)
        assert sim.seed == 123

    def test_ipred_plausible_magnitude(self, simple_setup):
        """IPRED values should be in a plausible range for 1-cmt IV with DOSE=100, V=20."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate(n_replicates=1)
        ipred_vals = sim.simulated_df["IPRED"].dropna()
        # Maximum possible concentration ≈ 100/20 = 5 at t=0+
        # Should be within [0, 10] for our obs times
        assert ipred_vals.max() < 100.0, f"IPRED suspiciously large: {ipred_vals.max()}"
        assert ipred_vals.min() >= 0.0, f"Negative IPRED: {ipred_vals.min()}"

    def test_eta_columns_present(self, simple_setup):
        """ETA columns should be present in the simulated DataFrame."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate(n_replicates=1)
        assert "ETA1" in sim.simulated_df.columns, "ETA1 column should be present"

    def test_zero_noise_simulation_is_exactly_deterministic(self):
        pop_model, result = _make_exact_advan1_setup(n_subj=2)

        sim = SimulationEngine(pop_model, result, seed=42).simulate(n_replicates=2)
        rep1 = sim.simulated_df[sim.simulated_df["REP"] == 1].reset_index(drop=True)
        rep2 = sim.simulated_df[sim.simulated_df["REP"] == 2].reset_index(drop=True)

        np.testing.assert_allclose(rep1["DV"].values, rep1["IPRED"].values, atol=1e-12)
        np.testing.assert_allclose(rep1["DV"].values, rep1["PRED"].values, atol=1e-12)
        np.testing.assert_allclose(rep1["DV"].values, rep2["DV"].values, atol=1e-12)

    def test_single_dose_zero_noise_matches_advan1_closed_form(self):
        pop_model, result = _make_exact_advan1_setup(n_subj=2)

        sim = SimulationEngine(pop_model, result, seed=42).simulate(n_replicates=1)
        rep1 = sim.simulated_df[sim.simulated_df["REP"] == 1]
        obs_times = np.array([1.0, 4.0, 8.0, 12.0, 24.0], dtype=float)
        expected = _advan1_superposition(obs_times, [(0.0, 100.0)], k=0.1, v=20.0)

        for _, grp in rep1.groupby("ID", sort=True):
            grp = grp.sort_values("TIME")
            np.testing.assert_allclose(grp["TIME"].to_numpy(dtype=float), obs_times, atol=1e-12)
            np.testing.assert_allclose(grp["IPRED"].to_numpy(dtype=float), expected, atol=1e-12)
            np.testing.assert_allclose(grp["PRED"].to_numpy(dtype=float), expected, atol=1e-12)
            np.testing.assert_allclose(grp["DV"].to_numpy(dtype=float), expected, atol=1e-12)
            np.testing.assert_allclose(grp["ETA1"].to_numpy(dtype=float), 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        ("compiled_code", "python_error_callable", "sigma_diag"),
        [
            (
                "Y = F + EPS(1)",
                lambda theta, eta, eps, f, ipred=None, dv=None, t=0.0: {
                    "Y": float(f) + float(eps[0])
                },
                [0.04],
            ),
            (
                "W = THETA(3)*F\nY = F + W*EPS(1)",
                lambda theta, eta, eps, f, ipred=None, dv=None, t=0.0: {
                    "Y": float(f) + float(theta[2]) * float(f) * float(eps[0])
                },
                [0.04],
            ),
            (
                "W = SQRT(THETA(3)**2 + (F*THETA(4))**2)\nY = F + W*EPS(1)\nIRES = DV - F\nIWRES = IRES/W",
                lambda theta, eta, eps, f, ipred=None, dv=None, t=0.0: {
                    "Y": float(f)
                    + np.sqrt(float(theta[2]) ** 2 + (float(f) * float(theta[3])) ** 2)
                    * float(eps[0])
                },
                [0.04],
            ),
            (
                "Y = F + EPS(1) + F*EPS(2)",
                lambda theta, eta, eps, f, ipred=None, dv=None, t=0.0: {
                    "Y": float(f) + float(eps[0]) + float(f) * float(eps[1])
                },
                [0.04, 0.09],
            ),
        ],
    )
    def test_common_error_fast_path_matches_generic_callable(
        self,
        compiled_code,
        python_error_callable,
        sigma_diag,
    ):
        compiler = NMTRANCompiler()
        fast_setup = _make_error_model_setup(
            compiler.compile_error(compiled_code),
            sigma_diag=sigma_diag,
        )
        slow_setup = _make_error_model_setup(
            python_error_callable,
            sigma_diag=sigma_diag,
        )

        fast_df = SimulationEngine(*fast_setup, seed=42).simulate(n_replicates=2).simulated_df
        slow_df = SimulationEngine(*slow_setup, seed=42).simulate(n_replicates=2).simulated_df

        np.testing.assert_allclose(
            fast_df[["DV", "IPRED", "PRED", "ETA1"]].to_numpy(dtype=float),
            slow_df[["DV", "IPRED", "PRED", "ETA1"]].to_numpy(dtype=float),
            atol=1e-12,
        )

    def test_default_advan2_batch_fast_path_matches_per_subject_fallback(self, monkeypatch):
        pop_model, result = _make_default_advan2_setup(n_subj=6)

        fast_engine = SimulationEngine(pop_model, result, seed=42)
        slow_engine = SimulationEngine(pop_model, result, seed=42)
        monkeypatch.setattr(
            slow_engine, "_prepare_default_advan2_batch_plan", lambda *_args, **_kwargs: None
        )

        fast_df = fast_engine.simulate(n_replicates=3).simulated_df
        slow_df = slow_engine.simulate(n_replicates=3).simulated_df

        np.testing.assert_allclose(
            fast_df[["DV", "IPRED", "PRED", "ETA1", "ETA2", "ETA3"]].to_numpy(dtype=float),
            slow_df[["DV", "IPRED", "PRED", "ETA1", "ETA2", "ETA3"]].to_numpy(dtype=float),
            atol=1e-12,
        )


# ── VPCEngine ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestVPCEngine:
    """Tests for VPCEngine.compute()."""

    def test_returns_vpc_result(self, simple_setup):
        """compute() should return a VPCResult."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        vpc = VPCEngine(engine)
        vpc_result = vpc.compute(n_replicates=10, n_bins=4)
        assert isinstance(vpc_result, VPCResult)

    def test_vpc_result_columns(self, simple_setup):
        """obs_percentiles and sim_percentiles should have expected columns."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        vpc = VPCEngine(engine)
        vpc_result = vpc.compute(n_replicates=10, n_bins=4)

        obs_cols = set(vpc_result.obs_percentiles.columns)
        assert "bin_mid" in obs_cols
        assert "p5" in obs_cols
        assert "p50" in obs_cols
        assert "p95" in obs_cols

        sim_cols = set(vpc_result.sim_percentiles.columns)
        assert "bin_mid" in sim_cols
        assert "p50_mid" in sim_cols
        assert "p5_lo" in sim_cols

    def test_vpc_n_replicates(self, simple_setup):
        """VPCResult.n_replicates should match the request."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        vpc = VPCEngine(engine)
        vpc_result = vpc.compute(n_replicates=20, n_bins=3)
        assert vpc_result.n_replicates == 20

    def test_vpc_raises_invalid_quantiles(self, simple_setup):
        """VPCEngine should raise ValueError for non-3-tuple quantiles."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        vpc = VPCEngine(engine)
        with pytest.raises(ValueError, match="3 elements"):
            vpc.compute(n_replicates=5, quantiles=(0.1, 0.5))  # wrong length

    def test_vpc_raises_for_unsorted_or_out_of_range_quantiles(self, simple_setup):
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        vpc = VPCEngine(engine)

        with pytest.raises(ValueError, match="strictly increasing"):
            vpc.compute(n_replicates=5, quantiles=(0.5, 0.1, 0.9))

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            vpc.compute(n_replicates=5, quantiles=(-0.1, 0.5, 0.9))

    def test_vpc_uses_requested_quantile_labels_for_custom_quantiles(self):
        rows = [
            {"ID": 1, "TIME": 1.0, "DV": 2.0, "PRED": 2.0, "IPRED": 2.0, "MDV": 0, "REP": 0},
            {"ID": 2, "TIME": 1.0, "DV": 4.0, "PRED": 4.0, "IPRED": 4.0, "MDV": 0, "REP": 0},
            {"ID": 1, "TIME": 2.0, "DV": 6.0, "PRED": 6.0, "IPRED": 6.0, "MDV": 0, "REP": 0},
            {"ID": 2, "TIME": 2.0, "DV": 8.0, "PRED": 8.0, "IPRED": 8.0, "MDV": 0, "REP": 0},
        ]
        for rep in (1, 2):
            rows.extend(
                [
                    {
                        "ID": 1,
                        "TIME": 1.0,
                        "DV": 2.0 + rep,
                        "PRED": 2.0,
                        "IPRED": 2.0,
                        "MDV": 0,
                        "REP": rep,
                    },
                    {
                        "ID": 2,
                        "TIME": 1.0,
                        "DV": 4.0 + rep,
                        "PRED": 4.0,
                        "IPRED": 4.0,
                        "MDV": 0,
                        "REP": rep,
                    },
                    {
                        "ID": 1,
                        "TIME": 2.0,
                        "DV": 6.0 + rep,
                        "PRED": 6.0,
                        "IPRED": 6.0,
                        "MDV": 0,
                        "REP": rep,
                    },
                    {
                        "ID": 2,
                        "TIME": 2.0,
                        "DV": 8.0 + rep,
                        "PRED": 8.0,
                        "IPRED": 8.0,
                        "MDV": 0,
                        "REP": rep,
                    },
                ]
            )

        class _FakeSimulationEngine:
            def __init__(self, df: pd.DataFrame) -> None:
                self.df = df

            def simulate(self, n_replicates: int = 1) -> SimulationResult:
                return SimulationResult(
                    simulated_df=self.df.copy(), seed=0, n_replicates=n_replicates
                )

        vpc_result = VPCEngine(_FakeSimulationEngine(pd.DataFrame(rows))).compute(
            n_replicates=2,
            n_bins=2,
            quantiles=(0.1, 0.5, 0.9),
        )

        assert vpc_result.quantiles == pytest.approx((0.1, 0.5, 0.9))
        assert {"p10", "p50", "p90"}.issubset(vpc_result.obs_percentiles.columns)
        assert {"p10_lo", "p50_mid", "p90_hi"}.issubset(vpc_result.sim_percentiles.columns)

    def test_identical_replicates_collapse_percentile_bands_exactly(self):
        rows = [
            {"ID": 1, "TIME": 1.0, "DV": 2.0, "PRED": 2.0, "IPRED": 2.0, "MDV": 0, "REP": 0},
            {"ID": 2, "TIME": 1.0, "DV": 2.0, "PRED": 2.0, "IPRED": 2.0, "MDV": 0, "REP": 0},
            {"ID": 1, "TIME": 2.0, "DV": 5.0, "PRED": 5.0, "IPRED": 5.0, "MDV": 0, "REP": 0},
            {"ID": 2, "TIME": 2.0, "DV": 5.0, "PRED": 5.0, "IPRED": 5.0, "MDV": 0, "REP": 0},
        ]
        for rep in (1, 2, 3):
            rows.extend(
                [
                    {
                        "ID": 1,
                        "TIME": 1.0,
                        "DV": 2.0,
                        "PRED": 2.0,
                        "IPRED": 2.0,
                        "MDV": 0,
                        "REP": rep,
                    },
                    {
                        "ID": 2,
                        "TIME": 1.0,
                        "DV": 2.0,
                        "PRED": 2.0,
                        "IPRED": 2.0,
                        "MDV": 0,
                        "REP": rep,
                    },
                    {
                        "ID": 1,
                        "TIME": 2.0,
                        "DV": 5.0,
                        "PRED": 5.0,
                        "IPRED": 5.0,
                        "MDV": 0,
                        "REP": rep,
                    },
                    {
                        "ID": 2,
                        "TIME": 2.0,
                        "DV": 5.0,
                        "PRED": 5.0,
                        "IPRED": 5.0,
                        "MDV": 0,
                        "REP": rep,
                    },
                ]
            )

        class _FakeSimulationEngine:
            def __init__(self, df: pd.DataFrame) -> None:
                self.df = df

            def simulate(self, n_replicates: int = 1) -> SimulationResult:
                return SimulationResult(
                    simulated_df=self.df.copy(), seed=0, n_replicates=n_replicates
                )

        vpc_result = VPCEngine(_FakeSimulationEngine(pd.DataFrame(rows))).compute(
            n_replicates=3,
            n_bins=2,
        )

        np.testing.assert_allclose(
            vpc_result.obs_percentiles[["p5", "p50", "p95"]].values,
            np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            vpc_result.sim_percentiles[
                [
                    "p5_lo",
                    "p5_mid",
                    "p5_hi",
                    "p50_lo",
                    "p50_mid",
                    "p50_hi",
                    "p95_lo",
                    "p95_mid",
                    "p95_hi",
                ]
            ].values,
            np.array(
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                ]
            ),
            atol=1e-12,
        )

    def test_vpc_simulated_percentile_bands_match_manual_quantiles(self):
        rows = [
            {"ID": 1, "TIME": 1.0, "DV": 1.0, "PRED": 1.0, "IPRED": 1.0, "MDV": 0, "REP": 0},
            {"ID": 2, "TIME": 1.0, "DV": 3.0, "PRED": 3.0, "IPRED": 3.0, "MDV": 0, "REP": 0},
            {"ID": 1, "TIME": 2.0, "DV": 10.0, "PRED": 10.0, "IPRED": 10.0, "MDV": 0, "REP": 0},
            {"ID": 2, "TIME": 2.0, "DV": 12.0, "PRED": 12.0, "IPRED": 12.0, "MDV": 0, "REP": 0},
            {"ID": 1, "TIME": 1.0, "DV": 1.0, "PRED": 1.0, "IPRED": 1.0, "MDV": 0, "REP": 1},
            {"ID": 2, "TIME": 1.0, "DV": 3.0, "PRED": 3.0, "IPRED": 3.0, "MDV": 0, "REP": 1},
            {"ID": 1, "TIME": 2.0, "DV": 10.0, "PRED": 10.0, "IPRED": 10.0, "MDV": 0, "REP": 1},
            {"ID": 2, "TIME": 2.0, "DV": 12.0, "PRED": 12.0, "IPRED": 12.0, "MDV": 0, "REP": 1},
            {"ID": 1, "TIME": 1.0, "DV": 2.0, "PRED": 2.0, "IPRED": 2.0, "MDV": 0, "REP": 2},
            {"ID": 2, "TIME": 1.0, "DV": 4.0, "PRED": 4.0, "IPRED": 4.0, "MDV": 0, "REP": 2},
            {"ID": 1, "TIME": 2.0, "DV": 20.0, "PRED": 20.0, "IPRED": 20.0, "MDV": 0, "REP": 2},
            {"ID": 2, "TIME": 2.0, "DV": 22.0, "PRED": 22.0, "IPRED": 22.0, "MDV": 0, "REP": 2},
        ]

        class _FakeSimulationEngine:
            def __init__(self, df: pd.DataFrame) -> None:
                self.df = df

            def simulate(self, n_replicates: int = 1) -> SimulationResult:
                return SimulationResult(
                    simulated_df=self.df.copy(), seed=0, n_replicates=n_replicates
                )

        result = VPCEngine(_FakeSimulationEngine(pd.DataFrame(rows))).compute(
            n_replicates=2,
            n_bins=2,
        )

        expected = np.array(
            [
                [1.15, 1.60, 2.05, 2.05, 2.50, 2.95, 2.95, 3.40, 3.85],
                [10.60, 15.10, 19.60, 11.50, 16.00, 20.50, 12.40, 16.90, 21.40],
            ]
        )
        np.testing.assert_allclose(
            result.sim_percentiles[
                [
                    "p5_lo",
                    "p5_mid",
                    "p5_hi",
                    "p50_lo",
                    "p50_mid",
                    "p50_hi",
                    "p95_lo",
                    "p95_mid",
                    "p95_hi",
                ]
            ].values,
            expected,
            atol=1e-12,
        )

    def test_prediction_corrected_vpc_uses_stratum_specific_reference(self):
        rows = [
            {
                "ID": 1,
                "TIME": 1.0,
                "DV": 2.0,
                "PRED": 2.0,
                "IPRED": 2.0,
                "MDV": 0,
                "REP": 0,
                "DOSE": 100,
            },
            {
                "ID": 2,
                "TIME": 1.0,
                "DV": 20.0,
                "PRED": 20.0,
                "IPRED": 20.0,
                "MDV": 0,
                "REP": 0,
                "DOSE": 200,
            },
            {
                "ID": 1,
                "TIME": 2.0,
                "DV": 4.0,
                "PRED": 4.0,
                "IPRED": 4.0,
                "MDV": 0,
                "REP": 0,
                "DOSE": 100,
            },
            {
                "ID": 2,
                "TIME": 2.0,
                "DV": 40.0,
                "PRED": 40.0,
                "IPRED": 40.0,
                "MDV": 0,
                "REP": 0,
                "DOSE": 200,
            },
        ]
        for rep in (1, 2):
            rows.extend(
                [
                    {
                        "ID": 1,
                        "TIME": 1.0,
                        "DV": 2.0,
                        "PRED": 2.0,
                        "IPRED": 2.0,
                        "MDV": 0,
                        "REP": rep,
                        "DOSE": 100,
                    },
                    {
                        "ID": 2,
                        "TIME": 1.0,
                        "DV": 20.0,
                        "PRED": 20.0,
                        "IPRED": 20.0,
                        "MDV": 0,
                        "REP": rep,
                        "DOSE": 200,
                    },
                    {
                        "ID": 1,
                        "TIME": 2.0,
                        "DV": 4.0,
                        "PRED": 4.0,
                        "IPRED": 4.0,
                        "MDV": 0,
                        "REP": rep,
                        "DOSE": 100,
                    },
                    {
                        "ID": 2,
                        "TIME": 2.0,
                        "DV": 40.0,
                        "PRED": 40.0,
                        "IPRED": 40.0,
                        "MDV": 0,
                        "REP": rep,
                        "DOSE": 200,
                    },
                ]
            )

        class _FakeSimulationEngine:
            def __init__(self, df: pd.DataFrame) -> None:
                self.df = df

            def simulate(self, n_replicates: int = 1) -> SimulationResult:
                return SimulationResult(
                    simulated_df=self.df.copy(), seed=0, n_replicates=n_replicates
                )

        vpc_result = VPCEngine(_FakeSimulationEngine(pd.DataFrame(rows))).compute(
            n_replicates=2,
            n_bins=2,
            stratify_by="DOSE",
            prediction_corrected=True,
        )

        dose100_obs = vpc_result.obs_percentiles[vpc_result.obs_percentiles["DOSE"] == 100]
        dose200_obs = vpc_result.obs_percentiles[vpc_result.obs_percentiles["DOSE"] == 200]
        np.testing.assert_allclose(
            dose100_obs[["p5", "p50", "p95"]].values, [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
        )
        np.testing.assert_allclose(
            dose200_obs[["p5", "p50", "p95"]].values, [[20.0, 20.0, 20.0], [40.0, 40.0, 40.0]]
        )


# ── simulate_new_design ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSimulateNewDesign:
    """Tests for SimulationEngine.simulate_new_design()."""

    def _dosing_df(self) -> pd.DataFrame:
        """Minimal dosing DataFrame: single 100 mg IV bolus at t=0."""
        return pd.DataFrame(
            {
                "TIME": [0.0],
                "AMT": [100.0],
                "EVID": [1],
                "CMT": [1],
                "MDV": [1],
            }
        )

    def test_returns_simulation_result(self, simple_setup):
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate_new_design(
            dosing_df=self._dosing_df(),
            obs_times=np.array([1.0, 4.0, 8.0]),
            n_subjects=3,
            n_replicates=2,
        )
        assert isinstance(sim, SimulationResult)

    def test_rep_column_present(self, simple_setup):
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate_new_design(
            dosing_df=self._dosing_df(),
            obs_times=np.array([1.0, 8.0]),
            n_subjects=4,
            n_replicates=2,
        )
        assert "REP" in sim.simulated_df.columns

    def test_correct_number_of_replicates(self, simple_setup):
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        n_rep = 3
        n_subj = 5
        obs = np.array([1.0, 4.0, 8.0])
        sim = engine.simulate_new_design(
            dosing_df=self._dosing_df(),
            obs_times=obs,
            n_subjects=n_subj,
            n_replicates=n_rep,
        )
        # REP=0 (observed placeholder) + REP=1..n_rep
        reps = sim.simulated_df["REP"].unique()
        assert n_rep in reps

    def test_n_subjects_matches(self, simple_setup):
        """Each replicate should have exactly n_subjects unique IDs."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        n_subj = 6
        sim = engine.simulate_new_design(
            dosing_df=self._dosing_df(),
            obs_times=np.array([1.0, 4.0]),
            n_subjects=n_subj,
            n_replicates=1,
        )
        rep1 = sim.simulated_df[sim.simulated_df["REP"] == 1]
        assert rep1["ID"].nunique() == n_subj

    def test_obs_times_match(self, simple_setup):
        """Observation times in the result should match the requested obs_times."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        obs = np.array([1.0, 4.0, 8.0, 24.0])
        sim = engine.simulate_new_design(
            dosing_df=self._dosing_df(),
            obs_times=obs,
            n_subjects=2,
            n_replicates=1,
        )
        rep1 = sim.simulated_df[sim.simulated_df["REP"] == 1]
        unique_times = sorted(rep1["TIME"].unique())
        np.testing.assert_array_almost_equal(unique_times, sorted(obs))

    def test_dv_finite_and_positive(self, simple_setup):
        """Simulated DV values should be finite and positive for this model."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate_new_design(
            dosing_df=self._dosing_df(),
            obs_times=np.array([1.0, 4.0, 8.0]),
            n_subjects=5,
            n_replicates=2,
        )
        rep_rows = sim.simulated_df[sim.simulated_df["REP"] > 0]
        # MDV=0 rows are actual observations
        obs_rows = rep_rows[rep_rows["MDV"] == 0]
        assert np.all(np.isfinite(obs_rows["DV"].values))

    def test_required_columns_present(self, simple_setup):
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        sim = engine.simulate_new_design(
            dosing_df=self._dosing_df(),
            obs_times=np.array([1.0, 8.0]),
            n_subjects=3,
            n_replicates=1,
        )
        required = {"ID", "TIME", "DV", "IPRED", "PRED", "REP"}
        assert required.issubset(set(sim.simulated_df.columns))

    def test_evid_inferred_when_absent(self, simple_setup):
        """dosing_df without EVID column should be handled gracefully."""
        pop_model, result = simple_setup
        engine = SimulationEngine(pop_model, result, seed=42)
        dosing_no_evid = pd.DataFrame({"TIME": [0.0], "AMT": [100.0]})
        sim = engine.simulate_new_design(
            dosing_df=dosing_no_evid,
            obs_times=np.array([1.0, 8.0]),
            n_subjects=2,
            n_replicates=1,
        )
        assert isinstance(sim, SimulationResult)
        assert len(sim.simulated_df) > 0

    def test_multidose_zero_noise_matches_superposition_closed_form(self):
        pop_model, result = _make_exact_advan1_setup(n_subj=1)
        engine = SimulationEngine(pop_model, result, seed=42)
        dosing_df = pd.DataFrame(
            {
                "TIME": [0.0, 12.0],
                "AMT": [100.0, 50.0],
                "EVID": [1, 1],
                "CMT": [1, 1],
                "MDV": [1, 1],
            }
        )
        obs_times = np.array([1.0, 4.0, 8.0, 13.0, 16.0, 24.0], dtype=float)

        sim = engine.simulate_new_design(
            dosing_df=dosing_df,
            obs_times=obs_times,
            n_subjects=2,
            n_replicates=1,
        )

        rep0 = sim.simulated_df[sim.simulated_df["REP"] == 0].sort_values(["ID", "TIME"])
        rep1 = sim.simulated_df[sim.simulated_df["REP"] == 1].sort_values(["ID", "TIME"])
        expected = _advan1_superposition(
            obs_times,
            [(0.0, 100.0), (12.0, 50.0)],
            k=0.1,
            v=20.0,
        )

        assert rep0["DV"].isna().all()
        for _, grp in rep1.groupby("ID", sort=True):
            np.testing.assert_allclose(grp["TIME"].to_numpy(dtype=float), obs_times, atol=1e-12)
            np.testing.assert_allclose(grp["IPRED"].to_numpy(dtype=float), expected, atol=1e-12)
            np.testing.assert_allclose(grp["PRED"].to_numpy(dtype=float), expected, atol=1e-12)
            np.testing.assert_allclose(grp["DV"].to_numpy(dtype=float), expected, atol=1e-12)
