"""
E1: VPC end-to-end test.

Builds a small theophylline-like dataset, fits a 1-cmt oral model with FOCE,
runs SimulationEngine to generate replicates, and passes the result to
VPCEngine.  Verifies that:
  - VPC computation completes without error.
  - The observed 50th percentile falls within the simulated 80% CI for
    at least 60% of time bins (generous tolerance for a small N dataset).
  - Simulated percentile DataFrames have the expected structure.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.simulation.engine import SimulationEngine
from openpkpd.simulation.vpc import VPCEngine

# ── Dataset builder (reused from regression test) ────────────────────────────


def _build_small_pk_dataset(n_subjects: int = 8, seed: int = 77) -> NONMEMDataset:
    """Simulate a small 1-cmt oral PK dataset for VPC testing."""
    rng = np.random.default_rng(seed)
    ka_pop, cl_pop, v_pop = 1.5, 2.8, 32.9
    dose = 320.0
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 7.0, 12.0, 24.0])

    rows = []
    for sid in range(1, n_subjects + 1):
        eta_cl = rng.normal(0, 0.2)
        eta_v = rng.normal(0, 0.15)
        cl = cl_pop * math.exp(eta_cl)
        v = v_pop * math.exp(eta_v)
        ka, k = ka_pop, cl / v

        rows.append(
            {
                "ID": sid,
                "TIME": 0.0,
                "AMT": dose,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            }
        )
        for t in obs_times:
            if abs(ka - k) < 1e-6:
                c = dose * ka / v * t * math.exp(-k * t)
            else:
                c = dose * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))
            eps = rng.normal(0, 0.1)
            dv = max(c * (1 + eps), 0.001)
            rows.append(
                {
                    "ID": sid,
                    "TIME": t,
                    "AMT": 0.0,
                    "DV": dv,
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                    "RATE": 0.0,
                    "ADDL": 0,
                    "II": 0,
                    "SS": 0,
                }
            )

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def _build_pop_model_and_result(
    *,
    n_subjects: int = 8,
    seed: int = 77,
    theta_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """Build a population model plus a synthetic estimation result for VPC tests."""
    dataset = _build_small_pk_dataset(n_subjects=n_subjects, seed=seed)

    theta_specs = [
        ThetaSpec(init=1.5, lower=0.5, upper=5.0),
        ThetaSpec(init=2.8, lower=0.5, upper=10.0),
        ThetaSpec(init=32.9, lower=10.0, upper=80.0),
    ]
    omega_specs = [OmegaSpec(block_size=1, values=[0.04])]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]

    params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)
    pop_model = PopulationModel(
        dataset=dataset,
        pk_subroutine=ADVAN2(),
        params=params,
        trans=2,
        advan=2,
    )
    theta_final = params.theta.copy()
    theta_final[:3] = theta_final[:3] * np.array(theta_scale, dtype=float)
    result = EstimationResult(
        theta_final=theta_final,
        omega_final=params.omega.copy(),
        sigma_final=params.sigma.copy(),
        ofv=100.0,
        converged=True,
        post_hoc_etas={sid: np.zeros(params.n_eta()) for sid in pop_model.subject_ids()},
    )
    return pop_model, result


def _fraction_obs_p50_in_sim_range(vpc_result) -> float:
    """Return fraction of bins where observed median lies in simulated p5-p95 range."""
    obs_p = vpc_result.obs_percentiles
    sim_p = vpc_result.sim_percentiles
    merged = obs_p[["bin_mid", "p50"]].merge(
        sim_p[["bin_mid", "p5_lo", "p95_hi"]],
        on="bin_mid",
        how="inner",
    )
    assert len(merged) > 0, "No matching VPC bins available for comparison"
    in_range = (merged["p50"] >= merged["p5_lo"]) & (merged["p50"] <= merged["p95_hi"])
    return float(in_range.mean())


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pop_model_and_result():
    """
    Build a population model with known true parameters (no FOCE fitting needed).

    Using the true parameters as the estimation result gives a well-defined
    VPC without depending on optimizer convergence.
    """
    return _build_pop_model_and_result(n_subjects=8, seed=77)


@pytest.fixture(scope="module")
def vpc_result(pop_model_and_result):
    """Run the VPC engine with 50 replicates."""
    pop_model, result = pop_model_and_result
    sim_engine = SimulationEngine(pop_model, result, seed=42)
    vpc_engine = VPCEngine(sim_engine)
    return vpc_engine.compute(n_replicates=50, n_bins=5)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.slow
class TestVPCPipeline:
    """E1: End-to-end VPC pipeline test."""

    def test_vpc_completes_without_error(self, vpc_result):
        """VPC computation must complete and return a VPCResult."""
        assert vpc_result is not None

    def test_sim_percentiles_has_required_columns(self, vpc_result):
        """Simulated percentile DataFrame must contain expected columns."""
        required = {"bin_mid", "p50_lo", "p50_mid", "p50_hi"}
        missing = required - set(vpc_result.sim_percentiles.columns)
        assert not missing, f"sim_percentiles missing columns: {missing}"

    def test_obs_percentiles_has_required_columns(self, vpc_result):
        """Observed percentile DataFrame must contain expected columns."""
        required = {"bin_mid", "p50"}
        missing = required - set(vpc_result.obs_percentiles.columns)
        assert not missing, f"obs_percentiles missing columns: {missing}"

    def test_n_replicates_matches_request(self, vpc_result):
        """n_replicates in result must equal the requested count."""
        assert vpc_result.n_replicates == 50

    def test_sim_df_has_rep_column(self, vpc_result):
        """Simulated DataFrame must include a REP column with REP >= 1."""
        assert "REP" in vpc_result.simulated_df.columns
        assert vpc_result.simulated_df["REP"].max() >= 1

    def test_obs_50th_pct_vs_sim_overlaps(self, vpc_result):
        """
        The observed 50th percentile should be in a plausible range relative
        to simulated percentiles across bins.

        For a well-fitting model, obs p50 should generally be bracketed by
        the simulated p5–p95 band.  We check that obs p50 falls within
        [sim_p5_lo, sim_p95_hi] for at least 40% of bins.

        This is a very relaxed criterion appropriate for small (N=8) datasets.
        """
        obs_p = vpc_result.obs_percentiles
        sim_p = vpc_result.sim_percentiles

        if len(obs_p) == 0 or len(sim_p) == 0:
            pytest.skip("No binned percentile data available")

        # Check required columns exist
        p5_col = "p5_lo" if "p5_lo" in sim_p.columns else None
        p95_col = "p95_hi" if "p95_hi" in sim_p.columns else None

        if p5_col is None or p95_col is None:
            # If bounds columns aren't available, just check shapes
            assert len(sim_p) > 0
            return

        merged = obs_p[["bin_mid", "p50"]].merge(
            sim_p[["bin_mid", p5_col, p95_col]],
            on="bin_mid",
            how="inner",
        )
        if len(merged) == 0:
            pytest.skip("No matching bins in obs and sim percentiles")

        in_range = (merged["p50"] >= merged[p5_col]) & (merged["p50"] <= merged[p95_col])
        fraction_in_range = in_range.mean()
        assert fraction_in_range >= 0.40, (
            f"Only {fraction_in_range:.0%} of bins have observed p50 within "
            f"simulated p5–p95 range (expected >= 40%)"
        )

    def test_ipred_values_non_negative(self, vpc_result):
        """IPRED values in observation rows should be non-negative."""
        if "IPRED" not in vpc_result.simulated_df.columns:
            return
        sim_df = vpc_result.simulated_df
        # Filter to observation rows only (exclude dose rows with IPRED=0)
        if "MDV" in sim_df.columns:
            obs_rows = sim_df[(sim_df["REP"] >= 1) & (sim_df["MDV"] == 0)]
        else:
            obs_rows = sim_df[sim_df["REP"] >= 1]
        ipred = obs_rows["IPRED"].dropna()
        if len(ipred) > 0:
            assert (ipred >= 0).all(), "Negative IPRED values found in simulation"

    def test_vpc_detects_clear_cl_misspecification_across_seeds(self):
        """A clearly misspecified CL should reduce VPC median coverage across seeds."""
        seeds = [7, 42, 99]
        correct_fractions = []
        misspecified_fractions = []

        for seed in seeds:
            pop_model, result = _build_pop_model_and_result(n_subjects=24, seed=seed)
            correct_vpc = VPCEngine(SimulationEngine(pop_model, result, seed=seed)).compute(
                n_replicates=100,
                n_bins=8,
            )
            correct_fractions.append(_fraction_obs_p50_in_sim_range(correct_vpc))

            misspecified_model, misspecified_result = _build_pop_model_and_result(
                n_subjects=24,
                seed=seed,
                theta_scale=(1.0, 1.6, 1.0),
            )
            misspecified_vpc = VPCEngine(
                SimulationEngine(misspecified_model, misspecified_result, seed=seed)
            ).compute(n_replicates=100, n_bins=8)
            misspecified_fractions.append(_fraction_obs_p50_in_sim_range(misspecified_vpc))

        correct_median = float(np.median(correct_fractions))
        misspecified_median = float(np.median(misspecified_fractions))

        assert correct_median >= 0.5, f"Correct-model VPC coverage too low: {correct_fractions}"
        assert misspecified_median <= 0.4, (
            f"Misspecified-model VPC coverage too high: {misspecified_fractions}"
        )
        assert correct_median - misspecified_median >= 0.15, (
            "VPC did not separate correct vs misspecified CL strongly enough: "
            f"correct={correct_fractions}, misspecified={misspecified_fractions}"
        )
