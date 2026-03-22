from __future__ import annotations

import json
import os

import numpy as np
import pytest

from openpkpd.nca.nca import NCAEngine
from openpkpd.simulation.engine import SimulationEngine
from openpkpd.simulation.npde import NPDEEngine
from openpkpd.simulation.vpc import VPCEngine
from tests.regression.diagnostic_helpers import (
    build_npc_result,
    build_pop_model_and_result,
    build_sse_result,
    fraction_obs_p50_in_sim_range,
    make_mock_npde_engine,
    theophylline_nca_profile,
)

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference_runs")


def load_reference(name: str) -> dict:
    with open(os.path.join(REFERENCE_DIR, f"{name}.json")) as f:
        return json.load(f)


@pytest.mark.regression
@pytest.mark.slow
class TestVPCDiagnosticRegression:
    @pytest.fixture(scope="class")
    def vpc_result(self):
        pop_model, est_result = build_pop_model_and_result(n_subjects=24, seed=7)
        return VPCEngine(SimulationEngine(pop_model, est_result, seed=7)).compute(
            n_replicates=100,
            n_bins=8,
        )

    def test_vpc_summary_matches_reference(self, vpc_result):
        ref = load_reference("diagnostic_vpc")
        np.testing.assert_allclose(vpc_result.obs_percentiles["p50"], ref["obs_p50"], atol=1e-6)
        np.testing.assert_allclose(
            vpc_result.sim_percentiles["p50_mid"], ref["sim_p50_mid"], atol=1e-6
        )
        np.testing.assert_allclose(vpc_result.sim_percentiles["p5_lo"], ref["sim_p5_lo"], atol=1e-6)
        np.testing.assert_allclose(
            vpc_result.sim_percentiles["p95_hi"], ref["sim_p95_hi"], atol=1e-6
        )

    def test_vpc_coverage_matches_reference(self, vpc_result):
        ref = load_reference("diagnostic_vpc")
        assert fraction_obs_p50_in_sim_range(vpc_result) == pytest.approx(
            ref["coverage_fraction"], abs=1e-6
        )


@pytest.mark.regression
@pytest.mark.slow
class TestNPCDiagnosticRegression:
    @pytest.fixture(scope="class")
    def npc_result(self):
        return build_npc_result(n_subjects=24, seed=7, n_replicates=100, n_bins=8)

    def test_npc_summary_matches_reference(self, npc_result):
        ref = load_reference("diagnostic_npc")
        assert npc_result.obs_below_lower == pytest.approx(ref["obs_below_lower"], abs=1e-6)
        assert npc_result.obs_within == pytest.approx(ref["obs_within"], abs=1e-6)
        assert npc_result.obs_above_upper == pytest.approx(ref["obs_above_upper"], abs=1e-6)
        assert npc_result.expected_within == pytest.approx(ref["expected_within"], abs=1e-12)
        assert npc_result.n_observations == ref["n_observations"]

    def test_npc_binned_summary_matches_reference(self, npc_result):
        ref = load_reference("diagnostic_npc")
        assert npc_result.binned is not None
        np.testing.assert_allclose(npc_result.binned["t_mid"], ref["binned_t_mid"], atol=1e-6)
        np.testing.assert_allclose(
            npc_result.binned["obs_within"], ref["binned_obs_within"], atol=1e-6
        )
        np.testing.assert_allclose(
            npc_result.binned["obs_below_lower"],
            ref["binned_obs_below_lower"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            npc_result.binned["obs_above_upper"],
            ref["binned_obs_above_upper"],
            atol=1e-6,
        )


@pytest.mark.regression
class TestNPDEDiagnosticRegression:
    @pytest.fixture(scope="class")
    def npde_result(self):
        engine = make_mock_npde_engine(
            n_subjects=20, n_obs=8, n_replicates=200, noise_sd=0.5, seed=7
        )
        return NPDEEngine(engine).compute(n_replicates=200, seed=7)

    def test_npde_summary_matches_reference(self, npde_result):
        ref = load_reference("diagnostic_npde")
        assert npde_result.mean_npde == pytest.approx(ref["mean_npde"], abs=5e-3)
        assert npde_result.var_npde == pytest.approx(ref["var_npde"], abs=5e-3)
        assert npde_result.sw_stat == pytest.approx(ref["sw_stat"], abs=5e-3)
        assert npde_result.sw_pvalue == pytest.approx(ref["sw_pvalue"], abs=5e-3)

    def test_npde_quantiles_match_reference(self, npde_result):
        ref = load_reference("diagnostic_npde")
        quantiles = np.quantile(
            npde_result.df["NPDE"].dropna().to_numpy(dtype=float), [0.05, 0.5, 0.95]
        )
        np.testing.assert_allclose(quantiles, ref["quantiles"], atol=5e-3)


@pytest.mark.regression
class TestNCADiagnosticRegression:
    def test_nca_summary_matches_reference(self):
        ref = load_reference("diagnostic_nca")
        times, conc, dose = theophylline_nca_profile()
        result = NCAEngine(auc_method="linear-log", min_points_lambda=4).compute_subject(
            times,
            conc,
            dose=dose,
            route="oral",
        )
        assert result.cmax == pytest.approx(ref["cmax"], abs=1e-10)
        assert result.tmax == pytest.approx(ref["tmax"], abs=1e-10)
        assert result.auc_last == pytest.approx(ref["auc_last"], abs=1e-10)
        assert result.auc_inf == pytest.approx(ref["auc_inf"], abs=1e-10)
        assert result.lambda_z == pytest.approx(ref["lambda_z"], abs=1e-10)
        assert result.t_half == pytest.approx(ref["t_half"], abs=1e-10)
        assert result.cl_f == pytest.approx(ref["cl_f"], abs=1e-10)
        assert result.vz_f == pytest.approx(ref["vz_f"], abs=1e-10)
        assert result.mrt == pytest.approx(ref["mrt"], abs=1e-10)


@pytest.mark.regression
@pytest.mark.slow
class TestSSEDiagnosticRegression:
    @pytest.fixture(scope="class")
    def sse_result(self):
        return build_sse_result(
            n_subjects=8,
            data_seed=11,
            run_seed=11,
            n_replicates=4,
            estimation_method="FO",
        )

    def test_sse_summary_matches_reference(self, sse_result):
        ref = load_reference("diagnostic_sse")
        names = ref["parameter_names"]

        assert sse_result.n_replicates == ref["n_replicates"]
        assert sse_result.parameter_names == names
        assert sse_result.convergence_rate == pytest.approx(ref["convergence_rate"], abs=1e-12)

        np.testing.assert_allclose(
            [sse_result.true_values[name] for name in names],
            [ref["true_values"][name] for name in names],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [float(sse_result.estimates[name].mean()) for name in names],
            [ref["mean_estimates"][name] for name in names],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            [sse_result.bias[name] for name in names],
            [ref["bias"][name] for name in names],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            [sse_result.rmse[name] for name in names],
            [ref["rmse"][name] for name in names],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            [sse_result.coverage_95[name] for name in names],
            [ref["coverage_95"][name] for name in names],
            atol=1e-12,
        )
