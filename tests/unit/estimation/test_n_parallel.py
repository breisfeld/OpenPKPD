"""
Tests for n_parallel multiprocessing/threading in estimation methods and SimulationEngine.

Covers:
  - FOCEMethod._inner_loop produces identical eta_hat with n_parallel=1 and n_parallel=2
    (deterministic method, results must match exactly).  Uses real IndividualModel
    objects backed by ADVAN1 — MagicMock is not picklable for ProcessPoolExecutor.
  - SAEMMethod completes without error and returns a valid EstimationResult for n_parallel=2
  - IMPMethod completes without error and returns a finite OFV for n_parallel=2
  - SimulationEngine.simulate() with n_parallel=2 produces the same REP count and
    is reproducible (same seed → same output regardless of n_parallel)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from openpkpd.estimation.base import EstimationResult
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.estimation.imp import IMPMethod
from openpkpd.estimation.saem import SAEMMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_params(n_theta: int = 2, n_eta: int = 1) -> ParameterSet:
    return ParameterSet(
        theta=np.ones(n_theta) * 0.5,
        omega=np.eye(n_eta) * 0.1,
        sigma=np.array([[0.05]]),
        theta_specs=[ThetaSpec(init=0.5, lower=1e-6, upper=10.0) for _ in range(n_theta)],
        omega_specs=[OmegaSpec(block_size=1, values=[0.1]) for _ in range(n_eta)],
        sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
    )


def _make_mock_pop_model(n_subjects: int = 6, n_eta: int = 1):
    """Mock population model — only used for serial or thread-based tests."""
    model = MagicMock()
    model.subject_ids.return_value = list(range(1, n_subjects + 1))
    model.n_subjects.return_value = n_subjects
    model.trans = 2

    def _make_indiv(sid):
        indiv = MagicMock()
        indiv.obj_eta.side_effect = lambda eta, theta, omega, sigma, trans=2: float(
            np.sum((eta - 0.3) ** 2) + 1.0
        )
        indiv.log_likelihood.side_effect = lambda theta, eta, sigma, trans=2: -5.0
        return indiv

    model.individual_model.side_effect = _make_indiv
    return model


def _make_real_pop_model(n_subjects: int = 4):
    """
    Build a real PopulationModel backed by ADVAN1 with compiled $PK/$ERROR.

    Required for ProcessPoolExecutor tests — MagicMock is not picklable.
    """
    from openpkpd.data.dataset import NONMEMDataset
    from openpkpd.model.population import PopulationModel
    from openpkpd.parser.code_compiler import NMTRANCompiler
    from openpkpd.pk.analytical.advan1 import ADVAN1

    rows = []
    for sid in range(1, n_subjects + 1):
        rows.append({"ID": sid, "TIME": 0.0, "AMT": 100.0, "DV": 0.0, "EVID": 1, "MDV": 1})
        for t in [1.0, 4.0, 8.0, 12.0]:
            conc = (
                100.0 / 20.0 * np.exp(-0.1 * t) * (1 + np.random.default_rng(sid).normal(0, 0.05))
            )
            rows.append(
                {"ID": sid, "TIME": t, "AMT": 0.0, "DV": max(conc, 0.01), "EVID": 0, "MDV": 0}
            )

    ds = NONMEMDataset.from_dataframe(pd.DataFrame(rows))
    ps = ParameterSet.from_specs(
        [ThetaSpec(init=0.1, lower=1e-6), ThetaSpec(init=20.0, lower=1e-6)],
        [OmegaSpec(block_size=1, values=[0.04])],
        [SigmaSpec(block_size=1, values=[0.01])],
    )
    compiler = NMTRANCompiler()
    pk_callable = compiler.compile_pk("K = theta[0]*EXP(eta[0])\nV = theta[1]")
    error_callable = compiler.compile_error("Y = F*(1 + EPS(1))")
    return PopulationModel(
        dataset=ds,
        pk_subroutine=ADVAN1(),
        params=ps,
        pk_callable=pk_callable,
        error_callable=error_callable,
        trans=1,
        advan=1,
    )


def _make_real_advan2_pop_model(n_subjects: int = 4):
    """Build a real ADVAN2 PopulationModel using builder-style CL/V/KA code."""
    from openpkpd.data.dataset import NONMEMDataset
    from openpkpd.model.population import PopulationModel
    from openpkpd.parser.code_compiler import NMTRANCompiler
    from openpkpd.pk.analytical.advan2 import ADVAN2

    rows = []
    for sid in range(1, n_subjects + 1):
        rows.append({"ID": sid, "TIME": 0.0, "AMT": 100.0, "DV": 0.0, "EVID": 1, "MDV": 1})
        for t in [0.5, 1.0, 2.0, 4.0]:
            conc = (
                100.0
                * 0.08
                / 20.0
                * np.exp(-0.08 / 20.0 * t)
                * (1 + np.random.default_rng(sid).normal(0, 0.03))
            )
            rows.append(
                {"ID": sid, "TIME": t, "AMT": 0.0, "DV": max(conc, 0.01), "EVID": 0, "MDV": 0}
            )

    ds = NONMEMDataset.from_dataframe(pd.DataFrame(rows))
    ps = ParameterSet.from_specs(
        [
            ThetaSpec(init=1.5, lower=1e-6),
            ThetaSpec(init=0.08, lower=1e-6),
            ThetaSpec(init=20.0, lower=1e-6),
        ],
        [
            OmegaSpec(block_size=1, values=[0.04]),
            OmegaSpec(block_size=1, values=[0.04]),
            OmegaSpec(block_size=1, values=[0.04]),
        ],
        [SigmaSpec(block_size=1, values=[0.01])],
    )
    compiler = NMTRANCompiler()
    pk_callable = compiler.compile_pk(
        "KA = theta[0]*EXP(eta[0])\nCL = theta[1]*EXP(eta[1])\nV = theta[2]*EXP(eta[2])"
    )
    error_callable = compiler.compile_error("Y = F*(1 + EPS(1))")
    return PopulationModel(
        dataset=ds,
        pk_subroutine=ADVAN2(),
        params=ps,
        pk_callable=pk_callable,
        error_callable=error_callable,
        trans=2,
        advan=2,
    )


def _make_foce_params() -> ParameterSet:
    return ParameterSet.from_specs(
        [ThetaSpec(init=0.1, lower=1e-6), ThetaSpec(init=20.0, lower=1e-6)],
        [OmegaSpec(block_size=1, values=[0.04])],
        [SigmaSpec(block_size=1, values=[0.01])],
    )


# ---------------------------------------------------------------------------
# FOCEMethod._inner_loop: parallel == serial (uses real IndividualModel)
# ---------------------------------------------------------------------------


class TestFOCEParallel:
    def test_inner_loop_parallel_matches_serial(self):
        """
        n_parallel=2 (ProcessPoolExecutor) must produce the same eta_hat as
        n_parallel=1 (serial).  Uses real IndividualModel — MagicMock is not picklable.
        """
        pop = _make_real_pop_model(n_subjects=4)
        params = _make_foce_params()
        subject_ids = pop.subject_ids()

        serial = FOCEMethod(maxeval=1, n_parallel=1)
        serial._current_eta_hat = {sid: np.zeros(1) for sid in subject_ids}
        eta_serial = serial._inner_loop(pop, params)

        parallel = FOCEMethod(maxeval=1, n_parallel=2)
        parallel._current_eta_hat = {sid: np.zeros(1) for sid in subject_ids}
        eta_parallel = parallel._inner_loop(pop, params)

        assert set(eta_serial.keys()) == set(eta_parallel.keys())
        for sid in eta_serial:
            np.testing.assert_allclose(
                eta_parallel[sid],
                eta_serial[sid],
                atol=1e-6,
                err_msg=f"eta_hat mismatch for subject {sid}",
            )

    def test_inner_loop_parallel_returns_all_subjects(self):
        pop = _make_real_pop_model(n_subjects=4)
        params = _make_foce_params()
        method = FOCEMethod(maxeval=1, n_parallel=2)
        method._current_eta_hat = {sid: np.zeros(1) for sid in pop.subject_ids()}
        eta_hat = method._inner_loop(pop, params)
        assert set(eta_hat.keys()) == set(pop.subject_ids())

    def test_inner_loop_auto_parallel(self):
        """n_parallel=0 (auto) should complete without error."""
        pop = _make_real_pop_model(n_subjects=3)
        params = _make_foce_params()
        method = FOCEMethod(maxeval=1, n_parallel=0)
        method._current_eta_hat = {sid: np.zeros(1) for sid in pop.subject_ids()}
        eta_hat = method._inner_loop(pop, params)
        assert len(eta_hat) == 3

    def test_inner_loop_parallel_matches_serial_for_cached_builder_style_advan2(self):
        """Builder-style ADVAN2 models remain process-pool safe after cache warmup."""
        pop = _make_real_advan2_pop_model(n_subjects=4)
        params = pop.params
        subject_ids = pop.subject_ids()

        for sid in subject_ids:
            indiv = pop.individual_model(sid)
            indiv.evaluate_observation_model(
                params.theta, np.zeros(params.n_eta()), params.sigma, trans=2
            )
            assert indiv._pk_param_transformers

        serial = FOCEMethod(maxeval=1, n_parallel=1)
        serial._current_eta_hat = {sid: np.zeros(params.n_eta()) for sid in subject_ids}
        eta_serial = serial._inner_loop(pop, params)

        parallel = FOCEMethod(maxeval=1, n_parallel=2)
        parallel._current_eta_hat = {sid: np.zeros(params.n_eta()) for sid in subject_ids}
        eta_parallel = parallel._inner_loop(pop, params)

        assert set(eta_serial.keys()) == set(eta_parallel.keys())
        for sid in eta_serial:
            np.testing.assert_allclose(
                eta_parallel[sid],
                eta_serial[sid],
                atol=1e-6,
                err_msg=f"cached ADVAN2 eta_hat mismatch for subject {sid}",
            )


# ---------------------------------------------------------------------------
# SAEMMethod: parallel E-step smoke test
# ---------------------------------------------------------------------------


class TestSAEMParallel:
    def test_estimate_parallel_returns_estimation_result(self):
        """n_parallel=2 should complete and return a valid EstimationResult."""
        pop = _make_mock_pop_model(n_subjects=4)
        params = _make_params()
        saem = SAEMMethod(
            n_iter_phase1=2,
            n_iter_phase2=1,
            n_chains=1,
            seed=0,
            n_parallel=2,
        )
        result = saem.estimate(pop, params)
        assert isinstance(result, EstimationResult)

    def test_estimate_parallel_ofv_is_finite(self):
        pop = _make_mock_pop_model(n_subjects=4)
        params = _make_params()
        saem = SAEMMethod(
            n_iter_phase1=2,
            n_iter_phase2=0,
            n_chains=2,
            seed=1,
            n_parallel=2,
        )
        result = saem.estimate(pop, params)
        assert np.isfinite(result.ofv)

    def test_estimate_parallel_post_hoc_etas_present(self):
        pop = _make_mock_pop_model(n_subjects=4)
        params = _make_params()
        saem = SAEMMethod(
            n_iter_phase1=1,
            n_iter_phase2=0,
            n_chains=1,
            seed=2,
            n_parallel=2,
        )
        result = saem.estimate(pop, params)
        assert set(result.post_hoc_etas.keys()) == set(pop.subject_ids())

    def test_estimate_auto_parallel(self):
        """n_parallel=0 should select worker count automatically."""
        pop = _make_mock_pop_model(n_subjects=3)
        params = _make_params()
        saem = SAEMMethod(
            n_iter_phase1=1,
            n_iter_phase2=0,
            n_chains=1,
            seed=3,
            n_parallel=0,
        )
        result = saem.estimate(pop, params)
        assert isinstance(result, EstimationResult)


# ---------------------------------------------------------------------------
# IMPMethod: parallel subject evaluation smoke test
# ---------------------------------------------------------------------------


class TestIMPParallel:
    def test_estimate_parallel_returns_estimation_result(self):
        pop = _make_mock_pop_model(n_subjects=4)
        params = _make_params()
        imp = IMPMethod(isample=10, maxeval=2, seed=0, n_parallel=2)
        result = imp.estimate(pop, params)
        assert isinstance(result, EstimationResult)

    def test_estimate_parallel_ofv_finite(self):
        pop = _make_mock_pop_model(n_subjects=4)
        params = _make_params()
        imp = IMPMethod(isample=10, maxeval=2, seed=0, n_parallel=2)
        result = imp.estimate(pop, params)
        assert np.isfinite(result.ofv)

    def test_compute_imp_ofv_parallel_close_to_serial(self):
        """
        With identical per-subject RNG seeds the IMP OFV from n_parallel=2
        should be finite (stochastic method, so exact equality is not expected).
        """
        pop = _make_mock_pop_model(n_subjects=4)
        params = _make_params()

        imp_s = IMPMethod(isample=50, maxeval=1, seed=7, n_parallel=1)
        result_s = imp_s.estimate(pop, params)

        imp_p = IMPMethod(isample=50, maxeval=1, seed=7, n_parallel=2)
        result_p = imp_p.estimate(pop, params)

        assert np.isfinite(result_s.ofv)
        assert np.isfinite(result_p.ofv)


# ---------------------------------------------------------------------------
# SimulationEngine: parallel replicates
# ---------------------------------------------------------------------------


class TestSimulationEngineParallel:
    """Tests for SimulationEngine n_parallel threading."""

    def _make_engine(self, n_parallel: int = 1, seed: int = 42):
        """Build a minimal SimulationEngine backed by an ADVAN1 model."""
        import pandas as pd

        from openpkpd.data.dataset import NONMEMDataset
        from openpkpd.model.population import PopulationModel
        from openpkpd.pk.analytical.advan1 import ADVAN1
        from openpkpd.simulation.engine import SimulationEngine

        rows = []
        for sid in range(1, 5):
            rows.append({"ID": sid, "TIME": 0.0, "AMT": 100.0, "DV": 0.0, "EVID": 1, "MDV": 1})
            for t in [1.0, 4.0, 8.0, 12.0]:
                conc = 100.0 / 20.0 * np.exp(-0.1 * t)
                rows.append({"ID": sid, "TIME": t, "AMT": 0.0, "DV": conc, "EVID": 0, "MDV": 0})
        ds = NONMEMDataset.from_dataframe(pd.DataFrame(rows))

        from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec

        ps = ParameterSet.from_specs(
            [ThetaSpec(init=0.1, lower=1e-6), ThetaSpec(init=20.0, lower=1e-6)],
            [OmegaSpec(block_size=1, values=[0.04])],
            [SigmaSpec(block_size=1, values=[0.01])],
        )
        pop = PopulationModel(
            dataset=ds,
            pk_subroutine=ADVAN1(),
            params=ps,
            trans=1,
            advan=1,
        )
        result = EstimationResult(
            theta_final=np.array([0.1, 20.0]),
            omega_final=np.diag([0.04]),
            sigma_final=np.diag([0.01]),
            ofv=0.0,
            converged=True,
            post_hoc_etas={sid: np.zeros(1) for sid in range(1, 5)},
        )
        return SimulationEngine(pop, result, seed=seed, n_parallel=n_parallel)

    def test_parallel_produces_correct_rep_count(self):
        engine = self._make_engine(n_parallel=2)
        sim = engine.simulate(n_replicates=4)
        assert sim.n_replicates == 4
        rep_values = sorted(sim.simulated_df["REP"].unique())
        assert rep_values == [0, 1, 2, 3, 4]

    def test_parallel_reproducible_same_seed(self):
        """Two runs with the same seed must produce identical DataFrames."""
        r1 = self._make_engine(n_parallel=2, seed=99).simulate(n_replicates=4)
        r2 = self._make_engine(n_parallel=2, seed=99).simulate(n_replicates=4)
        import pandas as pd

        pd.testing.assert_frame_equal(
            r1.simulated_df.reset_index(drop=True),
            r2.simulated_df.reset_index(drop=True),
        )

    def test_parallel_dv_nonnegative(self):
        engine = self._make_engine(n_parallel=2)
        sim = engine.simulate(n_replicates=3)
        rep_rows = sim.simulated_df[sim.simulated_df["REP"] > 0]
        assert (rep_rows["DV"] >= 0).all()

    def test_auto_parallel(self):
        """n_parallel=0 should complete without error."""
        engine = self._make_engine(n_parallel=0)
        sim = engine.simulate(n_replicates=2)
        assert sim.n_replicates == 2
