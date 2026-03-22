from __future__ import annotations

import math

import numpy as np
import pandas as pd

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan2 import ADVAN2


def build_small_pk_dataset(n_subjects: int = 8, seed: int = 77) -> NONMEMDataset:
    rng = np.random.default_rng(seed)
    ka_pop, cl_pop, v_pop = 1.5, 2.8, 32.9
    dose = 320.0
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 7.0, 12.0, 24.0])
    rows: list[dict[str, float | int]] = []
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
            c = (
                dose * ka / v * t * math.exp(-k * t)
                if abs(ka - k) < 1e-6
                else dose * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))
            )
            dv = max(c * (1 + rng.normal(0, 0.1)), 0.001)
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


def build_pop_model_and_result(
    *,
    n_subjects: int = 8,
    seed: int = 77,
    theta_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[PopulationModel, EstimationResult]:
    dataset = build_small_pk_dataset(n_subjects=n_subjects, seed=seed)
    params = ParameterSet.from_specs(
        [
            ThetaSpec(init=1.5, lower=0.5, upper=5.0),
            ThetaSpec(init=2.8, lower=0.5, upper=10.0),
            ThetaSpec(init=32.9, lower=10.0, upper=80.0),
        ],
        [OmegaSpec(block_size=1, values=[0.04])],
        [SigmaSpec(block_size=1, values=[0.01])],
    )
    pop_model = PopulationModel(
        dataset=dataset, pk_subroutine=ADVAN2(), params=params, trans=2, advan=2
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


def fraction_obs_p50_in_sim_range(vpc_result) -> float:
    merged = vpc_result.obs_percentiles[["bin_mid", "p50"]].merge(
        vpc_result.sim_percentiles[["bin_mid", "p5_lo", "p95_hi"]],
        on="bin_mid",
        how="inner",
    )
    in_range = (merged["p50"] >= merged["p5_lo"]) & (merged["p50"] <= merged["p95_hi"])
    return float(in_range.mean())


def build_npc_result(
    *,
    n_subjects: int = 24,
    seed: int = 7,
    n_replicates: int = 100,
    n_bins: int = 8,
):
    from openpkpd.simulation.engine import SimulationEngine
    from openpkpd.simulation.npc import NPCEngine

    pop_model, est_result = build_pop_model_and_result(n_subjects=n_subjects, seed=seed)
    sim_result = SimulationEngine(pop_model, est_result, seed=seed).simulate(
        n_replicates=n_replicates
    )
    return NPCEngine(sim_result).compute(n_bins=n_bins)


def build_sse_result(
    *,
    n_subjects: int = 8,
    data_seed: int = 11,
    run_seed: int = 11,
    n_replicates: int = 4,
    estimation_method: str = "FO",
):
    from openpkpd.simulation.sse import SSEEngine

    pop_model, est_result = build_pop_model_and_result(n_subjects=n_subjects, seed=data_seed)
    return SSEEngine(pop_model, est_result, estimation_method=estimation_method).run(
        n_replicates=n_replicates,
        seed=run_seed,
    )


def make_mock_npde_engine(
    n_subjects: int = 8,
    n_obs: int = 6,
    n_replicates: int = 200,
    noise_sd: float = 0.5,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    times = np.arange(1, n_obs + 1, dtype=float)
    true_conc = 10.0 * np.exp(-0.2 * times)
    records: list[dict[str, float | int]] = []
    for sid in range(1, n_subjects + 1):
        for t, mu in zip(times, true_conc, strict=False):
            records.append(
                {"ID": sid, "TIME": t, "DV": mu + rng.normal(0, noise_sd), "REP": 0, "MDV": 0}
            )
    for rep in range(1, n_replicates + 1):
        for sid in range(1, n_subjects + 1):
            for t, mu in zip(times, true_conc, strict=False):
                records.append(
                    {"ID": sid, "TIME": t, "DV": mu + rng.normal(0, noise_sd), "REP": rep, "MDV": 0}
                )
    sim_df = pd.DataFrame(records)

    class _MockResult:
        simulated_df = sim_df

    class _MockEngine:
        def simulate(self, n_replicates: int):
            return _MockResult()

    return _MockEngine()


def theophylline_nca_profile() -> tuple[np.ndarray, np.ndarray, float]:
    ka = 1.5
    cl = 2.8
    v = 32.9
    dose = 320.0
    k = cl / v
    times = np.array(
        [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]
    )
    conc = np.where(
        np.isclose(times, 0.0),
        0.0,
        dose * ka / (v * (ka - k)) * (np.exp(-k * times) - np.exp(-ka * times)),
    )
    return times, conc, dose
