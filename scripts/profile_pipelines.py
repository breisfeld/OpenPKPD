from __future__ import annotations

import argparse
import cProfile
import json
import math
import pstats
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.data.event_processor import DoseEvent, SubjectEvents
from openpkpd.estimation.base import EstimationResult
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.model.individual import IndividualModel
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.parser.code_compiler import NMTRANCompiler
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.pk.analytical.advan4 import ADVAN4
from openpkpd.simulation.engine import SimulationEngine
from openpkpd.simulation.npde import NPDEEngine
from openpkpd.simulation.vpc import VPCEngine
import openpkpd.simulation.npde as npde_mod
import openpkpd.simulation.vpc as vpc_mod


@dataclass
class StageStat:
    seconds: float = 0.0
    calls: int = 0


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
        k = cl / v
        rows.append({"ID": sid, "TIME": 0.0, "AMT": dose, "DV": 0.0, "EVID": 1, "MDV": 1, "CMT": 1, "RATE": 0.0, "ADDL": 0, "II": 0, "SS": 0})
        for t in obs_times:
            if abs(ka_pop - k) < 1e-6:
                conc = dose * ka_pop / v * t * math.exp(-k * t)
            else:
                conc = dose * ka_pop / (v * (ka_pop - k)) * (math.exp(-k * t) - math.exp(-ka_pop * t))
            dv = max(conc * (1 + rng.normal(0, 0.1)), 0.001)
            rows.append({"ID": sid, "TIME": t, "AMT": 0.0, "DV": dv, "EVID": 0, "MDV": 0, "CMT": 1, "RATE": 0.0, "ADDL": 0, "II": 0, "SS": 0})
    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def build_fit_model(n_subjects: int, seed: int, maxeval: int) -> Any:
    dataset = build_small_pk_dataset(n_subjects=n_subjects, seed=seed)
    return (
        ModelBuilder()
        .problem("Profiling FOCE workload")
        .dataset(dataset)
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.2, 5.0), (0.01, 2.5, 10.0), (1.0, 30.0, 80.0)])
        .omega([0.04, 0.02, 0.02])
        .sigma(0.01)
        .estimation(method="FOCE", interaction=True, maxeval=maxeval, n_parallel=1, print_interval=10_000)
        .build()
    )


def build_simulation_model_and_result(n_subjects: int, seed: int) -> tuple[PopulationModel, EstimationResult]:
    dataset = build_small_pk_dataset(n_subjects=n_subjects, seed=seed)
    params = ParameterSet.from_specs(
        [ThetaSpec(init=1.5, lower=0.5, upper=5.0), ThetaSpec(init=2.8, lower=0.5, upper=10.0), ThetaSpec(init=32.9, lower=10.0, upper=80.0)],
        [OmegaSpec(block_size=1, values=[0.04])],
        [SigmaSpec(block_size=1, values=[0.01])],
    )
    pop_model = PopulationModel(dataset=dataset, pk_subroutine=ADVAN2(), params=params, trans=2, advan=2)
    result = EstimationResult(
        theta_final=params.theta.copy(),
        omega_final=params.omega.copy(),
        sigma_final=params.sigma.copy(),
        ofv=100.0,
        converged=True,
        post_hoc_etas={sid: np.zeros(params.n_eta()) for sid in pop_model.subject_ids()},
    )
    return pop_model, result


_COVARIATE_DIAGNOSTIC_THETA = np.array([1.35, 2.8, 32.0, 0.75, 0.006, 0.004], dtype=float)
_COVARIATE_DIAGNOSTIC_OMEGA_DIAG = np.array([0.04, 0.03, 0.02], dtype=float)
_COVARIATE_DIAGNOSTIC_SIGMA = np.array([[0.02]], dtype=float)
_COVARIATE_DIAGNOSTIC_TIMES = np.array([0.5, 1.0, 2.0, 4.0, 7.0, 12.0, 24.0], dtype=float)
_COVARIATE_DIAGNOSTIC_DOSE = 320.0
_COVARIATE_DIAGNOSTIC_PK_CODE = (
    "KA = THETA(1)*EXP(ETA(1))\n"
    "CL = THETA(2)*EXP(ETA(2))\n"
    "V = THETA(3)*EXP(ETA(3))\n"
    "CL = CL * (WT/70.0)**THETA(4)\n"
    "V = V * (1 + THETA(5) * (AGE - 40.0))\n"
    "KA = KA * EXP(THETA(6) * (WT - 70.0))"
)


def build_covariate_diagnostics_dataset(
    n_subjects: int = 140,
    seed: int = 20260316,
) -> tuple[NONMEMDataset, dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    eta_map: dict[int, np.ndarray] = {}
    sigma_sd = math.sqrt(float(_COVARIATE_DIAGNOSTIC_SIGMA[0, 0]))
    for sid in range(1, n_subjects + 1):
        wt = float(rng.uniform(50.0, 100.0))
        age = float(rng.uniform(20.0, 70.0))
        eta = rng.normal(0.0, np.sqrt(_COVARIATE_DIAGNOSTIC_OMEGA_DIAG), size=3).astype(float)
        eta_map[sid] = eta
        ka = (
            float(_COVARIATE_DIAGNOSTIC_THETA[0])
            * math.exp(float(eta[0]))
            * math.exp(float(_COVARIATE_DIAGNOSTIC_THETA[5]) * (wt - 70.0))
        )
        cl = (
            float(_COVARIATE_DIAGNOSTIC_THETA[1])
            * math.exp(float(eta[1]))
            * (wt / 70.0) ** float(_COVARIATE_DIAGNOSTIC_THETA[3])
        )
        v = (
            float(_COVARIATE_DIAGNOSTIC_THETA[2])
            * math.exp(float(eta[2]))
            * (1.0 + float(_COVARIATE_DIAGNOSTIC_THETA[4]) * (age - 40.0))
        )
        k = cl / v
        rows.append(
            {
                "ID": sid,
                "TIME": 0.0,
                "AMT": _COVARIATE_DIAGNOSTIC_DOSE,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
                "WT": wt,
                "AGE": age,
            }
        )
        for t in _COVARIATE_DIAGNOSTIC_TIMES:
            if abs(ka - k) < 1e-8:
                conc = _COVARIATE_DIAGNOSTIC_DOSE * ka / v * float(t) * math.exp(-k * float(t))
            else:
                conc = (
                    _COVARIATE_DIAGNOSTIC_DOSE
                    * ka
                    / (v * (ka - k))
                    * (math.exp(-k * float(t)) - math.exp(-ka * float(t)))
                )
            dv = max(conc * (1.0 + rng.normal(0.0, sigma_sd)), 1e-4)
            rows.append(
                {
                    "ID": sid,
                    "TIME": float(t),
                    "AMT": 0.0,
                    "DV": float(dv),
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                    "RATE": 0.0,
                    "ADDL": 0,
                    "II": 0,
                    "SS": 0,
                    "WT": wt,
                    "AGE": age,
                }
            )
    return NONMEMDataset.from_dataframe(pd.DataFrame(rows)), eta_map


def build_covariate_diagnostics_model_and_result(
    n_subjects: int = 140,
    seed: int = 20260316,
) -> tuple[PopulationModel, EstimationResult]:
    dataset, eta_map = build_covariate_diagnostics_dataset(n_subjects=n_subjects, seed=seed)
    built = (
        ModelBuilder()
        .problem("Covariate-heavy diagnostics benchmark")
        .dataset(dataset)
        .covariates(["WT", "AGE"])
        .subroutines(advan=2, trans=2)
        .pk(_COVARIATE_DIAGNOSTIC_PK_CODE)
        .error("Y = F*(1 + EPS(1))")
        .theta(
            [
                (0.01, float(_COVARIATE_DIAGNOSTIC_THETA[0]), 5.0),
                (0.01, float(_COVARIATE_DIAGNOSTIC_THETA[1]), 10.0),
                (1.0, float(_COVARIATE_DIAGNOSTIC_THETA[2]), 100.0),
                (-2.0, float(_COVARIATE_DIAGNOSTIC_THETA[3]), 2.0),
                (-0.05, float(_COVARIATE_DIAGNOSTIC_THETA[4]), 0.05),
                (-0.05, float(_COVARIATE_DIAGNOSTIC_THETA[5]), 0.05),
            ]
        )
        .omega(_COVARIATE_DIAGNOSTIC_OMEGA_DIAG.tolist())
        .sigma(float(_COVARIATE_DIAGNOSTIC_SIGMA[0, 0]))
        .estimation(method="FOCE", interaction=True, maxeval=1, n_parallel=1, print_interval=10_000)
        .build()
    )
    n_obs = int(((dataset.df["EVID"].fillna(0) == 0) & (dataset.df["MDV"].fillna(0) == 0)).sum())
    result = EstimationResult(
        theta_final=_COVARIATE_DIAGNOSTIC_THETA.copy(),
        omega_final=np.diag(_COVARIATE_DIAGNOSTIC_OMEGA_DIAG),
        sigma_final=_COVARIATE_DIAGNOSTIC_SIGMA.copy(),
        ofv=0.0,
        converged=True,
        post_hoc_etas={sid: eta.copy() for sid, eta in eta_map.items()},
        n_subjects=n_subjects,
        n_observations=n_obs,
        method="PROFILE",
    )
    return built.population_model, result


def _build_symbolic_individual_model(
    *,
    pk_subroutine: Any,
    pk_code: str,
    error_code: str,
    n_eps: int,
    obs_times: np.ndarray,
    obs_dv: np.ndarray,
    dose_events: list[DoseEvent],
) -> IndividualModel:
    compiler = NMTRANCompiler()
    return IndividualModel(
        subject_events=SubjectEvents(
            subject_id=1,
            dose_events=dose_events,
            obs_times=np.asarray(obs_times, dtype=float),
            obs_dv=np.asarray(obs_dv, dtype=float),
            obs_cmt=np.ones(len(obs_times), dtype=int),
            obs_mdv=np.zeros(len(obs_times), dtype=int),
        ),
        pk_subroutine=pk_subroutine,
        pk_callable=compiler.compile_pk(pk_code),
        error_callable=compiler.compile_error(error_code),
        n_eps=n_eps,
    )


def build_symbolic_benchmark_cases() -> list[tuple[str, IndividualModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    return [
        (
            "advan1",
            _build_symbolic_individual_model(
                pk_subroutine=ADVAN1(),
                pk_code="CL = THETA(1)*EXP(ETA(1))\nV = THETA(2)*EXP(ETA(2))",
                error_code="IPRED = F\nW = IPRED * THETA(3)\nY = IPRED + W * EPS(1)",
                n_eps=1,
                obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=float),
                obs_dv=np.array([7.0, 6.6, 5.3, 3.6, 2.0, 0.8], dtype=float),
                dose_events=[DoseEvent(time=0.0, amount=280.0, compartment=1)],
            ),
            np.array([2.0, 25.0, 0.15]),
            np.array([0.03, -0.05]),
            np.diag([0.04, 0.02]),
            np.array([[1.0]]),
            2,
        ),
        (
            "advan3",
            _build_symbolic_individual_model(
                pk_subroutine=ADVAN3(),
                pk_code="CL = THETA(1)*EXP(ETA(1))\nV1 = THETA(2)*EXP(ETA(2))\nQ = THETA(3)\nV2 = THETA(4)",
                error_code="IPRED = F\nW = IPRED * THETA(5)\nY = IPRED + W * EPS(1)",
                n_eps=1,
                obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=float),
                obs_dv=np.array([7.1, 6.5, 5.2, 3.7, 1.9, 0.9], dtype=float),
                dose_events=[
                    DoseEvent(time=0.0, amount=260.0, compartment=1),
                    DoseEvent(time=10.0, amount=130.0, compartment=1),
                ],
            ),
            np.array([1.7, 12.0, 0.65, 18.0, 0.12]),
            np.array([0.04, -0.06]),
            np.diag([0.04, 0.03]),
            np.array([[1.0]]),
            4,
        ),
        (
            "advan4",
            _build_symbolic_individual_model(
                pk_subroutine=ADVAN4(),
                pk_code=(
                    "KA = THETA(1)*EXP(ETA(1))\n"
                    "CL = THETA(2)*EXP(ETA(2))\n"
                    "V2 = THETA(3)*EXP(ETA(3))\n"
                    "Q = THETA(4)\n"
                    "V3 = THETA(5)\n"
                    "K = CL/V2\n"
                    "K12 = Q/V2\n"
                    "K21 = Q/V3"
                ),
                error_code="IPRED = F\nW = IPRED * THETA(6)\nY = IPRED + W * EPS(1)",
                n_eps=1,
                obs_times=np.array([0.5, 1.0, 2.0, 4.0, 8.0, 14.0], dtype=float),
                obs_dv=np.array([5.8, 7.0, 7.4, 6.0, 3.2, 1.4], dtype=float),
                dose_events=[
                    DoseEvent(time=0.0, amount=220.0, compartment=1),
                    DoseEvent(time=9.0, amount=110.0, compartment=1),
                ],
            ),
            np.array([1.25, 1.85, 12.5, 0.7, 19.0, 0.14]),
            np.array([0.02, -0.04, 0.03]),
            np.diag([0.04, 0.03, 0.02]),
            np.array([[1.0]]),
            1,
        ),
    ]


def prewarm_symbolic_benchmark_cases(
    cases: list[tuple[str, IndividualModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]],
) -> None:
    for _name, model, theta, eta, omega, sigma, trans in cases:
        assert model.get_subject_derivative_kernel(trans) is not None
        model.obj_eta(eta, theta, omega, sigma, trans=trans)
        model.symbolic_obj_eta_value_grad(eta, theta, omega, sigma, trans=trans)
        model.prediction_eta_jacobian(theta, eta, sigma, trans=trans)
        model.eta_objective_hessian(theta, eta, omega, sigma, trans=trans)


def run_symbolic_workload(
    iterations: int,
    cases: list[tuple[str, IndividualModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    symbolic_cases = build_symbolic_benchmark_cases() if cases is None else cases
    for name, model, theta, eta0, omega, sigma, trans in symbolic_cases:
        kernel = model.get_subject_derivative_kernel(trans)
        assert kernel is not None, name
        checksum = 0.0
        grad_norm = 0.0
        jac_norm = 0.0
        hess_norm = 0.0
        phase_offsets = np.arange(len(eta0), dtype=float) * 0.7
        for step in range(iterations):
            phase = 2.0 * math.pi * step / max(iterations, 1)
            eta = eta0 + 0.03 * np.sin(phase + phase_offsets)
            checksum += float(model.obj_eta(eta, theta, omega, sigma, trans=trans))
            value, grad = model.symbolic_obj_eta_value_grad(eta, theta, omega, sigma, trans=trans)
            checksum += float(value)
            grad_norm += float(np.linalg.norm(grad))
            jac = model.prediction_eta_jacobian(theta, eta, sigma, trans=trans)
            jac_norm += float(np.linalg.norm(jac))
            hess = model.eta_objective_hessian(theta, eta, omega, sigma, trans=trans)
            hess_norm += float(np.linalg.norm(hess))
        results[name] = {
            "kernel": type(kernel).__name__,
            "iterations": int(iterations),
            "checksum": round(checksum, 6),
            "grad_norm_sum": round(grad_norm, 6),
            "jac_norm_sum": round(jac_norm, 6),
            "hess_norm_sum": round(hess_norm, 6),
        }
    return results


@contextmanager
def timed_patch(target: Any, attr: str, stats: dict[str, StageStat], label: str):
    original = getattr(target, attr)

    def wrapped(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            stat = stats.setdefault(label, StageStat())
            stat.seconds += time.perf_counter() - t0
            stat.calls += 1

    setattr(target, attr, wrapped)
    try:
        yield
    finally:
        setattr(target, attr, original)


def top_functions(profile: cProfile.Profile, limit: int) -> list[dict[str, Any]]:
    rows = []
    script_name = Path(__file__).name
    for (filename, line, funcname), (cc, nc, tt, ct, _callers) in pstats.Stats(profile).stats.items():
        if Path(filename).name == script_name:
            continue
        rows.append((ct, tt, nc, cc, filename, line, funcname))
    rows.sort(reverse=True)
    return [
        {
            "function": funcname,
            "location": f"{Path(filename).name}:{line}",
            "cumulative_seconds": round(float(ct), 6),
            "self_seconds": round(float(tt), 6),
            "ncalls": int(nc),
            "primitive_calls": int(cc),
        }
        for ct, tt, nc, cc, filename, line, funcname in rows[:limit]
    ]


def profile_workload(name: str, func: Any, patches: list[tuple[Any, str, str]], limit: int) -> dict[str, Any]:
    stage_stats: dict[str, StageStat] = {}
    profiler = cProfile.Profile()
    with ExitStack() as stack:
        for target, attr, label in patches:
            stack.enter_context(timed_patch(target, attr, stage_stats, label))
        t0 = time.perf_counter()
        result = profiler.runcall(func)
        wall_seconds = time.perf_counter() - t0
    return {
        "name": name,
        "wall_seconds": round(wall_seconds, 6),
        "result": result,
        "stage_totals": {k: asdict(v) for k, v in sorted(stage_stats.items())},
        "top_functions": top_functions(profiler, limit),
    }


def run_foce_workload(subjects: int, seed: int, maxeval: int) -> dict[str, Any]:
    built = build_fit_model(subjects, seed, maxeval)
    result = built.fit()
    return {"converged": bool(result.converged), "ofv": float(result.ofv), "n_subjects": subjects}


def run_vpc_workload(subjects: int, seed: int, replicates: int, bins: int) -> dict[str, Any]:
    pop_model, result = build_simulation_model_and_result(subjects, seed)
    vpc_result = VPCEngine(SimulationEngine(pop_model, result, seed=seed, n_parallel=1)).compute(
        n_replicates=replicates,
        n_bins=bins,
    )
    return {"n_subjects": subjects, "n_replicates": replicates, "sim_rows": int(len(vpc_result.simulated_df))}


def run_npde_workload(subjects: int, seed: int, replicates: int) -> dict[str, Any]:
    pop_model, result = build_simulation_model_and_result(subjects, seed)
    npde_result = NPDEEngine(SimulationEngine(pop_model, result, seed=seed, n_parallel=1)).compute(
        n_replicates=replicates,
        seed=seed,
    )
    return {"n_subjects": subjects, "n_replicates": replicates, "rows": int(len(npde_result.df))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile representative FOCE/VPC/NPDE/symbolic workloads.")
    parser.add_argument("--workloads", nargs="+", default=["all"], choices=["all", "foce", "vpc", "npde", "symbolic"])
    parser.add_argument("--foce-subjects", type=int, default=8)
    parser.add_argument("--foce-maxeval", type=int, default=25)
    parser.add_argument("--sim-subjects", type=int, default=24)
    parser.add_argument("--n-replicates", type=int, default=500)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--symbolic-iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--top-functions", type=int, default=15)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workloads = {"foce", "vpc", "npde", "symbolic"} if "all" in args.workloads else set(args.workloads)
    results: dict[str, Any] = {}

    if "foce" in workloads:
        results["foce"] = profile_workload(
            "foce",
            lambda: run_foce_workload(args.foce_subjects, args.seed, args.foce_maxeval),
            [
                (FOCEMethod, "_inner_loop", "foce.inner_loop"),
                (FOCEMethod, "_outer_ofv", "foce.outer_ofv"),
                (IndividualModel, "obj_eta", "individual.obj_eta"),
                (IndividualModel, "evaluate_observation_model", "individual.evaluate_observation_model"),
                (IndividualModel, "_evaluate_predictions", "individual._evaluate_predictions"),
            ],
            args.top_functions,
        )

    if "vpc" in workloads:
        results["vpc"] = profile_workload(
            "vpc",
            lambda: run_vpc_workload(args.sim_subjects, args.seed, args.n_replicates, args.n_bins),
            [
                (SimulationEngine, "simulate", "simulation.simulate"),
                (vpc_mod, "_compute_obs_percentiles", "vpc.obs_percentiles"),
                (vpc_mod, "_compute_sim_percentiles", "vpc.sim_percentiles"),
            ],
            args.top_functions,
        )

    if "npde" in workloads:
        results["npde"] = profile_workload(
            "npde",
            lambda: run_npde_workload(args.sim_subjects, args.seed, args.n_replicates),
            [
                (SimulationEngine, "simulate", "simulation.simulate"),
                (NPDEEngine, "_build_sim_matrix", "npde.build_sim_matrix"),
                (npde_mod, "_compute_pd", "npde.compute_pd"),
                (npde_mod, "_decorrelate", "npde.decorrelate"),
            ],
            args.top_functions,
        )

    if "symbolic" in workloads:
        symbolic_cases = build_symbolic_benchmark_cases()
        prewarm_symbolic_benchmark_cases(symbolic_cases)
        results["symbolic"] = profile_workload(
            "symbolic",
            lambda: run_symbolic_workload(args.symbolic_iterations, symbolic_cases),
            [
                (IndividualModel, "obj_eta", "individual.obj_eta"),
                (IndividualModel, "symbolic_obj_eta_value_grad", "individual.symbolic_obj_eta_value_grad"),
                (IndividualModel, "prediction_eta_jacobian", "individual.prediction_eta_jacobian"),
                (IndividualModel, "eta_objective_hessian", "individual.eta_objective_hessian"),
                (IndividualModel, "evaluate_observation_model", "individual.evaluate_observation_model"),
                (IndividualModel, "_evaluate_predictions", "individual._evaluate_predictions"),
            ],
            args.top_functions,
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()