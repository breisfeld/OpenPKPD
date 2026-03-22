from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from profile_pipelines import (
    build_covariate_diagnostics_model_and_result,
    build_fit_model,
    build_simulation_model_and_result,
    profile_workload,
)

from openpkpd.model import symbolic_eta
import openpkpd.simulation.npde as npde_mod
import openpkpd.simulation.vpc as vpc_mod
from openpkpd.model.individual import IndividualModel
from openpkpd.nca.nca import NCAEngine
from openpkpd.plots import diagnostics as diagnostics_mod
from openpkpd.simulation.engine import SimulationEngine
from openpkpd.simulation.npde import NPDEEngine
from openpkpd.simulation.vpc import VPCEngine

DEFAULT_JSON_OUT = Path("artifacts/profiling/analysis_baseline.json")


def build_nca_dataset(n_subjects: int = 2000, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dose = 100.0
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0], dtype=float)
    cl_tv, v_tv, ka_tv = 5.0, 50.0, 1.5
    sigma_prop = 0.15
    omega_cl = omega_v = omega_ka = 0.09
    rows: list[dict[str, float | int]] = []

    for sid in range(1, n_subjects + 1):
        eta_cl = rng.normal(0.0, math.sqrt(omega_cl))
        eta_v = rng.normal(0.0, math.sqrt(omega_v))
        eta_ka = rng.normal(0.0, math.sqrt(omega_ka))
        cl = cl_tv * math.exp(eta_cl)
        v = v_tv * math.exp(eta_v)
        ka = ka_tv * math.exp(eta_ka)
        k = cl / v
        if abs(ka - k) < 1e-8:
            ka += 1e-6

        rows.append({"ID": sid, "TIME": 0.0, "AMT": dose, "DV": 0.0, "EVID": 1, "MDV": 1})
        prefix = dose / v * ka / (ka - k)
        conc = prefix * (np.exp(-k * times[1:]) - np.exp(-ka * times[1:]))
        conc = np.maximum(conc, 0.0)
        eps = rng.normal(0.0, sigma_prop, size=len(conc))
        conc = np.maximum(conc * (1.0 + eps), 0.0)

        for obs_time, obs_conc in zip(times[1:], conc, strict=False):
            rows.append(
                {
                    "ID": sid,
                    "TIME": float(obs_time),
                    "AMT": 0.0,
                    "DV": float(obs_conc),
                    "EVID": 0,
                    "MDV": 0,
                }
            )

    return pd.DataFrame(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile representative diagnostics, VPC, NPDE, and NCA analysis routines.")
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["all"],
        choices=["all", "diagnostics", "diagnostics_covariate", "npde", "vpc", "nca"],
    )
    parser.add_argument("--fit-subjects", type=int, default=6)
    parser.add_argument("--fit-maxeval", type=int, default=12)
    parser.add_argument("--covariate-subjects", type=int, default=140)
    parser.add_argument("--sim-subjects", type=int, default=24)
    parser.add_argument("--npde-simulations", type=int, default=500)
    parser.add_argument("--vpc-replicates", type=int, default=500)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--nca-subjects", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--top-functions", type=int, default=15)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    return parser.parse_args(argv)


def _profile_covariate_diagnostics_mode(
    *,
    n_subjects: int,
    seed: int,
    top_functions: int,
    symbolic_enabled: bool,
) -> dict[str, Any]:
    requested_symbolic = bool(symbolic_enabled)
    original_flag = symbolic_eta.SYMPY_AVAILABLE
    effective_symbolic = requested_symbolic and getattr(symbolic_eta, "sp", None) is not None
    try:
        symbolic_eta.SYMPY_AVAILABLE = effective_symbolic
        pop_model, result = build_covariate_diagnostics_model_and_result(n_subjects=n_subjects, seed=seed)
        first_subject_id = pop_model.subject_ids()[0]
        indiv = pop_model.individual_model(first_subject_id)
        support = bool(indiv.supports_prediction_eta_jacobian(pop_model.trans))
        kernel = indiv.get_subject_derivative_kernel(pop_model.trans) if support else None
        warm_df = diagnostics_mod.compute_diagnostics(pop_model, result)
        profile = profile_workload(
            f"diagnostics_covariate_{'symbolic' if requested_symbolic else 'finite_difference'}",
            lambda: {"rows": int(len(diagnostics_mod.compute_diagnostics(pop_model, result)))},
            [
                (diagnostics_mod, "_prediction_eta_jacobian", "diagnostics.prediction_eta_jacobian"),
                (diagnostics_mod, "_finite_diff_jacobian", "diagnostics.finite_diff_jacobian"),
                (IndividualModel, "evaluate", "individual.evaluate"),
            ],
            top_functions,
        )
        return {
            "requested_symbolic": requested_symbolic,
            "effective_symbolic": effective_symbolic,
            "supports_prediction_eta_jacobian": support,
            "kernel": type(kernel).__name__ if kernel is not None else None,
            "warm_rows": int(len(warm_df)),
            "profile": profile,
        }
    finally:
        symbolic_eta.SYMPY_AVAILABLE = original_flag


def profile_covariate_diagnostics_comparison(
    n_subjects: int,
    seed: int,
    top_functions: int,
) -> dict[str, Any]:
    symbolic_result = _profile_covariate_diagnostics_mode(
        n_subjects=n_subjects,
        seed=seed,
        top_functions=top_functions,
        symbolic_enabled=True,
    )
    finite_diff_result = _profile_covariate_diagnostics_mode(
        n_subjects=n_subjects,
        seed=seed,
        top_functions=top_functions,
        symbolic_enabled=False,
    )
    symbolic_wall = float(symbolic_result["profile"]["wall_seconds"])
    fallback_wall = float(finite_diff_result["profile"]["wall_seconds"])
    return {
        "n_subjects": int(n_subjects),
        "symbolic": symbolic_result,
        "finite_difference": finite_diff_result,
        "speedup_vs_fallback": round(fallback_wall / symbolic_wall, 3) if symbolic_wall > 0.0 else None,
        "wall_seconds_saved": round(fallback_wall - symbolic_wall, 6),
    }


def run_profiles(args: argparse.Namespace) -> dict[str, Any]:
    workloads = {"diagnostics", "diagnostics_covariate", "npde", "vpc", "nca"} if "all" in args.workloads else set(args.workloads)
    results: dict[str, Any] = {
        "metadata": {
            "date": date.today().isoformat(),
            "seed": int(args.seed),
            "workloads": sorted(workloads),
            "parameters": {
                "fit_subjects": int(args.fit_subjects),
                "fit_maxeval": int(args.fit_maxeval),
                "covariate_subjects": int(args.covariate_subjects),
                "sim_subjects": int(args.sim_subjects),
                "npde_simulations": int(args.npde_simulations),
                "vpc_replicates": int(args.vpc_replicates),
                "n_bins": int(args.n_bins),
                "nca_subjects": int(args.nca_subjects),
                "top_functions": int(args.top_functions),
            },
        }
    }

    fit_pop = None
    fit_result = None
    if {"diagnostics", "npde"} & workloads:
        built = build_fit_model(args.fit_subjects, args.seed, args.fit_maxeval)
        fit_result = built.fit()
        fit_pop = built.population_model

    sim_pop = None
    sim_result = None
    if "vpc" in workloads:
        sim_pop, sim_result = build_simulation_model_and_result(args.sim_subjects, args.seed)

    if "diagnostics" in workloads:
        assert fit_pop is not None and fit_result is not None
        results["diagnostics"] = profile_workload(
            "diagnostics",
            lambda: {"rows": int(len(diagnostics_mod.compute_diagnostics(fit_pop, fit_result)))},
            [
                (diagnostics_mod, "_finite_diff_jacobian", "diagnostics.finite_diff_jacobian"),
                (diagnostics_mod, "_cwres_subject", "diagnostics.cwres_subject"),
                (IndividualModel, "evaluate", "individual.evaluate"),
            ],
            args.top_functions,
        )

    if "diagnostics_covariate" in workloads:
        results["diagnostics_covariate"] = profile_covariate_diagnostics_comparison(
            args.covariate_subjects,
            args.seed,
            args.top_functions,
        )

    if "npde" in workloads:
        assert fit_pop is not None and fit_result is not None
        results["npde"] = profile_workload(
            "npde",
            lambda: {"rows": int(len(diagnostics_mod.compute_npde(fit_pop, fit_result, n_simulations=args.npde_simulations, seed=args.seed, decorrelate=True)))},
            [
                (diagnostics_mod, "compute_diagnostics", "diagnostics.compute_diagnostics"),
                (SimulationEngine, "simulate", "simulation.simulate"),
                (NPDEEngine, "_build_sim_matrix", "npde.build_sim_matrix"),
                (npde_mod, "_compute_pd", "npde.compute_pd"),
                (npde_mod, "_decorrelate", "npde.decorrelate"),
            ],
            args.top_functions,
        )

    if "vpc" in workloads:
        assert sim_pop is not None and sim_result is not None
        results["vpc"] = profile_workload(
            "vpc",
            lambda: {"sim_rows": int(len(VPCEngine(SimulationEngine(sim_pop, sim_result, seed=args.seed, n_parallel=1)).compute(n_replicates=args.vpc_replicates, n_bins=args.n_bins).simulated_df))},
            [
                (SimulationEngine, "simulate", "simulation.simulate"),
                (vpc_mod, "_compute_obs_percentiles", "vpc.obs_percentiles"),
                (vpc_mod, "_compute_sim_percentiles", "vpc.sim_percentiles"),
            ],
            args.top_functions,
        )

    if "nca" in workloads:
        nca_df = build_nca_dataset(args.nca_subjects, args.seed)
        nca_engine = NCAEngine(
            auc_method="linear-log",
            lambda_z_method="auto",
            min_points_lambda=3,
            exclude_cmax=True,
        )
        results["nca"] = profile_workload(
            "nca",
            lambda: {"rows": int(len(nca_engine.compute_dataset(nca_df, id_col="ID", time_col="TIME", conc_col="DV", dose_col="AMT", dose_row_col="EVID", route="oral")))},
            [
                (NCAEngine, "compute_subject", "nca.compute_subject"),
                (NCAEngine, "_compute_auc", "nca.compute_auc"),
                (NCAEngine, "_compute_lambda_z", "nca.compute_lambda_z"),
            ],
            args.top_functions,
        )

    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_profiles(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.json_out}", file=sys.stderr)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())