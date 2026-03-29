#!/usr/bin/env python3
"""
Estimation method benchmark.

Measures wall-clock time, per-stage timing, and hot function call profiles
for FO, FOCE, FOCEI, SAEM, IMP, IMPMAP, and Bayesian backends on a standard
1-compartment oral PK dataset.

The results are saved as JSON and are intended as the *performance baseline*
against which future optimisations (e.g. the PyO3 obj_eta extension) are
compared.

Usage
-----
    uv run python scripts/benchmark_estimation.py
    uv run python scripts/benchmark_estimation.py --workloads fo foce focei saem imp impmap bayes_laplace bayes_nuts
    uv run python scripts/benchmark_estimation.py \\
        --json-out artifacts/profiling/estimation_baseline.json

Output schema (per workload)
----------------------------
    {
      "name":          str,          # workload label
      "wall_seconds":  float,        # total elapsed wall time
      "result": {
        "converged":   bool,
        "ofv":         float,
        "n_function_evals": int,
        "n_subjects":  int
      },
      "stage_totals": {              # cumulative time spent in each patched fn
        "<label>": {"seconds": float, "calls": int}, ...
      },
      "top_functions": [             # cProfile hottest functions
        {"function": str, "location": str,
         "cumulative_seconds": float, "self_seconds": float,
         "ncalls": int, "primitive_calls": int}, ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ── path bootstrap ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from profile_pipelines import build_small_pk_dataset, profile_workload  # noqa: E402

from openpkpd import ModelBuilder  # noqa: E402
from openpkpd.estimation.bayes import BAYESMethod  # noqa: E402
from openpkpd.estimation.fo import FOMethod  # noqa: E402
from openpkpd.estimation.foce import FOCEMethod  # noqa: E402
from openpkpd.estimation.imp import IMPMethod  # noqa: E402
from openpkpd.estimation.nuts import NUTSSampler  # noqa: E402
from openpkpd.estimation.saem import SAEMMethod  # noqa: E402
from openpkpd.model.individual import IndividualModel  # noqa: E402
from openpkpd.model.parameters import ParameterSet  # noqa: E402
from openpkpd.model.population import PopulationModel  # noqa: E402

# ── default workload parameters ───────────────────────────────────────────────
# These values are chosen so that:
#   • The run finishes in < 5 min on a typical laptop.
#   • Each method's hot path is exercised enough to be representative.
#   • The same dataset is reused across methods for comparability.
_DEFAULT_N_SUBJECTS = 12
_DEFAULT_SEED = 42
_DEFAULT_TOP_FN = 20

_FO_MAXEVAL = 500          # FO is fast; run to convergence
_FOCE_MAXEVAL = 300        # FOCE (no interaction)
_FOCEI_MAXEVAL = 200       # FOCEI (with interaction, more expensive)
_SAEM_K1 = 150             # SAEM stochastic phase iterations
_SAEM_K2 = 100             # SAEM convergence phase iterations
_IMP_ISAMPLE = 80
_IMP_MAXEVAL = 8
_BAYES_N_SAMPLES = 150
_BAYES_TUNE = 75

_DEFAULT_JSON_OUT = Path("artifacts/profiling/estimation_baseline.json")

# ── dataset / model helpers ───────────────────────────────────────────────────

def _build_population_model(n_subjects: int, seed: int) -> tuple[PopulationModel, ParameterSet]:
    """1-compartment oral PK (ADVAN2/TRANS2) with compiled callables."""
    built = (
        ModelBuilder()
        .problem("Benchmark oral PK workload")
        .dataset(build_small_pk_dataset(n_subjects=n_subjects, seed=seed))
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.5, 8.0), (0.01, 2.8, 15.0), (1.0, 32.9, 80.0)])
        .omega([0.04, 0.02, 0.02])
        .sigma(0.01)
        .build()
    )
    return built.population_model, built.population_model.params


def _result_summary(res: Any, n_subjects: int) -> dict[str, Any]:
    return {
        "converged": bool(res.converged),
        "ofv": float(res.ofv),
        "n_function_evals": int(getattr(res, "n_function_evals", -1)),
        "n_subjects": int(n_subjects),
    }


# ── individual workload runners ───────────────────────────────────────────────

def run_fo(n_subjects: int, seed: int, maxeval: int, top_fn: int) -> dict[str, Any]:
    pop, params = _build_population_model(n_subjects, seed)
    method = FOMethod(maxeval=maxeval, print_interval=100_000)
    return profile_workload(
        "fo",
        lambda: _result_summary(method.estimate(pop, params), n_subjects),
        [
            (FOMethod, "_compute_fo_ofv", "fo.compute_fo_ofv"),
            (IndividualModel, "evaluate_observation_model", "individual.evaluate_observation_model"),
            (IndividualModel, "_evaluate_predictions", "individual._evaluate_predictions"),
        ],
        top_fn,
    )


def run_foce(n_subjects: int, seed: int, maxeval: int, top_fn: int) -> dict[str, Any]:
    pop, params = _build_population_model(n_subjects, seed)
    method = FOCEMethod(interaction=False, maxeval=maxeval, print_interval=100_000, n_parallel=1)
    return profile_workload(
        "foce",
        lambda: _result_summary(method.estimate(pop, params), n_subjects),
        [
            (FOCEMethod, "_inner_loop", "foce.inner_loop"),
            (FOCEMethod, "_outer_ofv", "foce.outer_ofv"),
            (IndividualModel, "obj_eta", "individual.obj_eta"),
            (IndividualModel, "evaluate_observation_model", "individual.evaluate_observation_model"),
            (IndividualModel, "_evaluate_predictions", "individual._evaluate_predictions"),
        ],
        top_fn,
    )


def run_focei(n_subjects: int, seed: int, maxeval: int, top_fn: int) -> dict[str, Any]:
    pop, params = _build_population_model(n_subjects, seed)
    method = FOCEMethod(interaction=True, maxeval=maxeval, print_interval=100_000, n_parallel=1)
    return profile_workload(
        "focei",
        lambda: _result_summary(method.estimate(pop, params), n_subjects),
        [
            (FOCEMethod, "_inner_loop", "foce.inner_loop"),
            (FOCEMethod, "_outer_ofv", "foce.outer_ofv"),
            (IndividualModel, "obj_eta", "individual.obj_eta"),
            (IndividualModel, "evaluate_observation_model", "individual.evaluate_observation_model"),
            (IndividualModel, "_evaluate_predictions", "individual._evaluate_predictions"),
        ],
        top_fn,
    )


def run_saem(n_subjects: int, seed: int, k1: int, k2: int, top_fn: int) -> dict[str, Any]:
    pop, params = _build_population_model(n_subjects, seed)
    method = SAEMMethod(
        n_iter_phase1=k1,
        n_iter_phase2=k2,
        n_chains=1,
        seed=seed,
        print_interval=100_000,
        n_parallel=1,
    )
    # Note: SAEMMethod._e_step_one_subject is a @staticmethod; patching it via
    # timed_patch would convert it to an instance method (self injected), breaking
    # the argument count.  Instead we track work via IndividualModel.obj_eta,
    # which is the computational hot path inside _e_step_one_subject anyway.
    return profile_workload(
        "saem",
        lambda: _result_summary(method.estimate(pop, params), n_subjects),
        [
            (IndividualModel, "obj_eta", "individual.obj_eta"),
            (IndividualModel, "evaluate_observation_model", "individual.evaluate_observation_model"),
            (IndividualModel, "_evaluate_predictions", "individual._evaluate_predictions"),
        ],
        top_fn,
    )


def run_imp(n_subjects: int, seed: int, isample: int, maxeval: int, top_fn: int) -> dict[str, Any]:
    pop, params = _build_population_model(n_subjects, seed)
    method = IMPMethod(isample=isample, maxeval=maxeval, seed=seed, n_parallel=1)
    return profile_workload(
        "imp",
        lambda: _result_summary(method.estimate(pop, params), n_subjects),
        [
            (IMPMethod, "_compute_imp_ofv", "imp.compute_imp_ofv"),
            (IMPMethod, "_importance_sample", "imp.importance_sample"),
            (IndividualModel, "obj_eta", "individual.obj_eta"),
            (IndividualModel, "evaluate_observation_model", "individual.evaluate_observation_model"),
        ],
        top_fn,
    )


def run_impmap(
    n_subjects: int,
    seed: int,
    isample: int,
    maxeval: int,
    top_fn: int,
) -> dict[str, Any]:
    pop, params = _build_population_model(n_subjects, seed)
    method = IMPMethod(
        isample=isample,
        maxeval=maxeval,
        seed=seed,
        n_parallel=1,
        is_map=True,
    )
    return profile_workload(
        "impmap",
        lambda: _result_summary(method.estimate(pop, params), n_subjects),
        [
            (IMPMethod, "_warm_start_with_focei", "imp.warm_start_with_focei"),
            (IMPMethod, "_compute_imp_ofv", "imp.compute_imp_ofv"),
            (IMPMethod, "_importance_sample", "imp.importance_sample"),
            (FOCEMethod, "_inner_loop", "foce.inner_loop"),
            (FOCEMethod, "_outer_ofv", "foce.outer_ofv"),
            (IndividualModel, "obj_eta", "individual.obj_eta"),
            (IndividualModel, "evaluate_observation_model", "individual.evaluate_observation_model"),
        ],
        top_fn,
    )


def run_bayes_laplace(
    n_subjects: int, seed: int, n_samples: int, top_fn: int
) -> dict[str, Any]:
    pop, params = _build_population_model(n_subjects, seed)
    method = BAYESMethod(
        backend="laplace",
        n_samples=n_samples,
        n_chains=1,
        seed=seed,
        prior_sd_theta=2.0,
        maxeval=30,
    )
    return profile_workload(
        "bayes_laplace",
        lambda: {
            **_result_summary(method.estimate(pop, params), n_subjects),
            "backend_used": "laplace",
        },
        [
            (BAYESMethod, "_estimate_laplace", "bayes.estimate_laplace"),
            (FOCEMethod, "_inner_loop", "foce.inner_loop"),
            (FOCEMethod, "_outer_ofv", "foce.outer_ofv"),
            (IndividualModel, "evaluate_observation_model", "individual.evaluate_observation_model"),
        ],
        top_fn,
    )


def run_bayes_nuts(
    n_subjects: int,
    seed: int,
    n_samples: int,
    tune: int,
    top_fn: int,
) -> dict[str, Any]:
    pop, params = _build_population_model(n_subjects, seed)
    method = BAYESMethod(
        backend="nuts",
        n_samples=n_samples,
        n_chains=2,
        tune=tune,
        seed=seed,
        prior_sd_theta=2.0,
    )
    return profile_workload(
        "bayes_nuts",
        lambda: {
            **_result_summary((result := method.estimate(pop, params)), n_subjects),
            "backend_used": "nuts",
            "diagnostics": result.diagnostics,
        },
        [
            (BAYESMethod, "_estimate_nuts", "bayes.estimate_nuts"),
            (NUTSSampler, "sample", "nuts.sample"),
            (FOCEMethod, "_inner_loop", "foce.inner_loop"),
            (FOCEMethod, "_outer_ofv", "foce.outer_ofv"),
        ],
        top_fn,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark FO / FOCE / FOCEI / SAEM / IMP / IMPMAP estimation methods."
    )
    p.add_argument(
        "--workloads",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "fo",
            "foce",
            "focei",
            "saem",
            "imp",
            "impmap",
            "bayes_laplace",
            "bayes_nuts",
        ],
    )
    p.add_argument("--n-subjects", type=int, default=_DEFAULT_N_SUBJECTS)
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    p.add_argument("--fo-maxeval", type=int, default=_FO_MAXEVAL)
    p.add_argument("--foce-maxeval", type=int, default=_FOCE_MAXEVAL)
    p.add_argument("--focei-maxeval", type=int, default=_FOCEI_MAXEVAL)
    p.add_argument("--saem-k1", type=int, default=_SAEM_K1)
    p.add_argument("--saem-k2", type=int, default=_SAEM_K2)
    p.add_argument("--imp-isample", type=int, default=_IMP_ISAMPLE)
    p.add_argument("--imp-maxeval", type=int, default=_IMP_MAXEVAL)
    p.add_argument("--bayes-samples", type=int, default=_BAYES_N_SAMPLES)
    p.add_argument("--bayes-tune", type=int, default=_BAYES_TUNE)
    p.add_argument("--top-functions", type=int, default=_DEFAULT_TOP_FN)
    p.add_argument("--json-out", type=Path, default=_DEFAULT_JSON_OUT)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workloads: set[str] = (
        {"fo", "foce", "focei", "saem", "imp", "impmap", "bayes_laplace", "bayes_nuts"}
        if "all" in args.workloads else set(args.workloads)
    )

    results: dict[str, Any] = {
        "metadata": {
            "date": date.today().isoformat(),
            "seed": int(args.seed),
            "workloads": sorted(workloads),
            "parameters": {
                "n_subjects": int(args.n_subjects),
                "fo_maxeval": int(args.fo_maxeval),
                "foce_maxeval": int(args.foce_maxeval),
                "focei_maxeval": int(args.focei_maxeval),
                "saem_k1": int(args.saem_k1),
                "saem_k2": int(args.saem_k2),
                "imp_isample": int(args.imp_isample),
                "imp_maxeval": int(args.imp_maxeval),
                "bayes_samples": int(args.bayes_samples),
                "bayes_tune": int(args.bayes_tune),
                "top_functions": int(args.top_functions),
                "model": "1-cmt oral PK (ADVAN2/TRANS2), proportional error",
                "pk_params": "KA=1.5, CL=2.8, V=32.9 (pop), 7 obs/subject",
            },
        }
    }

    if "fo" in workloads:
        print(f"[fo]    running FO  (N={args.n_subjects}, maxeval={args.fo_maxeval})…",
              file=sys.stderr)
        results["fo"] = run_fo(args.n_subjects, args.seed, args.fo_maxeval, args.top_functions)

    if "foce" in workloads:
        print(f"[foce]  running FOCE (N={args.n_subjects}, maxeval={args.foce_maxeval})…",
              file=sys.stderr)
        results["foce"] = run_foce(
            args.n_subjects, args.seed, args.foce_maxeval, args.top_functions
        )

    if "focei" in workloads:
        print(f"[focei] running FOCEI (N={args.n_subjects}, maxeval={args.focei_maxeval})…",
              file=sys.stderr)
        results["focei"] = run_focei(
            args.n_subjects, args.seed, args.focei_maxeval, args.top_functions
        )

    if "saem" in workloads:
        print(
            f"[saem]  running SAEM (N={args.n_subjects}, K1={args.saem_k1}, K2={args.saem_k2})…",
            file=sys.stderr,
        )
        results["saem"] = run_saem(
            args.n_subjects, args.seed, args.saem_k1, args.saem_k2, args.top_functions
        )

    if "imp" in workloads:
        print(
            f"[imp]   running IMP (N={args.n_subjects}, isample={args.imp_isample}, maxeval={args.imp_maxeval})…",
            file=sys.stderr,
        )
        results["imp"] = run_imp(
            args.n_subjects,
            args.seed,
            args.imp_isample,
            args.imp_maxeval,
            args.top_functions,
        )

    if "impmap" in workloads:
        print(
            f"[impmap] running IMPMAP (N={args.n_subjects}, isample={args.imp_isample}, maxeval={args.imp_maxeval})…",
            file=sys.stderr,
        )
        results["impmap"] = run_impmap(
            args.n_subjects,
            args.seed,
            args.imp_isample,
            args.imp_maxeval,
            args.top_functions,
        )

    if "bayes_laplace" in workloads:
        print(
            f"[bayes_laplace] running BAYES(Laplace) (N={args.n_subjects}, samples={args.bayes_samples})…",
            file=sys.stderr,
        )
        results["bayes_laplace"] = run_bayes_laplace(
            args.n_subjects,
            args.seed,
            args.bayes_samples,
            args.top_functions,
        )

    if "bayes_nuts" in workloads:
        print(
            f"[bayes_nuts] running BAYES(NUTS) (N={args.n_subjects}, samples={args.bayes_samples}, tune={args.bayes_tune})…",
            file=sys.stderr,
        )
        results["bayes_nuts"] = run_bayes_nuts(
            args.n_subjects,
            args.seed,
            args.bayes_samples,
            args.bayes_tune,
            args.top_functions,
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.json_out}", file=sys.stderr)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
