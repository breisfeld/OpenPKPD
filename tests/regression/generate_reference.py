"""
Generate self-consistent golden reference files for regression tests.

Run once to populate tests/regression/reference_runs/:
    uv run python tests/regression/generate_reference.py

The JSON files are committed to the repo and serve as the regression baseline.
No NONMEM license required — baseline is self-consistent within openpkpd.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

# Ensure the repo root and src package are importable when run directly
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from openpkpd.data.dataset import NONMEMDataset  # noqa: E402
from openpkpd.estimation.bayes import BAYESMethod  # noqa: E402
from openpkpd.estimation.foce import FOCEMethod  # noqa: E402
from openpkpd.estimation.imp import IMPMethod  # noqa: E402
from openpkpd.estimation.laplacian import LaplacianMethod  # noqa: E402
from openpkpd.estimation.nonparametric import NonparametricMethod, NonparametricResult  # noqa: E402
from openpkpd.estimation.saem import SAEMMethod  # noqa: E402
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec  # noqa: E402
from openpkpd.model.population import PopulationModel  # noqa: E402
from openpkpd.nca.nca import NCAEngine  # noqa: E402
from openpkpd.pk.analytical.advan2 import ADVAN2  # noqa: E402
from openpkpd.simulation.engine import SimulationEngine  # noqa: E402
from openpkpd.simulation.npde import NPDEEngine  # noqa: E402
from openpkpd.simulation.vpc import VPCEngine  # noqa: E402
from tests.regression.diagnostic_helpers import (  # noqa: E402
    build_npc_result,
    build_pop_model_and_result,
    build_sse_result,
    fraction_obs_p50_in_sim_range,
    make_mock_npde_engine,
    theophylline_nca_profile,
)
from tests.regression.pd_model_helpers import (  # noqa: E402
    fit_direct_pd_case,
    fit_indirect_pd_case,
    fit_joint_pkpd_case,
    fit_mechanistic_pd_case,
    fit_population_pd_case,
    fit_sequential_from_pk_pd_case,
    fit_sequential_multi_subject_from_pk_pd_case,
    fit_sequential_pd_case,
    fit_turnover_pd_case,
)

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference_runs")


def _reference_metadata(
    *,
    name: str,
    method: str,
    seed: int | None = None,
    dataset: str | None = None,
    validation_level: str | None = None,
    external_method_references: list[str] | None = None,
    validation_note: str | None = None,
) -> dict:
    meta = {
        "name": name,
        "reference_kind": "internal-baseline",
        "externally_validated": False,
        "dataset": dataset or "synthetic_theophylline_like",
        "method": method,
        "seed": seed,
        "notes": (
            "Self-consistent openpkpd regression baseline. Useful for drift detection "
            "but not yet an external scientific reference."
        ),
    }
    if validation_level is not None:
        meta["validation_level"] = validation_level
    if external_method_references is not None:
        meta["external_method_references"] = list(external_method_references)
    if validation_note is not None:
        meta["validation_note"] = validation_note
    return meta


def _diagnostic_payload(*, metadata: dict, **values) -> dict:
    payload = {"_meta": metadata}
    payload.update(values)
    return payload


# ---------------------------------------------------------------------------
# Dataset builder (identical to test_regression.py)
# ---------------------------------------------------------------------------


def _build_theophylline_dataset() -> NONMEMDataset:
    rng = np.random.default_rng(42)
    ka_pop = 1.5
    cl_pop = 2.8
    v_pop = 32.9
    dose = 320.0
    obs_times = np.array([0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0, 24.0])

    rows = []
    for sid in range(1, 13):
        eta_ka = rng.normal(0, 0.3)
        eta_cl = rng.normal(0, 0.2)
        eta_v = rng.normal(0, 0.15)
        ka = ka_pop * math.exp(eta_ka)
        cl = cl_pop * math.exp(eta_cl)
        v = v_pop * math.exp(eta_v)
        k = cl / v

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
            dv = max(c * (1 + eps), 0.01)
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

    df = pd.DataFrame(rows)
    return NONMEMDataset.from_dataframe(df)


def _build_params() -> ParameterSet:
    theta_specs = [
        ThetaSpec(init=1.5, lower=0.5, upper=8.0),
        ThetaSpec(init=3.0, lower=0.5, upper=15.0),
        ThetaSpec(init=35.0, lower=10.0, upper=80.0),
    ]
    omega_specs = [OmegaSpec(block_size=1, values=[0.09])]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]
    return ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)


def _build_pop_model(dataset: NONMEMDataset, params: ParameterSet) -> PopulationModel:
    return PopulationModel(
        dataset=dataset,
        pk_subroutine=ADVAN2(),
        params=params,
        trans=2,
        advan=2,
    )


def _result_to_dict(result, *, metadata: dict | None = None) -> dict:
    payload = {
        "ofv": float(result.ofv),
        "theta": [float(x) for x in result.theta_final],
        "omega_diag": [float(x) for x in np.diag(result.omega_final)],
        "sigma_diag": [float(x) for x in np.diag(result.sigma_final)],
        "converged": bool(result.converged),
    }
    if metadata is not None:
        payload["_meta"] = metadata
    if hasattr(result, "posterior_ci_lo"):
        payload["posterior_ci_lo"] = [float(x) for x in result.posterior_ci_lo]
        payload["posterior_ci_hi"] = [float(x) for x in result.posterior_ci_hi]
        payload["backend_used"] = str(getattr(result, "backend_used", ""))
    if isinstance(result, NonparametricResult):
        payload["support_weights"] = [float(x) for x in result.support_weights]
        payload["empirical_mean"] = [float(x) for x in result.empirical_mean()]
        payload["empirical_variance"] = [float(x) for x in result.empirical_variance()]
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    dataset = _build_theophylline_dataset()

    # ── FOCE ──────────────────────────────────────────────────────────────────
    print("Running FOCE...")
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    foce = FOCEMethod(interaction=False, maxeval=400, print_interval=100)
    foce_result = foce.estimate(pop_model, params)
    foce_dict = _result_to_dict(
        foce_result,
        metadata=_reference_metadata(name="theophylline_foce", method="FOCE"),
    )
    path = os.path.join(REFERENCE_DIR, "theophylline_foce.json")
    with open(path, "w") as fh:
        json.dump(foce_dict, fh, indent=2)
    print(f"  OFV={foce_dict['ofv']:.4f}  converged={foce_dict['converged']}  -> {path}")

    # ── Laplacian ─────────────────────────────────────────────────────────────
    print("Running Laplacian...")
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    lap = LaplacianMethod(interaction=True, maxeval=400, print_interval=100)
    lap_result = lap.estimate(pop_model, params)
    lap_dict = _result_to_dict(
        lap_result,
        metadata=_reference_metadata(name="theophylline_laplacian", method="LAPLACIAN"),
    )
    path = os.path.join(REFERENCE_DIR, "theophylline_laplacian.json")
    with open(path, "w") as fh:
        json.dump(lap_dict, fh, indent=2)
    print(f"  OFV={lap_dict['ofv']:.4f}  converged={lap_dict['converged']}  -> {path}")

    # ── SAEM (reduced iterations for CI speed) ────────────────────────────────
    print("Running SAEM (K1=80, K2=40)...")
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    saem = SAEMMethod(n_iter_phase1=80, n_iter_phase2=40, seed=42, print_interval=20)
    saem_result = saem.estimate(pop_model, params)
    saem_dict = _result_to_dict(
        saem_result,
        metadata=_reference_metadata(name="theophylline_saem", method="SAEM", seed=42),
    )
    path = os.path.join(REFERENCE_DIR, "theophylline_saem.json")
    with open(path, "w") as fh:
        json.dump(saem_dict, fh, indent=2)
    print(f"  OFV={saem_dict['ofv']:.4f}  converged={saem_dict['converged']}  -> {path}")

    # ── IMP ───────────────────────────────────────────────────────────────────
    print("Running IMP...")
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    imp = IMPMethod(isample=200, maxeval=150, seed=42, print_interval=9999)
    imp_result = imp.estimate(pop_model, params)
    imp_dict = _result_to_dict(
        imp_result,
        metadata=_reference_metadata(name="theophylline_imp", method="IMP", seed=42),
    )
    path = os.path.join(REFERENCE_DIR, "theophylline_imp.json")
    with open(path, "w") as fh:
        json.dump(imp_dict, fh, indent=2)
    print(f"  OFV={imp_dict['ofv']:.4f}  converged={imp_dict['converged']}  -> {path}")

    print("Running IMPMAP...")
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    impmap = IMPMethod(isample=300, maxeval=50, seed=42, is_map=True, print_interval=9999)
    impmap_result = impmap.estimate(pop_model, params)
    impmap_dict = _result_to_dict(
        impmap_result,
        metadata=_reference_metadata(name="theophylline_impmap", method="IMPMAP", seed=42),
    )
    path = os.path.join(REFERENCE_DIR, "theophylline_impmap.json")
    with open(path, "w") as fh:
        json.dump(impmap_dict, fh, indent=2)
    print(f"  OFV={impmap_dict['ofv']:.4f}  converged={impmap_dict['converged']}  -> {path}")

    # ── BAYES (Laplace fallback) ──────────────────────────────────────────────
    print("Running BAYES (Laplace)...")
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    bayes = BAYESMethod(backend="laplace", n_samples=200, seed=42)
    bayes_result = bayes.estimate(pop_model, params)
    bayes_dict = _result_to_dict(
        bayes_result,
        metadata=_reference_metadata(
            name="theophylline_bayes_laplace", method="BAYES-laplace", seed=42
        ),
    )
    path = os.path.join(REFERENCE_DIR, "theophylline_bayes_laplace.json")
    with open(path, "w") as fh:
        json.dump(bayes_dict, fh, indent=2)
    print(f"  OFV={bayes_dict['ofv']:.4f}  converged={bayes_dict['converged']}  -> {path}")

    # ── Nonparametric ─────────────────────────────────────────────────────────
    print("Running nonparametric...")
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    nonparam = NonparametricMethod(base_method="FOCE", max_iter=60, maxeval=250, print_interval=999)
    nonparam_result = nonparam.estimate(pop_model, params)
    nonparam_dict = _result_to_dict(
        nonparam_result,
        metadata=_reference_metadata(
            name="theophylline_nonparametric", method="NONPARAMETRIC", seed=42
        ),
    )
    path = os.path.join(REFERENCE_DIR, "theophylline_nonparametric.json")
    with open(path, "w") as fh:
        json.dump(nonparam_dict, fh, indent=2)
    print(f"  OFV={nonparam_dict['ofv']:.4f}  converged={nonparam_dict['converged']}  -> {path}")

    # ── Direct PD regression baselines ────────────────────────────────────────
    for case_name in ["emax_direct_pd", "hill_direct_pd"]:
        print(f"Running {case_name}...")
        case, _data, result = fit_direct_pd_case(case_name)
        payload = _diagnostic_payload(
            metadata=_reference_metadata(
                name=case_name,
                method=f"{case['model_cls'].__name__}.fit",
                seed=case["seed"],
                dataset=case["dataset"],
                validation_level="fixed synthetic direct PD regression benchmark",
                validation_note="docs/user_guide/analysis_validation_gaps.md",
            ),
            true_params={k: float(v) for k, v in case["true_params"].items()},
            fit_init={k: float(v) for k, v in case["initial_params"].items()},
            params={k: float(v) for k, v in result.params.items()},
            ofv=float(result.ofv),
            aic=float(result.aic),
            converged=bool(result.converged),
            predicted=[float(x) for x in result.predicted],
        )
        path = os.path.join(REFERENCE_DIR, f"{case_name}.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  OFV={payload['ofv']:.4f} converged={payload['converged']} -> {path}")

    # ── Mechanistic / population PD regression baselines ─────────────────────
    print("Running effect_compartment_pd...")
    case, _data, result = fit_mechanistic_pd_case("effect_compartment_pd")
    payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="effect_compartment_pd",
            method=f"{case['model_cls'].__name__}.fit",
            seed=case["seed"],
            dataset=case["dataset"],
            validation_level="fixed synthetic mechanistic PD regression benchmark",
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        true_params={k: float(v) for k, v in case["true_params"].items()},
        fit_init={k: float(v) for k, v in case["initial_params"].items()},
        params={k: float(v) for k, v in result.params.items()},
        ofv=float(result.ofv),
        aic=float(result.aic),
        converged=bool(result.converged),
        predicted=[float(x) for x in result.predicted],
    )
    path = os.path.join(REFERENCE_DIR, "effect_compartment_pd.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  OFV={payload['ofv']:.4f} converged={payload['converged']} -> {path}")

    print("Running placebo_response_pd...")
    case, _data, result = fit_mechanistic_pd_case("placebo_response_pd")
    payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="placebo_response_pd",
            method=f"{case['model_cls'].__name__}.fit",
            seed=case["seed"],
            dataset=case["dataset"],
            validation_level="fixed synthetic placebo-response regression benchmark",
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        true_params={k: float(v) for k, v in case["true_params"].items()},
        fit_init={k: float(v) for k, v in case["initial_params"].items()},
        params={k: float(v) for k, v in result.params.items()},
        ofv=float(result.ofv),
        aic=float(result.aic),
        converged=bool(result.converged),
        predicted=[float(x) for x in result.predicted],
    )
    path = os.path.join(REFERENCE_DIR, "placebo_response_pd.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  OFV={payload['ofv']:.4f} converged={payload['converged']} -> {path}")

    print("Running tumor_growth_inhibition_pd...")
    case, _data, result = fit_mechanistic_pd_case("tumor_growth_inhibition_pd")
    payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="tumor_growth_inhibition_pd",
            method=f"{case['model_cls'].__name__}.fit",
            seed=case["seed"],
            dataset=case["dataset"],
            validation_level="fixed synthetic tumor-growth inhibition regression benchmark",
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        true_params={k: float(v) for k, v in case["true_params"].items()},
        fit_init={k: float(v) for k, v in case["initial_params"].items()},
        params={k: float(v) for k, v in result.params.items()},
        ofv=float(result.ofv),
        aic=float(result.aic),
        converged=bool(result.converged),
        predicted=[float(x) for x in result.predicted],
    )
    path = os.path.join(REFERENCE_DIR, "tumor_growth_inhibition_pd.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  OFV={payload['ofv']:.4f} converged={payload['converged']} -> {path}")

    print("Running turnover_pd...")
    case, _data, result = fit_turnover_pd_case("turnover_pd")
    payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="turnover_pd",
            method="TurnoverModel.fit",
            seed=case["seed"],
            dataset=case["dataset"],
            validation_level="fixed synthetic turnover regression benchmark",
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        true_params={k: float(v) for k, v in case["true_params"].items()},
        fit_init={k: float(v) for k, v in case["initial_params"].items()},
        params={k: float(v) for k, v in result.params.items()},
        ofv=float(result.ofv),
        aic=float(result.aic),
        converged=bool(result.converged),
        predicted=[float(x) for x in result.predicted],
    )
    path = os.path.join(REFERENCE_DIR, "turnover_pd.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  OFV={payload['ofv']:.4f} converged={payload['converged']} -> {path}")

    print("Running population_emax_pd...")
    case, _subjects, result = fit_population_pd_case("population_emax_pd")
    payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="population_emax_pd",
            method="PopulationPDModel.estimate",
            seed=case["seed"],
            dataset=case["dataset"],
            validation_level="fixed synthetic population PD regression benchmark",
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        true_theta={k: float(v) for k, v in case["true_theta"].items()},
        true_omega=float(case["true_omega"]),
        true_sigma2=float(case["true_sigma2"]),
        theta={k: float(v) for k, v in result.theta.items()},
        omega=[[float(x) for x in row] for row in result.omega.tolist()],
        sigma2=float(result.sigma2),
        ofv=float(result.ofv),
        aic=float(result.aic),
        converged=bool(result.converged),
        post_hoc_etas={
            str(k): [float(x) for x in v.tolist()] for k, v in result.post_hoc_etas.items()
        },
    )
    path = os.path.join(REFERENCE_DIR, "population_emax_pd.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  OFV={payload['ofv']:.4f} converged={payload['converged']} -> {path}")

    print("Running indirect_response_pd...")
    case, _data, result = fit_indirect_pd_case("indirect_response_pd")
    payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="indirect_response_pd",
            method="IndirectResponseModel.fit",
            seed=case["seed"],
            dataset=case["dataset"],
            validation_level="fixed synthetic indirect-response regression benchmark",
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        true_params={k: float(v) for k, v in case["true_params"].items()},
        fit_init={k: float(v) for k, v in case["initial_params"].items()},
        params={k: float(v) for k, v in result.params.items()},
        ofv=float(result.ofv),
        aic=float(result.aic),
        converged=bool(result.converged),
        predicted=[float(x) for x in result.predicted],
    )
    path = os.path.join(REFERENCE_DIR, "indirect_response_pd.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  OFV={payload['ofv']:.4f} converged={payload['converged']} -> {path}")

    print("Running sequential_emax_pd...")
    case, _data, result = fit_sequential_pd_case("sequential_emax_pd")
    payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="sequential_emax_pd",
            method="SequentialPKPDWorkflow.fit_pd",
            seed=case["seed"],
            dataset=case["dataset"],
            validation_level="fixed synthetic sequential PK/PD regression benchmark",
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        true_params={k: float(v) for k, v in case["true_params"].items()},
        fit_init={k: float(v) for k, v in case["initial_params"].items()},
        params={k: float(v) for k, v in result.params.items()},
        ofv=float(result.ofv),
        aic=float(result.aic),
        converged=bool(result.converged),
        predicted=[float(x) for x in result.predicted],
    )
    path = os.path.join(REFERENCE_DIR, "sequential_emax_pd.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  OFV={payload['ofv']:.4f} converged={payload['converged']} -> {path}")

    print("Running sequential_emax_from_pk_pd...")
    case, _data, result = fit_sequential_from_pk_pd_case("sequential_emax_from_pk_pd")
    payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="sequential_emax_from_pk_pd",
            method="SequentialPKPDWorkflow.fit_pd",
            seed=case["pd_seed"],
            dataset=case["dataset"],
            validation_level="fixed synthetic sequential PK-derived regression benchmark",
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        true_params={k: float(v) for k, v in case["true_params"].items()},
        fit_init={k: float(v) for k, v in case["initial_params"].items()},
        params={k: float(v) for k, v in result.params.items()},
        ofv=float(result.ofv),
        aic=float(result.aic),
        converged=bool(result.converged),
        predicted=[float(x) for x in result.predicted],
    )
    path = os.path.join(REFERENCE_DIR, "sequential_emax_from_pk_pd.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  OFV={payload['ofv']:.4f} converged={payload['converged']} -> {path}")

    print("Running sequential_multi_subject_from_pk_pd...")
    case, _data_by_sid, results = fit_sequential_multi_subject_from_pk_pd_case(
        "sequential_multi_subject_from_pk_pd"
    )
    payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="sequential_multi_subject_from_pk_pd",
            method="SequentialPKPDWorkflow.fit_pd",
            seed=case["pd_seed"],
            dataset=case["dataset"],
            validation_level="fixed synthetic multi-subject sequential PK-derived regression benchmark",
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        true_params={k: float(v) for k, v in case["true_params"].items()},
        fit_init={k: float(v) for k, v in case["initial_params"].items()},
        post_hoc_etas={
            str(sid): [float(x) for x in eta] for sid, eta in case["post_hoc_etas"].items()
        },
        subjects={
            str(sid): {
                "params": {k: float(v) for k, v in result.params.items()},
                "ofv": float(result.ofv),
                "aic": float(result.aic),
                "converged": bool(result.converged),
                "predicted": [float(x) for x in result.predicted],
            }
            for sid, result in results.items()
        },
    )
    path = os.path.join(REFERENCE_DIR, "sequential_multi_subject_from_pk_pd.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  converged={all(s['converged'] for s in payload['subjects'].values())} -> {path}")

    print("Running joint_emax_pkpd...")
    case, _built, result, predicted, pk_f = fit_joint_pkpd_case("joint_emax_pkpd")
    payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="joint_emax_pkpd",
            method="ModelBuilder.fit (FO)",
            seed=case["seed"],
            dataset=case["dataset"],
            validation_level="fixed synthetic joint PK/PD regression benchmark",
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        true_params={k: float(v) for k, v in case["true_params"].items()},
        true_omega_diag={k: float(v) for k, v in case["true_omega_diag"].items()},
        fit_init={k: float(v) for k, v in case["initial_params"].items()},
        omega_init=[float(x) for x in case["omega_init"]],
        n_subjects=int(case["n_subjects"]),
        params={
            name: float(result.theta_final[idx]) for idx, name in enumerate(case["param_names"])
        },
        omega_diag={
            name: float(result.omega_final[idx, idx])
            for idx, name in enumerate(case["omega_param_names"])
        },
        ofv=float(result.ofv),
        aic=float(result.aic),
        converged=bool(result.converged),
        predicted=[float(x) for x in predicted],
        pk_f=[float(x) for x in pk_f],
    )
    path = os.path.join(REFERENCE_DIR, "joint_emax_pkpd.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  OFV={payload['ofv']:.4f} converged={payload['converged']} -> {path}")

    # ── VPC diagnostic summary ────────────────────────────────────────────────
    print("Running VPC diagnostic baseline...")
    pop_model, est_result = build_pop_model_and_result(n_subjects=24, seed=7)
    vpc = VPCEngine(SimulationEngine(pop_model, est_result, seed=7)).compute(
        n_replicates=100, n_bins=8
    )
    vpc_payload = _diagnostic_payload(
        metadata=_reference_metadata(name="diagnostic_vpc", method="VPC", seed=7),
        coverage_fraction=fraction_obs_p50_in_sim_range(vpc),
        obs_p50=[float(x) for x in vpc.obs_percentiles["p50"]],
        sim_p50_mid=[float(x) for x in vpc.sim_percentiles["p50_mid"]],
        sim_p5_lo=[float(x) for x in vpc.sim_percentiles["p5_lo"]],
        sim_p95_hi=[float(x) for x in vpc.sim_percentiles["p95_hi"]],
    )
    path = os.path.join(REFERENCE_DIR, "diagnostic_vpc.json")
    with open(path, "w") as fh:
        json.dump(vpc_payload, fh, indent=2)
    print(f"  coverage={vpc_payload['coverage_fraction']:.4f} -> {path}")

    # ── NPC diagnostic summary ────────────────────────────────────────────────
    print("Running NPC diagnostic baseline...")
    npc_result = build_npc_result(n_subjects=24, seed=7, n_replicates=100, n_bins=8)
    npc_payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="diagnostic_npc",
            method="NPC",
            seed=7,
            validation_level="synthetic calibration and drift benchmark",
            external_method_references=[
                "Holford NHG (2005) The Visual Predictive Check — superiority to standard diagnostic plots. PAGE 14 (Abstract 738).",
            ],
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        obs_below_lower=float(npc_result.obs_below_lower),
        obs_within=float(npc_result.obs_within),
        obs_above_upper=float(npc_result.obs_above_upper),
        expected_within=float(npc_result.expected_within),
        n_observations=int(npc_result.n_observations),
        binned_t_mid=[float(x) for x in npc_result.binned["t_mid"]],
        binned_obs_within=[float(x) for x in npc_result.binned["obs_within"]],
        binned_obs_below_lower=[float(x) for x in npc_result.binned["obs_below_lower"]],
        binned_obs_above_upper=[float(x) for x in npc_result.binned["obs_above_upper"]],
    )
    path = os.path.join(REFERENCE_DIR, "diagnostic_npc.json")
    with open(path, "w") as fh:
        json.dump(npc_payload, fh, indent=2)
    print(f"  within={npc_payload['obs_within']:.4f} n={npc_payload['n_observations']} -> {path}")

    # ── NPDE diagnostic summary ───────────────────────────────────────────────
    print("Running NPDE diagnostic baseline...")
    npde_result = NPDEEngine(
        make_mock_npde_engine(n_subjects=20, n_obs=8, n_replicates=200, noise_sd=0.5, seed=7)
    ).compute(
        n_replicates=200,
        seed=7,
    )
    npde_vals = npde_result.df["NPDE"].dropna().to_numpy(dtype=float)
    npde_payload = _diagnostic_payload(
        metadata=_reference_metadata(name="diagnostic_npde", method="NPDE", seed=7),
        mean_npde=float(npde_result.mean_npde),
        var_npde=float(npde_result.var_npde),
        sw_stat=float(npde_result.sw_stat),
        sw_pvalue=float(npde_result.sw_pvalue),
        quantiles=[float(x) for x in np.quantile(npde_vals, [0.05, 0.5, 0.95])],
    )
    path = os.path.join(REFERENCE_DIR, "diagnostic_npde.json")
    with open(path, "w") as fh:
        json.dump(npde_payload, fh, indent=2)
    print(f"  mean={npde_payload['mean_npde']:.4f} var={npde_payload['var_npde']:.4f} -> {path}")

    # ── NCA diagnostic summary ────────────────────────────────────────────────
    print("Running NCA diagnostic baseline...")
    times, conc, dose = theophylline_nca_profile()
    nca_result = NCAEngine(auc_method="linear-log", min_points_lambda=4).compute_subject(
        times, conc, dose=dose, route="oral"
    )
    nca_payload = _diagnostic_payload(
        metadata=_reference_metadata(name="diagnostic_nca", method="NCA"),
        cmax=float(nca_result.cmax),
        tmax=float(nca_result.tmax),
        auc_last=float(nca_result.auc_last),
        auc_inf=float(nca_result.auc_inf),
        lambda_z=float(nca_result.lambda_z),
        t_half=float(nca_result.t_half),
        cl_f=float(nca_result.cl_f),
        vz_f=float(nca_result.vz_f),
        mrt=float(nca_result.mrt),
    )
    path = os.path.join(REFERENCE_DIR, "diagnostic_nca.json")
    with open(path, "w") as fh:
        json.dump(nca_payload, fh, indent=2)
    print(f"  auc_inf={nca_payload['auc_inf']:.4f} t_half={nca_payload['t_half']:.4f} -> {path}")

    # ── SSE diagnostic summary ────────────────────────────────────────────────
    print("Running SSE diagnostic baseline...")
    sse_result = build_sse_result(
        n_subjects=8,
        data_seed=11,
        run_seed=11,
        n_replicates=4,
        estimation_method="FO",
    )
    sse_payload = _diagnostic_payload(
        metadata=_reference_metadata(
            name="diagnostic_sse",
            method="SSE",
            seed=11,
            validation_level="internal simulation-reestimation benchmark",
            external_method_references=[
                "Holford NHG et al. (2000) Simulation of clinical trials. Annu Rev Pharmacol Toxicol 40:209-234.",
            ],
            validation_note="docs/user_guide/analysis_validation_gaps.md",
        ),
        n_replicates=int(sse_result.n_replicates),
        convergence_rate=float(sse_result.convergence_rate),
        parameter_names=list(sse_result.parameter_names),
        true_values={k: float(v) for k, v in sse_result.true_values.items()},
        mean_estimates={k: float(v) for k, v in sse_result.estimates.mean().items()},
        bias={k: float(v) for k, v in sse_result.bias.items()},
        rmse={k: float(v) for k, v in sse_result.rmse.items()},
        coverage_95={k: float(v) for k, v in sse_result.coverage_95.items()},
    )
    path = os.path.join(REFERENCE_DIR, "diagnostic_sse.json")
    with open(path, "w") as fh:
        json.dump(sse_payload, fh, indent=2)
    print(f"  convergence={sse_payload['convergence_rate']:.4f} -> {path}")

    print("\nDone. Commit the files under tests/regression/reference_runs/")


if __name__ == "__main__":
    main()
