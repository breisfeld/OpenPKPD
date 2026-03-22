"""External-validation benchmark against published WinNonlin-backed Indometh NCA output."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from openpkpd.nca.nca import NCAEngine

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "indometh.csv")
REFERENCE_PATH = os.path.join(HERE, "reference", "winnonlin_indometh_nca.json")


def _load_reference() -> dict:
    with open(REFERENCE_PATH) as f:
        return json.load(f)


def _load_zero_start_indometh() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    zero_rows = pd.DataFrame(
        {
            "Subject": sorted(df["Subject"].unique()),
            "time": 0.0,
            "conc": 0.0,
        }
    )
    combined = pd.concat([zero_rows, df], ignore_index=True)
    return combined.sort_values(["Subject", "time"], kind="stable").reset_index(drop=True)


def _compute_zero_start_results(
    auc_method: str,
    *,
    route: str = "oral",
    infusion_duration: float | None = None,
) -> dict[str, dict[str, float]]:
    engine = NCAEngine(auc_method=auc_method, exclude_cmax=True)
    df = _load_zero_start_indometh()
    results: dict[str, dict[str, float]] = {}
    for subject_id, group in df.groupby("Subject", sort=True):
        params = engine.compute_subject(
            times=group["time"].to_numpy(float),
            conc=group["conc"].to_numpy(float),
            dose=25.0,
            subject_id=int(subject_id),
            route=route,
            infusion_duration=infusion_duration,
        )
        results[str(int(subject_id))] = {
            "r_squared": params.r_squared,
            "lambda_z": params.lambda_z,
            "t_half": params.t_half,
            "cmax": params.cmax,
            "tmax": params.tmax,
            "auc_last": params.auc_last,
            "aumc_last": params.aumc_last,
            "auc_inf": params.auc_inf,
            "aumc_inf": params.aumc_inf,
            "cl_f": params.cl_f,
            "vz_f": params.vz_f,
            "mrt": params.mrt,
        }
    return results


def _compute_iv_bolus_results(auc_method: str) -> dict[str, dict[str, float]]:
    engine = NCAEngine(auc_method=auc_method, exclude_cmax=True)
    df = pd.read_csv(DATA_PATH)
    results: dict[str, dict[str, float]] = {}
    for subject_id, group in df.groupby("Subject", sort=True):
        params = engine.compute_subject(
            times=group["time"].to_numpy(float),
            conc=group["conc"].to_numpy(float),
            dose=25.0,
            subject_id=int(subject_id),
            route="IV",
        )
        results[str(int(subject_id))] = {
            "r_squared": params.r_squared,
            "lambda_z": params.lambda_z,
            "t_half": params.t_half,
            "c0": params.c0,
            "cmax": params.cmax,
            "tmax": params.tmax,
            "auc_last": params.auc_last,
            "aumc_last": params.aumc_last,
            "auc_inf": params.auc_inf,
            "aumc_inf": params.aumc_inf,
            "cl_f": params.cl_f,
            "vz_f": params.vz_f,
            "mrt": params.mrt,
        }
    return results


@pytest.mark.external_validation
@pytest.mark.parametrize(
    ("scenario", "auc_method", "mode", "route", "infusion_duration"),
    [
        ("linear_zero_start_core", "linear-trapezoidal", "zero_start", "oral", None),
        ("log_zero_start_core", "linear-log", "zero_start", "oral", None),
        ("linear_iv_bolus_core", "linear-trapezoidal", "iv_bolus", "IV", None),
        ("log_iv_bolus_core", "linear-log", "iv_bolus", "IV", None),
        ("linear_iv_infusion_supported", "linear-trapezoidal", "zero_start", "infusion", 0.25),
        ("log_iv_infusion_supported", "linear-log", "zero_start", "infusion", 0.25),
        ("linear_extravascular_supported", "linear-trapezoidal", "zero_start", "oral", None),
        ("log_extravascular_supported", "linear-log", "zero_start", "oral", None),
    ],
)
def test_indometh_core_nca_tracks_published_winnonlin_reference(
    scenario: str,
    auc_method: str,
    mode: str,
    route: str,
    infusion_duration: float | None,
) -> None:
    reference = _load_reference()["scenarios"][scenario]["subjects"]
    if mode == "zero_start":
        observed = _compute_zero_start_results(
            auc_method,
            route=route,
            infusion_duration=infusion_duration,
        )
    else:
        observed = _compute_iv_bolus_results(auc_method)

    for subject_id, expected_metrics in reference.items():
        for metric_name, expected_value in expected_metrics.items():
            assert observed[subject_id][metric_name] == pytest.approx(expected_value, abs=1e-6)
