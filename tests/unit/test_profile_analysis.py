from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _load_profile_analysis_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "profile_analysis.py"
    spec = importlib.util.spec_from_file_location("profile_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_nca_dataset_has_expected_shape_and_columns() -> None:
    module = _load_profile_analysis_module()

    df = module.build_nca_dataset(n_subjects=3, seed=11)

    assert len(df) == 33
    assert {"ID", "TIME", "AMT", "DV", "EVID", "MDV"}.issubset(df.columns)
    assert (df.groupby("ID").size() == 11).all()


def test_run_profiles_selects_requested_workloads(monkeypatch) -> None:
    module = _load_profile_analysis_module()

    seen: list[str] = []

    class _FakeBuiltModel:
        population_model = object()

        def fit(self):
            return object()

    def fake_profile_workload(name, func, patches, limit):
        seen.append(name)
        return {
            "name": name,
            "wall_seconds": 0.0,
            "result": {"ok": True},
            "stage_totals": {},
            "top_functions": [],
        }

    monkeypatch.setattr(module, "build_fit_model", lambda *args: _FakeBuiltModel())
    monkeypatch.setattr(
        module, "build_simulation_model_and_result", lambda *args: (object(), object())
    )
    monkeypatch.setattr(module, "build_nca_dataset", lambda *args: pd.DataFrame())
    monkeypatch.setattr(module, "profile_workload", fake_profile_workload)

    args = SimpleNamespace(
        workloads=["diagnostics", "nca"],
        fit_subjects=6,
        fit_maxeval=12,
        covariate_subjects=140,
        sim_subjects=24,
        npde_simulations=500,
        vpc_replicates=500,
        n_bins=8,
        nca_subjects=2000,
        seed=77,
        top_functions=15,
        json_out=Path("artifacts/profiling/test.json"),
    )

    results = module.run_profiles(args)

    assert seen == ["diagnostics", "nca"]
    assert set(results) == {"metadata", "diagnostics", "nca"}
    assert results["metadata"]["parameters"]["fit_subjects"] == 6


def test_run_profiles_selects_covariate_diagnostics_workload(monkeypatch) -> None:
    module = _load_profile_analysis_module()

    monkeypatch.setattr(
        module,
        "profile_covariate_diagnostics_comparison",
        lambda n_subjects, seed, top_functions: {
            "n_subjects": n_subjects,
            "seed": seed,
            "top_functions": top_functions,
        },
    )

    args = SimpleNamespace(
        workloads=["diagnostics_covariate"],
        fit_subjects=6,
        fit_maxeval=12,
        covariate_subjects=140,
        sim_subjects=24,
        npde_simulations=500,
        vpc_replicates=500,
        n_bins=8,
        nca_subjects=2000,
        seed=77,
        top_functions=15,
        json_out=Path("artifacts/profiling/test.json"),
    )

    results = module.run_profiles(args)

    assert set(results) == {"metadata", "diagnostics_covariate"}
    assert results["diagnostics_covariate"] == {
        "n_subjects": 140,
        "seed": 77,
        "top_functions": 15,
    }
