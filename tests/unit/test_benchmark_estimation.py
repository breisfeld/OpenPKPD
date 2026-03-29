from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_benchmark_estimation_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "benchmark_estimation.py"
    spec = importlib.util.spec_from_file_location("benchmark_estimation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_writes_bayes_nuts_diagnostics(tmp_path, monkeypatch) -> None:
    module = _load_benchmark_estimation_module()
    json_out = tmp_path / "estimation.json"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda argv=None: SimpleNamespace(
            workloads=["bayes_nuts"],
            n_subjects=6,
            seed=42,
            fo_maxeval=500,
            foce_maxeval=300,
            focei_maxeval=200,
            saem_k1=150,
            saem_k2=100,
            imp_isample=80,
            imp_maxeval=8,
            bayes_samples=12,
            bayes_tune=8,
            top_functions=10,
            json_out=json_out,
        ),
    )
    monkeypatch.setattr(
        module,
        "run_bayes_nuts",
        lambda n_subjects, seed, n_samples, tune, top_fn: {
            "name": "bayes_nuts",
            "wall_seconds": 1.23,
            "result": {
                "converged": False,
                "ofv": float("nan"),
                "n_function_evals": 0,
                "n_subjects": n_subjects,
                "backend_used": "nuts",
                "diagnostics": {
                    "nuts": {
                        "n_chains": 2,
                        "n_samples_per_chain": n_samples,
                        "n_warmup_per_chain": tune,
                        "log_prob_calls": 99,
                    }
                },
            },
            "stage_totals": {},
            "top_functions": [],
        },
    )

    rc = module.main([])

    assert rc == 0
    payload = json.loads(json_out.read_text())
    assert payload["metadata"]["parameters"]["bayes_samples"] == 12
    assert payload["metadata"]["parameters"]["bayes_tune"] == 8
    assert payload["bayes_nuts"]["result"]["diagnostics"]["nuts"]["n_chains"] == 2
    assert payload["bayes_nuts"]["result"]["diagnostics"]["nuts"]["log_prob_calls"] == 99


def test_build_population_model_uses_compiled_symbolic_subjects() -> None:
    pytest.importorskip("sympy")
    from openpkpd.model.symbolic_eta import prewarm_symbolic_caches

    prewarm_symbolic_caches()
    module = _load_benchmark_estimation_module()

    pop, _params = module._build_population_model(n_subjects=4, seed=42)
    subjects = list(pop._individual_models.values())

    assert subjects
    assert all(subject.pk_callable is not None for subject in subjects)
    assert all(subject.error_callable is not None for subject in subjects)
    assert all(subject.supports_eta_objective_gradient() for subject in subjects)
    assert all(subject.supports_theta_data_objective_gradient() for subject in subjects)
