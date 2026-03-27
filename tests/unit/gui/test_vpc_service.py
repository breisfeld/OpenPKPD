"""Focused tests for GUI VPC job orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import JobContext
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.fit_service import FitService
from openpkpd_gui.services.vpc_service import VPCConfig, VPCRunResult, VPCService


def _cache_fit_context(
    service: FitService, workspace: Workspace, monkeypatch, *, fit_run_id: str
) -> None:
    class _FakeResult:
        warnings: list[str] = []
        converged = True
        ofv = 12.34
        method = "FOCE"

    workspace.runs = [RunRecord(workflow="fit", run_id=fit_run_id, status=RunStatus.SUCCEEDED)]
    monkeypatch.setattr(service, "_generate_output_artifacts", lambda *_a, **_k: [])
    service._fit_run_result_from_estimation(
        JobContext(lambda *_args: None),
        workspace,
        "VPC demo",
        _FakeResult(),
        "FOCE",
        None,
        fit_run_id,
        project_id=workspace.active_project.project_id,
        scenario_id=workspace.active_scenario.scenario_id,
        dataset_path=str(Path(workspace.root_path or ".") / "theo.csv"),
        population_model=object(),
    )


def test_create_job_requires_cached_fit_context() -> None:
    workspace = Workspace(name="VPC demo")

    with pytest.raises(ValueError, match="requires a successful fit"):
        VPCService().create_job(workspace, fit_service=FitService())


@pytest.mark.unit
def test_create_job_and_apply_outcome_write_fit_linked_vpc_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = Workspace(name="VPC demo", root_path=str(tmp_path), workspace_id="vpc-demo")
    fit_service = FitService()
    vpc_service = VPCService()
    fit_run_id = "fit-run-123"
    vpc_run_id = "vpc-run-456"
    _cache_fit_context(fit_service, workspace, monkeypatch, fit_run_id=fit_run_id)

    class _FakeSimulationEngine:
        def __init__(self, *_args, **_kwargs):
            pass

    class _FakeVPCEngine:
        def __init__(self, *_args, **_kwargs):
            pass

        def compute(self, **_kwargs):
            class _Result:
                observed_df = pd.DataFrame({"ID": [1], "TIME": [1.0], "DV": [2.0], "REP": [0]})
                simulated_df = pd.DataFrame({"ID": [1], "TIME": [1.0], "DV": [2.1], "REP": [1]})
                obs_percentiles = pd.DataFrame(
                    {"bin_mid": [1.0], "p5": [1.0], "p50": [2.0], "p95": [3.0]}
                )
                sim_percentiles = pd.DataFrame(
                    {
                        "bin_mid": [1.0],
                        "p5_lo": [0.5],
                        "p5_mid": [1.0],
                        "p5_hi": [1.5],
                        "p50_lo": [1.5],
                        "p50_mid": [2.0],
                        "p50_hi": [2.5],
                        "p95_lo": [2.5],
                        "p95_mid": [3.0],
                        "p95_hi": [3.5],
                    }
                )

            return _Result()

    class _FakeFigure:
        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"PNG")

    monkeypatch.setattr("openpkpd_gui.services.vpc_service.SimulationEngine", _FakeSimulationEngine)
    monkeypatch.setattr("openpkpd_gui.services.vpc_service.VPCEngine", _FakeVPCEngine)
    monkeypatch.setattr(
        "openpkpd_gui.services.vpc_service.vpc_plot", lambda *_a, **_k: _FakeFigure()
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.vpc_service.simulation_panel", lambda *_a, **_k: _FakeFigure()
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.vpc_service.prediction_interval_plot",
        lambda *_a, **_k: _FakeFigure(),
    )

    runner = JobRunner()
    try:
        outcome = runner.submit(
            vpc_service.create_job(
                workspace,
                fit_service=fit_service,
                config=VPCConfig(n_replicates=64, n_bins=6, seed=123, prediction_corrected=True),
                run_id=vpc_run_id,
            )
        ).result(timeout=5)
    finally:
        runner.shutdown()

    run = RunRecord(workflow="vpc", run_id=vpc_run_id)
    run.mark_running()
    artifacts = vpc_service.apply_job_outcome(run, outcome)

    assert outcome.value is not None
    assert isinstance(outcome.value, VPCRunResult)
    assert run.status == RunStatus.SUCCEEDED
    assert len(artifacts) == 4
    assert {artifact.kind for artifact in artifacts} == {"plot", "table"}

    summary = next(artifact for artifact in artifacts if artifact.kind == "table")
    plot_types = {
        artifact.metadata.get("plot_type") for artifact in artifacts if artifact.kind == "plot"
    }
    assert summary.metadata.get("artifact_role") == "vpc_summary"
    assert summary.source_run_id == fit_run_id
    assert summary.metadata.get("fit_run_id") == fit_run_id
    assert summary.metadata.get("vpc_run_id") == vpc_run_id
    assert plot_types == {"vpc", "simulation_panel", "prediction_interval_plot"}
    assert Path(summary.path or "").exists()
    assert all(Path(artifact.path or "").exists() for artifact in artifacts)
    assert workspace.workspace_id in Path(summary.path or "").parts
    assert workspace.active_project.project_id in Path(summary.path or "").parts
    assert workspace.active_scenario.scenario_id in Path(summary.path or "").parts


@pytest.mark.unit
def test_apply_job_outcome_logs_plot_generation_warnings(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(name="VPC demo", root_path=str(tmp_path), workspace_id="vpc-demo")
    fit_service = FitService()
    vpc_service = VPCService()
    _cache_fit_context(fit_service, workspace, monkeypatch, fit_run_id="fit-run-123")

    class _FakeSimulationEngine:
        def __init__(self, *_args, **_kwargs):
            pass

    class _FakeVPCEngine:
        def __init__(self, *_args, **_kwargs):
            pass

        def compute(self, **_kwargs):
            class _Result:
                observed_df = pd.DataFrame({"ID": [1], "TIME": [1.0], "DV": [2.0], "REP": [0]})
                simulated_df = pd.DataFrame({"ID": [1], "TIME": [1.0], "DV": [2.1], "REP": [1]})
                obs_percentiles = pd.DataFrame(
                    {"bin_mid": [1.0], "p5": [1.0], "p50": [2.0], "p95": [3.0]}
                )
                sim_percentiles = pd.DataFrame(
                    {
                        "bin_mid": [1.0],
                        "p5_lo": [0.5],
                        "p5_mid": [1.0],
                        "p5_hi": [1.5],
                        "p50_lo": [1.5],
                        "p50_mid": [2.0],
                        "p50_hi": [2.5],
                        "p95_lo": [2.5],
                        "p95_mid": [3.0],
                        "p95_hi": [3.5],
                    }
                )

            return _Result()

    monkeypatch.setattr("openpkpd_gui.services.vpc_service.SimulationEngine", _FakeSimulationEngine)
    monkeypatch.setattr("openpkpd_gui.services.vpc_service.VPCEngine", _FakeVPCEngine)
    monkeypatch.setattr(
        "openpkpd_gui.services.vpc_service.vpc_plot",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("plot backend unavailable")),
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.vpc_service.simulation_panel",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("panel backend unavailable")),
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.vpc_service.prediction_interval_plot",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("interval backend unavailable")),
    )

    runner = JobRunner()
    try:
        outcome = runner.submit(
            vpc_service.create_job(workspace, fit_service=fit_service, run_id="vpc-run-456")
        ).result(timeout=5)
    finally:
        runner.shutdown()

    run = RunRecord(workflow="vpc", run_id="vpc-run-456")
    run.mark_running()
    artifacts = vpc_service.apply_job_outcome(run, outcome)

    assert run.status == RunStatus.SUCCEEDED
    assert len(artifacts) == 1
    assert any("Could not generate VPC plot" in line for line in run.log_lines)
    assert any("Could not generate simulation panel plot" in line for line in run.log_lines)
    assert any("Could not generate prediction interval plot" in line for line in run.log_lines)
