"""Focused tests for on-demand NPDE job orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import JobContext
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.fit_service import FitService
from openpkpd_gui.services.npde_service import NPDERunResult, NPDEService


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
        "NPDE demo",
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
    workspace = Workspace(name="NPDE demo")

    with pytest.raises(ValueError, match="requires a successful fit"):
        NPDEService().create_job(workspace, fit_service=FitService())


@pytest.mark.unit
def test_create_job_and_apply_outcome_write_fit_linked_npde_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = Workspace(name="NPDE demo", root_path=str(tmp_path), workspace_id="npde-demo")
    fit_service = FitService()
    npde_service = NPDEService()
    fit_run_id = "fit-run-123"
    npde_run_id = "npde-run-456"
    _cache_fit_context(fit_service, workspace, monkeypatch, fit_run_id=fit_run_id)

    monkeypatch.setattr(
        "openpkpd_gui.services.npde_service.compute_npde",
        lambda *_a, **_k: pd.DataFrame({"ID": [1], "TIME": [1.0], "PDE": [0.25], "NPDE": [-0.1]}),
    )

    class _FakeFigure:
        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"PNG")

    monkeypatch.setattr(
        "openpkpd_gui.services.npde_service.npde_plot", lambda *_a, **_k: _FakeFigure()
    )

    runner = JobRunner()
    try:
        outcome = runner.submit(
            npde_service.create_job(workspace, fit_service=fit_service, run_id=npde_run_id)
        ).result(timeout=5)
    finally:
        runner.shutdown()

    run = RunRecord(workflow="npde", run_id=npde_run_id)
    run.mark_running()
    artifacts = npde_service.apply_job_outcome(run, outcome)

    assert outcome.value is not None
    assert isinstance(outcome.value, NPDERunResult)
    assert run.status == RunStatus.SUCCEEDED
    assert len(artifacts) == 2

    table = next(artifact for artifact in artifacts if artifact.kind == "table")
    plot = next(artifact for artifact in artifacts if artifact.kind == "plot")
    assert table.metadata.get("artifact_role") == "npde_table"
    assert plot.metadata.get("plot_type") == "npde_plot"
    assert table.source_run_id == fit_run_id
    assert plot.source_run_id == fit_run_id
    assert table.metadata.get("fit_run_id") == fit_run_id
    assert plot.metadata.get("fit_run_id") == fit_run_id
    assert table.metadata.get("npde_run_id") == npde_run_id
    assert plot.metadata.get("npde_run_id") == npde_run_id
    assert Path(table.path or "").exists()
    assert Path(plot.path or "").exists()
    assert run.artifact_ids == [artifact.artifact_id for artifact in artifacts]
    assert workspace.workspace_id in Path(table.path or "").parts
    assert workspace.active_project.project_id in Path(table.path or "").parts
    assert workspace.active_scenario.scenario_id in Path(table.path or "").parts
