"""Focused tests for NCA preparation and job orchestration."""

from __future__ import annotations

from pathlib import Path

from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.nca_service import NCAConfig, NCARunResult, NCAService


def _write_dataset(path: Path) -> None:
    path.write_text(
        "ID,TIME,AMT,DV,EVID\n"
        "1,0,100,0,1\n1,1,0,5,0\n1,2,0,3,0\n1,4,0,1,0\n"
        "2,0,120,0,1\n2,1,0,6,0\n2,2,0,4,0\n2,4,0,1.5,0\n",
        encoding="utf-8",
    )


def test_prepare_run_requires_active_dataset() -> None:
    result = NCAService().prepare_run(Workspace(name="Demo"))

    assert result.ready is False
    assert any("Load and save a dataset" in issue.message for issue in result.validation.issues)


def test_create_job_and_apply_outcome_writes_nca_csv_artifact(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    _write_dataset(dataset_path)
    workspace = Workspace(name="NCA demo", root_path=str(tmp_path))
    workspace.active_dataset = DatasetAsset(source_path=str(dataset_path), display_name="theo.csv")
    service = NCAService()
    preparation = service.prepare_run(workspace)

    assert preparation.ready is True

    runner = JobRunner()
    try:
        outcome = runner.submit(
            service.create_job(workspace, config=NCAConfig(route="oral"), preparation=preparation)
        ).result(timeout=10)
    finally:
        runner.shutdown()

    run = RunRecord(workflow="nca")
    run.mark_running()
    artifacts = service.apply_job_outcome(run, outcome)

    assert outcome.value is not None
    assert isinstance(outcome.value, NCARunResult)
    assert run.status == RunStatus.SUCCEEDED
    assert len(artifacts) >= 1

    # CSV summary is always the first artifact
    csv_artifact = next(a for a in artifacts if a.kind == "table")
    assert csv_artifact.metadata.get("artifact_role") == "nca_summary"
    assert Path(csv_artifact.path or "").exists()
    preview_text = Path(csv_artifact.path or "").read_text(encoding="utf-8")
    assert "subject_id" in preview_text
    assert "auc_last" in preview_text
    assert workspace.workspace_id in Path(csv_artifact.path or "").parts
    assert workspace.active_project.project_id in Path(csv_artifact.path or "").parts
    assert workspace.active_scenario.scenario_id in Path(csv_artifact.path or "").parts

    # Plot artifacts (distributions + boxplot) are generated when matplotlib is available
    plot_artifacts = [a for a in artifacts if a.kind == "plot"]
    for plot in plot_artifacts:
        assert Path(plot.path or "").exists()
        assert plot.metadata.get("plot_type") in {"nca_distributions", "nca_boxplot"}


def test_prepare_run_uses_active_dataset_import_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "nmdata.csv"
    dataset_path.write_text(
        "1,0,999,100,0,1\n1,1,999,0,5,0\n1,2,999,0,3,0\n",
        encoding="utf-8",
    )
    workspace = Workspace(name="NCA nmdata")
    workspace.active_dataset = DatasetAsset(
        source_path=str(dataset_path),
        display_name="nmdata.csv",
        separator=",",
        input_columns=["ID", "TIME", "_DROP_3", "AMT", "DV", "EVID"],
        columns=["ID", "TIME", "AMT", "DV", "EVID", "MDV", "CMT"],
    )

    result = NCAService().prepare_run(workspace)

    assert result.ready is True
    assert result.subject_count == 1
    assert result.observation_count == 2
    assert result.row_count == 3


def test_apply_job_outcome_logs_nca_plot_warnings(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "theo.csv"
    _write_dataset(dataset_path)
    workspace = Workspace(name="NCA demo", root_path=str(tmp_path))
    workspace.active_dataset = DatasetAsset(source_path=str(dataset_path), display_name="theo.csv")
    service = NCAService()
    preparation = service.prepare_run(workspace)

    monkeypatch.setattr(
        "openpkpd_gui.services.nca_service.nca_distributions",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("distributions unavailable")),
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.nca_service.nca_boxplot",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boxplot unavailable")),
    )

    runner = JobRunner()
    try:
        outcome = runner.submit(
            service.create_job(workspace, config=NCAConfig(route="oral"), preparation=preparation)
        ).result(timeout=10)
    finally:
        runner.shutdown()

    run = RunRecord(workflow="nca")
    run.mark_running()
    artifacts = service.apply_job_outcome(run, outcome)

    assert run.status == RunStatus.SUCCEEDED
    assert len(artifacts) == 1
    assert any("Could not generate NCA distributions plot" in line for line in run.log_lines)
    assert any("Could not generate NCA boxplot" in line for line in run.log_lines)
