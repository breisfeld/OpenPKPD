"""Pure tests for NCA workflow helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.nca_service import NCAConfig, NCAPreparationResult
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.workflows.nca_workflow import (
    build_nca_workflow,
    can_start_nca_run,
    format_nca_preparation_summary,
    format_nca_run_summary,
    latest_nca_artifact,
    load_nca_preview_text,
    nca_config_matches_artifact,
    recommend_nca_next_action,
)


def test_format_nca_preparation_summary_reports_counts() -> None:
    summary = format_nca_preparation_summary(
        NCAPreparationResult(
            dataset_path="/tmp/theo.csv", subject_count=12, observation_count=120, row_count=132
        )
    )

    assert "Ready to run NCA" in summary
    assert "/tmp/theo.csv" in summary
    assert "12 subjects" in summary


def test_format_nca_run_summary_handles_success_and_empty_state() -> None:
    assert format_nca_run_summary(None) == "No NCA runs yet."
    run = RunRecord(workflow="nca", status=RunStatus.SUCCEEDED, summary_text="12 subjects")

    assert "Succeeded" in format_nca_run_summary(run)
    assert "12 subjects" in format_nca_run_summary(run)


def test_latest_nca_artifact_and_preview_text_use_csv_artifact(tmp_path: Path) -> None:
    csv_path = tmp_path / "nca-summary.csv"
    csv_path.write_text("subject_id,auc_last\n1,10.5\n2,12.0\n", encoding="utf-8")
    artifact = ArtifactRecord(
        kind="table",
        label="NCA summary",
        path=str(csv_path),
        metadata={"artifact_role": "nca_summary", "media_type": "text/csv"},
    )
    workspace = Workspace()
    workspace.artifacts = [artifact]

    assert latest_nca_artifact(workspace) is artifact
    preview = load_nca_preview_text(str(csv_path))
    assert "subject_id,auc_last" in preview
    assert "1,10.5" in preview


def test_can_start_nca_run_disables_unchanged_success_and_reenables_for_config_changes() -> None:
    preparation = NCAPreparationResult(dataset_path="/tmp/nca.csv")
    config = NCAConfig()
    latest_run = RunRecord(workflow="nca", status=RunStatus.SUCCEEDED, summary_text="2 subjects")
    latest_artifact = ArtifactRecord(
        kind="table",
        label="NCA summary",
        metadata={
            "artifact_role": "nca_summary",
            "route": "oral",
            "auc_method": "linear-log",
            "min_points_lambda": 3,
            "exclude_cmax": True,
        },
    )

    assert nca_config_matches_artifact(config, latest_artifact) is True
    assert can_start_nca_run(preparation, None, None, config) is True
    assert (
        can_start_nca_run(
            preparation, RunRecord(workflow="nca", status=RunStatus.FAILED), None, config
        )
        is True
    )
    assert can_start_nca_run(preparation, latest_run, latest_artifact, config) is False
    assert (
        can_start_nca_run(preparation, latest_run, latest_artifact, NCAConfig(route="IV")) is True
    )
    assert (
        can_start_nca_run(
            preparation,
            RunRecord(workflow="nca", status=RunStatus.RUNNING),
            latest_artifact,
            config,
        )
        is False
    )


def test_recommend_nca_next_action_tracks_blocked_and_matched_results() -> None:
    blocked = NCAPreparationResult()

    assert recommend_nca_next_action(blocked, None, None, NCAConfig()) == (
        "Open Data",
        "data",
        "Load a dataset in the Data workflow before starting NCA.",
    )

    ready = NCAPreparationResult(
        dataset_path="/tmp/nca.csv", subject_count=2, observation_count=2, row_count=4
    )
    artifact = ArtifactRecord(
        kind="table",
        label="NCA summary",
        metadata={
            "artifact_role": "nca_summary",
            "route": "oral",
            "auc_method": "linear-log",
            "min_points_lambda": 3,
            "exclude_cmax": True,
        },
    )
    run = RunRecord(workflow="nca", status=RunStatus.SUCCEEDED, summary_text="2 subjects")

    assert recommend_nca_next_action(ready, None, None, NCAConfig()) is None
    assert recommend_nca_next_action(ready, run, artifact, NCAConfig()) == (
        "Open latest CSV",
        "__open_latest_results__",
        "Latest NCA results already match the current options. Open the saved summary or change an option to rerun.",
    )

    running = RunRecord(workflow="nca", status=RunStatus.RUNNING)
    assert recommend_nca_next_action(ready, running, artifact, NCAConfig()) is None


@pytest.mark.unit
def test_nca_workflow_next_action_transitions_from_data_to_latest_results(
    tmp_path: Path, monkeypatch
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "nca.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n1,1,5\n", encoding="utf-8")
    results_path = tmp_path / "nca-summary.csv"
    results_path.write_text("subject_id,auc_last\n1,12.0\n", encoding="utf-8")

    qt_core, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="NCA next action")
    project_service = ProjectService()
    widget = build_nca_workflow(project, project_service=project_service)
    navigations: list[str] = []
    opened_urls: list[str] = []
    widget._navigate_to_workflow = lambda workflow_id: navigations.append(workflow_id)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        qt_gui.QDesktopServices,
        "openUrl",
        lambda url: opened_urls.append(url.toLocalFile()),
    )

    try:
        widget.show()
        app.processEvents()

        next_action_label = widget.findChild(qt_widgets.QLabel, "nca-next-action-label")
        next_action_button = widget.findChild(qt_widgets.QPushButton, "nca-next-action-button")
        run_button = widget.findChild(qt_widgets.QPushButton, "nca-run-button")

        assert next_action_label is not None
        assert next_action_button is not None
        assert run_button is not None
        assert next_action_button.text() == "Open Data"

        next_action_button.click()
        app.processEvents()

        project_service.attach_dataset(
            project,
            DatasetAsset(
                source_path=str(dataset_path), display_name="nca.csv", columns=["ID", "TIME", "DV"]
            ),
        )
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_label.isHidden() is True
        assert next_action_button.isHidden() is True
        assert run_button.isEnabled() is True

        run = RunRecord(workflow="nca", status=RunStatus.SUCCEEDED, summary_text="1 subject")
        artifact = ArtifactRecord(
            kind="table",
            label="NCA summary",
            path=str(results_path),
            source_run_id=run.run_id,
            metadata={
                "artifact_role": "nca_summary",
                "route": "oral",
                "auc_method": "linear-log",
                "min_points_lambda": 3,
                "exclude_cmax": True,
            },
        )
        run.artifact_ids.append(artifact.artifact_id)
        project_service.add_run(project, run)
        project_service.add_artifact(project, artifact)
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_button.text() == "Open latest CSV"
        assert "already match the current options" in next_action_label.text()

        next_action_button.click()
        app.processEvents()

        assert navigations == ["data"]
        assert opened_urls == [str(results_path)]
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
