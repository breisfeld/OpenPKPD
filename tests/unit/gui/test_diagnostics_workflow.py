"""Pure tests for Diagnostics workflow presentation helpers."""

from __future__ import annotations

import time
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.artifact_service import ArtifactService
from openpkpd_gui.services.npde_service import NPDERunResult, NPDEService
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.workflows.diagnostics_workflow import (
    build_diagnostics_workflow,
    filter_diagnostics_artifacts,
    format_diagnostics_artifact_summary,
    format_diagnostics_next_steps,
    format_diagnostics_stale_warning,
    format_diagnostics_status,
    format_npde_generation_status,
    latest_diagnostics_artifact,
    latest_diagnostics_role_artifact,
    latest_fit_run,
    plotting_backend_available,
    recommend_diagnostics_next_action,
)


def test_latest_fit_run_returns_most_recent_fit() -> None:
    first = RunRecord(workflow="fit", run_id="first")
    second = RunRecord(workflow="fit", run_id="second")
    other = RunRecord(workflow="nca", run_id="other")
    workspace = Workspace()
    workspace.runs = [first, other, second]

    assert latest_fit_run(workspace) is second


def test_diagnostics_status_reports_success_and_missing_fit_guidance() -> None:
    empty_workspace = Workspace(name="Demo")
    ready_workspace = Workspace()
    ready_workspace.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]

    empty_status = format_diagnostics_status(
        empty_workspace,
        diagnostics_api_available=True,
        plotting_available=True,
    )
    ready_status = format_diagnostics_status(
        ready_workspace,
        diagnostics_api_available=True,
        plotting_available=True,
    )

    assert "completed fit run" in empty_status
    assert "latest fit succeeded" in ready_status.lower()


def test_diagnostics_stale_warning_mentions_changed_inputs() -> None:
    workspace = Workspace(name="Demo")
    fit_run = RunRecord(workflow="fit")
    fit_run.mark_running()
    fit_run.mark_succeeded("ok")
    fit_run.finished_at = "2026-03-15T12:00:00+00:00"
    workspace.active_scenario.runs = [fit_run]
    workspace.active_scenario.dataset_updated_at = "2026-03-15T12:01:00+00:00"
    workspace.active_scenario.model_updated_at = "2026-03-15T12:02:00+00:00"

    warning = format_diagnostics_stale_warning(workspace)

    assert warning == (
        "Dataset and model changed since the latest successful fit. Diagnostics may be stale until you rerun the fit."
    )


def test_plotting_backend_available_checks_importability_not_only_spec() -> None:
    fake_module = SimpleNamespace(__spec__=ModuleSpec("matplotlib", loader=None))

    def fake_import_module(name: str):
        if name == "matplotlib":
            return fake_module
        msg = f"Unexpected import request: {name}"
        raise AssertionError(msg)

    with (
        patch(
            "openpkpd_gui.workflows.diagnostics_workflow.import_module",
            side_effect=fake_import_module,
        ),
        patch(
            "openpkpd_gui.workflows.diagnostics_workflow.find_spec",
            return_value=None,
        ),
    ):
        assert plotting_backend_available() is True


def test_next_steps_and_artifact_summary_include_key_context() -> None:
    artifact = ArtifactRecord(kind="plot", label="GOF panel")
    failed_run = RunRecord(workflow="fit", status=RunStatus.FAILED)
    workspace = Workspace()
    workspace.artifacts = [artifact]
    workspace.runs = [failed_run]

    steps = format_diagnostics_next_steps(workspace)
    summary = format_diagnostics_artifact_summary(workspace.artifacts)

    assert "Load a dataset" in steps
    assert "Open the Model workflow to configure a model" in steps
    assert "failed run" in steps
    assert "1 outputs" in summary
    assert "plot: 1" in summary


def test_recommend_diagnostics_next_action_tracks_fit_state() -> None:
    workspace = Workspace()

    assert recommend_diagnostics_next_action(workspace, []) == (
        "Open Data",
        "data",
        "Load a dataset before expecting diagnostics outputs for this scenario.",
    )

    workspace.active_dataset = object()  # type: ignore[assignment]
    assert recommend_diagnostics_next_action(workspace, []) == (
        "Open Model",
        "model",
        "Configure a model in the Model workflow first.",
    )

    workspace.active_model_spec = object()  # type: ignore[assignment]
    assert recommend_diagnostics_next_action(workspace, []) == (
        "Open Fit",
        "fit",
        "Run a fit to unlock diagnostics review for this scenario.",
    )

    workspace.runs = [RunRecord(workflow="fit", status=RunStatus.FAILED)]
    assert recommend_diagnostics_next_action(workspace, []) == (
        "Open Results",
        "results",
        "Inspect the failed fit in Results before expecting diagnostics outputs.",
    )

    diagnostics_table = ArtifactRecord(kind="table", label="Diagnostics table")
    assert recommend_diagnostics_next_action(workspace, [diagnostics_table]) is None


def test_diagnostics_filters_and_latest_artifact_use_plot_metadata() -> None:
    report = ArtifactRecord(
        kind="report",
        label="Report",
        metadata={"artifact_role": "report", "media_type": "text/html"},
    )
    gof = ArtifactRecord(
        kind="plot",
        label="GOF panel",
        metadata={"artifact_role": "plot", "plot_type": "gof_panel", "media_type": "image/png"},
    )
    residual = ArtifactRecord(
        kind="plot",
        label="Residual trends",
        metadata={
            "artifact_role": "plot",
            "plot_type": "residual_trends",
            "media_type": "image/png",
        },
    )

    assert filter_diagnostics_artifacts([report, gof, residual], "plot", "gof_panel") == [gof]
    assert latest_diagnostics_artifact([report, gof, residual]) is residual
    assert latest_diagnostics_artifact([report, gof, residual], plot_type="gof_panel") is gof


def test_successful_diagnostics_next_steps_and_table_lookup_reflect_current_outputs() -> None:
    diagnostics_table = ArtifactRecord(
        kind="table",
        label="Diagnostics table",
        metadata={"artifact_role": "diagnostics_table", "media_type": "text/csv"},
    )
    workspace = Workspace()
    workspace.active_dataset = object()  # type: ignore[assignment]
    workspace.active_model_spec = object()  # type: ignore[assignment]
    workspace.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]

    steps = format_diagnostics_next_steps(workspace)

    assert "Results quick review" in steps
    assert "diagnostics tables" in steps
    assert (
        latest_diagnostics_role_artifact([diagnostics_table], "diagnostics_table")
        is diagnostics_table
    )


@pytest.mark.unit
def test_diagnostics_workflow_next_action_advances_from_fit_to_results() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Diagnostics next action")
    project.active_dataset = object()  # type: ignore[assignment]
    project.active_model_spec = object()  # type: ignore[assignment]
    widget = build_diagnostics_workflow(project)
    navigations: list[str] = []
    widget._navigate_to_workflow = lambda workflow_id: navigations.append(workflow_id)  # type: ignore[attr-defined]

    try:
        widget.show()
        app.processEvents()

        next_action_label = widget.findChild(qt_widgets.QLabel, "diagnostics-next-action-label")
        next_action_button = widget.findChild(
            qt_widgets.QPushButton, "diagnostics-next-action-button"
        )

        assert next_action_label is not None
        assert next_action_button is not None
        assert next_action_button.text() == "Open Fit"

        next_action_button.click()
        app.processEvents()

        failed_run = RunRecord(workflow="fit", status=RunStatus.FAILED)
        project.active_scenario.runs = [failed_run]
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_button.text() == "Open Results"
        assert "failed fit" in next_action_label.text().lower()

        next_action_button.click()
        app.processEvents()

        succeeded_run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)
        artifact = ArtifactRecord(
            kind="table",
            label="Diagnostics table",
            source_run_id=succeeded_run.run_id,
            metadata={"artifact_role": "diagnostics_table", "media_type": "text/csv"},
        )
        succeeded_run.artifact_ids.append(artifact.artifact_id)
        project.active_scenario.runs = [failed_run, succeeded_run]
        project.active_scenario.artifacts = [artifact]
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert navigations == ["fit", "results"]
        assert next_action_label.isHidden() is True
        assert next_action_button.isHidden() is True
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_npde_generation_status_covers_ready_and_existing_artifact_states() -> None:
    workspace = Workspace()

    assert "completed fit run" in format_npde_generation_status(
        workspace,
        fit_context_available=False,
        npde_available=False,
    )

    workspace.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]

    assert "Generate NPDE on demand" in format_npde_generation_status(
        workspace,
        fit_context_available=True,
        npde_available=False,
    )
    assert "outputs are already available" in format_npde_generation_status(
        workspace,
        fit_context_available=True,
        npde_available=True,
    )


@pytest.mark.unit
def test_diagnostics_workflow_generates_npde_on_demand(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Diagnostics NPDE", root_path=str(tmp_path))
    project_service = ProjectService()
    artifact_service = ArtifactService()
    fit_run = RunRecord(workflow="fit", run_id="fit-run-1", status=RunStatus.SUCCEEDED)
    project_service.add_run(project, fit_run)

    artifact_path = tmp_path / "generated-npde.csv"
    artifact_path.write_text("ID,TIME,NPDE\n1,1.0,-0.1\n", encoding="utf-8")
    plot_path = tmp_path / "generated-npde.png"
    plot_path.write_bytes(b"PNG")

    class _FakeFitService:
        def latest_fit_context(self, workspace: Workspace):
            return object() if workspace is project else None

    class _FakeNPDEService(NPDEService):
        def create_job(
            self, workspace: Workspace, *, fit_service, run_id: str | None = None, **_kwargs
        ):
            table = ArtifactRecord(
                kind="table",
                label="Generated NPDE",
                path=str(artifact_path),
                source_run_id=fit_run.run_id,
                metadata={
                    "artifact_role": "npde_table",
                    "media_type": "text/csv",
                    "fit_run_id": fit_run.run_id,
                    "npde_run_id": run_id,
                },
            )
            plot = ArtifactRecord(
                kind="plot",
                label="Generated NPDE plot",
                path=str(plot_path),
                source_run_id=fit_run.run_id,
                metadata={
                    "artifact_role": "plot",
                    "plot_type": "npde_plot",
                    "media_type": "image/png",
                    "fit_run_id": fit_run.run_id,
                    "npde_run_id": run_id,
                },
            )
            return BackgroundJob(
                name="npde:demo",
                func=lambda _ctx: NPDERunResult(
                    summary_text="Generated NPDE", artifacts=[table, plot]
                ),
            )

    job_runner = JobRunner(max_workers=1)
    widget = build_diagnostics_workflow(
        project,
        artifact_service=artifact_service,
        fit_service=_FakeFitService(),
        npde_service=_FakeNPDEService(),
        project_service=project_service,
        job_runner=job_runner,
    )

    try:
        widget.show()
        app.processEvents()

        status_label = widget.findChild(qt_widgets.QLabel, "diagnostics-npde-status-label")
        generate_button = widget.findChild(
            qt_widgets.QPushButton, "diagnostics-generate-npde-button"
        )
        open_npde_action = widget.findChild(qt_gui.QAction, "diagnostics-view-npde-table-action")

        assert status_label is not None
        assert generate_button is not None
        assert open_npde_action is not None
        assert generate_button.isEnabled() is True
        assert "Generate NPDE on demand" in status_label.text()

        generate_button.click()
        app.processEvents()

        assert "running in the background" in status_label.text()

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            app.processEvents()
            if open_npde_action.isEnabled():
                break
            time.sleep(0.05)

        assert open_npde_action.isEnabled() is True
        assert generate_button.isEnabled() is False
        assert "outputs are already available" in status_label.text()
        assert len(project.artifacts) == 2
        assert {artifact.kind for artifact in project.artifacts} == {"plot", "table"}
        assert all(artifact.source_run_id == fit_run.run_id for artifact in project.artifacts)
        assert all(
            artifact.metadata.get("fit_run_id") == fit_run.run_id for artifact in project.artifacts
        )

        npde_runs = [run for run in project.runs if run.workflow == "npde"]
        assert len(npde_runs) == 1
        assert npde_runs[0].status == RunStatus.SUCCEEDED
        assert all(
            artifact.metadata.get("npde_run_id") == npde_runs[0].run_id
            for artifact in project.artifacts
        )
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
        job_runner.shutdown(wait=True)


@pytest.mark.unit
def test_diagnostics_workflow_does_not_shutdown_injected_job_runner() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )

    class _RecordingRunner:
        def __init__(self) -> None:
            self.shutdown_calls: list[bool] = []

        def shutdown(self, wait: bool = True) -> None:
            self.shutdown_calls.append(wait)

    job_runner = _RecordingRunner()
    widget = build_diagnostics_workflow(
        Workspace(name="Diagnostics lifecycle"), job_runner=job_runner
    )

    widget.show()
    app.processEvents()
    widget.close()
    widget.deleteLater()
    app.processEvents()

    assert job_runner.shutdown_calls == []


# ---------------------------------------------------------------------------
# P2-B: subject filter combo
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_diagnostics_workflow_has_subject_filter_combo() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    widget = build_diagnostics_workflow(Workspace(name="SubjectFilter"))
    try:
        combo = widget.findChild(qt_widgets.QComboBox, "diagnostics-subject-filter")
        assert combo is not None
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_diagnostics_subject_filter_has_all_subjects_default() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    widget = build_diagnostics_workflow(Workspace(name="SubjectDefault"))
    try:
        combo = widget.findChild(qt_widgets.QComboBox, "diagnostics-subject-filter")
        assert combo.itemText(0) == "All subjects"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_diagnostics_subject_filter_populated_from_csv(tmp_path: Path) -> None:
    """Subject filter is populated with subject IDs from the diagnostics CSV artifact."""
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not available")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )

    # Build a diagnostics CSV with known subjects
    csv_path = tmp_path / "diagnostics.csv"
    pd.DataFrame(
        {
            "ID": ["S1", "S1", "S2", "S3"],
            "TIME": [0.0, 1.0, 0.0, 0.0],
            "DV": [1.0, 2.0, 1.5, 1.2],
            "PRED": [1.1, 1.9, 1.4, 1.3],
            "IPRED": [1.05, 1.95, 1.45, 1.25],
            "CWRES": [0.1, -0.2, 0.3, 0.0],
            "IWRES": [0.05, -0.1, 0.15, 0.0],
        }
    ).to_csv(csv_path, index=False)

    workspace = Workspace(name="SubjectCSV")
    diag_artifact = ArtifactRecord(
        kind="table",
        label="Diagnostics CSV",
        path=str(csv_path),
        metadata={"artifact_role": "diagnostics_table"},
    )
    workspace.active_scenario.add_artifact(diag_artifact)

    widget = build_diagnostics_workflow(workspace)
    try:
        widget.show()
        app.processEvents()
        # Trigger a refresh to populate the combo
        if hasattr(widget, "_refresh_workflow"):
            widget._refresh_workflow()
        app.processEvents()

        combo = widget.findChild(qt_widgets.QComboBox, "diagnostics-subject-filter")
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "S1" in items
        assert "S2" in items
        assert "S3" in items
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
