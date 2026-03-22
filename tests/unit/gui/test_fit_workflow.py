"""Pure tests for Fit workflow presentation helpers."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec, ModelSpecMode
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob, JobStatus
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.fit_service import FitPreparationResult, FitRunResult
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.validation_service import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from openpkpd_gui.workflows.fit_workflow import (
    build_fit_workflow,
    can_start_fit_run,
    format_fit_preparation_summary,
    format_fit_run_summary,
    recommend_fit_next_action,
    validation_issue_target,
)


def test_format_fit_preparation_summary_reports_readiness() -> None:
    summary = format_fit_preparation_summary(
        FitPreparationResult(
            problem_title="One compartment",
            dataset_path="/tmp/theo.csv",
            mode=ModelSpecMode.BUILDER,
            estimation_method="FOCE",
            theta_count=3,
            eta_count=3,
            eps_count=1,
        )
    )

    assert "Fit needs attention" in summary
    assert "One compartment" in summary
    assert "builder" in summary
    assert "3 THETA" in summary


def test_format_fit_run_summary_reports_status() -> None:
    run = RunRecord(
        workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Demo • FOCE • converged=True"
    )

    summary = format_fit_run_summary(run)

    assert "Succeeded" in summary
    assert "FOCE" in summary


def test_format_fit_run_summary_reports_cancel_requested() -> None:
    run = RunRecord(workflow="fit", status=RunStatus.RUNNING)
    run.mark_cancel_requested()

    summary = format_fit_run_summary(run)

    assert "cancellation requested" in summary.lower()


def test_can_start_fit_run_requires_readiness_and_no_current_success() -> None:
    ready = FitPreparationResult(translation=SimpleNamespace(ok=True))
    not_ready = FitPreparationResult()

    assert can_start_fit_run(not_ready, None) is False
    assert can_start_fit_run(ready, None) is True
    assert can_start_fit_run(ready, RunRecord(workflow="fit", status=RunStatus.FAILED)) is True
    assert can_start_fit_run(ready, RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)) is False
    assert can_start_fit_run(ready, RunRecord(workflow="fit", status=RunStatus.RUNNING)) is False


def test_recommend_fit_next_action_tracks_blocked_and_completed_states() -> None:
    workspace = Workspace(name="Fit CTA")
    blocked = FitPreparationResult()

    assert recommend_fit_next_action(workspace, blocked, None) == (
        "Open Data",
        "data",
        "Load a dataset in the Data workflow before starting a fit.",
    )

    workspace.active_dataset = DatasetAsset(source_path="/tmp/theo.csv", display_name="theo.csv")
    assert recommend_fit_next_action(workspace, blocked, None) == (
        "Open Model",
        "model",
        "Configure a model in the Model workflow before starting a fit.",
    )

    workspace.active_model_spec = object()  # type: ignore[assignment]
    blocked.validation = ValidationResult(
        issues=[
            ValidationIssue(ValidationSeverity.ERROR, "$PK code is required.", field_name="pk_code")
        ]
    )
    assert recommend_fit_next_action(workspace, blocked, None) == (
        "Open Model",
        "model",
        "$PK code is required.",
    )

    ready = FitPreparationResult(translation=SimpleNamespace(ok=True))
    assert recommend_fit_next_action(workspace, ready, None) is None

    succeeded = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)
    assert recommend_fit_next_action(workspace, ready, succeeded) == (
        "Open Results",
        "results",
        "A successful fit is already available. Review the latest outputs in Results.",
    )

    running = RunRecord(workflow="fit", status=RunStatus.RUNNING)
    assert recommend_fit_next_action(workspace, ready, running) is None


def test_validation_issue_target_prefers_explicit_routing_metadata() -> None:
    issue = ValidationIssue(
        ValidationSeverity.ERROR,
        "Load a dataset first.",
        field_name="active_dataset",
        target_workflow="data",
        target_widget="data-source-path",
    )

    target = validation_issue_target(issue)

    assert target is not None
    assert target.workflow_id == "data"
    assert target.widget_object_name == "data-source-path"


def test_validation_issue_target_maps_model_translation_fields() -> None:
    issue = ValidationIssue(ValidationSeverity.ERROR, "$PK code is required.", field_name="pk_code")

    target = validation_issue_target(issue)

    assert target is not None
    assert target.workflow_id == "model"
    assert target.widget_object_name == "model-pk-code"


class _FakeFitService:
    def __init__(self, *, mode: str = "success") -> None:
        self.mode = mode
        self.started = threading.Event()
        self.finish_signal = threading.Event()

    def prepare_run(self, _workspace: Workspace) -> FitPreparationResult:
        return FitPreparationResult(
            translation=SimpleNamespace(ok=True),
            problem_title="Demo fit",
            dataset_path="/tmp/demo.csv",
            mode=ModelSpecMode.BUILDER,
            estimation_method="FOCE",
            theta_count=1,
            eta_count=1,
            eps_count=1,
        )

    def latest_run(self, workspace: Workspace) -> RunRecord | None:
        for run in reversed(workspace.runs):
            if run.workflow == "fit":
                return run
        return None

    def create_job(
        self,
        _workspace: Workspace,
        preparation=None,
        run_id: str | None = None,
        n_parallel: int = 0,
    ) -> BackgroundJob:
        title = preparation.problem_title if preparation is not None else "Demo fit"

        def _work(ctx):
            self.started.set()
            ctx.emit(f"Preparing fit for {title}", progress=0.1)
            if self.mode == "cancel":
                while not ctx.is_cancel_requested():
                    time.sleep(0.01)
                ctx.emit("Waiting for cooperative shutdown", progress=0.35)
                self.finish_signal.wait(timeout=5)
                ctx.check_cancelled()
            self.finish_signal.wait(timeout=5)
            ctx.emit("Generating fit outputs", progress=0.85)
            return FitRunResult(
                problem_title=title,
                estimation_method="FOCE",
                converged=True,
                ofv=12.34,
                summary_text=f"{title} • FOCE • converged=True • OFV=12.3400",
            )

        return BackgroundJob(name=f"fit:{title}", func=_work, job_id=run_id or "fit-job")

    def apply_job_outcome(self, run: RunRecord, outcome) -> list[object]:
        if outcome.status == JobStatus.SUCCEEDED and isinstance(outcome.value, FitRunResult):
            run.mark_succeeded(outcome.value.summary_text)
            return []
        if outcome.status == JobStatus.CANCELLED:
            run.mark_cancelled(outcome.error or "Cancelled by user.")
            return []
        run.mark_failed(outcome.error or "Fit failed.")
        return []


def _wait_for(predicate, app, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for GUI condition")


@pytest.mark.unit
def test_fit_workflow_next_action_transitions_from_data_to_results(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Fit next action")
    project_service = ProjectService()
    widget = build_fit_workflow(project, project_service=project_service)
    navigations: list[str] = []
    widget._navigate_to_workflow = lambda workflow_id: navigations.append(workflow_id)  # type: ignore[attr-defined]

    try:
        widget.show()
        app.processEvents()

        next_action_label = widget.findChild(qt_widgets.QLabel, "fit-next-action-label")
        next_action_button = widget.findChild(qt_widgets.QPushButton, "fit-next-action-button")
        run_button = widget.findChild(qt_widgets.QPushButton, "fit-run-button")

        assert next_action_label is not None
        assert next_action_button is not None
        assert run_button is not None
        assert next_action_button.text() == "Open Data"

        next_action_button.click()
        app.processEvents()

        project_service.attach_dataset(
            project, DatasetAsset(source_path=str(dataset_path), display_name="theo.csv")
        )
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_button.text() == "Open Model"
        assert "Configure a model" in next_action_label.text()

        next_action_button.click()
        app.processEvents()

        project_service.set_model_spec(
            project,
            ModelSpec(
                problem_title="Smoke",
                dataset_path=str(dataset_path),
                pk_code="CL = THETA(1) * EXP(ETA(1))",
                error_code="Y = F * (1 + EPS(1))",
                theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
                omega_values=[[0.3]],
                sigma_values=[[0.1]],
            ),
        )
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_label.isHidden() is True
        assert next_action_button.isHidden() is True
        assert run_button.isEnabled() is True

        run = RunRecord(workflow="fit")
        run.mark_succeeded("Finished")
        project_service.add_run(project, run)
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_button.text() == "Open Results"
        assert next_action_label.isHidden() is False

        next_action_button.click()
        app.processEvents()

        assert navigations == ["data", "model", "results"]
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_fit_workflow_streams_live_job_events() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Fit streaming")
    fit_service = _FakeFitService(mode="success")
    project_service = ProjectService()
    job_runner = JobRunner(max_workers=1)
    widget = build_fit_workflow(
        project, fit_service=fit_service, project_service=project_service, job_runner=job_runner
    )

    try:
        widget.show()
        app.processEvents()

        run_button = widget.findChild(qt_widgets.QPushButton, "fit-run-button")
        progress = widget.findChild(qt_widgets.QProgressBar, "fit-run-progress")
        log_output = widget.findChild(qt_widgets.QPlainTextEdit, "fit-log-output")
        run_label = widget.findChild(qt_widgets.QLabel, "fit-run-summary")

        assert run_button is not None
        assert progress is not None
        assert log_output is not None
        assert run_label is not None

        run_button.click()
        _wait_for(lambda: "Preparing fit for Demo fit" in log_output.toPlainText(), app)

        assert progress.isVisible() is True
        assert progress.maximum() == 100
        assert progress.value() >= 10
        assert run_label.text() == "Latest run — Running"

        fit_service.finish_signal.set()
        _wait_for(lambda: "Succeeded" in run_label.text(), app)

        assert "Generating fit outputs" in log_output.toPlainText()
        assert progress.isVisible() is False
    finally:
        fit_service.finish_signal.set()
        widget.close()
        widget.deleteLater()
        app.processEvents()
        job_runner.shutdown(wait=True)


@pytest.mark.unit
def test_fit_workflow_marks_cancel_requested_before_final_cancellation() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Fit cancellation")
    fit_service = _FakeFitService(mode="cancel")
    project_service = ProjectService()
    job_runner = JobRunner(max_workers=1)
    widget = build_fit_workflow(
        project, fit_service=fit_service, project_service=project_service, job_runner=job_runner
    )

    try:
        widget.show()
        app.processEvents()

        run_button = widget.findChild(qt_widgets.QPushButton, "fit-run-button")
        cancel_button = widget.findChild(qt_widgets.QPushButton, "fit-cancel-button")
        log_output = widget.findChild(qt_widgets.QPlainTextEdit, "fit-log-output")
        run_label = widget.findChild(qt_widgets.QLabel, "fit-run-summary")

        assert run_button is not None
        assert cancel_button is not None
        assert log_output is not None
        assert run_label is not None

        run_button.click()
        _wait_for(lambda: fit_service.started.is_set(), app)

        cancel_button.click()
        _wait_for(lambda: "cancellation requested" in run_label.text().lower(), app)

        assert cancel_button.isEnabled() is False
        assert (
            "Cancellation requested. Waiting for the current step to finish."
            in log_output.toPlainText()
        )

        fit_service.finish_signal.set()
        _wait_for(lambda: run_label.text() == "Latest run — Cancelled", app)
        assert "Waiting for cooperative shutdown" in log_output.toPlainText()
    finally:
        fit_service.finish_signal.set()
        widget.close()
        widget.deleteLater()
        app.processEvents()
        job_runner.shutdown(wait=True)
