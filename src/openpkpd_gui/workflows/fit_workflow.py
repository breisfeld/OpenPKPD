"""Fit workflow widget for readiness checks and background run orchestration."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    load_gui_preferences,
    save_gui_preferences,
    with_dismissed_hint,
)
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob, JobEvent, JobOutcome
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.artifact_service import ArtifactService
from openpkpd_gui.services.fit_service import FitPreparationResult, FitService
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.validation_service import ValidationIssue
from openpkpd_gui.widgets.collapsible_section import build_collapsible_section
from openpkpd_gui.widgets.convergence_plot import build_convergence_plot_widget
from openpkpd_gui.widgets.combined_header import build_combined_header
from openpkpd_gui.widgets.dismissible_hint import build_dismissible_hint
from openpkpd_gui.widgets.responsive_layout import (
    install_responsive_box_layouts,
    install_responsive_splitters,
)
from openpkpd_gui.widgets.scrollable_page import build_scrollable_page

FIT_RESPONSIVE_LAYOUT_BREAKPOINT = 1000


@dataclass(frozen=True, slots=True)
class ValidationIssueTarget:
    workflow_id: str
    widget_object_name: str | None = None


_FIT_VALIDATION_TARGETS: dict[str, ValidationIssueTarget] = {
    "active_dataset": ValidationIssueTarget("data", "data-source-path"),
    "active_model_spec": ValidationIssueTarget("model", "model-problem-title"),
    "problem_title": ValidationIssueTarget("model", "model-problem-title"),
    "dataset_path": ValidationIssueTarget("model", "model-dataset-path"),
    "pk_code": ValidationIssueTarget("model", "model-pk-code"),
    "error_code": ValidationIssueTarget("model", "model-error-code"),
    "des_code": ValidationIssueTarget("model", "model-des-code"),
    "control_stream_text": ValidationIssueTarget("model", "model-control-stream-text"),
    "theta_rows": ValidationIssueTarget("model", "model-theta-table"),
    "omega_values": ValidationIssueTarget("model", "model-omega-table"),
    "sigma_values": ValidationIssueTarget("model", "model-sigma-table"),
}


def validation_issue_target(issue: ValidationIssue) -> ValidationIssueTarget | None:
    if issue.target_workflow or issue.target_widget:
        if issue.target_workflow is None:
            return None
        return ValidationIssueTarget(issue.target_workflow, issue.target_widget)
    if issue.field_name is None:
        return None
    return _FIT_VALIDATION_TARGETS.get(issue.field_name)


def format_fit_preparation_summary(
    preparation: FitPreparationResult,
    latest_run: RunRecord | None = None,
) -> str:
    """Return a compact readiness summary for the Fit workflow."""
    title = preparation.problem_title or "Untitled fit"
    mode = preparation.mode.value.replace("_", " ") if preparation.mode is not None else "no model"
    dataset = preparation.dataset_path or "dataset unresolved"
    method = preparation.estimation_method or "unspecified"
    if latest_run is not None and latest_run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
        status = "Fit in progress"
    elif preparation.ready:
        status = "Ready to start fit"
    else:
        status = "Fit needs attention"
    return (
        f"{status} — {title} • {mode} • method {method} • dataset {dataset} "
        f"• {preparation.theta_count} THETA • {preparation.eta_count} ETA • {preparation.eps_count} EPS"
    )


def format_fit_run_summary(run: RunRecord | None) -> str:
    """Render a concise status line for the latest fit run."""
    if run is None:
        return "No fit runs yet."
    if run.status == RunStatus.SUCCEEDED and run.summary_text:
        return f"Latest run — Succeeded • {run.summary_text}"
    if run.status == RunStatus.FAILED and run.error_text:
        return f"Latest run — Failed • {run.error_text}"
    if run.status == RunStatus.CANCELLED:
        return "Latest run — Cancelled"
    if run.status == RunStatus.RUNNING and run.cancel_requested_at is not None:
        return "Latest run — Running • cancellation requested"
    return f"Latest run — {run.status.value.title()}"


def can_start_fit_run(preparation: FitPreparationResult, latest_run: RunRecord | None) -> bool:
    """Return whether the Fit workflow should allow starting a new fit."""
    if not preparation.ready:
        return False
    if latest_run is None:
        return True
    return latest_run.status not in {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.SUCCEEDED}


def recommend_fit_next_action(
    project: Workspace,
    preparation: FitPreparationResult,
    latest_run: RunRecord | None,
) -> tuple[str, str, str] | None:
    """Return the primary CTA for blocked or already-completed Fit states."""
    if latest_run is not None and latest_run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
        return None
    if project.active_dataset is None:
        return (
            "Open Data",
            "data",
            "Load a dataset in the Data workflow before starting a fit.",
        )
    if project.active_model_spec is None:
        return (
            "Open Model",
            "model",
            "Configure a model in the Model workflow before starting a fit.",
        )
    if not preparation.ready:
        for issue in preparation.validation.issues:
            target = validation_issue_target(issue)
            if target is None:
                continue
            if target.workflow_id == "data":
                return ("Open Data", "data", issue.message)
            if target.workflow_id == "model":
                return ("Open Model", "model", issue.message)
        return (
            "Open Model",
            "model",
            "Resolve the model validation issues before starting a fit.",
        )
    if latest_run is not None and latest_run.status == RunStatus.SUCCEEDED:
        return (
            "Open Results",
            "results",
            "A successful fit is already available. Review the latest outputs in Results.",
        )
    return None


def build_fit_workflow(
    project: Workspace,
    fit_service: FitService | None = None,
    project_service: ProjectService | None = None,
    job_runner: JobRunner | None = None,
    preferences: list | None = None,
):
    """Build the first real Fit workflow page."""
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    fit_service = fit_service or FitService()
    project_service = project_service or ProjectService()
    artifact_service = ArtifactService()
    job_runner = job_runner or JobRunner(max_workers=1)
    preparation = fit_service.prepare_run(project)

    root, _, layout, scroll_area = build_scrollable_page(
        qt_widgets, root_object_name="fit-workflow"
    )

    combined_header, refresh_combined_header = build_combined_header(
        root,
        project,
        workflow_id="fit",
        workflow_label="Fit",
        status_workflow_ids=("data", "model", "fit", "results"),
    )

    title_label = qt_widgets.QLabel("Fit workflow")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 4)
    title_font.setBold(True)
    title_label.setFont(title_font)

    _hint_prefs = load_gui_preferences()

    def _save_dismissed_hint_fit():
        save_gui_preferences(with_dismissed_hint(load_gui_preferences(), "hint_fit"))

    hint_widget, _ = build_dismissible_hint(
        "Review saved dataset/model readiness, then launch a background fit run. "
        "This first slice focuses on preparation, run state, and log capture.",
        dismissed="hint_fit" in _hint_prefs.dismissed_hints,
        on_dismiss=_save_dismissed_hint_fit,
    )

    preparation_label = qt_widgets.QLabel(format_fit_preparation_summary(preparation))
    preparation_label.setObjectName("fit-preparation-summary")
    preparation_label.setWordWrap(True)

    next_action_label = qt_widgets.QLabel("")
    next_action_label.setObjectName("fit-next-action-label")
    next_action_label.setWordWrap(True)
    next_action_label.setVisible(False)

    next_action_button = qt_widgets.QPushButton("")
    next_action_button.setObjectName("fit-next-action-button")
    next_action_button.setProperty("primaryAction", True)
    next_action_button.setMinimumHeight(36)
    next_action_button.setVisible(False)

    validation_list = qt_widgets.QListWidget()
    validation_list.setObjectName("fit-validation-list")

    run_label = qt_widgets.QLabel(format_fit_run_summary(fit_service.latest_run(project)))
    run_label.setObjectName("fit-run-summary")
    run_label.setWordWrap(True)

    phase_label = qt_widgets.QLabel("Ready")
    phase_label.setObjectName("fit-phase-label")
    phase_font = phase_label.font()
    phase_font.setPointSize(phase_font.pointSize() + 1)
    phase_font.setBold(True)
    phase_label.setFont(phase_font)
    phase_label.setWordWrap(True)

    log_output = qt_widgets.QPlainTextEdit()
    log_output.setObjectName("fit-log-output")
    log_output.setReadOnly(True)
    log_output.setPlaceholderText("Run logs will stream here while the fit is running.")

    log_section, _, log_layout, _log_toggle = build_collapsible_section(
        None,
        title="Detailed log",
        object_name="fit-run-log-section",
        expanded=False,
    )
    log_layout.addWidget(log_output)

    content_row_widget = qt_widgets.QSplitter(root)
    content_row_widget.setObjectName("fit-content-row")
    content_row_widget.setChildrenCollapsible(False)
    content_row_widget.setHandleWidth(8)

    preparation_panel = qt_widgets.QWidget(content_row_widget)
    preparation_panel.setObjectName("fit-preparation-panel")
    preparation_layout = qt_widgets.QVBoxLayout(preparation_panel)
    preparation_layout.setContentsMargins(12, 12, 12, 12)
    preparation_layout.setSpacing(8)
    preparation_layout.addWidget(preparation_label)
    preparation_layout.addWidget(next_action_label)
    preparation_layout.addWidget(next_action_button)
    preparation_layout.addWidget(validation_list, 1)

    convergence_widget, _add_ofv_point, _reset_convergence, _finalize_convergence = (
        build_convergence_plot_widget((qt_core, qt_gui, qt_widgets))
    )

    run_panel = qt_widgets.QWidget(content_row_widget)
    run_panel.setObjectName("fit-run-panel")
    run_layout = qt_widgets.QVBoxLayout(run_panel)
    run_layout.setContentsMargins(12, 12, 12, 12)
    run_layout.setSpacing(8)
    run_layout.addWidget(run_label)
    run_layout.addWidget(phase_label)
    run_layout.addWidget(convergence_widget)
    run_layout.addWidget(log_section, 1)

    content_row_widget.addWidget(preparation_panel)
    content_row_widget.addWidget(run_panel)
    content_row_widget.setStretchFactor(0, 2)
    content_row_widget.setStretchFactor(1, 3)
    content_row_widget.setSizes([400, 600])

    cancel_button = qt_widgets.QPushButton("Cancel")
    cancel_button.setObjectName("fit-cancel-button")
    cancel_button.setEnabled(False)
    run_button = qt_widgets.QPushButton("Run fit")
    run_button.setObjectName("fit-run-button")
    run_button.setProperty("primaryAction", True)
    run_button.setEnabled(can_start_fit_run(preparation, fit_service.latest_run(project)))
    run_button.setToolTip("Start fit run (Ctrl+R)")
    cancel_button.setToolTip("Cancel running fit (Escape)")

    action_row_widget = qt_widgets.QWidget(root)
    action_row_widget.setObjectName("fit-action-row")
    action_row = qt_widgets.QHBoxLayout(action_row_widget)
    action_row.setContentsMargins(0, 0, 0, 0)
    action_row.setSpacing(8)

    layout.addWidget(combined_header)
    layout.addWidget(title_label)
    layout.addWidget(hint_widget)
    layout.addWidget(content_row_widget, 1)

    run_progress = qt_widgets.QProgressBar()
    run_progress.setObjectName("fit-run-progress")
    run_progress.setRange(0, 0)
    run_progress.setFixedHeight(20)
    run_progress.setVisible(False)

    action_row.addStretch(1)
    action_row.addWidget(run_progress)
    action_row.addWidget(cancel_button)
    action_row.addWidget(run_button)
    layout.addWidget(action_row_widget)

    _apply_responsive_box_layout = install_responsive_box_layouts(
        root,
        breakpoint=FIT_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        layouts=(action_row,),
    )

    _apply_responsive_splitter = install_responsive_splitters(
        root,
        breakpoint=FIT_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        splitters=(content_row_widget,),
    )

    def _apply_responsive_layout(width: int | None = None) -> None:
        _apply_responsive_box_layout(width)
        _apply_responsive_splitter(width)

    run_shortcut = qt_gui.QShortcut(qt_gui.QKeySequence("Ctrl+R"), root)
    run_shortcut.activated.connect(lambda: run_button.click() if run_button.isEnabled() else None)
    cancel_shortcut = qt_gui.QShortcut(qt_gui.QKeySequence("Escape"), root)
    cancel_shortcut.activated.connect(
        lambda: cancel_button.click() if cancel_button.isEnabled() else None
    )

    future = None
    current_job: BackgroundJob | None = None
    active_run: RunRecord | None = None
    streamed_event_count = 0
    next_action_target = [""]
    pending_events: deque[JobEvent] = deque()
    pending_events_lock = threading.Lock()
    poll_timer = qt_core.QTimer(root)
    poll_timer.setInterval(100)

    def _refresh_next_action(latest_run: RunRecord | None) -> None:
        action = recommend_fit_next_action(project, preparation, latest_run)
        if action is None:
            next_action_target[0] = ""
            next_action_label.clear()
            next_action_label.setVisible(False)
            next_action_button.setText("")
            next_action_button.setToolTip("")
            next_action_button.setVisible(False)
            return
        button_text, workflow_id, summary = action
        next_action_target[0] = workflow_id
        next_action_label.setText(summary)
        next_action_label.setVisible(True)
        next_action_button.setText(button_text)
        next_action_button.setToolTip(summary)
        next_action_button.setVisible(True)

    def _render_validation(latest_run: RunRecord | None = None) -> None:
        validation_list.clear()
        if not preparation.validation.issues:
            current_run = active_run or latest_run
            if current_run is not None and current_run.status == RunStatus.SUCCEEDED:
                validation_list.addItem("Ready to run again.")
            elif current_run is not None and current_run.status in {
                RunStatus.PENDING,
                RunStatus.RUNNING,
            }:
                validation_list.addItem("Fit is running\u2026")
            elif current_run is None or current_run.status == RunStatus.FAILED:
                validation_list.addItem("Fit is ready to start.")
            return
        for issue in preparation.validation.issues:
            icon = "⚠" if issue.severity.value == "error" else "ℹ"
            item = qt_widgets.QListWidgetItem(f"{icon} {issue.message}")
            tooltip_parts = []
            if issue.field_name:
                tooltip_parts.append(f"Field: {issue.field_name}")
            target = validation_issue_target(issue)
            if target is not None:
                tooltip_parts.append(
                    f"Activate to open {target.workflow_id.title()} and focus the relevant field."
                )
                item.setData(
                    qt_core.Qt.ItemDataRole.UserRole,
                    (target.workflow_id, target.widget_object_name),
                )
            if tooltip_parts:
                item.setToolTip("\n".join(tooltip_parts))
            validation_list.addItem(item)

    def _activate_validation_issue(item) -> None:
        target = item.data(qt_core.Qt.ItemDataRole.UserRole)
        if not isinstance(target, tuple) or not target:
            return
        workflow_id = str(target[0])
        widget_object_name = str(target[1]) if len(target) > 1 and target[1] else None
        focus_callback = getattr(root, "_focus_workflow_widget", None)
        if callable(focus_callback):
            focus_callback(workflow_id, widget_object_name)
            return
        navigate_callback = getattr(root, "_navigate_to_workflow", None)
        if callable(navigate_callback):
            navigate_callback(workflow_id)

    def _render_run(run: RunRecord | None) -> None:
        run_label.setText(format_fit_run_summary(run))
        log_output.setPlainText("\n".join(run.log_lines) if run is not None else "")
        if run is None or run.status == RunStatus.PENDING:
            phase_label.setText("Ready")
        elif run.status == RunStatus.RUNNING:
            phase_label.setText("Running…")
        elif run.status == RunStatus.SUCCEEDED:
            phase_label.setText("Completed")
        elif run.status == RunStatus.FAILED:
            phase_label.setText("Failed")
        if run is not None and run.status == RunStatus.FAILED:
            run_panel.setStyleSheet("QWidget#fit-run-panel { border-left: 4px solid #e74c3c; }")
        elif run is not None and run.status == RunStatus.SUCCEEDED:
            run_panel.setStyleSheet("QWidget#fit-run-panel { border-left: 4px solid #27ae60; }")
        else:
            run_panel.setStyleSheet("")

    def _enqueue_job_event(event: JobEvent) -> None:
        with pending_events_lock:
            pending_events.append(event)

    def _clear_pending_events() -> None:
        with pending_events_lock:
            pending_events.clear()

    def _drain_pending_events() -> None:
        nonlocal streamed_event_count
        with pending_events_lock:
            drained = list(pending_events)
            pending_events.clear()
        if not drained or active_run is None:
            return
        for event in drained:
            streamed_event_count += 1
            if event.kind == "ofv":
                # Parse "iteration,ofv_value" and update live plot (don't log these)
                try:
                    iter_str, ofv_str = event.message.split(",", 1)
                    _add_ofv_point(int(iter_str), float(ofv_str))
                except (ValueError, AttributeError):
                    pass
                continue
            active_run.add_log(f"[{event.kind}] {event.message}")
            if event.kind == "info":
                phase_label.setText(event.message)
            if event.progress is not None:
                run_progress.setRange(0, 100)
                run_progress.setValue(max(0, min(100, round(event.progress * 100))))
        _render_run(active_run)

    def _reset_run_controls() -> None:
        run_progress.setVisible(False)
        run_progress.setRange(0, 0)
        cancel_button.setEnabled(False)
        cancel_button.setText("Cancel")
        # Keep convergence plot visible after run completes (shows final result)

    def _notify_project_state_changed() -> None:
        callback = getattr(root, "_project_state_changed", None)
        if callable(callback):
            callback()

    def _navigate_to_next_action() -> None:
        workflow_id = next_action_target[0]
        if not workflow_id:
            return
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(workflow_id)

    def _refresh_preparation() -> None:
        nonlocal preparation
        preparation = fit_service.prepare_run(project)
        latest_run = fit_service.latest_run(project)
        refresh_combined_header()
        preparation_label.setText(
            format_fit_preparation_summary(preparation, active_run or latest_run)
        )
        run_button.setEnabled(future is None and can_start_fit_run(preparation, latest_run))
        cancel_button.setEnabled(
            future is not None and active_run is not None and active_run.cancel_requested_at is None
        )
        _refresh_next_action(active_run or latest_run)
        _render_validation(latest_run)
        _render_run(active_run or latest_run)

    def _poll_future() -> None:
        nonlocal active_run, current_job, future, streamed_event_count
        _drain_pending_events()
        if future is None or not future.done():
            return
        outcome = future.result()
        _drain_pending_events()
        run = active_run or fit_service.latest_run(project)
        if run is not None:
            outcome_for_apply = outcome
            if streamed_event_count:
                outcome_for_apply = JobOutcome(
                    job_id=outcome.job_id,
                    name=outcome.name,
                    status=outcome.status,
                    value=outcome.value,
                    error=outcome.error,
                    events=outcome.events[streamed_event_count:],
                )
            artifacts = fit_service.apply_job_outcome(run, outcome_for_apply)
            for artifact in artifacts:
                artifact_service.register(project, artifact)
                run.add_log(f"[artifact] {artifact.kind}: {artifact.label}")
            # Finalize the OFV plot with the authoritative history
            if outcome.value is not None and hasattr(outcome.value, "ofv_history"):
                _finalize_convergence(list(outcome.value.ofv_history))
            _render_run(run)
            _notify_project_state_changed()
        future = None
        current_job = None
        active_run = None
        streamed_event_count = 0
        _clear_pending_events()
        poll_timer.stop()
        _reset_run_controls()
        _refresh_preparation()

    def _start_run() -> None:
        nonlocal active_run, current_job, future, streamed_event_count
        _refresh_preparation()
        if not can_start_fit_run(preparation, fit_service.latest_run(project)):
            return
        run = RunRecord(workflow="fit")
        run.mark_running()
        run.add_log("Fit run submitted.")
        project_service.add_run(project, run)
        active_run = run
        streamed_event_count = 0
        _clear_pending_events()
        _render_run(run)
        _notify_project_state_changed()
        n_parallel = preferences[0].n_parallel if preferences else 0
        current_job = fit_service.create_job(
            project, preparation=preparation, run_id=run.run_id, n_parallel=n_parallel
        )
        future = job_runner.submit(current_job, on_event=_enqueue_job_event)
        run_button.setEnabled(False)
        cancel_button.setEnabled(True)
        cancel_button.setText("Cancel")
        run_progress.setVisible(True)
        run_progress.setRange(0, 0)
        _reset_convergence()
        convergence_widget.setVisible(True)
        poll_timer.start()

    def _cancel_run() -> None:
        nonlocal current_job
        if future is None or current_job is None or active_run is None:
            return
        current_job.request_cancel()
        cancel_button.setEnabled(False)
        cancel_button.setText("Cancelling…")
        active_run.mark_cancel_requested()
        active_run.add_log(
            "[status] Cancellation requested. Waiting for the current step to finish."
        )
        _render_run(active_run)
        _notify_project_state_changed()

    cancel_button.clicked.connect(_cancel_run)
    run_button.clicked.connect(_start_run)
    next_action_button.clicked.connect(_navigate_to_next_action)
    poll_timer.timeout.connect(_poll_future)
    validation_list.itemActivated.connect(_activate_validation_issue)
    root.destroyed.connect(lambda *_args: job_runner.shutdown(wait=False))
    root._apply_responsive_layout = _apply_responsive_layout  # type: ignore[attr-defined]
    root._refresh_workflow = _refresh_preparation  # type: ignore[attr-defined]
    root._refresh_context_header = refresh_combined_header  # type: ignore[attr-defined]

    _refresh_preparation()
    _apply_responsive_layout()
    return root
