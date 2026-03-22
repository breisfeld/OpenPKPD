"""NCA workflow widget for running standalone non-compartmental analyses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    default_workspace_root_path,
    load_gui_preferences,
    save_gui_preferences,
    with_dismissed_hint,
)
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.artifact_service import ArtifactService
from openpkpd_gui.services.nca_service import NCAConfig, NCAPreparationResult, NCAService
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.widgets.dismissible_hint import build_dismissible_hint
from openpkpd_gui.widgets.responsive_layout import (
    install_responsive_box_layouts,
    install_responsive_splitters,
)
from openpkpd_gui.widgets.scrollable_page import build_scrollable_page

NCA_RESPONSIVE_LAYOUT_BREAKPOINT = 1000


def format_nca_preparation_summary(preparation: NCAPreparationResult) -> str:
    """Return a compact readiness summary for the NCA workflow."""
    dataset = preparation.dataset_path or "no dataset"
    status = "Ready to run NCA" if preparation.ready else "NCA needs attention"
    return (
        f"{status} — dataset {dataset} • {preparation.subject_count} subjects • "
        f"{preparation.observation_count} observations • {preparation.row_count} rows"
    )


def format_nca_run_summary(run: RunRecord | None) -> str:
    """Render a concise status line for the latest NCA run."""
    if run is None:
        return "No NCA runs yet."
    if run.status == RunStatus.SUCCEEDED and run.summary_text:
        return f"Latest NCA — Succeeded • {run.summary_text}"
    if run.status == RunStatus.FAILED and run.error_text:
        return f"Latest NCA — Failed • {run.error_text}"
    if run.status == RunStatus.CANCELLED:
        return "Latest NCA — Cancelled"
    return f"Latest NCA — {run.status.value.title()}"


def nca_config_matches_artifact(config: NCAConfig, artifact: ArtifactRecord | None) -> bool:
    """Return whether the latest NCA artifact reflects the current NCA options."""
    if artifact is None or artifact.metadata.get("artifact_role") != "nca_summary":
        return False
    min_points = artifact.metadata.get("min_points_lambda")
    try:
        artifact_min_points = int(min_points) if min_points is not None else None
    except (TypeError, ValueError):
        artifact_min_points = None
    exclude_cmax = artifact.metadata.get("exclude_cmax")
    if isinstance(exclude_cmax, str):
        artifact_exclude_cmax = exclude_cmax.strip().lower() == "true"
    else:
        artifact_exclude_cmax = exclude_cmax
    return (
        artifact.metadata.get("route") == config.route
        and artifact.metadata.get("auc_method") == config.auc_method
        and artifact_min_points == config.min_points_lambda
        and artifact_exclude_cmax == config.exclude_cmax
    )


def can_start_nca_run(
    preparation: NCAPreparationResult,
    latest_run: RunRecord | None,
    latest_artifact: ArtifactRecord | None,
    config: NCAConfig,
) -> bool:
    """Return whether the NCA workflow should allow starting a new run."""
    if not preparation.ready:
        return False
    if latest_run is None:
        return True
    if latest_run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
        return False
    if latest_run.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
        return True
    return not nca_config_matches_artifact(config, latest_artifact)


def recommend_nca_next_action(
    preparation: NCAPreparationResult,
    latest_run: RunRecord | None,
    latest_artifact: ArtifactRecord | None,
    config: NCAConfig,
) -> tuple[str, str, str] | None:
    """Return the primary CTA for blocked or already-satisfied NCA states."""
    if latest_run is not None and latest_run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
        return None
    if not preparation.ready:
        message = "Load a dataset in the Data workflow before starting NCA."
        return ("Open Data", "data", message)
    if (
        latest_run is not None
        and latest_run.status == RunStatus.SUCCEEDED
        and latest_artifact is not None
        and nca_config_matches_artifact(config, latest_artifact)
    ):
        return (
            "Open latest CSV",
            "__open_latest_results__",
            "Latest NCA results already match the current options. Open the saved summary or change an option to rerun.",
        )
    return None


def latest_nca_artifact(project: Workspace) -> ArtifactRecord | None:
    """Return the newest NCA summary artifact for the selected scenario."""
    for artifact in reversed(project.active_scenario.artifacts):
        if artifact.metadata.get("artifact_role") == "nca_summary":
            return artifact
    return None


def load_nca_preview_text(path: str | None, *, max_rows: int = 8) -> str:
    """Load a small text preview for an NCA results CSV artifact."""
    if not path:
        return "No NCA results are available yet."
    csv_path = Path(path)
    if not csv_path.exists():
        return f"Latest NCA results file is not available on disk: {csv_path}"
    preview = pd.read_csv(csv_path, nrows=max_rows)
    preview = preview.astype(object).where(preview.notna(), "")
    return preview.to_csv(index=False)


def build_nca_workflow(
    project: Workspace,
    nca_service: NCAService | None = None,
    project_service: ProjectService | None = None,
    job_runner: JobRunner | None = None,
):
    """Build the first real NCA workflow page."""
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    nca_service = nca_service or NCAService()
    project_service = project_service or ProjectService()
    artifact_service = ArtifactService()
    job_runner = job_runner or JobRunner(max_workers=1)
    preparation = nca_service.prepare_run(project)

    root, _, layout, scroll_area = build_scrollable_page(
        qt_widgets, root_object_name="nca-workflow"
    )

    title_label = qt_widgets.QLabel("NCA workflow")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 4)
    title_font.setBold(True)
    title_label.setFont(title_font)

    _hint_prefs = load_gui_preferences()

    def _save_dismissed_hint_nca():
        save_gui_preferences(with_dismissed_hint(load_gui_preferences(), "hint_nca"))

    hint_widget, _ = build_dismissible_hint(
        "Run standalone non-compartmental analysis on the active dataset, export the subject-level "
        "summary table, and preview the latest results.",
        dismissed="hint_nca" in _hint_prefs.dismissed_hints,
        on_dismiss=_save_dismissed_hint_nca,
    )

    preparation_label = qt_widgets.QLabel(format_nca_preparation_summary(preparation))
    preparation_label.setObjectName("nca-preparation-summary")
    preparation_label.setWordWrap(True)

    next_action_label = qt_widgets.QLabel("")
    next_action_label.setObjectName("nca-next-action-label")
    next_action_label.setWordWrap(True)
    next_action_label.setVisible(False)

    next_action_button = qt_widgets.QPushButton("")
    next_action_button.setObjectName("nca-next-action-button")
    next_action_button.setProperty("primaryAction", True)
    next_action_button.setMinimumHeight(36)
    next_action_button.setVisible(False)

    validation_list = qt_widgets.QListWidget()
    validation_list.setObjectName("nca-validation-list")

    options_row_widget = qt_widgets.QWidget(root)
    options_row_widget.setObjectName("nca-options-row")
    options_row = qt_widgets.QHBoxLayout(options_row_widget)
    options_row.setContentsMargins(0, 0, 0, 0)
    options_row.setSpacing(8)
    route_combo = qt_widgets.QComboBox()
    route_combo.setObjectName("nca-route-combo")
    route_combo.addItems(["oral", "IV", "infusion"])
    auc_method_combo = qt_widgets.QComboBox()
    auc_method_combo.setObjectName("nca-auc-method-combo")
    auc_method_combo.addItems(["linear-log", "linear-trapezoidal", "linear-up-log-down"])
    min_points_spin = qt_widgets.QSpinBox()
    min_points_spin.setObjectName("nca-min-points-spin")
    min_points_spin.setRange(3, 8)
    min_points_spin.setValue(3)
    exclude_cmax_check = qt_widgets.QCheckBox("Exclude Cmax from terminal regression")
    exclude_cmax_check.setObjectName("nca-exclude-cmax-check")
    exclude_cmax_check.setChecked(True)
    options_row.addWidget(qt_widgets.QLabel("Route"))
    options_row.addWidget(route_combo)
    options_row.addWidget(qt_widgets.QLabel("AUC method"))
    options_row.addWidget(auc_method_combo, 1)
    options_row.addWidget(qt_widgets.QLabel("Min λz points"))
    options_row.addWidget(min_points_spin)
    options_row.addWidget(exclude_cmax_check)

    run_label = qt_widgets.QLabel(format_nca_run_summary(nca_service.latest_run(project)))
    run_label.setObjectName("nca-run-summary")
    run_label.setWordWrap(True)

    results_label = qt_widgets.QLabel("No NCA results are available yet.")
    results_label.setObjectName("nca-results-summary")
    results_label.setWordWrap(True)

    preview_output = qt_widgets.QPlainTextEdit()
    preview_output.setObjectName("nca-results-preview")
    preview_output.setReadOnly(True)
    preview_output.setPlaceholderText("The latest NCA CSV preview will appear here.")

    log_output = qt_widgets.QPlainTextEdit()
    log_output.setObjectName("nca-log-output")
    log_output.setReadOnly(True)
    log_output.setPlaceholderText("NCA run logs will appear here after execution.")

    content_row_widget = qt_widgets.QSplitter(root)
    content_row_widget.setObjectName("nca-content-row")
    content_row_widget.setChildrenCollapsible(False)
    content_row_widget.setHandleWidth(8)

    readiness_panel = qt_widgets.QWidget(content_row_widget)
    readiness_panel.setObjectName("nca-readiness-panel")
    readiness_layout = qt_widgets.QVBoxLayout(readiness_panel)
    readiness_layout.setContentsMargins(12, 12, 12, 12)
    readiness_layout.setSpacing(8)
    readiness_layout.addWidget(preparation_label)
    readiness_layout.addWidget(next_action_label)
    readiness_layout.addWidget(next_action_button)
    readiness_layout.addWidget(validation_list, 1)

    results_panel = qt_widgets.QWidget(content_row_widget)
    results_panel.setObjectName("nca-results-panel")
    results_layout = qt_widgets.QVBoxLayout(results_panel)
    results_layout.setContentsMargins(12, 12, 12, 12)
    results_layout.setSpacing(8)
    results_layout.addWidget(run_label)
    results_layout.addWidget(results_label)
    results_layout.addWidget(preview_output, 1)
    results_layout.addWidget(log_output, 1)

    content_row_widget.addWidget(readiness_panel)
    content_row_widget.addWidget(results_panel)
    content_row_widget.setStretchFactor(0, 2)
    content_row_widget.setStretchFactor(1, 3)
    content_row_widget.setSizes([400, 600])

    cancel_button = qt_widgets.QPushButton("Cancel")
    cancel_button.setObjectName("nca-cancel-button")
    cancel_button.setEnabled(False)
    run_button = qt_widgets.QPushButton("Run NCA")
    run_button.setObjectName("nca-run-button")
    run_button.setProperty("primaryAction", True)
    run_button.setEnabled(False)
    open_results_button = qt_widgets.QPushButton("Open latest CSV")
    open_results_button.setObjectName("nca-open-latest-results-button")
    open_results_button.setEnabled(False)
    open_folder_button = qt_widgets.QPushButton("Open artifacts folder")
    open_folder_button.setObjectName("nca-open-artifacts-folder-button")
    open_folder_button.setEnabled(False)

    action_row_widget = qt_widgets.QWidget(root)
    action_row_widget.setObjectName("nca-action-row")
    action_row = qt_widgets.QHBoxLayout(action_row_widget)
    action_row.setContentsMargins(0, 0, 0, 0)
    action_row.setSpacing(8)

    layout.addWidget(title_label)
    layout.addWidget(hint_widget)
    layout.addWidget(options_row_widget)
    layout.addWidget(content_row_widget, 1)
    run_progress = qt_widgets.QProgressBar()
    run_progress.setObjectName("nca-run-progress")
    run_progress.setRange(0, 0)
    run_progress.setFixedHeight(20)
    run_progress.setVisible(False)

    action_row.addWidget(open_results_button)
    action_row.addWidget(open_folder_button)
    action_row.addStretch(1)
    action_row.addWidget(run_progress)
    action_row.addWidget(cancel_button)
    action_row.addWidget(run_button)
    layout.addWidget(action_row_widget)

    _apply_responsive_box_layout = install_responsive_box_layouts(
        root,
        breakpoint=NCA_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        layouts=(options_row, action_row),
    )

    _apply_responsive_splitter = install_responsive_splitters(
        root,
        breakpoint=NCA_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        splitters=(content_row_widget,),
    )

    def _apply_responsive_layout(width: int | None = None) -> None:
        _apply_responsive_box_layout(width)
        _apply_responsive_splitter(width)

    future = None
    next_action_target = [""]
    poll_timer = qt_core.QTimer(root)
    poll_timer.setInterval(100)

    def _current_config() -> NCAConfig:
        return NCAConfig(
            route=route_combo.currentText(),
            auc_method=auc_method_combo.currentText(),
            min_points_lambda=min_points_spin.value(),
            exclude_cmax=exclude_cmax_check.isChecked(),
        )

    def _notify_project_state_changed() -> None:
        callback = getattr(root, "_project_state_changed", None)
        if callable(callback):
            callback()

    def _refresh_next_action() -> None:
        action = recommend_nca_next_action(
            preparation,
            nca_service.latest_run(project),
            latest_nca_artifact(project),
            _current_config(),
        )
        if action is None:
            next_action_target[0] = ""
            next_action_label.clear()
            next_action_label.setVisible(False)
            next_action_button.setText("")
            next_action_button.setToolTip("")
            next_action_button.setVisible(False)
            return
        button_text, target, summary = action
        next_action_target[0] = target
        next_action_label.setText(summary)
        next_action_label.setVisible(True)
        next_action_button.setText(button_text)
        next_action_button.setToolTip(summary)
        next_action_button.setVisible(True)

    def _navigate_to_next_action() -> None:
        target = next_action_target[0]
        if not target:
            return
        if target == "__open_latest_results__":
            _open_latest_results()
            return
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(target)

    def _update_run_availability() -> None:
        run_button.setEnabled(
            future is None
            and can_start_nca_run(
                preparation,
                nca_service.latest_run(project),
                latest_nca_artifact(project),
                _current_config(),
            )
        )
        _refresh_next_action()

    def _render_validation() -> None:
        validation_list.clear()
        if not preparation.validation.issues:
            validation_list.addItem("NCA is ready to start.")
            return
        for issue in preparation.validation.issues:
            prefix = issue.severity.value.title()
            field = f" [{issue.field_name}]" if issue.field_name else ""
            validation_list.addItem(f"{prefix}{field}: {issue.message}")

    def _render_run(run: RunRecord | None) -> None:
        run_label.setText(format_nca_run_summary(run))
        log_output.setPlainText("\n".join(run.log_lines) if run is not None else "")

    def _render_latest_results() -> None:
        artifact = latest_nca_artifact(project)
        if artifact is None:
            results_label.setText("No NCA results are available yet.")
            preview_output.setPlainText("No NCA results are available yet.")
            open_results_button.setEnabled(False)
            open_folder_button.setEnabled(False)
            return
        subject_count = artifact.metadata.get("subject_count")
        subject_text = f"{subject_count} subjects" if subject_count is not None else artifact.label
        results_label.setText(
            f"Latest results • {subject_text} • {artifact.path or 'in-memory metadata'}"
        )
        preview_output.setPlainText(load_nca_preview_text(artifact.path))
        open_results_button.setEnabled(bool(artifact.path))
        open_folder_button.setEnabled(bool(artifact.path))

    def _refresh() -> None:
        nonlocal preparation
        preparation = nca_service.prepare_run(project)
        preparation_label.setText(format_nca_preparation_summary(preparation))
        _update_run_availability()
        _render_validation()
        _render_run(nca_service.latest_run(project))
        _render_latest_results()

    def _poll_future() -> None:
        nonlocal future
        if future is None or not future.done():
            return
        outcome = future.result()
        run = nca_service.latest_run(project)
        if run is not None:
            artifacts = nca_service.apply_job_outcome(run, outcome)
            for artifact in artifacts:
                artifact_service.register(project, artifact)
                run.add_log(f"[artifact] {artifact.kind}: {artifact.label}")
            _notify_project_state_changed()
        future = None
        poll_timer.stop()
        run_progress.setVisible(False)
        cancel_button.setEnabled(False)
        _refresh()

    def _start_run() -> None:
        nonlocal future
        _refresh()
        if not can_start_nca_run(
            preparation,
            nca_service.latest_run(project),
            latest_nca_artifact(project),
            _current_config(),
        ):
            return
        run = RunRecord(workflow="nca")
        run.mark_running()
        run.add_log("NCA run submitted.")
        project_service.add_run(project, run)
        _render_run(run)
        _refresh_next_action()
        _notify_project_state_changed()
        future = job_runner.submit(
            nca_service.create_job(
                project, config=_current_config(), preparation=preparation, run_id=run.run_id
            )
        )
        run_button.setEnabled(False)
        cancel_button.setEnabled(True)
        run_progress.setVisible(True)
        poll_timer.start()

    def _cancel_run() -> None:
        nonlocal future
        if future is None:
            return
        future.cancel()
        future = None
        poll_timer.stop()
        run_progress.setVisible(False)
        cancel_button.setEnabled(False)
        run = nca_service.latest_run(project)
        if run is not None:
            run.mark_cancelled()
            _render_run(run)
            _notify_project_state_changed()
        _refresh()

    def _open_latest_results() -> None:
        artifact = latest_nca_artifact(project)
        if artifact is None or not artifact.path:
            return
        qt_gui.QDesktopServices.openUrl(qt_core.QUrl.fromLocalFile(artifact.path))

    def _open_artifact_folder() -> None:
        artifact = latest_nca_artifact(project)
        target = (
            Path(artifact.path).resolve().parent
            if artifact is not None and artifact.path
            else default_workspace_root_path()
        )
        qt_gui.QDesktopServices.openUrl(qt_core.QUrl.fromLocalFile(str(target)))

    cancel_button.clicked.connect(_cancel_run)
    run_button.clicked.connect(_start_run)
    open_results_button.clicked.connect(_open_latest_results)
    open_folder_button.clicked.connect(_open_artifact_folder)
    next_action_button.clicked.connect(_navigate_to_next_action)
    poll_timer.timeout.connect(_poll_future)
    route_combo.currentTextChanged.connect(lambda *_args: _update_run_availability())
    auc_method_combo.currentTextChanged.connect(lambda *_args: _update_run_availability())
    min_points_spin.valueChanged.connect(lambda *_args: _update_run_availability())
    exclude_cmax_check.toggled.connect(lambda *_args: _update_run_availability())
    root.destroyed.connect(lambda *_args: job_runner.shutdown(wait=False))
    root._apply_responsive_layout = _apply_responsive_layout  # type: ignore[attr-defined]
    root._refresh_workflow = _refresh  # type: ignore[attr-defined]

    _refresh()
    _apply_responsive_layout()
    return root
