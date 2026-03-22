"""Diagnostics workflow widget for fit-readiness summaries and outputs."""

from __future__ import annotations

from collections import Counter
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    load_gui_preferences,
    save_gui_preferences,
    with_dismissed_hint,
)
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.artifact_service import ArtifactService
from openpkpd_gui.services.fit_service import FitService
from openpkpd_gui.services.npde_service import NPDEService
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.workflow_state_service import describe_fit_input_changes
from openpkpd_gui.widgets.combined_header import build_combined_header
from openpkpd_gui.widgets.dismissible_hint import build_dismissible_hint
from openpkpd_gui.widgets.output_browser import (
    build_output_preview_panel,
    export_output_file,
    open_output_file,
)
from openpkpd_gui.widgets.responsive_layout import (
    install_responsive_box_layouts,
    install_responsive_splitters,
)
from openpkpd_gui.widgets.scrollable_page import build_scrollable_page
from openpkpd_gui.workflows.results_workflow import (
    artifact_plot_type_options,
    artifact_role_options,
    filter_artifacts,
    latest_artifact,
)

DIAGNOSTICS_RESPONSIVE_LAYOUT_BREAKPOINT = 1000


def latest_fit_run(project: Workspace) -> RunRecord | None:
    """Return the most recent fit run for the selected scenario, if any."""
    for run in reversed(project.active_scenario.runs):
        if run.workflow == "fit":
            return run
    return None


def diagnostics_backend_available() -> bool:
    """Return whether the diagnostics plotting APIs are import-discoverable."""
    return find_spec("openpkpd.plots.diagnostics") is not None


def plotting_backend_available() -> bool:
    """Return whether matplotlib can be imported in the current environment."""
    try:
        import_module("matplotlib")
    except Exception:
        return False
    return True


def diagnostics_artifacts(
    project: Workspace,
    run: RunRecord | None = None,
    artifact_service: ArtifactService | None = None,
) -> list[ArtifactRecord]:
    """Return selected-scenario artifacts relevant to diagnostics browsing."""
    artifact_service = artifact_service or ArtifactService()
    if run is None:
        return list(project.active_scenario.artifacts)
    run_artifacts = artifact_service.list_for_run(project, run.run_id)
    if run_artifacts:
        return run_artifacts
    return [
        artifact
        for artifact in project.active_scenario.artifacts
        if artifact.artifact_id in run.artifact_ids
    ]


def format_diagnostics_overview(project: Workspace) -> str:
    """Return a compact overview of selected-scenario diagnostics prerequisites and outputs."""
    run = latest_fit_run(project)
    dataset_text = "dataset ready" if project.active_dataset is not None else "no dataset"
    model_text = "model ready" if project.active_model_spec is not None else "no model"
    fit_text = run.status.value.title() if run is not None else "No fit run"
    return (
        f"Scenario {project.active_scenario.name} • {dataset_text} • {model_text} • latest fit {fit_text} • "
        f"{len(project.active_scenario.artifacts)} scenario outputs"
    )


def format_diagnostics_stale_warning(project: Workspace) -> str:
    """Return a stale-diagnostics warning when saved inputs outpace the latest successful fit."""
    subject = describe_fit_input_changes(project.active_scenario)
    if subject is None:
        return ""
    return f"{subject} since the latest successful fit. Diagnostics may be stale until you rerun the fit."


def format_diagnostics_status(
    project: Workspace,
    *,
    diagnostics_api_available: bool | None = None,
    plotting_available: bool | None = None,
) -> str:
    """Summarize whether the project is in a good state for diagnostics work."""
    diagnostics_api_available = (
        diagnostics_backend_available()
        if diagnostics_api_available is None
        else diagnostics_api_available
    )
    plotting_available = (
        plotting_backend_available() if plotting_available is None else plotting_available
    )
    run = latest_fit_run(project)
    if run is None:
        return (
            "Diagnostics need a completed fit run before residuals and GOF outputs can be reviewed."
        )
    if run.status == RunStatus.RUNNING:
        return (
            "A fit is currently running. Refresh after it completes to review diagnostic readiness."
        )
    if run.status == RunStatus.FAILED:
        return "The latest fit failed. Resolve model or data issues before expecting diagnostics output."
    if diagnostics_api_available and plotting_available:
        return (
            "The latest fit succeeded and the plotting stack looks available for future GOF, "
            "residual, ETA, and profile rendering."
        )
    if diagnostics_api_available:
        return (
            "The latest fit succeeded, but matplotlib is not currently discoverable for figure "
            "rendering in this environment."
        )
    return "The latest fit succeeded, but diagnostics plotting APIs are not currently discoverable."


def format_diagnostics_next_steps(project: Workspace) -> str:
    """Return concise next-step guidance for the Diagnostics page."""
    steps: list[str] = []
    if project.active_dataset is None:
        steps.append("• Load a dataset in Data.")
    if project.active_model_spec is None:
        steps.append("• Open the Model workflow to configure a model.")

    run = latest_fit_run(project)
    if run is None:
        steps.append("• Run a fit in Fit to generate estimation outputs.")
    elif run.status == RunStatus.RUNNING:
        steps.append("• Wait for the active fit to complete, then refresh this page.")
    elif run.status == RunStatus.FAILED:
        steps.append("• Inspect the failed run in Results and correct the model before rerunning.")
    else:
        steps.append("• Review fit logs and outputs in Results.")
        steps.append(
            "• Use Results quick review and this page to inspect GOF, residuals, ETA/profile plots, "
            "exported diagnostics tables, and on-demand NPDE outputs."
        )
    return "\n".join(steps)


def recommend_diagnostics_next_action(
    project: Workspace,
    artifacts: list[ArtifactRecord],
) -> tuple[str, str, str] | None:
    """Return the primary CTA for an empty or blocked Diagnostics page."""
    if artifacts:
        return None
    if project.active_dataset is None:
        return (
            "Open Data",
            "data",
            "Load a dataset before expecting diagnostics outputs for this scenario.",
        )
    if project.active_model_spec is None:
        return (
            "Open Model",
            "model",
            "Configure a model in the Model workflow first.",
        )
    run = latest_fit_run(project)
    if run is None:
        return (
            "Open Fit",
            "fit",
            "Run a fit to unlock diagnostics review for this scenario.",
        )
    if run.status == RunStatus.RUNNING:
        return (
            "Open Fit",
            "fit",
            "A fit is active. Monitor progress in Fit and refresh Diagnostics when it completes.",
        )
    if run.status == RunStatus.FAILED:
        return (
            "Open Results",
            "results",
            "Inspect the failed fit in Results before expecting diagnostics outputs.",
        )
    return (
        "Open Results",
        "results",
        "The latest fit succeeded. Review the saved run outputs in Results while diagnostics outputs are not yet available.",
    )


def format_npde_generation_status(
    project: Workspace,
    *,
    fit_context_available: bool,
    npde_available: bool,
    generation_running: bool = False,
) -> str:
    """Summarize whether NPDE can be generated for the latest fit."""
    run = latest_fit_run(project)
    if run is None:
        return "NPDE generation needs a completed fit run before it can start."
    if run.status == RunStatus.RUNNING:
        return "A fit is currently running. Wait for it to finish before generating NPDE."
    if run.status == RunStatus.FAILED:
        return "The latest fit failed. Resolve fit issues before generating NPDE."
    if generation_running:
        return "NPDE generation is running in the background for the latest fit."
    if npde_available:
        return "NPDE outputs are already available for the latest fit."
    if fit_context_available:
        return "Generate NPDE on demand to create background-linked outputs for the latest fit."
    return "NPDE generation requires a fit completed in this session. Re-run the fit to enable it."


def format_diagnostics_artifact_summary(artifacts: list[ArtifactRecord]) -> str:
    """Return artifact counts grouped by kind for diagnostics browsing."""
    if not artifacts:
        return "0 outputs"
    counts = Counter(artifact.kind for artifact in artifacts)
    detail = " • ".join(f"{kind}: {count}" for kind, count in sorted(counts.items()))
    return f"{len(artifacts)} outputs • {detail}"


def format_diagnostics_artifact_label(artifact: ArtifactRecord) -> str:
    """Render one diagnostics artifact row."""
    path_text = artifact.path or "in-memory metadata"
    return f"{artifact.kind}: {artifact.label} • {path_text}"


def filter_diagnostics_artifacts(
    artifacts: list[ArtifactRecord],
    role_filter: str = "All roles",
    plot_type_filter: str = "All plot types",
) -> list[ArtifactRecord]:
    """Filter diagnostics artifacts by logical role and plot type."""
    return filter_artifacts(artifacts, "All kinds", role_filter, plot_type_filter)


def latest_diagnostics_artifact(
    artifacts: list[ArtifactRecord],
    *,
    plot_type: str | None = None,
) -> ArtifactRecord | None:
    """Return the newest diagnostics artifact, optionally for one plot type."""
    return latest_artifact(artifacts, role="plot", plot_type=plot_type)


def latest_diagnostics_role_artifact(
    artifacts: list[ArtifactRecord],
    role: str,
) -> ArtifactRecord | None:
    """Return the newest diagnostics-related artifact for one logical role."""
    return latest_artifact(artifacts, role=role)


def build_diagnostics_workflow(
    project: Workspace,
    artifact_service: ArtifactService | None = None,
    fit_service: FitService | None = None,
    npde_service: NPDEService | None = None,
    project_service: ProjectService | None = None,
    job_runner: JobRunner | None = None,
):
    """Build a conservative first Diagnostics workflow page."""
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    artifact_service = artifact_service or ArtifactService()
    fit_service = fit_service or FitService()
    npde_service = npde_service or NPDEService()
    project_service = project_service or ProjectService()
    owns_job_runner = job_runner is None
    job_runner = job_runner or JobRunner(max_workers=1)

    root, _, layout, scroll_area = build_scrollable_page(
        qt_widgets, root_object_name="diagnostics-workflow"
    )

    combined_header, refresh_combined_header = build_combined_header(
        root,
        project,
        workflow_id="diagnostics",
        workflow_label="Diagnostics",
        status_workflow_ids=("data", "model", "fit", "results", "diagnostics", "advanced"),
    )

    title_label = qt_widgets.QLabel("Diagnostics workflow")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 4)
    title_font.setBold(True)
    title_label.setFont(title_font)

    _hint_prefs = load_gui_preferences()

    def _save_dismissed_hint_diagnostics():
        save_gui_preferences(with_dismissed_hint(load_gui_preferences(), "hint_diagnostics"))

    hint_widget, _ = build_dismissible_hint(
        "Use this page to judge whether the current scenario is ready for diagnostic review and "
        "to browse any saved outputs tied to the latest fit run.",
        dismissed="hint_diagnostics" in _hint_prefs.dismissed_hints,
        on_dismiss=_save_dismissed_hint_diagnostics,
    )

    overview_label = qt_widgets.QLabel(format_diagnostics_overview(project))
    overview_label.setObjectName("diagnostics-overview-label")
    overview_label.setWordWrap(True)

    stale_warning_label = qt_widgets.QLabel(format_diagnostics_stale_warning(project))
    stale_warning_label.setObjectName("diagnostics-stale-warning-label")
    stale_warning_label.setWordWrap(True)
    stale_warning_label.setVisible(bool(stale_warning_label.text()))

    status_label = qt_widgets.QLabel(format_diagnostics_status(project))
    status_label.setObjectName("diagnostics-status-label")
    status_label.setWordWrap(True)

    next_steps_label = qt_widgets.QLabel(format_diagnostics_next_steps(project))
    next_steps_label.setObjectName("diagnostics-next-steps-label")
    next_steps_label.setWordWrap(True)

    next_action_label = qt_widgets.QLabel("")
    next_action_label.setObjectName("diagnostics-next-action-label")
    next_action_label.setWordWrap(True)
    next_action_label.setVisible(False)

    next_action_button = qt_widgets.QPushButton("")
    next_action_button.setObjectName("diagnostics-next-action-button")
    next_action_button.setProperty("primaryAction", True)
    next_action_button.setMinimumHeight(36)
    next_action_button.setVisible(False)

    capabilities_label = qt_widgets.QLabel(
        "Available diagnostics include GOF panels, DV/PRED views, CWRES plots, Q-Q summaries, "
        "mean profiles, spaghetti plots, ETA histograms, ETA pairs, exported diagnostics CSV tables, "
        "and on-demand NPDE output generation."
    )
    capabilities_label.setWordWrap(True)

    npde_status_label = qt_widgets.QLabel(
        format_npde_generation_status(
            project,
            fit_context_available=False,
            npde_available=False,
        )
    )
    npde_status_label.setObjectName("diagnostics-npde-status-label")
    npde_status_label.setWordWrap(True)

    filter_row_widget = qt_widgets.QWidget(root)
    filter_row_widget.setObjectName("diagnostics-filter-row")
    filter_row = qt_widgets.QHBoxLayout(filter_row_widget)
    filter_row.setContentsMargins(0, 0, 0, 0)
    filter_row.setSpacing(8)
    filter_row.addWidget(qt_widgets.QLabel("Role"))
    artifact_role_combo = qt_widgets.QComboBox()
    artifact_role_combo.setObjectName("diagnostics-artifact-role-filter")
    artifact_role_combo.setProperty("persistComboSelection", True)
    filter_row.addWidget(artifact_role_combo, 1)
    filter_row.addWidget(qt_widgets.QLabel("Plot type"))
    artifact_plot_type_combo = qt_widgets.QComboBox()
    artifact_plot_type_combo.setObjectName("diagnostics-artifact-plot-type-filter")
    artifact_plot_type_combo.setProperty("persistComboSelection", True)
    filter_row.addWidget(artifact_plot_type_combo, 1)

    artifact_summary_label = qt_widgets.QLabel("0 outputs")
    artifact_summary_label.setObjectName("diagnostics-artifact-summary-label")
    artifact_summary_label.setWordWrap(True)

    artifacts_list = qt_widgets.QListWidget()
    artifacts_list.setObjectName("diagnostics-artifacts-list")
    artifacts_list.setProperty("persistListSelection", True)

    preview_panel_obj = build_output_preview_panel(
        root,
        object_prefix="diagnostics-artifact",
        placeholder_text="Preview supported for HTML, text, and image diagnostics outputs.",
    )
    artifact_preview_title = preview_panel_obj.title_label
    artifact_preview_metadata = preview_panel_obj.metadata_label
    artifact_preview_stack = preview_panel_obj.stack

    artifact_browser_row_widget = qt_widgets.QSplitter(root)
    artifact_browser_row_widget.setObjectName("diagnostics-content-row")
    artifact_browser_row_widget.setChildrenCollapsible(False)
    artifact_browser_row_widget.setHandleWidth(8)

    artifact_list_panel = qt_widgets.QWidget(artifact_browser_row_widget)
    artifact_list_panel.setObjectName("diagnostics-artifact-list-panel")
    artifact_list_column = qt_widgets.QVBoxLayout(artifact_list_panel)
    artifact_list_column.setContentsMargins(12, 12, 12, 12)
    artifact_list_column.setSpacing(8)
    artifact_list_column.addWidget(artifact_summary_label)
    artifact_list_column.addWidget(artifacts_list, 1)

    artifact_preview_panel = qt_widgets.QWidget(artifact_browser_row_widget)
    artifact_preview_panel.setObjectName("diagnostics-artifact-preview-panel")
    artifact_preview_column = qt_widgets.QVBoxLayout(artifact_preview_panel)
    artifact_preview_column.setContentsMargins(12, 12, 12, 12)
    artifact_preview_column.setSpacing(8)
    artifact_preview_column.addWidget(artifact_preview_title)
    artifact_preview_column.addWidget(artifact_preview_metadata)
    artifact_preview_column.addWidget(artifact_preview_stack, 1)

    artifact_browser_row_widget.addWidget(artifact_list_panel)
    artifact_browser_row_widget.addWidget(artifact_preview_panel)
    artifact_browser_row_widget.setStretchFactor(0, 2)
    artifact_browser_row_widget.setStretchFactor(1, 3)
    artifact_browser_row_widget.setSizes([400, 600])

    view_menu = qt_widgets.QMenu(root)
    open_gof_action = view_menu.addAction("Open GOF panel")
    open_gof_action.setObjectName("diagnostics-view-gof-panel-action")
    open_gof_action.setEnabled(False)
    open_residual_action = view_menu.addAction("Open residual trends")
    open_residual_action.setObjectName("diagnostics-view-residual-trends-action")
    open_residual_action.setEnabled(False)
    open_diagnostics_table_action = view_menu.addAction("Open diagnostics CSV")
    open_diagnostics_table_action.setObjectName("diagnostics-view-diagnostics-table-action")
    open_diagnostics_table_action.setEnabled(False)
    open_npde_table_action = view_menu.addAction("Open NPDE CSV")
    open_npde_table_action.setObjectName("diagnostics-view-npde-table-action")
    open_npde_table_action.setEnabled(False)
    view_dropdown = qt_widgets.QToolButton()
    view_dropdown.setObjectName("diagnostics-view-dropdown-button")
    view_dropdown.setText("View ▾")
    view_dropdown.setMenu(view_menu)
    view_dropdown.setPopupMode(qt_widgets.QToolButton.ToolButtonPopupMode.InstantPopup)

    output_menu = qt_widgets.QMenu(root)
    open_artifact_action = output_menu.addAction("Open output")
    open_artifact_action.setObjectName("diagnostics-output-open-action")
    open_artifact_action.setEnabled(False)
    export_artifact_action = output_menu.addAction("Save copy…")
    export_artifact_action.setObjectName("diagnostics-output-save-copy-action")
    export_artifact_action.setEnabled(False)
    open_folder_action = output_menu.addAction("Open folder")
    open_folder_action.setObjectName("diagnostics-output-open-folder-action")
    open_folder_action.setEnabled(False)
    output_dropdown = qt_widgets.QToolButton()
    output_dropdown.setObjectName("diagnostics-output-dropdown-button")
    output_dropdown.setText("Output ▾")
    output_dropdown.setMenu(output_menu)
    output_dropdown.setPopupMode(qt_widgets.QToolButton.ToolButtonPopupMode.InstantPopup)

    generate_npde_button = qt_widgets.QPushButton("Generate NPDE outputs")
    generate_npde_button.setObjectName("diagnostics-generate-npde-button")
    generate_npde_button.setProperty("primaryAction", True)
    generate_npde_button.setEnabled(False)

    action_row_widget = qt_widgets.QWidget(root)
    action_row_widget.setObjectName("diagnostics-action-row")
    action_row = qt_widgets.QHBoxLayout(action_row_widget)
    action_row.setContentsMargins(0, 0, 0, 0)
    action_row.setSpacing(8)

    layout.addWidget(combined_header)
    layout.addWidget(title_label)
    layout.addWidget(hint_widget)
    layout.addWidget(overview_label)
    layout.addWidget(stale_warning_label)
    layout.addWidget(status_label)
    layout.addWidget(next_steps_label)
    layout.addWidget(next_action_label)
    layout.addWidget(next_action_button)
    layout.addWidget(capabilities_label)
    layout.addWidget(npde_status_label)
    layout.addWidget(filter_row_widget)
    layout.addWidget(artifact_browser_row_widget, 1)

    action_row.addWidget(view_dropdown)
    action_row.addWidget(output_dropdown)
    action_row.addStretch(1)
    action_row.addWidget(generate_npde_button)
    layout.addWidget(action_row_widget)

    _apply_responsive_box_layout = install_responsive_box_layouts(
        root,
        breakpoint=DIAGNOSTICS_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        layouts=(filter_row, action_row),
    )

    _apply_responsive_splitter = install_responsive_splitters(
        root,
        breakpoint=DIAGNOSTICS_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        splitters=(artifact_browser_row_widget,),
    )

    def _apply_responsive_layout(width: int | None = None) -> None:
        _apply_responsive_box_layout(width)
        _apply_responsive_splitter(width)

    current_artifacts: list[ArtifactRecord] = []
    filtered_artifacts: list[ArtifactRecord] = []
    current_artifact: ArtifactRecord | None = None
    current_fit_context_available = False
    future = None
    next_action_target = [""]
    poll_timer = qt_core.QTimer(root)
    poll_timer.setInterval(100)

    def _refresh_next_action() -> None:
        action = recommend_diagnostics_next_action(project, current_artifacts)
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

    def _navigate_to_next_action() -> None:
        workflow_id = next_action_target[0]
        if not workflow_id:
            return
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(workflow_id)

    def _refresh_filter_options(artifacts: list[ArtifactRecord]) -> None:
        current_role = artifact_role_combo.currentText() or "All roles"
        current_plot_type = artifact_plot_type_combo.currentText() or "All plot types"
        role_options = artifact_role_options(artifacts)
        plot_type_options = artifact_plot_type_options(artifacts)
        artifact_role_combo.blockSignals(True)
        artifact_plot_type_combo.blockSignals(True)
        artifact_role_combo.clear()
        artifact_plot_type_combo.clear()
        artifact_role_combo.addItems(role_options)
        artifact_plot_type_combo.addItems(plot_type_options)
        artifact_role_combo.setCurrentText(
            current_role if current_role in role_options else "All roles"
        )
        artifact_plot_type_combo.setCurrentText(
            current_plot_type if current_plot_type in plot_type_options else "All plot types"
        )
        artifact_role_combo.blockSignals(False)
        artifact_plot_type_combo.blockSignals(False)

    def _refresh_quick_actions() -> None:
        open_gof_action.setEnabled(
            bool(latest_diagnostics_artifact(current_artifacts, plot_type="gof_panel"))
        )
        open_residual_action.setEnabled(
            bool(latest_diagnostics_artifact(current_artifacts, plot_type="residual_trends"))
        )
        open_diagnostics_table_action.setEnabled(
            bool(
                (
                    artifact := latest_diagnostics_role_artifact(
                        current_artifacts, "diagnostics_table"
                    )
                )
                and artifact.path
            )
        )
        open_npde_table_action.setEnabled(
            bool(
                (artifact := latest_diagnostics_role_artifact(current_artifacts, "npde_table"))
                and artifact.path
            )
        )
        generate_npde_button.setEnabled(
            future is None
            and latest_fit_run(project) is not None
            and latest_fit_run(project).status == RunStatus.SUCCEEDED
            and current_fit_context_available
            and not bool(latest_diagnostics_role_artifact(current_artifacts, "npde_table"))
        )

    def _notify_project_state_changed() -> None:
        callback = getattr(root, "_project_state_changed", None)
        if callable(callback):
            callback()

    def _render_artifact_preview(artifact: ArtifactRecord | None) -> None:
        nonlocal current_artifact
        current_artifact = artifact
        preview_panel_obj.render(
            artifact,
            empty_label="Select a diagnostics output to preview.",
            on_has_path=lambda has_path: (
                export_artifact_action.setEnabled(has_path),
                open_artifact_action.setEnabled(has_path),
                open_folder_action.setEnabled(has_path),
            ),
        )

    def _open_selected_artifact() -> None:
        open_output_file(current_artifact)

    def _export_selected_artifact() -> None:
        export_output_file(current_artifact, title="Save diagnostics output copy", parent=root)

    def _open_list_item(item) -> None:
        index = artifacts_list.row(item)
        if index < 0 or index >= len(filtered_artifacts):
            return
        open_output_file(filtered_artifacts[index])

    def _open_selected_artifact_folder() -> None:
        if current_artifact is None or not current_artifact.path:
            return
        qt_gui.QDesktopServices.openUrl(
            qt_core.QUrl.fromLocalFile(str(Path(current_artifact.path).resolve().parent))
        )

    def _open_plot_type(plot_type: str) -> None:
        open_output_file(latest_diagnostics_artifact(current_artifacts, plot_type=plot_type))

    def _open_role_artifact(role: str) -> None:
        open_output_file(latest_diagnostics_role_artifact(current_artifacts, role))

    def _render_artifacts(artifacts: list[ArtifactRecord]) -> None:
        nonlocal filtered_artifacts
        selected_artifact_id = (
            current_artifact.artifact_id if current_artifact is not None else None
        )
        _refresh_filter_options(artifacts)
        role_filter = artifact_role_combo.currentText() or "All roles"
        plot_type_filter = artifact_plot_type_combo.currentText() or "All plot types"
        filtered_artifacts = filter_diagnostics_artifacts(artifacts, role_filter, plot_type_filter)
        artifact_summary_label.setText(format_diagnostics_artifact_summary(filtered_artifacts))
        _refresh_quick_actions()
        artifacts_list.clear()
        if not filtered_artifacts:
            artifacts_list.addItem("No diagnostics outputs are available for this filter.")
            _render_artifact_preview(None)
            return
        for artifact in filtered_artifacts:
            item = qt_widgets.QListWidgetItem(format_diagnostics_artifact_label(artifact))
            item.setData(qt_core.Qt.ItemDataRole.UserRole, artifact.artifact_id)
            artifacts_list.addItem(item)
        selected_index = next(
            (
                index
                for index, artifact in enumerate(filtered_artifacts)
                if artifact.artifact_id == selected_artifact_id
            ),
            0,
        )
        artifacts_list.setCurrentRow(selected_index)

    def _handle_selection_changed(index: int) -> None:
        if index < 0 or index >= len(filtered_artifacts):
            _render_artifact_preview(None)
            return
        _render_artifact_preview(filtered_artifacts[index])

    def _refresh() -> None:
        nonlocal current_artifacts, current_fit_context_available
        refresh_combined_header()
        run = latest_fit_run(project)
        current_artifacts = diagnostics_artifacts(
            project, run=run, artifact_service=artifact_service
        )
        current_fit_context_available = fit_service.latest_fit_context(project) is not None
        overview_label.setText(format_diagnostics_overview(project))
        stale_warning_label.setText(format_diagnostics_stale_warning(project))
        stale_warning_label.setVisible(bool(stale_warning_label.text()))
        status_label.setText(format_diagnostics_status(project))
        next_steps_label.setText(format_diagnostics_next_steps(project))
        npde_status_label.setText(
            format_npde_generation_status(
                project,
                fit_context_available=current_fit_context_available,
                npde_available=bool(
                    latest_diagnostics_role_artifact(current_artifacts, "npde_table")
                ),
                generation_running=future is not None,
            )
        )
        _refresh_next_action()
        if not current_artifacts:
            artifact_summary_label.setText("0 outputs")
            artifacts_list.clear()
            artifacts_list.addItem("No diagnostic-related outputs are available yet.")
            _refresh_quick_actions()
            _render_artifact_preview(None)
            return
        _render_artifacts(current_artifacts)

    def _poll_future() -> None:
        nonlocal future
        if future is None or not future.done():
            return
        outcome = future.result()
        run = npde_service.latest_run(project)
        if run is not None:
            artifacts = npde_service.apply_job_outcome(run, outcome)
            for artifact in artifacts:
                artifact_service.register(project, artifact)
                run.add_log(f"[artifact] {artifact.kind}: {artifact.label}")
            _notify_project_state_changed()
        future = None
        poll_timer.stop()
        _refresh()

    def _start_npde_generation() -> None:
        nonlocal future
        _refresh()
        fit_run = latest_fit_run(project)
        if (
            fit_run is None
            or fit_run.status != RunStatus.SUCCEEDED
            or not current_fit_context_available
        ):
            return
        run = RunRecord(workflow="npde")
        try:
            job = npde_service.create_job(project, fit_service=fit_service, run_id=run.run_id)
        except ValueError:
            _refresh()
            return
        run.mark_running()
        run.add_log("NPDE generation submitted.")
        project_service.add_run(project, run)
        _notify_project_state_changed()
        future = job_runner.submit(job)
        generate_npde_button.setEnabled(False)
        npde_status_label.setText(
            format_npde_generation_status(
                project,
                fit_context_available=current_fit_context_available,
                npde_available=False,
                generation_running=True,
            )
        )
        poll_timer.start()

    artifact_role_combo.currentTextChanged.connect(
        lambda _text: _render_artifacts(current_artifacts)
    )
    artifact_plot_type_combo.currentTextChanged.connect(
        lambda _text: _render_artifacts(current_artifacts)
    )
    artifacts_list.currentRowChanged.connect(_handle_selection_changed)
    artifacts_list.itemActivated.connect(_open_list_item)
    artifacts_list.itemDoubleClicked.connect(_open_list_item)
    open_gof_action.triggered.connect(lambda: _open_plot_type("gof_panel"))
    open_residual_action.triggered.connect(lambda: _open_plot_type("residual_trends"))
    generate_npde_button.clicked.connect(_start_npde_generation)
    open_diagnostics_table_action.triggered.connect(
        lambda: _open_role_artifact("diagnostics_table")
    )
    open_npde_table_action.triggered.connect(lambda: _open_role_artifact("npde_table"))
    export_artifact_action.triggered.connect(_export_selected_artifact)
    open_artifact_action.triggered.connect(_open_selected_artifact)
    open_folder_action.triggered.connect(_open_selected_artifact_folder)
    next_action_button.clicked.connect(_navigate_to_next_action)
    poll_timer.timeout.connect(_poll_future)
    if owns_job_runner:
        root.destroyed.connect(lambda *_args: job_runner.shutdown(wait=False))
    root._refresh_workflow = _refresh  # type: ignore[attr-defined]
    root._apply_responsive_layout = _apply_responsive_layout  # type: ignore[attr-defined]
    root._refresh_context_header = refresh_combined_header  # type: ignore[attr-defined]

    _refresh()
    _apply_responsive_layout()
    return root
