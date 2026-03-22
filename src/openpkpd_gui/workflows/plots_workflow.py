"""Plots workflow widget for browsing selected-scenario plot outputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    default_workspace_root_path,
    load_gui_preferences,
    save_gui_preferences,
    with_dismissed_hint,
)
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.workflow_state_service import describe_fit_input_changes
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
from openpkpd_gui.workflows.results_workflow import (
    ANALYSIS_FILTER_ALL,
    artifact_analysis_type,
    artifact_analysis_type_options,
    artifact_plot_type,
    artifact_plot_type_options,
    artifact_role,
    latest_artifact,
    review_runs,
    run_analysis_type,
    run_workflow_label,
    scenario_runs_by_id,
)

PLOTS_RESPONSIVE_LAYOUT_BREAKPOINT = 1000


def plot_artifacts(project: Workspace) -> list[ArtifactRecord]:
    """Return selected-scenario plot artifacts, newest first."""
    plots = [
        artifact
        for artifact in project.active_scenario.artifacts
        if artifact_role(artifact) == "plot"
    ]
    return list(reversed(plots))


def format_plots_overview(project: Workspace, plots: list[ArtifactRecord]) -> str:
    """Return a compact overview of selected-scenario plotting state."""
    runs_by_id = scenario_runs_by_id(project)
    analysis_types = {artifact_analysis_type(artifact, runs_by_id) for artifact in plots}
    analysis_text = ", ".join(sorted(analysis_types)) if analysis_types else "none yet"
    latest_run = next(iter(review_runs(project)), None)
    if latest_run is None:
        latest_text = "latest status No runs yet"
    else:
        run_label = run_analysis_type(latest_run)
        workflow_label = run_workflow_label(latest_run)
        if workflow_label != run_label:
            run_label = f"{run_label} / {workflow_label}"
        latest_text = f"latest {run_label} {latest_run.status.value.title()}"
    return f"Scenario {project.active_scenario.name} • {latest_text} • {len(plots)} saved plots • analyses {analysis_text}"


def format_plots_stale_warning(project: Workspace) -> str:
    """Return a stale-plots warning when saved inputs outpace the latest successful fit."""
    subject = describe_fit_input_changes(project.active_scenario)
    if subject is None:
        return ""
    return f"{subject} since the latest successful fit. Plot previews may be stale until you rerun the fit."


def format_plots_status(
    project: Workspace, plots: list[ArtifactRecord], analysis_filter: str = ANALYSIS_FILTER_ALL
) -> str:
    """Summarize the current scenario's plotting readiness."""
    runs = review_runs(project, analysis_filter)
    latest_run = runs[0] if runs else None
    analysis_text = "analysis" if analysis_filter == ANALYSIS_FILTER_ALL else analysis_filter
    if latest_run is None:
        if analysis_filter == ANALYSIS_FILTER_ALL:
            return "Run an analysis to generate saved plots for this scenario. No analysis runs are available yet."
        return (
            f"Run a {analysis_filter.lower()} analysis to generate saved plots for this scenario. "
            f"No {analysis_filter.lower()} runs are available yet."
        )
    if latest_run.status.value == "running" and not plots:
        return f"A {analysis_text.lower()} run is currently running. Refresh after it completes to browse any new plots."
    if latest_run.status.value == "failed" and not plots:
        return f"The latest {analysis_text.lower()} run failed. Fix the issue before expecting plot outputs."
    if not plots:
        return f"No saved {analysis_text.lower()} plots were found for this scenario."
    return "Saved plots can be previewed here, filtered by analysis type and plot type, and opened from disk."


def recommend_plots_next_action(
    project: Workspace, plots: list[ArtifactRecord]
) -> tuple[str, str, str] | None:
    """Return the primary CTA for an empty or blocked Plots page."""
    if plots:
        return None
    if project.active_dataset is None:
        return (
            "Open Data",
            "data",
            "Load a dataset before expecting saved plots for this scenario.",
        )
    if project.active_model_spec is None:
        return (
            "Open Model",
            "model",
            "Configure a model in the Model workflow first.",
        )
    runs = review_runs(project)
    if not runs:
        return (
            "Open Fit",
            "fit",
            "Run a fit to generate saved plots that can be previewed here.",
        )
    latest_run = runs[0]
    if latest_run.status == RunStatus.RUNNING:
        return (
            "Open Results",
            "results",
            "A run is still in progress. Review current logs in Results and refresh Plots when it finishes.",
        )
    if latest_run.status == RunStatus.FAILED:
        return (
            "Open Results",
            "results",
            "The latest analysis failed. Inspect the run details in Results before expecting plot outputs.",
        )
    return (
        "Open Results",
        "results",
        "Saved plots are not available yet. Review the latest run outputs and output list in Results.",
    )


def format_plot_summary(plots: list[ArtifactRecord]) -> str:
    """Render plot counts grouped by plot type."""
    if not plots:
        return "0 plots"
    counts = Counter(artifact_plot_type(artifact) or "untyped" for artifact in plots)
    detail = " • ".join(f"{plot_type}: {count}" for plot_type, count in sorted(counts.items()))
    return f"{len(plots)} plots • {detail}"


def format_plot_label(artifact: ArtifactRecord) -> str:
    """Render one plot row for the Plots workflow."""
    plot_type = artifact_plot_type(artifact) or "untyped"
    path_text = artifact.path or "in-memory metadata"
    return f"{artifact.label} • {plot_type} • {path_text}"


def build_plots_workflow(project: Workspace):
    """Build a plot-focused workflow page for the selected scenario."""
    qt_core, qt_gui, qt_widgets = load_qt_modules()

    root = qt_widgets.QWidget()
    root.setObjectName("plots-workflow")
    layout = qt_widgets.QVBoxLayout(root)

    title_label = qt_widgets.QLabel("Plots workflow")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 4)
    title_font.setBold(True)
    title_label.setFont(title_font)

    _hint_prefs = load_gui_preferences()

    def _save_dismissed_hint_plots():
        save_gui_preferences(with_dismissed_hint(load_gui_preferences(), "hint_plots"))

    hint_widget, _ = build_dismissible_hint(
        "Browse saved plots for the selected scenario. This page breaks plots down by "
        "analysis type and plot type while keeping plotting review separate from logs and general results.",
        dismissed="hint_plots" in _hint_prefs.dismissed_hints,
        on_dismiss=_save_dismissed_hint_plots,
    )

    overview_label = qt_widgets.QLabel(format_plots_overview(project, plot_artifacts(project)))
    overview_label.setObjectName("plots-overview-label")
    overview_label.setWordWrap(True)

    stale_warning_label = qt_widgets.QLabel(format_plots_stale_warning(project))
    stale_warning_label.setObjectName("plots-stale-warning-label")
    stale_warning_label.setWordWrap(True)
    stale_warning_label.setVisible(bool(stale_warning_label.text()))

    status_label = qt_widgets.QLabel(format_plots_status(project, plot_artifacts(project)))
    status_label.setObjectName("plots-status-label")
    status_label.setWordWrap(True)

    next_action_label = qt_widgets.QLabel("")
    next_action_label.setObjectName("plots-next-action-label")
    next_action_label.setWordWrap(True)
    next_action_label.setVisible(False)

    next_action_button = qt_widgets.QPushButton("")
    next_action_button.setObjectName("plots-next-action-button")
    next_action_button.setProperty("primaryAction", True)
    next_action_button.setMinimumHeight(36)
    next_action_button.setVisible(False)

    filter_row_widget = qt_widgets.QWidget(root)
    filter_row_widget.setObjectName("plots-filter-row")
    filter_row = qt_widgets.QHBoxLayout(filter_row_widget)
    filter_row.setContentsMargins(0, 0, 0, 0)
    filter_row.setSpacing(8)
    filter_row.addWidget(qt_widgets.QLabel("Analysis"))
    analysis_type_combo = qt_widgets.QComboBox()
    analysis_type_combo.setObjectName("plots-analysis-filter")
    analysis_type_combo.setProperty("persistComboSelection", True)
    filter_row.addWidget(analysis_type_combo, 1)
    filter_row.addWidget(qt_widgets.QLabel("Plot type"))
    plot_type_combo = qt_widgets.QComboBox()
    plot_type_combo.setObjectName("plots-plot-type-filter")
    plot_type_combo.setProperty("persistComboSelection", True)
    filter_row.addWidget(plot_type_combo, 1)

    summary_label = qt_widgets.QLabel("0 plots")
    summary_label.setObjectName("plots-summary-label")
    summary_label.setWordWrap(True)

    plots_list = qt_widgets.QListWidget()
    plots_list.setObjectName("plots-list")
    plots_list.setProperty("persistListSelection", True)

    preview_panel_obj = build_output_preview_panel(
        root,
        object_prefix="plots",
        placeholder_text="Preview supported for HTML, text, and image plot outputs.",
    )
    # Expose individual widgets for legacy code that references them by name.
    preview_title = preview_panel_obj.title_label
    preview_metadata = preview_panel_obj.metadata_label
    preview_stack = preview_panel_obj.stack

    content_row_widget = qt_widgets.QSplitter(root)
    content_row_widget.setObjectName("plots-content-row")
    content_row_widget.setChildrenCollapsible(False)
    content_row_widget.setHandleWidth(8)

    list_panel = qt_widgets.QWidget(content_row_widget)
    list_panel.setObjectName("plots-list-panel")
    list_column = qt_widgets.QVBoxLayout(list_panel)
    list_column.setContentsMargins(12, 12, 12, 12)
    list_column.setSpacing(8)
    list_column.addWidget(summary_label)
    list_column.addWidget(plots_list, 1)

    preview_panel = qt_widgets.QWidget(content_row_widget)
    preview_panel.setObjectName("plots-preview-panel")
    preview_column = qt_widgets.QVBoxLayout(preview_panel)
    preview_column.setContentsMargins(12, 12, 12, 12)
    preview_column.setSpacing(8)
    preview_column.addWidget(preview_title)
    preview_column.addWidget(preview_metadata)
    preview_column.addWidget(preview_stack, 1)

    content_row_widget.addWidget(list_panel)
    content_row_widget.addWidget(preview_panel)
    content_row_widget.setStretchFactor(0, 2)
    content_row_widget.setStretchFactor(1, 3)
    content_row_widget.setSizes([400, 600])

    open_menu = qt_widgets.QMenu(root)
    open_latest_action = open_menu.addAction("Open latest plot")
    open_latest_action.setObjectName("plots-open-latest-action")
    open_latest_action.setEnabled(False)
    open_selected_action = open_menu.addAction("Open selected plot")
    open_selected_action.setObjectName("plots-open-selected-action")
    open_selected_action.setEnabled(False)
    export_copy_action = open_menu.addAction("Save copy…")
    export_copy_action.setObjectName("plots-save-copy-action")
    export_copy_action.setEnabled(False)
    open_folder_action = open_menu.addAction("Open folder")
    open_folder_action.setObjectName("plots-open-folder-action")
    open_folder_action.setEnabled(False)
    open_dropdown = qt_widgets.QToolButton()
    open_dropdown.setObjectName("plots-open-dropdown-button")
    open_dropdown.setText("Open ▾")
    open_dropdown.setMenu(open_menu)
    open_dropdown.setPopupMode(qt_widgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    next_action_target = [""]

    action_row_widget = qt_widgets.QWidget(root)
    action_row_widget.setObjectName("plots-action-row")
    action_row = qt_widgets.QHBoxLayout(action_row_widget)
    action_row.setContentsMargins(0, 0, 0, 0)
    action_row.setSpacing(8)
    action_row.addWidget(open_dropdown)
    action_row.addStretch(1)

    layout.addWidget(title_label)
    layout.addWidget(hint_widget)
    layout.addWidget(overview_label)
    layout.addWidget(stale_warning_label)
    layout.addWidget(status_label)
    layout.addWidget(next_action_label)
    layout.addWidget(next_action_button)
    layout.addWidget(filter_row_widget)
    layout.addWidget(content_row_widget, 1)
    layout.addWidget(action_row_widget)

    _apply_responsive_box_layout = install_responsive_box_layouts(
        root,
        breakpoint=PLOTS_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=root.width,
        layouts=(filter_row, action_row),
    )

    _apply_responsive_splitter = install_responsive_splitters(
        root,
        breakpoint=PLOTS_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=root.width,
        splitters=(content_row_widget,),
    )

    def _apply_responsive_layout(width: int | None = None) -> None:
        _apply_responsive_box_layout(width)
        _apply_responsive_splitter(width)

    current_plots: list[ArtifactRecord] = []
    filtered_plots: list[ArtifactRecord] = []
    current_plot: ArtifactRecord | None = None

    def _refresh_next_action() -> None:
        action = recommend_plots_next_action(project, current_plots)
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

    def _analysis_filter_text() -> str:
        return analysis_type_combo.currentText() or ANALYSIS_FILTER_ALL

    def _refresh_filter_options(plots: list[ArtifactRecord]) -> None:
        runs_by_id = scenario_runs_by_id(project)
        current_analysis = _analysis_filter_text()
        current_plot_type = plot_type_combo.currentText() or "All plot types"
        analysis_options = artifact_analysis_type_options(plots, runs_by_id)
        options = artifact_plot_type_options(plots)
        analysis_type_combo.blockSignals(True)
        plot_type_combo.blockSignals(True)
        analysis_type_combo.clear()
        plot_type_combo.clear()
        analysis_type_combo.addItems(analysis_options)
        plot_type_combo.addItems(options)
        analysis_type_combo.setCurrentText(
            current_analysis if current_analysis in analysis_options else ANALYSIS_FILTER_ALL
        )
        plot_type_combo.setCurrentText(
            current_plot_type if current_plot_type in options else "All plot types"
        )
        analysis_type_combo.blockSignals(False)
        plot_type_combo.blockSignals(False)

    def _render_preview(artifact: ArtifactRecord | None) -> None:
        nonlocal current_plot
        current_plot = artifact
        preview_panel_obj.render(
            artifact,
            scenario_runs_by_id(project),
            empty_label="Select a plot to preview.",
            on_has_path=lambda has_path: (
                open_selected_action.setEnabled(has_path),
                export_copy_action.setEnabled(has_path),
                open_folder_action.setEnabled(has_path),
            ),
        )

    def _open_artifact(artifact: ArtifactRecord | None) -> None:
        open_output_file(artifact)

    def _open_list_item(item) -> None:
        index = plots_list.row(item)
        if index < 0 or index >= len(filtered_plots):
            return
        open_output_file(filtered_plots[index])

    def _open_latest_plot() -> None:
        open_output_file(latest_artifact(filtered_plots, role="plot"))

    def _export_selected_plot() -> None:
        export_output_file(current_plot, title="Save plot copy", parent=root)

    def _open_plot_folder() -> None:
        artifact = current_plot or latest_artifact(filtered_plots, role="plot")
        if artifact is not None and artifact.path:
            target = Path(artifact.path).resolve().parent
        elif project.root_path:
            target = Path(project.root_path).resolve()
        else:
            target = default_workspace_root_path()
        qt_gui.QDesktopServices.openUrl(qt_core.QUrl.fromLocalFile(str(target)))

    def _render_plots() -> None:
        nonlocal filtered_plots
        runs_by_id = scenario_runs_by_id(project)
        selected_plot_id = current_plot.artifact_id if current_plot is not None else None
        _refresh_filter_options(current_plots)
        analysis_filter = _analysis_filter_text()
        plot_type_filter = plot_type_combo.currentText() or "All plot types"
        filtered_plots = [
            artifact
            for artifact in current_plots
            if analysis_filter == ANALYSIS_FILTER_ALL
            or artifact_analysis_type(artifact, runs_by_id) == analysis_filter
            if plot_type_filter == "All plot types"
            or artifact_plot_type(artifact) == plot_type_filter
        ]
        status_label.setText(format_plots_status(project, filtered_plots, analysis_filter))
        summary_label.setText(format_plot_summary(filtered_plots))
        open_latest_action.setEnabled(bool(latest_artifact(filtered_plots, role="plot")))
        plots_list.clear()
        if not filtered_plots:
            plots_list.addItem("No plots are available for this filter.")
            _render_preview(None)
            return
        for artifact in filtered_plots:
            item = qt_widgets.QListWidgetItem(format_plot_label(artifact))
            item.setData(qt_core.Qt.ItemDataRole.UserRole, artifact.artifact_id)
            plots_list.addItem(item)
        selected_index = next(
            (
                index
                for index, artifact in enumerate(filtered_plots)
                if artifact.artifact_id == selected_plot_id
            ),
            0,
        )
        plots_list.setCurrentRow(selected_index)

    def _handle_selection_changed(index: int) -> None:
        if index < 0 or index >= len(filtered_plots):
            _render_preview(None)
            return
        _render_preview(filtered_plots[index])

    def _refresh() -> None:
        nonlocal current_plots
        current_plots = plot_artifacts(project)
        overview_label.setText(format_plots_overview(project, current_plots))
        stale_warning_label.setText(format_plots_stale_warning(project))
        stale_warning_label.setVisible(bool(stale_warning_label.text()))
        _refresh_next_action()
        _render_plots()

    analysis_type_combo.currentTextChanged.connect(lambda _text: _render_plots())
    plot_type_combo.currentTextChanged.connect(lambda _text: _render_plots())
    plots_list.currentRowChanged.connect(_handle_selection_changed)
    plots_list.itemActivated.connect(_open_list_item)
    plots_list.itemDoubleClicked.connect(_open_list_item)
    open_latest_action.triggered.connect(_open_latest_plot)
    export_copy_action.triggered.connect(_export_selected_plot)
    open_selected_action.triggered.connect(lambda: _open_artifact(current_plot))
    open_folder_action.triggered.connect(_open_plot_folder)
    next_action_button.clicked.connect(_navigate_to_next_action)
    root._apply_responsive_layout = _apply_responsive_layout  # type: ignore[attr-defined]
    root._refresh_workflow = _refresh  # type: ignore[attr-defined]

    _refresh()
    _apply_responsive_layout()
    return root
