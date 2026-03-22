"""Scenario overview workflow page for the desktop GUI shell."""

from __future__ import annotations

from pathlib import Path

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    default_workspace_root_path,
    load_gui_preferences,
    save_gui_preferences,
    with_dismissed_hint,
)
from openpkpd_gui.domain.run_record import RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.workflow_state_service import (
    WorkflowState,
    WorkflowStateId,
    latest_run_for_workflow,
    workflow_state_for,
)
from openpkpd_gui.widgets.collapsible_section import build_collapsible_section
from openpkpd_gui.widgets.combined_header import build_combined_header
from openpkpd_gui.widgets.dismissible_hint import build_dismissible_hint
from openpkpd_gui.widgets.overview_readiness_card import (
    OverviewReadinessCardSpec,
    build_overview_readiness_card,
)
from openpkpd_gui.widgets.responsive_layout import install_responsive_box_layouts
from openpkpd_gui.widgets.semantic_state import set_semantic_state
from openpkpd_gui.widgets.workflow_shortcut_section import (
    WorkflowShortcutGroupSpec,
    WorkflowShortcutSpec,
    build_grouped_workflow_shortcut_section,
    refresh_workflow_shortcut_buttons,
)
from openpkpd_gui.workflows.results_workflow import (
    format_artifact_summary,
    format_fit_review_summary,
    latest_artifact,
    latest_artifact_for_plot_group,
)

RESPONSIVE_LAYOUT_BREAKPOINT = 900


def review_outputs_state(workspace: Workspace) -> WorkflowState:
    """Return the combined review-output state for the active scenario."""
    scenario = workspace.active_scenario
    latest_fit_run = latest_run_for_workflow(scenario, "fit")
    has_review_outputs = (
        any(
            latest_artifact(scenario.artifacts, role=role) is not None
            for role in ("report", "plot", "diagnostics_table", "npde_table")
        )
        or latest_artifact_for_plot_group(scenario.artifacts, "gof") is not None
    )
    if latest_fit_run is not None and latest_fit_run.status == RunStatus.RUNNING:
        return WorkflowState(
            "review",
            WorkflowStateId.RUNNING,
            "Running",
            "Latest review outputs will update when the current fit finishes.",
        )
    if latest_fit_run is not None and latest_fit_run.status == RunStatus.FAILED:
        return WorkflowState(
            "review",
            WorkflowStateId.NEEDS_ATTENTION,
            "Needs attention",
            "Review outputs may be stale because the latest fit needs attention.",
        )
    if has_review_outputs:
        return WorkflowState(
            "review",
            WorkflowStateId.RESULTS_AVAILABLE,
            "Results available",
            "Reports, plots, or diagnostics are ready to inspect.",
        )
    if latest_fit_run is not None and latest_fit_run.status == RunStatus.SUCCEEDED:
        return WorkflowState(
            "review",
            WorkflowStateId.READY,
            "Ready",
            "A successful fit exists; review outputs can now be opened as they are generated.",
        )
    return WorkflowState(
        "review",
        WorkflowStateId.NOT_STARTED,
        "Not started",
        "Review outputs will appear after this scenario has generated saved outputs.",
    )


def recommended_next_action(workspace: Workspace) -> tuple[str, str, str]:
    """Return the primary CTA for the active scenario."""
    scenario = workspace.active_scenario
    fit_state = workflow_state_for(workspace, "fit")
    review_state = review_outputs_state(workspace)
    advanced_state = workflow_state_for(workspace, "advanced")
    if scenario.active_dataset is None:
        return ("Open Data", "data", "Import or load a dataset to start work in this scenario.")
    if scenario.active_model_spec is None:
        return ("Open Model", "model", "A dataset is ready; open Model to configure one next.")
    if fit_state.state in {WorkflowStateId.NOT_STARTED, WorkflowStateId.READY}:
        return ("Open Fit", "fit", "Dataset and model are ready for estimation.")
    if fit_state.state == WorkflowStateId.RUNNING:
        return ("Open Fit", "fit", "A fit run is active; monitor progress and logs in Fit.")
    if fit_state.state == WorkflowStateId.NEEDS_ATTENTION:
        return (
            "Open Fit",
            "fit",
            "Resolve the latest fit issue before relying on downstream outputs.",
        )
    if review_state.state in {WorkflowStateId.READY, WorkflowStateId.RESULTS_AVAILABLE}:
        return ("Open Results", "results", "Review the latest reports, plots, and saved artifacts.")
    if advanced_state.state in {WorkflowStateId.READY, WorkflowStateId.RESULTS_AVAILABLE}:
        return (
            "Open Advanced",
            "advanced",
            "Validation and design workflows are available for this scenario.",
        )
    return ("Open Dashboard", "dashboard", "Scenario guidance is available here.")


def format_recent_activity(workspace: Workspace) -> str:
    """Summarize the most recent scenario activity."""
    scenario = workspace.active_scenario
    latest_run = scenario.runs[-1] if scenario.runs else None
    latest_output = scenario.artifacts[-1] if scenario.artifacts else None
    if latest_run is None and latest_output is None:
        return "No runs or saved outputs have been recorded for this scenario yet."
    segments: list[str] = []
    if latest_run is not None:
        detail = latest_run.summary_text or latest_run.error_text or latest_run.status.value.title()
        segments.append(
            f"Latest run: {latest_run.workflow} • {latest_run.status.value.title()} • {detail}"
        )
    if latest_output is not None:
        location = Path(latest_output.path).name if latest_output.path else "in-memory metadata"
        segments.append(f"Latest artifact: {latest_output.label} • {location}")
    return " • ".join(segments)


def format_latest_output_summary(workspace: Workspace) -> str:
    """Summarize the latest scenario outputs available from Overview."""
    artifacts = workspace.active_scenario.artifacts
    available = []
    if (artifact := latest_artifact(artifacts, role="report")) is not None:
        available.append(f"report: {artifact.label}")
    if (artifact := latest_artifact(artifacts, role="plot")) is not None:
        available.append(f"plot: {artifact.label}")
    if (artifact := latest_artifact_for_plot_group(artifacts, "gof")) is not None:
        available.append(f"GOF: {artifact.label}")
    if not available:
        fit_state = workflow_state_for(workspace, "fit")
        if fit_state.state == WorkflowStateId.RUNNING:
            return "Latest report and plot shortcuts will appear when the current fit finishes."
        if fit_state.state == WorkflowStateId.RESULTS_AVAILABLE:
            return "Recent fit results exist; latest-output shortcuts will appear as artifacts are saved."
        return "No saved report or plot outputs are available yet."
    return f"Latest outputs ready: {', '.join(available)}."


def format_review_workflows_summary(
    results_state: WorkflowState, plots_state: WorkflowState
) -> str:
    """Summarize the state of results and plots workflows."""
    if (
        results_state.state == WorkflowStateId.NOT_STARTED
        and plots_state.state == WorkflowStateId.NOT_STARTED
    ):
        return "Results and plots workflows unlock after a successful fit begins producing review outputs."
    return (
        f"Results: {results_state.label} — {results_state.summary} "
        f"Plots: {plots_state.label} — {plots_state.summary}"
    )


def format_follow_up_summary(
    diagnostics_state: WorkflowState, advanced_state: WorkflowState
) -> str:
    """Summarize the state of follow-up review workflows."""
    if (
        diagnostics_state.state == WorkflowStateId.NOT_STARTED
        and advanced_state.state == WorkflowStateId.NOT_STARTED
    ):
        return "Diagnostics and advanced validation workflows unlock after a successful fit."
    return (
        f"Diagnostics: {diagnostics_state.label} — {diagnostics_state.summary} "
        f"Advanced: {advanced_state.label} — {advanced_state.summary}"
    )


def format_additional_workflows_summary(
    nca_state: WorkflowState, covariate_state: WorkflowState
) -> str:
    """Summarize the state of additional scenario workflows."""
    if (
        nca_state.state == WorkflowStateId.NOT_STARTED
        and covariate_state.state == WorkflowStateId.NOT_STARTED
    ):
        return (
            "NCA and covariate workflows unlock after the scenario has saved inputs to work from."
        )
    return (
        f"NCA: {nca_state.label} — {nca_state.summary} "
        f"Covariate: {covariate_state.label} — {covariate_state.summary}"
    )


def build_overview_workflow(project: Workspace):
    """Build the scenario Overview workflow page."""
    qt_core, qt_gui, qt_widgets = load_qt_modules()

    root = qt_widgets.QWidget()
    root.setObjectName("overview-workflow")
    layout = qt_widgets.QVBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)

    scroll_area = qt_widgets.QScrollArea(root)
    scroll_area.setObjectName("overview-scroll-area")
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(qt_widgets.QFrame.Shape.NoFrame)

    content = qt_widgets.QWidget(scroll_area)
    content.setObjectName("overview-content")
    content_layout = qt_widgets.QVBoxLayout(content)
    content_layout.setContentsMargins(12, 12, 12, 12)
    content_layout.setSpacing(12)
    scroll_area.setWidget(content)
    layout.addWidget(scroll_area)

    combined_header, refresh_combined_header = build_combined_header(
        content,
        project,
        workflow_id="dashboard",
        workflow_label="Dashboard",
        status_workflow_ids=("data", "model", "fit", "results", "advanced"),
    )

    hero_panel = qt_widgets.QFrame(content)
    hero_panel.setObjectName("overview-hero-panel")
    hero_panel.setFrameShape(qt_widgets.QFrame.Shape.StyledPanel)
    hero_layout = qt_widgets.QVBoxLayout(hero_panel)
    hero_layout.setContentsMargins(12, 12, 12, 12)
    hero_layout.setSpacing(6)

    eyebrow_label = qt_widgets.QLabel("◆ Scenario dashboard", hero_panel)
    eyebrow_label.setObjectName("overview-eyebrow-label")
    eyebrow_label.setProperty("semanticRole", "eyebrow")
    eyebrow_font = eyebrow_label.font()
    eyebrow_font.setBold(True)
    eyebrow_label.setFont(eyebrow_font)

    title_label = qt_widgets.QLabel("Scenario Overview", hero_panel)
    title_label.setObjectName("overview-title-label")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 5)
    title_font.setBold(True)
    title_label.setFont(title_font)

    _hint_prefs = load_gui_preferences()

    def _save_dismissed_hint_overview():
        save_gui_preferences(with_dismissed_hint(load_gui_preferences(), "hint_overview"))

    hint_widget, _ = build_dismissible_hint(
        "Use this page to confirm what is ready, what happened most recently, and where to go next.",
        dismissed="hint_overview" in _hint_prefs.dismissed_hints,
        object_name="overview-hint-label",
        on_dismiss=_save_dismissed_hint_overview,
    )
    output_path = (
        Path(project.root_path) / ".openpkpd_gui_artifacts"
        if project.root_path
        else default_workspace_root_path() / ".openpkpd_gui_artifacts"
    )
    output_row_widget = qt_widgets.QWidget(content)
    output_row_widget.setObjectName("overview-output-folder-row")
    output_row = qt_widgets.QHBoxLayout(output_row_widget)
    output_row.setContentsMargins(0, 0, 0, 0)
    output_row.setSpacing(6)
    output_label = qt_widgets.QLabel(
        f"Output folder: <a href='file:///{output_path}'>{output_path}</a>"
    )
    output_label.setObjectName("overview-output-folder-label")
    output_label.setOpenExternalLinks(True)
    output_label.setWordWrap(True)
    open_output_button = qt_widgets.QPushButton("Open folder")
    open_output_button.setObjectName("overview-open-output-folder-button")
    open_output_button.clicked.connect(
        lambda: qt_gui.QDesktopServices.openUrl(qt_core.QUrl.fromLocalFile(str(output_path)))
    )
    output_row.addWidget(output_label, 1)
    output_row.addWidget(open_output_button)

    hero_layout.addWidget(eyebrow_label)
    hero_layout.addWidget(title_label)
    hero_layout.addWidget(hint_widget)
    hero_layout.addWidget(output_row_widget)

    readiness_group = qt_widgets.QGroupBox("✓ Scenario readiness", content)
    readiness_group.setObjectName("overview-readiness-group")
    readiness_layout = qt_widgets.QGridLayout(readiness_group)
    readiness_layout.setContentsMargins(12, 12, 12, 12)
    readiness_layout.setHorizontalSpacing(10)
    readiness_layout.setVerticalSpacing(10)
    readiness_card_widgets: dict[str, object] = {}
    readiness_cards: dict[str, tuple[object, object]] = {}
    readiness_order = (
        OverviewReadinessCardSpec("data", "Dataset", "◫"),
        OverviewReadinessCardSpec("model", "Model", "ƒ"),
        OverviewReadinessCardSpec("fit", "Fit", "∑"),
        OverviewReadinessCardSpec("nca", "NCA", "∫"),
        OverviewReadinessCardSpec("covariate", "Covariate", "⊕"),
        OverviewReadinessCardSpec("review", "Review outputs", "◌"),
        OverviewReadinessCardSpec("advanced", "Validation & design", "⋯"),
    )
    for index, spec in enumerate(readiness_order):
        card, state_label, summary_label = build_overview_readiness_card(content, spec=spec)
        readiness_layout.addWidget(card, index // 2, index % 2)
        readiness_cards[spec.key] = (state_label, summary_label)
        readiness_card_widgets[spec.key] = card

    next_steps_group = qt_widgets.QGroupBox("→ Recommended next step", content)
    next_steps_group.setObjectName("overview-next-steps-group")
    next_steps_layout = qt_widgets.QVBoxLayout(next_steps_group)
    next_steps_layout.setContentsMargins(12, 12, 12, 12)
    next_steps_layout.setSpacing(8)
    next_steps_label = qt_widgets.QLabel()
    next_steps_label.setObjectName("overview-next-action-label")
    next_steps_label.setWordWrap(True)
    next_action_button = qt_widgets.QPushButton()
    next_action_button.setObjectName("overview-next-action-button")
    next_action_button.setProperty("primaryAction", True)
    next_action_button.setMinimumHeight(36)
    next_steps_layout.addWidget(next_steps_label)
    next_steps_layout.addWidget(next_action_button)

    latest_outputs_group = qt_widgets.QGroupBox("⇣ Latest outputs", content)
    latest_outputs_group.setObjectName("overview-latest-outputs-group")
    latest_outputs_layout = qt_widgets.QVBoxLayout(latest_outputs_group)
    latest_outputs_layout.setContentsMargins(12, 12, 12, 12)
    latest_outputs_layout.setSpacing(8)
    latest_output_summary_label = qt_widgets.QLabel()
    latest_output_summary_label.setObjectName("overview-latest-output-summary-label")
    latest_output_summary_label.setWordWrap(True)
    latest_details_section, _, latest_details_layout, latest_details_toggle = (
        build_collapsible_section(
            latest_outputs_group,
            title="Details & quick actions",
            object_name="overview-latest-output-details-section",
            expanded=False,
        )
    )
    fit_review_summary_label = qt_widgets.QLabel()
    fit_review_summary_label.setObjectName("overview-fit-review-summary-label")
    fit_review_summary_label.setWordWrap(True)
    shortcut_row = qt_widgets.QHBoxLayout()
    shortcut_row.setContentsMargins(0, 0, 0, 0)
    shortcut_row.setSpacing(8)
    open_latest_report_button = qt_widgets.QPushButton("Open latest report")
    open_latest_report_button.setObjectName("overview-open-latest-report-button")
    open_latest_report_button.setMinimumHeight(32)
    open_latest_plot_button = qt_widgets.QPushButton("Open latest plot")
    open_latest_plot_button.setObjectName("overview-open-latest-plot-button")
    open_latest_plot_button.setMinimumHeight(32)
    shortcut_row.addWidget(open_latest_report_button)
    shortcut_row.addWidget(open_latest_plot_button)
    shortcut_row.addStretch(1)
    latest_outputs_layout.addWidget(latest_output_summary_label)
    latest_details_layout.addWidget(fit_review_summary_label)
    latest_details_layout.addLayout(shortcut_row)
    latest_outputs_layout.addWidget(latest_details_section)

    (
        available_workflows_group,
        workflow_summary_labels,
        workflow_rows,
        workflow_buttons,
        workflow_button_labels,
    ) = build_grouped_workflow_shortcut_section(
        content,
        title="☰ Available workflows",
        object_name="overview-available-workflows-group",
        group_specs=(
            WorkflowShortcutGroupSpec(
                "review",
                "Review outputs",
                "overview-review-workflows-summary-label",
                (
                    WorkflowShortcutSpec("results", "Results", "overview-open-results-button"),
                    WorkflowShortcutSpec("plots", "Plots", "overview-open-plots-button"),
                ),
            ),
            WorkflowShortcutGroupSpec(
                "follow_up",
                "Follow-up analyses",
                "overview-follow-up-summary-label",
                (
                    WorkflowShortcutSpec(
                        "diagnostics",
                        "Diagnostics",
                        "overview-open-diagnostics-button",
                    ),
                    WorkflowShortcutSpec("advanced", "Advanced", "overview-open-advanced-button"),
                ),
            ),
            WorkflowShortcutGroupSpec(
                "additional",
                "Additional workflows",
                "overview-additional-workflows-summary-label",
                (
                    WorkflowShortcutSpec("nca", "NCA", "overview-open-nca-button"),
                    WorkflowShortcutSpec(
                        "covariate", "Covariate", "overview-open-covariate-button"
                    ),
                ),
            ),
        ),
    )
    review_summary_label = workflow_summary_labels["review"]
    follow_up_summary_label = workflow_summary_labels["follow_up"]
    additional_summary_label = workflow_summary_labels["additional"]
    review_row = workflow_rows["review"]
    follow_up_row = workflow_rows["follow_up"]
    additional_row = workflow_rows["additional"]
    review_buttons = workflow_buttons["review"]
    follow_up_buttons = workflow_buttons["follow_up"]
    additional_buttons = workflow_buttons["additional"]
    review_button_labels = workflow_button_labels["review"]
    follow_up_button_labels = workflow_button_labels["follow_up"]
    additional_button_labels = workflow_button_labels["additional"]

    activity_group, _, activity_layout, _activity_toggle = build_collapsible_section(
        content,
        title="◷ Recent activity",
        object_name="overview-activity-group",
        expanded=False,
        framed=True,
    )
    activity_label = qt_widgets.QLabel()
    activity_label.setObjectName("overview-activity-label")
    activity_label.setWordWrap(True)
    artifact_summary_label = qt_widgets.QLabel()
    artifact_summary_label.setObjectName("overview-artifact-summary-label")
    artifact_summary_label.setWordWrap(True)
    activity_layout.addWidget(activity_label)
    activity_layout.addWidget(artifact_summary_label)

    primary_row = qt_widgets.QWidget(content)
    primary_row.setObjectName("overview-primary-row")
    primary_layout = qt_widgets.QHBoxLayout(primary_row)
    primary_layout.setContentsMargins(0, 0, 0, 0)
    primary_layout.setSpacing(12)
    primary_layout.addWidget(next_steps_group, 1)
    primary_layout.addWidget(latest_outputs_group, 1)

    secondary_row = qt_widgets.QWidget(content)
    secondary_row.setObjectName("overview-secondary-row")
    secondary_layout = qt_widgets.QHBoxLayout(secondary_row)
    secondary_layout.setContentsMargins(0, 0, 0, 0)
    secondary_layout.setSpacing(12)

    readiness_column = qt_widgets.QWidget(secondary_row)
    readiness_column.setObjectName("overview-readiness-column")
    readiness_column_layout = qt_widgets.QVBoxLayout(readiness_column)
    readiness_column_layout.setContentsMargins(0, 0, 0, 0)
    readiness_column_layout.setSpacing(0)
    readiness_column_layout.addWidget(readiness_group)

    workflows_column = qt_widgets.QWidget(secondary_row)
    workflows_column.setObjectName("overview-workflows-column")
    workflows_column_layout = qt_widgets.QVBoxLayout(workflows_column)
    workflows_column_layout.setContentsMargins(0, 0, 0, 0)
    workflows_column_layout.setSpacing(12)
    workflows_column_layout.addWidget(available_workflows_group)
    workflows_column_layout.addStretch(1)

    secondary_layout.addWidget(readiness_column, 3)
    secondary_layout.addWidget(workflows_column, 2)

    content_layout.addWidget(combined_header)
    content_layout.addWidget(hero_panel)
    content_layout.addWidget(primary_row)
    content_layout.addWidget(secondary_row)
    content_layout.addWidget(activity_group)
    content_layout.addStretch(1)

    def _sync_readiness_grid(compact: bool) -> None:
        if compact:
            for index, spec in enumerate(readiness_order):
                readiness_layout.addWidget(readiness_card_widgets[spec.key], index, 0)
        else:
            for index, spec in enumerate(readiness_order):
                readiness_layout.addWidget(readiness_card_widgets[spec.key], index // 2, index % 2)

    _apply_responsive_layout = install_responsive_box_layouts(
        root,
        breakpoint=RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0)
            or scroll_area.viewport().width()
            or root.width()
        ),
        layouts=(
            primary_layout,
            secondary_layout,
            shortcut_row,
            review_row,
            follow_up_row,
            additional_row,
        ),
        on_mode_changed=_sync_readiness_grid,
    )

    next_action_target = ["overview"]

    def _call_root_callback(callback_name: str) -> None:
        callback = getattr(root, callback_name, None)
        if callable(callback):
            callback()

    def _navigate_to_next_action() -> None:
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(next_action_target[0])

    def _navigate_to_workflow(workflow_id: str) -> None:
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(workflow_id)

    def _refresh() -> None:
        refresh_combined_header()
        data_state = workflow_state_for(project, "data")
        model_state = workflow_state_for(project, "model")
        fit_state = workflow_state_for(project, "fit")
        review_state = review_outputs_state(project)
        nca_state = workflow_state_for(project, "nca")
        covariate_state = workflow_state_for(project, "covariate")
        results_state = workflow_state_for(project, "results")
        plots_state = workflow_state_for(project, "plots")
        diagnostics_state = workflow_state_for(project, "diagnostics")
        advanced_state = workflow_state_for(project, "advanced")
        for key, state in {
            "data": data_state,
            "model": model_state,
            "fit": fit_state,
            "nca": nca_state,
            "covariate": covariate_state,
            "review": review_state,
            "advanced": advanced_state,
        }.items():
            state_label, summary_label = readiness_cards[key]
            state_label.setText(state.label)
            summary_label.setText(state.summary)
            set_semantic_state(state_label, state)
            set_semantic_state(readiness_card_widgets[key], state)

        button_text, workflow_id, summary = recommended_next_action(project)
        next_action_target[0] = workflow_id
        next_steps_label.setText(summary)
        next_action_button.setText(button_text)
        next_action_button.setToolTip(summary)

        artifacts = project.active_scenario.artifacts
        latest_report = latest_artifact(artifacts, role="report")
        latest_plot = latest_artifact(artifacts, role="plot")
        latest_output_summary_label.setText(format_latest_output_summary(project))
        fit_review_summary_label.setText(format_fit_review_summary(artifacts))
        latest_details_toggle.setEnabled(bool(artifacts))
        open_latest_report_button.setEnabled(bool(latest_report is not None and latest_report.path))
        open_latest_plot_button.setEnabled(bool(latest_plot is not None and latest_plot.path))
        review_summary_label.setText(format_review_workflows_summary(results_state, plots_state))
        refresh_workflow_shortcut_buttons(
            review_buttons,
            review_button_labels,
            {"results": results_state, "plots": plots_state},
        )
        follow_up_summary_label.setText(format_follow_up_summary(diagnostics_state, advanced_state))
        refresh_workflow_shortcut_buttons(
            follow_up_buttons,
            follow_up_button_labels,
            {"diagnostics": diagnostics_state, "advanced": advanced_state},
        )
        additional_summary_label.setText(
            format_additional_workflows_summary(nca_state, covariate_state)
        )
        refresh_workflow_shortcut_buttons(
            additional_buttons,
            additional_button_labels,
            {"nca": nca_state, "covariate": covariate_state},
        )
        activity_label.setText(format_recent_activity(project))
        artifact_summary_label.setText(format_artifact_summary(artifacts))

    next_action_button.clicked.connect(_navigate_to_next_action)
    open_latest_report_button.clicked.connect(
        lambda: _call_root_callback("_project_open_latest_report")
    )
    open_latest_plot_button.clicked.connect(
        lambda: _call_root_callback("_project_open_latest_plot")
    )
    review_buttons["results"].clicked.connect(lambda: _call_root_callback("_navigate_to_results"))
    review_buttons["plots"].clicked.connect(lambda: _navigate_to_workflow("results"))
    follow_up_buttons["diagnostics"].clicked.connect(lambda: _navigate_to_workflow("diagnostics"))
    follow_up_buttons["advanced"].clicked.connect(lambda: _navigate_to_workflow("advanced"))
    additional_buttons["nca"].clicked.connect(lambda: _navigate_to_workflow("nca"))
    additional_buttons["covariate"].clicked.connect(lambda: _navigate_to_workflow("covariate"))
    root._refresh_workflow = _refresh  # type: ignore[attr-defined]
    root._apply_responsive_layout = _apply_responsive_layout  # type: ignore[attr-defined]
    root._refresh_context_header = refresh_combined_header  # type: ignore[attr-defined]

    _refresh()
    _apply_responsive_layout()
    return root
