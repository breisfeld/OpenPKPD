"""Workspace landing workflow for project-level actions and handoff."""

from __future__ import annotations

from pathlib import Path

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.workflow_state_service import workflow_state_for
from openpkpd_gui.workflows.overview_workflow import format_recent_activity, recommended_next_action


def format_home_workspace_summary(
    project: Workspace, *, snapshot_path: str | None, is_dirty: bool
) -> str:
    """Summarize workspace-level context for the Home landing page."""
    project_count = len(project.projects)
    scenario_count = sum(len(project_model.scenarios) for project_model in project.projects)
    recent_count = len(project.recent_files)
    location = "Unsaved project snapshot"
    if snapshot_path:
        resolved = Path(snapshot_path)
        location = f"Snapshot: {resolved.name} • {resolved.parent}"
    elif project.root_path:
        location = f"Workspace: {project.root_path}"
    dirty_suffix = " • Unsaved changes pending" if is_dirty else ""
    return (
        f"{project_count} project{'s' if project_count != 1 else ''} • "
        f"{scenario_count} scenario{'s' if scenario_count != 1 else ''} • "
        f"{recent_count} recent snapshot{'s' if recent_count != 1 else ''}{dirty_suffix}\n"
        f"{location}"
    )


def format_home_active_context(project: Workspace) -> str:
    """Summarize the active project and scenario on the Home landing page."""
    data_state = workflow_state_for(project, "data")
    model_state = workflow_state_for(project, "model")
    fit_state = workflow_state_for(project, "fit")
    return (
        f"Active project: {project.active_project.name}\n"
        f"Active scenario: {project.active_scenario.name}\n"
        f"Data: {data_state.label} • {data_state.summary}\n"
        f"Model: {model_state.label} • {model_state.summary}\n"
        f"Fit: {fit_state.label} • {fit_state.summary}"
    )


def build_home_workflow(project: Workspace):
    """Build the workspace Home landing page."""
    _qt_core, _, qt_widgets = load_qt_modules()

    root = qt_widgets.QWidget()
    root.setObjectName("home-workflow")
    layout = qt_widgets.QVBoxLayout(root)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(12)

    title_label = qt_widgets.QLabel("Workspace Home")
    title_label.setObjectName("home-title-label")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 6)
    title_font.setBold(True)
    title_label.setFont(title_font)

    intro_label = qt_widgets.QLabel(
        "Start a new project, open a saved snapshot, or continue the active scenario from here."
    )
    intro_label.setObjectName("home-intro-label")
    intro_label.setWordWrap(True)

    workspace_summary_label = qt_widgets.QLabel("")
    workspace_summary_label.setObjectName("home-workspace-summary-label")
    workspace_summary_label.setWordWrap(True)

    active_context_label = qt_widgets.QLabel("")
    active_context_label.setObjectName("home-active-context-label")
    active_context_label.setWordWrap(True)

    recent_activity_label = qt_widgets.QLabel("")
    recent_activity_label.setObjectName("home-recent-activity-label")
    recent_activity_label.setWordWrap(True)

    next_action_label = qt_widgets.QLabel("")
    next_action_label.setObjectName("home-next-action-label")
    next_action_label.setWordWrap(True)

    action_row = qt_widgets.QWidget(root)
    action_row.setObjectName("home-action-row")
    action_layout = qt_widgets.QHBoxLayout(action_row)
    action_layout.setContentsMargins(0, 0, 0, 0)
    action_layout.setSpacing(8)

    new_project_button = qt_widgets.QPushButton("New Project…")
    new_project_button.setObjectName("home-new-project-button")
    open_project_button = qt_widgets.QPushButton("Open Project Snapshot…")
    open_project_button.setObjectName("home-open-project-button")
    next_action_button = qt_widgets.QPushButton("Open Overview")
    next_action_button.setObjectName("home-next-action-button")
    next_action_button.setProperty("primaryAction", True)
    next_action_button.setMinimumHeight(36)
    open_overview_button = qt_widgets.QPushButton("Open Overview")
    open_overview_button.setObjectName("home-open-overview-button")

    for button in (
        new_project_button,
        open_project_button,
        next_action_button,
        open_overview_button,
    ):
        action_layout.addWidget(button)
    action_layout.addStretch(1)

    recent_summary_label = qt_widgets.QLabel("")
    recent_summary_label.setObjectName("home-recent-summary-label")
    recent_summary_label.setWordWrap(True)

    recent_buttons: list[object] = []
    for index in range(3):
        button = qt_widgets.QPushButton("")
        button.setObjectName(f"home-recent-snapshot-button-{index}")
        button.setVisible(False)
        recent_buttons.append(button)

    layout.addWidget(title_label)
    layout.addWidget(intro_label)
    layout.addWidget(workspace_summary_label)
    layout.addWidget(active_context_label)
    layout.addWidget(recent_activity_label)
    layout.addWidget(next_action_label)
    layout.addWidget(action_row)
    layout.addWidget(recent_summary_label)
    for button in recent_buttons:
        layout.addWidget(button)
    layout.addStretch(1)

    next_action_target = ["overview"]

    def _navigate_to(workflow_id: str) -> None:
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(workflow_id)

    def _refresh() -> None:
        snapshot_path_getter = getattr(root, "_current_snapshot_path", None)
        snapshot_path = snapshot_path_getter() if callable(snapshot_path_getter) else None
        dirty_getter = getattr(root, "_project_dirty", None)
        is_dirty = bool(dirty_getter()) if callable(dirty_getter) else False
        workspace_summary_label.setText(
            format_home_workspace_summary(project, snapshot_path=snapshot_path, is_dirty=is_dirty)
        )
        active_context_label.setText(format_home_active_context(project))
        recent_activity_label.setText(format_recent_activity(project))
        button_text, workflow_id, summary = recommended_next_action(project)
        next_action_target[0] = workflow_id
        next_action_label.setText(summary)
        next_action_button.setText(button_text)
        if not project.recent_files:
            recent_summary_label.setText("No recent project snapshots yet.")
        else:
            recent_summary_label.setText("Recent project snapshots")
        for index, button in enumerate(recent_buttons):
            if index >= len(project.recent_files):
                button.setVisible(False)
                button.setText("")
                button.setProperty("snapshotPath", "")
                continue
            snapshot = Path(project.recent_files[index]).resolve()
            button.setText(f"{snapshot.name} — {snapshot.parent}")
            button.setToolTip(str(snapshot))
            button.setProperty("snapshotPath", str(snapshot))
            button.setVisible(True)

    def _open_recent_snapshot(button) -> None:
        snapshot_path = str(button.property("snapshotPath") or "")
        if not snapshot_path:
            return
        callback = getattr(root, "_open_recent_snapshot", None)
        if callable(callback):
            callback(snapshot_path)

    new_project_button.clicked.connect(lambda: getattr(root, "_create_project", lambda: None)())
    open_project_button.clicked.connect(
        lambda: getattr(root, "_choose_project_snapshot_to_open", lambda: None)()
    )
    next_action_button.clicked.connect(lambda: _navigate_to(next_action_target[0]))
    open_overview_button.clicked.connect(lambda: _navigate_to("dashboard"))
    for button in recent_buttons:
        button.clicked.connect(lambda _checked=False, widget=button: _open_recent_snapshot(widget))

    root._refresh_workflow = _refresh  # type: ignore[attr-defined]
    root._refresh_context_header = _refresh  # type: ignore[attr-defined]
    return root
