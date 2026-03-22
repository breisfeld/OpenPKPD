"""Dashboard workflow — merged workspace home and scenario overview."""

from __future__ import annotations

from pathlib import Path

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.workflows.overview_workflow import build_overview_workflow


def build_dashboard_workflow(project: Workspace):
    """Build the Dashboard page combining workspace bar and scenario overview."""
    _, _, qt_widgets = load_qt_modules()

    root = qt_widgets.QWidget()
    root.setObjectName("dashboard-workflow")
    root_layout = qt_widgets.QVBoxLayout(root)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(0)

    # --- Compact workspace bar ---
    workspace_bar = qt_widgets.QFrame(root)
    workspace_bar.setObjectName("dashboard-workspace-bar")
    workspace_bar.setFrameShape(qt_widgets.QFrame.Shape.StyledPanel)
    bar_layout = qt_widgets.QHBoxLayout(workspace_bar)
    bar_layout.setContentsMargins(8, 4, 8, 4)
    bar_layout.setSpacing(6)

    workspace_name_label = qt_widgets.QLabel("")
    workspace_name_label.setObjectName("dashboard-workspace-name-label")
    name_font = workspace_name_label.font()
    name_font.setBold(True)
    workspace_name_label.setFont(name_font)

    bar_layout.addWidget(workspace_name_label)
    bar_layout.addStretch(1)

    new_project_button = qt_widgets.QPushButton("New Project…")
    new_project_button.setObjectName("dashboard-new-project-button")

    open_snapshot_button = qt_widgets.QPushButton("Open Snapshot…")
    open_snapshot_button.setObjectName("dashboard-open-snapshot-button")

    bar_layout.addWidget(new_project_button)
    bar_layout.addWidget(open_snapshot_button)

    recent_buttons: list[object] = []
    for index in range(2):
        btn = qt_widgets.QPushButton("")
        btn.setObjectName(f"dashboard-recent-snapshot-button-{index}")
        btn.setVisible(False)
        bar_layout.addWidget(btn)
        recent_buttons.append(btn)

    # --- Separator ---
    separator = qt_widgets.QFrame(root)
    separator.setFrameShape(qt_widgets.QFrame.Shape.HLine)
    separator.setFrameShadow(qt_widgets.QFrame.Shadow.Sunken)
    separator.setObjectName("dashboard-separator")

    # --- Overview content ---
    overview_widget = build_overview_workflow(project)
    overview_widget.setParent(root)

    root_layout.addWidget(workspace_bar)
    root_layout.addWidget(separator)
    root_layout.addWidget(overview_widget, 1)

    def _refresh_workspace_bar() -> None:
        workspace_name_label.setText(project.name or "Untitled Project")
        for index, btn in enumerate(recent_buttons):
            if index >= len(project.recent_files):
                btn.setVisible(False)
                btn.setText("")
                btn.setProperty("snapshotPath", "")
                continue
            snapshot = Path(project.recent_files[index]).resolve()
            btn.setText(snapshot.name)
            btn.setToolTip(str(snapshot))
            btn.setProperty("snapshotPath", str(snapshot))
            btn.setVisible(True)

    def _open_recent_snapshot(btn) -> None:
        snapshot_path = str(btn.property("snapshotPath") or "")
        if not snapshot_path:
            return
        callback = getattr(root, "_open_recent_snapshot", None)
        if callable(callback):
            callback(snapshot_path)

    def _refresh() -> None:
        _refresh_workspace_bar()
        overview_refresh = getattr(overview_widget, "_refresh_workflow", None)
        if callable(overview_refresh):
            overview_refresh()

    def _forward_callback(name: str):
        """Forward a root callback attribute to overview_widget."""
        value = getattr(root, name, None)
        if value is not None:
            setattr(overview_widget, name, value)

    def _sync_callbacks() -> None:
        for name in (
            "_navigate_to_workflow",
            "_project_open_latest_report",
            "_project_open_latest_plot",
            "_navigate_to_results",
            "_create_project",
            "_choose_project_snapshot_to_open",
            "_current_snapshot_path",
            "_project_dirty",
        ):
            _forward_callback(name)

    new_project_button.clicked.connect(lambda: getattr(root, "_create_project", lambda: None)())
    open_snapshot_button.clicked.connect(
        lambda: getattr(root, "_choose_project_snapshot_to_open", lambda: None)()
    )
    for btn in recent_buttons:
        btn.clicked.connect(lambda _checked=False, widget=btn: _open_recent_snapshot(widget))

    def _refresh_with_callbacks() -> None:
        _sync_callbacks()
        _refresh()

    root._refresh_workflow = _refresh_with_callbacks  # type: ignore[attr-defined]
    root._refresh_context_header = _refresh_with_callbacks  # type: ignore[attr-defined]
    root._apply_responsive_layout = getattr(
        overview_widget, "_apply_responsive_layout", lambda: None
    )  # type: ignore[attr-defined]

    _refresh_workspace_bar()
    return root
