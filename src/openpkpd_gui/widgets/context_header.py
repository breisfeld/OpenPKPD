"""Shared context header widget for GUI workflows."""

from __future__ import annotations

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.workflow_state_service import (
    latest_run_for_workflows,
    workflow_state_for,
)
from openpkpd_gui.widgets.semantic_state import install_semantic_state_styles, set_semantic_state


def _context_run(project: Workspace, workflow_id: str) -> RunRecord | None:
    scenario = project.active_scenario
    if workflow_id in {"home", "overview", "dashboard"}:
        workflow_ids = {run.workflow for run in scenario.runs}
        return latest_run_for_workflows(scenario, workflow_ids) if workflow_ids else None
    if workflow_id in {"results", "plots", "diagnostics", "advanced"}:
        return latest_run_for_workflows(scenario, {"fit", "vpc", "bootstrap", "design", "npde"})
    return latest_run_for_workflows(scenario, {workflow_id})


def _run_summary(run: RunRecord | None) -> str:
    if run is None:
        return "No recent run"
    if run.status == RunStatus.SUCCEEDED:
        detail = run.summary_text or "Succeeded"
    elif run.status == RunStatus.FAILED:
        detail = run.error_text or "Failed"
    else:
        detail = run.summary_text or run.status.value.title()
    return f"Latest run: {run.workflow} • {detail}"


def build_context_header(root, project: Workspace, *, workflow_id: str, workflow_label: str):
    """Build a reusable breadcrumb/context header and a refresh callback."""
    qt_core, _, qt_widgets = load_qt_modules()

    container = qt_widgets.QWidget(root)
    container.setObjectName(f"{workflow_id}-context-header")
    container.setProperty("surfaceRole", "workflow-header")
    install_semantic_state_styles(container)
    layout = qt_widgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    breadcrumb_label = qt_widgets.QLabel()
    breadcrumb_label.setObjectName(f"{workflow_id}-context-breadcrumb")
    breadcrumb_label.setTextInteractionFlags(qt_core.Qt.TextInteractionFlag.TextSelectableByMouse)

    summary_label = qt_widgets.QLabel()
    summary_label.setObjectName(f"{workflow_id}-context-summary")
    summary_label.setWordWrap(True)

    state_label = qt_widgets.QLabel()
    state_label.setObjectName(f"{workflow_id}-context-state")
    state_label.setWordWrap(True)

    layout.addWidget(breadcrumb_label)
    layout.addWidget(summary_label)
    layout.addWidget(state_label)

    def _refresh() -> None:
        scenario = project.active_scenario
        active_project = project.active_project
        state = workflow_state_for(project, workflow_id)
        dirty_callback = getattr(root, "_project_dirty", None)
        dirty_text = "Workspace clean"
        if callable(dirty_callback) and bool(dirty_callback()):
            dirty_text = "Unsaved workspace changes"
        breadcrumb_label.setText(
            f"Workspace / {active_project.name} / {scenario.name} / {workflow_label}"
        )
        summary_label.setText(f"{dirty_text} • {_run_summary(_context_run(project, workflow_id))}")
        state_label.setText(f"Status: {state.label} — {state.summary}")
        set_semantic_state(state_label, state)

    _refresh()
    return container, _refresh
