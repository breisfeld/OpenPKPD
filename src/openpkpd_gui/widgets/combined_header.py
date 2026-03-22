"""Single-row header combining breadcrumb and workflow status chips."""

from __future__ import annotations

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.workflow_state_service import workflow_state_for, workflow_states_for
from openpkpd_gui.widgets.context_header import _context_run, _run_summary
from openpkpd_gui.widgets.semantic_state import install_semantic_state_styles, set_semantic_state
from openpkpd_gui.workflows.registry import DEFAULT_WORKFLOWS


def build_combined_header(
    root,
    project: Workspace,
    *,
    workflow_id: str,
    workflow_label: str,
    status_workflow_ids: tuple[str, ...],
):
    """Build a compact single-row header: breadcrumb left · status chips right.

    Returns (container_widget, refresh_fn).
    """
    _, _, qt_widgets = load_qt_modules()

    object_prefix = f"{workflow_id}-workflow"
    workflow_labels = {w.workflow_id: w.label for w in DEFAULT_WORKFLOWS}

    container = qt_widgets.QWidget(root)
    container.setObjectName(f"{object_prefix}-combined-header")
    container.setProperty("surfaceRole", "workflow-header")
    install_semantic_state_styles(container)

    layout = qt_widgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 2, 0, 2)
    layout.setSpacing(8)

    breadcrumb_label = qt_widgets.QLabel()
    breadcrumb_label.setObjectName(f"{object_prefix}-breadcrumb")
    layout.addWidget(breadcrumb_label, 1)

    chips: dict[str, object] = {}
    for wid in status_workflow_ids:
        chip = qt_widgets.QLabel()
        chip.setObjectName(f"{object_prefix}-status-chip-{wid}")
        chip.setMargin(4)
        chip.setFrameShape(qt_widgets.QFrame.Shape.StyledPanel)
        layout.addWidget(chip)
        chips[wid] = chip

    def _refresh() -> None:
        scenario = project.active_scenario
        active_project = project.active_project
        state = workflow_state_for(project, workflow_id)
        dirty_callback = getattr(root, "_project_dirty", None)
        dirty_text = "Clean"
        if callable(dirty_callback) and bool(dirty_callback()):
            dirty_text = "Unsaved changes"
        breadcrumb_label.setText(
            f"Workspace / {active_project.name} / {scenario.name} / {workflow_label}"
        )
        run_text = _run_summary(_context_run(project, workflow_id))
        breadcrumb_label.setToolTip(
            f"{dirty_text} • {run_text}\nStatus: {state.label} — {state.summary}"
        )
        states = workflow_states_for(project, status_workflow_ids)
        for wid, chip in chips.items():
            chip_state = states[wid]
            label = workflow_labels.get(wid, wid.title())
            chip.setText(f"{label}: {chip_state.label}")
            chip.setToolTip(chip_state.summary)
            set_semantic_state(chip, chip_state)

    _refresh()
    return container, _refresh
