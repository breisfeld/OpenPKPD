"""Compact shared workflow status strip widget."""

from __future__ import annotations

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.workflow_state_service import workflow_states_for
from openpkpd_gui.widgets.semantic_state import install_semantic_state_styles, set_semantic_state
from openpkpd_gui.workflows.registry import DEFAULT_WORKFLOWS


def build_workflow_status_strip(
    root, project: Workspace, *, object_prefix: str, workflow_ids: tuple[str, ...]
):
    """Build a row of lightweight status chips plus a refresh callback."""
    _qt_core, _, qt_widgets = load_qt_modules()

    widget = qt_widgets.QWidget(root)
    widget.setObjectName(f"{object_prefix}-status-strip")
    widget.setProperty("surfaceRole", "workflow-status-strip")
    install_semantic_state_styles(widget)
    layout = qt_widgets.QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    workflow_labels = {workflow.workflow_id: workflow.label for workflow in DEFAULT_WORKFLOWS}
    chips: dict[str, object] = {}
    for workflow_id in workflow_ids:
        chip = qt_widgets.QLabel()
        chip.setObjectName(f"{object_prefix}-status-chip-{workflow_id}")
        chip.setMargin(6)
        chip.setFrameShape(qt_widgets.QFrame.Shape.StyledPanel)
        chip.setWordWrap(True)
        layout.addWidget(chip)
        chips[workflow_id] = chip
    layout.addStretch(1)

    def _refresh() -> None:
        states = workflow_states_for(project, workflow_ids)
        for workflow_id, chip in chips.items():
            state = states[workflow_id]
            label = workflow_labels.get(workflow_id, workflow_id.title())
            chip.setText(f"{label}: {state.label}")
            chip.setToolTip(state.summary)
            set_semantic_state(chip, state)

    _refresh()
    return widget, _refresh
