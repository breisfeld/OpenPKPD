"""Shared helpers for state-aware workflow shortcut sections."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.services.workflow_state_service import WorkflowState, WorkflowStateId
from openpkpd_gui.widgets.semantic_state import install_semantic_state_styles, set_semantic_state


@dataclass(frozen=True, slots=True)
class WorkflowShortcutSpec:
    """Declarative configuration for one workflow shortcut button."""

    workflow_id: str
    label: str
    object_name: str


@dataclass(frozen=True, slots=True)
class WorkflowShortcutGroupSpec:
    """Declarative configuration for a grouped workflow shortcut subsection."""

    key: str
    title: str
    summary_object_name: str
    workflow_specs: tuple[WorkflowShortcutSpec, ...]


def format_workflow_shortcut_text(label: str, state: WorkflowState) -> str:
    """Return state-aware text for a workflow shortcut."""
    return f"Open {label} ({state.label})"


def format_workflow_shortcut_tooltip(label: str, state: WorkflowState) -> str:
    """Return tooltip text for a workflow shortcut."""
    return f"Open {label}. Status: {state.label} — {state.summary}"


def build_workflow_shortcut_section(root, *, title: str, summary_object_name: str, workflow_specs):
    """Build a section with a summary label and workflow shortcut buttons."""
    _qt_core, _, qt_widgets = load_qt_modules()

    group = qt_widgets.QGroupBox(title, root)
    section_name = summary_object_name.removesuffix("-summary-label")
    group.setObjectName(f"{section_name}-section")
    install_semantic_state_styles(group)
    layout = qt_widgets.QVBoxLayout(group)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)
    summary_label = qt_widgets.QLabel()
    summary_label.setObjectName(summary_object_name)
    summary_label.setWordWrap(True)

    row = qt_widgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    buttons: dict[str, object] = {}
    labels: dict[str, str] = {}
    for spec in workflow_specs:
        button = qt_widgets.QPushButton(f"Open {spec.label}")
        button.setObjectName(spec.object_name)
        button.setMinimumHeight(32)
        row.addWidget(button)
        buttons[spec.workflow_id] = button
        labels[spec.workflow_id] = spec.label
    row.addStretch(1)

    layout.addWidget(summary_label)
    layout.addLayout(row)
    return group, summary_label, row, buttons, labels


def build_grouped_workflow_shortcut_section(root, *, title: str, object_name: str, group_specs):
    """Build one compact section containing multiple workflow shortcut groups."""
    _qt_core, _, qt_widgets = load_qt_modules()

    group = qt_widgets.QGroupBox(title, root)
    group.setObjectName(object_name)
    install_semantic_state_styles(group)
    layout = qt_widgets.QVBoxLayout(group)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(12)

    summary_labels: dict[str, object] = {}
    rows: dict[str, object] = {}
    buttons: dict[str, dict[str, object]] = {}
    labels: dict[str, dict[str, str]] = {}

    for index, group_spec in enumerate(group_specs):
        section_name = group_spec.summary_object_name.removesuffix("-summary-label")
        section = qt_widgets.QWidget(group)
        section.setObjectName(f"{section_name}-section")
        section_layout = qt_widgets.QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(6)

        heading_label = qt_widgets.QLabel(group_spec.title, section)
        heading_label.setObjectName(f"{section_name}-heading")
        heading_font = heading_label.font()
        heading_font.setBold(True)
        heading_label.setFont(heading_font)

        summary_label = qt_widgets.QLabel(section)
        summary_label.setObjectName(group_spec.summary_object_name)
        summary_label.setWordWrap(True)

        row = qt_widgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        section_buttons: dict[str, object] = {}
        section_labels: dict[str, str] = {}
        for spec in group_spec.workflow_specs:
            button = qt_widgets.QPushButton(f"Open {spec.label}")
            button.setObjectName(spec.object_name)
            button.setMinimumHeight(32)
            row.addWidget(button)
            section_buttons[spec.workflow_id] = button
            section_labels[spec.workflow_id] = spec.label
        row.addStretch(1)

        section_layout.addWidget(heading_label)
        section_layout.addWidget(summary_label)
        section_layout.addLayout(row)
        layout.addWidget(section)

        if index < len(group_specs) - 1:
            divider = qt_widgets.QFrame(group)
            divider.setFrameShape(qt_widgets.QFrame.Shape.HLine)
            divider.setFrameShadow(qt_widgets.QFrame.Shadow.Plain)
            divider.setObjectName(f"{section_name}-divider")
            layout.addWidget(divider)

        summary_labels[group_spec.key] = summary_label
        rows[group_spec.key] = row
        buttons[group_spec.key] = section_buttons
        labels[group_spec.key] = section_labels

    return group, summary_labels, rows, buttons, labels


def refresh_workflow_shortcut_buttons(
    buttons: dict[str, object],
    labels: dict[str, str],
    states: dict[str, WorkflowState],
    *,
    enable_not_started: Iterable[str] = (),
) -> None:
    """Refresh shortcut button text, enablement, and tooltip from workflow states."""
    enabled_when_not_started = set(enable_not_started)
    for workflow_id, button in buttons.items():
        state = states[workflow_id]
        label = labels[workflow_id]
        button.setText(format_workflow_shortcut_text(label, state))
        button.setEnabled(
            workflow_id in enabled_when_not_started or state.state != WorkflowStateId.NOT_STARTED
        )
        button.setToolTip(format_workflow_shortcut_tooltip(label, state))
        set_semantic_state(button, state)
