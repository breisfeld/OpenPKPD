"""Helpers for lightweight semantic state styling in Qt widgets."""

from __future__ import annotations

from openpkpd_gui.services.workflow_state_service import WorkflowState, WorkflowStateId

SEMANTIC_STATE_STYLESHEET = """
QLabel[semanticState], QPushButton[semanticState] {
    font-weight: 600;
}
QLabel[semanticState] {
    border-style: solid;
    border-width: 1px;
    border-radius: 6px;
    padding: 2px 8px;
}
QFrame[semanticState="not_started"], QLabel[semanticState="not_started"], QPushButton[semanticState="not_started"] {
    color: #5f6b7a;
    border-color: #c8d0da;
    background-color: #f4f6f8;
}
QFrame[semanticState="ready"], QLabel[semanticState="ready"], QPushButton[semanticState="ready"] {
    color: #25663b;
    border-color: #9ad4ad;
    background-color: #eef9f1;
}
QFrame[semanticState="running"], QLabel[semanticState="running"], QPushButton[semanticState="running"] {
    color: #0f5f7a;
    border-color: #8fd0e4;
    background-color: #edf8fc;
}
QFrame[semanticState="needs_attention"], QLabel[semanticState="needs_attention"], QPushButton[semanticState="needs_attention"] {
    color: #8a4f00;
    border-color: #e8c07c;
    background-color: #fff6e8;
}
QFrame[semanticState="results_available"], QLabel[semanticState="results_available"], QPushButton[semanticState="results_available"] {
    color: #5b3ea6;
    border-color: #ccbdf6;
    background-color: #f4efff;
}
QPushButton[semanticState] {
    border-width: 2px;
    border-style: solid;
    border-radius: 10px;
    padding: 8px 12px;
    text-align: left;
}
QPushButton[semanticState]:focus:!disabled {
    border-color: #334155;
}
QPushButton[semanticState="not_started"]:hover:!disabled {
    border-color: #94a3b8;
    background-color: #e9edf2;
}
QPushButton[semanticState="not_started"]:pressed:!disabled,
QPushButton[semanticState="not_started"]:checked:!disabled {
    border-color: #64748b;
    background-color: #dde3ea;
    padding-top: 9px;
    padding-bottom: 7px;
}
QPushButton[semanticState="ready"]:hover:!disabled {
    border-color: #67b981;
    background-color: #dff3e5;
}
QPushButton[semanticState="ready"]:pressed:!disabled,
QPushButton[semanticState="ready"]:checked:!disabled {
    border-color: #25663b;
    background-color: #cfead8;
    padding-top: 9px;
    padding-bottom: 7px;
}
QPushButton[semanticState="running"]:hover:!disabled {
    border-color: #54b4d3;
    background-color: #dff2f8;
}
QPushButton[semanticState="running"]:pressed:!disabled,
QPushButton[semanticState="running"]:checked:!disabled {
    border-color: #0f5f7a;
    background-color: #cde9f2;
    padding-top: 9px;
    padding-bottom: 7px;
}
QPushButton[semanticState="needs_attention"]:hover:!disabled {
    border-color: #d8a54c;
    background-color: #ffedd2;
}
QPushButton[semanticState="needs_attention"]:pressed:!disabled,
QPushButton[semanticState="needs_attention"]:checked:!disabled {
    border-color: #8a4f00;
    background-color: #ffe1b4;
    padding-top: 9px;
    padding-bottom: 7px;
}
QPushButton[semanticState="results_available"]:hover:!disabled {
    border-color: #a993ec;
    background-color: #eae2ff;
}
QPushButton[semanticState="results_available"]:pressed:!disabled,
QPushButton[semanticState="results_available"]:checked:!disabled {
    border-color: #5b3ea6;
    background-color: #ddd0ff;
    padding-top: 9px;
    padding-bottom: 7px;
}
QPushButton[semanticState]:disabled {
    color: #7b8593;
}
QLabel[semanticRole="eyebrow"] {
    color: #5f6b7a;
}
"""


def semantic_state_name(state: WorkflowState | WorkflowStateId | str) -> str:
    """Normalize a workflow state value into a stylesheet-friendly name."""
    if isinstance(state, WorkflowState):
        return state.state.value
    if isinstance(state, WorkflowStateId):
        return state.value
    return str(state)


def set_semantic_state(widget, state: WorkflowState | WorkflowStateId | str) -> None:
    """Apply the normalized semantic state property to one widget."""
    widget.setProperty("semanticState", semantic_state_name(state))
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)


def install_semantic_state_styles(widget) -> None:
    """Install the shared semantic state stylesheet on one widget if needed."""
    existing = widget.styleSheet()
    if SEMANTIC_STATE_STYLESHEET in existing:
        return
    widget.setStyleSheet(f"{existing}\n{SEMANTIC_STATE_STYLESHEET}".strip())
