"""Small reusable collapsible section helper for progressive disclosure."""

from __future__ import annotations

from openpkpd_gui.app.runtime import load_qt_modules


def build_collapsible_section(
    parent,
    *,
    title: str,
    object_name: str,
    expanded: bool = False,
    framed: bool = False,
):
    """Return a collapsible section with a toggle header and content area."""
    qt_core, _, qt_widgets = load_qt_modules()

    root = qt_widgets.QWidget(parent)
    root.setObjectName(object_name)
    root.setProperty("collapsibleSection", True)
    root.setProperty("collapsibleFrame", framed)

    layout = qt_widgets.QVBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    toggle = qt_widgets.QToolButton(root)
    toggle.setObjectName(f"{object_name}-toggle")
    toggle.setProperty("collapsibleHeader", True)
    toggle.setCheckable(True)
    toggle.setChecked(expanded)
    toggle.setCursor(qt_core.Qt.CursorShape.PointingHandCursor)
    toggle.setToolButtonStyle(qt_core.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    toggle.setSizePolicy(
        qt_widgets.QSizePolicy.Policy.Expanding, qt_widgets.QSizePolicy.Policy.Fixed
    )
    toggle.setText(title)

    content = qt_widgets.QWidget(root)
    content.setObjectName(f"{object_name}-content")
    content.setProperty("collapsibleContent", True)
    content_layout = qt_widgets.QVBoxLayout(content)
    content_layout.setContentsMargins(12, 0, 12, 12)
    content_layout.setSpacing(8)

    layout.addWidget(toggle)
    layout.addWidget(content)

    def _set_expanded(is_expanded: bool) -> None:
        toggle.setArrowType(
            qt_core.Qt.ArrowType.DownArrow if is_expanded else qt_core.Qt.ArrowType.RightArrow
        )
        content.setVisible(is_expanded)
        root.setProperty("expanded", is_expanded)
        root.updateGeometry()

    toggle.toggled.connect(_set_expanded)
    root._set_expanded = _set_expanded  # type: ignore[attr-defined]
    root._toggle = toggle  # type: ignore[attr-defined]
    _set_expanded(expanded)
    return root, content, content_layout, toggle
