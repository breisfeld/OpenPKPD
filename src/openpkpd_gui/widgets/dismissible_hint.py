"""Collapsible one-time hint label with a dismiss button."""

from __future__ import annotations

from openpkpd_gui.app.runtime import load_qt_modules


def build_dismissible_hint(
    text: str,
    *,
    dismissed: bool = False,
    object_name: str = "dismissible-hint",
    on_dismiss=None,
):
    """Return ``(container_widget, set_dismissed_fn)``.

    ``container_widget`` is the QWidget to insert into a layout.
    ``set_dismissed_fn(bool)`` hides (True) or shows (False) the widget.
    ``on_dismiss`` is an optional zero-argument callback invoked when the user
    clicks the ✕ button.
    """
    qt_core, _, qt_widgets = load_qt_modules()

    container = qt_widgets.QWidget()
    container.setObjectName(object_name)
    row = qt_widgets.QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)

    label = qt_widgets.QLabel(text)
    label.setWordWrap(True)

    dismiss_btn = qt_widgets.QToolButton()
    dismiss_btn.setObjectName("dismissible-hint-dismiss")
    dismiss_btn.setText("\u00d7")  # U+00D7 MULTIPLICATION SIGN — in every Latin-1 font
    dismiss_btn.setFixedSize(20, 20)
    dismiss_btn.setToolTip("Dismiss this hint")

    row.addWidget(label, 1)
    row.addWidget(dismiss_btn, 0, qt_core.Qt.AlignmentFlag.AlignTop)

    container.setVisible(not dismissed)

    def _dismiss():
        container.setVisible(False)
        if on_dismiss is not None:
            on_dismiss()

    dismiss_btn.clicked.connect(_dismiss)

    def set_dismissed(value: bool) -> None:
        container.setVisible(not value)

    return container, set_dismissed
