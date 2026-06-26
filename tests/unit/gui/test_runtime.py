"""Runtime checks for optional Qt GUI loading."""

from __future__ import annotations

from openpkpd_gui.app.runtime import qt_widgets_available, qt_widgets_error


def test_qt_runtime_status_is_consistent() -> None:
    error = qt_widgets_error()

    if qt_widgets_available():
        assert error is None
    else:
        assert error is not None
        assert "PySide6" in error
        # The message must point the user at the actionable install command.
        assert 'pip install "openpkpd[gui]"' in error
