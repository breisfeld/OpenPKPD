"""Runtime helpers for optional Qt GUI loading."""

from __future__ import annotations


def load_qt_modules():
    """Import and return the Qt modules needed by the desktop shell."""
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "PySide6 is not installed in this environment, so Qt GUI modules are "
            "unavailable. The desktop GUI is an optional extra; install it with: "
            'pip install "openpkpd[gui]"'
        ) from exc
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "PySide6 is installed, but Qt GUI modules could not be loaded in this "
            f"environment. Original error: {exc}"
        ) from exc
    return QtCore, QtGui, QtWidgets


def qt_widgets_available() -> bool:
    """Return True when Qt GUI modules can be imported successfully."""
    try:
        load_qt_modules()
    except RuntimeError:
        return False
    return True


def qt_widgets_error() -> str | None:
    """Return a readable Qt GUI import error, if one exists."""
    try:
        load_qt_modules()
    except RuntimeError as exc:
        return str(exc)
    return None
