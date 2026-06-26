"""Runtime checks for optional Qt GUI loading."""

from __future__ import annotations

import sys

import pytest

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


def test_gui_entrypoint_missing_dependency_names_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing ``[gui]`` dependency must surface the actionable install hint.

    The ``openpkpd-gui`` entry point imports ``openpkpd_gui.app.main`` and calls
    ``main()``.  In a core-only install the shell pulls in ``platformdirs`` (a
    ``[gui]``-only dependency) before Qt is ever touched, so ``main()`` must
    convert that ``ModuleNotFoundError`` into the same install guidance rather
    than letting a raw traceback escape.
    """

    class _BlockPlatformdirs:
        def find_spec(self, name, path=None, target=None):  # noqa: ANN001, ANN202
            if name == "platformdirs" or name.startswith("platformdirs."):
                raise ModuleNotFoundError(
                    "No module named 'platformdirs'", name="platformdirs"
                )
            return None

    # Force a fresh import of the shell so the blocked dependency is re-attempted.
    for mod in list(sys.modules):
        if mod.startswith("openpkpd_gui.shell") or mod.split(".", 1)[0] == "platformdirs":
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockPlatformdirs(), *sys.meta_path])

    from openpkpd_gui.app.main import main

    with pytest.raises(RuntimeError) as excinfo:
        main()

    message = str(excinfo.value)
    assert "platformdirs" in message
    assert 'pip install "openpkpd[gui]"' in message
