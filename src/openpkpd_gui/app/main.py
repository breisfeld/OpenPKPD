"""Main entry point for the OpenPKPD desktop GUI."""

from __future__ import annotations

import importlib.resources
import sys

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.shell.main_window import create_main_window



def configure_application_style(app) -> None:
    """Apply a conservative modern Qt-native style to the GUI."""
    from openpkpd_gui.app.theme import LIGHT, build_palette, build_stylesheet

    _, qt_gui, qt_widgets = load_qt_modules()
    fusion_style = qt_widgets.QStyleFactory.create("Fusion")
    if fusion_style is not None:
        fusion_style.setObjectName("Fusion")
        app.setStyle(fusion_style)
    else:
        app.setStyle("Fusion")

    base_ss = build_stylesheet(LIGHT)
    app.setPalette(build_palette(LIGHT, qt_gui))
    app.setStyleSheet(base_ss)
    app.setProperty("openpkpd_base_stylesheet", base_ss)


def main() -> int:
    """Launch the desktop GUI and return the Qt exit code."""
    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(sys.argv)
    app.setApplicationName("OpenPKPD")
    app.setOrganizationName("OpenPKPD")
    configure_application_style(app)

    _, qt_gui, _ = load_qt_modules()
    icon_ref = importlib.resources.files("openpkpd_gui.resources").joinpath("icon.ico")
    with importlib.resources.as_file(icon_ref) as icon_path:
        app.setWindowIcon(qt_gui.QIcon(str(icon_path)))

    window = create_main_window(Workspace())
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
