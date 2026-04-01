"""Tests for output_browser interactive plot panel (P2-A)."""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import pytest


def _get_app():
    from openpkpd_gui.app.runtime import load_qt_modules

    qt_core, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    return (qt_core, qt_gui, qt_widgets), app


def _make_panel(prefix="test"):
    from openpkpd_gui.widgets.output_browser import build_output_preview_panel

    _qt_modules, _app = _get_app()
    panel = build_output_preview_panel(None, object_prefix=prefix)
    return panel, _qt_modules


def _write_minimal_png(path: Path) -> None:
    """Write a valid 1×1 white PNG to *path*."""
    def chunk(name: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + name + data
        return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat_raw = b"\x00\xff\xff\xff"  # filter=none, R=G=B=255
    idat_data = zlib.compress(idat_raw)
    png = sig + chunk(b"IHDR", ihdr_data) + chunk(b"IDAT", idat_data) + chunk(b"IEND", b"")
    path.write_bytes(png)


class TestBuildOutputPreviewPanel:
    def test_returns_panel_instance(self):
        from openpkpd_gui.widgets.output_browser import OutputPreviewPanel

        panel, _ = _make_panel()
        assert isinstance(panel, OutputPreviewPanel)

    def test_stack_has_mpl_canvas_container(self):
        panel, (qt_core, qt_gui, qt_widgets) = _make_panel()
        assert panel.mpl_canvas_container is not None

    def test_mpl_canvas_present(self):
        panel, _ = _make_panel()
        assert panel.mpl_canvas is not None

    def test_mpl_ax_present(self):
        panel, _ = _make_panel()
        assert panel.mpl_ax is not None

    def test_mpl_canvas_objectname(self):
        panel, _ = _make_panel("diag")
        assert panel.mpl_canvas.objectName() == "diag-preview-mpl-canvas"

    def test_mpl_toolbar_objectname(self):
        panel, (qt_core, qt_gui, qt_widgets) = _make_panel("diag")
        container = panel.mpl_canvas_container
        toolbar = container.findChild(qt_widgets.QWidget, "diag-preview-mpl-toolbar")
        assert toolbar is not None

    def test_stack_contains_mpl_container(self):
        panel, (qt_core, qt_gui, qt_widgets) = _make_panel()
        found = False
        for i in range(panel.stack.count()):
            if panel.stack.widget(i) is panel.mpl_canvas_container:
                found = True
        assert found


class TestOutputPreviewPanelImageRendering:
    def test_image_artifact_shows_mpl_canvas(self, tmp_path):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("img")
        img_path = tmp_path / "plot.png"
        _write_minimal_png(img_path)

        artifact = ArtifactRecord(kind="image", label="Test plot", path=str(img_path))
        panel.render(artifact)
        assert panel.stack.currentWidget() is panel.mpl_canvas_container

    def test_image_artifact_not_showing_scroll(self, tmp_path):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("img2")
        img_path = tmp_path / "plot.png"
        _write_minimal_png(img_path)

        artifact = ArtifactRecord(kind="image", label="Test plot", path=str(img_path))
        panel.render(artifact)
        # Should NOT be showing the old static scroll widget
        assert panel.stack.currentWidget() is not panel.scroll

    def test_mpl_ax_not_empty_after_image_render(self, tmp_path):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("img3")
        img_path = tmp_path / "plot.png"
        _write_minimal_png(img_path)

        artifact = ArtifactRecord(kind="image", label="Test plot", path=str(img_path))
        panel.render(artifact)
        # After rendering an image, axes should have children (the AxesImage)
        assert len(panel.mpl_ax.images) > 0

    def test_none_artifact_shows_placeholder(self):
        panel, _ = _make_panel("none")
        panel.render(None)
        assert panel.stack.currentWidget() is panel.placeholder

    def test_missing_file_shows_placeholder(self):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("miss")
        artifact = ArtifactRecord(
            kind="image", label="Missing", path="/nonexistent/path/plot.png"
        )
        panel.render(artifact)
        assert panel.stack.currentWidget() is panel.placeholder

    def test_html_artifact_shows_browser(self, tmp_path):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("html")
        html_path = tmp_path / "report.html"
        html_path.write_text("<html><body>Hello</body></html>")

        artifact = ArtifactRecord(kind="report", label="Report", path=str(html_path))
        panel.render(artifact)
        assert panel.stack.currentWidget() is panel.browser

    def test_text_artifact_shows_browser(self, tmp_path):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("txt")
        txt_path = tmp_path / "output.txt"
        txt_path.write_text("Some text output")

        artifact = ArtifactRecord(kind="log", label="Log", path=str(txt_path))
        panel.render(artifact)
        assert panel.stack.currentWidget() is panel.browser

    def test_second_image_replaces_first(self, tmp_path):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("two")
        img1 = tmp_path / "plot1.png"
        img2 = tmp_path / "plot2.png"
        _write_minimal_png(img1)
        _write_minimal_png(img2)

        for path, label in [(img1, "Plot 1"), (img2, "Plot 2")]:
            artifact = ArtifactRecord(kind="image", label=label, path=str(path))
            panel.render(artifact)

        assert panel.stack.currentWidget() is panel.mpl_canvas_container
        assert len(panel.mpl_ax.images) > 0


# ---------------------------------------------------------------------------
# P2-B: render_highlighted_artifact
# ---------------------------------------------------------------------------


def _make_diag_df():
    """Return a minimal diagnostics DataFrame with two subjects."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not available")
    return pd.DataFrame(
        {
            "ID": ["001", "001", "002", "002"],
            "TIME": [0.0, 1.0, 0.0, 2.0],
            "DV": [1.0, 2.0, 1.5, 2.5],
            "PRED": [1.1, 1.9, 1.4, 2.6],
            "IPRED": [1.05, 1.95, 1.45, 2.55],
            "CWRES": [0.1, -0.2, 0.3, -0.1],
            "IWRES": [0.05, -0.1, 0.15, -0.05],
        }
    )


class TestRenderHighlightedArtifact:
    def test_returns_true_for_supported_plot_type(self):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("hi1")
        df = _make_diag_df()
        artifact = ArtifactRecord(
            kind="plot",
            label="DV vs IPRED",
            path=None,
            metadata={"plot_type": "dv_vs_ipred"},
        )
        result = panel.render_highlighted_artifact(df, artifact, "001")
        assert result is True

    def test_returns_false_for_unsupported_plot_type(self):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("hi2")
        df = _make_diag_df()
        artifact = ArtifactRecord(
            kind="plot",
            label="GOF Panel",
            path=None,
            metadata={"plot_type": "gof_panel"},
        )
        result = panel.render_highlighted_artifact(df, artifact, "001")
        assert result is False

    def test_returns_false_when_no_mpl_canvas(self):
        from openpkpd_gui.domain.artifact import ArtifactRecord
        from openpkpd_gui.widgets.output_browser import OutputPreviewPanel

        _qt_modules, _app = _get_app()
        qt_core, qt_gui, qt_widgets = _qt_modules
        stack = qt_widgets.QStackedWidget()
        placeholder = qt_widgets.QLabel()
        stack.addWidget(placeholder)
        panel_no_mpl = OutputPreviewPanel(
            title_label=qt_widgets.QLabel(),
            metadata_label=qt_widgets.QLabel(),
            placeholder=placeholder,
            browser=qt_widgets.QTextBrowser(),
            image_label=qt_widgets.QLabel(),
            scroll=qt_widgets.QScrollArea(),
            stack=stack,
            qt_core=qt_core,
            qt_gui=qt_gui,
            table_widget=None,
            mpl_canvas_container=None,
            mpl_canvas=None,
            mpl_ax=None,
        )
        df = _make_diag_df()
        artifact = ArtifactRecord(
            kind="plot", label="DV vs IPRED", path=None, metadata={"plot_type": "dv_vs_ipred"}
        )
        result = panel_no_mpl.render_highlighted_artifact(df, artifact, "001")
        assert result is False

    def test_shows_mpl_canvas_container_after_highlight(self):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("hi3")
        df = _make_diag_df()
        artifact = ArtifactRecord(
            kind="plot",
            label="CWRES vs TIME",
            path=None,
            metadata={"plot_type": "cwres_vs_time"},
        )
        panel.render_highlighted_artifact(df, artifact, "001")
        assert panel.stack.currentWidget() is panel.mpl_canvas_container

    def test_all_supported_gof_plot_types_return_true(self):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        supported = [
            "dv_vs_ipred",
            "dv_vs_pred",
            "cwres_vs_time",
            "cwres_vs_pred",
            "abs_iwres_vs_ipred",
        ]
        df = _make_diag_df()
        for plot_type in supported:
            panel, _ = _make_panel(f"hi_{plot_type}")
            artifact = ArtifactRecord(
                kind="plot",
                label=plot_type,
                path=None,
                metadata={"plot_type": plot_type},
            )
            result = panel.render_highlighted_artifact(df, artifact, "001")
            assert result is True, f"Expected True for plot_type={plot_type}"

    def test_subject_not_in_df_still_renders(self):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("hi4")
        df = _make_diag_df()
        artifact = ArtifactRecord(
            kind="plot",
            label="DV vs IPRED",
            path=None,
            metadata={"plot_type": "dv_vs_ipred"},
        )
        # Subject "999" doesn't exist in df — should still render without error
        result = panel.render_highlighted_artifact(df, artifact, "999")
        assert result is True

    def test_no_metadata_returns_false(self):
        from openpkpd_gui.domain.artifact import ArtifactRecord

        panel, _ = _make_panel("hi5")
        df = _make_diag_df()
        artifact = ArtifactRecord(kind="plot", label="No meta", path=None, metadata=None)
        result = panel.render_highlighted_artifact(df, artifact, "001")
        assert result is False
