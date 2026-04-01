"""Tests for model_diagram — compartment SVG diagram widget (P1-E)."""

from __future__ import annotations

import re

import pytest

from openpkpd_gui.widgets.model_diagram import (
    W,
    H,
    get_diagram_svg,
    build_model_diagram_widget,
    _DIAGRAM_MAP,
)


# ---------------------------------------------------------------------------
# SVG content tests (no Qt needed)
# ---------------------------------------------------------------------------


class TestGetDiagramSvg:
    """get_diagram_svg() should return valid SVG for all known (advan, trans) pairs."""

    @pytest.mark.parametrize(
        "advan, trans",
        [
            (1, 1),   # 1-cmt IV
            (2, 2),   # 1-cmt oral
            (1, 2),   # 1-cmt MM
            (3, 4),   # 2-cmt IV
            (4, 4),   # 2-cmt oral
            (5, 1),   # N-cmt general
            (11, 4),  # 3-cmt IV
            (12, 4),  # 3-cmt oral
            (6, 1),   # ODE ADVAN6
            (8, 1),   # ODE ADVAN8
        ],
    )
    def test_known_models_return_svg(self, advan, trans):
        svg = get_diagram_svg(advan, trans)
        assert svg.startswith("<svg"), f"Expected SVG for ADVAN{advan}/TRANS{trans}"
        assert "</svg>" in svg

    def test_unknown_model_returns_custom_diagram(self):
        svg = get_diagram_svg(99, 99)
        assert "<svg" in svg
        assert "Custom" in svg

    def test_svg_viewbox_dimensions(self):
        """All diagrams must use the standard viewport."""
        for (advan, trans), svg in _DIAGRAM_MAP.items():
            assert f'viewBox="0 0 {W} {H}"' in svg, (
                f"ADVAN{advan}/TRANS{trans} SVG missing expected viewBox"
            )

    def test_svg_contains_defs_arrowhead(self):
        """All diagrams must include the arrowhead marker definition."""
        svg = get_diagram_svg(2, 2)
        assert '<defs>' in svg
        assert 'marker id="arr"' in svg

    def test_1cmt_iv_mentions_v_and_cl(self):
        svg = get_diagram_svg(1, 1)
        assert "Central" in svg
        assert "CL" in svg

    def test_1cmt_oral_mentions_depot_and_central(self):
        svg = get_diagram_svg(2, 2)
        assert "Depot" in svg
        assert "Central" in svg
        assert "Ka" in svg

    def test_2cmt_iv_mentions_periph(self):
        svg = get_diagram_svg(3, 4)
        assert "Periph" in svg
        assert "Central" in svg

    def test_2cmt_oral_has_depot(self):
        svg = get_diagram_svg(4, 4)
        assert "Depot" in svg
        assert "Periph" in svg

    def test_ode_models_mention_user_ode(self):
        for advan in (6, 8):
            svg = get_diagram_svg(advan, 1)
            assert "ODE" in svg or "ADVAN" in svg, (
                f"ODE diagram for ADVAN{advan} should mention ODE/ADVAN"
            )

    def test_all_svgs_are_valid_xml_structure(self):
        """Each SVG should open and close its root element."""
        for (advan, trans), svg in _DIAGRAM_MAP.items():
            open_count = svg.count("<svg")
            close_count = svg.count("</svg>")
            assert open_count == 1 and close_count == 1, (
                f"ADVAN{advan}/TRANS{trans}: malformed SVG (open={open_count}, close={close_count})"
            )

    def test_diagrams_have_rect_elements(self):
        """Every diagram should have at least one compartment box."""
        for (advan, trans), svg in _DIAGRAM_MAP.items():
            assert "<rect" in svg, f"ADVAN{advan}/TRANS{trans}: no compartment boxes found"

    def test_diagrams_have_line_arrows(self):
        """Most diagrams (except single-box models) should have at least one arrow."""
        # Single-box models: ADVAN1/TRANS1 (1-cmt IV), ADVAN5 (N-cmt general), ODE
        single_box = {(1, 1), (5, 1), (6, 1), (8, 1)}
        for (advan, trans), svg in _DIAGRAM_MAP.items():
            if (advan, trans) in single_box:
                continue
            assert "<line" in svg, (
                f"ADVAN{advan}/TRANS{trans}: expected at least one arrow line"
            )


# ---------------------------------------------------------------------------
# Qt widget tests
# ---------------------------------------------------------------------------


def _get_app():
    from openpkpd_gui.app.runtime import load_qt_modules
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    return qt_widgets.QApplication.instance() or qt_widgets.QApplication([])


class TestBuildModelDiagramWidget:
    def test_widget_has_correct_object_name(self):
        _get_app()
        widget = build_model_diagram_widget(2, 2)
        assert widget.objectName() == "model-diagram-svg"

    def test_widget_has_fixed_size(self):
        _get_app()
        widget = build_model_diagram_widget(2, 2)
        assert widget.width() == W
        assert widget.height() == H

    def test_widget_has_update_diagram_method(self):
        _get_app()
        widget = build_model_diagram_widget(2, 2)
        assert callable(getattr(widget, "update_diagram", None))

    def test_update_diagram_does_not_raise(self):
        """Calling update_diagram with various ADVAN/TRANS should not raise."""
        _get_app()
        widget = build_model_diagram_widget(1, 1)
        for advan, trans in [(2, 2), (3, 4), (4, 4), (6, 1), (99, 99)]:
            widget.update_diagram(advan, trans)  # must not raise

    def test_default_diagram_loaded_on_construction(self):
        """Widget should already have loaded its initial SVG."""
        _get_app()
        widget = build_model_diagram_widget(2, 2)
        # QSvgWidget.renderer() is valid if SVG was loaded successfully
        assert widget.renderer().isValid()


# ---------------------------------------------------------------------------
# Integration: diagram widget is present in build_model_workflow
# ---------------------------------------------------------------------------


class TestModelWorkflowDiagramIntegration:
    def test_diagram_widget_present_in_model_workflow(self):
        from openpkpd_gui.app.runtime import load_qt_modules
        from openpkpd_gui.workflows.model_workflow import build_model_workflow
        from openpkpd_gui.domain.workspace import Workspace
        from PySide6 import QtSvgWidgets

        qt_core, qt_gui, qt_widgets = load_qt_modules()
        _get_app()

        ws = Workspace()
        widget = build_model_workflow(ws)
        diagram = widget.findChild(QtSvgWidgets.QSvgWidget, "model-diagram-svg")
        assert diagram is not None, "model-diagram-svg widget not found in model workflow"

    def test_diagram_widget_has_valid_svg_loaded(self):
        from openpkpd_gui.app.runtime import load_qt_modules
        from openpkpd_gui.workflows.model_workflow import build_model_workflow
        from openpkpd_gui.domain.workspace import Workspace
        from PySide6 import QtSvgWidgets

        qt_core, qt_gui, qt_widgets = load_qt_modules()
        _get_app()

        ws = Workspace()
        widget = build_model_workflow(ws)
        diagram = widget.findChild(QtSvgWidgets.QSvgWidget, "model-diagram-svg")
        assert diagram is not None
        assert diagram.renderer().isValid()
