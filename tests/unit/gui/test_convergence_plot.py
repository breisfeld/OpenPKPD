"""Tests for convergence_plot widget (P1-A)."""

from __future__ import annotations

import pytest


def _get_app():
    from openpkpd_gui.app.runtime import load_qt_modules

    qt_core, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    return (qt_core, qt_gui, qt_widgets), app


def _make():
    from openpkpd_gui.widgets.convergence_plot import build_convergence_plot_widget

    qt_modules, _app = _get_app()
    widget, add_ofv, reset, finalize = build_convergence_plot_widget(qt_modules)
    return widget, add_ofv, reset, finalize, qt_modules


class TestBuildConvergencePlotWidget:
    def test_returns_group_box(self):
        widget, _, _, _, qt_modules = _make()
        qt_core, qt_gui, qt_widgets = qt_modules
        assert isinstance(widget, qt_widgets.QGroupBox)

    def test_correct_object_name(self):
        widget, _, _, _, _ = _make()
        assert widget.objectName() == "fit-convergence-group"

    def test_hidden_by_default(self):
        widget, _, _, _, _ = _make()
        assert widget.isHidden()

    def test_canvas_present(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        widget, _, _, _, qt_modules = _make()
        qt_core, qt_gui, qt_widgets = qt_modules
        canvas = widget.findChild(qt_widgets.QWidget, "fit-convergence-canvas")
        assert canvas is not None

    def test_status_label_present(self):
        widget, _, _, _, qt_modules = _make()
        qt_core, qt_gui, qt_widgets = qt_modules
        label = widget.findChild(qt_widgets.QLabel, "fit-convergence-status")
        assert label is not None

    def test_status_label_empty_initially(self):
        widget, _, _, _, qt_modules = _make()
        qt_core, qt_gui, qt_widgets = qt_modules
        label = widget.findChild(qt_widgets.QLabel, "fit-convergence-status")
        assert label.text() == ""

    def test_add_ofv_point_updates_status(self):
        widget, add_ofv, _, _, qt_modules = _make()
        qt_core, qt_gui, qt_widgets = qt_modules
        add_ofv(1, 500.0)
        label = widget.findChild(qt_widgets.QLabel, "fit-convergence-status")
        assert "500" in label.text()
        assert "1" in label.text()

    def test_add_multiple_points(self):
        widget, add_ofv, _, _, qt_modules = _make()
        qt_core, qt_gui, qt_widgets = qt_modules
        for i in range(1, 6):
            add_ofv(i, 500.0 - i * 10)
        label = widget.findChild(qt_widgets.QLabel, "fit-convergence-status")
        assert "5" in label.text()
        assert "450" in label.text()

    def test_reset_clears_status(self):
        widget, add_ofv, reset, _, qt_modules = _make()
        qt_core, qt_gui, qt_widgets = qt_modules
        add_ofv(1, 500.0)
        reset()
        label = widget.findChild(qt_widgets.QLabel, "fit-convergence-status")
        assert label.text() == ""

    def test_finalize_updates_status(self):
        widget, add_ofv, _, finalize, qt_modules = _make()
        qt_core, qt_gui, qt_widgets = qt_modules
        finalize([500.0, 480.0, 460.0, 440.0])
        label = widget.findChild(qt_widgets.QLabel, "fit-convergence-status")
        assert "440" in label.text()

    def test_finalize_empty_history(self):
        widget, _, _, finalize, _ = _make()
        # Should not raise
        finalize([])

    def test_add_ofv_large_value_suppressed_from_status(self):
        widget, add_ofv, _, _, qt_modules = _make()
        qt_core, qt_gui, qt_widgets = qt_modules
        # OFV >= 1e6 is a penalty value and shouldn't be shown
        add_ofv(1, 1e8)
        label = widget.findChild(qt_widgets.QLabel, "fit-convergence-status")
        assert label.text() == ""

    def test_reset_after_finalize(self):
        widget, _, reset, finalize, qt_modules = _make()
        qt_core, qt_gui, qt_widgets = qt_modules
        finalize([400.0, 380.0])
        reset()
        label = widget.findChild(qt_widgets.QLabel, "fit-convergence-status")
        assert label.text() == ""

    def test_multiple_reset_cycles(self):
        widget, add_ofv, reset, _, qt_modules = _make()
        for _ in range(3):
            for i in range(5):
                add_ofv(i + 1, 500.0 - i * 5)
            reset()
        # After all resets, status should be clear
        qt_core, qt_gui, qt_widgets = qt_modules
        label = widget.findChild(qt_widgets.QLabel, "fit-convergence-status")
        assert label.text() == ""


# ---------------------------------------------------------------------------
# Estimation method iteration_callback tests (pure, no Qt)
# ---------------------------------------------------------------------------


class TestIterationCallbackSAEM:
    def test_saem_accepts_iteration_callback(self):
        from openpkpd.estimation.saem import SAEMMethod

        calls: list[tuple[int, float]] = []
        est = SAEMMethod(iteration_callback=lambda i, v: calls.append((i, v)))
        assert est.iteration_callback is not None

    def test_saem_default_callback_is_none(self):
        from openpkpd.estimation.saem import SAEMMethod

        est = SAEMMethod()
        assert est.iteration_callback is None


class TestIterationCallbackFOCE:
    def test_foce_accepts_iteration_callback(self):
        from openpkpd.estimation.foce import FOCEMethod

        calls: list[tuple[int, float]] = []
        est = FOCEMethod(iteration_callback=lambda i, v: calls.append((i, v)))
        assert est.iteration_callback is not None

    def test_foce_default_callback_is_none(self):
        from openpkpd.estimation.foce import FOCEMethod

        est = FOCEMethod()
        assert est.iteration_callback is None


class TestIterationCallbackIMP:
    def test_imp_accepts_iteration_callback(self):
        from openpkpd.estimation.imp import IMPMethod

        calls: list[tuple[int, float]] = []
        est = IMPMethod(iteration_callback=lambda i, v: calls.append((i, v)))
        assert est.iteration_callback is not None

    def test_imp_default_callback_is_none(self):
        from openpkpd.estimation.imp import IMPMethod

        est = IMPMethod()
        assert est.iteration_callback is None


class TestGetEstimationMethodCallback:
    def test_foce_passes_callback_through(self):
        from openpkpd.estimation import get_estimation_method
        from openpkpd.estimation.foce import FOCEMethod

        calls: list = []
        est = get_estimation_method("FOCE", iteration_callback=lambda i, v: calls.append((i, v)))
        assert isinstance(est, FOCEMethod)
        assert est.iteration_callback is not None

    def test_saem_passes_callback_through(self):
        from openpkpd.estimation import get_estimation_method
        from openpkpd.estimation.saem import SAEMMethod

        calls: list = []
        est = get_estimation_method("SAEM", iteration_callback=lambda i, v: calls.append((i, v)))
        assert isinstance(est, SAEMMethod)
        assert est.iteration_callback is not None

    def test_imp_passes_callback_through(self):
        from openpkpd.estimation import get_estimation_method
        from openpkpd.estimation.imp import IMPMethod

        calls: list = []
        est = get_estimation_method("IMP", iteration_callback=lambda i, v: calls.append((i, v)))
        assert isinstance(est, IMPMethod)
        assert est.iteration_callback is not None

    def test_fo_ignores_callback(self):
        from openpkpd.estimation import get_estimation_method
        from openpkpd.estimation.fo import FOMethod

        # Should not raise; FO pops iteration_callback
        est = get_estimation_method("FO", iteration_callback=lambda i, v: None)
        assert isinstance(est, FOMethod)

    def test_laplacian_ignores_callback(self):
        from openpkpd.estimation import get_estimation_method
        from openpkpd.estimation.laplacian import LaplacianMethod

        est = get_estimation_method("LAPLACIAN", iteration_callback=lambda i, v: None)
        assert isinstance(est, LaplacianMethod)

    def test_bayes_ignores_callback(self):
        from openpkpd.estimation import get_estimation_method
        from openpkpd.estimation.bayes import BAYESMethod

        est = get_estimation_method("BAYES", iteration_callback=lambda i, v: None)
        assert isinstance(est, BAYESMethod)


# ---------------------------------------------------------------------------
# Integration: fit workflow contains convergence widget
# ---------------------------------------------------------------------------


class TestFitWorkflowConvergenceIntegration:
    def test_convergence_widget_in_fit_workflow(self):
        from openpkpd_gui.app.runtime import load_qt_modules
        from openpkpd_gui.workflows.fit_workflow import build_fit_workflow
        from openpkpd_gui.domain.workspace import Workspace

        qt_core, qt_gui, qt_widgets = load_qt_modules()
        app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])

        ws = Workspace()
        widget = build_fit_workflow(ws)
        group = widget.findChild(qt_widgets.QGroupBox, "fit-convergence-group")
        assert group is not None, "Convergence group box not found in fit workflow"

    def test_convergence_widget_hidden_on_fresh_workflow(self):
        from openpkpd_gui.app.runtime import load_qt_modules
        from openpkpd_gui.workflows.fit_workflow import build_fit_workflow
        from openpkpd_gui.domain.workspace import Workspace

        qt_core, qt_gui, qt_widgets = load_qt_modules()
        app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])

        ws = Workspace()
        widget = build_fit_workflow(ws)
        group = widget.findChild(qt_widgets.QGroupBox, "fit-convergence-group")
        assert group is not None
        assert group.isHidden(), "Convergence panel should be hidden when no fit is running"
