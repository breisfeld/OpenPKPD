"""Tests for scenario_comparison widget (P1-D)."""

from __future__ import annotations

import pytest

from openpkpd_gui.widgets.scenario_comparison import (
    _parse_summary,
    build_comparison_rows,
    build_scenario_comparison_widget,
)


# ---------------------------------------------------------------------------
# _parse_summary — pure function, no Qt
# ---------------------------------------------------------------------------


class TestParseSummary:
    def test_all_fields_extracted(self):
        text = "1-cmt IV • FOCE • converged=True • OFV=1234.5678"
        result = _parse_summary(text)
        assert result["ofv"] == "1234.5678"
        assert result["converged"] == "Yes"
        assert result["method"] == "FOCE"

    def test_not_converged(self):
        text = "2-cmt oral • SAEM • converged=False • OFV=987.1"
        result = _parse_summary(text)
        assert result["ofv"] == "987.1"
        assert result["converged"] == "No"
        assert result["method"] == "SAEM"

    def test_negative_ofv(self):
        text = "Model • IMP • converged=True • OFV=-42.3"
        result = _parse_summary(text)
        assert result["ofv"] == "-42.3"

    def test_scientific_notation_ofv(self):
        text = "Model • FO • converged=True • OFV=1.23e+04"
        result = _parse_summary(text)
        assert result["ofv"] == "1.23e+04"

    def test_empty_string_returns_dashes(self):
        result = _parse_summary("")
        assert result["ofv"] == "—"
        assert result["converged"] == "—"
        assert result["method"] == "—"

    def test_missing_ofv_returns_dash(self):
        text = "No OFV info here"
        result = _parse_summary(text)
        assert result["ofv"] == "—"

    def test_missing_converged_returns_dash(self):
        text = "• FOCE • OFV=100.0"
        result = _parse_summary(text)
        assert result["converged"] == "—"


# ---------------------------------------------------------------------------
# build_comparison_rows — pure function using Workspace domain objects
# ---------------------------------------------------------------------------


def _make_workspace_with_scenarios(n_siblings: int = 2, with_successful_fit: bool = True):
    """Build a minimal Workspace with n_siblings peer scenarios."""
    from openpkpd_gui.domain.workspace import Workspace, Scenario
    from openpkpd_gui.domain.run_record import RunRecord, RunStatus

    ws = Workspace()
    # Rename current scenario for clarity
    ws.active_scenario.name = "Base"
    base_id = ws.active_scenario.scenario_id

    for i in range(n_siblings):
        scenario = Scenario(name=f"Alt{i + 1}")
        ws.active_project.add_scenario(scenario, make_active=False)
        if with_successful_fit:
            run = RunRecord(workflow="fit")
            run.mark_running()
            run.mark_succeeded(
                f"Alt{i + 1} • FOCE • converged=True • OFV={200.0 + i * 10:.4f}"
            )
            scenario.add_run(run)

    # Reset active scenario to "Base"
    ws.active_project.active_scenario_id = base_id
    return ws


class TestBuildComparisonRows:
    def test_no_siblings_returns_empty(self):
        from openpkpd_gui.domain.workspace import Workspace

        ws = Workspace()
        rows = build_comparison_rows(ws)
        assert rows == []

    def test_one_sibling_returns_one_row(self):
        ws = _make_workspace_with_scenarios(n_siblings=1)
        rows = build_comparison_rows(ws)
        assert len(rows) == 1

    def test_two_siblings_returns_two_rows(self):
        ws = _make_workspace_with_scenarios(n_siblings=2)
        rows = build_comparison_rows(ws)
        assert len(rows) == 2

    def test_row_keys_present(self):
        ws = _make_workspace_with_scenarios(n_siblings=1)
        row = build_comparison_rows(ws)[0]
        for key in ("scenario_id", "name", "relation", "ofv", "converged", "method", "fit_runs", "runs", "artifacts"):
            assert key in row, f"Missing key: {key}"

    def test_successful_fit_ofv_extracted(self):
        ws = _make_workspace_with_scenarios(n_siblings=1, with_successful_fit=True)
        row = build_comparison_rows(ws)[0]
        assert row["ofv"] != "—"
        float(row["ofv"])  # should be parseable

    def test_no_fit_run_ofv_is_dash(self):
        ws = _make_workspace_with_scenarios(n_siblings=1, with_successful_fit=False)
        row = build_comparison_rows(ws)[0]
        assert row["ofv"] == "—"

    def test_converged_yes_extracted(self):
        ws = _make_workspace_with_scenarios(n_siblings=1, with_successful_fit=True)
        row = build_comparison_rows(ws)[0]
        assert row["converged"] == "Yes"

    def test_rows_sorted_by_ofv_ascending(self):
        ws = _make_workspace_with_scenarios(n_siblings=3, with_successful_fit=True)
        rows = build_comparison_rows(ws)
        ofv_values = [float(row["ofv"]) for row in rows if row["ofv"] != "—"]
        assert ofv_values == sorted(ofv_values)

    def test_no_fit_rows_sorted_last(self):
        from openpkpd_gui.domain.workspace import Workspace, Scenario
        from openpkpd_gui.domain.run_record import RunRecord

        ws = Workspace()
        ws.active_scenario.name = "Base"
        base_id = ws.active_scenario.scenario_id

        fit_scenario = Scenario(name="WithFit")
        ws.active_project.add_scenario(fit_scenario, make_active=False)
        run = RunRecord(workflow="fit")
        run.mark_running()
        run.mark_succeeded("WithFit • FOCE • converged=True • OFV=100.0")
        fit_scenario.add_run(run)

        no_fit_scenario = Scenario(name="NoFit")
        ws.active_project.add_scenario(no_fit_scenario, make_active=False)

        ws.active_project.active_scenario_id = base_id

        rows = build_comparison_rows(ws)
        assert len(rows) == 2
        assert rows[0]["ofv"] != "—"  # WithFit first
        assert rows[1]["ofv"] == "—"   # NoFit last

    def test_relation_child_detected(self):
        from openpkpd_gui.domain.workspace import Workspace, Scenario

        ws = Workspace()
        current_id = ws.active_scenario.scenario_id
        child = Scenario(name="Child", parent_scenario_id=current_id)
        ws.active_project.add_scenario(child, make_active=False)
        ws.active_project.active_scenario_id = current_id

        rows = build_comparison_rows(ws)
        assert rows[0]["relation"] == "child"

    def test_relation_parent_detected(self):
        from openpkpd_gui.domain.workspace import Workspace, Scenario

        ws = Workspace()
        base_id = ws.active_scenario.scenario_id
        parent = Scenario(name="Parent")
        ws.active_project.add_scenario(parent, make_active=False)
        ws.active_scenario.parent_scenario_id = parent.scenario_id
        ws.active_project.active_scenario_id = base_id

        rows = build_comparison_rows(ws)
        assert rows[0]["relation"] == "parent"

    def test_current_scenario_excluded(self):
        from openpkpd_gui.domain.workspace import Workspace

        ws = Workspace()
        rows = build_comparison_rows(ws)
        current_id = ws.active_scenario.scenario_id
        assert all(row["scenario_id"] != current_id for row in rows)


# ---------------------------------------------------------------------------
# Qt widget tests
# ---------------------------------------------------------------------------


def _get_app():
    from openpkpd_gui.app.runtime import load_qt_modules

    qt_core, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    return (qt_core, qt_gui, qt_widgets), app


class TestBuildScenarioComparisonWidget:
    def _make(self, n_siblings: int = 2, with_fit: bool = True):
        ws = _make_workspace_with_scenarios(n_siblings, with_successful_fit=with_fit)
        qt_modules, _app = _get_app()
        widget, refresh = build_scenario_comparison_widget(ws, qt_modules)
        return widget, refresh, ws, qt_modules

    def test_returns_group_box(self):
        widget, _, _, qt_modules = self._make()
        qt_core, qt_gui, qt_widgets = qt_modules
        assert isinstance(widget, qt_widgets.QGroupBox)

    def test_correct_object_name(self):
        widget, _, _, _ = self._make()
        assert widget.objectName() == "results-scenario-comparison-group"

    def test_table_present(self):
        widget, _, _, qt_modules = self._make()
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "results-scenario-comparison-table")
        assert table is not None

    def test_table_row_count_matches_siblings(self):
        widget, _, _, qt_modules = self._make(n_siblings=3)
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "results-scenario-comparison-table")
        assert table.rowCount() == 3

    def test_table_not_hidden_when_siblings_exist(self):
        widget, _, _, qt_modules = self._make(n_siblings=1)
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "results-scenario-comparison-table")
        assert not table.isHidden()

    def test_empty_label_shown_when_no_siblings(self):
        from openpkpd_gui.domain.workspace import Workspace

        ws = Workspace()
        qt_modules, _app = _get_app()
        widget, refresh = build_scenario_comparison_widget(ws, qt_modules)
        qt_core, qt_gui, qt_widgets = qt_modules
        empty_label = widget.findChild(qt_widgets.QLabel, "results-scenario-comparison-empty")
        assert empty_label is not None
        assert not empty_label.isHidden()

    def test_table_hidden_when_no_siblings(self):
        from openpkpd_gui.domain.workspace import Workspace

        ws = Workspace()
        qt_modules, _app = _get_app()
        widget, refresh = build_scenario_comparison_widget(ws, qt_modules)
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "results-scenario-comparison-table")
        assert table.isHidden()

    def test_correct_column_count(self):
        widget, _, _, qt_modules = self._make()
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "results-scenario-comparison-table")
        assert table.columnCount() == 7  # Scenario, Relation, OFV, Converged, Method, Fit runs, Outputs

    def test_ofv_cell_value(self):
        widget, _, _, qt_modules = self._make(n_siblings=1, with_fit=True)
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "results-scenario-comparison-table")
        ofv_col = 2  # "OFV"
        ofv_text = table.item(0, ofv_col).text()
        assert ofv_text != "—"
        float(ofv_text)  # parseable

    def test_converged_cell_yes(self):
        widget, _, _, qt_modules = self._make(n_siblings=1, with_fit=True)
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "results-scenario-comparison-table")
        converged_col = 3  # "Converged"
        assert table.item(0, converged_col).text() == "Yes"

    def test_refresh_updates_table(self):
        from openpkpd_gui.domain.workspace import Workspace

        ws = Workspace()
        qt_modules, _app = _get_app()
        widget, refresh = build_scenario_comparison_widget(ws, qt_modules)
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "results-scenario-comparison-table")
        assert table.rowCount() == 0

        ws2 = _make_workspace_with_scenarios(n_siblings=2)
        refresh(ws2)
        assert table.rowCount() == 2

    def test_scenario_id_stored_in_user_role(self):
        widget, _, ws, qt_modules = self._make(n_siblings=1)
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "results-scenario-comparison-table")
        name_item = table.item(0, 0)
        scenario_id = name_item.data(qt_core.Qt.ItemDataRole.UserRole)
        assert isinstance(scenario_id, str)
        assert len(scenario_id) > 0

    def test_not_editable(self):
        widget, _, _, qt_modules = self._make(n_siblings=1)
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "results-scenario-comparison-table")
        item = table.item(0, 0)
        assert not (item.flags() & qt_core.Qt.ItemFlag.ItemIsEditable)
