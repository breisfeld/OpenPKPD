"""Tests for column_mapping widget — P1-B data column mapping wizard."""

from __future__ import annotations

import pytest

from openpkpd_gui.widgets.column_mapping import (
    auto_detect_mapping,
    needs_mapping,
    build_column_mapping_widget,
    REQUIRED_NONMEM,
    ALL_NONMEM,
)


# ---------------------------------------------------------------------------
# needs_mapping — pure function, no Qt
# ---------------------------------------------------------------------------


class TestNeedsMapping:
    def test_standard_columns_need_no_mapping(self):
        assert not needs_mapping(["ID", "TIME", "DV"])

    def test_standard_plus_extras_need_no_mapping(self):
        assert not needs_mapping(["ID", "TIME", "DV", "AMT", "EVID"])

    def test_case_insensitive_match_needs_no_mapping(self):
        assert not needs_mapping(["id", "time", "dv"])

    def test_missing_dv_needs_mapping(self):
        assert needs_mapping(["ID", "TIME", "CONC"])

    def test_missing_id_needs_mapping(self):
        assert needs_mapping(["SUBJ", "TIME", "DV"])

    def test_completely_different_columns_need_mapping(self):
        assert needs_mapping(["SUBJECT", "HOURS", "CONCENTRATION", "DOSE"])

    def test_empty_columns_need_mapping(self):
        assert needs_mapping([])


# ---------------------------------------------------------------------------
# auto_detect_mapping — pure function, no Qt
# ---------------------------------------------------------------------------


class TestAutoDetectMapping:
    def test_exact_match_returned_unchanged(self):
        result = auto_detect_mapping(["ID", "TIME", "DV"])
        assert result == ["ID", "TIME", "DV"]

    def test_case_insensitive_exact_match(self):
        result = auto_detect_mapping(["id", "time", "dv"])
        assert result == ["ID", "TIME", "DV"]

    def test_subject_alias_maps_to_id(self):
        result = auto_detect_mapping(["SUBJECT", "TIME", "DV"])
        assert result[0] == "ID"

    def test_subj_alias_maps_to_id(self):
        result = auto_detect_mapping(["SUBJ", "TIME", "DV"])
        assert result[0] == "ID"

    def test_conc_alias_maps_to_dv(self):
        result = auto_detect_mapping(["ID", "TIME", "CONC"])
        assert result[2] == "DV"

    def test_concentration_alias_maps_to_dv(self):
        result = auto_detect_mapping(["ID", "TIME", "CONCENTRATION"])
        assert result[2] == "DV"

    def test_dose_alias_maps_to_amt(self):
        result = auto_detect_mapping(["ID", "TIME", "DV", "DOSE"])
        assert result[3] == "AMT"

    def test_unknown_column_kept_as_is(self):
        result = auto_detect_mapping(["ID", "TIME", "DV", "WEIGHT"])
        assert result[3] == "WEIGHT"

    def test_no_duplicate_assignments(self):
        # Even if two columns look like ID, only one gets mapped
        result = auto_detect_mapping(["SUBJ", "SUBJECT", "TIME", "DV"])
        id_count = result.count("ID")
        assert id_count <= 1, f"ID assigned twice: {result}"

    def test_result_length_matches_input(self):
        cols = ["A", "B", "C", "D", "E"]
        result = auto_detect_mapping(cols)
        assert len(result) == len(cols)

    def test_empty_input_returns_empty(self):
        assert auto_detect_mapping([]) == []

    def test_typical_non_standard_dataset(self):
        # Underscore-separated names like PLASMA_CONC and DOSE_MG should map
        # correctly after normalisation strips non-alphanumeric separators.
        cols = ["PATIENT", "TIME_H", "PLASMA_CONC", "DOSE_MG", "EVENT"]
        result = auto_detect_mapping(cols)
        assert "ID" in result, f"ID not detected in {list(zip(cols, result))}"
        assert "TIME" in result, f"TIME not detected in {list(zip(cols, result))}"
        assert "DV" in result, f"DV not detected in {list(zip(cols, result))}"
        assert "AMT" in result, f"AMT not detected in {list(zip(cols, result))}"

    def test_evid_alias(self):
        result = auto_detect_mapping(["ID", "TIME", "DV", "EVID"])
        assert result[3] == "EVID"

    def test_mdv_alias(self):
        result = auto_detect_mapping(["ID", "TIME", "DV", "MDV"])
        assert result[3] == "MDV"


# ---------------------------------------------------------------------------
# Qt widget tests
# ---------------------------------------------------------------------------


def _get_app():
    from openpkpd_gui.app.runtime import load_qt_modules
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    return (qt_core, qt_gui, qt_widgets), (
        qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    )


class TestBuildColumnMappingWidget:
    def _make(self, columns):
        qt_modules, _app = _get_app()
        widget, get_cols, refresh = build_column_mapping_widget(columns, qt_modules)
        return widget, get_cols, refresh, qt_modules

    def test_widget_is_group_box(self):
        widget, _, _, qt_modules = self._make(["ID", "TIME", "DV"])
        qt_core, qt_gui, qt_widgets = qt_modules
        assert isinstance(widget, qt_widgets.QGroupBox)

    def test_widget_hidden_by_default(self):
        widget, _, _ = self._make(["ID", "TIME", "DV"])[0:3]
        assert not widget.isVisible()

    def test_widget_has_correct_object_name(self):
        widget, _, _, _ = self._make(["ID", "TIME", "DV"])
        assert widget.objectName() == "data-column-mapping-group"

    def test_get_input_columns_exact_match(self):
        widget, get_cols, _, _ = self._make(["ID", "TIME", "DV"])
        result = get_cols()
        assert result == ["ID", "TIME", "DV"]

    def test_get_input_columns_auto_detected(self):
        widget, get_cols, _, _ = self._make(["SUBJECT", "HOURS", "CONC"])
        result = get_cols()
        assert result[0] == "ID"
        assert result[1] == "TIME"
        assert result[2] == "DV"

    def test_get_input_columns_unknown_kept_as_is(self):
        widget, get_cols, _, _ = self._make(["ID", "TIME", "DV", "WEIGHT"])
        result = get_cols()
        assert result[3] == "WEIGHT"

    def test_refresh_updates_columns(self):
        widget, get_cols, refresh, _ = self._make(["ID", "TIME", "DV"])
        refresh(["SUBJECT", "HOURS", "CONC"])
        result = get_cols()
        assert result[0] == "ID"
        assert result[1] == "TIME"
        assert result[2] == "DV"

    def test_apply_button_accessible_on_widget(self):
        widget, _, _, _ = self._make(["ID", "TIME", "DV"])
        assert hasattr(widget, "_apply_btn")

    def test_auto_detect_button_present(self):
        widget, _, _, qt_modules = self._make(["SUBJ", "TIME_H", "CONC"])
        qt_core, qt_gui, qt_widgets = qt_modules
        btn = widget.findChild(qt_widgets.QPushButton, "data-column-mapping-auto-btn")
        assert btn is not None

    def test_mapping_table_present(self):
        widget, _, _, qt_modules = self._make(["ID", "TIME", "DV"])
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "data-column-mapping-table")
        assert table is not None

    def test_table_has_correct_row_count(self):
        widget, _, _, qt_modules = self._make(["ID", "TIME", "DV", "AMT"])
        qt_core, qt_gui, qt_widgets = qt_modules
        table = widget.findChild(qt_widgets.QTableWidget, "data-column-mapping-table")
        assert table.rowCount() == 4

    def test_result_length_matches_input(self):
        cols = ["A", "B", "C", "D", "E"]
        widget, get_cols, _, _ = self._make(cols)
        assert len(get_cols()) == len(cols)


# ---------------------------------------------------------------------------
# DatasetService.peek_csv_columns tests
# ---------------------------------------------------------------------------


class TestPeekCsvColumns:
    def test_returns_correct_columns(self, tmp_path):
        from openpkpd_gui.services.data_service import DatasetService

        csv = tmp_path / "data.csv"
        csv.write_text("ID,TIME,DV,AMT\n1,0,5.0,100\n")
        svc = DatasetService()
        cols = svc.peek_csv_columns(str(csv))
        assert cols == ["ID", "TIME", "DV", "AMT"]

    def test_returns_empty_for_missing_file(self):
        from openpkpd_gui.services.data_service import DatasetService

        svc = DatasetService()
        cols = svc.peek_csv_columns("/nonexistent/path/data.csv")
        assert cols == []

    def test_returns_empty_for_empty_string(self):
        from openpkpd_gui.services.data_service import DatasetService

        svc = DatasetService()
        cols = svc.peek_csv_columns("")
        assert cols == []

    def test_respects_separator_option(self, tmp_path):
        from openpkpd_gui.services.data_service import DatasetService, DatasetImportOptions

        csv = tmp_path / "data.tsv"
        csv.write_text("ID\tTIME\tDV\n1\t0\t5.0\n")
        svc = DatasetService()
        opts = DatasetImportOptions(separator="\t")
        cols = svc.peek_csv_columns(str(csv), options=opts)
        assert cols == ["ID", "TIME", "DV"]

    def test_does_not_load_full_file(self, tmp_path):
        """peek should be fast — only reads the header, not all rows."""
        from openpkpd_gui.services.data_service import DatasetService
        import time

        # Write a large file
        csv = tmp_path / "large.csv"
        rows = ["ID,TIME,DV"] + [f"{i},{i*0.1:.1f},{i*2:.1f}" for i in range(100_000)]
        csv.write_text("\n".join(rows))
        svc = DatasetService()
        start = time.monotonic()
        cols = svc.peek_csv_columns(str(csv))
        elapsed = time.monotonic() - start
        assert cols == ["ID", "TIME", "DV"]
        assert elapsed < 1.0, f"peek_csv_columns took {elapsed:.2f}s — likely reading full file"


# ---------------------------------------------------------------------------
# Integration: mapping widget present in built data workflow
# ---------------------------------------------------------------------------


class TestDataWorkflowMappingIntegration:
    def test_mapping_widget_in_data_workflow(self):
        from openpkpd_gui.app.runtime import load_qt_modules
        from openpkpd_gui.workflows.data_workflow import build_data_workflow
        from openpkpd_gui.domain.workspace import Workspace

        qt_core, qt_gui, qt_widgets = load_qt_modules()
        app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])

        ws = Workspace()
        widget = build_data_workflow(ws)
        mapping_group = widget.findChild(qt_widgets.QGroupBox, "data-column-mapping-group")
        assert mapping_group is not None, "Column mapping group box not found in data workflow"

    def test_mapping_widget_hidden_on_fresh_workflow(self):
        from openpkpd_gui.app.runtime import load_qt_modules
        from openpkpd_gui.workflows.data_workflow import build_data_workflow
        from openpkpd_gui.domain.workspace import Workspace

        qt_core, qt_gui, qt_widgets = load_qt_modules()
        app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])

        ws = Workspace()
        widget = build_data_workflow(ws)
        mapping_group = widget.findChild(qt_widgets.QGroupBox, "data-column-mapping-group")
        assert mapping_group is not None
        assert not mapping_group.isVisible(), (
            "Mapping panel should be hidden when no dataset is loaded"
        )
