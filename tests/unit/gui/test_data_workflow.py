"""Pure tests for the Data workflow presentation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.data_service import DatasetService, ExampleDataset
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.widgets.link_formatting import compact_path_label, compact_url_label, copy_link
from openpkpd_gui.widgets.table_headers import configure_resizable_table_columns
from openpkpd_gui.workflows.data_workflow import (
    build_data_workflow,
    filter_example_datasets,
    format_data_draft_status,
    format_dataset_import_summary,
    format_dataset_summary,
    format_example_dataset_contents,
    format_example_dataset_details,
    format_example_dataset_details_html,
    format_example_dataset_hint,
    format_example_dataset_option,
    recommend_data_next_action,
)


def test_format_dataset_summary_handles_empty_state() -> None:
    assert "No dataset loaded yet" in format_dataset_summary(None)


def test_format_dataset_summary_includes_key_counts() -> None:
    dataset_asset = DatasetAsset(
        display_name="theo.csv",
        row_count=132,
        subject_count=12,
        observation_count=120,
    )

    summary = format_dataset_summary(dataset_asset)

    assert "theo.csv" in summary
    assert "132 rows" in summary
    assert "12 subjects" in summary
    assert "120 observations" in summary


def test_format_dataset_import_summary_reports_options() -> None:
    dataset_asset = DatasetAsset(separator=";", ignore_char="@")

    summary = format_dataset_import_summary(dataset_asset)

    assert "';'" in summary
    assert "@" in summary


def test_format_dataset_import_summary_handles_whitespace_mode() -> None:
    dataset_asset = DatasetAsset(treat_as_whitespace=True)

    summary = format_dataset_import_summary(dataset_asset)

    assert "whitespace" in summary


def test_format_data_draft_status_reports_unsaved_state() -> None:
    assert format_data_draft_status(True) == "Unsaved data import changes"
    assert format_data_draft_status(False) == ""


def test_recommend_data_next_action_tracks_handoff_states() -> None:
    workspace = Workspace(name="Data CTA")

    assert recommend_data_next_action(workspace, has_unsaved_changes=False) is None

    workspace.active_dataset = DatasetAsset(source_path="/tmp/theo.csv", display_name="theo.csv")
    assert recommend_data_next_action(workspace, has_unsaved_changes=True) is None
    assert recommend_data_next_action(workspace, has_unsaved_changes=False) == (
        "Open Model",
        "model",
        "A dataset is ready; open Model to configure one next.",
    )

    workspace.active_model_spec = ModelSpec(dataset_path="/tmp/other.csv")
    assert recommend_data_next_action(workspace, has_unsaved_changes=False) == (
        "Open Model",
        "model",
        "The active dataset changed — open Model to update the dataset path before fitting.",
    )

    workspace.active_model_spec = ModelSpec(dataset_path="/tmp/theo.csv")
    assert recommend_data_next_action(workspace, has_unsaved_changes=False) == (
        "Open Fit",
        "fit",
        "Dataset and saved model are ready for estimation.",
    )

    workspace.add_run(RunRecord(workflow="fit", status=RunStatus.SUCCEEDED))
    assert recommend_data_next_action(workspace, has_unsaved_changes=False) == (
        "Open Results",
        "results",
        "A successful fit is already available. Review the latest outputs in Results.",
    )


@pytest.mark.unit
def test_data_workflow_tracks_dirty_and_clean_transitions(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Data workflow")
    widget = build_data_workflow(project)
    dirty_transitions: list[bool] = []
    widget._project_state_changed = lambda: dirty_transitions.append(widget._has_unsaved_changes())  # type: ignore[attr-defined]

    try:
        widget.show()
        app.processEvents()

        path_input = widget.findChild(qt_widgets.QLineEdit, "data-source-path")
        unsaved_label = widget.findChild(qt_widgets.QLabel, "data-unsaved-label")

        assert path_input is not None
        assert unsaved_label is not None

        path_input.setText(str(dataset_path))
        app.processEvents()

        assert widget._has_unsaved_changes() is True  # type: ignore[attr-defined]
        assert unsaved_label.text() == "Unsaved data import changes"
        assert project.active_dataset is None

        path_input.editingFinished.emit()
        app.processEvents()

        assert widget._has_unsaved_changes() is False  # type: ignore[attr-defined]
        assert unsaved_label.text() == ""
        assert project.active_dataset is not None
        assert project.active_dataset.source_path == str(dataset_path)
        assert dirty_transitions == [True, False]
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_data_preview_table_columns_are_user_resizable() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    widget = build_data_workflow(Workspace(name="Data workflow"))

    try:
        widget.show()
        app.processEvents()

        preview_table = widget.findChild(qt_widgets.QTableWidget, "data-preview-table")

        assert preview_table is not None
        preview_table.setColumnCount(1)
        preview_table.setHorizontalHeaderLabels(["ID"])
        configure_resizable_table_columns(preview_table, qt_widgets)
        assert (
            preview_table.horizontalHeader().sectionResizeMode(0)
            == qt_widgets.QHeaderView.ResizeMode.Interactive
        )
        assert preview_table.horizontalHeader().stretchLastSection() is False
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_data_workflow_panels_are_user_adjustable() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    qt_core, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    widget = build_data_workflow(Workspace(name="Data workflow"))

    try:
        widget.resize(1400, 900)
        widget.show()
        app.processEvents()

        splitter = widget.findChild(qt_widgets.QSplitter, "data-content-row")

        assert splitter is not None
        assert splitter.count() == 3
        assert splitter.orientation() == qt_core.Qt.Orientation.Horizontal

        splitter.setSizes([120, 620, 260])
        app.processEvents()

        assert len(splitter.sizes()) == 3

        widget.resize(700, 900)
        app.processEvents()

        assert splitter.orientation() == qt_core.Qt.Orientation.Vertical
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_data_workflow_next_action_transitions_from_model_to_fit_to_results(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Data workflow")
    project_service = ProjectService()
    widget = build_data_workflow(project, project_service=project_service)
    navigations: list[str] = []
    widget._navigate_to_workflow = lambda workflow_id: navigations.append(workflow_id)  # type: ignore[attr-defined]

    try:
        widget.show()
        app.processEvents()

        next_action_label = widget.findChild(qt_widgets.QLabel, "data-next-action-label")
        next_action_button = widget.findChild(qt_widgets.QPushButton, "data-next-action-button")

        assert next_action_label is not None
        assert next_action_button is not None
        assert next_action_button.isHidden() is True

        project_service.attach_dataset(
            project,
            DatasetAsset(
                source_path=str(dataset_path), display_name="theo.csv", columns=["ID", "TIME", "DV"]
            ),
        )
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_button.text() == "Open Model"
        next_action_button.click()
        app.processEvents()

        project_service.set_model_spec(
            project,
            ModelSpec(
                problem_title="One compartment",
                dataset_path=str(dataset_path),
                pk_code="CL = THETA(1)",
                error_code="Y = F",
                theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
                omega_values=[[0.3]],
                sigma_values=[[0.1]],
            ),
        )
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_button.text() == "Open Fit"
        assert "ready for estimation" in next_action_label.text()
        next_action_button.click()
        app.processEvents()

        project_service.add_run(project, RunRecord(workflow="fit", status=RunStatus.SUCCEEDED))
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_button.text() == "Open Results"
        next_action_button.click()
        app.processEvents()

        assert navigations == ["model", "fit", "results"]
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def _write_catalog_nmdata_example(tmp_path: Path) -> tuple[Path, Path]:
    catalog_root = tmp_path / "catalog"
    shared_data_root = tmp_path / "shared_data"
    bundle_dir = catalog_root / "pk" / "oral" / "nmdata_drop"
    dataset_path = shared_data_root / "nmdata" / "drop.csv"
    bundle_dir.mkdir(parents=True)
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("1,0,999,70,5\n", encoding="utf-8")
    (bundle_dir / "model.ctl").write_text(
        "$PROBLEM Drop demo\n"
        "$DATA ../../../../shared_data/nmdata/drop.csv\n"
        "$INPUT ID TIME SKIP=DROP WT DV\n"
        "$THETA 1\n"
        "$OMEGA 0.1\n"
        "$SIGMA 0.1\n"
        "$PK\nCL=THETA(1)\n"
        "$ERROR\nY=F+EPS(1)\n",
        encoding="utf-8",
    )
    (bundle_dir / "manifest.yaml").write_text(
        "\n".join(
            [
                "manifest_version: 1",
                "id: nmdata_drop",
                "title: NMdata drop example",
                "description: Headerless dataset with INPUT DROP metadata.",
                "category: pk",
                "primary_mode: control_stream",
                "route: oral",
                "difficulty: advanced",
                "tags: [nmdata, drop]",
                "files:",
                "  dataset: ../../../../shared_data/nmdata/drop.csv",
                "  control_stream: model.ctl",
                "gui:",
                "  load_dataset: true",
                "  load_control_stream: true",
                "source:",
                "  kind: imported",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return catalog_root, shared_data_root


@pytest.mark.unit
def test_data_workflow_load_example_populates_preview_for_nmdata_drop(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    catalog_root, shared_data_root = _write_catalog_nmdata_example(tmp_path)
    dataset_service = DatasetService(catalog_root=catalog_root, shared_data_root=shared_data_root)

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Data workflow")
    widget = build_data_workflow(project, dataset_service=dataset_service)

    try:
        widget.show()
        app.processEvents()

        selector = widget.findChild(qt_widgets.QComboBox, "data-example-selector")
        button = widget.findChild(qt_widgets.QPushButton, "data-load-example-button")
        preview_table = widget.findChild(qt_widgets.QTableWidget, "data-preview-table")

        assert selector is not None
        assert button is not None
        assert preview_table is not None

        selector.setCurrentIndex(1)
        app.processEvents()
        button.click()
        app.processEvents()

        assert project.active_dataset is not None
        assert project.active_dataset.columns[:4] == ["ID", "TIME", "WT", "DV"]
        assert preview_table.columnCount() >= 4
        assert preview_table.rowCount() == 1
        assert preview_table.horizontalHeaderItem(2).text() == "WT"
        assert preview_table.item(0, 2).text() in {"70", "70.0"}
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def _catalog_example(**overrides: object) -> ExampleDataset:
    values: dict[str, object] = {
        "key": "demo_dataset",
        "label": "Demo dataset",
        "description": "Demo dataset.\n\nHelpful details.",
        "dataset_path": "/tmp/shared_data/demo/demo.csv",
        "category": "pk",
        "route": "oral",
        "difficulty": "starter",
        "tags": ("demo", "oral"),
        "manifest_path": "/tmp/examples/catalog/pk/oral/demo/manifest.yaml",
        "source_kind": "catalog_manifest",
        "readme_path": None,
        "source_license": None,
        "source_url": None,
    }
    values.update(overrides)
    return ExampleDataset(**values)


def test_format_example_dataset_option_uses_example_label() -> None:
    assert format_example_dataset_option(_catalog_example()) == "Demo dataset"


def test_filter_example_datasets_matches_catalog_metadata() -> None:
    examples = [
        _catalog_example(),
        _catalog_example(
            key="nmdata_demo",
            label="NMdata body-weight example",
            description="Imported NMdata example.",
            dataset_path="/tmp/shared_data/nmautoverse/xgxr1.csv",
            manifest_path="/tmp/examples/catalog/pk/iv/nmdata_demo/manifest.yaml",
            route="iv",
            difficulty="advanced",
            tags=("nmdata", "covariate"),
            readme_path="/tmp/examples/catalog/pk/iv/nmdata_demo/README.md",
            source_kind="imported",
            source_license="MIT + file LICENSE",
            source_url="https://github.com/NMautoverse/NMdata",
        ),
    ]

    assert [example.key for example in filter_example_datasets(examples, "nmdata mit")] == [
        "nmdata_demo"
    ]
    assert [example.key for example in filter_example_datasets(examples, "starter oral")] == [
        "demo_dataset"
    ]


def test_format_example_dataset_hint_reports_availability() -> None:
    example = _catalog_example()

    assert "Curated examples" in format_example_dataset_hint([example])
    assert "No curated example datasets" in format_example_dataset_hint([])
    assert 'Showing 1 of 1 curated example dataset for "demo"' in format_example_dataset_hint(
        [example],
        visible_count=1,
        filter_text="demo",
    )


def test_format_example_dataset_contents_reports_catalog_metadata() -> None:
    contents = format_example_dataset_contents(_catalog_example())

    assert "Dataset: demo.csv" in contents
    assert "Category: pk" in contents
    assert "Route: oral" in contents
    assert "Difficulty: starter" in contents


def test_format_example_dataset_details_reports_selection_prompt() -> None:
    assert "Select a curated example dataset" in format_example_dataset_details(None)


def test_format_example_dataset_details_includes_provenance_metadata() -> None:
    example = _catalog_example(
        readme_path="/tmp/examples/catalog/pk/oral/demo/README.md",
        source_kind="imported",
        source_license="MIT + file LICENSE",
        source_url="https://github.com/NMautoverse/NMdata",
        tags=("demo", "nmdata"),
    )

    details = format_example_dataset_details(example)

    assert "Demo dataset" in details
    assert "Helpful details." in details
    assert "Source file: /tmp/shared_data/demo/demo.csv" in details
    assert "Provenance: kind: imported" in details
    assert "license: MIT + file LICENSE" in details
    assert "url: https://github.com/NMautoverse/NMdata" in details
    assert "Bundle notes: /tmp/examples/catalog/pk/oral/demo/README.md" in details
    assert "Tags: demo, nmdata" in details


def test_format_example_dataset_details_html_includes_clickable_actions() -> None:
    example = _catalog_example(
        readme_path="/tmp/examples/catalog/pk/oral/demo/README.md",
        source_kind="imported",
        source_license="MIT + file LICENSE",
        source_url="https://github.com/NMautoverse/NMdata",
    )

    details = format_example_dataset_details_html(example)

    assert "<b>Demo dataset</b>" in details
    assert 'href="file:///tmp/shared_data/demo/demo.csv"' in details
    assert 'href="file:///tmp/examples/catalog/pk/oral/demo"' in details
    assert 'href="file:///tmp/examples/catalog/pk/oral/demo/README.md"' in details
    assert 'href="https://github.com/NMautoverse/NMdata"' in details
    assert copy_link("/tmp/shared_data/demo/demo.csv", label="Copy path") in details
    assert copy_link("https://github.com/NMautoverse/NMdata", label="Copy URL") in details
    assert compact_path_label("/tmp/shared_data/demo/demo.csv") in details
    assert compact_path_label("/tmp/examples/catalog/pk/oral/demo/README.md") in details
    assert compact_url_label("https://github.com/NMautoverse/NMdata") in details
    assert "Open dataset" in details
    assert "Open bundle folder" in details
    assert "Open bundle notes" in details
    assert "Open upstream source" in details
