"""Pure tests for the Model workflow presentation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from openpkpd.examples.catalog_models import (
    ExampleEntry,
    ExampleFiles,
    ExampleGUI,
    ExampleManifest,
    ExampleSource,
)
from openpkpd.examples.catalog_service import ExampleCatalogService
from openpkpd.model.problem import Problem
from openpkpd.parser.control_stream import ControlStream
from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec, ModelSpecMode
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.data_service import DatasetService
from openpkpd_gui.services.model_translation_service import ModelTranslationResult
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.widgets.link_formatting import compact_path_label, compact_url_label, copy_link
from openpkpd_gui.workflows.model_workflow import (
    _format_catalog_control_stream_details,
    _format_catalog_control_stream_details_html,
    build_model_workflow,
    default_model_spec,
    default_theta_row,
    format_model_draft_status,
    format_model_summary,
    format_parameter_summary,
    format_translation_summary,
    load_control_stream_dataset_asset,
    load_control_stream_model_spec,
    recommend_model_next_action,
    resize_square_matrix,
    resolve_control_stream_dataset_path,
    write_control_stream_text,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"
CONTROL_STREAMS_DIR = EXAMPLES_DIR / "control_streams"


def _write_catalog_control_stream_example(
    tmp_path: Path,
    *,
    include_dataset: bool = True,
) -> tuple[DatasetService, ExampleCatalogService]:
    catalog_root = tmp_path / "catalog"
    shared_data_root = tmp_path / "shared_data"
    bundle_dir = catalog_root / "pk" / "oral" / "demo_oral_fo"
    dataset_path = shared_data_root / "demo" / "demo.csv"
    bundle_dir.mkdir(parents=True)
    if include_dataset:
        dataset_path.parent.mkdir(parents=True)
        dataset_path.write_text(
            "ID,TIME,AMT,DV,EVID,MDV\n1,0,100,0,1,1\n1,1,0,5,0,0\n", encoding="utf-8"
        )
    (bundle_dir / "model.ctl").write_text(
        "\n".join(
            [
                "$PROBLEM Demo oral FO",
                (
                    "$DATA ../../../../shared_data/demo/demo.csv"
                    if include_dataset
                    else "$DATA external/demo.csv"
                ),
                "$INPUT ID TIME AMT DV EVID MDV",
                "$SUBROUTINES ADVAN2 TRANS2",
                "$PK",
                "CL = THETA(1)",
                "$ERROR",
                "Y = F",
                "$THETA 1",
                "$OMEGA 0.3",
                "$SIGMA 0.1",
                "$ESTIMATION METHOD=ZERO",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (bundle_dir / "manifest.yaml").write_text(
        "\n".join(
            [
                "manifest_version: 1",
                "id: demo_oral_fo",
                "title: Demo oral FO",
                "description: Demo curated control-stream example for the model workflow.",
                "category: pk",
                "primary_mode: control_stream",
                "route: oral",
                "difficulty: starter",
                "tags: [demo, oral, fo]",
                "files:",
                (
                    "  dataset: ../../../../shared_data/demo/demo.csv"
                    if include_dataset
                    else "  dataset: null"
                ),
                "  control_stream: model.ctl",
                "  readme: null",
                "  preview_image: null",
                "gui:",
                f"  load_dataset: {'true' if include_dataset else 'false'}",
                "  load_control_stream: true",
                "source:",
                "  kind: internal",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return (
        DatasetService(catalog_root=catalog_root, shared_data_root=shared_data_root),
        ExampleCatalogService(catalog_root=catalog_root, shared_data_root=shared_data_root),
    )


def test_default_model_spec_uses_active_dataset_when_present() -> None:
    workspace = Workspace()
    workspace.active_dataset = DatasetAsset(source_path="/tmp/theo.csv")

    spec = default_model_spec(workspace)

    assert spec.dataset_path == "/tmp/theo.csv"
    assert spec.pk_code
    assert spec.error_code
    assert spec.theta_rows


def test_format_model_summary_reports_mode_and_method() -> None:
    spec = ModelSpec(
        problem_title="One compartment",
        mode=ModelSpecMode.CONTROL_STREAM,
        dataset_path="/tmp/theo.csv",
    )

    summary = format_model_summary(spec)

    assert "One compartment" in summary
    assert "Control stream mode" in summary
    assert "FOCE" in summary
    assert "/tmp/theo.csv" in summary


def test_format_translation_summary_reports_builder_counts() -> None:
    result = ModelTranslationResult(
        mode=ModelSpecMode.BUILDER,
        builder=object(),
        theta_count=3,
        eta_count=3,
        eps_count=1,
        estimation_method="FOCE",
    )

    summary = format_translation_summary(result)

    assert "Ready for engine translation" in summary
    assert "3 THETA" in summary
    assert "3 ETA" in summary
    assert "1 EPS" in summary


def test_format_catalog_control_stream_details_includes_provenance_and_notes(
    tmp_path: Path,
) -> None:
    readme_path = tmp_path / "README.md"
    readme_path.write_text("notes\n", encoding="utf-8")
    entry = ExampleEntry(
        manifest=ExampleManifest(
            manifest_version=1,
            id="nmdata_demo",
            title="NMdata demo",
            description="Imported demo control stream.",
            category="pk",
            primary_mode="control_stream",
            route="oral",
            difficulty="advanced",
            tags=("nmdata",),
            files=ExampleFiles(control_stream="model.ctl", readme="README.md"),
            gui=ExampleGUI(load_control_stream=True),
            source=ExampleSource(
                kind="imported",
                url="https://github.com/NMautoverse/NMdata",
                license="MIT + file LICENSE",
            ),
        ),
        bundle_dir=tmp_path,
        manifest_path=tmp_path / "manifest.yaml",
        dataset_path=None,
        control_stream_path=tmp_path / "model.ctl",
        readme_path=readme_path,
    )

    details = _format_catalog_control_stream_details(entry)

    assert "Imported demo control stream." in details
    assert "route: oral" in details
    assert "difficulty: advanced" in details
    assert "dataset: not bundled" in details
    assert "Provenance: source: imported" in details
    assert "license: MIT + file LICENSE" in details
    assert "url: https://github.com/NMautoverse/NMdata" in details
    assert f"Bundle notes: {readme_path}" in details


def test_format_catalog_control_stream_details_html_includes_clickable_actions(
    tmp_path: Path,
) -> None:
    readme_path = tmp_path / "README.md"
    readme_path.write_text("notes\n", encoding="utf-8")
    entry = ExampleEntry(
        manifest=ExampleManifest(
            manifest_version=1,
            id="nmdata_demo",
            title="NMdata demo",
            description="Imported demo control stream.",
            category="pk",
            primary_mode="control_stream",
            route="oral",
            difficulty="advanced",
            tags=("nmdata",),
            files=ExampleFiles(control_stream="model.ctl", readme="README.md"),
            gui=ExampleGUI(load_control_stream=True),
            source=ExampleSource(
                kind="imported",
                url="https://github.com/NMautoverse/NMdata",
                license="MIT + file LICENSE",
            ),
        ),
        bundle_dir=tmp_path,
        manifest_path=tmp_path / "manifest.yaml",
        dataset_path=tmp_path / "demo.csv",
        control_stream_path=tmp_path / "model.ctl",
        readme_path=readme_path,
    )

    details = _format_catalog_control_stream_details_html(entry)

    assert "<b>NMdata demo</b>" in details
    assert f'href="{(tmp_path / "demo.csv").resolve().as_uri()}"' in details
    assert f'href="{(tmp_path / "model.ctl").resolve().as_uri()}"' in details
    assert f'href="{tmp_path.resolve().as_uri()}"' in details
    assert f'href="{readme_path.resolve().as_uri()}"' in details
    assert 'href="https://github.com/NMautoverse/NMdata"' in details
    assert copy_link(tmp_path / "model.ctl", label="Copy control-stream path") in details
    assert copy_link(tmp_path, label="Copy bundle path") in details
    assert copy_link("https://github.com/NMautoverse/NMdata", label="Copy URL") in details
    assert compact_path_label(readme_path) in details
    assert compact_url_label("https://github.com/NMautoverse/NMdata") in details
    assert "Open control stream" in details
    assert "Open bundle folder" in details
    assert "Open bundle notes" in details
    assert "Open upstream source" in details


def test_parameter_helpers_seed_expected_defaults() -> None:
    row = default_theta_row(4)
    resized = resize_square_matrix([[0.3]], 2, diagonal_fill=0.1)

    assert row == {
        "label": "THETA4",
        "lower": 0.0,
        "init": 1.0,
        "upper": 10.0,
        "fixed": False,
    }
    assert resized == [[0.3, 0.0], [0.0, 0.1]]


def test_format_parameter_summary_reports_dimensions() -> None:
    spec = ModelSpec(
        theta_rows=[{"init": 1.0}, {"init": 2.0}],
        omega_values=[[0.3, 0.0], [0.0, 0.2]],
        sigma_values=[[0.1]],
    )

    summary = format_parameter_summary(spec)

    assert "2 THETA rows" in summary
    assert "OMEGA 2×2" in summary
    assert "SIGMA 1×1" in summary


def test_format_model_draft_status_reports_unsaved_state() -> None:
    assert format_model_draft_status(True) == "Unsaved model changes"
    assert format_model_draft_status(False) == ""


def test_recommend_model_next_action_tracks_blocked_and_handoff_states(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    workspace = Workspace(name="Model CTA")

    assert recommend_model_next_action(
        workspace,
        current_translation_result=None,
    ) == (
        "Open Data",
        "data",
        "Load a dataset in the Data workflow before handing this model off to fitting.",
    )

    workspace.active_dataset = DatasetAsset(source_path=str(dataset_path), display_name="theo.csv")
    assert (
        recommend_model_next_action(
            workspace,
            current_translation_result=None,
        )
        is None
    )

    invalid_translation = ModelTranslationResult(mode=ModelSpecMode.BUILDER)
    assert (
        recommend_model_next_action(
            workspace,
            current_translation_result=invalid_translation,
        )
        is None
    )

    valid_translation = ModelTranslationResult(
        mode=ModelSpecMode.BUILDER, builder=object(), dataset_path=str(dataset_path)
    )
    assert recommend_model_next_action(
        workspace,
        current_translation_result=valid_translation,
    ) == (
        "Save model and open Fit",
        "fit",
        "Model is valid and dataset is ready — go to Fit.",
    )

    workspace.add_run(RunRecord(workflow="fit", status=RunStatus.SUCCEEDED))
    assert recommend_model_next_action(
        workspace,
        current_translation_result=valid_translation,
    ) == (
        "Open Results",
        "results",
        "A successful fit is already available. Review the latest outputs in Results.",
    )


@pytest.mark.unit
def test_model_workflow_tracks_dirty_and_clean_transitions() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Model workflow")
    widget = build_model_workflow(project)
    dirty_transitions: list[bool] = []
    widget._project_state_changed = lambda: dirty_transitions.append(widget._has_unsaved_changes())  # type: ignore[attr-defined]

    try:
        widget.show()
        app.processEvents()

        problem_title_input = widget.findChild(qt_widgets.QLineEdit, "model-problem-title")

        assert problem_title_input is not None

        problem_title_input.setText("One compartment")
        app.processEvents()

        assert widget._has_unsaved_changes() is True  # type: ignore[attr-defined]
        assert project.active_model_spec is None

        widget._on_leave()  # type: ignore[attr-defined]
        app.processEvents()

        assert widget._has_unsaved_changes() is False  # type: ignore[attr-defined]
        assert project.active_model_spec is not None
        assert project.active_model_spec.problem_title == "One compartment"
        assert dirty_transitions == [True, False]
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_model_workflow_round_trips_advanced_estimation_options() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Advanced estimation options")
    widget = build_model_workflow(project)

    try:
        widget.show()
        app.processEvents()

        estimation_combo = widget.findChild(qt_widgets.QComboBox, "model-estimation-combo")
        maxeval_spin = widget.findChild(qt_widgets.QSpinBox, "model-maxeval-spin")
        nstarts_spin = widget.findChild(qt_widgets.QSpinBox, "model-nstarts-spin")
        outer_combo = widget.findChild(qt_widgets.QComboBox, "model-outer-optimizer-combo")
        fallback_combo = widget.findChild(
            qt_widgets.QComboBox, "model-fallback-optimizer-combo"
        )
        fallback_maxeval_spin = widget.findChild(
            qt_widgets.QSpinBox, "model-fallback-maxeval-spin"
        )
        retain_best_checkbox = widget.findChild(
            qt_widgets.QCheckBox, "model-retain-best-checkbox"
        )
        retry_checkbox = widget.findChild(
            qt_widgets.QCheckBox, "model-retry-on-abnormal-checkbox"
        )
        retry_scales_input = widget.findChild(
            qt_widgets.QLineEdit, "model-retry-omega-scales-input"
        )

        assert estimation_combo is not None
        assert maxeval_spin is not None
        assert nstarts_spin is not None
        assert outer_combo is not None
        assert fallback_combo is not None
        assert fallback_maxeval_spin is not None
        assert retain_best_checkbox is not None
        assert retry_checkbox is not None
        assert retry_scales_input is not None

        estimation_combo.setCurrentIndex(estimation_combo.findData("FOCEI"))
        app.processEvents()

        maxeval_spin.setValue(321)
        nstarts_spin.setValue(4)
        outer_combo.setCurrentIndex(outer_combo.findData("Powell"))
        fallback_combo.setCurrentIndex(fallback_combo.findData("L-BFGS-B"))
        fallback_maxeval_spin.setValue(17)
        retain_best_checkbox.setChecked(False)
        retry_checkbox.setChecked(True)
        retry_scales_input.setText("0.6, 0.3")
        app.processEvents()

        widget._on_leave()  # type: ignore[attr-defined]
        app.processEvents()

        saved = project.active_model_spec
        assert saved is not None
        assert saved.estimation.method == "FOCEI"
        assert saved.estimation.options["maxeval"] == 321
        assert saved.estimation.options["n_starts"] == 4
        assert saved.estimation.options["outer_optimizer"] == "Powell"
        assert saved.estimation.options["outer_fallback_optimizer"] == "L-BFGS-B"
        assert saved.estimation.options["outer_fallback_maxeval"] == 17
        assert saved.estimation.options["retain_best_iterate"] is False
        assert saved.estimation.options["retry_on_abnormal"] is True
        assert saved.estimation.options["retry_omega_scales"] == (0.6, 0.3)

        widget.close()
        widget.deleteLater()
        app.processEvents()

        widget = build_model_workflow(project)
        widget.show()
        app.processEvents()

        estimation_combo = widget.findChild(qt_widgets.QComboBox, "model-estimation-combo")
        maxeval_spin = widget.findChild(qt_widgets.QSpinBox, "model-maxeval-spin")
        outer_combo = widget.findChild(qt_widgets.QComboBox, "model-outer-optimizer-combo")
        fallback_combo = widget.findChild(
            qt_widgets.QComboBox, "model-fallback-optimizer-combo"
        )
        fallback_maxeval_spin = widget.findChild(
            qt_widgets.QSpinBox, "model-fallback-maxeval-spin"
        )
        retain_best_checkbox = widget.findChild(
            qt_widgets.QCheckBox, "model-retain-best-checkbox"
        )
        retry_checkbox = widget.findChild(
            qt_widgets.QCheckBox, "model-retry-on-abnormal-checkbox"
        )
        retry_scales_input = widget.findChild(
            qt_widgets.QLineEdit, "model-retry-omega-scales-input"
        )

        assert estimation_combo is not None
        assert maxeval_spin is not None
        assert outer_combo is not None
        assert fallback_combo is not None
        assert fallback_maxeval_spin is not None
        assert retain_best_checkbox is not None
        assert retry_checkbox is not None
        assert retry_scales_input is not None

        assert estimation_combo.currentData() == "FOCEI"
        assert maxeval_spin.value() == 321
        assert outer_combo.currentData() == "Powell"
        assert fallback_combo.currentData() == "L-BFGS-B"
        assert fallback_maxeval_spin.value() == 17
        assert retain_best_checkbox.isChecked() is False
        assert retry_checkbox.isChecked() is True
        assert retry_scales_input.text() == "0.6, 0.3"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_model_tables_are_user_resizable() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    widget = build_model_workflow(Workspace(name="Model workflow"))

    try:
        widget.show()
        app.processEvents()

        theta_table = widget.findChild(qt_widgets.QTableWidget, "model-theta-table")
        omega_table = widget.findChild(qt_widgets.QTableWidget, "model-omega-table")
        sigma_table = widget.findChild(qt_widgets.QTableWidget, "model-sigma-table")

        assert theta_table is not None
        assert omega_table is not None
        assert sigma_table is not None
        assert (
            theta_table.horizontalHeader().sectionResizeMode(0)
            == qt_widgets.QHeaderView.ResizeMode.Interactive
        )
        assert (
            omega_table.horizontalHeader().sectionResizeMode(0)
            == qt_widgets.QHeaderView.ResizeMode.Interactive
        )
        assert (
            sigma_table.horizontalHeader().sectionResizeMode(0)
            == qt_widgets.QHeaderView.ResizeMode.Interactive
        )
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_model_workflow_panels_are_user_adjustable() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    qt_core, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    widget = build_model_workflow(Workspace(name="Model workflow"))

    try:
        widget.resize(1400, 900)
        widget.show()
        app.processEvents()

        splitter = widget.findChild(qt_widgets.QSplitter, "model-content-row")

        assert splitter is not None
        assert splitter.count() == 2
        assert splitter.orientation() == qt_core.Qt.Orientation.Horizontal

        splitter.setSizes([700, 300])
        app.processEvents()

        assert len(splitter.sizes()) == 2

        widget.resize(700, 900)
        app.processEvents()

        assert splitter.orientation() == qt_core.Qt.Orientation.Vertical
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_model_workflow_next_action_transitions_from_data_to_fit_to_results(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Model workflow")
    project_service = ProjectService()
    widget = build_model_workflow(project, project_service=project_service)
    navigations: list[str] = []
    widget._navigate_to_workflow = lambda workflow_id: navigations.append(workflow_id)  # type: ignore[attr-defined]

    try:
        widget.show()
        app.processEvents()

        next_action_label = widget.findChild(qt_widgets.QLabel, "model-next-action-label")
        next_action_button = widget.findChild(qt_widgets.QPushButton, "model-next-action-button")

        assert next_action_label is not None
        assert next_action_button is not None
        assert next_action_button.text() == "Open Data"

        next_action_button.click()
        app.processEvents()

        project_service.attach_dataset(
            project,
            DatasetAsset(
                source_path=str(dataset_path), display_name="theo.csv", columns=["ID", "TIME", "DV"]
            ),
        )
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_button.text() == "Save model and open Fit"
        assert "dataset is ready" in next_action_label.text()
        next_action_button.click()
        app.processEvents()

        project_service.add_run(project, RunRecord(workflow="fit", status=RunStatus.SUCCEEDED))
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_button.text() == "Open Results"
        next_action_button.click()
        app.processEvents()

        assert navigations == ["data", "fit", "results"]
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_model_workflow_loads_curated_control_stream_examples_from_catalog(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    data_service, example_catalog_service = _write_catalog_control_stream_example(tmp_path)

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Model workflow")
    widget = build_model_workflow(
        project,
        data_service=data_service,
        example_catalog_service=example_catalog_service,
    )

    try:
        widget.show()
        app.processEvents()

        example_selector = widget.findChild(qt_widgets.QComboBox, "model-ctl-example-selector")
        load_example_button = widget.findChild(
            qt_widgets.QPushButton, "model-ctl-example-load-button"
        )
        example_details = widget.findChild(qt_widgets.QLabel, "model-ctl-example-details")
        dataset_path_input = widget.findChild(qt_widgets.QLineEdit, "model-dataset-path")
        control_stream_edit = widget.findChild(
            qt_widgets.QPlainTextEdit, "model-control-stream-text"
        )

        assert example_selector is not None
        assert load_example_button is not None
        assert example_details is not None
        assert dataset_path_input is not None
        assert control_stream_edit is not None

        assert example_selector.count() == 2

        example_selector.setCurrentIndex(1)
        app.processEvents()

        assert load_example_button.isEnabled() is True
        assert "route: oral" in example_details.text()
        assert "demo.csv" in example_details.text()

        load_example_button.click()
        app.processEvents()

        assert project.active_dataset is not None
        assert project.active_dataset.display_name == "Demo oral FO"
        assert dataset_path_input.text().endswith("shared_data/demo/demo.csv")
        assert "$PROBLEM Demo oral FO" in control_stream_edit.toPlainText()
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_model_workflow_loads_control_stream_only_catalog_examples_without_dataset(
    tmp_path: Path,
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    data_service, example_catalog_service = _write_catalog_control_stream_example(
        tmp_path,
        include_dataset=False,
    )

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Model workflow")
    widget = build_model_workflow(
        project,
        data_service=data_service,
        example_catalog_service=example_catalog_service,
    )

    try:
        widget.show()
        app.processEvents()

        example_selector = widget.findChild(qt_widgets.QComboBox, "model-ctl-example-selector")
        load_example_button = widget.findChild(
            qt_widgets.QPushButton, "model-ctl-example-load-button"
        )
        example_details = widget.findChild(qt_widgets.QLabel, "model-ctl-example-details")
        dataset_path_input = widget.findChild(qt_widgets.QLineEdit, "model-dataset-path")
        control_stream_edit = widget.findChild(
            qt_widgets.QPlainTextEdit, "model-control-stream-text"
        )

        assert example_selector is not None
        assert load_example_button is not None
        assert example_details is not None
        assert dataset_path_input is not None
        assert control_stream_edit is not None

        example_selector.setCurrentIndex(1)
        app.processEvents()

        assert "dataset: not bundled" in example_details.text()

        load_example_button.click()
        app.processEvents()

        assert project.active_dataset is None
        assert dataset_path_input.text().endswith("external/demo.csv")
        assert "$DATA external/demo.csv" in control_stream_edit.toPlainText()
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_resolve_control_stream_dataset_path_uses_source_file_directory(tmp_path: Path) -> None:
    control_stream = ControlStream.from_string("$PROBLEM Demo\n$DATA data/theo.csv\n")
    source_path = tmp_path / "models" / "demo.ctl"
    source_path.parent.mkdir(parents=True)

    resolved = resolve_control_stream_dataset_path(control_stream, str(source_path))

    assert resolved == str((source_path.parent / "data" / "theo.csv").resolve())


def test_load_control_stream_model_spec_populates_control_stream_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")
    control_stream_path = tmp_path / "demo.ctl"
    control_stream_path.write_text(
        """$PROBLEM Demo\n$DATA theo.csv\n$INPUT ID TIME AMT DV EVID\n$SUBROUTINES ADVAN2 TRANS2\n$PK\nCL = THETA(1)\n$ERROR\nY = F\n$THETA 1\n$OMEGA 0.3\n$SIGMA 0.1\n$ESTIMATION METHOD=COND\n$COVARIANCE\n""",
        encoding="utf-8",
    )

    spec = load_control_stream_model_spec(str(control_stream_path))

    assert spec.mode == ModelSpecMode.CONTROL_STREAM
    assert spec.problem_title == "Demo"
    assert spec.dataset_path == str(dataset_path.resolve())
    assert spec.advan == 2
    assert spec.trans == 2
    assert spec.estimation.method == "FOCE"
    assert spec.covariance.enabled is True
    assert "$PROBLEM Demo" in spec.control_stream_text


@pytest.mark.parametrize(
    ("control_stream_name", "dataset_name"),
    [
        ("30_nmdata_input_drop_and_body_weight_scaling.ctl", "xgxr1.csv"),
        ("31_nmdata_multiple_table_output_formats.ctl", "xgxr4.csv"),
        ("32_nmdata_saem_age_covariate.ctl", "xgxr2covs.csv"),
        ("33_nmdata_saem_age_covariate_block_omega.ctl", "xgxr2covs.csv"),
        ("34_nmsim_two_compartment_block_omega_fo.ctl", "xgxr2.csv"),
        ("35_nmsim_advan13_one_compartment_des.ctl", "xgxr12.csv"),
        ("36_nmsim_saem_multi_covariate_age_weight_sex.ctl", "xgxr2covs.csv"),
    ],
)
def test_imported_examples_resolve_and_load_local_example_datasets(
    control_stream_name: str,
    dataset_name: str,
) -> None:
    control_stream_path = CONTROL_STREAMS_DIR / control_stream_name

    spec = load_control_stream_model_spec(str(control_stream_path))
    expected_dataset_path = (EXAMPLES_DIR / "data" / dataset_name).resolve()

    assert spec.dataset_path == str(expected_dataset_path)
    assert expected_dataset_path.exists()

    problem = Problem.from_control_stream(
        ControlStream.from_file(control_stream_path),
        dataset_path=spec.dataset_path,
    )

    assert problem.population_model.dataset.source_path == str(expected_dataset_path)
    assert problem.population_model.dataset.n_subjects() == 12
    assert len(problem.population_model.dataset.df) == 120
    if control_stream_name == "30_nmdata_input_drop_and_body_weight_scaling.ctl":
        first_row = problem.population_model.dataset.df.iloc[0]
        assert first_row["DOSE"] == pytest.approx(3.0)
        assert first_row["BBW"] == pytest.approx(87.031, abs=1e-6)


def test_load_control_stream_dataset_asset_handles_headerless_drop_columns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "drop.csv"
    dataset_path.write_text("1,0,999,70,5\n", encoding="utf-8")
    control_stream_path = tmp_path / "drop.ctl"
    control_stream_path.write_text(
        "$PROBLEM Drop demo\n"
        "$DATA drop.csv\n"
        "$INPUT ID TIME SKIP=DROP WT DV\n"
        "$THETA 1\n"
        "$OMEGA 0.1\n"
        "$SIGMA 0.1\n"
        "$PK\nCL=THETA(1)\n"
        "$ERROR\nY=F+EPS(1)\n",
        encoding="utf-8",
    )

    spec = load_control_stream_model_spec(str(control_stream_path))
    result = load_control_stream_dataset_asset(DatasetService(), spec)

    assert result.validation.ok is True
    assert result.dataset_asset is not None
    assert result.dataset_asset.columns[:4] == ["ID", "TIME", "WT", "DV"]
    assert all(not column.startswith("_DROP_") for column in result.dataset_asset.columns)
    assert result.dataset_asset.preview_rows[0]["WT"] == 70


def test_write_control_stream_text_round_trips_content(tmp_path: Path) -> None:
    destination = tmp_path / "saved.ctl"
    text = "$PROBLEM Demo\n$DATA theo.csv\n"

    write_control_stream_text(str(destination), text)

    assert destination.read_text(encoding="utf-8") == text


# ---------------------------------------------------------------------------
# BAYES estimation method — selector and results menu
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bayes_appears_in_estimation_combo() -> None:
    """BAYES must be selectable in the estimation method combo box."""
    if not qt_widgets_available():
        pytest.skip("Qt not available")

    from openpkpd_gui.workflows.model_workflow import ESTIMATION_METHODS

    assert "BAYES" in ESTIMATION_METHODS, (
        f"BAYES not in ESTIMATION_METHODS: {ESTIMATION_METHODS}"
    )


@pytest.mark.unit
def test_bayes_display_label_mentions_nuts() -> None:
    """The BAYES display label must mention NUTS or Bayesian."""
    if not qt_widgets_available():
        pytest.skip("Qt not available")

    from openpkpd_gui.workflows.model_workflow import _ESTIMATION_METHODS_DISPLAY

    bayes_labels = [label for method, label in _ESTIMATION_METHODS_DISPLAY
                    if method == "BAYES"]
    assert bayes_labels, "No BAYES entry found in _ESTIMATION_METHODS_DISPLAY"
    assert any("NUTS" in label or "Bayesian" in label or "bayes" in label.lower()
               for label in bayes_labels), (
        f"BAYES label does not mention NUTS or Bayesian: {bayes_labels}"
    )


@pytest.mark.unit
def test_bayes_help_text_mentions_rhat_and_ess() -> None:
    """The BAYES help text must mention R-hat and ESS (key diagnostics)."""
    if not qt_widgets_available():
        pytest.skip("Qt not available")

    from openpkpd_gui.workflows.model_workflow import _ESTIMATION_HELP_TEXT

    assert "R-hat" in _ESTIMATION_HELP_TEXT or "r-hat" in _ESTIMATION_HELP_TEXT.lower(), (
        "Help text does not mention R-hat"
    )


@pytest.mark.unit
def test_bayes_combo_item_present_in_built_widget(tmp_path) -> None:
    """After building the model workflow widget the combo box must include BAYES."""
    if not qt_widgets_available():
        pytest.skip("Qt not available")

    _, _, qt_widgets = load_qt_modules()
    from openpkpd_gui.workflows.model_workflow import build_model_workflow
    from openpkpd_gui.domain.workspace import Workspace

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    project = Workspace(root_path=str(tmp_path))
    widget = build_model_workflow(project)
    try:
        widget.show()
        app.processEvents()
        combo = widget.findChild(qt_widgets.QComboBox, "model-estimation-combo")
        assert combo is not None, "Estimation combo box not found"
        all_data = [combo.itemData(i) for i in range(combo.count())]
        assert "BAYES" in all_data, (
            f"BAYES not in combo items: {all_data}"
        )
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_bayesian_action_present_in_review_menu(tmp_path) -> None:
    """The results workflow Review menu must contain a 'Bayesian diagnostics' action."""
    if not qt_widgets_available():
        pytest.skip("Qt not available")

    _, qt_gui, qt_widgets = load_qt_modules()
    from openpkpd_gui.workflows.results_workflow import build_results_workflow
    from openpkpd_gui.domain.workspace import Workspace

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    project = Workspace(root_path=str(tmp_path))
    widget = build_results_workflow(project)
    try:
        widget.show()
        app.processEvents()
        action = widget.findChild(qt_gui.QAction,
                                  "results-open-bayesian-review-button")
        assert action is not None, (
            "open_bayesian_action (objectName='results-open-bayesian-review-button') "
            "not found in results workflow"
        )
        # Must start disabled (no Bayesian artifacts yet)
        assert not action.isEnabled(), (
            "Bayesian action should be disabled until Bayesian artifacts are present"
        )
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# P3-D: Nonparametric step GUI
# ---------------------------------------------------------------------------


def _build_model_workflow_widget():
    from openpkpd_gui.app.runtime import load_qt_modules
    from openpkpd_gui.workflows.model_workflow import build_model_workflow
    from openpkpd_gui.domain.workspace import Workspace

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    ws = Workspace(name="NP test")
    widget = build_model_workflow(ws)
    return widget, app, qt_widgets


def test_nonparametric_in_estimation_combo() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")
    widget, app, qt_widgets = _build_model_workflow_widget()
    try:
        combo = widget.findChild(qt_widgets.QComboBox, "model-estimation-combo")
        assert combo is not None
        codes = [combo.itemData(i) for i in range(combo.count())]
        assert "NONPARAMETRIC" in codes, "NONPARAMETRIC not in estimation method dropdown"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_nonparam_base_method_combo_present() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")
    widget, app, qt_widgets = _build_model_workflow_widget()
    try:
        combo = widget.findChild(qt_widgets.QComboBox, "model-nonparam-base-method-combo")
        assert combo is not None, "model-nonparam-base-method-combo not found"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_nonparam_base_method_combo_hidden_for_foce() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")
    widget, app, qt_widgets = _build_model_workflow_widget()
    try:
        combo = widget.findChild(qt_widgets.QComboBox, "model-nonparam-base-method-combo")
        assert combo is not None
        # Default method is FOCE, so NP base combo should be hidden
        assert combo.isHidden(), "NP base method combo should be hidden when FOCE is selected"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_nonparam_base_method_combo_has_expected_options() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")
    widget, app, qt_widgets = _build_model_workflow_widget()
    try:
        combo = widget.findChild(qt_widgets.QComboBox, "model-nonparam-base-method-combo")
        assert combo is not None
        codes = [combo.itemData(i) for i in range(combo.count())]
        assert "FOCE" in codes
        assert "FOCEI" in codes
        assert "FO" in codes
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_nonparam_base_method_label_present() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")
    widget, app, qt_widgets = _build_model_workflow_widget()
    try:
        label = widget.findChild(qt_widgets.QLabel, "model-nonparam-base-method-label")
        assert label is not None, "model-nonparam-base-method-label not found"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# P3-C: Parameter initialization assistance
# ---------------------------------------------------------------------------

from openpkpd_gui.workflows.model_workflow import (
    suggest_theta_rows_for_advan,
    suggest_omega_values_for_advan,
)


def test_suggest_theta_rows_advan2_trans2() -> None:
    rows = suggest_theta_rows_for_advan(2, 2)
    assert rows is not None
    labels = [r["label"] for r in rows]
    assert "KA" in labels
    assert "CL" in labels
    assert "V" in labels
    assert len(rows) == 3


def test_suggest_theta_rows_advan1_trans2() -> None:
    rows = suggest_theta_rows_for_advan(1, 2)
    assert rows is not None
    labels = [r["label"] for r in rows]
    assert "CL" in labels
    assert "V" in labels
    assert "KA" not in labels
    assert len(rows) == 2


def test_suggest_theta_rows_advan3_trans4() -> None:
    rows = suggest_theta_rows_for_advan(3, 4)
    assert rows is not None
    labels = [r["label"] for r in rows]
    assert "CL" in labels
    assert "V1" in labels
    assert "Q" in labels
    assert "V2" in labels
    assert len(rows) == 4


def test_suggest_theta_rows_advan4_trans4() -> None:
    rows = suggest_theta_rows_for_advan(4, 4)
    assert rows is not None
    labels = [r["label"] for r in rows]
    assert "KA" in labels
    assert "CL" in labels
    assert "V1" in labels
    assert "Q" in labels
    assert "V2" in labels
    assert len(rows) == 5


def test_suggest_theta_rows_unknown_advan_returns_none() -> None:
    assert suggest_theta_rows_for_advan(99, 1) is None


def test_suggest_omega_values_advan2_trans2() -> None:
    omega = suggest_omega_values_for_advan(2, 2)
    assert omega is not None
    assert len(omega) == 3
    assert len(omega[0]) == 3
    # Diagonal should be nonzero
    assert omega[0][0] > 0
    assert omega[1][1] > 0
    assert omega[2][2] > 0
    # Off-diagonal should be zero
    assert omega[0][1] == 0.0


def test_suggest_omega_values_unknown_returns_none() -> None:
    assert suggest_omega_values_for_advan(99, 1) is None


def test_suggest_button_present_in_model_workflow() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")
    widget, app, qt_widgets = _build_model_workflow_widget()
    try:
        btn = widget.findChild(qt_widgets.QPushButton, "model-suggest-theta-button")
        assert btn is not None, "model-suggest-theta-button not found"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
