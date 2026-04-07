"""Smoke test for the deferred-import Qt shell."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from openpkpd_gui.app.main import configure_application_style
from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.serialization_service import ProjectSnapshotService
from openpkpd_gui.shell.main_window import create_main_window as _create_main_window
from openpkpd_gui.widgets.semantic_state import SEMANTIC_STATE_STYLESHEET
from openpkpd_gui.workflows.registry import DEFAULT_WORKFLOWS


class FakeSettingsStore:
    """Small in-memory settings store for shell smoke tests."""

    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def value(self, key: str, default: object | None = None):
        return self._values.get(key, default)

    def setValue(self, key: str, value: object) -> None:
        self._values[key] = value

    def remove(self, key: str) -> None:
        self._values.pop(key, None)


def create_main_window(*args, **kwargs):
    kwargs.setdefault("settings_store", FakeSettingsStore())
    return _create_main_window(*args, **kwargs)


def _find_tree_child(node, label: str):
    if hasattr(node, "topLevelItemCount"):
        for index in range(node.topLevelItemCount()):
            item = node.topLevelItem(index)
            if item is not None and item.text(0) == label:
                return item
        return None
    for index in range(node.childCount()):
        item = node.child(index)
        if item is not None and item.text(0) == label:
            return item
    return None


def _select_nav_path(nav, *labels: str):
    item = None
    current_node = nav
    for label in labels:
        item = _find_tree_child(current_node, label)
        assert item is not None, f"Navigation item not found: {' / '.join(labels)}"
        nav.expandItem(item)
        current_node = item
    nav.setCurrentItem(item)
    return item


_WORKFLOW_GROUP_BY_LABEL = {
    workflow.label: workflow.section
    for workflow in DEFAULT_WORKFLOWS
    if workflow.workflow_id not in {"home", "overview"}
}


def _scenario_workflow_path(project: Workspace, workflow_label: str) -> tuple[str, ...]:
    base_path = ("Workspace", project.active_project.name, project.active_scenario.name)
    if workflow_label == "Dashboard":
        return (*base_path, workflow_label)
    group_label = _WORKFLOW_GROUP_BY_LABEL.get(workflow_label)
    if group_label is None:
        return (*base_path, workflow_label)
    return (*base_path, group_label, workflow_label)


def _find_scenario_workflow_item(scenario_item, workflow_label: str):
    if workflow_label == "Dashboard":
        return _find_tree_child(scenario_item, workflow_label)
    group_label = _WORKFLOW_GROUP_BY_LABEL.get(workflow_label)
    parent = scenario_item if group_label is None else _find_tree_child(scenario_item, group_label)
    assert parent is not None
    return _find_tree_child(parent, workflow_label)


def _snapshot_project_payload(snapshot_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(snapshot_path, "r") as archive:
        return dict(json.loads(archive.read("workspace.json").decode("utf-8"))["project"])


@pytest.mark.unit
def test_create_main_window_smoke() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    window = create_main_window(Workspace(name="Smoke"))

    assert window.windowTitle() == "OpenPKPD — Smoke"
    assert app.applicationName() in {"", "OpenPKPD"}
    assert window.findChild(qt_gui.QAction, "file-new-project-action") is not None
    assert window.findChild(qt_gui.QAction, "file-open-project-action") is not None
    assert window.findChild(qt_gui.QAction, "file-save-project-action") is not None
    assert window.findChild(qt_gui.QAction, "file-save-project-as-action") is not None
    assert window.findChild(qt_gui.QAction, "file-close-action") is not None
    exit_action = window.findChild(qt_gui.QAction, "file-exit-action")
    assert exit_action is not None
    assert exit_action.text() in {"Quit", "Exit"}  # "Quit" on macOS, "Exit" elsewhere
    assert window.findChild(qt_gui.QAction, "workspace-new-project-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-new-scenario-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-duplicate-project-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-duplicate-scenario-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-save-project-snapshot-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-save-scenario-snapshot-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-load-project-snapshot-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-load-scenario-snapshot-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-rename-project-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-rename-scenario-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-edit-project-details-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-edit-scenario-details-action") is not None
    assert window.findChild(qt_widgets.QWidget, "project-details-page") is not None
    assert window.findChild(qt_widgets.QWidget, "scenario-details-page") is not None
    assert window.findChild(qt_gui.QAction, "workspace-delete-project-action") is not None
    assert window.findChild(qt_gui.QAction, "workspace-delete-scenario-action") is not None
    assert window.findChild(qt_gui.QAction, "navigate-dashboard-action") is not None
    assert window.findChild(qt_gui.QAction, "navigate-nca-action") is not None
    assert window.findChild(qt_gui.QAction, "navigate-results-action") is not None
    assert window.findChild(qt_gui.QAction, "navigate-diagnostics-action") is not None
    assert window.findChild(qt_gui.QAction, "inputs-import-dataset-action") is not None
    assert window.findChild(qt_gui.QAction, "inputs-open-control-stream-action") is not None
    assert window.findChild(qt_gui.QAction, "results-open-latest-report-action") is not None
    assert window.findChild(qt_gui.QAction, "results-save-latest-report-copy-action") is not None
    assert window.findChild(qt_gui.QAction, "results-export-latest-report-pdf-action") is not None
    assert window.findChild(qt_gui.QAction, "results-open-latest-plot-action") is not None
    assert window.findChild(qt_gui.QAction, "results-save-latest-plot-copy-action") is not None
    assert window.findChild(qt_gui.QAction, "diagnostics-open-gof-panel-action") is not None
    assert window.findChild(qt_gui.QAction, "diagnostics-open-residual-trends-action") is not None
    assert window.findChild(qt_gui.QAction, "diagnostics-open-artifact-folder-action") is not None
    assert window.findChild(qt_gui.QAction, "settings-preferences-action") is not None
    assert window.findChild(qt_gui.QAction, "results-open-latest-plot-button") is not None
    assert window.findChild(qt_gui.QAction, "results-save-latest-plot-copy-button") is not None
    assert window.findChild(qt_gui.QAction, "results-export-latest-report-pdf-button") is not None
    assert window.findChild(qt_widgets.QToolButton, "results-review-menu-button") is not None
    assert window.findChild(qt_widgets.QToolButton, "results-export-menu-button") is not None
    assert window.findChild(qt_gui.QAction, "help-about-action") is not None
    assert window.findChild(qt_widgets.QToolBar, "main-toolbar") is None
    assert window.findChild(qt_widgets.QWidget, "shell-sidebar") is not None
    assert window.findChild(qt_widgets.QWidget, "sidebar-action-bar") is None
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    assert nav is not None
    assert stack is not None
    assert window.findChild(qt_widgets.QToolButton, "sidebar-workspace-menu-button") is None
    assert window.findChild(qt_widgets.QToolButton, "sidebar-new-project-button") is None
    assert window.findChild(qt_widgets.QToolButton, "sidebar-save-project-snapshot-button") is None
    assert nav.topLevelItemCount() == 1
    assert nav.topLevelItem(0).text(0) == "Workspace"
    project_item = _find_tree_child(nav.topLevelItem(0), "Project 1")
    assert project_item is not None
    scenario_item = _find_tree_child(project_item, "Baseline")
    assert scenario_item is not None
    assert _find_tree_child(scenario_item, "Dashboard") is not None
    assert _find_tree_child(scenario_item, "Inputs") is not None
    assert _find_tree_child(scenario_item, "Analyses") is not None
    assert _find_tree_child(scenario_item, "Review") is not None
    assert _find_scenario_workflow_item(scenario_item, "Data") is not None
    assert _find_scenario_workflow_item(scenario_item, "Fit") is not None
    assert _find_scenario_workflow_item(scenario_item, "Results") is not None
    assert stack.currentWidget().objectName() == "dashboard-workflow"
    assert window.findChild(qt_widgets.QWidget, "overview-workflow") is not None
    assert window.findChild(qt_widgets.QWidget, "data-workflow") is not None
    assert window.findChild(qt_widgets.QWidget, "model-workflow") is not None
    assert window.findChild(qt_widgets.QWidget, "fit-workflow") is not None
    assert window.findChild(qt_widgets.QWidget, "nca-workflow") is not None
    assert window.findChild(qt_widgets.QWidget, "results-workflow") is not None
    assert window.findChild(qt_widgets.QWidget, "diagnostics-workflow") is not None
    assert window.findChild(qt_widgets.QWidget, "covariate-workflow") is not None
    assert window.findChild(qt_widgets.QWidget, "advanced-workflow") is not None
    assert window.minimumSizeHint().height() < 900
    assert window.findChild(qt_widgets.QScrollArea, "data-workflow-scroll-area") is not None
    assert window.findChild(qt_widgets.QScrollArea, "model-workflow-scroll-area") is not None
    assert window.findChild(qt_widgets.QScrollArea, "results-workflow-scroll-area") is not None


@pytest.mark.unit
def test_home_workflow_renders_workspace_context_and_hands_off_to_current_work(
    tmp_path: Path, monkeypatch
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")
    recent_snapshot = tmp_path / "recent-project.opkpd"
    recent_snapshot.write_text("placeholder", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Home")
    project.recent_files = [str(recent_snapshot)]
    project.root_path = str(tmp_path)
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(
            source_path=str(dataset_path), display_name="dataset.csv", columns=["ID", "TIME", "DV"]
        ),
    )

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    home_page = window.findChild(qt_widgets.QWidget, "dashboard-workflow")

    assert nav is not None
    assert stack is not None
    assert home_page is not None

    _select_nav_path(nav, "Workspace")
    app.processEvents()

    summary_label = home_page.findChild(qt_widgets.QLabel, "dashboard-workspace-name-label")
    next_action_button = home_page.findChild(qt_widgets.QPushButton, "overview-next-action-button")
    new_project_button = home_page.findChild(qt_widgets.QPushButton, "dashboard-new-project-button")
    recent_button = home_page.findChild(
        qt_widgets.QPushButton, "dashboard-recent-snapshot-button-0"
    )

    assert summary_label is not None
    assert next_action_button is not None
    assert new_project_button is not None
    assert recent_button is not None
    assert stack.currentWidget().objectName() == "dashboard-workflow"
    assert summary_label.text() == "Home"
    assert "recent-project.opkpd" in recent_button.text()
    assert next_action_button.text() == "Open Model"

    next_action_button.click()
    app.processEvents()

    assert stack.currentWidget().objectName() == "model-workflow"

    _select_nav_path(nav, "Workspace")
    app.processEvents()

    window._prompt_for_name_override = lambda **kwargs: "Landing Project"  # type: ignore[attr-defined]

    new_project_button.click()
    app.processEvents()

    assert project.active_project.name == "Landing Project"
    assert stack.currentWidget().objectName() == "dashboard-workflow"
    assert _find_tree_child(nav.topLevelItem(0), "Landing Project") is not None


@pytest.mark.unit
def test_workspace_menu_action_creates_project_and_selects_baseline_scenario(monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    window = create_main_window(project)
    window.show()
    window.activateWindow()
    app.processEvents()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    file_new_project_action = window.findChild(qt_gui.QAction, "file-new-project-action")
    new_project_action = window.findChild(qt_gui.QAction, "workspace-new-project-action")

    assert nav is not None
    assert stack is not None
    assert file_new_project_action is not None
    assert new_project_action is not None

    window._prompt_for_name_override = lambda **kwargs: "Dose Escalation"  # type: ignore[attr-defined]

    file_new_project_action.trigger()
    app.processEvents()

    assert len(project.projects) == 2
    assert project.active_project.name == "Dose Escalation"
    assert project.active_scenario.name == "Baseline"
    assert stack.currentWidget().objectName() == "dashboard-workflow"
    assert _find_tree_child(nav.topLevelItem(0), "Dose Escalation") is not None
    assert window.windowTitle() == "OpenPKPD — Workspace Actions *"


@pytest.mark.unit
def test_workspace_menu_action_creates_scenario_and_clones_saved_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "baseline.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(dataset_path), display_name="baseline.csv", row_count=1),
    )
    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Baseline model",
            dataset_path=str(dataset_path),
            pk_code="CL = THETA(1)",
            error_code="Y = F",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )
    baseline_run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)
    baseline_artifact = ArtifactRecord(
        kind="report", label="Baseline report", path=str(tmp_path / "report.html")
    )
    baseline_run.artifact_ids.append(baseline_artifact.artifact_id)
    project_service.add_run(project, baseline_run)
    project_service.add_artifact(project, baseline_artifact)

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    data_path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    sidebar_summary = window.findChild(qt_widgets.QLabel, "sidebar-project-path")
    new_scenario_action = window.findChild(qt_gui.QAction, "workspace-new-scenario-action")

    assert nav is not None
    assert stack is not None
    assert data_path_input is not None
    assert sidebar_summary is not None
    assert new_scenario_action is not None

    window._prompt_for_name_override = lambda **kwargs: "Variant B"  # type: ignore[attr-defined]

    new_scenario_action.trigger()
    app.processEvents()

    assert len(project.active_project.scenarios) == 2
    assert project.active_scenario.name == "Variant B"
    assert project.active_scenario.active_dataset is not None
    assert project.active_scenario.active_dataset.display_name == "baseline.csv"
    assert project.active_scenario.active_model_spec is not None
    assert project.active_scenario.active_model_spec.problem_title == "Baseline model"
    assert project.active_scenario.runs == []
    assert project.active_scenario.artifacts == []
    assert stack.currentWidget().objectName() == "dashboard-workflow"
    assert data_path_input.text() == str(dataset_path)
    project_item = _find_tree_child(nav.topLevelItem(0), project.active_project.name)
    assert project_item is not None
    scenario_item = _find_tree_child(project_item, "Variant B")
    assert scenario_item is not None
    assert "Parent scenario: Baseline" in scenario_item.toolTip(0)
    assert "Parent scenario: Baseline" in sidebar_summary.text()


@pytest.mark.unit
def test_workspace_menu_action_duplicates_scenario_and_clones_saved_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "baseline.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(dataset_path), display_name="baseline.csv", row_count=1),
    )
    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Baseline model", dataset_path=str(dataset_path), pk_code="CL = THETA(1)"
        ),
    )
    baseline_run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)
    baseline_artifact = ArtifactRecord(
        kind="report", label="Baseline report", path=str(tmp_path / "report.html")
    )
    project_service.add_run(project, baseline_run)
    project_service.add_artifact(project, baseline_artifact)

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    data_path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    duplicate_scenario_action = window.findChild(
        qt_gui.QAction, "workspace-duplicate-scenario-action"
    )

    assert nav is not None
    assert stack is not None
    assert data_path_input is not None
    assert duplicate_scenario_action is not None

    window._prompt_for_name_override = lambda **kwargs: "Baseline Copy"  # type: ignore[attr-defined]

    duplicate_scenario_action.trigger()
    app.processEvents()

    assert len(project.active_project.scenarios) == 2
    assert project.active_scenario.name == "Baseline Copy"
    assert project.active_scenario.active_dataset is not None
    assert project.active_scenario.active_dataset.display_name == "baseline.csv"
    assert project.active_scenario.active_model_spec is not None
    assert project.active_scenario.active_model_spec.problem_title == "Baseline model"
    assert project.active_scenario.runs == []
    assert project.active_scenario.artifacts == []
    assert stack.currentWidget().objectName() == "dashboard-workflow"
    assert data_path_input.text() == str(dataset_path)
    project_item = _find_tree_child(nav.topLevelItem(0), project.active_project.name)
    assert project_item is not None
    assert _find_tree_child(project_item, "Baseline Copy") is not None


@pytest.mark.unit
def test_inputs_menu_action_imports_dataset_into_active_scenario(
    monkeypatch, tmp_path: Path
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "menu-import.csv"
    dataset_path.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Input menu")
    window = create_main_window(project)

    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    import_dataset_action = window.findChild(qt_gui.QAction, "inputs-import-dataset-action")

    assert stack is not None
    assert path_input is not None
    assert import_dataset_action is not None

    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(dataset_path), "Delimited data (*.csv *.txt)"),
    )

    import_dataset_action.trigger()
    app.processEvents()

    assert stack.currentWidget().objectName() == "data-workflow"
    assert project.active_dataset is not None
    assert project.active_dataset.source_path == str(dataset_path)
    assert path_input.text() == str(dataset_path)
    assert window.windowTitle() == "OpenPKPD — Input menu *"


@pytest.mark.unit
def test_inputs_menu_action_opens_nonmem_file_in_model_editor(monkeypatch, tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")
    control_stream_path = tmp_path / "menu-demo.ctl"
    control_stream_path.write_text(
        """$PROBLEM Demo\n$DATA theo.csv\n$INPUT ID TIME AMT DV EVID\n$SUBROUTINES ADVAN2 TRANS2\n$PK\nCL = THETA(1)\n$ERROR\nY = F\n$THETA 1\n$OMEGA 0.3\n$SIGMA 0.1\n$ESTIMATION METHOD=COND\n$COVARIANCE\n""",
        encoding="utf-8",
    )

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Input menu")
    window = create_main_window(project)

    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    mode_radio_ctl = window.findChild(qt_widgets.QRadioButton, "model-mode-radio-ctl")
    dataset_path_input = window.findChild(qt_widgets.QLineEdit, "model-dataset-path")
    control_stream_edit = window.findChild(qt_widgets.QPlainTextEdit, "model-control-stream-text")
    open_control_stream_action = window.findChild(
        qt_gui.QAction, "inputs-open-control-stream-action"
    )

    assert stack is not None
    assert mode_radio_ctl is not None
    assert dataset_path_input is not None
    assert control_stream_edit is not None
    assert open_control_stream_action is not None

    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (
            str(control_stream_path),
            "NONMEM control stream (*.ctl *.mod *.txt)",
        ),
    )

    open_control_stream_action.trigger()
    app.processEvents()

    assert stack.currentWidget().objectName() == "model-workflow"
    assert mode_radio_ctl.isChecked() is True
    assert dataset_path_input.text() == str(dataset_path.resolve())
    assert "$PROBLEM Demo" in control_stream_edit.toPlainText()
    assert project.active_model_spec is None
    assert window.windowTitle() == "OpenPKPD — Input menu *"


@pytest.mark.unit
def test_workspace_menu_action_duplicates_project_and_selects_copied_scenario(
    monkeypatch, tmp_path: Path
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    baseline_path = tmp_path / "baseline.csv"
    variant_path = tmp_path / "variant.csv"
    baseline_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")
    variant_path.write_text("ID,TIME,DV\n1,0,1\n", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(baseline_path), display_name="baseline.csv", row_count=1),
    )
    project_service.create_scenario(project, name="Variant A")
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(variant_path), display_name="variant.csv", row_count=1),
    )

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    sidebar_summary = window.findChild(qt_widgets.QLabel, "sidebar-project-path")
    duplicate_project_action = window.findChild(
        qt_gui.QAction, "workspace-duplicate-project-action"
    )

    assert nav is not None
    assert stack is not None
    assert path_input is not None
    assert sidebar_summary is not None
    assert duplicate_project_action is not None

    window._prompt_for_name_override = lambda **kwargs: "Project Copy"  # type: ignore[attr-defined]

    duplicate_project_action.trigger()
    app.processEvents()

    assert len(project.projects) == 2
    assert project.active_project.name == "Project Copy"
    assert len(project.active_project.scenarios) == 2
    assert project.active_scenario.name == "Variant A"
    assert project.active_scenario.active_dataset is not None
    assert project.active_scenario.active_dataset.display_name == "variant.csv"
    assert project.active_scenario.runs == []
    assert project.active_scenario.artifacts == []
    assert stack.currentWidget().objectName() == "dashboard-workflow"
    assert path_input.text() == str(variant_path)
    project_item = _find_tree_child(nav.topLevelItem(0), "Project Copy")
    assert project_item is not None
    assert _find_tree_child(project_item, "Baseline") is not None
    scenario_item = _find_tree_child(project_item, "Variant A")
    assert scenario_item is not None
    assert "Parent scenario: Baseline" in scenario_item.toolTip(0)
    assert "Parent scenario: Baseline" in sidebar_summary.text()


@pytest.mark.unit
def test_workspace_menu_action_renames_project_without_reloading_data_editor(monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    rename_project_action = window.findChild(qt_gui.QAction, "workspace-rename-project-action")

    assert nav is not None
    assert stack is not None
    assert path_input is not None
    assert rename_project_action is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Data"))
    app.processEvents()
    path_input.setText("/tmp/unsaved.csv")

    window._prompt_for_name_override = lambda **kwargs: "Dose Escalation"  # type: ignore[attr-defined]

    rename_project_action.trigger()
    app.processEvents()

    assert project.active_project.name == "Dose Escalation"
    assert path_input.text() == "/tmp/unsaved.csv"
    assert stack.currentWidget().objectName() == "data-workflow"
    assert _find_tree_child(nav.topLevelItem(0), "Dose Escalation") is not None
    assert window.windowTitle() == "OpenPKPD — Workspace Actions *"


@pytest.mark.unit
def test_workspace_menu_action_renames_scenario_without_reloading_model_editor(monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    problem_title_input = window.findChild(qt_widgets.QLineEdit, "model-problem-title")
    rename_scenario_action = window.findChild(qt_gui.QAction, "workspace-rename-scenario-action")

    assert nav is not None
    assert stack is not None
    assert problem_title_input is not None
    assert rename_scenario_action is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Model"))
    app.processEvents()
    problem_title_input.setText("Unsaved title")

    window._prompt_for_name_override = lambda **kwargs: "Baseline A"  # type: ignore[attr-defined]

    rename_scenario_action.trigger()
    app.processEvents()

    assert project.active_scenario.name == "Baseline A"
    assert problem_title_input.text() == "Unsaved title"
    assert stack.currentWidget().objectName() == "model-workflow"
    project_item = _find_tree_child(nav.topLevelItem(0), project.active_project.name)
    assert project_item is not None
    assert _find_tree_child(project_item, "Baseline A") is not None
    assert window.windowTitle() == "OpenPKPD — Workspace Actions *"


@pytest.mark.unit
def test_selecting_project_item_shows_details_page_and_saves_changes() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    sidebar_summary = window.findChild(qt_widgets.QLabel, "sidebar-project-path")
    name_input = window.findChild(qt_widgets.QLineEdit, "project-details-pane-name-input")
    description_input = window.findChild(
        qt_widgets.QPlainTextEdit, "project-details-pane-description-input"
    )
    references_input = window.findChild(
        qt_widgets.QPlainTextEdit, "project-details-pane-references-input"
    )
    notes_input = window.findChild(qt_widgets.QPlainTextEdit, "project-details-pane-notes-input")
    save_button = window.findChild(qt_widgets.QPushButton, "project-details-pane-save-button")

    assert nav is not None
    assert stack is not None
    assert sidebar_summary is not None
    assert name_input is not None
    assert description_input is not None
    assert references_input is not None
    assert notes_input is not None
    assert save_button is not None

    _select_nav_path(nav, "Workspace", project.active_project.name)
    app.processEvents()

    assert stack.currentWidget().objectName() == "project-details-page"
    assert name_input.text() == "Project 1"

    name_input.setText("  Dose Escalation  ")
    description_input.setPlainText("  Lead project overview\nSecond line  ")
    references_input.setPlainText("  PMID:12345\nDOI:10.1000/example  ")
    notes_input.setPlainText("  Lead project note\nThird line  ")
    app.processEvents()

    assert save_button.isEnabled() is True
    assert window.windowTitle() == "OpenPKPD — Workspace Actions *"

    save_button.click()
    app.processEvents()

    project_item = _find_tree_child(nav.topLevelItem(0), "Dose Escalation")

    assert project.active_project.name == "Dose Escalation"
    assert project.active_project.metadata["description"] == "Lead project overview\nSecond line"
    assert project.active_project.metadata["references"] == "PMID:12345\nDOI:10.1000/example"
    assert project.active_project.metadata["notes"] == "Lead project note\nThird line"
    assert name_input.text() == "Dose Escalation"
    assert save_button.isEnabled() is False
    assert stack.currentWidget().objectName() == "project-details-page"
    assert project_item is not None
    assert "Description:\nLead project overview\nSecond line" in project_item.toolTip(0)
    assert "References:\nPMID:12345\nDOI:10.1000/example" in project_item.toolTip(0)
    assert "Notes:\nLead project note\nThird line" in project_item.toolTip(0)
    assert "Project: Dose Escalation" in sidebar_summary.text()
    assert "Project description: Lead project overview …" in sidebar_summary.text()
    assert "Project references: PMID:12345 …" in sidebar_summary.text()
    assert "Project notes: Lead project note …" in sidebar_summary.text()
    assert window.windowTitle() == "OpenPKPD — Workspace Actions *"


@pytest.mark.unit
def test_selecting_scenario_item_opens_overview_and_edit_action_shows_details_page_and_saves_changes() -> (
    None
):
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    sidebar_summary = window.findChild(qt_widgets.QLabel, "sidebar-project-path")
    edit_details_action = window.findChild(qt_gui.QAction, "workspace-edit-scenario-details-action")
    name_input = window.findChild(qt_widgets.QLineEdit, "scenario-details-pane-name-input")
    description_input = window.findChild(
        qt_widgets.QPlainTextEdit, "scenario-details-pane-description-input"
    )
    references_input = window.findChild(
        qt_widgets.QPlainTextEdit, "scenario-details-pane-references-input"
    )
    notes_input = window.findChild(qt_widgets.QPlainTextEdit, "scenario-details-pane-notes-input")
    save_button = window.findChild(qt_widgets.QPushButton, "scenario-details-pane-save-button")

    assert nav is not None
    assert stack is not None
    assert sidebar_summary is not None
    assert edit_details_action is not None
    assert name_input is not None
    assert description_input is not None
    assert references_input is not None
    assert notes_input is not None
    assert save_button is not None

    _select_nav_path(nav, "Workspace", project.active_project.name, project.active_scenario.name)
    app.processEvents()

    assert stack.currentWidget().objectName() == "dashboard-workflow"

    edit_details_action.trigger()
    app.processEvents()

    assert stack.currentWidget().objectName() == "scenario-details-page"
    assert name_input.text() == "Baseline"

    name_input.setText("  Scenario A  ")
    description_input.setPlainText("  Scenario overview\nLine two  ")
    references_input.setPlainText("  Ref-001\nDOI:10.1000/scenario  ")
    notes_input.setPlainText("  Scenario note\nLine three  ")
    app.processEvents()

    assert save_button.isEnabled() is True
    assert window.windowTitle() == "OpenPKPD — Workspace Actions *"

    save_button.click()
    app.processEvents()

    project_item = _find_tree_child(nav.topLevelItem(0), project.active_project.name)
    scenario_item = (
        _find_tree_child(project_item, "Scenario A") if project_item is not None else None
    )

    assert project.active_scenario.name == "Scenario A"
    assert project.active_scenario.metadata["description"] == "Scenario overview\nLine two"
    assert project.active_scenario.metadata["references"] == "Ref-001\nDOI:10.1000/scenario"
    assert project.active_scenario.metadata["notes"] == "Scenario note\nLine three"
    assert name_input.text() == "Scenario A"
    assert save_button.isEnabled() is False
    assert stack.currentWidget().objectName() == "scenario-details-page"
    assert scenario_item is not None
    assert "Description:\nScenario overview\nLine two" in scenario_item.toolTip(0)
    assert "References:\nRef-001\nDOI:10.1000/scenario" in scenario_item.toolTip(0)
    assert "Notes:\nScenario note\nLine three" in scenario_item.toolTip(0)
    assert "Scenario: Scenario A" in sidebar_summary.text()
    assert "Scenario description: Scenario overview …" in sidebar_summary.text()
    assert "Scenario references: Ref-001 …" in sidebar_summary.text()
    assert "Scenario notes: Scenario note …" in sidebar_summary.text()
    assert window.windowTitle() == "OpenPKPD — Workspace Actions *"


@pytest.mark.unit
def test_workspace_menu_action_deletes_scenario_and_reloads_fallback_model(monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    project_service = ProjectService()
    project_service.set_model_spec(
        project,
        ModelSpec(problem_title="Baseline model", pk_code="CL = THETA(1)", error_code="Y = F"),
    )
    variant = project_service.create_scenario(project, name="Variant B")
    project_service.set_model_spec(
        project,
        ModelSpec(problem_title="Variant model", pk_code="CL = THETA(2)", error_code="Y = F"),
    )
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    problem_title_input = window.findChild(qt_widgets.QLineEdit, "model-problem-title")
    delete_scenario_action = window.findChild(qt_gui.QAction, "workspace-delete-scenario-action")

    assert nav is not None
    assert stack is not None
    assert problem_title_input is not None
    assert delete_scenario_action is not None

    warning_messages: list[tuple[str, str]] = []

    def _confirm_delete(*args, **_kwargs):
        warning_messages.append((args[1], args[2]))
        return qt_widgets.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(qt_widgets.QMessageBox, "warning", _confirm_delete)

    _select_nav_path(nav, "Workspace", project.active_project.name, variant.name, "Inputs", "Model")
    app.processEvents()

    delete_scenario_action.trigger()
    app.processEvents()

    assert len(project.active_project.scenarios) == 1
    assert project.active_scenario.name == "Baseline"
    assert stack.currentWidget().objectName() == "model-workflow"
    assert problem_title_input.text() == "Baseline model"
    project_item = _find_tree_child(nav.topLevelItem(0), project.active_project.name)
    assert project_item is not None
    assert _find_tree_child(project_item, "Variant B") is None
    assert any(
        title == "Delete Scenario" and "Variant B" in message for title, message in warning_messages
    )


@pytest.mark.unit
def test_workspace_menu_action_deletes_project_and_reloads_fallback_data(
    monkeypatch, tmp_path: Path
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    baseline_path = tmp_path / "baseline.csv"
    variant_path = tmp_path / "variant.csv"
    baseline_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")
    variant_path.write_text("ID,TIME,DV\n1,0,1\n", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(baseline_path), display_name="baseline.csv", row_count=1),
    )
    project_service.create_project(project, name="Project B")
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(variant_path), display_name="variant.csv", row_count=1),
    )
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    delete_project_action = window.findChild(qt_gui.QAction, "workspace-delete-project-action")

    assert nav is not None
    assert stack is not None
    assert path_input is not None
    assert delete_project_action is not None

    warning_messages: list[tuple[str, str]] = []

    def _confirm_delete(*args, **_kwargs):
        warning_messages.append((args[1], args[2]))
        return qt_widgets.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(qt_widgets.QMessageBox, "warning", _confirm_delete)

    _select_nav_path(nav, *_scenario_workflow_path(project, "Data"))
    app.processEvents()

    delete_project_action.trigger()
    app.processEvents()

    assert len(project.projects) == 1
    assert project.active_project.name == "Project 1"
    assert project.active_scenario.name == "Baseline"
    assert stack.currentWidget().objectName() == "data-workflow"
    assert path_input.text() == str(baseline_path)
    assert _find_tree_child(nav.topLevelItem(0), "Project B") is None
    assert any(
        title == "Delete Project" and "Project B" in message for title, message in warning_messages
    )


@pytest.mark.unit
def test_workspace_menu_action_blocks_deleting_last_scenario(monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workspace Actions")
    window = create_main_window(project)
    delete_scenario_action = window.findChild(qt_gui.QAction, "workspace-delete-scenario-action")

    assert delete_scenario_action is not None

    info_messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        qt_widgets.QMessageBox,
        "information",
        lambda *args, **_kwargs: (
            info_messages.append((args[1], args[2])) or qt_widgets.QMessageBox.StandardButton.Ok
        ),
    )

    delete_scenario_action.trigger()
    app.processEvents()

    assert len(project.active_project.scenarios) == 1
    assert any("At least one scenario must remain" in message for _title, message in info_messages)


@pytest.mark.unit
def test_shell_save_and_open_project_snapshot_refreshes_live_workflows(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n1,1,5\n", encoding="utf-8")
    report_path = tmp_path / "report.html"
    report_path.write_text("<html><body>report</body></html>", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )

    project = Workspace(name="Snapshot Smoke")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(
            source_path=str(dataset_path),
            display_name="theo.csv",
            columns=["ID", "TIME", "DV"],
            row_count=2,
        ),
    )
    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Snapshotd model",
            dataset_path=str(dataset_path),
            pk_code="CL = THETA(1)",
            error_code="Y = F",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Loaded from snapshot")
    artifact = ArtifactRecord(
        kind="report",
        label="Report",
        path=str(report_path),
        source_run_id=run.run_id,
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)

    window = create_main_window(project)
    save_snapshot = getattr(window, "_save_project_snapshot", None)
    open_snapshot = getattr(window, "_open_project_snapshot", None)

    assert isinstance(save_snapshot, Callable)
    assert isinstance(open_snapshot, Callable)

    snapshot_path = tmp_path / "snapshot-smoke.opkpd"
    assert save_snapshot(snapshot_path)
    assert snapshot_path.exists()

    data_path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    model_title_input = window.findChild(qt_widgets.QLineEdit, "model-problem-title")
    results_overview = window.findChild(qt_widgets.QLabel, "results-overview-label")

    assert data_path_input is not None
    assert model_title_input is not None
    assert results_overview is not None

    data_path_input.setText("/tmp/unsaved.csv")
    model_title_input.setText("Unsaved title")
    project.name = "Mutated"
    project.active_dataset = None
    project.active_model_spec = None
    project.runs = []
    project.artifacts = []

    assert open_snapshot(snapshot_path)
    app.processEvents()

    assert window.windowTitle() == "OpenPKPD — Snapshot Smoke"
    assert project.active_dataset is not None
    assert Path(project.active_dataset.source_path or "").read_text(encoding="utf-8") == (
        dataset_path.read_text(encoding="utf-8")
    )
    assert project.active_model_spec is not None
    assert project.active_model_spec.problem_title == "Snapshotd model"
    assert data_path_input.text() == project.active_dataset.source_path
    assert model_title_input.text() == "Snapshotd model"
    assert "1 review runs" in results_overview.text()
    assert project.artifacts
    assert Path(project.artifacts[0].path or "").read_text(
        encoding="utf-8"
    ) == report_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_workspace_menu_action_exports_project_snapshot_without_replacing_current_project_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    baseline_path = tmp_path / "baseline.csv"
    dose_path = tmp_path / "dose.csv"
    baseline_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")
    dose_path.write_text("ID,TIME,DV\n1,0,10\n", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )

    project = Workspace(name="Snapshot Export")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(baseline_path), display_name="baseline.csv"),
    )
    exported_project = project_service.create_project(project, name="Dose Escalation")
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(dose_path), display_name="dose.csv"),
    )

    window = create_main_window(project)
    save_snapshot = getattr(window, "_save_project_snapshot", None)
    is_dirty = getattr(window, "_is_project_dirty", None)
    export_action = window.findChild(qt_gui.QAction, "workspace-save-project-snapshot-action")
    save_action = window.findChild(qt_gui.QAction, "file-save-project-action")

    assert isinstance(save_snapshot, Callable)
    assert isinstance(is_dirty, Callable)
    assert export_action is not None
    assert save_action is not None

    workspace_path = tmp_path / "workspace-main.opkpd"
    export_path = tmp_path / "dose-escalation.opkpd"

    assert save_snapshot(workspace_path)
    assert project.recent_files == [str(workspace_path.resolve())]
    assert is_dirty() is False

    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(export_path), "OpenPKPD project snapshots (*.opkpd)"),
    )

    export_action.trigger()
    app.processEvents()

    assert export_path.exists()
    assert project.recent_files == [str(workspace_path.resolve())]
    assert is_dirty() is False

    project_service.set_scenario_notes(project, notes="Saved after export")
    assert is_dirty() is True

    save_action.trigger()
    app.processEvents()

    assert is_dirty() is False

    workspace_payload = _snapshot_project_payload(workspace_path)
    export_payload = _snapshot_project_payload(export_path)

    assert len(workspace_payload["projects"]) == 2
    assert len(export_payload["projects"]) == 1
    assert export_payload["name"] == exported_project.name
    assert export_payload["projects"][0]["name"] == exported_project.name
    assert (
        workspace_payload["projects"][1]["scenarios"][0]["metadata"]["notes"]
        == "Saved after export"
    )
    assert export_payload["projects"][0]["scenarios"][0]["metadata"] == {}


@pytest.mark.unit
def test_workspace_menu_action_loads_project_snapshot_without_replacing_current_project_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    current_path = tmp_path / "current.csv"
    imported_baseline_path = tmp_path / "imported-baseline.csv"
    imported_branch_path = tmp_path / "imported-branch.csv"
    current_path.write_text("ID,TIME,DV\n1,0,1\n", encoding="utf-8")
    imported_baseline_path.write_text("ID,TIME,DV\n1,0,5\n", encoding="utf-8")
    imported_branch_path.write_text("ID,TIME,DV\n1,0,15\n", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )

    project = Workspace(name="Snapshot Import")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(current_path), display_name="current.csv"),
    )

    window = create_main_window(project)
    save_snapshot = getattr(window, "_save_project_snapshot", None)
    is_dirty = getattr(window, "_is_project_dirty", None)
    data_path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    load_action = window.findChild(qt_gui.QAction, "workspace-load-project-snapshot-action")
    save_action = window.findChild(qt_gui.QAction, "file-save-project-action")

    assert isinstance(save_snapshot, Callable)
    assert isinstance(is_dirty, Callable)
    assert data_path_input is not None
    assert load_action is not None
    assert save_action is not None

    workspace_path = tmp_path / "workspace-main.opkpd"
    assert save_snapshot(workspace_path)
    assert project.recent_files == [str(workspace_path.resolve())]
    assert is_dirty() is False

    source_workspace = project_service.new_workspace(
        name="Imported Workspace", root_path=str(tmp_path)
    )
    imported_project = project_service.create_project(source_workspace, name="Dose Escalation")
    project_service.attach_dataset(
        source_workspace,
        DatasetAsset(source_path=str(imported_baseline_path), display_name="imported-baseline.csv"),
    )
    project_service.set_project_notes(source_workspace, notes="Imported project")
    project_service.create_scenario(source_workspace, name="High Dose")
    project_service.attach_dataset(
        source_workspace,
        DatasetAsset(source_path=str(imported_branch_path), display_name="imported-branch.csv"),
    )
    project_service.set_scenario_notes(source_workspace, notes="Imported branch")

    project_snapshot = tmp_path / "dose-escalation.opkpd"
    snapshot_service = ProjectSnapshotService()
    snapshot_service.save_snapshot(
        snapshot_service.export_workspace_for_project(
            source_workspace,
            project_id=imported_project.project_id,
        ),
        project_snapshot,
    )

    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (
            str(project_snapshot),
            "OpenPKPD project snapshots (*.opkpd *.pkp *.zip)",
        ),
    )

    load_action.trigger()
    app.processEvents()

    assert len(project.projects) == 2
    assert project.active_project.name == "Dose Escalation"
    assert project.active_scenario.name == "High Dose"
    assert project.active_project.metadata["notes"] == "Imported project"
    assert project.active_scenario.metadata["notes"] == "Imported branch"
    assert Path(project.active_dataset.source_path or "").read_text(
        encoding="utf-8"
    ) == imported_branch_path.read_text(encoding="utf-8")
    assert data_path_input.text() == project.active_dataset.source_path
    assert project.recent_files == [str(workspace_path.resolve())]
    assert is_dirty() is True

    save_action.trigger()
    app.processEvents()

    assert is_dirty() is False
    workspace_payload = _snapshot_project_payload(workspace_path)
    assert len(workspace_payload["projects"]) == 2
    assert workspace_payload["projects"][1]["name"] == "Dose Escalation"
    assert (
        workspace_payload["projects"][1]["scenarios"][1]["metadata"]["notes"] == "Imported branch"
    )


@pytest.mark.unit
def test_workspace_menu_action_loads_scenario_snapshot_into_active_project_without_replacing_current_project_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    current_path = tmp_path / "current.csv"
    imported_baseline_path = tmp_path / "scenario-import-baseline.csv"
    imported_branch_path = tmp_path / "scenario-import-branch.csv"
    current_path.write_text("ID,TIME,DV\n1,0,1\n", encoding="utf-8")
    imported_baseline_path.write_text("ID,TIME,DV\n1,0,7\n", encoding="utf-8")
    imported_branch_path.write_text("ID,TIME,DV\n1,0,17\n", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )

    project = Workspace(name="Scenario Snapshot Import")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(current_path), display_name="current.csv"),
    )

    window = create_main_window(project)
    save_snapshot = getattr(window, "_save_project_snapshot", None)
    is_dirty = getattr(window, "_is_project_dirty", None)
    data_path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    load_action = window.findChild(qt_gui.QAction, "workspace-load-scenario-snapshot-action")
    save_action = window.findChild(qt_gui.QAction, "file-save-project-action")

    assert isinstance(save_snapshot, Callable)
    assert isinstance(is_dirty, Callable)
    assert data_path_input is not None
    assert load_action is not None
    assert save_action is not None

    workspace_path = tmp_path / "workspace-main.opkpd"
    assert save_snapshot(workspace_path)
    assert project.recent_files == [str(workspace_path.resolve())]
    assert is_dirty() is False

    source_workspace = project_service.new_workspace(
        name="Imported Workspace", root_path=str(tmp_path)
    )
    project_service.attach_dataset(
        source_workspace,
        DatasetAsset(
            source_path=str(imported_baseline_path), display_name="scenario-import-baseline.csv"
        ),
    )
    imported_scenario = project_service.create_scenario(source_workspace, name="Branch A")
    project_service.attach_dataset(
        source_workspace,
        DatasetAsset(
            source_path=str(imported_branch_path), display_name="scenario-import-branch.csv"
        ),
    )
    project_service.set_scenario_notes(source_workspace, notes="Imported scenario")

    scenario_snapshot = tmp_path / "branch-a.opkpd"
    snapshot_service = ProjectSnapshotService()
    snapshot_service.save_snapshot(
        snapshot_service.export_workspace_for_scenario(
            source_workspace, scenario_id=imported_scenario.scenario_id
        ),
        scenario_snapshot,
    )

    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (
            str(scenario_snapshot),
            "OpenPKPD project snapshots (*.opkpd *.pkp *.zip)",
        ),
    )

    load_action.trigger()
    app.processEvents()

    assert len(project.active_project.scenarios) == 2
    assert project.active_scenario.name == "Branch A"
    assert project.active_scenario.parent_scenario_id is None
    assert project.active_scenario.metadata["notes"] == "Imported scenario"
    assert Path(project.active_dataset.source_path or "").read_text(
        encoding="utf-8"
    ) == imported_branch_path.read_text(encoding="utf-8")
    assert data_path_input.text() == project.active_dataset.source_path
    assert project.recent_files == [str(workspace_path.resolve())]
    assert is_dirty() is True

    save_action.trigger()
    app.processEvents()

    assert is_dirty() is False
    workspace_payload = _snapshot_project_payload(workspace_path)
    assert len(workspace_payload["projects"]) == 1
    assert len(workspace_payload["projects"][0]["scenarios"]) == 2
    assert workspace_payload["projects"][0]["scenarios"][1]["name"] == "Branch A"
    assert workspace_payload["projects"][0]["scenarios"][1]["parent_scenario_id"] is None


@pytest.mark.unit
def test_tree_item_context_menu_exposes_item_appropriate_actions() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Tree context")
    ProjectService().create_project(project, name="Project B")
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    context_action_names = getattr(window, "_tree_context_menu_action_names", None)

    assert nav is not None
    assert callable(context_action_names)

    workspace_item = _find_tree_child(nav, "Workspace")
    assert workspace_item is not None
    project_item = _find_tree_child(workspace_item, "Project B")
    assert project_item is not None
    scenario_item = _find_tree_child(project_item, "Baseline")
    assert scenario_item is not None
    model_item = _find_scenario_workflow_item(scenario_item, "Model")
    assert model_item is not None
    results_item = _find_scenario_workflow_item(scenario_item, "Results")
    assert results_item is not None
    diagnostics_item = _find_scenario_workflow_item(scenario_item, "Diagnostics")
    assert diagnostics_item is not None

    assert context_action_names(workspace_item) == (
        "workspace-new-project-action",
        "separator",
        "workspace-load-project-snapshot-action",
    )
    assert context_action_names(project_item) == (
        "workspace-new-scenario-action",
        "separator",
        "workspace-duplicate-project-action",
        "workspace-save-project-snapshot-action",
        "workspace-load-scenario-snapshot-action",
        "separator",
        "workspace-rename-project-action",
        "workspace-edit-project-details-action",
        "separator",
        "workspace-delete-project-action",
    )
    assert context_action_names(scenario_item) == (
        "workspace-new-scenario-action",
        "separator",
        "workspace-duplicate-scenario-action",
        "workspace-save-scenario-snapshot-action",
        "workspace-load-scenario-snapshot-action",
        "separator",
        "workspace-rename-scenario-action",
        "workspace-edit-scenario-details-action",
        "separator",
        "workspace-delete-scenario-action",
    )
    assert context_action_names(model_item) == (
        "workspace-new-scenario-action",
        "separator",
        "workspace-duplicate-scenario-action",
        "workspace-save-scenario-snapshot-action",
        "workspace-load-scenario-snapshot-action",
        "separator",
        "inputs-open-control-stream-action",
        "separator",
        "workspace-rename-scenario-action",
        "workspace-edit-scenario-details-action",
        "separator",
        "workspace-delete-scenario-action",
    )
    assert context_action_names(results_item) == (
        "workspace-new-scenario-action",
        "separator",
        "workspace-duplicate-scenario-action",
        "workspace-save-scenario-snapshot-action",
        "workspace-load-scenario-snapshot-action",
        "separator",
        "results-open-latest-report-action",
        "results-save-latest-report-copy-action",
        "results-export-latest-report-pdf-action",
        "results-open-latest-plot-action",
        "results-save-latest-plot-copy-action",
        "results-open-artifact-folder-action",
        "separator",
        "workspace-rename-scenario-action",
        "workspace-edit-scenario-details-action",
        "separator",
        "workspace-delete-scenario-action",
    )
    assert context_action_names(diagnostics_item) == (
        "workspace-new-scenario-action",
        "separator",
        "workspace-duplicate-scenario-action",
        "workspace-save-scenario-snapshot-action",
        "workspace-load-scenario-snapshot-action",
        "separator",
        "diagnostics-open-gof-panel-action",
        "diagnostics-open-residual-trends-action",
        "diagnostics-open-artifact-folder-action",
        "separator",
        "workspace-rename-scenario-action",
        "workspace-edit-scenario-details-action",
        "separator",
        "workspace-delete-scenario-action",
    )


@pytest.mark.unit
def test_tree_item_context_menu_applies_scenario_action_to_clicked_workflow_branch(
    monkeypatch,
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Tree context")
    project_service = ProjectService()
    project_service.create_scenario(project, name="Variant A")
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    prepare_tree_context_item = getattr(window, "_prepare_tree_context_item", None)
    rename_scenario_action = window.findChild(qt_gui.QAction, "workspace-rename-scenario-action")

    assert nav is not None
    assert callable(prepare_tree_context_item)
    assert rename_scenario_action is not None

    _select_nav_path(nav, "Workspace", project.active_project.name, "Baseline", "Inputs", "Data")
    app.processEvents()
    assert project.active_scenario.name == "Baseline"

    workspace_item = _find_tree_child(nav, "Workspace")
    assert workspace_item is not None
    project_item = _find_tree_child(workspace_item, project.active_project.name)
    assert project_item is not None
    variant_scenario_item = _find_tree_child(project_item, "Variant A")
    assert variant_scenario_item is not None
    variant_model_item = _find_scenario_workflow_item(variant_scenario_item, "Model")
    assert variant_model_item is not None

    window._prompt_for_name_override = lambda **kwargs: "Variant B"  # type: ignore[attr-defined]

    assert prepare_tree_context_item(variant_model_item) is True
    rename_scenario_action.trigger()
    app.processEvents()

    assert project.active_scenario.name == "Variant B"
    assert nav.currentItem() is not None
    assert nav.currentItem().text(0) == "Model"

    workspace_item = _find_tree_child(nav, "Workspace")
    assert workspace_item is not None
    project_item = _find_tree_child(workspace_item, project.active_project.name)
    assert project_item is not None
    renamed_scenario_item = _find_tree_child(project_item, "Variant B")
    assert renamed_scenario_item is not None
    assert _find_scenario_workflow_item(renamed_scenario_item, "Model") is not None


@pytest.mark.unit
def test_results_workflow_previews_html_artifact(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    report_path = tmp_path / "report.html"
    report_path.write_text("<html><body><h1>Preview me</h1></body></html>", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Preview")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Preview fit finished")
    artifact = ArtifactRecord(
        kind="report",
        label="Preview report",
        path=str(report_path),
        source_run_id=run.run_id,
        metadata={"media_type": "text/html"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    preview_title = window.findChild(qt_widgets.QLabel, "results-artifact-preview-title")
    preview_browser = window.findChild(qt_widgets.QTextBrowser, "results-artifact-preview-browser")
    open_button = window.findChild(qt_widgets.QPushButton, "results-artifact-open-button")
    preview_toggle = window.findChild(
        qt_widgets.QToolButton, "results-artifact-preview-section-toggle"
    )

    assert nav is not None
    assert preview_title is not None
    assert preview_browser is not None
    assert open_button is not None
    assert preview_toggle is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()

    assert preview_toggle.isChecked() is True
    assert preview_title.text() == "Preview report"
    assert "Preview me" in preview_browser.toPlainText()
    assert open_button.isEnabled() is True


@pytest.mark.unit
def test_results_menu_action_opens_latest_report(tmp_path: Path, monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    report_path = tmp_path / "latest-report.html"
    report_path.write_text("<html><body>Latest report</body></html>", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Latest report")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
    artifact = ArtifactRecord(
        kind="report",
        label="Latest report",
        path=str(report_path),
        source_run_id=run.run_id,
        metadata={"media_type": "text/html"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)

    opened_paths: list[str] = []
    monkeypatch.setattr(
        qt_gui.QDesktopServices,
        "openUrl",
        lambda url: opened_paths.append(url.toLocalFile()) or True,
    )

    window = create_main_window(project)
    action = window.findChild(qt_gui.QAction, "results-open-latest-report-action")

    assert action is not None

    action.trigger()
    app.processEvents()

    assert opened_paths == [str(report_path)]


@pytest.mark.unit
def test_diagnostics_menu_action_opens_latest_gof_panel(tmp_path: Path, monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    gof_path = tmp_path / "gof-panel.png"
    gof_path.write_bytes(b"gof-panel")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Diagnostics menu")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
    artifact = ArtifactRecord(
        kind="plot",
        label="GOF panel",
        path=str(gof_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "plot", "media_type": "image/png", "plot_type": "gof_panel"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)

    opened_paths: list[str] = []
    monkeypatch.setattr(
        qt_gui.QDesktopServices,
        "openUrl",
        lambda url: opened_paths.append(url.toLocalFile()) or True,
    )

    window = create_main_window(project)
    action = window.findChild(qt_gui.QAction, "diagnostics-open-gof-panel-action")

    assert action is not None

    action.trigger()
    app.processEvents()

    assert opened_paths == [str(gof_path)]


@pytest.mark.unit
def test_results_workflow_shows_latest_report_summary_and_exports_copy(
    tmp_path: Path, monkeypatch
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    report_path = tmp_path / "report.html"
    report_path.write_text("<html><body>Export me</body></html>", encoding="utf-8")
    export_path = tmp_path / "report-copy.html"

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Export report")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
    artifact = ArtifactRecord(
        kind="report",
        label="HTML report",
        path=str(report_path),
        source_run_id=run.run_id,
        metadata={"media_type": "text/html", "artifact_role": "report"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)
    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_k: (str(export_path), "All files (*)"),
    )

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    report_summary_label = window.findChild(qt_widgets.QLabel, "results-report-summary-label")
    export_button = window.findChild(qt_gui.QAction, "results-artifact-export-button")
    open_latest_report_button = window.findChild(
        qt_gui.QAction, "results-open-latest-report-button"
    )

    assert nav is not None
    assert report_summary_label is not None
    assert export_button is not None
    assert open_latest_report_button is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()

    assert "Latest report" in report_summary_label.text()
    assert "HTML report" in report_summary_label.text()
    assert open_latest_report_button.isEnabled() is True
    assert export_button.isEnabled() is True

    export_button.trigger()
    app.processEvents()

    assert export_path.read_text(encoding="utf-8") == report_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_results_workflow_opens_selected_artifact_from_list_activation(
    tmp_path: Path, monkeypatch
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    artifact_path = tmp_path / "results-activation-report.html"
    artifact_path.write_text("<html><body>Activate me</body></html>", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Results activation")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
    artifact = ArtifactRecord(
        kind="report",
        label="Results activation report",
        path=str(artifact_path),
        source_run_id=run.run_id,
        metadata={"media_type": "text/html", "artifact_role": "report"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)

    opened_paths: list[str] = []
    monkeypatch.setattr(
        qt_gui.QDesktopServices,
        "openUrl",
        lambda url: opened_paths.append(url.toLocalFile()) or True,
    )

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    artifacts_list = window.findChild(qt_widgets.QListWidget, "results-artifacts-list")

    assert nav is not None
    assert artifacts_list is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()

    item = artifacts_list.item(0)
    assert item is not None

    artifacts_list.itemActivated.emit(item)
    app.processEvents()

    assert opened_paths == [str(artifact_path)]


@pytest.mark.unit
def test_results_menu_action_saves_latest_plot_copy(tmp_path: Path, monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    plot_path = tmp_path / "latest-plot.png"
    plot_path.write_bytes(b"not-a-real-png-but-good-enough-for-save-action")
    export_path = tmp_path / "latest-plot-copy.png"

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Latest plot copy")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
    artifact = ArtifactRecord(
        kind="plot",
        label="Latest plot",
        path=str(plot_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "plot", "media_type": "image/png", "plot_type": "gof_panel"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)
    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_k: (str(export_path), "All files (*)"),
    )

    window = create_main_window(project)
    action = window.findChild(qt_gui.QAction, "results-save-latest-plot-copy-action")

    assert action is not None

    action.trigger()
    app.processEvents()

    assert export_path.exists()
    assert export_path.read_bytes() == plot_path.read_bytes()


@pytest.mark.unit
def test_results_menu_action_saves_latest_report_copy(tmp_path: Path, monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    report_path = tmp_path / "latest-report.html"
    report_path.write_text("<html><body>latest report copy</body></html>", encoding="utf-8")
    export_path = tmp_path / "latest-report-copy.html"

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Latest report copy")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
    artifact = ArtifactRecord(
        kind="report",
        label="Latest report",
        path=str(report_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "report", "media_type": "text/html"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)
    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_k: (str(export_path), "All files (*)"),
    )

    window = create_main_window(project)
    action = window.findChild(qt_gui.QAction, "results-save-latest-report-copy-action")

    assert action is not None

    action.trigger()
    app.processEvents()

    assert export_path.exists()
    assert export_path.read_text(encoding="utf-8") == report_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_results_menu_action_exports_latest_report_pdf(tmp_path: Path, monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    report_path = tmp_path / "latest-report.html"
    report_path.write_text("<html><body>latest report pdf</body></html>", encoding="utf-8")
    export_path = tmp_path / "latest-report.pdf"

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Latest report PDF")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
    artifact = ArtifactRecord(
        kind="report",
        label="Latest report",
        path=str(report_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "report", "media_type": "text/html"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)
    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_k: (str(export_path), "PDF files (*.pdf)"),
    )

    export_calls: list[tuple[str, str]] = []

    def _fake_export(self, *, parent, source_path, destination_path, timeout_ms=30_000):
        export_calls.append((str(source_path), str(destination_path)))
        Path(destination_path).write_bytes(b"%PDF-1.4\n%fake\n")
        return True, None

    monkeypatch.setattr(
        "openpkpd_gui.services.report_export_service.ReportExportService.export_html_report_to_pdf",
        _fake_export,
    )

    window = create_main_window(project)
    action = window.findChild(qt_gui.QAction, "results-export-latest-report-pdf-action")

    assert action is not None

    action.trigger()
    app.processEvents()

    assert export_calls == [(str(report_path), str(export_path))]
    assert export_path.exists()
    assert export_path.read_bytes().startswith(b"%PDF-1.4")


@pytest.mark.unit
def test_results_workflow_button_exports_latest_report_pdf(tmp_path: Path, monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    report_path = tmp_path / "workflow-report.html"
    report_path.write_text("<html><body>workflow report pdf</body></html>", encoding="utf-8")
    export_path = tmp_path / "workflow-report.pdf"

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Workflow report PDF")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
    artifact = ArtifactRecord(
        kind="report",
        label="Workflow report",
        path=str(report_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "report", "media_type": "text/html"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)
    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_k: (str(export_path), "PDF files (*.pdf)"),
    )

    export_calls: list[tuple[str, str]] = []

    def _fake_export(self, *, parent, source_path, destination_path, timeout_ms=30_000):
        export_calls.append((str(source_path), str(destination_path)))
        Path(destination_path).write_bytes(b"%PDF-1.4\n%workflow\n")
        return True, None

    monkeypatch.setattr(
        "openpkpd_gui.services.report_export_service.ReportExportService.export_html_report_to_pdf",
        _fake_export,
    )

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    export_pdf_button = window.findChild(qt_gui.QAction, "results-export-latest-report-pdf-button")

    assert nav is not None
    assert export_pdf_button is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()

    assert export_pdf_button.isEnabled() is True

    export_pdf_button.trigger()
    app.processEvents()

    assert export_calls == [(str(report_path), str(export_path))]
    assert export_path.exists()
    assert export_path.read_bytes().startswith(b"%PDF-1.4")


@pytest.mark.unit
def test_shell_save_action_blocks_when_workflow_editor_is_dirty(monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    window = create_main_window(Workspace(name="Dirty save"))
    save_action = window.findChild(qt_gui.QAction, "file-save-project-action")
    data_path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    is_project_dirty = getattr(window, "_is_project_dirty", None)
    warning_messages: list[tuple[str, str]] = []

    assert save_action is not None
    assert data_path_input is not None
    assert callable(is_project_dirty)

    def _record_warning(*args, **_kwargs):
        warning_messages.append((args[1], args[2]))
        return qt_widgets.QMessageBox.StandardButton.Ok

    def _unexpected_save_dialog(*_args, **_kwargs):
        raise AssertionError("Save dialog should not open while editor changes are pending")

    monkeypatch.setattr(qt_widgets.QMessageBox, "warning", _record_warning)
    monkeypatch.setattr(qt_widgets.QFileDialog, "getSaveFileName", _unexpected_save_dialog)

    data_path_input.setText("/tmp/unsaved.csv")
    app.processEvents()
    save_action.trigger()
    app.processEvents()

    assert is_project_dirty() is True
    assert any(
        title == "Save Project Snapshot" and "Pending editor changes: Data" in message
        for title, message in warning_messages
    )


@pytest.mark.unit
def test_shell_open_snapshot_prompt_can_cancel_dirty_editor_discard(
    tmp_path: Path, monkeypatch
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    window = create_main_window(Workspace(name="Dirty open"))
    open_with_prompt = getattr(window, "_open_project_snapshot_with_prompt", None)
    model_title_input = window.findChild(qt_widgets.QLineEdit, "model-problem-title")
    warning_messages: list[tuple[str, str]] = []

    assert callable(open_with_prompt)
    assert model_title_input is not None

    def _record_warning(*args, **_kwargs):
        warning_messages.append((args[1], args[2]))
        return qt_widgets.QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(qt_widgets.QMessageBox, "warning", _record_warning)

    model_title_input.setText("Unsaved title")
    app.processEvents()

    assert open_with_prompt(tmp_path / "other-project.opkpd") is False
    assert any(title == "Unsaved workflow edits" for title, _message in warning_messages)


@pytest.mark.unit
def test_open_recent_missing_snapshot_is_removed_from_recent_projects(
    tmp_path: Path, monkeypatch
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    missing_snapshot = tmp_path / "missing.opkpd"
    existing_snapshot = tmp_path / "existing.opkpd"
    existing_snapshot.write_text("placeholder", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(
        name="Recent snapshots",
        recent_files=[str(missing_snapshot), str(existing_snapshot)],
    )
    info_messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        qt_widgets.QMessageBox,
        "information",
        lambda *args, **_kwargs: (
            info_messages.append((args[1], args[2])) or qt_widgets.QMessageBox.StandardButton.Ok
        ),
    )

    window = create_main_window(project)
    recent_action = window.findChild(qt_gui.QAction, "file-open-recent-action-0")

    assert recent_action is not None

    recent_action.trigger()
    app.processEvents()

    assert str(missing_snapshot.resolve()) not in project.recent_files
    assert any("no longer available" in message for _title, message in info_messages)


@pytest.mark.unit
def test_configure_application_style_applies_fusion_palette() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )

    configure_application_style(app)

    stylesheet = app.styleSheet()
    assert "QMainWindow" in stylesheet
    assert "QPushButton:hover:!disabled" in stylesheet
    assert "QPushButton:pressed:!disabled" in stylesheet
    assert "QPushButton:default:!disabled" in stylesheet
    assert 'QPushButton[primaryAction="true"]' in stylesheet
    assert "QToolButton:hover:!disabled" in stylesheet
    assert "QToolButton:pressed:!disabled" in stylesheet
    assert 'QToolButton[collapsibleHeader="true"]:checked' in stylesheet
    assert app.palette().color(qt_gui.QPalette.ColorRole.Window).name().lower() == "#f3f6fb"


@pytest.mark.unit
def test_semantic_state_stylesheet_includes_button_interaction_feedback() -> None:
    assert 'QPushButton[semanticState="ready"]:hover:!disabled' in SEMANTIC_STATE_STYLESHEET
    assert (
        'QPushButton[semanticState="results_available"]:pressed:!disabled'
        in SEMANTIC_STATE_STYLESHEET
    )


@pytest.mark.unit
def test_main_window_marks_primary_actions_distinct_from_utility_buttons() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    window = create_main_window(Workspace(name="Primary actions"))

    fit_run_button = window.findChild(qt_widgets.QPushButton, "fit-run-button")
    overview_next_action_button = window.findChild(
        qt_widgets.QPushButton, "overview-next-action-button"
    )
    results_refresh_button = window.findChild(qt_widgets.QToolButton, "results-review-menu-button")

    assert fit_run_button is not None
    assert overview_next_action_button is not None
    assert results_refresh_button is not None
    assert fit_run_button.property("primaryAction") is True
    assert overview_next_action_button.property("primaryAction") is True
    assert results_refresh_button.property("primaryAction") is not True


def _layout_item_names(layout) -> list[str]:
    names: list[str] = []
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget()
        names.append(widget.objectName() if widget is not None else "<stretch>")
    return names


@pytest.mark.unit
def test_action_rows_group_utility_buttons_before_primary_actions() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    window = create_main_window(Workspace(name="Action row ordering"))

    fit_action_row = window.findChild(qt_widgets.QWidget, "fit-action-row")
    nca_action_row = window.findChild(qt_widgets.QWidget, "nca-action-row")
    diagnostics_action_row = window.findChild(qt_widgets.QWidget, "diagnostics-action-row")
    advanced_vpc_actions_row = window.findChild(qt_widgets.QWidget, "advanced-vpc-actions-row")

    assert fit_action_row is not None
    assert nca_action_row is not None
    assert diagnostics_action_row is not None
    assert advanced_vpc_actions_row is not None
    assert _layout_item_names(fit_action_row.layout()) == [
        "<stretch>",
        "fit-run-progress",
        "fit-cancel-button",
        "fit-run-button",
    ]
    assert _layout_item_names(nca_action_row.layout()) == [
        "nca-open-latest-results-button",
        "nca-open-artifacts-folder-button",
        "<stretch>",
        "nca-run-progress",
        "nca-cancel-button",
        "nca-run-button",
    ]
    assert _layout_item_names(diagnostics_action_row.layout()) == [
        "diagnostics-view-dropdown-button",
        "diagnostics-output-dropdown-button",
        "<stretch>",
        "diagnostics-generate-npde-button",
    ]
    assert _layout_item_names(advanced_vpc_actions_row.layout()) == [
        "advanced-open-vpc-plot-button",
        "advanced-open-vpc-summary-button",
        "<stretch>",
        "advanced-vpc-progress",
        "advanced-cancel-vpc-button",
        "advanced-generate-vpc-button",
    ]


@pytest.mark.unit
def test_fit_workflow_refreshes_when_opened_after_data_and_model_save(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Smoke")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    fit_widget = window.findChild(qt_widgets.QWidget, "fit-workflow")
    fit_content_row = window.findChild(qt_widgets.QWidget, "fit-content-row")
    fit_action_row = window.findChild(qt_widgets.QWidget, "fit-action-row")
    fit_preparation_panel = window.findChild(qt_widgets.QWidget, "fit-preparation-panel")
    fit_run_panel = window.findChild(qt_widgets.QWidget, "fit-run-panel")
    preparation_label = window.findChild(qt_widgets.QLabel, "fit-preparation-summary")
    next_action_label = window.findChild(qt_widgets.QLabel, "fit-next-action-label")
    next_action_button = window.findChild(qt_widgets.QPushButton, "fit-next-action-button")
    validation_list = window.findChild(qt_widgets.QListWidget, "fit-validation-list")
    run_label = window.findChild(qt_widgets.QLabel, "fit-run-summary")
    run_button = window.findChild(qt_widgets.QPushButton, "fit-run-button")

    assert nav is not None
    assert fit_widget is not None
    assert fit_content_row is not None
    assert fit_action_row is not None
    assert fit_preparation_panel is not None
    assert fit_run_panel is not None
    assert preparation_label is not None
    assert next_action_label is not None
    assert next_action_button is not None
    assert validation_list is not None
    assert run_label is not None
    assert run_button is not None
    assert "Fit needs attention" in preparation_label.text()
    assert next_action_button.text() == "Open Data"
    assert next_action_label.text() == "Load a dataset in the Data workflow before starting a fit."
    assert run_label.text() == "No fit runs yet."
    assert run_button.isEnabled() is False

    project_service.attach_dataset(
        project, DatasetAsset(source_path=str(dataset_path), display_name="theo.csv")
    )
    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Smoke",
            dataset_path=str(dataset_path),
            pk_code="CL = THETA(1) * EXP(ETA(1))",
            error_code="Y = F * (1 + EPS(1))",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    app.processEvents()

    fit_widget._apply_responsive_layout(1200)
    app.processEvents()

    assert fit_content_row.orientation().name == "Horizontal"
    assert fit_action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert "Ready to start fit" in preparation_label.text()
    assert next_action_label.isHidden() is True
    assert next_action_button.isHidden() is True
    assert validation_list.item(0).text() == "Fit is ready to start."
    assert run_button.isEnabled() is True

    fit_run = RunRecord(
        workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Smoke fit finished"
    )
    project_service.add_run(project, fit_run)
    fit_widget._refresh_workflow()
    app.processEvents()

    assert run_label.text() == "Latest run — Succeeded • Smoke fit finished"
    assert next_action_button.text() == "Open Results"
    assert (
        next_action_label.text()
        == "A successful fit is already available. Review the latest outputs in Results."
    )
    assert run_button.isEnabled() is False

    fit_widget._apply_responsive_layout(760)
    app.processEvents()

    assert fit_content_row.orientation().name == "Vertical"
    assert fit_action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom

    fit_widget._apply_responsive_layout(1200)
    app.processEvents()

    assert fit_content_row.orientation().name == "Horizontal"
    assert fit_action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight


@pytest.mark.unit
def test_fit_next_action_button_navigates_to_data_model_and_results(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Fit next actions")
    project_service = ProjectService()
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    fit_button = window.findChild(qt_widgets.QPushButton, "fit-next-action-button")

    assert nav is not None
    assert stack is not None
    assert fit_button is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    app.processEvents()

    assert fit_button.text() == "Open Data"
    fit_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "data-workflow"

    project_service.attach_dataset(
        project, DatasetAsset(source_path=str(dataset_path), display_name="theo.csv")
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    app.processEvents()

    assert fit_button.text() == "Open Model"
    fit_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "model-workflow"

    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Smoke",
            dataset_path=str(dataset_path),
            pk_code="CL = THETA(1) * EXP(ETA(1))",
            error_code="Y = F * (1 + EPS(1))",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    app.processEvents()

    assert fit_button.isHidden() is True

    fit_run = RunRecord(workflow="fit")
    fit_run.mark_succeeded("Smoke fit finished")
    project_service.add_run(project, fit_run)
    fit_page = window.findChild(qt_widgets.QWidget, "fit-workflow")
    assert fit_page is not None
    fit_page._refresh_workflow()  # type: ignore[attr-defined]

    _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    app.processEvents()

    assert fit_button.text() == "Open Results"
    fit_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "results-workflow"


@pytest.mark.unit
def test_fit_validation_issue_activation_navigates_to_data_and_model_workflows() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Validation routing")
    window = create_main_window(project)
    window.show()
    window.activateWindow()
    app.processEvents()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    validation_list = window.findChild(qt_widgets.QListWidget, "fit-validation-list")
    data_path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    model_title_input = window.findChild(qt_widgets.QLineEdit, "model-problem-title")

    assert nav is not None
    assert stack is not None
    assert validation_list is not None
    assert data_path_input is not None
    assert model_title_input is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    app.processEvents()

    first_issue = validation_list.item(0)
    assert first_issue is not None

    validation_list.itemActivated.emit(first_issue)
    app.processEvents()

    assert stack.currentWidget().objectName() == "data-workflow"
    assert data_path_input.hasFocus() is True

    _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    app.processEvents()

    second_issue = validation_list.item(1)
    assert second_issue is not None

    validation_list.itemActivated.emit(second_issue)
    app.processEvents()

    assert stack.currentWidget().objectName() == "model-workflow"
    assert model_title_input.hasFocus() is True


@pytest.mark.unit
def test_fit_validation_issue_activation_focuses_model_translation_field(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Validation routing")
    window = create_main_window(project)
    window.show()
    window.activateWindow()
    app.processEvents()
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    validation_list = window.findChild(qt_widgets.QListWidget, "fit-validation-list")
    model_pk_edit = window.findChild(qt_widgets.QPlainTextEdit, "model-pk-code")

    assert nav is not None
    assert stack is not None
    assert validation_list is not None
    assert model_pk_edit is not None

    project_service.attach_dataset(
        project, DatasetAsset(source_path=str(dataset_path), display_name="theo.csv")
    )
    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Smoke",
            dataset_path=str(dataset_path),
            pk_code="",
            error_code="Y = F * (1 + EPS(1))",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    app.processEvents()

    pk_issue = next(
        (
            validation_list.item(index)
            for index in range(validation_list.count())
            if "Field: pk_code" in validation_list.item(index).toolTip()
        ),
        None,
    )
    assert pk_issue is not None

    validation_list.itemActivated.emit(pk_issue)
    app.processEvents()

    assert stack.currentWidget().objectName() == "model-workflow"
    assert model_pk_edit.hasFocus() is True


@pytest.mark.unit
def test_data_workflow_refreshes_when_opened_if_controls_are_pristine(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "external.csv"

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Smoke")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    data_page = window.findChild(qt_widgets.QWidget, "data-workflow")
    example_row = window.findChild(qt_widgets.QWidget, "data-example-row")
    import_row = window.findChild(qt_widgets.QWidget, "data-import-row")
    options_row = window.findChild(qt_widgets.QWidget, "data-options-row")
    content_row = window.findChild(qt_widgets.QWidget, "data-content-row")
    columns_panel = window.findChild(qt_widgets.QWidget, "data-columns-panel")
    preview_panel = window.findChild(qt_widgets.QWidget, "data-preview-panel")
    validation_panel = window.findChild(qt_widgets.QWidget, "data-validation-panel")
    path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    separator_input = window.findChild(qt_widgets.QLineEdit, "data-separator-input")
    ignore_char_input = window.findChild(qt_widgets.QLineEdit, "data-ignore-char-input")
    summary_label = window.findChild(qt_widgets.QLabel, "data-summary-label")
    next_action_label = window.findChild(qt_widgets.QLabel, "data-next-action-label")
    next_action_button = window.findChild(qt_widgets.QPushButton, "data-next-action-button")

    assert nav is not None
    assert data_page is not None
    assert example_row is not None
    assert import_row is not None
    assert options_row is not None
    assert content_row is not None
    assert columns_panel is not None
    assert preview_panel is not None
    assert validation_panel is not None
    assert path_input is not None
    assert separator_input is not None
    assert ignore_char_input is not None
    assert summary_label is not None
    assert next_action_label is not None
    assert next_action_button is not None

    project_service.attach_dataset(
        project,
        DatasetAsset(
            source_path=str(dataset_path),
            display_name="external.csv",
            separator=";",
            ignore_char="#",
            columns=["ID", "TIME", "DV"],
            row_count=2,
        ),
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Data"))
    app.processEvents()

    data_page._apply_responsive_layout(1280)
    app.processEvents()

    assert example_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert import_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert options_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert content_row.orientation().name == "Horizontal"
    assert path_input.text() == str(dataset_path)
    assert separator_input.text() == ";"
    assert ignore_char_input.text() == "#"
    assert "external.csv" in summary_label.text()
    assert next_action_button.text() == "Open Model"
    assert "open Model to configure one next" in next_action_label.text()

    data_page._apply_responsive_layout(760)
    app.processEvents()

    assert example_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert import_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert options_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert content_row.orientation().name == "Vertical"

    data_page._apply_responsive_layout(1280)
    app.processEvents()

    assert example_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert import_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert options_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert content_row.orientation().name == "Horizontal"


@pytest.mark.unit
def test_data_workflow_preserves_unsaved_inputs_when_opened(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "external.csv"

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Smoke")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    separator_input = window.findChild(qt_widgets.QLineEdit, "data-separator-input")
    unsaved_label = window.findChild(qt_widgets.QLabel, "data-unsaved-label")

    assert nav is not None
    assert path_input is not None
    assert separator_input is not None
    assert unsaved_label is not None

    path_input.setText("/tmp/unsaved.csv")
    separator_input.setText("|")
    app.processEvents()
    project_service.attach_dataset(
        project, DatasetAsset(source_path=str(dataset_path), display_name="external.csv")
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Data"))
    app.processEvents()

    assert path_input.text() == "/tmp/unsaved.csv"
    assert separator_input.text() == "|"
    assert unsaved_label.text() == "Unsaved data import changes"
    assert unsaved_label.isHidden() is False
    assert window.windowTitle() == "OpenPKPD — Smoke *"


@pytest.mark.unit
def test_data_workflow_clears_unsaved_indicator_after_load(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "external.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    window = create_main_window(Workspace(name="Smoke"))

    path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    unsaved_label = window.findChild(qt_widgets.QLabel, "data-unsaved-label")

    assert path_input is not None
    assert unsaved_label is not None

    path_input.setText(str(dataset_path))
    app.processEvents()

    assert unsaved_label.text() == "Unsaved data import changes"

    path_input.editingFinished.emit()
    app.processEvents()

    assert unsaved_label.text() == ""
    assert unsaved_label.isHidden() is True


@pytest.mark.unit
def test_results_workflow_refreshes_when_opened_after_run_and_artifact_added() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Smoke")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    results_page = window.findChild(qt_widgets.QWidget, "results-workflow")
    results_content_row = window.findChild(qt_widgets.QWidget, "results-content-row")
    results_filter_row = window.findChild(qt_widgets.QWidget, "results-artifact-filter-row")
    results_fit_review_row = window.findChild(qt_widgets.QToolButton, "results-review-menu-button")
    results_artifact_action_row = window.findChild(
        qt_widgets.QWidget, "results-artifact-action-row"
    )
    analysis_filter = window.findChild(qt_widgets.QComboBox, "results-analysis-filter")
    overview_label = window.findChild(qt_widgets.QLabel, "results-overview-label")
    runs_list = window.findChild(qt_widgets.QListWidget, "results-runs-list")
    artifacts_list = window.findChild(qt_widgets.QListWidget, "results-artifacts-list")
    artifact_summary_label = window.findChild(qt_widgets.QLabel, "results-artifact-summary-label")

    assert nav is not None
    assert results_page is not None
    assert results_content_row is not None
    assert results_filter_row is not None
    assert results_fit_review_row is not None
    assert results_artifact_action_row is not None
    assert analysis_filter is not None
    assert overview_label is not None
    assert runs_list is not None
    assert artifacts_list is not None
    assert artifact_summary_label is not None
    assert "0 review runs" in overview_label.text()

    fit_run = RunRecord(
        workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Smoke fit finished"
    )
    fit_artifact = ArtifactRecord(
        kind="report",
        label="Summary",
        path="summary.html",
        source_run_id=fit_run.run_id,
    )
    fit_run.artifact_ids.append(fit_artifact.artifact_id)
    nca_run = RunRecord(
        workflow="nca", status=RunStatus.SUCCEEDED, summary_text="Smoke NCA finished"
    )
    nca_artifact = ArtifactRecord(
        kind="table",
        label="NCA summary",
        path="nca.csv",
        source_run_id=nca_run.run_id,
        metadata={"artifact_role": "nca_summary", "media_type": "text/csv"},
    )
    nca_run.artifact_ids.append(nca_artifact.artifact_id)
    project_service.add_run(project, fit_run)
    project_service.add_run(project, nca_run)
    project_service.add_artifact(project, fit_artifact)
    project_service.add_artifact(project, nca_artifact)

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()
    results_page._apply_responsive_layout(1400)
    app.processEvents()

    assert results_content_row.orientation().name == "Horizontal"
    assert results_filter_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert results_fit_review_row is not None
    assert (
        results_artifact_action_row.layout().direction()
        == qt_widgets.QBoxLayout.Direction.LeftToRight
    )
    assert "2 review runs" in overview_label.text()
    assert "2 outputs" in overview_label.text()
    assert analysis_filter.itemText(0) == "All analyses"
    assert analysis_filter.findText("Fit") >= 0
    assert analysis_filter.findText("NCA") >= 0
    assert runs_list.item(0).text().startswith("NCA • Succeeded")
    assert "table: 1" in artifact_summary_label.text()

    analysis_filter.setCurrentText("Fit")
    app.processEvents()
    assert runs_list.count() == 1
    assert runs_list.item(0).text().startswith("Fit • Succeeded")
    assert "report" in artifacts_list.item(0).text().lower()
    assert "report: 1" in artifact_summary_label.text()

    analysis_filter.setCurrentText("NCA")
    app.processEvents()
    assert runs_list.count() == 1
    assert runs_list.item(0).text().startswith("NCA • Succeeded")
    assert "table" in artifacts_list.item(0).text().lower()

    results_page._apply_responsive_layout(800)
    app.processEvents()

    assert results_content_row.orientation().name == "Vertical"
    assert results_filter_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert results_fit_review_row is not None
    assert (
        results_artifact_action_row.layout().direction()
        == qt_widgets.QBoxLayout.Direction.TopToBottom
    )

    results_page._apply_responsive_layout(1400)
    app.processEvents()

    assert results_content_row.orientation().name == "Horizontal"
    assert results_filter_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert results_fit_review_row is not None
    assert (
        results_artifact_action_row.layout().direction()
        == qt_widgets.QBoxLayout.Direction.LeftToRight
    )


@pytest.mark.unit
def test_tree_navigation_switches_selected_scenario_context_across_workflows(
    tmp_path: Path,
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    baseline_path = tmp_path / "baseline.csv"
    baseline_path.write_text("ID,TIME,DV\n1,0,0\n1,1,5\n", encoding="utf-8")
    variant_path = tmp_path / "variant.csv"
    variant_path.write_text("ID,TIME,DV\n2,0,0\n2,1,7\n", encoding="utf-8")
    baseline_report = tmp_path / "baseline-report.html"
    baseline_report.write_text("<html><body>baseline</body></html>", encoding="utf-8")
    baseline_plot = tmp_path / "baseline-plot.png"
    baseline_plot.write_bytes(b"not-a-real-png-but-good-enough-for-selection")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Scenario switch smoke")
    project_service = ProjectService()

    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(baseline_path), display_name="baseline.csv", row_count=2),
    )
    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Baseline model",
            dataset_path=str(baseline_path),
            pk_code="CL = THETA(1)",
            error_code="Y = F",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )
    baseline_run = RunRecord(
        workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Baseline fit finished"
    )
    baseline_artifact = ArtifactRecord(
        kind="report",
        label="Baseline report",
        path=str(baseline_report),
        source_run_id=baseline_run.run_id,
        metadata={"artifact_role": "report", "media_type": "text/html"},
    )
    baseline_plot_artifact = ArtifactRecord(
        kind="plot",
        label="Baseline GOF",
        path=str(baseline_plot),
        source_run_id=baseline_run.run_id,
        metadata={"artifact_role": "plot", "media_type": "image/png", "plot_type": "gof_panel"},
    )
    baseline_run.artifact_ids.append(baseline_artifact.artifact_id)
    baseline_run.artifact_ids.append(baseline_plot_artifact.artifact_id)
    project_service.add_run(project, baseline_run)
    project_service.add_artifact(project, baseline_artifact)
    project_service.add_artifact(project, baseline_plot_artifact)

    project_service.create_scenario(project, name="Variant A")
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(variant_path), display_name="variant.csv", row_count=2),
    )
    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Variant model",
            dataset_path=str(variant_path),
            pk_code="V = THETA(1)",
            error_code="Y = F",
            theta_rows=[{"init": 2.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.2]],
            sigma_values=[[0.05]],
        ),
    )

    window = create_main_window(project)
    window.show()
    window.activateWindow()
    app.processEvents()
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    data_path_input = window.findChild(qt_widgets.QLineEdit, "data-source-path")
    model_title_input = window.findChild(qt_widgets.QLineEdit, "model-problem-title")
    results_overview = window.findChild(qt_widgets.QLabel, "results-overview-label")
    diagnostics_overview = window.findChild(qt_widgets.QLabel, "diagnostics-overview-label")

    assert nav is not None
    assert data_path_input is not None
    assert model_title_input is not None
    assert results_overview is not None
    assert diagnostics_overview is not None

    _select_nav_path(nav, "Workspace", project.active_project.name, "Baseline", "Inputs", "Data")
    app.processEvents()
    assert data_path_input.text() == str(baseline_path)

    _select_nav_path(nav, "Workspace", project.active_project.name, "Baseline", "Inputs", "Model")
    app.processEvents()
    assert model_title_input.text() == "Baseline model"

    _select_nav_path(nav, "Workspace", project.active_project.name, "Baseline", "Review", "Results")
    app.processEvents()
    assert "Scenario Baseline" in results_overview.text()
    assert "1 review runs" in results_overview.text()

    _select_nav_path(
        nav, "Workspace", project.active_project.name, "Baseline", "Review", "Diagnostics"
    )
    app.processEvents()
    assert "Scenario Baseline" in diagnostics_overview.text()
    assert "latest fit Succeeded" in diagnostics_overview.text()

    _select_nav_path(nav, "Workspace", project.active_project.name, "Variant A", "Inputs", "Data")
    app.processEvents()
    assert data_path_input.text() == str(variant_path)

    _select_nav_path(nav, "Workspace", project.active_project.name, "Variant A", "Inputs", "Model")
    app.processEvents()
    assert model_title_input.text() == "Variant model"

    _select_nav_path(
        nav, "Workspace", project.active_project.name, "Variant A", "Review", "Results"
    )
    app.processEvents()
    assert "Scenario Variant A" in results_overview.text()
    assert "0 review runs" in results_overview.text()

    _select_nav_path(
        nav, "Workspace", project.active_project.name, "Variant A", "Review", "Diagnostics"
    )
    app.processEvents()
    assert "Scenario Variant A" in diagnostics_overview.text()
    assert "latest fit No fit run" in diagnostics_overview.text()


@pytest.mark.unit
def test_results_workflow_opens_latest_plot_from_page(tmp_path: Path, monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    plot_path = tmp_path / "results-latest-plot.png"
    plot_path.write_bytes(b"not-a-real-png-but-good-enough-for-results-open")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Results plot button")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
    artifact = ArtifactRecord(
        kind="plot",
        label="Results latest plot",
        path=str(plot_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "plot", "media_type": "image/png", "plot_type": "gof_panel"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)

    opened_paths: list[str] = []
    monkeypatch.setattr(
        qt_gui.QDesktopServices,
        "openUrl",
        lambda url: opened_paths.append(url.toLocalFile()) or True,
    )

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    open_latest_plot_button = window.findChild(qt_gui.QAction, "results-open-latest-plot-button")

    assert nav is not None
    assert open_latest_plot_button is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()

    assert open_latest_plot_button.isEnabled() is True

    open_latest_plot_button.trigger()
    app.processEvents()

    assert opened_paths == [str(plot_path)]


@pytest.mark.unit
def test_results_workflow_saves_latest_plot_copy_from_page(tmp_path: Path, monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    plot_path = tmp_path / "results-page-latest-plot.png"
    plot_path.write_bytes(b"not-a-real-png-but-good-enough-for-results-export")
    export_path = tmp_path / "results-page-latest-plot-copy.png"

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Results plot export button")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
    artifact = ArtifactRecord(
        kind="plot",
        label="Results page latest plot",
        path=str(plot_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "plot", "media_type": "image/png", "plot_type": "gof_panel"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)
    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_k: (str(export_path), "All files (*)"),
    )

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    save_latest_plot_copy_button = window.findChild(
        qt_gui.QAction, "results-save-latest-plot-copy-button"
    )

    assert nav is not None
    assert save_latest_plot_copy_button is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()

    assert save_latest_plot_copy_button.isEnabled() is True

    save_latest_plot_copy_button.trigger()
    app.processEvents()

    assert export_path.exists()
    assert export_path.read_bytes() == plot_path.read_bytes()


@pytest.mark.unit
def test_results_workflow_opens_latest_bayesian_plot_from_review_menu(
    tmp_path: Path, monkeypatch
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    trace_path = tmp_path / "results-bayesian-trace.png"
    trace_path.write_bytes(b"not-a-real-png-but-good-enough-for-bayesian-open")
    older_plot_path = tmp_path / "results-older-gof.png"
    older_plot_path.write_bytes(b"older-gof")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Results Bayesian review")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Bayesian fit finished")
    gof_artifact = ArtifactRecord(
        kind="plot",
        label="Older GOF panel",
        path=str(older_plot_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "plot", "media_type": "image/png", "plot_type": "gof_panel"},
    )
    bayes_artifact = ArtifactRecord(
        kind="plot",
        label="Bayesian trace",
        path=str(trace_path),
        source_run_id=run.run_id,
        metadata={
            "artifact_role": "plot",
            "media_type": "image/png",
            "plot_type": "mcmc_trace_by_chain",
        },
    )
    posterior_table = ArtifactRecord(
        kind="table",
        label="Posterior summary",
        path=str(tmp_path / "posterior-summary.csv"),
        source_run_id=run.run_id,
        metadata={"artifact_role": "posterior_summary_table", "media_type": "text/csv"},
    )
    Path(posterior_table.path).write_text("parameter,mean\nTHETA(1),1.0\n", encoding="utf-8")
    for artifact in (gof_artifact, bayes_artifact, posterior_table):
        run.artifact_ids.append(artifact.artifact_id)
        project_service.add_artifact(project, artifact)
    project_service.add_run(project, run)

    opened_paths: list[str] = []
    monkeypatch.setattr(
        qt_gui.QDesktopServices,
        "openUrl",
        lambda url: opened_paths.append(url.toLocalFile()) or True,
    )

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    review_button = window.findChild(qt_widgets.QToolButton, "results-review-menu-button")
    bayesian_action = window.findChild(qt_gui.QAction, "results-open-bayesian-review-button")
    artifact_list = window.findChild(qt_widgets.QListWidget, "results-artifacts-list")

    assert nav is not None
    assert review_button is not None
    assert bayesian_action is not None
    assert artifact_list is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()

    assert review_button.isEnabled() is True
    assert bayesian_action.isEnabled() is True
    labels = [artifact_list.item(i).text() for i in range(artifact_list.count())]
    assert any("Bayesian trace  [plot]" == label for label in labels)
    assert any("Posterior summary  [table]" == label for label in labels)

    bayesian_action.trigger()
    app.processEvents()

    assert opened_paths == [str(trace_path)]


@pytest.mark.unit
def test_diagnostics_workflow_opens_selected_artifact_from_list_activation(
    tmp_path: Path, monkeypatch
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    artifact_path = tmp_path / "diagnostics-activation.png"
    artifact_path.write_bytes(b"not-a-real-png-but-good-enough-for-open-action")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Diagnostics activation")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)
    artifact = ArtifactRecord(
        kind="plot",
        label="Diagnostics activation plot",
        path=str(artifact_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "plot", "media_type": "image/png", "plot_type": "gof_panel"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)

    opened_paths: list[str] = []

    def fake_open_url(url) -> bool:
        opened_paths.append(url.toLocalFile())
        return True

    monkeypatch.setattr(qt_gui.QDesktopServices, "openUrl", fake_open_url)

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    artifacts_list = window.findChild(qt_widgets.QListWidget, "diagnostics-artifacts-list")

    assert nav is not None
    assert artifacts_list is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Diagnostics"))
    app.processEvents()

    item = artifacts_list.item(0)
    assert item is not None

    artifacts_list.itemActivated.emit(item)
    app.processEvents()

    assert opened_paths == [str(artifact_path)]


@pytest.mark.unit
def test_diagnostics_workflow_exports_selected_artifact_copy(tmp_path: Path, monkeypatch) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    artifact_path = tmp_path / "diagnostics-export.png"
    export_path = tmp_path / "diagnostics-export-copy.png"
    artifact_path.write_bytes(b"not-a-real-png-but-good-enough-for-export")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Diagnostics export")
    project_service = ProjectService()
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)
    artifact = ArtifactRecord(
        kind="plot",
        label="Diagnostics export plot",
        path=str(artifact_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "plot", "media_type": "image/png", "plot_type": "gof_panel"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)
    monkeypatch.setattr(
        qt_widgets.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_k: (str(export_path), "All files (*)"),
    )

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    export_action = window.findChild(qt_gui.QAction, "diagnostics-output-save-copy-action")

    assert nav is not None
    assert export_action is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Diagnostics"))
    app.processEvents()

    assert export_action.isEnabled() is True

    export_action.trigger()
    app.processEvents()

    assert export_path.exists()
    assert export_path.read_bytes() == artifact_path.read_bytes()


@pytest.mark.unit
def test_model_save_invalidates_stale_results_and_diagnostics(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "baseline.csv"
    report_path = tmp_path / "summary.html"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")
    report_path.write_text("<html><body>report</body></html>", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Smoke")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(source_path=str(dataset_path), display_name="baseline.csv", row_count=1),
    )
    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Baseline model",
            dataset_path=str(dataset_path),
            pk_code="CL = THETA(1)",
            error_code="Y = F",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )
    run = RunRecord(
        workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Baseline fit finished"
    )
    artifact = ArtifactRecord(
        kind="report",
        label="Summary",
        path=str(report_path),
        source_run_id=run.run_id,
        metadata={"artifact_role": "report", "media_type": "text/html"},
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)

    window = create_main_window(project)
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    problem_title_input = window.findChild(qt_widgets.QLineEdit, "model-problem-title")
    results_overview = window.findChild(qt_widgets.QLabel, "results-overview-label")
    results_artifact_summary = window.findChild(qt_widgets.QLabel, "results-artifact-summary-label")
    diagnostics_overview = window.findChild(qt_widgets.QLabel, "diagnostics-overview-label")
    diagnostics_next_steps = window.findChild(qt_widgets.QLabel, "diagnostics-next-steps-label")
    diagnostics_artifact_summary = window.findChild(
        qt_widgets.QLabel, "diagnostics-artifact-summary-label"
    )

    assert nav is not None
    assert problem_title_input is not None
    assert results_overview is not None
    assert results_artifact_summary is not None
    assert diagnostics_overview is not None
    assert diagnostics_next_steps is not None
    assert diagnostics_artifact_summary is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()
    assert "1 review runs" in results_overview.text()
    assert "1 outputs" in results_artifact_summary.text()

    _select_nav_path(nav, *_scenario_workflow_path(project, "Diagnostics"))
    app.processEvents()
    assert "latest fit Succeeded" in diagnostics_overview.text()
    assert "1 outputs" in diagnostics_artifact_summary.text()

    _select_nav_path(nav, *_scenario_workflow_path(project, "Model"))
    app.processEvents()
    problem_title_input.setText("Updated model")
    # Navigate away — auto-save triggers on leave
    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()

    assert project.active_scenario.runs == []
    assert project.active_scenario.artifacts == []

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()
    assert "0 review runs" in results_overview.text()
    assert "0 outputs" in results_artifact_summary.text()

    _select_nav_path(nav, *_scenario_workflow_path(project, "Diagnostics"))
    app.processEvents()
    assert "latest fit No fit run" in diagnostics_overview.text()
    assert "Run a fit in Fit to generate estimation outputs." in diagnostics_next_steps.text()
    assert "0 outputs" in diagnostics_artifact_summary.text()


@pytest.mark.unit
def test_fit_stale_warning_banners_appear_in_results_plots_and_diagnostics() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project_service = ProjectService()
    project = Workspace(name="Stale banners")

    project_service.attach_dataset(
        project,
        DatasetAsset(source_path="/tmp/data.csv", display_name="data.csv", row_count=2),
    )
    project_service.set_model_spec(
        project,
        ModelSpec(problem_title="Baseline", dataset_path="/tmp/data.csv", pk_code="CL = THETA(1)"),
    )

    fit_run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)
    fit_run.finished_at = "2026-03-15T12:00:00+00:00"
    project_service.add_run(project, fit_run)

    window = create_main_window(project)
    window.show()
    window.activateWindow()
    app.processEvents()
    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    results_warning = window.findChild(qt_widgets.QLabel, "results-stale-warning-label")
    diagnostics_warning = window.findChild(qt_widgets.QLabel, "diagnostics-stale-warning-label")

    assert nav is not None
    assert stack is not None
    assert results_warning is not None
    assert diagnostics_warning is not None

    project.active_model_spec = ModelSpec(
        problem_title="Updated",
        dataset_path="/tmp/data.csv",
        pk_code="V = THETA(2)",
    )
    project.active_scenario.model_updated_at = "2026-03-15T12:05:00+00:00"

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()
    assert stack.currentWidget().objectName() == "results-workflow"
    assert "Results may be stale" in results_warning.text()
    assert results_warning.isHidden() is False

    _select_nav_path(nav, *_scenario_workflow_path(project, "Diagnostics"))
    app.processEvents()
    assert stack.currentWidget().objectName() == "diagnostics-workflow"
    assert "Diagnostics may be stale" in diagnostics_warning.text()
    assert diagnostics_warning.isHidden() is False


@pytest.mark.unit
def test_model_workflow_refreshes_when_opened_if_editor_is_pristine() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Smoke")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    model_page = window.findChild(qt_widgets.QWidget, "model-workflow")
    content_row = window.findChild(qt_widgets.QWidget, "model-content-row")
    configuration_panel = window.findChild(qt_widgets.QWidget, "model-configuration-panel")
    translation_panel = window.findChild(qt_widgets.QWidget, "model-translation-panel")
    dataset_path_input = window.findChild(qt_widgets.QLineEdit, "model-dataset-path")
    summary_label = window.findChild(qt_widgets.QLabel, "model-summary-label")
    next_action_button = window.findChild(qt_widgets.QPushButton, "model-next-action-button")

    assert nav is not None
    assert model_page is not None
    assert content_row is not None
    assert configuration_panel is not None
    assert translation_panel is not None
    assert dataset_path_input is not None
    assert summary_label is not None
    assert next_action_button is not None

    project_service.attach_dataset(
        project, DatasetAsset(source_path="/tmp/from-data.csv", display_name="from-data.csv")
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Model"))
    app.processEvents()

    model_page._apply_responsive_layout(1280)
    app.processEvents()

    assert content_row.orientation().name == "Horizontal"
    assert dataset_path_input.text() == "/tmp/from-data.csv"
    assert "/tmp/from-data.csv" in summary_label.text()
    assert next_action_button.isHidden() is True

    model_page._apply_responsive_layout(760)
    app.processEvents()

    assert content_row.orientation().name == "Vertical"

    model_page._apply_responsive_layout(1280)
    app.processEvents()

    assert content_row.orientation().name == "Horizontal"


@pytest.mark.unit
def test_data_and_model_next_action_buttons_hand_off_between_workflows(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Smoke")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    data_page = window.findChild(qt_widgets.QWidget, "data-workflow")
    model_page = window.findChild(qt_widgets.QWidget, "model-workflow")
    data_button = window.findChild(qt_widgets.QPushButton, "data-next-action-button")
    model_button = window.findChild(qt_widgets.QPushButton, "model-next-action-button")

    assert nav is not None
    assert stack is not None
    assert data_page is not None
    assert model_page is not None
    assert data_button is not None
    assert model_button is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Model"))
    app.processEvents()

    assert model_button.text() == "Open Data"
    model_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "data-workflow"

    project_service.attach_dataset(
        project,
        DatasetAsset(
            source_path=str(dataset_path), display_name="theo.csv", columns=["ID", "TIME", "DV"]
        ),
    )
    data_page._refresh_workflow()  # type: ignore[attr-defined]
    model_page._refresh_workflow()  # type: ignore[attr-defined]
    app.processEvents()

    _select_nav_path(nav, *_scenario_workflow_path(project, "Data"))
    app.processEvents()

    assert data_button.text() == "Open Model"
    data_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "model-workflow"
    assert model_button.isHidden() is False
    assert model_button.text() == "Save model and open Fit"

    # Navigate away — auto-save triggers on leave
    _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    app.processEvents()
    _select_nav_path(nav, *_scenario_workflow_path(project, "Model"))
    app.processEvents()

    assert model_button.text() == "Save model and open Fit"
    model_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "fit-workflow"

    project_service.add_run(project, RunRecord(workflow="fit", status=RunStatus.SUCCEEDED))

    _select_nav_path(nav, *_scenario_workflow_path(project, "Data"))
    app.processEvents()

    assert data_button.text() == "Open Results"
    data_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "results-workflow"

    _select_nav_path(nav, *_scenario_workflow_path(project, "Model"))
    app.processEvents()

    assert model_button.text() == "Open Results"
    model_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "results-workflow"


@pytest.mark.unit
def test_model_workflow_preserves_unsaved_edits_when_opened() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Smoke")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    problem_title_input = window.findChild(qt_widgets.QLineEdit, "model-problem-title")
    dataset_path_input = window.findChild(qt_widgets.QLineEdit, "model-dataset-path")

    assert nav is not None
    assert problem_title_input is not None
    assert dataset_path_input is not None

    problem_title_input.setText("Unsaved title")
    app.processEvents()
    project_service.attach_dataset(
        project, DatasetAsset(source_path="/tmp/from-data.csv", display_name="from-data.csv")
    )

    # Navigate to Data and back — edits should survive (only the LEAVING page auto-saves, not the arriving page)
    _select_nav_path(nav, *_scenario_workflow_path(project, "Data"))
    app.processEvents()
    # Note: leaving Model auto-saved, so re-navigate to Model to verify it was committed
    _select_nav_path(nav, *_scenario_workflow_path(project, "Model"))
    app.processEvents()

    assert problem_title_input.text() == "Unsaved title"
    assert dataset_path_input.text() == ""
    assert window.windowTitle() == "OpenPKPD — Smoke *"


@pytest.mark.unit
def test_model_workflow_refreshes_after_save_when_project_model_changes() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Smoke")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    problem_title_input = window.findChild(qt_widgets.QLineEdit, "model-problem-title")
    dataset_path_input = window.findChild(qt_widgets.QLineEdit, "model-dataset-path")
    model_page = window.findChild(qt_widgets.QWidget, "model-workflow")

    assert nav is not None
    assert problem_title_input is not None
    assert dataset_path_input is not None
    assert model_page is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Model"))
    app.processEvents()

    problem_title_input.setText("Saved title")
    app.processEvents()
    assert model_page._has_unsaved_changes() is True  # type: ignore[attr-defined]
    # Navigate away — auto-save fires on leave
    _select_nav_path(nav, *_scenario_workflow_path(project, "Data"))
    app.processEvents()
    _select_nav_path(nav, *_scenario_workflow_path(project, "Model"))
    app.processEvents()
    assert model_page._has_unsaved_changes() is False  # type: ignore[attr-defined]

    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Externally changed",
            dataset_path="/tmp/other.csv",
            pk_code="CL = THETA(1)",
            error_code="Y = F",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Data"))
    _select_nav_path(nav, *_scenario_workflow_path(project, "Model"))
    app.processEvents()

    assert problem_title_input.text() == "Externally changed"
    assert dataset_path_input.text() == "/tmp/other.csv"


@pytest.mark.unit
def test_nca_workflow_refreshes_and_reflows_responsive_rows(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "nca.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n1,1,5\n2,0,0\n2,1,7\n", encoding="utf-8")
    results_path = tmp_path / "nca-results.csv"
    results_path.write_text("ID,CMAX,AUC\n1,5,12\n2,7,15\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="NCA")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    nca_page = window.findChild(qt_widgets.QWidget, "nca-workflow")
    options_row = window.findChild(qt_widgets.QWidget, "nca-options-row")
    content_row = window.findChild(qt_widgets.QWidget, "nca-content-row")
    action_row = window.findChild(qt_widgets.QWidget, "nca-action-row")
    readiness_panel = window.findChild(qt_widgets.QWidget, "nca-readiness-panel")
    results_panel = window.findChild(qt_widgets.QWidget, "nca-results-panel")
    preparation_label = window.findChild(qt_widgets.QLabel, "nca-preparation-summary")
    next_action_label = window.findChild(qt_widgets.QLabel, "nca-next-action-label")
    next_action_button = window.findChild(qt_widgets.QPushButton, "nca-next-action-button")
    run_label = window.findChild(qt_widgets.QLabel, "nca-run-summary")
    results_label = window.findChild(qt_widgets.QLabel, "nca-results-summary")
    route_combo = window.findChild(qt_widgets.QComboBox, "nca-route-combo")
    run_button = window.findChild(qt_widgets.QPushButton, "nca-run-button")
    open_results_button = window.findChild(qt_widgets.QPushButton, "nca-open-latest-results-button")

    assert nav is not None
    assert nca_page is not None
    assert options_row is not None
    assert content_row is not None
    assert action_row is not None
    assert readiness_panel is not None
    assert results_panel is not None
    assert preparation_label is not None
    assert next_action_label is not None
    assert next_action_button is not None
    assert run_label is not None
    assert results_label is not None
    assert route_combo is not None
    assert run_button is not None
    assert open_results_button is not None
    assert run_button.isEnabled() is False
    assert next_action_button.text() == "Open Data"
    assert next_action_label.text() == "Load a dataset in the Data workflow before starting NCA."

    project_service.attach_dataset(
        project,
        DatasetAsset(
            source_path=str(dataset_path),
            display_name="nca.csv",
            columns=["ID", "TIME", "DV"],
            row_count=4,
        ),
    )
    run = RunRecord(
        workflow="nca", status=RunStatus.SUCCEEDED, summary_text="2 subjects • 2 result rows"
    )
    artifact = ArtifactRecord(
        kind="table",
        label="NCA Summary",
        path=str(results_path),
        source_run_id=run.run_id,
        metadata={
            "artifact_role": "nca_summary",
            "subject_count": 2,
            "route": "oral",
            "auc_method": "linear-log",
            "min_points_lambda": 3,
            "exclude_cmax": True,
        },
    )
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)

    _select_nav_path(nav, *_scenario_workflow_path(project, "NCA"))
    app.processEvents()

    nca_page._apply_responsive_layout(1280)
    app.processEvents()

    assert options_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert content_row.orientation().name == "Horizontal"
    assert action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert "Ready to run NCA" in preparation_label.text()
    assert next_action_button.text() == "Open latest CSV"
    assert "already match the current options" in next_action_label.text()
    assert "Latest NCA — Succeeded" in run_label.text()
    assert "Latest results" in results_label.text()
    assert open_results_button.isEnabled() is True
    assert run_button.isEnabled() is False

    route_combo.setCurrentText("IV")
    app.processEvents()

    assert run_button.isEnabled() is True
    assert next_action_label.isHidden() is True
    assert next_action_button.isHidden() is True

    nca_page._apply_responsive_layout(760)
    app.processEvents()

    assert options_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert content_row.orientation().name == "Vertical"
    assert action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom

    nca_page._apply_responsive_layout(1280)
    app.processEvents()

    assert options_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert content_row.orientation().name == "Horizontal"
    assert action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight


@pytest.mark.unit
def test_covariate_workflow_refreshes_and_reflows_responsive_rows(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Covariate")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    covariate_page = window.findChild(qt_widgets.QWidget, "covariate-workflow")
    candidate_controls_row = window.findChild(
        qt_widgets.QWidget, "covariate-candidate-controls-row"
    )
    content_row = window.findChild(qt_widgets.QWidget, "covariate-content-row")
    action_row = window.findChild(qt_widgets.QWidget, "covariate-action-row")
    configuration_panel = window.findChild(qt_widgets.QWidget, "covariate-configuration-panel")
    results_panel = window.findChild(qt_widgets.QWidget, "covariate-results-panel")
    candidates_table = window.findChild(qt_widgets.QTableWidget, "covariate-candidates-table")
    status_label = window.findChild(qt_widgets.QLabel, "covariate-status-label")
    next_action_label = window.findChild(qt_widgets.QLabel, "covariate-next-action-label")
    next_action_button = window.findChild(qt_widgets.QPushButton, "covariate-next-action-button")
    add_candidate_button = window.findChild(
        qt_widgets.QPushButton, "covariate-add-candidate-button"
    )
    run_button = window.findChild(qt_widgets.QPushButton, "covariate-run-button")

    assert nav is not None
    assert covariate_page is not None
    assert candidate_controls_row is not None
    assert content_row is not None
    assert action_row is not None
    assert configuration_panel is not None
    assert results_panel is not None
    assert candidates_table is not None
    assert status_label is not None
    assert next_action_label is not None
    assert next_action_button is not None
    assert add_candidate_button is not None
    assert run_button is not None
    assert run_button.isEnabled() is False
    assert next_action_button.text() == "Open Data"

    add_candidate_button.click()
    app.processEvents()

    assert candidates_table.rowCount() == 1
    assert next_action_button.text() == "Open Data"

    project_service.attach_dataset(
        project, DatasetAsset(source_path=str(dataset_path), display_name="theo.csv")
    )
    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Smoke",
            dataset_path=str(dataset_path),
            pk_code="CL = THETA(1) * EXP(ETA(1))",
            error_code="Y = F * (1 + EPS(1))",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )
    covariate_run = RunRecord(
        workflow="covariate",
        status=RunStatus.SUCCEEDED,
        summary_text="Smoke • 1 accepted • base OFV=200.0 • final OFV=190.0",
    )
    project_service.add_run(project, covariate_run)

    _select_nav_path(nav, *_scenario_workflow_path(project, "Covariate"))
    app.processEvents()

    covariate_page._apply_responsive_layout(1200)
    app.processEvents()

    assert (
        candidate_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    )
    assert content_row.orientation().name == "Horizontal"
    assert action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert "Latest run — Succeeded" in status_label.text()
    assert "1 accepted" in status_label.text()
    assert next_action_label.isHidden() is True
    assert next_action_button.isHidden() is True
    assert run_button.isEnabled() is True

    covariate_page._apply_responsive_layout(760)
    app.processEvents()

    assert (
        candidate_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    )
    assert content_row.orientation().name == "Vertical"
    assert action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom

    covariate_page._apply_responsive_layout(1200)
    app.processEvents()

    assert (
        candidate_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    )
    assert content_row.orientation().name == "Horizontal"
    assert action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight


@pytest.mark.unit
def test_nca_and_covariate_next_action_buttons_navigate_or_act(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text(
        "ID,TIME,AMT,DV,EVID,WT\n1,0,100,0,1,70\n1,1,0,5,0,70\n", encoding="utf-8"
    )

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Secondary analyses")
    project_service = ProjectService()
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    nca_button = window.findChild(qt_widgets.QPushButton, "nca-next-action-button")
    covariate_button = window.findChild(qt_widgets.QPushButton, "covariate-next-action-button")
    candidates_table = window.findChild(qt_widgets.QTableWidget, "covariate-candidates-table")

    assert nav is not None
    assert stack is not None
    assert nca_button is not None
    assert covariate_button is not None
    assert candidates_table is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "NCA"))
    app.processEvents()

    assert nca_button.text() == "Open Data"
    nca_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "data-workflow"

    project_service.attach_dataset(
        project,
        DatasetAsset(
            source_path=str(dataset_path),
            display_name="theo.csv",
            columns=["ID", "TIME", "DV", "WT"],
        ),
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Covariate"))
    app.processEvents()

    assert covariate_button.text() == "Open Model"
    covariate_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "model-workflow"

    project_service.set_model_spec(
        project,
        ModelSpec(
            problem_title="Smoke",
            dataset_path=str(dataset_path),
            pk_code="CL = THETA(1) * EXP(ETA(1))",
            error_code="Y = F * (1 + EPS(1))",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0, "label": "CL"}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Covariate"))
    app.processEvents()

    assert covariate_button.text() == "Add candidate"
    covariate_button.click()
    app.processEvents()

    assert candidates_table.rowCount() == 1
    assert covariate_button.isHidden() is True


@pytest.mark.unit
def test_advanced_workflow_refreshes_and_reflows_responsive_rows(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    summary_path = tmp_path / "vpc-summary.html"
    summary_path.write_text("<html><body>vpc summary</body></html>", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Advanced")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    advanced_page = window.findChild(qt_widgets.QWidget, "advanced-workflow")
    tab_widget = window.findChild(qt_widgets.QTabWidget, "advanced-tab-widget")
    vpc_controls_row = window.findChild(qt_widgets.QWidget, "advanced-vpc-controls-row")
    vpc_actions_row = window.findChild(qt_widgets.QWidget, "advanced-vpc-actions-row")
    bootstrap_controls_row = window.findChild(qt_widgets.QWidget, "advanced-bootstrap-controls-row")
    bootstrap_actions_row = window.findChild(qt_widgets.QWidget, "advanced-bootstrap-actions-row")
    design_controls_row = window.findChild(qt_widgets.QWidget, "advanced-design-controls-row")
    design_actions_row = window.findChild(qt_widgets.QWidget, "advanced-design-actions-row")
    artifact_scope_row = window.findChild(qt_widgets.QWidget, "advanced-artifact-scope-row")
    artifact_content_row = window.findChild(qt_widgets.QWidget, "advanced-artifact-content-row")
    artifact_actions_row = window.findChild(qt_widgets.QWidget, "advanced-artifact-actions-row")
    artifact_list_panel = window.findChild(qt_widgets.QWidget, "advanced-artifact-list-panel")
    artifact_preview_panel = window.findChild(qt_widgets.QWidget, "advanced-artifact-preview-panel")
    vpc_settings_toggle = window.findChild(
        qt_widgets.QToolButton, "advanced-vpc-settings-section-toggle"
    )
    bootstrap_settings_toggle = window.findChild(
        qt_widgets.QToolButton, "advanced-bootstrap-settings-section-toggle"
    )
    design_settings_toggle = window.findChild(
        qt_widgets.QToolButton, "advanced-design-settings-section-toggle"
    )
    artifact_preview_toggle = window.findChild(
        qt_widgets.QToolButton, "advanced-artifact-preview-section-toggle"
    )
    artifact_scope_summary = window.findChild(qt_widgets.QLabel, "advanced-artifact-scope-summary")
    preview_title = window.findChild(qt_widgets.QLabel, "advanced-preview-title")
    vpc_next_action_label = window.findChild(qt_widgets.QLabel, "advanced-vpc-next-action-label")
    vpc_next_action_button = window.findChild(
        qt_widgets.QPushButton, "advanced-vpc-next-action-button"
    )
    bootstrap_next_action_button = window.findChild(
        qt_widgets.QPushButton, "advanced-bootstrap-next-action-button"
    )
    design_next_action_button = window.findChild(
        qt_widgets.QPushButton, "advanced-design-next-action-button"
    )
    open_vpc_summary_button = window.findChild(
        qt_widgets.QPushButton, "advanced-open-vpc-summary-button"
    )

    assert nav is not None
    assert advanced_page is not None
    assert tab_widget is not None
    assert vpc_controls_row is not None
    assert vpc_actions_row is not None
    assert bootstrap_controls_row is not None
    assert bootstrap_actions_row is not None
    assert design_controls_row is not None
    assert design_actions_row is not None
    assert artifact_scope_row is not None
    assert artifact_content_row is not None
    assert artifact_actions_row is not None
    assert artifact_list_panel is not None
    assert artifact_preview_panel is not None
    assert vpc_settings_toggle is not None
    assert bootstrap_settings_toggle is not None
    assert design_settings_toggle is not None
    assert artifact_preview_toggle is not None
    assert artifact_scope_summary is not None
    assert preview_title is not None
    assert vpc_next_action_label is not None
    assert vpc_next_action_button is not None
    assert bootstrap_next_action_button is not None
    assert design_next_action_button is not None
    assert open_vpc_summary_button is not None

    project_service.attach_dataset(
        project, DatasetAsset(source_path="/tmp/theo.csv", display_name="theo.csv")
    )
    project_service.set_model_spec(
        project, ModelSpec(problem_title="Smoke", dataset_path="/tmp/theo.csv")
    )
    fit_run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)
    artifact = ArtifactRecord(
        kind="report",
        label="VPC Summary",
        path=str(summary_path),
        source_run_id=fit_run.run_id,
        metadata={"artifact_role": "vpc_summary", "media_type": "text/html"},
    )
    fit_run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, fit_run)
    project_service.add_artifact(project, artifact)

    _select_nav_path(nav, *_scenario_workflow_path(project, "Advanced"))
    app.processEvents()

    advanced_page._apply_responsive_layout(1200)
    app.processEvents()

    assert vpc_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert vpc_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert (
        bootstrap_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    )
    assert bootstrap_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert design_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert design_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert artifact_scope_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert artifact_content_row.orientation().name == "Horizontal"
    assert artifact_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight

    tab_widget.setCurrentIndex(3)
    app.processEvents()

    assert "Showing 1 of 1" in artifact_scope_summary.text()
    assert preview_title.text() == "VPC Summary"
    assert vpc_next_action_button.text() == "Open latest VPC summary"
    assert "Latest VPC outputs are already available" in vpc_next_action_label.text()
    assert bootstrap_next_action_button.text() == "Open Fit"
    assert design_next_action_button.text() == "Open Fit"
    assert open_vpc_summary_button.isEnabled() is True

    advanced_page._apply_responsive_layout(760)
    app.processEvents()

    assert vpc_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert vpc_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert (
        bootstrap_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    )
    assert bootstrap_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert design_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert design_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert artifact_scope_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert artifact_content_row.orientation().name == "Vertical"
    assert artifact_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom

    advanced_page._apply_responsive_layout(1200)
    app.processEvents()

    assert vpc_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert vpc_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert (
        bootstrap_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    )
    assert bootstrap_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert design_controls_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert design_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert artifact_scope_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert artifact_content_row.orientation().name == "Horizontal"
    assert artifact_actions_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight


@pytest.mark.unit
def test_advanced_next_action_buttons_navigate_or_open_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    vpc_plot_path = tmp_path / "vpc-plot.png"
    vpc_plot_path.write_bytes(b"PNG")
    bootstrap_summary_path = tmp_path / "bootstrap-summary.csv"
    bootstrap_summary_path.write_text("parameter,mean\nTHETA(1),1.2\n", encoding="utf-8")
    design_summary_path = tmp_path / "design-summary.txt"
    design_summary_path.write_text("Design summary", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Advanced CTAs")
    window = create_main_window(project)
    project_service = ProjectService()
    opened_urls: list[str] = []
    monkeypatch.setattr(
        qt_gui.QDesktopServices, "openUrl", lambda url: opened_urls.append(url.toLocalFile())
    )

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    vpc_button = window.findChild(qt_widgets.QPushButton, "advanced-vpc-next-action-button")
    bootstrap_button = window.findChild(
        qt_widgets.QPushButton, "advanced-bootstrap-next-action-button"
    )
    design_button = window.findChild(qt_widgets.QPushButton, "advanced-design-next-action-button")

    assert nav is not None
    assert stack is not None
    assert vpc_button is not None
    assert bootstrap_button is not None
    assert design_button is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Advanced"))
    app.processEvents()

    assert vpc_button.text() == "Open Fit"
    assert bootstrap_button.text() == "Open Fit"
    assert design_button.text() == "Open Fit"

    vpc_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "fit-workflow"

    fit_run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)
    project_service.add_run(project, fit_run)
    project_service.add_artifact(
        project,
        ArtifactRecord(
            kind="plot",
            label="VPC Plot",
            path=str(vpc_plot_path),
            source_run_id=fit_run.run_id,
            metadata={"artifact_role": "plot", "plot_type": "vpc", "media_type": "image/png"},
        ),
    )
    project_service.add_artifact(
        project,
        ArtifactRecord(
            kind="table",
            label="Bootstrap Summary",
            path=str(bootstrap_summary_path),
            source_run_id=fit_run.run_id,
            metadata={"artifact_role": "bootstrap_summary", "media_type": "text/csv"},
        ),
    )
    project_service.add_artifact(
        project,
        ArtifactRecord(
            kind="report",
            label="Design Summary",
            path=str(design_summary_path),
            source_run_id=fit_run.run_id,
            metadata={"artifact_role": "design_summary", "media_type": "text/plain"},
        ),
    )

    _select_nav_path(nav, *_scenario_workflow_path(project, "Advanced"))
    app.processEvents()

    assert vpc_button.text() == "Open latest VPC plot"
    assert bootstrap_button.text() == "Open latest bootstrap summary"
    assert design_button.text() == "Open latest design summary"

    vpc_button.click()
    bootstrap_button.click()
    design_button.click()
    app.processEvents()

    assert opened_urls == [
        str(vpc_plot_path),
        str(bootstrap_summary_path),
        str(design_summary_path),
    ]


@pytest.mark.unit
def test_diagnostics_workflow_refreshes_when_opened_after_project_updates() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Smoke")
    window = create_main_window(project)
    project_service = ProjectService()

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    diagnostics_page = window.findChild(qt_widgets.QWidget, "diagnostics-workflow")
    diagnostics_filter_row = window.findChild(qt_widgets.QWidget, "diagnostics-filter-row")
    diagnostics_content_row = window.findChild(qt_widgets.QWidget, "diagnostics-content-row")
    diagnostics_action_row = window.findChild(qt_widgets.QWidget, "diagnostics-action-row")
    diagnostics_list_panel = window.findChild(qt_widgets.QWidget, "diagnostics-artifact-list-panel")
    diagnostics_preview_panel = window.findChild(
        qt_widgets.QWidget, "diagnostics-artifact-preview-panel"
    )
    overview_label = window.findChild(qt_widgets.QLabel, "diagnostics-overview-label")
    next_steps_label = window.findChild(qt_widgets.QLabel, "diagnostics-next-steps-label")
    artifact_summary_label = window.findChild(
        qt_widgets.QLabel, "diagnostics-artifact-summary-label"
    )

    assert nav is not None
    assert diagnostics_page is not None
    assert diagnostics_filter_row is not None
    assert diagnostics_content_row is not None
    assert diagnostics_action_row is not None
    assert diagnostics_list_panel is not None
    assert diagnostics_preview_panel is not None
    assert overview_label is not None
    assert next_steps_label is not None
    assert artifact_summary_label is not None
    assert "no dataset" in overview_label.text()

    project_service.attach_dataset(
        project, DatasetAsset(source_path="/tmp/theo.csv", display_name="theo.csv")
    )
    project_service.set_model_spec(
        project, ModelSpec(problem_title="Smoke", dataset_path="/tmp/theo.csv")
    )
    run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)
    artifact = ArtifactRecord(kind="plot", label="GOF", path="gof.png", source_run_id=run.run_id)
    run.artifact_ids.append(artifact.artifact_id)
    project_service.add_run(project, run)
    project_service.add_artifact(project, artifact)

    _select_nav_path(nav, *_scenario_workflow_path(project, "Diagnostics"))
    app.processEvents()

    diagnostics_page._apply_responsive_layout(1200)
    app.processEvents()

    assert (
        diagnostics_filter_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    )
    assert diagnostics_content_row.orientation().name == "Horizontal"
    assert (
        diagnostics_action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    )
    assert "dataset ready" in overview_label.text()
    assert "model ready" in overview_label.text()
    assert "latest fit Succeeded" in overview_label.text()
    assert "Review fit logs and outputs in Results." in next_steps_label.text()
    assert "plot: 1" in artifact_summary_label.text()

    diagnostics_page._apply_responsive_layout(760)
    app.processEvents()

    assert (
        diagnostics_filter_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    )
    assert diagnostics_content_row.orientation().name == "Vertical"
    assert (
        diagnostics_action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    )

    diagnostics_page._apply_responsive_layout(1200)
    app.processEvents()

    assert (
        diagnostics_filter_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    )
    assert diagnostics_content_row.orientation().name == "Horizontal"
    assert (
        diagnostics_action_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    )


@pytest.mark.unit
def test_review_workflow_next_action_buttons_navigate_between_pages() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Review next actions")
    project_service = ProjectService()
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    results_button = window.findChild(qt_widgets.QPushButton, "results-next-action-button")
    diagnostics_button = window.findChild(qt_widgets.QPushButton, "diagnostics-next-action-button")

    assert nav is not None
    assert stack is not None
    assert results_button is not None
    assert diagnostics_button is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Results"))
    app.processEvents()

    assert results_button.text() == "Open Data"
    results_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "data-workflow"

    project_service.attach_dataset(
        project, DatasetAsset(source_path="/tmp/theo.csv", display_name="theo.csv")
    )
    project_service.set_model_spec(
        project, ModelSpec(problem_title="Smoke", dataset_path="/tmp/theo.csv")
    )

    failed_run = RunRecord(workflow="fit", status=RunStatus.FAILED)
    project_service.add_run(project, failed_run)

    _select_nav_path(nav, *_scenario_workflow_path(project, "Diagnostics"))
    app.processEvents()

    assert diagnostics_button.text() == "Open Results"
    diagnostics_button.click()
    app.processEvents()
    assert stack.currentWidget().objectName() == "results-workflow"


@pytest.mark.unit
def test_fit_context_header_and_status_strip_refresh_after_run() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project_service = ProjectService()
    project = project_service.new_workspace(name="Shell Demo")
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    fit_page = window.findChild(qt_widgets.QWidget, "fit-workflow")
    assert nav is not None
    assert fit_page is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    app.processEvents()

    breadcrumb = fit_page.findChild(qt_widgets.QLabel, "fit-workflow-breadcrumb")
    state_label = fit_page.findChild(qt_widgets.QLabel, "fit-phase-label")
    results_chip = fit_page.findChild(qt_widgets.QLabel, "fit-workflow-status-chip-results")
    assert breadcrumb is not None
    assert state_label is not None
    assert results_chip is not None
    assert breadcrumb.text().endswith("/ Fit")
    assert state_label.text() == "Ready"

    project_service.attach_dataset(
        project, DatasetAsset(source_path="data.csv", display_name="Data")
    )
    project_service.set_model_spec(
        project, ModelSpec(problem_title="Baseline", pk_code="CL = THETA(1)")
    )
    run = RunRecord(workflow="fit")
    run.mark_succeeded("Finished")
    project_service.add_run(project, run)
    fit_page._refresh_workflow()
    app.processEvents()

    assert state_label.text() == "Completed"
    assert "Results available" in results_chip.text()


@pytest.mark.unit
def test_navigation_workflow_tooltips_include_state_summary() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project_service = ProjectService()
    project = project_service.new_workspace(name="Shell Demo")
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    assert nav is not None

    fit_item = _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    assert "Status: Not started" in fit_item.toolTip(0)

    project_service.attach_dataset(
        project, DatasetAsset(source_path="data.csv", display_name="Data")
    )
    project_service.set_model_spec(
        project, ModelSpec(problem_title="Baseline", pk_code="CL = THETA(1)")
    )
    run = RunRecord(workflow="fit")
    run.mark_succeeded("Finished")
    project_service.add_run(project, run)

    fit_page = window.findChild(qt_widgets.QWidget, "fit-workflow")
    assert fit_page is not None
    fit_page._project_state_changed()
    app.processEvents()

    fit_item = _select_nav_path(nav, *_scenario_workflow_path(project, "Fit"))
    assert "Status: Results available" in fit_item.toolTip(0)


@pytest.mark.unit
def test_overview_workflow_updates_next_step_and_latest_output_shortcuts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project_service = ProjectService()
    project = project_service.new_workspace(name="Overview Demo")
    window = create_main_window(project)

    nav = window.findChild(qt_widgets.QTreeWidget, "workflow-nav")
    stack = window.findChild(qt_widgets.QStackedWidget, "workflow-stack")
    overview_page = window.findChild(qt_widgets.QWidget, "overview-workflow")
    overview_scroll_area = window.findChild(qt_widgets.QScrollArea, "overview-scroll-area")
    overview_hero_panel = window.findChild(qt_widgets.QFrame, "overview-hero-panel")
    overview_eyebrow_label = window.findChild(qt_widgets.QLabel, "overview-eyebrow-label")
    overview_primary_row = window.findChild(qt_widgets.QWidget, "overview-primary-row")
    overview_secondary_row = window.findChild(qt_widgets.QWidget, "overview-secondary-row")
    overview_readiness_column = window.findChild(qt_widgets.QWidget, "overview-readiness-column")
    overview_workflows_column = window.findChild(qt_widgets.QWidget, "overview-workflows-column")
    overview_fit_card = window.findChild(qt_widgets.QFrame, "overview-readiness-card-fit")
    overview_fit_chip = window.findChild(qt_widgets.QLabel, "dashboard-workflow-status-chip-fit")
    overview_available_workflows_group = window.findChild(
        qt_widgets.QGroupBox,
        "overview-available-workflows-group",
    )
    overview_review_section = window.findChild(
        qt_widgets.QWidget, "overview-review-workflows-section"
    )
    overview_follow_up_section = window.findChild(qt_widgets.QWidget, "overview-follow-up-section")
    overview_additional_section = window.findChild(
        qt_widgets.QWidget, "overview-additional-workflows-section"
    )
    fit_heading_label = window.findChild(qt_widgets.QLabel, "overview-readiness-fit-heading")
    fit_icon_label = window.findChild(qt_widgets.QLabel, "overview-readiness-fit-icon")
    next_action_button = window.findChild(qt_widgets.QPushButton, "overview-next-action-button")
    next_action_label = window.findChild(qt_widgets.QLabel, "overview-next-action-label")
    fit_state_label = window.findChild(qt_widgets.QLabel, "overview-readiness-fit-state")
    nca_state_label = window.findChild(qt_widgets.QLabel, "overview-readiness-nca-state")
    covariate_state_label = window.findChild(
        qt_widgets.QLabel, "overview-readiness-covariate-state"
    )
    latest_output_summary = window.findChild(
        qt_widgets.QLabel, "overview-latest-output-summary-label"
    )
    latest_output_toggle = window.findChild(
        qt_widgets.QToolButton,
        "overview-latest-output-details-section-toggle",
    )
    review_summary = window.findChild(qt_widgets.QLabel, "overview-review-workflows-summary-label")
    follow_up_summary = window.findChild(qt_widgets.QLabel, "overview-follow-up-summary-label")
    additional_summary = window.findChild(
        qt_widgets.QLabel, "overview-additional-workflows-summary-label"
    )
    report_button = window.findChild(qt_widgets.QPushButton, "overview-open-latest-report-button")
    latest_plot_button = window.findChild(
        qt_widgets.QPushButton, "overview-open-latest-plot-button"
    )
    results_button = window.findChild(qt_widgets.QPushButton, "overview-open-results-button")
    plots_button = window.findChild(qt_widgets.QPushButton, "overview-open-plots-button")
    diagnostics_button = window.findChild(
        qt_widgets.QPushButton, "overview-open-diagnostics-button"
    )
    advanced_button = window.findChild(qt_widgets.QPushButton, "overview-open-advanced-button")
    nca_button = window.findChild(qt_widgets.QPushButton, "overview-open-nca-button")
    covariate_button = window.findChild(qt_widgets.QPushButton, "overview-open-covariate-button")

    assert nav is not None
    assert stack is not None
    assert overview_page is not None
    assert overview_scroll_area is not None
    assert overview_hero_panel is not None
    assert overview_eyebrow_label is not None
    assert overview_primary_row is not None
    assert overview_secondary_row is not None
    assert overview_readiness_column is not None
    assert overview_workflows_column is not None
    assert overview_fit_card is not None
    assert overview_fit_chip is not None
    assert overview_available_workflows_group is not None
    assert overview_review_section is not None
    assert overview_follow_up_section is not None
    assert overview_additional_section is not None
    assert fit_heading_label is not None
    assert fit_icon_label is not None
    assert next_action_button is not None
    assert next_action_label is not None
    assert fit_state_label is not None
    assert nca_state_label is not None
    assert covariate_state_label is not None
    assert latest_output_summary is not None
    assert latest_output_toggle is not None
    assert review_summary is not None
    assert follow_up_summary is not None
    assert additional_summary is not None
    assert report_button is not None
    assert latest_plot_button is not None
    assert results_button is not None
    assert plots_button is not None
    assert diagnostics_button is not None
    assert advanced_button is not None
    assert nca_button is not None
    assert covariate_button is not None

    _select_nav_path(nav, *_scenario_workflow_path(project, "Dashboard"))
    app.processEvents()

    assert stack.currentWidget().objectName() == "dashboard-workflow"
    assert overview_scroll_area.widget() is not None
    overview_page._apply_responsive_layout(1200)
    app.processEvents()
    assert "Scenario dashboard" in overview_eyebrow_label.text()
    assert overview_eyebrow_label.property("semanticRole") == "eyebrow"
    assert fit_heading_label.text() == "Fit"
    assert fit_icon_label.text() == "∑"
    assert overview_primary_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert (
        overview_secondary_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    )
    assert next_action_button.text() == "Open Data"
    assert "Import or load a dataset" in next_action_label.text()
    assert fit_state_label.text() == "Not started"
    assert nca_state_label.text() == "Not started"
    assert covariate_state_label.text() == "Not started"
    assert latest_output_toggle.isChecked() is False
    assert report_button.isEnabled() is False
    assert latest_plot_button.isEnabled() is False
    assert results_button.isEnabled() is False
    assert plots_button.isEnabled() is False
    assert diagnostics_button.isEnabled() is False
    assert advanced_button.isEnabled() is False
    assert nca_button.isEnabled() is False
    assert covariate_button.isEnabled() is False
    assert results_button.text() == "Open Results (Not started)"
    assert plots_button.text() == "Open Plots (Not started)"
    assert diagnostics_button.text() == "Open Diagnostics (Not started)"
    assert advanced_button.text() == "Open Advanced (Not started)"
    assert nca_button.text() == "Open NCA (Not started)"
    assert covariate_button.text() == "Open Covariate (Not started)"
    assert fit_state_label.property("semanticState") == "not_started"
    assert overview_fit_card.property("semanticState") == "not_started"
    assert overview_fit_chip.property("semanticState") == "not_started"
    assert results_button.property("semanticState") == "not_started"
    assert "Status: Not started" in results_button.toolTip()
    assert "Status: Not started" in plots_button.toolTip()
    assert "Status: Not started" in diagnostics_button.toolTip()
    assert "Status: Not started" in advanced_button.toolTip()
    assert "Status: Not started" in nca_button.toolTip()
    assert "Status: Not started" in covariate_button.toolTip()
    assert "unlock after a successful fit begins producing review outputs" in review_summary.text()
    assert "unlock after a successful fit" in follow_up_summary.text()
    assert "unlock after the scenario has saved inputs" in additional_summary.text()

    overview_page._apply_responsive_layout(640)
    app.processEvents()

    assert overview_primary_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    assert (
        overview_secondary_row.layout().direction() == qt_widgets.QBoxLayout.Direction.TopToBottom
    )

    overview_page._apply_responsive_layout(1200)
    app.processEvents()

    assert overview_primary_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    assert (
        overview_secondary_row.layout().direction() == qt_widgets.QBoxLayout.Direction.LeftToRight
    )

    report_path = tmp_path / "latest-report.html"
    plot_path = tmp_path / "latest-plot.png"
    report_path.write_text("<html>report</html>", encoding="utf-8")
    plot_path.write_text("plot", encoding="utf-8")

    project_service.attach_dataset(
        project, DatasetAsset(source_path="data.csv", display_name="Data")
    )
    overview_page._refresh_workflow()
    app.processEvents()

    assert nca_state_label.text() == "Ready"
    assert covariate_state_label.text() == "Needs attention"
    assert results_button.isEnabled() is False
    assert plots_button.isEnabled() is False
    assert nca_button.isEnabled() is True
    assert covariate_button.isEnabled() is True
    assert results_button.text() == "Open Results (Not started)"
    assert plots_button.text() == "Open Plots (Not started)"
    assert nca_button.text() == "Open NCA (Ready)"
    assert covariate_button.text() == "Open Covariate (Needs attention)"
    assert nca_state_label.property("semanticState") == "ready"
    assert covariate_state_label.property("semanticState") == "needs_attention"
    assert nca_button.property("semanticState") == "ready"
    assert covariate_button.property("semanticState") == "needs_attention"
    assert "Status: Ready" in nca_button.toolTip()
    assert "Status: Needs attention" in covariate_button.toolTip()
    assert "NCA: Ready" in additional_summary.text()
    assert "Covariate: Needs attention" in additional_summary.text()

    project_service.set_model_spec(
        project, ModelSpec(problem_title="Baseline", pk_code="CL = THETA(1)")
    )
    overview_page._refresh_workflow()
    app.processEvents()

    assert covariate_state_label.text() == "Ready"
    assert results_button.isEnabled() is True
    assert plots_button.isEnabled() is False
    assert results_button.text() == "Open Results (Ready)"
    assert plots_button.text() == "Open Plots (Not started)"
    assert results_button.property("semanticState") == "ready"
    assert plots_button.property("semanticState") == "not_started"
    assert "Status: Ready" in results_button.toolTip()
    assert "Status: Not started" in plots_button.toolTip()
    assert "Results: Ready" in review_summary.text()
    assert "Plots: Not started" in review_summary.text()

    run = RunRecord(workflow="fit")
    run.mark_succeeded("Finished")
    report = ArtifactRecord(
        kind="report",
        label="HTML report",
        path=str(report_path),
        source_run_id=run.run_id,
    )
    plot = ArtifactRecord(
        kind="plot",
        label="DV vs PRED",
        path=str(plot_path),
        source_run_id=run.run_id,
    )
    run.artifact_ids.extend([report.artifact_id, plot.artifact_id])
    project_service.add_run(project, run)
    project_service.add_artifact(project, report)
    project_service.add_artifact(project, plot)

    opened_paths: list[str] = []
    monkeypatch.setattr(
        qt_gui.QDesktopServices,
        "openUrl",
        lambda url: opened_paths.append(url.toLocalFile()) or True,
    )

    overview_page._refresh_workflow()
    app.processEvents()

    assert next_action_button.text() == "Open Results"
    assert "Review the latest reports" in next_action_label.text()
    assert fit_state_label.text() == "Results available"
    assert "Latest outputs ready" in latest_output_summary.text()
    assert report_button.isEnabled() is True
    assert latest_plot_button.isEnabled() is True
    assert results_button.isEnabled() is True
    assert plots_button.isEnabled() is True
    assert diagnostics_button.isEnabled() is True
    assert advanced_button.isEnabled() is True
    assert nca_button.isEnabled() is True
    assert covariate_button.isEnabled() is True
    assert results_button.text() == "Open Results (Results available)"
    assert plots_button.text() == "Open Plots (Results available)"
    assert diagnostics_button.text() == "Open Diagnostics (Ready)"
    assert advanced_button.text() == "Open Advanced (Ready)"
    assert nca_button.text() == "Open NCA (Ready)"
    assert covariate_button.text() == "Open Covariate (Ready)"
    assert fit_state_label.property("semanticState") == "results_available"
    assert overview_fit_card.property("semanticState") == "results_available"
    assert overview_fit_chip.property("semanticState") == "results_available"
    assert results_button.property("semanticState") == "results_available"
    assert plots_button.property("semanticState") == "results_available"
    assert diagnostics_button.property("semanticState") == "ready"
    assert "Status: Results available" in results_button.toolTip()
    assert "Status: Results available" in plots_button.toolTip()
    assert "Status: Ready" in diagnostics_button.toolTip()
    assert "Status: Ready" in advanced_button.toolTip()
    assert "Status: Ready" in nca_button.toolTip()
    assert "Status: Ready" in covariate_button.toolTip()
    assert "Results: Results available" in review_summary.text()
    assert "Plots: Results available" in review_summary.text()
    assert "Diagnostics: Ready" in follow_up_summary.text()
    assert "Advanced: Ready" in follow_up_summary.text()
    assert "NCA: Ready" in additional_summary.text()
    assert "Covariate: Ready" in additional_summary.text()

    report_button.click()
    latest_plot_button.click()
    app.processEvents()

    assert opened_paths == [str(report_path), str(plot_path)]

    next_action_button.click()
    app.processEvents()

    assert stack.currentWidget().objectName() == "results-workflow"

    _select_nav_path(nav, *_scenario_workflow_path(project, "Dashboard"))
    app.processEvents()
    plots_button.click()
    app.processEvents()

    assert stack.currentWidget().objectName() == "results-workflow"

    _select_nav_path(nav, *_scenario_workflow_path(project, "Dashboard"))
    app.processEvents()
    diagnostics_button.click()
    app.processEvents()

    assert stack.currentWidget().objectName() == "diagnostics-workflow"

    _select_nav_path(nav, *_scenario_workflow_path(project, "Dashboard"))
    app.processEvents()
    advanced_button.click()
    app.processEvents()

    assert stack.currentWidget().objectName() == "advanced-workflow"

    _select_nav_path(nav, *_scenario_workflow_path(project, "Dashboard"))
    app.processEvents()
    nca_button.click()
    app.processEvents()

    assert stack.currentWidget().objectName() == "nca-workflow"

    _select_nav_path(nav, *_scenario_workflow_path(project, "Dashboard"))
    app.processEvents()
    covariate_button.click()
    app.processEvents()

    assert stack.currentWidget().objectName() == "covariate-workflow"
