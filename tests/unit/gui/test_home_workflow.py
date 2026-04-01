"""Tests for the Home landing workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.workflows.home_workflow import (
    build_home_workflow,
    format_home_active_context,
    format_home_workspace_summary,
)


def test_format_home_workspace_summary_includes_snapshot_recent_and_dirty_state() -> None:
    workspace = Workspace(name="Home")
    workspace.recent_files = ["/tmp/one.opkpd", "/tmp/two.opkpd"]

    summary = format_home_workspace_summary(
        workspace,
        snapshot_path="/tmp/current.opkpd",
        is_dirty=True,
    )

    assert "1 project • 1 scenario • 2 recent snapshots" in summary
    assert "Unsaved changes pending" in summary
    assert "current.opkpd" in summary


def test_format_home_active_context_reports_active_project_and_workflow_states(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    workspace = Workspace(name="Home")
    ProjectService().attach_dataset(
        workspace,
        DatasetAsset(source_path=str(dataset_path), display_name="theo.csv"),
    )

    context = format_home_active_context(workspace)

    assert "Active project: Project 1" in context
    assert "Active scenario: Baseline" in context
    assert "Data: Ready" in context
    assert "Model: Needs attention" in context


@pytest.mark.unit
def test_home_workflow_refreshes_actions_and_recent_snapshot_buttons(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    recent_snapshot = tmp_path / "recent.opkpd"
    recent_snapshot.write_text("placeholder", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    workspace = Workspace(name="Home")
    workspace.recent_files = [str(recent_snapshot)]
    widget = build_home_workflow(workspace)
    navigations: list[str] = []
    opens: list[str] = []
    created = [0]

    widget._navigate_to_workflow = lambda workflow_id: navigations.append(workflow_id)  # type: ignore[attr-defined]
    widget._open_recent_snapshot = lambda path: opens.append(path)  # type: ignore[attr-defined]
    widget._choose_project_snapshot_to_open = lambda: opens.append("choose")  # type: ignore[attr-defined]
    widget._create_project = lambda: created.__setitem__(0, created[0] + 1)  # type: ignore[attr-defined]
    widget._current_snapshot_path = lambda: str(tmp_path / "current.opkpd")  # type: ignore[attr-defined]
    widget._project_dirty = lambda: True  # type: ignore[attr-defined]

    try:
        widget.show()
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        summary_label = widget.findChild(qt_widgets.QLabel, "home-workspace-summary-label")
        next_action_button = widget.findChild(qt_widgets.QPushButton, "home-next-action-button")
        new_project_button = widget.findChild(qt_widgets.QPushButton, "home-new-project-button")
        open_project_button = widget.findChild(qt_widgets.QPushButton, "home-open-project-button")
        open_overview_button = widget.findChild(qt_widgets.QPushButton, "home-open-overview-button")
        recent_button = widget.findChild(qt_widgets.QPushButton, "home-recent-snapshot-button-0")

        assert summary_label is not None
        assert next_action_button is not None
        assert new_project_button is not None
        assert open_project_button is not None
        assert open_overview_button is not None
        assert recent_button is not None
        assert "Unsaved changes pending" in summary_label.text()
        assert next_action_button.text() == "Open Data"
        assert recent_button.isHidden() is False

        new_project_button.click()
        open_project_button.click()
        next_action_button.click()
        open_overview_button.click()
        recent_button.click()
        app.processEvents()

        assert created[0] == 1
        assert opens == ["choose", str(recent_snapshot.resolve())]
        assert navigations == ["data", "dashboard"]
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# P3-E: Scenario cloning / branching UI
# ---------------------------------------------------------------------------


def test_home_workflow_has_duplicate_scenario_button() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    workspace = Workspace(name="Home")
    widget = build_home_workflow(workspace)
    try:
        btn = widget.findChild(qt_widgets.QPushButton, "home-duplicate-scenario-button")
        assert btn is not None, "home-duplicate-scenario-button not found"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_home_workflow_duplicate_scenario_button_calls_callback() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    workspace = Workspace(name="Home")
    widget = build_home_workflow(workspace)
    duplicated = [0]
    widget._duplicate_scenario = lambda: duplicated.__setitem__(0, duplicated[0] + 1)  # type: ignore[attr-defined]
    try:
        widget.show()
        app.processEvents()
        btn = widget.findChild(qt_widgets.QPushButton, "home-duplicate-scenario-button")
        assert btn is not None
        btn.click()
        app.processEvents()
        assert duplicated[0] == 1, "Duplicate callback not triggered"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
