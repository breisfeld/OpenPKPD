"""Pure tests for covariate workflow presentation helpers and SCM service."""

from __future__ import annotations

from pathlib import Path

import pytest

from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec, ModelSpecMode
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import JobStatus
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.scm_service import SCMCandidate, SCMRunResult, SCMService
from openpkpd_gui.services.workflow_state_service import WorkflowStateId, workflow_state_for
from openpkpd_gui.workflows.covariate_workflow import (
    build_covariate_workflow,
    covariate_blocking_message,
    format_scm_result_summary,
    format_scm_step_summary,
    recommend_covariate_next_action,
)

# ── format_scm_step_summary ────────────────────────────────────────────────────


class TestFormatScmStepSummary:
    def _row(self, **kwargs):
        defaults = {
            "type": "forward",
            "rel": "CL~WT(power)",
            "delta_ofv": -5.2,
            "p_value": 0.023,
            "accepted": True,
        }
        defaults.update(kwargs)
        return defaults

    def test_accepted_shows_accepted(self):
        summary = format_scm_step_summary(self._row(accepted=True))
        assert "ACCEPTED" in summary

    def test_rejected_shows_rejected(self):
        summary = format_scm_step_summary(self._row(accepted=False))
        assert "rejected" in summary

    def test_step_type_uppercased(self):
        summary = format_scm_step_summary(self._row(type="forward"))
        assert "[FORWARD]" in summary

    def test_backward_step_type(self):
        summary = format_scm_step_summary(self._row(type="backward"))
        assert "[BACKWARD]" in summary

    def test_relationship_included(self):
        summary = format_scm_step_summary(self._row(rel="V~AGE(linear)"))
        assert "V~AGE(linear)" in summary

    def test_delta_ofv_signed(self):
        summary = format_scm_step_summary(self._row(delta_ofv=-3.14))
        assert "-3.140" in summary

    def test_p_value_included(self):
        summary = format_scm_step_summary(self._row(p_value=0.0315))
        assert "0.0315" in summary

    def test_missing_keys_do_not_raise(self):
        summary = format_scm_step_summary({})
        assert isinstance(summary, str)


# ── format_scm_result_summary ──────────────────────────────────────────────────


class TestFormatScmResultSummary:
    def test_no_run_and_no_result_returns_default(self):
        summary = format_scm_result_summary(None, None)
        assert "No SCM runs yet" in summary

    def test_result_summary_text_preferred(self):
        result = SCMRunResult(
            summary_text="Demo • 2 accepted • base OFV=200.0 • final OFV=185.0",
            step_rows=[],
            accepted_count=2,
            final_ofv=185.0,
            base_ofv=200.0,
        )
        summary = format_scm_result_summary(result, None)
        assert "2 accepted" in summary
        assert "185" in summary

    def test_succeeded_run_shows_summary_text(self):
        run = RunRecord(
            workflow="covariate",
            status=RunStatus.SUCCEEDED,
            summary_text="Demo • FOCE done",
        )
        summary = format_scm_result_summary(None, run)
        assert "Succeeded" in summary
        assert "Demo" in summary

    def test_failed_run_shows_error(self):
        run = RunRecord(workflow="covariate", status=RunStatus.FAILED, error_text="No builder")
        summary = format_scm_result_summary(None, run)
        assert "Failed" in summary
        assert "No builder" in summary

    def test_running_run_shows_status(self):
        run = RunRecord(workflow="covariate", status=RunStatus.RUNNING)
        summary = format_scm_result_summary(None, run)
        assert "Running" in summary or "running" in summary.lower()


def test_recommend_covariate_next_action_tracks_blocked_and_empty_states():
    workspace = Workspace(name="Covariate CTA")

    class _Preparation:
        def __init__(self, ready, issues=None):
            self.ready = ready
            self.validation = type("Validation", (), {"issues": issues or []})()

    assert recommend_covariate_next_action(workspace, _Preparation(False), None, 0) == (
        "Open Data",
        "data",
        "Load a dataset in the Data workflow before running SCM.",
    )

    workspace.active_dataset = DatasetAsset(source_path="/tmp/theo.csv", display_name="theo.csv")
    assert recommend_covariate_next_action(workspace, _Preparation(False), None, 0) == (
        "Open Model",
        "model",
        "Configure a model in the Model workflow before running SCM.",
    )

    workspace.active_model_spec = object()  # type: ignore[assignment]
    issue = type(
        "Issue", (), {"message": "SCM requires a builder-mode model. Switch to Model Builder mode."}
    )()
    assert recommend_covariate_next_action(workspace, _Preparation(False, [issue]), None, 0) == (
        "Open Model",
        "model",
        "SCM requires a builder-mode model. Switch to Model Builder mode.",
    )

    assert recommend_covariate_next_action(workspace, _Preparation(True), None, 0) == (
        "Add candidate",
        "__add_candidate__",
        "Add at least one candidate before running SCM.",
    )
    assert recommend_covariate_next_action(workspace, _Preparation(True), None, 1) is None
    running = RunRecord(workflow="covariate", status=RunStatus.RUNNING)
    assert recommend_covariate_next_action(workspace, _Preparation(True), running, 0) is None


def test_covariate_blocking_message_reports_builder_mode_requirement(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV,WT\n1,0,0,70\n", encoding="utf-8")
    workspace = Workspace(name="Covariate blocking")
    workspace.active_dataset = DatasetAsset(
        source_path=str(dataset_path), display_name="theo.csv", columns=["ID", "TIME", "DV", "WT"]
    )
    workspace.active_model_spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        problem_title="Demo",
        dataset_path=str(dataset_path),
        control_stream_text="$PROBLEM Demo\n$DATA theo.csv\n$THETA 1\n$OMEGA 0.1\n$SIGMA 0.1\n$PK\nCL=THETA(1)\n$ERROR\nY=F+EPS(1)",
    )

    preparation = type(
        "Preparation",
        (),
        {
            "ready": False,
            "validation": type(
                "Validation",
                (),
                {
                    "issues": [
                        type(
                            "Issue",
                            (),
                            {
                                "message": "SCM requires a builder-mode model. Switch to Model Builder mode."
                            },
                        )()
                    ]
                },
            )(),
        },
    )()

    assert covariate_blocking_message(workspace, preparation) == (
        "SCM requires a builder-mode model. Switch to Model Builder mode."
    )


def test_covariate_workflow_state_requires_builder_mode(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV,WT\n1,0,0,70\n", encoding="utf-8")
    workspace = Workspace(name="Covariate state")
    workspace.active_dataset = DatasetAsset(
        source_path=str(dataset_path), display_name="theo.csv", columns=["ID", "TIME", "DV", "WT"]
    )
    workspace.active_model_spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        problem_title="Demo",
        dataset_path=str(dataset_path),
        control_stream_text="$PROBLEM Demo",
    )

    state = workflow_state_for(workspace, workflow_id="covariate")

    assert state.state == WorkflowStateId.NEEDS_ATTENTION
    assert "builder-mode model" in state.summary


# ── SCMCandidate ───────────────────────────────────────────────────────────────


class TestSCMCandidate:
    def test_defaults(self):
        c = SCMCandidate(parameter="CL", covariate="WT", effect="power")
        assert c.reference == 70.0

    def test_custom_reference(self):
        c = SCMCandidate(parameter="V", covariate="AGE", effect="linear", reference=40.0)
        assert c.reference == 40.0


# ── SCMRunResult ───────────────────────────────────────────────────────────────


class TestSCMRunResult:
    def test_fields_stored(self):
        rows = [
            {
                "type": "forward",
                "rel": "CL~WT(power)",
                "delta_ofv": -4.1,
                "p_value": 0.04,
                "accepted": True,
            }
        ]
        r = SCMRunResult(
            summary_text="ok",
            step_rows=rows,
            accepted_count=1,
            final_ofv=190.0,
            base_ofv=200.0,
        )
        assert r.accepted_count == 1
        assert r.final_ofv == 190.0
        assert len(r.step_rows) == 1


@pytest.mark.unit
def test_scm_service_runs_with_integer_dv_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV,WT,AGE\n1,0,0,70,55\n1,1,1,70,55\n", encoding="utf-8")

    project = Workspace(name="Covariate integer DV")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(
            source_path=str(dataset_path),
            display_name="theo.csv",
            columns=["ID", "TIME", "DV", "WT", "AGE"],
        ),
    )
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

    service = SCMService()
    preparation = service.prepare(project)
    runner = JobRunner(max_workers=1)

    try:
        outcome = runner.submit(
            service.create_job(
                project,
                candidates=[
                    SCMCandidate(parameter="CL", covariate="WT", effect="power", reference=70.0)
                ],
                n_jobs=1,
                preparation=preparation,
            )
        ).result(timeout=30)
    finally:
        runner.shutdown(wait=True)

    assert preparation.ready is True
    assert outcome.status == JobStatus.SUCCEEDED
    assert isinstance(outcome.value, SCMRunResult)
    assert "accepted" in outcome.value.summary_text


@pytest.mark.unit
def test_scm_service_uses_active_dataset_import_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "nmdata.csv"
    dataset_path.write_text(
        "1,0,999,70,0,55\n1,1,999,70,1,55\n2,0,999,90,0,65\n2,1,999,90,1.2,65\n",
        encoding="utf-8",
    )

    project = Workspace(name="Covariate NMData")
    project.active_dataset = DatasetAsset(
        source_path=str(dataset_path),
        display_name="nmdata.csv",
        separator=",",
        input_columns=["ID", "TIME", "_DROP_3", "WT", "DV", "AGE"],
        columns=["ID", "TIME", "WT", "DV", "AGE", "EVID", "MDV"],
    )
    project.active_model_spec = ModelSpec(
        problem_title="Covariate smoke",
        dataset_path=str(dataset_path),
        pk_code="CL = THETA(1) * EXP(ETA(1))",
        error_code="Y = F * (1 + EPS(1))",
        theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0, "label": "CL"}],
        omega_values=[[0.3]],
        sigma_values=[[0.1]],
    )

    service = SCMService()
    preparation = service.prepare(project)
    runner = JobRunner(max_workers=1)

    try:
        outcome = runner.submit(
            service.create_job(
                project,
                candidates=[
                    SCMCandidate(parameter="CL", covariate="WT", effect="power", reference=70.0)
                ],
                n_jobs=1,
                preparation=preparation,
            )
        ).result(timeout=30)
    finally:
        runner.shutdown(wait=True)

    assert preparation.ready is True
    assert outcome.status == JobStatus.SUCCEEDED
    assert isinstance(outcome.value, SCMRunResult)
    assert outcome.value.base_ofv >= outcome.value.final_ofv


@pytest.mark.unit
def test_covariate_workflow_next_action_transitions_from_data_to_add_candidate(
    tmp_path: Path,
) -> None:
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
    project = Workspace(name="Covariate next action")
    project_service = ProjectService()
    widget = build_covariate_workflow(project, project_service=project_service)
    navigations: list[str] = []
    widget._navigate_to_workflow = lambda workflow_id: navigations.append(workflow_id)  # type: ignore[attr-defined]

    try:
        widget.show()
        app.processEvents()

        next_action_label = widget.findChild(qt_widgets.QLabel, "covariate-next-action-label")
        next_action_button = widget.findChild(
            qt_widgets.QPushButton, "covariate-next-action-button"
        )
        candidates_table = widget.findChild(qt_widgets.QTableWidget, "covariate-candidates-table")

        assert next_action_label is not None
        assert next_action_button is not None
        assert candidates_table is not None
        assert next_action_button.text() == "Open Data"

        next_action_button.click()
        app.processEvents()

        project_service.attach_dataset(
            project,
            DatasetAsset(
                source_path=str(dataset_path),
                display_name="theo.csv",
                columns=["ID", "TIME", "DV", "WT"],
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
                problem_title="Smoke",
                dataset_path=str(dataset_path),
                pk_code="CL = THETA(1) * EXP(ETA(1))",
                error_code="Y = F * (1 + EPS(1))",
                theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0, "label": "CL"}],
                omega_values=[[0.3]],
                sigma_values=[[0.1]],
            ),
        )
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_button.text() == "Add candidate"
        assert "Add at least one candidate" in next_action_label.text()

        next_action_button.click()
        app.processEvents()

        assert candidates_table.rowCount() == 1
        assert next_action_label.isHidden() is True
        assert next_action_button.isHidden() is True
        assert navigations == ["data", "model"]
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_covariate_workflow_uses_dataset_covariate_dropdown(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV,WT,AGE,SEX\n1,0,0,70,55,M\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Covariate dropdown")
    project_service = ProjectService()
    project_service.attach_dataset(
        project,
        DatasetAsset(
            source_path=str(dataset_path),
            display_name="theo.csv",
            columns=["ID", "TIME", "DV", "WT", "AGE", "SEX"],
        ),
    )
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
    widget = build_covariate_workflow(project, project_service=project_service)

    try:
        widget.show()
        app.processEvents()

        add_button = widget.findChild(qt_widgets.QPushButton, "covariate-add-candidate-button")
        candidates_table = widget.findChild(qt_widgets.QTableWidget, "covariate-candidates-table")

        assert add_button is not None
        assert candidates_table is not None

        add_button.click()
        app.processEvents()

        parameter_combo = candidates_table.cellWidget(0, 0)
        covariate_combo = candidates_table.cellWidget(0, 1)

        assert isinstance(parameter_combo, qt_widgets.QComboBox)
        assert parameter_combo.isEditable() is False
        assert parameter_combo.currentText() == "CL"
        assert isinstance(covariate_combo, qt_widgets.QComboBox)
        assert covariate_combo.isEditable() is False
        assert [covariate_combo.itemText(index) for index in range(covariate_combo.count())] == [
            "WT",
            "AGE",
            "SEX",
        ]
        assert (
            candidates_table.horizontalHeader().sectionResizeMode(0)
            == qt_widgets.QHeaderView.ResizeMode.Interactive
        )

        covariate_combo.setCurrentText("AGE")
        app.processEvents()

        assert project.active_scenario.metadata["scm_candidates"][0]["covariate"] == "AGE"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_covariate_workflow_surfaces_builder_mode_blocker_in_status(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV,WT\n1,0,0,70\n", encoding="utf-8")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Covariate blocked")
    project.active_dataset = DatasetAsset(
        source_path=str(dataset_path),
        display_name="theo.csv",
        columns=["ID", "TIME", "DV", "WT"],
    )
    project.active_model_spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        problem_title="Demo",
        dataset_path=str(dataset_path),
        control_stream_text="$PROBLEM Demo\n$DATA theo.csv",
    )
    widget = build_covariate_workflow(project, project_service=ProjectService())

    try:
        widget.show()
        app.processEvents()

        run_button = widget.findChild(qt_widgets.QPushButton, "covariate-run-button")
        next_action_button = widget.findChild(
            qt_widgets.QPushButton, "covariate-next-action-button"
        )
        status_label = widget.findChild(qt_widgets.QLabel, "covariate-status-label")
        add_button = widget.findChild(qt_widgets.QPushButton, "covariate-add-candidate-button")

        assert run_button is not None
        assert next_action_button is not None
        assert status_label is not None
        assert add_button is not None

        add_button.click()
        app.processEvents()

        assert run_button.isEnabled() is False
        assert next_action_button.text() == "Open Model"
        assert "builder-mode model" in status_label.text()
        assert "builder-mode model" in run_button.toolTip()
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
