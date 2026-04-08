"""Unit tests for overview_workflow.py pure functions.

These cover the entire state machine of review_outputs_state(),
recommended_next_action(), and all format_* helpers without touching Qt.
"""

from __future__ import annotations

import pytest

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec, ModelSpecMode
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.workflow_state_service import WorkflowState, WorkflowStateId
from openpkpd_gui.workflows.overview_workflow import (
    format_additional_workflows_summary,
    format_follow_up_summary,
    format_latest_output_summary,
    format_recent_activity,
    format_review_workflows_summary,
    recommended_next_action,
    review_outputs_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ws() -> Workspace:
    return Workspace(name="W")


def _state(state_id: WorkflowStateId) -> WorkflowState:
    return WorkflowState(
        workflow_id="x",
        state=state_id,
        label=state_id.value,
        summary="",
    )


def _succeeded_fit_run() -> RunRecord:
    run = RunRecord(workflow="fit")
    run.mark_running()
    run.mark_succeeded("ok")
    return run


def _failed_fit_run() -> RunRecord:
    run = RunRecord(workflow="fit")
    run.mark_running()
    run.mark_failed("convergence failed")
    return run


def _running_fit_run() -> RunRecord:
    run = RunRecord(workflow="fit")
    run.mark_running()
    return run


def _artifact(role: str, plot_type: str | None = None) -> ArtifactRecord:
    meta: dict[str, str | int | float | bool | None] = {"artifact_role": role}
    if plot_type is not None:
        meta["plot_type"] = plot_type
    return ArtifactRecord(kind="plot" if role == "plot" else "report", label=role, metadata=meta)


# ===========================================================================
# review_outputs_state
# ===========================================================================


def test_review_outputs_no_fit_run_is_not_started() -> None:
    ws = _ws()
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.NOT_STARTED


def test_review_outputs_running_fit_returns_running() -> None:
    ws = _ws()
    ws.active_scenario.runs = [_running_fit_run()]
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.RUNNING


def test_review_outputs_failed_fit_returns_needs_attention() -> None:
    ws = _ws()
    ws.active_scenario.runs = [_failed_fit_run()]
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.NEEDS_ATTENTION


def test_review_outputs_with_report_artifact_returns_results_available() -> None:
    ws = _ws()
    ws.active_scenario.runs = [_succeeded_fit_run()]
    ws.active_scenario.artifacts = [_artifact("report")]
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.RESULTS_AVAILABLE


def test_review_outputs_with_plot_artifact_returns_results_available() -> None:
    ws = _ws()
    ws.active_scenario.runs = [_succeeded_fit_run()]
    ws.active_scenario.artifacts = [_artifact("plot")]
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.RESULTS_AVAILABLE


def test_review_outputs_with_diagnostics_table_returns_results_available() -> None:
    ws = _ws()
    ws.active_scenario.runs = [_succeeded_fit_run()]
    ws.active_scenario.artifacts = [_artifact("diagnostics_table")]
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.RESULTS_AVAILABLE


def test_review_outputs_with_npde_table_returns_results_available() -> None:
    ws = _ws()
    ws.active_scenario.runs = [_succeeded_fit_run()]
    ws.active_scenario.artifacts = [_artifact("npde_table")]
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.RESULTS_AVAILABLE


def test_review_outputs_with_gof_plot_returns_results_available() -> None:
    ws = _ws()
    ws.active_scenario.runs = [_succeeded_fit_run()]
    ws.active_scenario.artifacts = [_artifact("plot", plot_type="gof_dv_pred")]
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.RESULTS_AVAILABLE


def test_review_outputs_succeeded_fit_no_artifacts_is_ready() -> None:
    ws = _ws()
    ws.active_scenario.runs = [_succeeded_fit_run()]
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.READY


def test_review_outputs_running_fit_takes_precedence_over_artifacts() -> None:
    """Running trumps RESULTS_AVAILABLE."""
    ws = _ws()
    ws.active_scenario.runs = [_running_fit_run()]
    ws.active_scenario.artifacts = [_artifact("report")]
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.RUNNING


def test_review_outputs_failed_fit_takes_precedence_over_artifacts() -> None:
    """Failed run → NEEDS_ATTENTION even when artifacts exist."""
    ws = _ws()
    ws.active_scenario.runs = [_failed_fit_run()]
    ws.active_scenario.artifacts = [_artifact("report")]
    state = review_outputs_state(ws)
    assert state.state == WorkflowStateId.NEEDS_ATTENTION


# ===========================================================================
# recommended_next_action
# ===========================================================================


def test_recommended_action_no_dataset_go_to_data() -> None:
    ws = _ws()
    label, workflow, _ = recommended_next_action(ws)
    assert workflow == "data"
    assert "data" in label.lower()


def test_recommended_action_has_dataset_no_model_go_to_model() -> None:
    ws = _ws()
    ws.active_scenario.active_dataset = DatasetAsset(
        source_path="data.csv", display_name="Data"
    )
    label, workflow, _ = recommended_next_action(ws)
    assert workflow == "model"


def test_recommended_action_ready_to_fit_go_to_fit() -> None:
    ws = _ws()
    ws.active_scenario.active_dataset = DatasetAsset(
        source_path="data.csv", display_name="Data", columns=["ID", "TIME", "DV"]
    )
    ws.active_scenario.active_model_spec = ModelSpec(
        problem_title="Demo",
        mode=ModelSpecMode.BUILDER,
        pk_code="CL = THETA(1)",
    )
    label, workflow, _ = recommended_next_action(ws)
    assert workflow == "fit"


def test_recommended_action_running_fit_go_to_fit() -> None:
    ws = _ws()
    ws.active_scenario.active_dataset = DatasetAsset(
        source_path="data.csv", display_name="Data", columns=["ID", "TIME", "DV"]
    )
    ws.active_scenario.active_model_spec = ModelSpec(
        problem_title="Demo", mode=ModelSpecMode.BUILDER, pk_code="CL = THETA(1)"
    )
    ws.active_scenario.runs = [_running_fit_run()]
    label, workflow, _ = recommended_next_action(ws)
    assert workflow == "fit"
    assert "monitor" in _.lower() or "active" in _.lower() or "running" in _.lower()


def test_recommended_action_failed_fit_go_to_fit() -> None:
    ws = _ws()
    ws.active_scenario.active_dataset = DatasetAsset(
        source_path="data.csv", display_name="Data", columns=["ID", "TIME", "DV"]
    )
    ws.active_scenario.active_model_spec = ModelSpec(
        problem_title="Demo", mode=ModelSpecMode.BUILDER, pk_code="CL = THETA(1)"
    )
    ws.active_scenario.runs = [_failed_fit_run()]
    label, workflow, _ = recommended_next_action(ws)
    assert workflow == "fit"


def test_recommended_action_results_available_go_to_results() -> None:
    ws = _ws()
    ws.active_scenario.active_dataset = DatasetAsset(
        source_path="data.csv", display_name="Data", columns=["ID", "TIME", "DV"]
    )
    ws.active_scenario.active_model_spec = ModelSpec(
        problem_title="Demo", mode=ModelSpecMode.BUILDER, pk_code="CL = THETA(1)"
    )
    ws.active_scenario.runs = [_succeeded_fit_run()]
    ws.active_scenario.artifacts = [_artifact("report")]
    label, workflow, _ = recommended_next_action(ws)
    assert workflow == "results"


# ===========================================================================
# format_recent_activity
# ===========================================================================


def test_format_recent_activity_no_runs_or_artifacts() -> None:
    ws = _ws()
    text = format_recent_activity(ws)
    assert "no runs" in text.lower() or "no" in text.lower()


def test_format_recent_activity_with_run() -> None:
    ws = _ws()
    run = _succeeded_fit_run()
    ws.active_scenario.runs = [run]
    text = format_recent_activity(ws)
    assert "fit" in text.lower()
    assert "succeeded" in text.lower() or "success" in text.lower()


def test_format_recent_activity_with_run_and_artifact() -> None:
    ws = _ws()
    ws.active_scenario.runs = [_succeeded_fit_run()]
    artifact = ArtifactRecord(kind="report", label="HTML report", path="/tmp/report.html")
    ws.active_scenario.artifacts = [artifact]
    text = format_recent_activity(ws)
    assert "fit" in text.lower()
    assert "html report" in text.lower() or "report" in text.lower()


def test_format_recent_activity_artifact_no_path() -> None:
    ws = _ws()
    artifact = ArtifactRecord(kind="table", label="Summary table", path=None)
    ws.active_scenario.artifacts = [artifact]
    text = format_recent_activity(ws)
    assert "in-memory" in text.lower() or "memory" in text.lower()


def test_format_recent_activity_failed_run_shows_error_text() -> None:
    ws = _ws()
    run = _failed_fit_run()
    ws.active_scenario.runs = [run]
    text = format_recent_activity(ws)
    assert "failed" in text.lower()


# ===========================================================================
# format_latest_output_summary
# ===========================================================================


def test_format_latest_output_summary_no_outputs() -> None:
    ws = _ws()
    text = format_latest_output_summary(ws)
    assert "no saved" in text.lower() or "not available" in text.lower() or "available" in text.lower()


def test_format_latest_output_summary_with_report() -> None:
    ws = _ws()
    ws.active_scenario.artifacts = [_artifact("report")]
    text = format_latest_output_summary(ws)
    assert "report" in text.lower()


def test_format_latest_output_summary_with_plot() -> None:
    ws = _ws()
    ws.active_scenario.artifacts = [_artifact("plot")]
    text = format_latest_output_summary(ws)
    assert "plot" in text.lower()


def test_format_latest_output_summary_with_gof_plot() -> None:
    ws = _ws()
    ws.active_scenario.artifacts = [_artifact("plot", plot_type="gof_dv_pred")]
    text = format_latest_output_summary(ws)
    assert "gof" in text.lower() or "plot" in text.lower()


def test_format_latest_output_summary_running_fit_no_outputs() -> None:
    ws = _ws()
    ws.active_scenario.runs = [_running_fit_run()]
    text = format_latest_output_summary(ws)
    # Running fit with no artifacts → mentions current fit
    assert "current fit" in text.lower() or "running" in text.lower() or "finish" in text.lower()


# ===========================================================================
# format_review_workflows_summary
# ===========================================================================


def test_format_review_workflows_both_not_started() -> None:
    text = format_review_workflows_summary(
        _state(WorkflowStateId.NOT_STARTED),
        _state(WorkflowStateId.NOT_STARTED),
    )
    assert "unlock" in text.lower() or "after" in text.lower()


def test_format_review_workflows_one_ready() -> None:
    text = format_review_workflows_summary(
        _state(WorkflowStateId.READY),
        _state(WorkflowStateId.NOT_STARTED),
    )
    assert "results" in text.lower() or "ready" in text.lower()


def test_format_review_workflows_results_available() -> None:
    text = format_review_workflows_summary(
        _state(WorkflowStateId.RESULTS_AVAILABLE),
        _state(WorkflowStateId.RESULTS_AVAILABLE),
    )
    assert "results" in text.lower()


# ===========================================================================
# format_follow_up_summary
# ===========================================================================


def test_format_follow_up_both_not_started() -> None:
    text = format_follow_up_summary(
        _state(WorkflowStateId.NOT_STARTED),
        _state(WorkflowStateId.NOT_STARTED),
    )
    assert "unlock" in text.lower() or "after" in text.lower()


def test_format_follow_up_one_ready() -> None:
    text = format_follow_up_summary(
        _state(WorkflowStateId.READY),
        _state(WorkflowStateId.NOT_STARTED),
    )
    assert "diagnostics" in text.lower() or "ready" in text.lower()


def test_format_follow_up_both_results_available() -> None:
    text = format_follow_up_summary(
        _state(WorkflowStateId.RESULTS_AVAILABLE),
        _state(WorkflowStateId.RESULTS_AVAILABLE),
    )
    assert "diagnostics" in text.lower() or "advanced" in text.lower()


# ===========================================================================
# format_additional_workflows_summary
# ===========================================================================


def test_format_additional_both_not_started() -> None:
    text = format_additional_workflows_summary(
        _state(WorkflowStateId.NOT_STARTED),
        _state(WorkflowStateId.NOT_STARTED),
    )
    assert "unlock" in text.lower() or "after" in text.lower()


def test_format_additional_nca_ready() -> None:
    text = format_additional_workflows_summary(
        _state(WorkflowStateId.READY),
        _state(WorkflowStateId.NOT_STARTED),
    )
    assert "nca" in text.lower() or "ready" in text.lower()


def test_format_additional_both_results_available() -> None:
    text = format_additional_workflows_summary(
        _state(WorkflowStateId.RESULTS_AVAILABLE),
        _state(WorkflowStateId.RESULTS_AVAILABLE),
    )
    assert "nca" in text.lower() or "covariate" in text.lower()
