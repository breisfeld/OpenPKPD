"""Unit tests for WorkflowStateService — state machine covering all workflows × states."""

from __future__ import annotations

import time

import pytest

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec, ModelSpecMode
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Scenario, Workspace
from openpkpd_gui.services.workflow_state_service import (
    WorkflowStateId,
    describe_fit_input_changes,
    fit_input_changes_since_latest_successful_fit,
    latest_artifact,
    latest_run_for_workflow,
    latest_run_for_workflows,
    latest_successful_run_for_workflow,
    resolve_scenario,
    workflow_state_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ws() -> Workspace:
    return Workspace(name="W")


def _scenario() -> Scenario:
    return Scenario()


def _run(workflow: str, status: RunStatus) -> RunRecord:
    run = RunRecord(workflow=workflow)
    if status in (RunStatus.RUNNING, RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
        run.mark_running()
    if status == RunStatus.SUCCEEDED:
        run.mark_succeeded("ok")
    elif status == RunStatus.FAILED:
        run.mark_failed("err")
    elif status == RunStatus.CANCELLED:
        run.mark_cancelled()
    return run


def _artifact(kind: str = "report", role: str = "", plot_type: str = "") -> ArtifactRecord:
    meta: dict[str, str | int | float | bool | None] = {}
    if role:
        meta["artifact_role"] = role
    if plot_type:
        meta["plot_type"] = plot_type
    return ArtifactRecord(kind=kind, label="A", metadata=meta)


def _dataset() -> DatasetAsset:
    return DatasetAsset(source_path="/data/demo.csv", display_name="demo.csv")


def _builder_model() -> ModelSpec:
    return ModelSpec(
        problem_title="Demo",
        mode=ModelSpecMode.BUILDER,
        pk_code="CL = THETA(1)",
        error_code="Y = F",
        theta_rows=[{"init": 1.0}],
        omega_values=[[0.1]],
        sigma_values=[[0.05]],
    )


def _cs_model() -> ModelSpec:
    return ModelSpec(
        problem_title="Demo",
        mode=ModelSpecMode.CONTROL_STREAM,
        control_stream_text="$PROBLEM\n$THETA 1\n",
    )


# ---------------------------------------------------------------------------
# resolve_scenario
# ---------------------------------------------------------------------------


def test_resolve_scenario_active_fallback() -> None:
    ws = _ws()
    s = resolve_scenario(ws)
    assert s is ws.active_scenario


def test_resolve_scenario_by_id() -> None:
    ws = _ws()
    extra = Scenario(name="Extra")
    ws.active_project.add_scenario(extra)
    s = resolve_scenario(ws, scenario_id=extra.scenario_id)
    assert s.scenario_id == extra.scenario_id


def test_resolve_scenario_by_project_id() -> None:
    ws = _ws()
    pid = ws.active_project_id
    s = resolve_scenario(ws, project_id=pid)
    assert s is ws.active_scenario


def test_resolve_scenario_unknown_id_falls_back_to_active() -> None:
    ws = _ws()
    s = resolve_scenario(ws, scenario_id="nonexistent")
    assert s is ws.active_scenario


# ---------------------------------------------------------------------------
# latest_run_for_workflow
# ---------------------------------------------------------------------------


def test_latest_run_for_workflow_returns_newest() -> None:
    s = _scenario()
    r1 = _run("fit", RunStatus.SUCCEEDED)
    r2 = _run("fit", RunStatus.FAILED)
    s.add_run(r1)
    s.add_run(r2)
    assert latest_run_for_workflow(s, "fit") is r2


def test_latest_run_for_workflow_ignores_other_workflows() -> None:
    s = _scenario()
    s.add_run(_run("nca", RunStatus.SUCCEEDED))
    assert latest_run_for_workflow(s, "fit") is None


def test_latest_run_for_workflow_none_when_empty() -> None:
    assert latest_run_for_workflow(_scenario(), "fit") is None


# ---------------------------------------------------------------------------
# latest_successful_run_for_workflow
# ---------------------------------------------------------------------------


def test_latest_successful_skips_failed() -> None:
    s = _scenario()
    ok = _run("fit", RunStatus.SUCCEEDED)
    s.add_run(ok)
    s.add_run(_run("fit", RunStatus.FAILED))
    assert latest_successful_run_for_workflow(s, "fit") is ok


def test_latest_successful_none_when_all_failed() -> None:
    s = _scenario()
    s.add_run(_run("fit", RunStatus.FAILED))
    assert latest_successful_run_for_workflow(s, "fit") is None


# ---------------------------------------------------------------------------
# latest_run_for_workflows
# ---------------------------------------------------------------------------


def test_latest_run_for_workflows_matches_any() -> None:
    s = _scenario()
    vpc = _run("vpc", RunStatus.SUCCEEDED)
    bs = _run("bootstrap", RunStatus.RUNNING)
    s.add_run(vpc)
    s.add_run(bs)
    result = latest_run_for_workflows(s, {"vpc", "bootstrap", "design"})
    assert result is bs


def test_latest_run_for_workflows_none_if_no_match() -> None:
    s = _scenario()
    s.add_run(_run("fit", RunStatus.SUCCEEDED))
    assert latest_run_for_workflows(s, {"vpc", "bootstrap"}) is None


# ---------------------------------------------------------------------------
# latest_artifact
# ---------------------------------------------------------------------------


def test_latest_artifact_by_kind() -> None:
    s = _scenario()
    s.add_artifact(_artifact("report"))
    s.add_artifact(_artifact("plot"))
    found = latest_artifact(s, kind="plot")
    assert found is not None
    assert found.kind == "plot"


def test_latest_artifact_by_role() -> None:
    s = _scenario()
    s.add_artifact(_artifact("table", role="nca_summary"))
    s.add_artifact(_artifact("table", role="fit_summary"))
    found = latest_artifact(s, role="nca_summary")
    assert found is not None
    assert found.metadata.get("artifact_role") == "nca_summary"


def test_latest_artifact_by_plot_type() -> None:
    s = _scenario()
    s.add_artifact(_artifact("plot", plot_type="gof_panel"))
    s.add_artifact(_artifact("plot", plot_type="vpc"))
    found = latest_artifact(s, plot_type="vpc")
    assert found is not None
    assert found.metadata.get("plot_type") == "vpc"


def test_latest_artifact_returns_newest_matching() -> None:
    s = _scenario()
    old = _artifact("report")
    new = _artifact("report")
    s.add_artifact(old)
    s.add_artifact(new)
    assert latest_artifact(s, kind="report") is new


def test_latest_artifact_none_when_no_match() -> None:
    s = _scenario()
    s.add_artifact(_artifact("report"))
    assert latest_artifact(s, kind="plot") is None


def test_latest_artifact_combined_filters() -> None:
    s = _scenario()
    s.add_artifact(_artifact("plot", role="", plot_type="gof_panel"))
    s.add_artifact(_artifact("table", role="nca_summary", plot_type=""))
    # kind=plot AND plot_type=gof_panel
    found = latest_artifact(s, kind="plot", plot_type="gof_panel")
    assert found is not None
    # kind=plot AND plot_type=nca_summary — no match
    assert latest_artifact(s, kind="plot", plot_type="nca_summary") is None


# ---------------------------------------------------------------------------
# fit_input_changes_since_latest_successful_fit
# ---------------------------------------------------------------------------


def test_no_successful_fit_returns_empty() -> None:
    s = _scenario()
    assert fit_input_changes_since_latest_successful_fit(s) == ()


def test_no_changes_since_fit_returns_empty() -> None:
    s = _scenario()
    # Attach dataset and model BEFORE marking fit succeeded
    s.dataset_updated_at = "2024-01-01T00:00:00+00:00"
    s.model_updated_at = "2024-01-01T00:00:00+00:00"
    run = _run("fit", RunStatus.SUCCEEDED)
    # Simulate fit finished AFTER inputs were updated
    run.finished_at = "2024-06-01T00:00:00+00:00"
    s.add_run(run)
    assert fit_input_changes_since_latest_successful_fit(s) == ()


def test_dataset_changed_after_fit() -> None:
    s = _scenario()
    run = _run("fit", RunStatus.SUCCEEDED)
    run.finished_at = "2024-01-01T00:00:00+00:00"
    s.add_run(run)
    # Dataset updated AFTER fit
    time.sleep(0.01)
    s.dataset_updated_at = "2025-01-01T00:00:00+00:00"
    result = fit_input_changes_since_latest_successful_fit(s)
    assert "dataset" in result


def test_model_changed_after_fit() -> None:
    s = _scenario()
    run = _run("fit", RunStatus.SUCCEEDED)
    run.finished_at = "2024-01-01T00:00:00+00:00"
    s.add_run(run)
    s.model_updated_at = "2025-01-01T00:00:00+00:00"
    result = fit_input_changes_since_latest_successful_fit(s)
    assert "model" in result


def test_both_changed_after_fit() -> None:
    s = _scenario()
    run = _run("fit", RunStatus.SUCCEEDED)
    run.finished_at = "2024-01-01T00:00:00+00:00"
    s.add_run(run)
    s.dataset_updated_at = "2025-01-01T00:00:00+00:00"
    s.model_updated_at = "2025-01-01T00:00:00+00:00"
    result = fit_input_changes_since_latest_successful_fit(s)
    assert "dataset" in result and "model" in result


def test_running_fit_suppresses_staleness() -> None:
    """A currently-running fit should not be reported as stale."""
    s = _scenario()
    succeeded = _run("fit", RunStatus.SUCCEEDED)
    succeeded.finished_at = "2024-01-01T00:00:00+00:00"
    s.add_run(succeeded)
    s.dataset_updated_at = "2025-01-01T00:00:00+00:00"
    # Add a currently-running fit AFTER the stale input — should suppress staleness report
    s.add_run(_run("fit", RunStatus.RUNNING))
    assert fit_input_changes_since_latest_successful_fit(s) == ()


# ---------------------------------------------------------------------------
# describe_fit_input_changes
# ---------------------------------------------------------------------------


def test_describe_no_changes_returns_none() -> None:
    assert describe_fit_input_changes(_scenario()) is None


def test_describe_dataset_only() -> None:
    s = _scenario()
    run = _run("fit", RunStatus.SUCCEEDED)
    run.finished_at = "2024-01-01T00:00:00+00:00"
    s.add_run(run)
    s.dataset_updated_at = "2025-01-01T00:00:00+00:00"
    assert describe_fit_input_changes(s) == "Dataset changed"


def test_describe_model_only() -> None:
    s = _scenario()
    run = _run("fit", RunStatus.SUCCEEDED)
    run.finished_at = "2024-01-01T00:00:00+00:00"
    s.add_run(run)
    s.model_updated_at = "2025-01-01T00:00:00+00:00"
    assert describe_fit_input_changes(s) == "Model changed"


def test_describe_both_changed() -> None:
    s = _scenario()
    run = _run("fit", RunStatus.SUCCEEDED)
    run.finished_at = "2024-01-01T00:00:00+00:00"
    s.add_run(run)
    s.dataset_updated_at = "2025-01-01T00:00:00+00:00"
    s.model_updated_at = "2025-01-01T00:00:00+00:00"
    assert describe_fit_input_changes(s) == "Dataset and model changed"


# ---------------------------------------------------------------------------
# workflow_state_for — home (always READY)
# ---------------------------------------------------------------------------


def test_home_always_ready() -> None:
    ws = _ws()
    state = workflow_state_for(ws, "home")
    assert state.state == WorkflowStateId.READY


# ---------------------------------------------------------------------------
# workflow_state_for — overview
# ---------------------------------------------------------------------------


def test_overview_not_started_empty() -> None:
    ws = _ws()
    state = workflow_state_for(ws, "overview")
    assert state.state == WorkflowStateId.NOT_STARTED


def test_overview_ready_has_dataset() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    state = workflow_state_for(ws, "overview")
    assert state.state == WorkflowStateId.READY


def test_overview_running_has_running_run() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.RUNNING))
    assert workflow_state_for(ws, "overview").state == WorkflowStateId.RUNNING


def test_overview_needs_attention_has_failed_run() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.FAILED))
    assert workflow_state_for(ws, "overview").state == WorkflowStateId.NEEDS_ATTENTION


def test_overview_results_available_has_runs() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.SUCCEEDED))
    assert workflow_state_for(ws, "overview").state == WorkflowStateId.RESULTS_AVAILABLE


# ---------------------------------------------------------------------------
# workflow_state_for — data
# ---------------------------------------------------------------------------


def test_data_not_started_clean() -> None:
    ws = _ws()
    assert workflow_state_for(ws, "data").state == WorkflowStateId.NOT_STARTED


def test_data_ready_has_dataset() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    assert workflow_state_for(ws, "data").state == WorkflowStateId.READY


def test_data_needs_attention_has_model_but_no_dataset() -> None:
    ws = _ws()
    ws.active_model_spec = _builder_model()
    assert workflow_state_for(ws, "data").state == WorkflowStateId.NEEDS_ATTENTION


def test_data_needs_attention_has_runs_but_no_dataset() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.SUCCEEDED))
    assert workflow_state_for(ws, "data").state == WorkflowStateId.NEEDS_ATTENTION


# ---------------------------------------------------------------------------
# workflow_state_for — model
# ---------------------------------------------------------------------------


def test_model_not_started_clean() -> None:
    ws = _ws()
    assert workflow_state_for(ws, "model").state == WorkflowStateId.NOT_STARTED


def test_model_needs_attention_has_dataset_no_model() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    assert workflow_state_for(ws, "model").state == WorkflowStateId.NEEDS_ATTENTION


def test_model_ready_has_model() -> None:
    ws = _ws()
    ws.active_model_spec = _builder_model()
    assert workflow_state_for(ws, "model").state == WorkflowStateId.READY


# ---------------------------------------------------------------------------
# workflow_state_for — fit
# ---------------------------------------------------------------------------


def test_fit_not_started_clean() -> None:
    ws = _ws()
    assert workflow_state_for(ws, "fit").state == WorkflowStateId.NOT_STARTED


def test_fit_needs_attention_only_dataset() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    assert workflow_state_for(ws, "fit").state == WorkflowStateId.NEEDS_ATTENTION


def test_fit_ready_has_dataset_and_model() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    ws.active_model_spec = _builder_model()
    assert workflow_state_for(ws, "fit").state == WorkflowStateId.READY


def test_fit_running() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.RUNNING))
    assert workflow_state_for(ws, "fit").state == WorkflowStateId.RUNNING


def test_fit_needs_attention_last_failed() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.FAILED))
    assert workflow_state_for(ws, "fit").state == WorkflowStateId.NEEDS_ATTENTION


def test_fit_results_available_succeeded() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.SUCCEEDED))
    assert workflow_state_for(ws, "fit").state == WorkflowStateId.RESULTS_AVAILABLE


# ---------------------------------------------------------------------------
# workflow_state_for — nca
# ---------------------------------------------------------------------------


def test_nca_not_started_clean() -> None:
    ws = _ws()
    assert workflow_state_for(ws, "nca").state == WorkflowStateId.NOT_STARTED


def test_nca_ready_has_dataset() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    assert workflow_state_for(ws, "nca").state == WorkflowStateId.READY


def test_nca_running() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("nca", RunStatus.RUNNING))
    assert workflow_state_for(ws, "nca").state == WorkflowStateId.RUNNING


def test_nca_needs_attention_last_failed() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("nca", RunStatus.FAILED))
    assert workflow_state_for(ws, "nca").state == WorkflowStateId.NEEDS_ATTENTION


def test_nca_results_available_has_artifact() -> None:
    ws = _ws()
    ws.active_scenario.add_artifact(_artifact("table", role="nca_summary"))
    assert workflow_state_for(ws, "nca").state == WorkflowStateId.RESULTS_AVAILABLE


# ---------------------------------------------------------------------------
# workflow_state_for — results
# ---------------------------------------------------------------------------


def test_results_not_started_clean() -> None:
    ws = _ws()
    assert workflow_state_for(ws, "results").state == WorkflowStateId.NOT_STARTED


def test_results_ready_has_dataset_and_model() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    ws.active_model_spec = _builder_model()
    assert workflow_state_for(ws, "results").state == WorkflowStateId.READY


def test_results_running_fit_running() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.RUNNING))
    assert workflow_state_for(ws, "results").state == WorkflowStateId.RUNNING


def test_results_available_has_runs() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.SUCCEEDED))
    assert workflow_state_for(ws, "results").state == WorkflowStateId.RESULTS_AVAILABLE


def test_results_available_has_report_artifact() -> None:
    ws = _ws()
    ws.active_scenario.add_artifact(_artifact("report"))
    assert workflow_state_for(ws, "results").state == WorkflowStateId.RESULTS_AVAILABLE


# ---------------------------------------------------------------------------
# workflow_state_for — plots
# ---------------------------------------------------------------------------


def test_plots_not_started_clean() -> None:
    ws = _ws()
    assert workflow_state_for(ws, "plots").state == WorkflowStateId.NOT_STARTED


def test_plots_results_available_has_plot() -> None:
    ws = _ws()
    ws.active_scenario.add_artifact(_artifact("plot"))
    assert workflow_state_for(ws, "plots").state == WorkflowStateId.RESULTS_AVAILABLE


def test_plots_ready_after_successful_fit() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.SUCCEEDED))
    assert workflow_state_for(ws, "plots").state == WorkflowStateId.READY


def test_plots_needs_attention_fit_failed() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.FAILED))
    assert workflow_state_for(ws, "plots").state == WorkflowStateId.NEEDS_ATTENTION


def test_plots_running_fit_running() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.RUNNING))
    assert workflow_state_for(ws, "plots").state == WorkflowStateId.RUNNING


# ---------------------------------------------------------------------------
# workflow_state_for — diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_not_started_clean() -> None:
    ws = _ws()
    assert workflow_state_for(ws, "diagnostics").state == WorkflowStateId.NOT_STARTED


def test_diagnostics_results_available_has_diagnostics_table() -> None:
    ws = _ws()
    ws.active_scenario.add_artifact(_artifact("table", role="diagnostics_table"))
    assert workflow_state_for(ws, "diagnostics").state == WorkflowStateId.RESULTS_AVAILABLE


def test_diagnostics_results_available_has_gof_plot() -> None:
    ws = _ws()
    ws.active_scenario.add_artifact(_artifact("plot", plot_type="gof_panel"))
    assert workflow_state_for(ws, "diagnostics").state == WorkflowStateId.RESULTS_AVAILABLE


def test_diagnostics_ready_after_successful_fit() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.SUCCEEDED))
    assert workflow_state_for(ws, "diagnostics").state == WorkflowStateId.READY


def test_diagnostics_needs_attention_fit_failed() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.FAILED))
    assert workflow_state_for(ws, "diagnostics").state == WorkflowStateId.NEEDS_ATTENTION


def test_diagnostics_running_fit_running() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.RUNNING))
    assert workflow_state_for(ws, "diagnostics").state == WorkflowStateId.RUNNING


# ---------------------------------------------------------------------------
# workflow_state_for — covariate
# ---------------------------------------------------------------------------


def test_covariate_not_started_clean() -> None:
    ws = _ws()
    assert workflow_state_for(ws, "covariate").state == WorkflowStateId.NOT_STARTED


def test_covariate_needs_attention_only_dataset() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    assert workflow_state_for(ws, "covariate").state == WorkflowStateId.NEEDS_ATTENTION


def test_covariate_needs_attention_control_stream_model() -> None:
    """Control stream models not allowed for SCM — needs builder mode."""
    ws = _ws()
    ws.active_dataset = _dataset()
    ws.active_model_spec = _cs_model()
    assert workflow_state_for(ws, "covariate").state == WorkflowStateId.NEEDS_ATTENTION


def test_covariate_ready_builder_model_and_dataset() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    ws.active_model_spec = _builder_model()
    assert workflow_state_for(ws, "covariate").state == WorkflowStateId.READY


def test_covariate_running() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("covariate", RunStatus.RUNNING))
    assert workflow_state_for(ws, "covariate").state == WorkflowStateId.RUNNING


def test_covariate_needs_attention_failed() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("covariate", RunStatus.FAILED))
    assert workflow_state_for(ws, "covariate").state == WorkflowStateId.NEEDS_ATTENTION


def test_covariate_results_available_succeeded() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("covariate", RunStatus.SUCCEEDED))
    assert workflow_state_for(ws, "covariate").state == WorkflowStateId.RESULTS_AVAILABLE


# ---------------------------------------------------------------------------
# workflow_state_for — advanced
# ---------------------------------------------------------------------------


def test_advanced_not_started_clean() -> None:
    ws = _ws()
    assert workflow_state_for(ws, "advanced").state == WorkflowStateId.NOT_STARTED


def test_advanced_ready_after_successful_fit() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.SUCCEEDED))
    assert workflow_state_for(ws, "advanced").state == WorkflowStateId.READY


def test_advanced_needs_attention_fit_failed() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("fit", RunStatus.FAILED))
    assert workflow_state_for(ws, "advanced").state == WorkflowStateId.NEEDS_ATTENTION


def test_advanced_running_vpc() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("vpc", RunStatus.RUNNING))
    assert workflow_state_for(ws, "advanced").state == WorkflowStateId.RUNNING


def test_advanced_running_bootstrap() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run("bootstrap", RunStatus.RUNNING))
    assert workflow_state_for(ws, "advanced").state == WorkflowStateId.RUNNING


def test_advanced_results_available_vpc_artifact() -> None:
    ws = _ws()
    ws.active_scenario.add_artifact(_artifact("plot", plot_type="vpc"))
    assert workflow_state_for(ws, "advanced").state == WorkflowStateId.RESULTS_AVAILABLE


def test_advanced_results_available_bootstrap_summary() -> None:
    ws = _ws()
    ws.active_scenario.add_artifact(_artifact("table", role="bootstrap_summary"))
    assert workflow_state_for(ws, "advanced").state == WorkflowStateId.RESULTS_AVAILABLE


# ---------------------------------------------------------------------------
# workflow_state_for — unknown workflow
# ---------------------------------------------------------------------------


def test_unknown_workflow_returns_ready() -> None:
    ws = _ws()
    state = workflow_state_for(ws, "completely_unknown_workflow")
    assert state.state == WorkflowStateId.READY
