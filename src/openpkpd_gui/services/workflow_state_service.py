"""Shared workflow-state computation for the desktop GUI shell."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.model_spec import ModelSpec, ModelSpecMode
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Scenario, Workspace


class WorkflowStateId(StrEnum):
    """High-level user-facing state for one workflow."""

    NOT_STARTED = "not_started"
    NEEDS_ATTENTION = "needs_attention"
    READY = "ready"
    RUNNING = "running"
    RESULTS_AVAILABLE = "results_available"


@dataclass(frozen=True, slots=True)
class WorkflowState:
    """Computed state and summary for one workflow node."""

    workflow_id: str
    state: WorkflowStateId
    label: str
    summary: str


def _state(workflow_id: str, state: WorkflowStateId, summary: str) -> WorkflowState:
    labels = {
        WorkflowStateId.NOT_STARTED: "Not started",
        WorkflowStateId.NEEDS_ATTENTION: "Needs attention",
        WorkflowStateId.READY: "Ready",
        WorkflowStateId.RUNNING: "Running",
        WorkflowStateId.RESULTS_AVAILABLE: "Results available",
    }
    return WorkflowState(workflow_id=workflow_id, state=state, label=labels[state], summary=summary)


def resolve_scenario(
    workspace: Workspace,
    *,
    project_id: str | None = None,
    scenario_id: str | None = None,
) -> Scenario:
    """Resolve a scenario for workflow-state evaluation."""
    if scenario_id is not None:
        resolved = workspace.find_scenario(scenario_id, project_id=project_id)
        if resolved is not None:
            _project, scenario = resolved
            return scenario
    if project_id is not None:
        project = workspace.find_project(project_id)
        if project is not None:
            return project.active_scenario
    return workspace.active_scenario


def latest_run_for_workflow(scenario: Scenario, workflow_id: str) -> RunRecord | None:
    """Return the newest run for one workflow in one scenario."""
    for run in reversed(scenario.runs):
        if run.workflow == workflow_id:
            return run
    return None


def latest_successful_run_for_workflow(scenario: Scenario, workflow_id: str) -> RunRecord | None:
    """Return the newest successful run for one workflow in one scenario."""
    for run in reversed(scenario.runs):
        if run.workflow == workflow_id and run.status == RunStatus.SUCCEEDED:
            return run
    return None


def latest_run_for_workflows(scenario: Scenario, workflow_ids: set[str]) -> RunRecord | None:
    """Return the newest run matching any workflow in one scenario."""
    for run in reversed(scenario.runs):
        if run.workflow in workflow_ids:
            return run
    return None


def latest_artifact(
    scenario: Scenario,
    *,
    kind: str | None = None,
    role: str | None = None,
    plot_type: str | None = None,
) -> ArtifactRecord | None:
    """Return the newest scenario artifact matching the requested criteria."""
    for artifact in reversed(scenario.artifacts):
        if kind is not None and artifact.kind != kind:
            continue
        artifact_role = str(artifact.metadata.get("artifact_role") or artifact.kind or "").strip()
        if role is not None and artifact_role != role:
            continue
        artifact_plot_type = artifact.metadata.get("plot_type")
        if plot_type is not None and str(artifact_plot_type or "") != plot_type:
            continue
        return artifact
    return None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def fit_input_changes_since_latest_successful_fit(scenario: Scenario) -> tuple[str, ...]:
    """Return input kinds changed since the latest successful fit."""
    latest_fit = latest_successful_run_for_workflow(scenario, "fit")
    latest_fit_finished_at = _parse_timestamp(
        latest_fit.finished_at if latest_fit is not None else None
    )
    latest_fit_run = latest_run_for_workflow(scenario, "fit")
    if latest_fit_finished_at is None:
        return ()
    if latest_fit_run is not None and latest_fit_run.status == RunStatus.RUNNING:
        return ()
    changed_inputs: list[str] = []
    dataset_updated_at = _parse_timestamp(scenario.dataset_updated_at)
    model_updated_at = _parse_timestamp(scenario.model_updated_at)
    if dataset_updated_at is not None and dataset_updated_at > latest_fit_finished_at:
        changed_inputs.append("dataset")
    if model_updated_at is not None and model_updated_at > latest_fit_finished_at:
        changed_inputs.append("model")
    return tuple(changed_inputs)


def fit_outputs_are_stale(scenario: Scenario) -> bool:
    """Return whether current saved fit-review outputs are stale."""
    return bool(fit_input_changes_since_latest_successful_fit(scenario))


def describe_fit_input_changes(scenario: Scenario) -> str | None:
    """Return a user-facing subject for stale fit inputs, if any."""
    changes = fit_input_changes_since_latest_successful_fit(scenario)
    if not changes:
        return None
    if changes == ("dataset",):
        return "Dataset changed"
    if changes == ("model",):
        return "Model changed"
    return "Dataset and model changed"


def workflow_state_for(
    workspace: Workspace,
    workflow_id: str,
    *,
    project_id: str | None = None,
    scenario_id: str | None = None,
) -> WorkflowState:
    """Compute the current state for one workflow in one scenario."""
    scenario = resolve_scenario(workspace, project_id=project_id, scenario_id=scenario_id)
    has_dataset = scenario.active_dataset is not None
    active_model_spec = scenario.active_model_spec
    has_model = active_model_spec is not None
    covariate_model_ready = (
        isinstance(active_model_spec, ModelSpec) and active_model_spec.mode == ModelSpecMode.BUILDER
    )
    fit_run = latest_run_for_workflow(scenario, "fit")
    nca_run = latest_run_for_workflow(scenario, "nca")
    covariate_run = latest_run_for_workflow(scenario, "covariate")
    advanced_run = latest_run_for_workflows(scenario, {"vpc", "bootstrap", "design"})
    has_plot = latest_artifact(scenario, kind="plot") is not None
    has_report = latest_artifact(scenario, kind="report") is not None
    has_nca_output = latest_artifact(scenario, role="nca_summary") is not None
    has_diagnostics_output = any(
        latest_artifact(scenario, role=role) is not None
        for role in ("diagnostics_table", "npde_table")
    ) or any(
        latest_artifact(scenario, plot_type=plot_type) is not None
        for plot_type in ("gof_panel", "residual_trends", "cwres_vs_time", "cwres_vs_pred")
    )
    has_advanced_output = (
        any(
            latest_artifact(scenario, role=role) is not None
            for role in (
                "vpc_summary",
                "bootstrap_summary",
                "bootstrap_ci_table",
                "bootstrap_samples",
                "design_summary",
                "design_metrics",
                "design_schedule",
                "design_fim",
                "design_expected_se",
            )
        )
        or latest_artifact(scenario, plot_type="vpc") is not None
    )

    if workflow_id == "home":
        return _state(
            "home", WorkflowStateId.READY, "Workspace navigation and recent work are available."
        )
    if workflow_id == "overview":
        if any(run.status == RunStatus.RUNNING for run in scenario.runs):
            return _state(
                "overview",
                WorkflowStateId.RUNNING,
                "One or more scenario workflows are currently running.",
            )
        if any(run.status == RunStatus.FAILED for run in scenario.runs):
            return _state(
                "overview",
                WorkflowStateId.NEEDS_ATTENTION,
                "A recent scenario run failed and should be reviewed.",
            )
        if scenario.runs or scenario.artifacts:
            return _state(
                "overview",
                WorkflowStateId.RESULTS_AVAILABLE,
                "This scenario already has recent runs or saved outputs to review.",
            )
        if has_dataset or has_model:
            return _state(
                "overview",
                WorkflowStateId.READY,
                "Saved inputs are available and the scenario is ready for the next step.",
            )
        return _state(
            "overview",
            WorkflowStateId.NOT_STARTED,
            "Start by importing a dataset or opening a saved scenario snapshot.",
        )
    if workflow_id == "data":
        if has_dataset:
            return _state(
                "data", WorkflowStateId.READY, "A saved dataset is available for this scenario."
            )
        if has_model or scenario.runs or scenario.artifacts:
            return _state(
                "data",
                WorkflowStateId.NEEDS_ATTENTION,
                "Downstream work exists without a saved dataset.",
            )
        return _state(
            "data",
            WorkflowStateId.NOT_STARTED,
            "No dataset has been imported for this scenario yet.",
        )
    if workflow_id == "model":
        if has_model:
            return _state(
                "model", WorkflowStateId.READY, "A saved model specification is available."
            )
        if has_dataset:
            return _state(
                "model",
                WorkflowStateId.NEEDS_ATTENTION,
                "A dataset is ready; open Model to configure one.",
            )
        return _state(
            "model",
            WorkflowStateId.NOT_STARTED,
            "No saved model is available for this scenario yet.",
        )
    if workflow_id == "fit":
        if fit_run is not None:
            if fit_run.status == RunStatus.RUNNING:
                return _state(
                    "fit",
                    WorkflowStateId.RUNNING,
                    "A fit run is currently executing in the background.",
                )
            if fit_run.status == RunStatus.FAILED:
                return _state(
                    "fit",
                    WorkflowStateId.NEEDS_ATTENTION,
                    "The latest fit failed and needs review before rerunning.",
                )
            if fit_run.status == RunStatus.SUCCEEDED:
                return _state(
                    "fit",
                    WorkflowStateId.RESULTS_AVAILABLE,
                    "The latest fit succeeded and generated reviewable outputs.",
                )
        if has_dataset and has_model:
            return _state(
                "fit", WorkflowStateId.READY, "Saved dataset and model are available for fitting."
            )
        if has_dataset or has_model:
            return _state(
                "fit",
                WorkflowStateId.NEEDS_ATTENTION,
                "Fit needs both a saved dataset and a saved model.",
            )
        return _state(
            "fit", WorkflowStateId.NOT_STARTED, "Fit prerequisites have not been prepared yet."
        )
    if workflow_id == "nca":
        if nca_run is not None:
            if nca_run.status == RunStatus.RUNNING:
                return _state(
                    "nca",
                    WorkflowStateId.RUNNING,
                    "An NCA run is currently executing in the background.",
                )
            if nca_run.status == RunStatus.FAILED:
                return _state(
                    "nca",
                    WorkflowStateId.NEEDS_ATTENTION,
                    "The latest NCA run failed and needs review.",
                )
        if has_nca_output:
            return _state(
                "nca",
                WorkflowStateId.RESULTS_AVAILABLE,
                "Latest NCA output is available for preview and export.",
            )
        if has_dataset:
            return _state(
                "nca", WorkflowStateId.READY, "A saved dataset is available for standalone NCA."
            )
        return _state(
            "nca", WorkflowStateId.NOT_STARTED, "NCA needs a saved dataset before it can run."
        )
    if workflow_id == "results":
        if fit_run is not None and fit_run.status == RunStatus.RUNNING:
            return _state(
                "results",
                WorkflowStateId.RUNNING,
                "A fit run is in progress; new results will appear after completion.",
            )
        if has_report or scenario.runs:
            return _state(
                "results",
                WorkflowStateId.RESULTS_AVAILABLE,
                "Run history and saved result artifacts are available.",
            )
        if has_dataset and has_model:
            return _state(
                "results",
                WorkflowStateId.READY,
                "Run this scenario to populate result history and artifacts.",
            )
        return _state(
            "results",
            WorkflowStateId.NOT_STARTED,
            "Results will appear after this scenario has been run.",
        )
    if workflow_id == "plots":
        if fit_run is not None and fit_run.status == RunStatus.RUNNING:
            return _state(
                "plots",
                WorkflowStateId.RUNNING,
                "A fit run is still running; plot outputs are not final yet.",
            )
        if has_plot:
            return _state(
                "plots",
                WorkflowStateId.RESULTS_AVAILABLE,
                "Saved plot artifacts are available for review.",
            )
        if fit_run is not None and fit_run.status == RunStatus.SUCCEEDED:
            return _state(
                "plots",
                WorkflowStateId.READY,
                "A successful fit exists; plot outputs can now be reviewed as they appear.",
            )
        if fit_run is not None and fit_run.status == RunStatus.FAILED:
            return _state(
                "plots",
                WorkflowStateId.NEEDS_ATTENTION,
                "The latest fit failed, so expected plot outputs may be missing.",
            )
        return _state(
            "plots",
            WorkflowStateId.NOT_STARTED,
            "Plot outputs will appear after a successful fit produces them.",
        )
    if workflow_id == "diagnostics":
        if fit_run is not None and fit_run.status == RunStatus.RUNNING:
            return _state(
                "diagnostics",
                WorkflowStateId.RUNNING,
                "A fit run is still running; diagnostics will be clearer after completion.",
            )
        if has_diagnostics_output:
            return _state(
                "diagnostics",
                WorkflowStateId.RESULTS_AVAILABLE,
                "Diagnostics tables or review plots are available.",
            )
        if fit_run is not None and fit_run.status == RunStatus.SUCCEEDED:
            return _state(
                "diagnostics",
                WorkflowStateId.READY,
                "The latest fit succeeded; diagnostics review is now available.",
            )
        if fit_run is not None and fit_run.status == RunStatus.FAILED:
            return _state(
                "diagnostics",
                WorkflowStateId.NEEDS_ATTENTION,
                "The latest fit failed; resolve fit issues before expecting diagnostics output.",
            )
        return _state(
            "diagnostics",
            WorkflowStateId.NOT_STARTED,
            "Diagnostics need a completed fit before outputs can be reviewed.",
        )
    if workflow_id == "covariate":
        if covariate_run is not None:
            if covariate_run.status == RunStatus.RUNNING:
                return _state(
                    "covariate",
                    WorkflowStateId.RUNNING,
                    "A covariate search is running in the background.",
                )
            if covariate_run.status == RunStatus.FAILED:
                return _state(
                    "covariate",
                    WorkflowStateId.NEEDS_ATTENTION,
                    "The latest covariate search failed and needs review.",
                )
            if covariate_run.status == RunStatus.SUCCEEDED:
                return _state(
                    "covariate",
                    WorkflowStateId.RESULTS_AVAILABLE,
                    "Covariate search results are available for review.",
                )
        if has_dataset and has_model and not covariate_model_ready:
            return _state(
                "covariate",
                WorkflowStateId.NEEDS_ATTENTION,
                "SCM requires a builder-mode model. Switch the Model workflow from Control Stream to Builder mode.",
            )
        if has_dataset and has_model:
            return _state(
                "covariate",
                WorkflowStateId.READY,
                "Saved dataset and base model are available for SCM.",
            )
        if has_dataset or has_model:
            return _state(
                "covariate",
                WorkflowStateId.NEEDS_ATTENTION,
                "Covariate search needs both a saved dataset and a saved model.",
            )
        return _state(
            "covariate",
            WorkflowStateId.NOT_STARTED,
            "Covariate search prerequisites have not been prepared yet.",
        )
    if workflow_id == "advanced":
        if advanced_run is not None and advanced_run.status == RunStatus.RUNNING:
            return _state(
                "advanced",
                WorkflowStateId.RUNNING,
                "A validation or design workflow is currently executing.",
            )
        if has_advanced_output:
            return _state(
                "advanced",
                WorkflowStateId.RESULTS_AVAILABLE,
                "Validation or design artifacts are available for review.",
            )
        if fit_run is not None and fit_run.status == RunStatus.SUCCEEDED:
            return _state(
                "advanced",
                WorkflowStateId.READY,
                "A successful fit is available for VPC, bootstrap, or design work.",
            )
        if fit_run is not None and fit_run.status == RunStatus.FAILED:
            return _state(
                "advanced",
                WorkflowStateId.NEEDS_ATTENTION,
                "The latest fit failed; validation and design need a successful fit first.",
            )
        return _state(
            "advanced",
            WorkflowStateId.NOT_STARTED,
            "Validation and design workflows unlock after a successful fit.",
        )
    return _state(workflow_id, WorkflowStateId.READY, "Workflow state is available.")


def workflow_states_for(
    workspace: Workspace,
    workflow_ids: tuple[str, ...],
    *,
    project_id: str | None = None,
    scenario_id: str | None = None,
) -> dict[str, WorkflowState]:
    """Return workflow states for a set of workflow ids."""
    return {
        workflow_id: workflow_state_for(
            workspace,
            workflow_id,
            project_id=project_id,
            scenario_id=scenario_id,
        )
        for workflow_id in workflow_ids
    }
