"""Unit tests for the new GUI domain and service layers."""

from __future__ import annotations

import json

import pytest

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import EstimationConfig, ModelSpec, ModelSpecMode
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.artifact_service import ArtifactService
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.validation_service import ValidationResult
from openpkpd_gui.services.workflow_state_service import (
    WorkflowStateId,
    describe_fit_input_changes,
    fit_outputs_are_stale,
    workflow_state_for,
)


def test_workspace_round_trip_and_services() -> None:
    project_service = ProjectService()
    artifact_service = ArtifactService()
    workspace = project_service.new_workspace(name="Workspace Demo", root_path="/tmp/project")
    project_service.rename_project(workspace, name="GUI Scaffold")

    dataset = DatasetAsset(
        source_path="data.csv",
        display_name="Data",
        separator=",",
        ignore_char="#",
        input_columns=["ID", "TIME", "_DROP_3", "DV"],
        columns=["ID", "TIME", "DV"],
        row_count=42,
        subject_count=6,
        observation_count=36,
        preview_rows=[{"ID": 1, "TIME": 0.0, "DV": 0.0}],
    )
    model = ModelSpec(problem_title="Example", mode=ModelSpecMode.BUILDER, pk_code="CL=THETA(1)")
    run = RunRecord(workflow="fit")
    run.mark_running()
    run.add_log("running")
    run.mark_succeeded("ok")
    artifact = ArtifactRecord(kind="report", label="HTML report", path="report.html")

    project_service.attach_dataset(workspace, dataset)
    project_service.set_model_spec(workspace, model)
    project_service.add_run(workspace, run)
    artifact_service.register(workspace, artifact)

    restored = Workspace.from_dict(workspace.to_dict())

    assert restored.active_project.name == "GUI Scaffold"
    assert restored.active_dataset is not None
    assert restored.active_dataset.row_count == 42
    assert restored.active_dataset.observation_count == 36
    assert restored.active_dataset.ignore_char == "#"
    assert restored.active_dataset.input_columns == ["ID", "TIME", "_DROP_3", "DV"]
    assert restored.active_model_spec is not None
    assert restored.active_model_spec.problem_title == "Example"
    assert restored.runs[0].status == RunStatus.SUCCEEDED
    assert restored.artifacts[0].label == "HTML report"


def test_model_spec_round_trip_preserves_executable_code_trust_state() -> None:
    model = ModelSpec(
        problem_title="Example",
        mode=ModelSpecMode.BUILDER,
        pk_code="CL=THETA(1)",
        executable_code_trusted=False,
        executable_code_origin="imported_snapshot",
    )

    restored = ModelSpec.from_dict(model.to_dict())

    assert restored.executable_code_trusted is False
    assert restored.executable_code_origin == "imported_snapshot"


def test_workspace_to_dict_handles_recursive_estimation_options() -> None:
    workspace = Workspace(name="Recursive options")
    recursive_options: dict[str, object] = {}
    recursive_options["self"] = recursive_options
    workspace.active_model_spec = ModelSpec(
        problem_title="Example",
        estimation=EstimationConfig(method="FOCE", options=recursive_options),
    )

    payload = workspace.to_dict()

    assert (
        payload["projects"][0]["scenarios"][0]["active_model_spec"]["estimation"]["options"][
            "self"
        ]["self"]
        == "<recursive>"
    )
    json.dumps(payload, sort_keys=True, separators=(",", ":"))


def test_run_record_preserves_cancel_request_round_trip() -> None:
    run = RunRecord(workflow="fit")
    run.mark_running()
    run.mark_cancel_requested()
    payload = run.to_dict()

    restored = RunRecord.from_dict(payload)

    assert restored.status == RunStatus.RUNNING
    assert restored.cancel_requested_at is not None
    assert restored.finished_at is None


def test_workspace_round_trip_preserves_saved_input_timestamps() -> None:
    workspace = Workspace(name="Workspace Demo")
    workspace.active_dataset = DatasetAsset(source_path="data.csv", display_name="Data")
    workspace.active_model_spec = ModelSpec(problem_title="Example")
    workspace.active_scenario.dataset_updated_at = "2026-03-15T10:00:00+00:00"
    workspace.active_scenario.model_updated_at = "2026-03-15T11:00:00+00:00"

    restored = Workspace.from_dict(workspace.to_dict())

    assert restored.active_scenario.dataset_updated_at == "2026-03-15T10:00:00+00:00"
    assert restored.active_scenario.model_updated_at == "2026-03-15T11:00:00+00:00"


def test_workspace_scenario_branching_clones_inputs_without_outputs() -> None:
    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Demo", root_path="/tmp/workspace")

    workspace.active_dataset = DatasetAsset(
        source_path="data.csv",
        display_name="Data",
        separator=",",
        columns=["ID", "TIME", "DV"],
        row_count=3,
    )
    workspace.active_model_spec = ModelSpec(
        problem_title="Baseline model",
        mode=ModelSpecMode.BUILDER,
        pk_code="CL = THETA(1)",
    )
    workspace.add_run(RunRecord(workflow="fit"))
    workspace.add_artifact(
        ArtifactRecord(kind="report", label="Baseline report", path="report.html")
    )

    baseline = workspace.active_scenario
    branch = project_service.create_scenario(workspace, name="Scenario 2")

    assert len(workspace.active_project.scenarios) == 2
    assert branch.parent_scenario_id == baseline.scenario_id
    assert branch.active_dataset is not None
    assert branch.active_dataset.display_name == "Data"
    assert branch.active_model_spec is not None
    assert branch.active_model_spec.problem_title == "Baseline model"
    assert branch.runs == []
    assert branch.artifacts == []

    project = project_service.create_project(workspace, name="Project 2")
    assert workspace.active_project is project
    assert project.active_scenario.name == "Baseline"


def test_workspace_rename_helpers_update_active_project_and_scenario() -> None:
    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Demo")

    project_service.rename_project(workspace, name="Dose Escalation")
    project_service.rename_scenario(workspace, name="Baseline A")

    assert workspace.active_project.name == "Dose Escalation"
    assert workspace.active_scenario.name == "Baseline A"


def test_workspace_note_helpers_normalize_and_clear_notes() -> None:
    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Demo")

    project_service.set_project_notes(workspace, notes="  Lead project\nwith context  ")
    project_service.set_scenario_notes(workspace, notes="  Branch notes  ")

    assert workspace.active_project.metadata["notes"] == "Lead project\nwith context"
    assert workspace.active_scenario.metadata["notes"] == "Branch notes"

    project_service.set_project_notes(workspace, notes="   ")
    project_service.set_scenario_notes(workspace, notes="")

    assert "notes" not in workspace.active_project.metadata
    assert "notes" not in workspace.active_scenario.metadata


def test_workspace_project_details_helpers_normalize_and_clear_metadata() -> None:
    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Demo")

    project_service.update_project_details(
        workspace,
        name="  Dose Escalation  ",
        description="  Lead project\nwith context  ",
        references="  PMID:12345\nDOI:10.1000/example  ",
        notes="  Primary notes  ",
    )

    assert workspace.active_project.name == "Dose Escalation"
    assert workspace.active_project.metadata["description"] == "Lead project\nwith context"
    assert workspace.active_project.metadata["references"] == "PMID:12345\nDOI:10.1000/example"
    assert workspace.active_project.metadata["notes"] == "Primary notes"

    project_service.update_project_details(
        workspace,
        name="Dose Escalation",
        description="   ",
        references="",
        notes="  ",
    )

    assert workspace.active_project.name == "Dose Escalation"
    assert "description" not in workspace.active_project.metadata
    assert "references" not in workspace.active_project.metadata
    assert "notes" not in workspace.active_project.metadata


def test_workspace_scenario_details_helpers_normalize_and_clear_metadata() -> None:
    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Demo")

    project_service.update_scenario_details(
        workspace,
        name="  Scenario A  ",
        description="  Branch description\nwith context  ",
        references="  Ref-001\nDOI:10.1000/scenario  ",
        notes="  Branch notes  ",
    )

    assert workspace.active_scenario.name == "Scenario A"
    assert workspace.active_scenario.metadata["description"] == "Branch description\nwith context"
    assert workspace.active_scenario.metadata["references"] == "Ref-001\nDOI:10.1000/scenario"
    assert workspace.active_scenario.metadata["notes"] == "Branch notes"

    project_service.update_scenario_details(
        workspace,
        name="Scenario A",
        description="   ",
        references="",
        notes="  ",
    )

    assert workspace.active_scenario.name == "Scenario A"
    assert "description" not in workspace.active_scenario.metadata
    assert "references" not in workspace.active_scenario.metadata
    assert "notes" not in workspace.active_scenario.metadata


def test_workspace_duplicate_helpers_clone_saved_inputs_without_outputs() -> None:
    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Demo")

    workspace.active_dataset = DatasetAsset(source_path="data.csv", display_name="Data")
    workspace.active_model_spec = ModelSpec(problem_title="Baseline", pk_code="CL = THETA(1)")
    workspace.active_project.metadata["notes"] = "Project note"
    workspace.active_scenario.metadata["notes"] = "Scenario note"
    workspace.add_run(RunRecord(workflow="fit"))
    workspace.add_artifact(
        ArtifactRecord(kind="report", label="Baseline report", path="report.html")
    )

    baseline = workspace.active_scenario
    source_project = workspace.active_project
    duplicated_scenario = project_service.duplicate_scenario(workspace, name="Baseline Copy")

    assert duplicated_scenario.name == "Baseline Copy"
    assert duplicated_scenario.parent_scenario_id == baseline.scenario_id
    assert duplicated_scenario.active_dataset is not None
    assert duplicated_scenario.active_dataset.display_name == "Data"
    assert duplicated_scenario.active_model_spec is not None
    assert duplicated_scenario.active_model_spec.problem_title == "Baseline"
    assert duplicated_scenario.metadata["notes"] == "Scenario note"
    assert duplicated_scenario.runs == []
    assert duplicated_scenario.artifacts == []

    duplicated_project = project_service.duplicate_project(workspace, name="Project Copy")

    assert duplicated_project.name == "Project Copy"
    assert duplicated_project.metadata["notes"] == "Project note"
    assert len(duplicated_project.scenarios) == len(source_project.scenarios)
    assert duplicated_project.active_scenario.name == "Baseline Copy"
    assert duplicated_project.active_scenario.active_dataset is not None
    assert duplicated_project.active_scenario.active_dataset.display_name == "Data"
    assert duplicated_project.active_scenario.metadata["notes"] == "Scenario note"
    assert duplicated_project.active_scenario.runs == []
    assert duplicated_project.active_scenario.artifacts == []


def test_workflow_state_service_tracks_fit_and_review_lifecycle() -> None:
    project_service = ProjectService()
    artifact_service = ArtifactService()
    workspace = project_service.new_workspace(name="Workspace Demo")

    assert workflow_state_for(workspace, "overview").state == WorkflowStateId.NOT_STARTED
    assert workflow_state_for(workspace, "fit").state == WorkflowStateId.NOT_STARTED
    assert workflow_state_for(workspace, "results").state == WorkflowStateId.NOT_STARTED

    project_service.attach_dataset(
        workspace,
        DatasetAsset(source_path="data.csv", display_name="Data", columns=["ID", "TIME", "DV"]),
    )
    assert workflow_state_for(workspace, "overview").state == WorkflowStateId.READY
    assert workflow_state_for(workspace, "fit").state == WorkflowStateId.NEEDS_ATTENTION

    project_service.set_model_spec(
        workspace,
        ModelSpec(problem_title="Baseline", mode=ModelSpecMode.BUILDER, pk_code="CL = THETA(1)"),
    )
    assert workflow_state_for(workspace, "fit").state == WorkflowStateId.READY
    assert workflow_state_for(workspace, "results").state == WorkflowStateId.READY

    run = RunRecord(workflow="fit")
    run.mark_running()
    project_service.add_run(workspace, run)
    assert workflow_state_for(workspace, "overview").state == WorkflowStateId.RUNNING
    assert workflow_state_for(workspace, "fit").state == WorkflowStateId.RUNNING

    run.mark_succeeded("Objective function minimised")
    artifact_service.register(
        workspace,
        ArtifactRecord(kind="report", label="HTML report", path="report.html"),
    )

    assert workflow_state_for(workspace, "overview").state == WorkflowStateId.RESULTS_AVAILABLE
    assert workflow_state_for(workspace, "fit").state == WorkflowStateId.RESULTS_AVAILABLE
    assert workflow_state_for(workspace, "results").state == WorkflowStateId.RESULTS_AVAILABLE


def test_saved_input_changes_invalidate_scenario_outputs() -> None:
    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Demo")

    baseline_dataset = DatasetAsset(source_path="data.csv", display_name="Data", row_count=3)
    baseline_model = ModelSpec(
        problem_title="Baseline", dataset_path="data.csv", pk_code="CL = THETA(1)"
    )

    project_service.attach_dataset(workspace, baseline_dataset)
    project_service.set_model_spec(workspace, baseline_model)

    preserved_run = RunRecord(workflow="fit")
    preserved_artifact = ArtifactRecord(kind="report", label="Baseline report", path="report.html")
    project_service.add_run(workspace, preserved_run)
    project_service.add_artifact(workspace, preserved_artifact)

    project_service.attach_dataset(workspace, DatasetAsset.from_dict(baseline_dataset.to_dict()))
    project_service.set_model_spec(workspace, ModelSpec.from_dict(baseline_model.to_dict()))

    assert [run.run_id for run in workspace.runs] == [preserved_run.run_id]
    assert [artifact.artifact_id for artifact in workspace.artifacts] == [
        preserved_artifact.artifact_id
    ]

    project_service.attach_dataset(
        workspace,
        DatasetAsset(source_path="updated.csv", display_name="Updated", row_count=4),
    )

    assert workspace.runs == []
    assert workspace.artifacts == []

    project_service.add_run(workspace, RunRecord(workflow="fit"))
    project_service.add_artifact(
        workspace,
        ArtifactRecord(kind="plot", label="GOF", path="gof.png"),
    )

    project_service.set_model_spec(
        workspace,
        ModelSpec(problem_title="Updated", dataset_path="updated.csv", pk_code="V = THETA(2)"),
    )

    assert workspace.runs == []
    assert workspace.artifacts == []


def test_fit_output_staleness_tracks_dataset_and_model_changes_since_latest_success() -> None:
    workspace = Workspace(name="Workspace Demo")
    run = RunRecord(workflow="fit")
    run.mark_running()
    run.mark_succeeded("ok")
    run.finished_at = "2026-03-15T12:00:00+00:00"
    workspace.active_scenario.runs = [run]
    workspace.active_scenario.dataset_updated_at = "2026-03-15T12:05:00+00:00"
    workspace.active_scenario.model_updated_at = "2026-03-15T12:06:00+00:00"

    assert fit_outputs_are_stale(workspace.active_scenario) is True
    assert describe_fit_input_changes(workspace.active_scenario) == "Dataset and model changed"


def test_fit_output_staleness_is_clear_when_inputs_predate_latest_success() -> None:
    workspace = Workspace(name="Workspace Demo")
    run = RunRecord(workflow="fit")
    run.mark_running()
    run.mark_succeeded("ok")
    run.finished_at = "2026-03-15T12:00:00+00:00"
    workspace.active_scenario.runs = [run]
    workspace.active_scenario.dataset_updated_at = "2026-03-15T11:59:00+00:00"
    workspace.active_scenario.model_updated_at = "2026-03-15T11:58:00+00:00"

    assert fit_outputs_are_stale(workspace.active_scenario) is False
    assert describe_fit_input_changes(workspace.active_scenario) is None


def test_resaving_same_dataset_and_model_preserves_input_timestamps_and_fresh_outputs() -> None:
    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Demo")
    dataset = DatasetAsset(source_path="data.csv", display_name="Data", row_count=3)
    model = ModelSpec(problem_title="Baseline", dataset_path="data.csv", pk_code="CL = THETA(1)")

    project_service.attach_dataset(workspace, dataset)
    project_service.set_model_spec(workspace, model)
    workspace.active_scenario.dataset_updated_at = "2026-03-15T11:00:00+00:00"
    workspace.active_scenario.model_updated_at = "2026-03-15T11:30:00+00:00"

    run = RunRecord(workflow="fit")
    run.mark_running()
    run.mark_succeeded("ok")
    run.finished_at = "2026-03-15T12:00:00+00:00"
    workspace.active_scenario.runs = [run]

    project_service.attach_dataset(
        workspace,
        DatasetAsset(source_path="data.csv", display_name="Data", row_count=3),
    )
    project_service.set_model_spec(
        workspace,
        ModelSpec(problem_title="Baseline", dataset_path="data.csv", pk_code="CL = THETA(1)"),
    )

    assert workspace.active_scenario.dataset_updated_at == "2026-03-15T11:00:00+00:00"
    assert workspace.active_scenario.model_updated_at == "2026-03-15T11:30:00+00:00"
    assert fit_outputs_are_stale(workspace.active_scenario) is False
    assert describe_fit_input_changes(workspace.active_scenario) is None


def test_workspace_delete_helpers_select_fallback_project_and_scenario() -> None:
    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Demo")

    project_service.rename_project(workspace, name="Project A")
    project_service.rename_scenario(workspace, name="Baseline A")
    project_service.create_scenario(workspace, name="Variant A")
    project_b = project_service.create_project(workspace, name="Project B")
    project_service.rename_scenario(workspace, name="Baseline B")
    project_service.create_scenario(workspace, name="Variant B")

    removed_scenario = project_service.delete_scenario(workspace)

    assert removed_scenario.name == "Variant B"
    assert workspace.active_project is project_b
    assert workspace.active_scenario.name == "Baseline B"

    removed_project = project_service.delete_project(workspace)

    assert removed_project.name == "Project B"
    assert workspace.active_project.name == "Project A"
    assert workspace.active_scenario.name == "Variant A"


def test_workspace_delete_helpers_reject_last_project_and_scenario() -> None:
    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Demo")

    with pytest.raises(ValueError):
        project_service.delete_project(workspace)

    with pytest.raises(ValueError):
        project_service.delete_scenario(workspace)


def test_validation_result_reports_error_state() -> None:
    result = ValidationResult()
    result.add_warning("Missing optional column", field_name="AMT")
    assert result.ok is True

    result.add_error("Missing required column", field_name="DV")
    assert result.ok is False
    assert len(result.issues) == 2


def test_validation_result_preserves_optional_routing_metadata() -> None:
    result = ValidationResult()

    result.add_error(
        "Load a dataset first.",
        field_name="active_dataset",
        target_workflow="data",
        target_widget="data-source-path",
    )

    assert result.issues[0].target_workflow == "data"
    assert result.issues[0].target_widget == "data-source-path"
