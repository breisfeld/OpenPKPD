"""Tests for workspace snapshot serialization."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec
from openpkpd_gui.domain.run_record import RunRecord
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.serialization_service import ProjectSnapshotService, SnapshotPayload


def _archive_slug(value: str, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return cleaned or default


def test_workspace_snapshot_round_trip_preserves_inputs_outputs_and_provenance(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "theo.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n1,1,5\n", encoding="utf-8")
    report_path = tmp_path / "report.html"
    report_path.write_text("<html><body>report</body></html>", encoding="utf-8")

    workspace = Workspace(name="Serialization Demo", root_path=str(tmp_path))
    workspace.active_dataset = DatasetAsset(
        source_path=str(dataset_path),
        display_name="theo.csv",
        columns=["ID", "TIME", "DV"],
        row_count=2,
        preview_rows=[{"ID": 1, "TIME": 0, "DV": 0}],
    )
    workspace.active_model_spec = ModelSpec(
        problem_title="Theophylline",
        dataset_path=str(dataset_path),
        pk_code="CL = THETA(1)",
        error_code="Y = F",
        theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
        omega_values=[[0.3]],
        sigma_values=[[0.1]],
    )

    run = RunRecord(workflow="fit")
    run.mark_running()
    run.add_log("[progress] Building model")
    run.mark_succeeded("Fit completed")
    workspace.add_run(run)

    report_artifact = ArtifactRecord(
        kind="report",
        label="Fit report",
        path=str(report_path),
        source_run_id=run.run_id,
        metadata={"format": "html"},
    )
    plot_artifact = ArtifactRecord(
        kind="plot",
        label="GOF panel",
        source_run_id=run.run_id,
        metadata={"plot_family": "gof"},
    )
    workspace.add_artifact(report_artifact)
    workspace.add_artifact(plot_artifact)
    run.artifact_ids.extend([report_artifact.artifact_id, plot_artifact.artifact_id])

    service = ProjectSnapshotService()
    snapshot_path = tmp_path / "demo.opkpd"
    manifest = service.save_snapshot(
        workspace,
        snapshot_path,
        artifact_payloads={
            plot_artifact.artifact_id: SnapshotPayload(
                file_name="gof.png",
                data=b"\x89PNG\r\nplot-bytes",
                media_type="image/png",
                metadata={"title": "GOF panel"},
            )
        },
    )

    assert snapshot_path.exists()
    assert manifest.format_name == "openpkpd.gui.project_package"
    assert manifest.format_version == 3
    assert manifest.provenance["workspace_name"] == "Serialization Demo"
    assert manifest.provenance["workspace_id"] == workspace.workspace_id
    assert manifest.provenance["active_project_id"] == workspace.active_project.project_id
    assert manifest.provenance["active_scenario_id"] == workspace.active_scenario.scenario_id
    assert any(resource.role == "scenario_dataset" for resource in manifest.resources)
    assert any(resource.media_type == "image/png" for resource in manifest.resources)

    with zipfile.ZipFile(snapshot_path, "r") as archive:
        archive_names = set(archive.namelist())
        project_dir = (
            f"projects/{_archive_slug(workspace.active_project.name, 'project')}"
            f"-{workspace.active_project.project_id}"
        )
        scenario_dir = (
            f"{project_dir}/scenarios/{_archive_slug(workspace.active_scenario.name, 'scenario')}"
            f"-{workspace.active_scenario.scenario_id}"
        )
        assert "workspace.json" in archive_names
        assert f"{project_dir}/" in archive_names
        assert f"{project_dir}/metadata.json" in archive_names
        assert f"{scenario_dir}/" in archive_names
        assert f"{scenario_dir}/metadata.json" in archive_names
        assert f"{scenario_dir}/data/" in archive_names
        assert f"{scenario_dir}/models/" in archive_names
        assert f"{scenario_dir}/outputs/" in archive_names
        assert f"{scenario_dir}/reports/" in archive_names
        assert f"{scenario_dir}/results/" in archive_names
        assert f"{scenario_dir}/plots/" in archive_names
        assert f"{scenario_dir}/models/model_spec.json" in archive_names
        assert f"{scenario_dir}/outputs/runs.json" in archive_names
        assert f"{scenario_dir}/results/artifacts.json" in archive_names
        assert any(
            path.startswith(f"{scenario_dir}/data/") and path != f"{scenario_dir}/data/"
            for path in archive_names
        )
        assert any(
            path.startswith(f"{scenario_dir}/reports/") and path.endswith("report.html")
            for path in archive_names
        )
        assert any(
            path.startswith(f"{scenario_dir}/plots/") and path.endswith("gof.png")
            for path in archive_names
        )

        manifest_payload = json.loads(archive.read("workspace.json").decode("utf-8"))
        assert manifest_payload["format_name"] == "openpkpd.gui.project_package"
        assert manifest_payload["format_version"] == 3
        assert manifest_payload["project"]["workspace_id"] == workspace.workspace_id
        assert (
            manifest_payload["project"]["active_project_id"] == workspace.active_project.project_id
        )
        assert (
            manifest_payload["provenance"]["active_scenario_id"]
            == workspace.active_scenario.scenario_id
        )
        assert len(manifest_payload["resources"]) == 3

        project_metadata = json.loads(archive.read(f"{project_dir}/metadata.json").decode("utf-8"))
        assert project_metadata["project"]["project_id"] == workspace.active_project.project_id
        assert (
            project_metadata["provenance"]["active_scenario_id"]
            == workspace.active_scenario.scenario_id
        )

        scenario_metadata = json.loads(
            archive.read(f"{scenario_dir}/metadata.json").decode("utf-8")
        )
        assert scenario_metadata["scenario"]["scenario_id"] == workspace.active_scenario.scenario_id
        assert scenario_metadata["dataset"]["display_name"] == "theo.csv"
        assert scenario_metadata["model_spec_path"] == "models/model_spec.json"
        assert scenario_metadata["runs_path"] == "outputs/runs.json"
        assert scenario_metadata["artifacts_path"] == "results/artifacts.json"
        assert len(scenario_metadata["resources"]) == 3

    loaded = service.load_snapshot(snapshot_path, extract_dir=tmp_path / "extracted")

    assert loaded.project.name == workspace.name
    assert loaded.project.active_model_spec is not None
    assert loaded.project.active_model_spec.problem_title == "Theophylline"
    assert loaded.project.runs[0].summary_text == "Fit completed"
    assert loaded.project.active_dataset is not None
    assert Path(loaded.project.active_dataset.source_path or "").read_text(encoding="utf-8") == (
        dataset_path.read_text(encoding="utf-8")
    )

    restored_report = next(
        artifact for artifact in loaded.project.artifacts if artifact.kind == "report"
    )
    restored_plot = next(
        artifact for artifact in loaded.project.artifacts if artifact.kind == "plot"
    )
    assert Path(restored_report.path or "").read_text(encoding="utf-8") == report_path.read_text(
        encoding="utf-8"
    )
    assert Path(restored_plot.path or "").read_bytes() == b"\x89PNG\r\nplot-bytes"
    assert loaded.manifest.provenance["openpkpd_version"]


def test_workspace_snapshot_round_trip_preserves_multiple_scenarios(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    baseline_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")
    branch_path = tmp_path / "branch.csv"
    branch_path.write_text("ID,TIME,DV\n1,0,10\n", encoding="utf-8")
    report_path = tmp_path / "baseline-report.html"
    report_path.write_text("<html><body>baseline</body></html>", encoding="utf-8")

    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Snapshot", root_path=str(tmp_path))
    workspace.active_dataset = DatasetAsset(
        source_path=str(baseline_path), display_name="baseline.csv"
    )
    workspace.active_model_spec = ModelSpec(
        problem_title="Baseline",
        dataset_path=str(baseline_path),
        pk_code="CL = THETA(1)",
        error_code="Y = F",
        theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
        omega_values=[[0.3]],
        sigma_values=[[0.1]],
    )
    baseline_run = RunRecord(workflow="fit")
    baseline_run.mark_succeeded("Baseline finished")
    workspace.add_run(baseline_run)
    workspace.add_artifact(
        ArtifactRecord(
            kind="report",
            label="Baseline report",
            path=str(report_path),
            source_run_id=baseline_run.run_id,
        )
    )
    baseline_scenario_id = workspace.active_scenario.scenario_id

    branch = project_service.create_scenario(workspace, name="Branch")
    branch.active_dataset = DatasetAsset(source_path=str(branch_path), display_name="branch.csv")
    branch.active_model_spec = ModelSpec(
        problem_title="Branch",
        dataset_path=str(branch_path),
        pk_code="CL = THETA(1)",
        error_code="Y = F",
        theta_rows=[{"init": 2.0, "lower": 0.0, "upper": 10.0}],
        omega_values=[[0.2]],
        sigma_values=[[0.05]],
    )

    service = ProjectSnapshotService()
    snapshot_path = tmp_path / "workspace.opkpd"
    manifest = service.save_snapshot(workspace, snapshot_path)

    dataset_resources = [
        resource for resource in manifest.resources if resource.role == "scenario_dataset"
    ]
    assert len(dataset_resources) == 2
    assert {resource.scenario_id for resource in dataset_resources} == {
        baseline_scenario_id,
        branch.scenario_id,
    }

    loaded = service.load_snapshot(snapshot_path, extract_dir=tmp_path / "workspace-extracted")
    assert len(loaded.project.projects) == 1
    assert len(loaded.project.projects[0].scenarios) == 2

    loaded_baseline_match = loaded.project.find_scenario(baseline_scenario_id)
    loaded_branch_match = loaded.project.find_scenario(branch.scenario_id)
    assert loaded_baseline_match is not None
    assert loaded_branch_match is not None

    _, loaded_baseline = loaded_baseline_match
    _, loaded_branch = loaded_branch_match
    assert Path(loaded_baseline.active_dataset.source_path or "").read_text(
        encoding="utf-8"
    ) == baseline_path.read_text(encoding="utf-8")
    assert Path(loaded_branch.active_dataset.source_path or "").read_text(
        encoding="utf-8"
    ) == branch_path.read_text(encoding="utf-8")
    assert Path(loaded_baseline.artifacts[0].path or "").read_text(
        encoding="utf-8"
    ) == report_path.read_text(encoding="utf-8")
    assert loaded_branch.runs == []
    assert loaded_branch.artifacts == []


def test_project_snapshot_service_exports_active_project_as_standalone_workspace(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.csv"
    variant_path = tmp_path / "variant.csv"
    baseline_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")
    variant_path.write_text("ID,TIME,DV\n1,0,10\n", encoding="utf-8")

    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Snapshot", root_path=str(tmp_path))
    workspace.recent_files = [str(tmp_path / "workspace.opkpd")]
    workspace.metadata["owner"] = "OpenPKPD"
    project_service.attach_dataset(
        workspace,
        DatasetAsset(source_path=str(baseline_path), display_name="baseline.csv"),
    )
    project = project_service.create_project(workspace, name="Dose Escalation")
    project_service.attach_dataset(
        workspace,
        DatasetAsset(source_path=str(variant_path), display_name="variant.csv"),
    )
    baseline_scenario_id = project.active_scenario.scenario_id
    branch = project_service.create_scenario(workspace, name="High Dose")
    project_service.attach_dataset(
        workspace,
        DatasetAsset(source_path=str(variant_path), display_name="variant.csv"),
    )

    service = ProjectSnapshotService()
    export_workspace = service.export_workspace_for_project(
        workspace, project_id=project.project_id
    )

    assert export_workspace.name == "Dose Escalation"
    assert export_workspace.root_path is None
    assert export_workspace.recent_files == []
    assert export_workspace.metadata == workspace.metadata
    assert len(export_workspace.projects) == 1
    assert export_workspace.active_project.name == "Dose Escalation"
    assert {scenario.name for scenario in export_workspace.active_project.scenarios} == {
        "Baseline",
        "High Dose",
    }

    snapshot_path = tmp_path / "dose-escalation.opkpd"
    manifest = service.save_snapshot(export_workspace, snapshot_path)

    dataset_resources = [
        resource for resource in manifest.resources if resource.role == "scenario_dataset"
    ]
    assert len(dataset_resources) == 2
    assert {resource.scenario_id for resource in dataset_resources} == {
        baseline_scenario_id,
        branch.scenario_id,
    }

    loaded = service.load_snapshot(snapshot_path, extract_dir=tmp_path / "project-extracted")
    assert loaded.project.name == "Dose Escalation"
    assert loaded.project.recent_files == []
    assert len(loaded.project.projects) == 1
    assert loaded.project.active_project.name == "Dose Escalation"
    assert {scenario.name for scenario in loaded.project.active_project.scenarios} == {
        "Baseline",
        "High Dose",
    }


def test_project_snapshot_service_exports_active_scenario_as_standalone_workspace(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.csv"
    branch_path = tmp_path / "branch.csv"
    baseline_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")
    branch_path.write_text("ID,TIME,DV\n1,0,20\n", encoding="utf-8")

    project_service = ProjectService()
    workspace = project_service.new_workspace(name="Workspace Snapshot", root_path=str(tmp_path))
    project_service.attach_dataset(
        workspace,
        DatasetAsset(source_path=str(baseline_path), display_name="baseline.csv"),
    )
    branch = project_service.create_scenario(workspace, name="Branch A")
    project_service.attach_dataset(
        workspace,
        DatasetAsset(source_path=str(branch_path), display_name="branch.csv"),
    )

    service = ProjectSnapshotService()
    export_workspace = service.export_workspace_for_scenario(
        workspace, scenario_id=branch.scenario_id
    )

    assert export_workspace.name == "Project 1 - Branch A"
    assert export_workspace.root_path is None
    assert export_workspace.recent_files == []
    assert len(export_workspace.projects) == 1
    assert export_workspace.active_project.name == "Project 1"
    assert len(export_workspace.active_project.scenarios) == 1
    assert export_workspace.active_scenario.name == "Branch A"
    assert export_workspace.active_scenario.parent_scenario_id == branch.parent_scenario_id

    snapshot_path = tmp_path / "branch-a.opkpd"
    manifest = service.save_snapshot(export_workspace, snapshot_path)

    dataset_resources = [
        resource for resource in manifest.resources if resource.role == "scenario_dataset"
    ]
    assert len(dataset_resources) == 1
    assert dataset_resources[0].scenario_id == branch.scenario_id

    loaded = service.load_snapshot(snapshot_path, extract_dir=tmp_path / "scenario-extracted")
    assert loaded.project.name == "Project 1 - Branch A"
    assert len(loaded.project.projects) == 1
    assert len(loaded.project.active_project.scenarios) == 1
    assert loaded.project.active_scenario.name == "Branch A"
    assert Path(loaded.project.active_scenario.active_dataset.source_path or "").read_text(
        encoding="utf-8"
    ) == (branch_path.read_text(encoding="utf-8"))


def test_project_service_imports_active_project_snapshot_with_outputs_and_lineage(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.csv"
    branch_path = tmp_path / "branch.csv"
    report_path = tmp_path / "branch-report.html"
    baseline_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")
    branch_path.write_text("ID,TIME,DV\n1,0,15\n", encoding="utf-8")
    report_path.write_text("<html><body>imported project</body></html>", encoding="utf-8")

    project_service = ProjectService()
    source_workspace = project_service.new_workspace(
        name="Source Workspace", root_path=str(tmp_path)
    )
    source_project = project_service.create_project(source_workspace, name="Dose Escalation")
    project_service.attach_dataset(
        source_workspace,
        DatasetAsset(source_path=str(baseline_path), display_name="baseline.csv"),
    )
    project_service.set_model_spec(
        source_workspace,
        ModelSpec(
            problem_title="Baseline",
            dataset_path=str(baseline_path),
            pk_code="CL = THETA(1)",
            error_code="Y = F",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        ),
    )
    project_service.set_project_notes(source_workspace, notes="Imported project")
    branch = project_service.create_scenario(source_workspace, name="High Dose")
    project_service.attach_dataset(
        source_workspace,
        DatasetAsset(source_path=str(branch_path), display_name="branch.csv"),
    )
    project_service.set_model_spec(
        source_workspace,
        ModelSpec(
            problem_title="High Dose",
            dataset_path=str(branch_path),
            pk_code="CL = THETA(1)",
            error_code="Y = F",
            theta_rows=[{"init": 2.0, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.2]],
            sigma_values=[[0.05]],
        ),
    )
    project_service.set_scenario_notes(source_workspace, notes="Imported branch")
    branch_run = RunRecord(workflow="fit")
    branch_run.mark_succeeded("Imported fit")
    branch_report = ArtifactRecord(
        kind="report",
        label="Imported report",
        path=str(report_path),
        source_run_id=branch_run.run_id,
    )
    branch_run.artifact_ids.append(branch_report.artifact_id)
    project_service.add_run(source_workspace, branch_run)
    project_service.add_artifact(source_workspace, branch_report)

    snapshot_service = ProjectSnapshotService()
    project_snapshot = tmp_path / "dose-escalation.opkpd"
    snapshot_service.save_snapshot(
        snapshot_service.export_workspace_for_project(
            source_workspace,
            project_id=source_project.project_id,
        ),
        project_snapshot,
    )
    loaded = snapshot_service.load_snapshot(
        project_snapshot, extract_dir=tmp_path / "project-extracted"
    )

    destination_workspace = project_service.new_workspace(
        name="Destination", root_path=str(tmp_path)
    )
    imported_project = project_service.import_project(destination_workspace, loaded.project)

    assert len(destination_workspace.projects) == 2
    assert destination_workspace.active_project.project_id == imported_project.project_id
    assert imported_project.name == "Dose Escalation"
    assert imported_project.metadata["notes"] == "Imported project"
    assert imported_project.active_scenario.name == "High Dose"

    imported_baseline = next(
        scenario for scenario in imported_project.scenarios if scenario.name == "Baseline"
    )
    imported_branch = next(
        scenario for scenario in imported_project.scenarios if scenario.name == "High Dose"
    )
    assert imported_baseline.scenario_id != source_project.scenarios[0].scenario_id
    assert imported_branch.scenario_id != branch.scenario_id
    assert imported_branch.parent_scenario_id == imported_baseline.scenario_id
    assert imported_branch.metadata["notes"] == "Imported branch"
    assert imported_branch.runs[0].summary_text == "Imported fit"
    assert Path(imported_branch.active_dataset.source_path or "").read_text(
        encoding="utf-8"
    ) == branch_path.read_text(encoding="utf-8")
    assert Path(imported_branch.artifacts[0].path or "").read_text(
        encoding="utf-8"
    ) == report_path.read_text(encoding="utf-8")


def test_project_service_imports_active_scenario_snapshot_with_outputs_and_clears_missing_parent(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.csv"
    branch_path = tmp_path / "branch.csv"
    report_path = tmp_path / "branch-report.html"
    baseline_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")
    branch_path.write_text("ID,TIME,DV\n1,0,25\n", encoding="utf-8")
    report_path.write_text("<html><body>imported scenario</body></html>", encoding="utf-8")

    project_service = ProjectService()
    source_workspace = project_service.new_workspace(
        name="Source Workspace", root_path=str(tmp_path)
    )
    project_service.attach_dataset(
        source_workspace,
        DatasetAsset(source_path=str(baseline_path), display_name="baseline.csv"),
    )
    branch = project_service.create_scenario(source_workspace, name="Branch A")
    project_service.attach_dataset(
        source_workspace,
        DatasetAsset(source_path=str(branch_path), display_name="branch.csv"),
    )
    project_service.set_model_spec(
        source_workspace,
        ModelSpec(
            problem_title="Branch A",
            dataset_path=str(branch_path),
            pk_code="CL = THETA(1)",
            error_code="Y = F",
            theta_rows=[{"init": 2.5, "lower": 0.0, "upper": 10.0}],
            omega_values=[[0.2]],
            sigma_values=[[0.05]],
        ),
    )
    project_service.set_scenario_notes(source_workspace, notes="Imported scenario")
    branch_run = RunRecord(workflow="fit")
    branch_run.mark_succeeded("Imported branch fit")
    branch_report = ArtifactRecord(
        kind="report",
        label="Imported branch report",
        path=str(report_path),
        source_run_id=branch_run.run_id,
    )
    branch_run.artifact_ids.append(branch_report.artifact_id)
    project_service.add_run(source_workspace, branch_run)
    project_service.add_artifact(source_workspace, branch_report)

    snapshot_service = ProjectSnapshotService()
    scenario_snapshot = tmp_path / "branch-a.opkpd"
    snapshot_service.save_snapshot(
        snapshot_service.export_workspace_for_scenario(
            source_workspace, scenario_id=branch.scenario_id
        ),
        scenario_snapshot,
    )
    loaded = snapshot_service.load_snapshot(
        scenario_snapshot, extract_dir=tmp_path / "scenario-extracted"
    )

    destination_workspace = project_service.new_workspace(
        name="Destination", root_path=str(tmp_path)
    )
    imported_scenario = project_service.import_scenario(destination_workspace, loaded.project)

    assert len(destination_workspace.active_project.scenarios) == 2
    assert destination_workspace.active_scenario.scenario_id == imported_scenario.scenario_id
    assert imported_scenario.name == "Branch A"
    assert imported_scenario.scenario_id != branch.scenario_id
    assert imported_scenario.parent_scenario_id is None
    assert imported_scenario.metadata["notes"] == "Imported scenario"
    assert imported_scenario.runs[0].summary_text == "Imported branch fit"
    assert Path(imported_scenario.active_dataset.source_path or "").read_text(
        encoding="utf-8"
    ) == branch_path.read_text(encoding="utf-8")
    assert Path(imported_scenario.artifacts[0].path or "").read_text(
        encoding="utf-8"
    ) == report_path.read_text(encoding="utf-8")


def test_snapshot_round_trip_preserves_fit_state_payloads(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    workspace = Workspace(name="Fit State Demo", root_path=str(tmp_path))
    workspace.active_dataset = DatasetAsset(
        source_path=str(dataset_path),
        display_name="data.csv",
    )
    run = RunRecord(workflow="fit")
    run.mark_succeeded("Fit done")
    workspace.add_run(run)

    proj_id = workspace.active_project.project_id
    scen_id = workspace.active_scenario.scenario_id
    fit_state = json.dumps(
        {
            "format_version": 1,
            "workspace_id": workspace.workspace_id,
            "project_id": proj_id,
            "scenario_id": scen_id,
            "fit_run_id": run.run_id,
            "problem_title": "Demo",
            "estimation_method": "FOCE",
            "dataset_path": str(dataset_path),
            "estimation_result": {
                "theta_final": [1.0, 2.0],
                "omega_final": [[0.1]],
                "sigma_final": [[0.05]],
                "ofv": -100.0,
                "converged": True,
                "condition_number": None,
                "eta_shrinkage": [],
                "eps_shrinkage": [],
                "post_hoc_etas": {"1": [0.01, -0.02]},
                "ofv_history": [-200.0, -150.0, -100.0],
                "warnings": [],
                "n_function_evals": 50,
                "elapsed_time": 3.14,
                "method": "FOCE",
                "message": "",
                "n_observations": 10,
                "n_subjects": 2,
            },
        }
    ).encode()

    service = ProjectSnapshotService()
    snapshot_path = tmp_path / "fit-state.opkpd"
    manifest = service.save_snapshot(
        workspace,
        snapshot_path,
        fit_state_payloads={(proj_id, scen_id): fit_state},
    )

    fit_state_resources = [
        resource for resource in manifest.resources if resource.role == "fit_state"
    ]
    assert len(fit_state_resources) == 1
    assert fit_state_resources[0].project_id == proj_id
    assert fit_state_resources[0].scenario_id == scen_id

    with zipfile.ZipFile(snapshot_path, "r") as archive:
        archive_names = set(archive.namelist())
        assert any("fit_state.json" in name for name in archive_names)

    loaded = service.load_snapshot(snapshot_path, extract_dir=tmp_path / "extracted")
    assert (proj_id, scen_id) in loaded.fit_state_payloads
    restored = json.loads(loaded.fit_state_payloads[(proj_id, scen_id)])
    assert restored["estimation_result"]["theta_final"] == [1.0, 2.0]
    assert restored["estimation_result"]["converged"] is True
    assert restored["fit_run_id"] == run.run_id


def test_project_snapshot_marks_missing_external_resources(tmp_path: Path) -> None:
    workspace = Workspace(name="Missing Resources")
    workspace.active_dataset = DatasetAsset(
        source_path=str(tmp_path / "missing.csv"), display_name="missing.csv"
    )
    workspace.add_artifact(
        ArtifactRecord(kind="report", label="Missing report", path=str(tmp_path / "missing.html"))
    )

    service = ProjectSnapshotService()
    manifest = service.save_snapshot(workspace, tmp_path / "missing.opkpd")

    assert len(manifest.resources) == 2
    assert all(resource.missing for resource in manifest.resources)
