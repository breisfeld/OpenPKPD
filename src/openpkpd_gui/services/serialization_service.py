"""Package-based workspace serialization for the desktop GUI."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import platform
import re
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from openpkpd_gui.domain.workspace import Project, Scenario, Workspace


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: str, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return cleaned or default


def _media_type(path: str | None, fallback: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(path or "")
    return guessed or fallback


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _openpkpd_version() -> str:
    try:
        import openpkpd

        return str(getattr(openpkpd, "__version__", "unknown"))
    except Exception:
        return "unknown"


WORKSPACE_METADATA_FILE = "workspace.json"
LEGACY_MANIFEST_FILE = "manifest.json"


def _project_archive_dir(project: Project) -> str:
    return f"projects/{_slug(project.name, 'project')}-{project.project_id}"


def _scenario_archive_dir(project: Project, scenario: Scenario) -> str:
    return (
        f"{_project_archive_dir(project)}/scenarios/"
        f"{_slug(scenario.name, 'scenario')}-{scenario.scenario_id}"
    )


def _artifact_archive_category(kind: str, metadata: Mapping[str, object]) -> str:
    artifact_role = str(metadata.get("artifact_role", "")).strip().lower()
    normalized_kind = kind.strip().lower()
    if artifact_role == "report" or normalized_kind == "report":
        return "reports"
    if artifact_role == "plot" or normalized_kind == "plot":
        return "plots"
    if normalized_kind in {"result", "results", "table", "dataset", "nca"}:
        return "results"
    return "outputs"


@dataclass(slots=True)
class SnapshotPayload:
    """Binary payload supplied by the caller for an in-memory resource."""

    file_name: str
    data: bytes
    media_type: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SnapshotResource:
    """One packaged file or unresolved external reference."""

    role: str
    label: str
    resource_id: str = field(default_factory=lambda: uuid4().hex)
    file_name: str | None = None
    archive_path: str | None = None
    original_path: str | None = None
    extracted_path: str | None = None
    artifact_id: str | None = None
    source_run_id: str | None = None
    project_id: str | None = None
    scenario_id: str | None = None
    media_type: str | None = None
    sha256: str | None = None
    size_bytes: int = 0
    missing: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "label": self.label,
            "resource_id": self.resource_id,
            "file_name": self.file_name,
            "archive_path": self.archive_path,
            "original_path": self.original_path,
            "extracted_path": self.extracted_path,
            "artifact_id": self.artifact_id,
            "source_run_id": self.source_run_id,
            "project_id": self.project_id,
            "scenario_id": self.scenario_id,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "missing": self.missing,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SnapshotResource:
        return cls(
            role=str(payload["role"]),
            label=str(payload["label"]),
            resource_id=str(payload.get("resource_id", uuid4().hex)),
            file_name=str(payload["file_name"]) if payload.get("file_name") else None,
            archive_path=str(payload["archive_path"]) if payload.get("archive_path") else None,
            original_path=str(payload["original_path"]) if payload.get("original_path") else None,
            extracted_path=str(payload["extracted_path"])
            if payload.get("extracted_path")
            else None,
            artifact_id=str(payload["artifact_id"]) if payload.get("artifact_id") else None,
            source_run_id=str(payload["source_run_id"]) if payload.get("source_run_id") else None,
            project_id=(str(payload["project_id"]) if payload.get("project_id") else None),
            scenario_id=str(payload["scenario_id"]) if payload.get("scenario_id") else None,
            media_type=str(payload["media_type"]) if payload.get("media_type") else None,
            sha256=str(payload["sha256"]) if payload.get("sha256") else None,
            size_bytes=int(payload.get("size_bytes", 0)),
            missing=bool(payload.get("missing", False)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class SnapshotManifest:
    """Top-level snapshot manifest stored as JSON inside the archive."""

    project: dict[str, object]
    resources: list[SnapshotResource] = field(default_factory=list)
    format_name: str = "openpkpd.gui.project_package"
    format_version: int = 3
    created_at: str = field(default_factory=_timestamp)
    provenance: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "created_at": self.created_at,
            "provenance": dict(self.provenance),
            "project": dict(self.project),
            "resources": [resource.to_dict() for resource in self.resources],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SnapshotManifest:
        return cls(
            project=dict(payload.get("project", {})),
            resources=[
                SnapshotResource.from_dict(dict(item)) for item in payload.get("resources", [])
            ],
            format_name=str(payload.get("format_name", "openpkpd.gui.project_package")),
            format_version=int(payload.get("format_version", 3)),
            created_at=str(payload.get("created_at", _timestamp())),
            provenance=dict(payload.get("provenance", {})),
        )


@dataclass(slots=True)
class LoadedProjectSnapshot:
    """Loaded snapshot containing the restored workspace and manifest."""

    project: Workspace
    manifest: SnapshotManifest
    extracted_root: str
    fit_state_payloads: dict[tuple[str, str], bytes] = field(default_factory=dict)


class ProjectSnapshotService:
    """Persist a GUI workspace as a self-contained zipped project package."""

    def export_workspace_for_project(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
    ) -> Workspace:
        source_project = self._require_project(workspace, project_id or workspace.active_project_id)
        export_workspace = Workspace.from_dict(workspace.to_dict())
        export_project = self._require_project(export_workspace, source_project.project_id)
        export_workspace.name = source_project.name
        export_workspace.root_path = None
        export_workspace.recent_files = []
        export_workspace.projects = [export_project]
        export_workspace.active_project_id = export_project.project_id
        return export_workspace

    def export_workspace_for_scenario(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        scenario_id: str | None = None,
    ) -> Workspace:
        source_project_id = project_id or workspace.active_project_id
        source_scenario_id = scenario_id or workspace.active_scenario.scenario_id
        source_project, source_scenario = self._require_scenario(
            workspace,
            source_scenario_id,
            project_id=source_project_id,
        )
        export_workspace = self.export_workspace_for_project(
            workspace,
            project_id=source_project.project_id,
        )
        export_project = export_workspace.active_project
        export_scenario = self._require_scenario(
            export_workspace,
            source_scenario.scenario_id,
            project_id=export_project.project_id,
        )[1]
        export_workspace.name = f"{source_project.name} - {source_scenario.name}"
        export_project.scenarios = [export_scenario]
        export_project.active_scenario_id = export_scenario.scenario_id
        return export_workspace

    def save_snapshot(
        self,
        workspace: Workspace,
        destination: str | Path,
        *,
        dataset_payload: SnapshotPayload | None = None,
        artifact_payloads: Mapping[str, SnapshotPayload] | None = None,
        fit_state_payloads: Mapping[tuple[str, str], bytes] | None = None,
    ) -> SnapshotManifest:
        snapshot_path = Path(destination)
        artifact_payloads = artifact_payloads or {}
        fit_state_payloads = fit_state_payloads or {}
        manifest = SnapshotManifest(
            project=workspace.to_dict(), provenance=self._provenance(workspace)
        )

        with tempfile.TemporaryDirectory(prefix="openpkpd_snapshot_build_") as staging_dir:
            staging_root = Path(staging_dir)
            scenario_resources: dict[tuple[str, str], list[SnapshotResource]] = {}
            for project_model, scenario in workspace.iter_scenarios():
                scenario_key = (project_model.project_id, scenario.scenario_id)
                if scenario.active_dataset is not None:
                    dataset_resource = self._store_dataset(
                        staging_root,
                        project_model,
                        scenario,
                        dataset_payload=(
                            dataset_payload if scenario is workspace.active_scenario else None
                        ),
                    )
                    manifest.resources.append(dataset_resource)
                    scenario_resources.setdefault(scenario_key, []).append(dataset_resource)
                for artifact in scenario.artifacts:
                    artifact_resource = self._store_artifact(
                        staging_root,
                        project=project_model,
                        scenario=scenario,
                        artifact_id=artifact.artifact_id,
                        label=artifact.label,
                        kind=artifact.kind,
                        original_path=artifact.path,
                        source_run_id=artifact.source_run_id,
                        metadata=dict(artifact.metadata),
                        payload=artifact_payloads.get(artifact.artifact_id),
                    )
                    manifest.resources.append(artifact_resource)
                    scenario_resources.setdefault(scenario_key, []).append(artifact_resource)
                fit_state_data = fit_state_payloads.get(
                    (project_model.project_id, scenario.scenario_id)
                )
                if fit_state_data:
                    fit_state_resource = self._store_fit_state(
                        staging_root,
                        project=project_model,
                        scenario=scenario,
                        data=fit_state_data,
                    )
                    manifest.resources.append(fit_state_resource)

            self._write_workspace_metadata(staging_root, manifest)
            for project_model in workspace.projects:
                self._write_project_metadata(
                    staging_root,
                    workspace=workspace,
                    project=project_model,
                    manifest=manifest,
                )
                for scenario in project_model.scenarios:
                    self._write_scenario_metadata(
                        staging_root,
                        project=project_model,
                        scenario=scenario,
                        resources=scenario_resources.get(
                            (project_model.project_id, scenario.scenario_id), []
                        ),
                    )

            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(snapshot_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file_path in sorted(staging_root.rglob("*")):
                    archive.write(file_path, file_path.relative_to(staging_root).as_posix())
        return manifest

    def _write_workspace_metadata(self, staging_root: Path, manifest: SnapshotManifest) -> None:
        manifest_path = staging_root / WORKSPACE_METADATA_FILE
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_project_metadata(
        self,
        staging_root: Path,
        *,
        workspace: Workspace,
        project: Project,
        manifest: SnapshotManifest,
    ) -> None:
        project_root = staging_root / _project_archive_dir(project)
        project_root.mkdir(parents=True, exist_ok=True)
        scenario_index = [
            {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "path": Path(_scenario_archive_dir(project, scenario))
                .relative_to(Path(_project_archive_dir(project)))
                .as_posix(),
            }
            for scenario in project.scenarios
        ]
        project_payload = project.to_dict(include_selection=True)
        project_payload["scenarios"] = scenario_index
        (project_root / "metadata.json").write_text(
            json.dumps(
                {
                    "format_name": manifest.format_name,
                    "format_version": manifest.format_version,
                    "workspace": {
                        "name": workspace.name,
                        "workspace_id": workspace.workspace_id,
                        "active_project_id": workspace.active_project_id,
                    },
                    "project": project_payload,
                    "provenance": dict(manifest.provenance),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _write_scenario_metadata(
        self,
        staging_root: Path,
        *,
        project: Project,
        scenario: Scenario,
        resources: list[SnapshotResource],
    ) -> None:
        scenario_root = staging_root / _scenario_archive_dir(project, scenario)
        for folder_name in ("data", "models", "outputs", "reports", "results", "plots"):
            (scenario_root / folder_name).mkdir(parents=True, exist_ok=True)
        if scenario.active_model_spec is not None:
            (scenario_root / "models" / "model_spec.json").write_text(
                json.dumps(scenario.active_model_spec.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if scenario.runs:
            (scenario_root / "outputs" / "runs.json").write_text(
                json.dumps([run.to_dict() for run in scenario.runs], indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if scenario.artifacts:
            (scenario_root / "results" / "artifacts.json").write_text(
                json.dumps(
                    [artifact.to_dict() for artifact in scenario.artifacts],
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        dataset_resource = next(
            (resource for resource in resources if resource.role == "scenario_dataset"), None
        )
        (scenario_root / "metadata.json").write_text(
            json.dumps(
                {
                    "scenario": {
                        "name": scenario.name,
                        "scenario_id": scenario.scenario_id,
                        "parent_scenario_id": scenario.parent_scenario_id,
                        "created_at": scenario.created_at,
                        "updated_at": scenario.updated_at,
                        "metadata": dict(scenario.metadata),
                    },
                    "dataset": scenario.active_dataset.to_dict()
                    if scenario.active_dataset is not None
                    else None,
                    "dataset_archive_path": dataset_resource.archive_path
                    if dataset_resource
                    else None,
                    "model_spec_path": (
                        "models/model_spec.json" if scenario.active_model_spec is not None else None
                    ),
                    "runs_path": "outputs/runs.json" if scenario.runs else None,
                    "artifacts_path": "results/artifacts.json" if scenario.artifacts else None,
                    "resources": [resource.to_dict() for resource in resources],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _store_fit_state(
        self,
        staging_root: Path,
        *,
        project: Project,
        scenario: Scenario,
        data: bytes,
    ) -> SnapshotResource:
        scenario_dir = _scenario_archive_dir(project, scenario)
        archive_rel = f"{scenario_dir}/outputs/fit_state.json"
        target = staging_root / archive_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return SnapshotResource(
            role="fit_state",
            label=f"Fit state ({scenario.name})",
            file_name="fit_state.json",
            archive_path=archive_rel,
            project_id=project.project_id,
            scenario_id=scenario.scenario_id,
            media_type="application/json",
            sha256=_sha256(data),
            size_bytes=len(data),
        )

    def load_snapshot(
        self,
        source: str | Path,
        *,
        extract_dir: str | Path | None = None,
    ) -> LoadedProjectSnapshot:
        snapshot_path = Path(source)
        extraction_root = (
            Path(extract_dir)
            if extract_dir is not None
            else Path(tempfile.mkdtemp(prefix="openpkpd_snapshot_"))
        )
        extraction_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(snapshot_path, "r") as archive:
            archive.extractall(extraction_root)

        manifest_path = extraction_root / WORKSPACE_METADATA_FILE
        if not manifest_path.exists():
            manifest_path = extraction_root / LEGACY_MANIFEST_FILE
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = SnapshotManifest.from_dict(manifest_payload)
        workspace = Workspace.from_dict(manifest.project)

        fit_state_payloads: dict[tuple[str, str], bytes] = {}
        for resource in manifest.resources:
            if resource.archive_path:
                resource_path = extraction_root / resource.archive_path
                resource.extracted_path = str(resource_path)
            scenario_match = None
            if resource.scenario_id:
                scenario_match = workspace.find_scenario(
                    resource.scenario_id, project_id=resource.project_id
                )
            if scenario_match is None:
                continue
            _project, scenario = scenario_match
            if (
                resource.role == "scenario_dataset"
                and scenario.active_dataset is not None
                and resource.extracted_path
            ):
                scenario.active_dataset.source_path = resource.extracted_path
                if scenario.active_model_spec is not None:
                    scenario.active_model_spec.dataset_path = resource.extracted_path
            if resource.role == "artifact" and resource.artifact_id and resource.extracted_path:
                for artifact in scenario.artifacts:
                    if artifact.artifact_id == resource.artifact_id:
                        artifact.path = resource.extracted_path
                        break
            if (
                resource.role == "fit_state"
                and resource.extracted_path
                and resource.project_id
                and resource.scenario_id
            ):
                fit_state_path = Path(resource.extracted_path)
                if fit_state_path.is_file():
                    fit_state_payloads[(resource.project_id, resource.scenario_id)] = (
                        fit_state_path.read_bytes()
                    )

        return LoadedProjectSnapshot(
            project=workspace,
            manifest=manifest,
            extracted_root=str(extraction_root),
            fit_state_payloads=fit_state_payloads,
        )

    def _store_dataset(
        self,
        staging_root: Path,
        project: Project,
        scenario: Scenario,
        *,
        dataset_payload: SnapshotPayload | None,
    ) -> SnapshotResource:
        dataset = scenario.active_dataset
        assert dataset is not None
        label = dataset.display_name or f"{project.name}-{scenario.name}-dataset"
        if dataset.source_path and Path(dataset.source_path).is_file():
            data = Path(dataset.source_path).read_bytes()
            return self._write_resource(
                staging_root,
                role="scenario_dataset",
                label=label,
                file_name=Path(dataset.source_path).name,
                data=data,
                original_path=dataset.source_path,
                project_id=project.project_id,
                scenario_id=scenario.scenario_id,
                media_type=_media_type(dataset.source_path, fallback="text/csv"),
                metadata={
                    "project_name": project.name,
                    "scenario_name": scenario.name,
                    "display_name": dataset.display_name,
                    "separator": dataset.separator,
                    "treat_as_whitespace": dataset.treat_as_whitespace,
                    "ignore_char": dataset.ignore_char,
                },
            )
        if dataset_payload is not None:
            return self._write_resource(
                staging_root,
                role="scenario_dataset",
                label=label,
                file_name=dataset_payload.file_name,
                data=dataset_payload.data,
                original_path=dataset.source_path,
                project_id=project.project_id,
                scenario_id=scenario.scenario_id,
                media_type=dataset_payload.media_type
                or _media_type(dataset_payload.file_name, "text/csv"),
                metadata={
                    "project_name": project.name,
                    "scenario_name": scenario.name,
                    "display_name": dataset.display_name,
                    **dataset_payload.metadata,
                },
            )
        return SnapshotResource(
            role="scenario_dataset",
            label=label,
            file_name=Path(dataset.source_path).name if dataset.source_path else None,
            original_path=dataset.source_path,
            project_id=project.project_id,
            scenario_id=scenario.scenario_id,
            media_type=_media_type(dataset.source_path, fallback="text/csv"),
            missing=True,
            metadata={
                "project_name": project.name,
                "scenario_name": scenario.name,
                "display_name": dataset.display_name,
            },
        )

    def _store_artifact(
        self,
        staging_root: Path,
        *,
        project: Project,
        scenario: Scenario,
        artifact_id: str,
        label: str,
        kind: str,
        original_path: str | None,
        source_run_id: str | None,
        metadata: dict[str, object],
        payload: SnapshotPayload | None,
    ) -> SnapshotResource:
        if original_path and Path(original_path).is_file():
            data = Path(original_path).read_bytes()
            return self._write_resource(
                staging_root,
                role="artifact",
                label=label,
                file_name=Path(original_path).name,
                data=data,
                original_path=original_path,
                artifact_id=artifact_id,
                source_run_id=source_run_id,
                project_id=project.project_id,
                scenario_id=scenario.scenario_id,
                media_type=_media_type(original_path),
                metadata={
                    "kind": kind,
                    "project_name": project.name,
                    "scenario_name": scenario.name,
                    **metadata,
                },
            )
        if payload is not None:
            return self._write_resource(
                staging_root,
                role="artifact",
                label=label,
                file_name=payload.file_name,
                data=payload.data,
                original_path=original_path,
                artifact_id=artifact_id,
                source_run_id=source_run_id,
                project_id=project.project_id,
                scenario_id=scenario.scenario_id,
                media_type=payload.media_type or _media_type(payload.file_name),
                metadata={
                    "kind": kind,
                    "project_name": project.name,
                    "scenario_name": scenario.name,
                    **metadata,
                    **payload.metadata,
                },
            )
        return SnapshotResource(
            role="artifact",
            label=label,
            file_name=Path(original_path).name if original_path else None,
            original_path=original_path,
            artifact_id=artifact_id,
            source_run_id=source_run_id,
            project_id=project.project_id,
            scenario_id=scenario.scenario_id,
            media_type=_media_type(original_path),
            missing=True,
            metadata={
                "kind": kind,
                "project_name": project.name,
                "scenario_name": scenario.name,
                **metadata,
            },
        )

    def _write_resource(
        self,
        staging_root: Path,
        *,
        role: str,
        label: str,
        file_name: str,
        data: bytes,
        original_path: str | None,
        media_type: str,
        metadata: dict[str, object],
        artifact_id: str | None = None,
        source_run_id: str | None = None,
        project_id: str | None = None,
        scenario_id: str | None = None,
    ) -> SnapshotResource:
        resource = SnapshotResource(
            role=role,
            label=label,
            file_name=file_name,
            original_path=original_path,
            artifact_id=artifact_id,
            source_run_id=source_run_id,
            project_id=project_id,
            scenario_id=scenario_id,
            media_type=media_type,
            sha256=_sha256(data),
            size_bytes=len(data),
            metadata=metadata,
        )
        project_dir = _slug(str(metadata.get("project_name") or project_id or "project"), "project")
        scenario_dir = _slug(
            str(metadata.get("scenario_name") or scenario_id or "scenario"), "scenario"
        )
        if role == "scenario_dataset":
            category = "data"
        else:
            category = _artifact_archive_category(str(metadata.get("kind", "artifact")), metadata)
        archive_rel = (
            f"projects/{project_dir}-{project_id or 'project'}/scenarios/"
            f"{scenario_dir}-{scenario_id or 'scenario'}/{category}/"
            f"{resource.resource_id}-{_slug(file_name, 'resource')}"
        )
        target_path = staging_root / archive_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
        resource.archive_path = archive_rel
        return resource

    def _provenance(self, workspace: Workspace) -> dict[str, object]:
        active_project = workspace.active_project
        active_scenario = workspace.active_scenario
        return {
            "workspace_id": workspace.workspace_id,
            "workspace_name": workspace.name,
            "workspace_updated_at": workspace.updated_at,
            "active_project_id": active_project.project_id,
            "active_project_name": active_project.name,
            "active_scenario_id": active_scenario.scenario_id,
            "active_scenario_name": active_scenario.name,
            "serialized_at": _timestamp(),
            "serialized_by": "openpkpd_gui.services.serialization_service.ProjectSnapshotService",
            "openpkpd_version": _openpkpd_version(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        }

    def _require_project(self, workspace: Workspace, project_id: str | None) -> Project:
        project = workspace.find_project(project_id)
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        return project

    def _require_scenario(
        self,
        workspace: Workspace,
        scenario_id: str | None,
        *,
        project_id: str | None = None,
    ) -> tuple[Project, Scenario]:
        if scenario_id is None:
            raise KeyError("Unknown scenario_id: None")
        match = workspace.find_scenario(scenario_id, project_id=project_id)
        if match is None:
            raise KeyError(f"Unknown scenario_id: {scenario_id}")
        return match
