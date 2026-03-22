"""Workspace lifecycle helpers for the desktop GUI."""

from __future__ import annotations

from pathlib import Path

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec
from openpkpd_gui.domain.run_record import RunRecord
from openpkpd_gui.domain.workspace import Project, Scenario, Workspace


def _normalized_metadata_text(value: str) -> str:
    return value.strip()


def _set_metadata_text(metadata: dict[str, object], *, key: str, value: str) -> bool:
    normalized = _normalized_metadata_text(value)
    current = metadata.get(key)
    current_text = current.strip() if isinstance(current, str) else ""
    if normalized == current_text:
        return False
    if normalized:
        metadata[key] = normalized
    else:
        metadata.pop(key, None)
    return True


def _set_metadata_notes(metadata: dict[str, object], notes: str) -> bool:
    return _set_metadata_text(metadata, key="notes", value=notes)


def _clone_scenario(
    scenario: Scenario,
    *,
    name: str | None = None,
    parent_scenario_id: str | None = None,
    copy_outputs: bool = False,
) -> Scenario:
    clone = Scenario(
        name=name or scenario.name,
        parent_scenario_id=parent_scenario_id,
        active_dataset=(
            DatasetAsset.from_dict(scenario.active_dataset.to_dict())
            if scenario.active_dataset is not None
            else None
        ),
        active_model_spec=(
            ModelSpec.from_dict(scenario.active_model_spec.to_dict())
            if scenario.active_model_spec is not None
            else None
        ),
        metadata=dict(scenario.metadata),
    )
    if copy_outputs:
        clone.runs = [RunRecord.from_dict(run.to_dict()) for run in scenario.runs]
        clone.artifacts = [
            ArtifactRecord.from_dict(artifact.to_dict()) for artifact in scenario.artifacts
        ]
    return clone


def _clone_project(
    project: Project,
    *,
    name: str,
    copy_outputs: bool = False,
) -> Project:
    cloned_scenarios: list[Scenario] = []
    scenario_id_map: dict[str, str] = {}
    for source_scenario in project.scenarios:
        clone = _clone_scenario(
            source_scenario, name=source_scenario.name, copy_outputs=copy_outputs
        )
        scenario_id_map[source_scenario.scenario_id] = clone.scenario_id
        cloned_scenarios.append(clone)
    for source_scenario, clone in zip(project.scenarios, cloned_scenarios, strict=False):
        clone.parent_scenario_id = (
            scenario_id_map.get(source_scenario.parent_scenario_id)
            if source_scenario.parent_scenario_id is not None
            else None
        )
    return Project(
        name=name,
        scenarios=cloned_scenarios,
        active_scenario_id=scenario_id_map.get(project.active_scenario_id),
        metadata=dict(project.metadata),
    )


def _serialized_input(value: DatasetAsset | ModelSpec | None) -> dict[str, object] | None:
    return value.to_dict() if value is not None else None


def _invalidate_active_scenario_outputs(workspace: Workspace) -> None:
    scenario = workspace.active_scenario
    if not scenario.runs and not scenario.artifacts:
        return
    scenario.runs = []
    scenario.artifacts = []
    scenario.touch()
    workspace.active_project.touch()
    workspace.touch()


class ProjectService:
    """Create and mutate GUI workspaces in a single place."""

    MAX_RECENT_FILES = 8

    def new_workspace(
        self,
        name: str = "Untitled Workspace",
        root_path: str | None = None,
    ) -> Workspace:
        return Workspace(name=name, root_path=root_path)

    def create_project(self, workspace: Workspace, name: str | None = None) -> Project:
        project = Project(name=name or f"Project {len(workspace.projects) + 1}")
        workspace.add_project(project)
        return project

    def rename_project(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        name: str,
    ) -> Project:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        normalized_name = name.strip() or project.name
        if normalized_name == project.name:
            return project
        project.name = normalized_name
        project.touch()
        workspace.touch()
        return project

    def update_project_details(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        name: str,
        description: str,
        references: str,
        notes: str,
    ) -> Project:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")

        changed = False
        normalized_name = name.strip() or project.name
        if normalized_name != project.name:
            project.name = normalized_name
            changed = True
        if _set_metadata_text(project.metadata, key="description", value=description):
            changed = True
        if _set_metadata_text(project.metadata, key="references", value=references):
            changed = True
        if _set_metadata_notes(project.metadata, notes):
            changed = True
        if not changed:
            return project

        project.touch()
        workspace.touch()
        return project

    def set_project_notes(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        notes: str,
    ) -> Project:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        if not _set_metadata_notes(project.metadata, notes):
            return project
        project.touch()
        workspace.touch()
        return project

    def delete_project(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
    ) -> Project:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        if len(workspace.projects) <= 1:
            raise ValueError("Workspace must contain at least one project")
        index = next(
            candidate_index
            for candidate_index, candidate in enumerate(workspace.projects)
            if candidate.project_id == project.project_id
        )
        workspace.projects.pop(index)
        if workspace.active_project_id == project.project_id:
            fallback_index = min(index, len(workspace.projects) - 1)
            workspace.active_project_id = workspace.projects[fallback_index].project_id
        workspace.touch()
        return project

    def duplicate_project(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        name: str | None = None,
        copy_outputs: bool = False,
    ) -> Project:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        duplicate = _clone_project(
            project,
            name=(name.strip() if name is not None else "") or f"{project.name} Copy",
            copy_outputs=copy_outputs,
        )
        workspace.add_project(duplicate)
        return duplicate

    def import_project(
        self,
        workspace: Workspace,
        imported_workspace: Workspace,
        *,
        project_id: str | None = None,
        name: str | None = None,
    ) -> Project:
        project = (
            imported_workspace.find_project(project_id)
            if project_id is not None
            else imported_workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        imported_project = _clone_project(
            project,
            name=(name.strip() if name is not None else "") or project.name,
            copy_outputs=True,
        )
        workspace.add_project(imported_project)
        return imported_project

    def create_scenario(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        name: str | None = None,
        copy_outputs: bool = False,
    ) -> Scenario:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        parent = project.active_scenario
        scenario = parent.snapshot_clone(
            name=name or f"Scenario {len(project.scenarios) + 1}",
            copy_outputs=copy_outputs,
        )
        project.add_scenario(scenario)
        workspace.active_project_id = project.project_id
        workspace.touch()
        return scenario

    def rename_scenario(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        scenario_id: str | None = None,
        name: str,
    ) -> Scenario:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        scenario = (
            project.find_scenario(scenario_id)
            if scenario_id is not None
            else project.active_scenario
        )
        if scenario is None:
            raise KeyError(f"Unknown scenario_id: {scenario_id}")
        normalized_name = name.strip() or scenario.name
        if normalized_name == scenario.name:
            return scenario
        scenario.name = normalized_name
        scenario.touch()
        project.touch()
        workspace.touch()
        return scenario

    def update_scenario_details(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        scenario_id: str | None = None,
        name: str,
        description: str,
        references: str,
        notes: str,
    ) -> Scenario:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        scenario = (
            project.find_scenario(scenario_id)
            if scenario_id is not None
            else project.active_scenario
        )
        if scenario is None:
            raise KeyError(f"Unknown scenario_id: {scenario_id}")

        changed = False
        normalized_name = name.strip() or scenario.name
        if normalized_name != scenario.name:
            scenario.name = normalized_name
            changed = True
        if _set_metadata_text(scenario.metadata, key="description", value=description):
            changed = True
        if _set_metadata_text(scenario.metadata, key="references", value=references):
            changed = True
        if _set_metadata_notes(scenario.metadata, notes):
            changed = True
        if not changed:
            return scenario

        scenario.touch()
        project.touch()
        workspace.touch()
        return scenario

    def set_scenario_notes(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        scenario_id: str | None = None,
        notes: str,
    ) -> Scenario:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        scenario = (
            project.find_scenario(scenario_id)
            if scenario_id is not None
            else project.active_scenario
        )
        if scenario is None:
            raise KeyError(f"Unknown scenario_id: {scenario_id}")
        if not _set_metadata_notes(scenario.metadata, notes):
            return scenario
        scenario.touch()
        project.touch()
        workspace.touch()
        return scenario

    def duplicate_scenario(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        scenario_id: str | None = None,
        name: str | None = None,
        copy_outputs: bool = False,
    ) -> Scenario:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        scenario = (
            project.find_scenario(scenario_id)
            if scenario_id is not None
            else project.active_scenario
        )
        if scenario is None:
            raise KeyError(f"Unknown scenario_id: {scenario_id}")
        duplicate = _clone_scenario(
            scenario,
            name=(name.strip() if name is not None else "") or f"{scenario.name} Copy",
            parent_scenario_id=scenario.scenario_id,
            copy_outputs=copy_outputs,
        )
        project.add_scenario(duplicate)
        workspace.active_project_id = project.project_id
        workspace.touch()
        return duplicate

    def import_scenario(
        self,
        workspace: Workspace,
        imported_workspace: Workspace,
        *,
        destination_project_id: str | None = None,
        source_project_id: str | None = None,
        source_scenario_id: str | None = None,
        name: str | None = None,
    ) -> Scenario:
        destination_project = (
            workspace.find_project(destination_project_id)
            if destination_project_id is not None
            else workspace.active_project
        )
        if destination_project is None:
            raise KeyError(f"Unknown destination_project_id: {destination_project_id}")
        source_project_id = source_project_id or imported_workspace.active_project_id
        source_scenario_id = source_scenario_id or imported_workspace.active_scenario.scenario_id
        match = imported_workspace.find_scenario(source_scenario_id, project_id=source_project_id)
        if match is None:
            raise KeyError(f"Unknown source_scenario_id: {source_scenario_id}")
        _source_project, source_scenario = match
        imported_scenario = _clone_scenario(
            source_scenario,
            name=(name.strip() if name is not None else "") or source_scenario.name,
            parent_scenario_id=None,
            copy_outputs=True,
        )
        destination_project.add_scenario(imported_scenario)
        workspace.active_project_id = destination_project.project_id
        workspace.touch()
        return imported_scenario

    def delete_scenario(
        self,
        workspace: Workspace,
        *,
        project_id: str | None = None,
        scenario_id: str | None = None,
    ) -> Scenario:
        project = (
            workspace.find_project(project_id)
            if project_id is not None
            else workspace.active_project
        )
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        scenario = (
            project.find_scenario(scenario_id)
            if scenario_id is not None
            else project.active_scenario
        )
        if scenario is None:
            raise KeyError(f"Unknown scenario_id: {scenario_id}")
        if len(project.scenarios) <= 1:
            raise ValueError("Project must contain at least one scenario")
        index = next(
            candidate_index
            for candidate_index, candidate in enumerate(project.scenarios)
            if candidate.scenario_id == scenario.scenario_id
        )
        project.scenarios.pop(index)
        if project.active_scenario_id == scenario.scenario_id:
            fallback_index = min(index, len(project.scenarios) - 1)
            project.active_scenario_id = project.scenarios[fallback_index].scenario_id
        workspace.active_project_id = project.project_id
        project.touch()
        workspace.touch()
        return scenario

    def select_project(self, workspace: Workspace, project_id: str) -> Project:
        return workspace.set_active_project(project_id)

    def select_scenario(
        self,
        workspace: Workspace,
        scenario_id: str,
        *,
        project_id: str | None = None,
    ) -> Scenario:
        return workspace.set_active_scenario(scenario_id, project_id=project_id)

    def attach_dataset(self, workspace: Workspace, dataset: DatasetAsset) -> None:
        previous_dataset = _serialized_input(workspace.active_dataset)
        previous_dataset_updated_at = workspace.active_scenario.dataset_updated_at
        workspace.active_dataset = dataset
        if previous_dataset == _serialized_input(dataset):
            workspace.active_scenario.dataset_updated_at = previous_dataset_updated_at
        if previous_dataset != _serialized_input(dataset):
            _invalidate_active_scenario_outputs(workspace)
        if workspace.root_path is None and dataset.source_path:
            workspace.root_path = str(Path(dataset.source_path).resolve().parent)
        workspace.touch()

    def set_model_spec(self, workspace: Workspace, model_spec: ModelSpec) -> None:
        previous_model_spec = _serialized_input(workspace.active_model_spec)
        previous_model_updated_at = workspace.active_scenario.model_updated_at
        workspace.active_model_spec = model_spec
        if previous_model_spec == _serialized_input(model_spec):
            workspace.active_scenario.model_updated_at = previous_model_updated_at
        if previous_model_spec != _serialized_input(model_spec):
            _invalidate_active_scenario_outputs(workspace)
        if workspace.root_path is None and model_spec.dataset_path:
            workspace.root_path = str(Path(model_spec.dataset_path).resolve().parent)
        workspace.touch()

    def add_run(self, workspace: Workspace, run: RunRecord) -> None:
        workspace.add_run(run)

    def add_artifact(self, workspace: Workspace, artifact: ArtifactRecord) -> None:
        workspace.add_artifact(artifact)

    def remember_recent_file(
        self,
        project: Workspace,
        path: str | Path,
        *,
        limit: int | None = None,
    ) -> None:
        resolved_path = str(Path(path).resolve())
        recent_files = [item for item in project.recent_files if item != resolved_path]
        recent_files.insert(0, resolved_path)
        project.recent_files = recent_files[: (limit or self.MAX_RECENT_FILES)]
        project.touch()

    def clear_recent_files(self, project: Workspace) -> None:
        if not project.recent_files:
            return
        project.recent_files = []
        project.touch()
