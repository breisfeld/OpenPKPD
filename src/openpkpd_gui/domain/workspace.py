"""Workspace, project, and scenario state models for the desktop GUI."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec
from openpkpd_gui.domain.run_record import RunRecord


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Scenario:
    """One editable analysis branch within a project."""

    name: str = "Baseline"
    scenario_id: str = field(default_factory=lambda: uuid4().hex)
    parent_scenario_id: str | None = None
    created_at: str = field(default_factory=_timestamp)
    updated_at: str = field(default_factory=_timestamp)
    dataset_updated_at: str | None = None
    model_updated_at: str | None = None
    active_dataset: DatasetAsset | None = None
    active_model_spec: ModelSpec | None = None
    runs: list[RunRecord] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = _timestamp()

    def add_run(self, run: RunRecord) -> None:
        self.runs.append(run)
        self.touch()

    def add_artifact(self, artifact: ArtifactRecord) -> None:
        self.artifacts.append(artifact)
        self.touch()

    def snapshot_clone(self, *, name: str, copy_outputs: bool = False) -> Scenario:
        clone = Scenario(
            name=name,
            parent_scenario_id=self.scenario_id,
            dataset_updated_at=self.dataset_updated_at,
            model_updated_at=self.model_updated_at,
            active_dataset=(
                DatasetAsset.from_dict(self.active_dataset.to_dict())
                if self.active_dataset is not None
                else None
            ),
            active_model_spec=(
                ModelSpec.from_dict(self.active_model_spec.to_dict())
                if self.active_model_spec is not None
                else None
            ),
            metadata=dict(self.metadata),
        )
        if copy_outputs:
            clone.runs = [RunRecord.from_dict(run.to_dict()) for run in self.runs]
            clone.artifacts = [
                ArtifactRecord.from_dict(artifact.to_dict()) for artifact in self.artifacts
            ]
        return clone

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scenario_id": self.scenario_id,
            "parent_scenario_id": self.parent_scenario_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "dataset_updated_at": self.dataset_updated_at,
            "model_updated_at": self.model_updated_at,
            "active_dataset": (
                self.active_dataset.to_dict() if self.active_dataset is not None else None
            ),
            "active_model_spec": (
                self.active_model_spec.to_dict() if self.active_model_spec is not None else None
            ),
            "runs": [run.to_dict() for run in self.runs],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Scenario:
        if "scenario_id" not in payload:
            raise ValueError(
                "Workspace data is corrupted: 'scenario_id' is missing from scenario payload"
            )
        scenario = cls(
            name=str(payload.get("name", "Baseline")),
            scenario_id=str(payload["scenario_id"]),
            parent_scenario_id=(
                str(payload["parent_scenario_id"]) if payload.get("parent_scenario_id") else None
            ),
            created_at=str(payload.get("created_at", _timestamp())),
            updated_at=str(payload.get("updated_at", _timestamp())),
            dataset_updated_at=(
                str(payload["dataset_updated_at"]) if payload.get("dataset_updated_at") else None
            ),
            model_updated_at=(
                str(payload["model_updated_at"]) if payload.get("model_updated_at") else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )
        if payload.get("active_dataset"):
            scenario.active_dataset = DatasetAsset.from_dict(dict(payload["active_dataset"]))
        if payload.get("active_model_spec"):
            scenario.active_model_spec = ModelSpec.from_dict(dict(payload["active_model_spec"]))
        scenario.runs = [RunRecord.from_dict(dict(item)) for item in payload.get("runs", [])]
        scenario.artifacts = [
            ArtifactRecord.from_dict(dict(item)) for item in payload.get("artifacts", [])
        ]
        return scenario


def _default_scenario() -> Scenario:
    return Scenario(name="Baseline")


@dataclass(slots=True)
class Project:
    """A named project containing one or more scenarios."""

    name: str = "Project 1"
    project_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=_timestamp)
    updated_at: str = field(default_factory=_timestamp)
    scenarios: list[Scenario] = field(default_factory=list)
    active_scenario_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenarios:
            self.scenarios = [_default_scenario()]
        if self.active_scenario_id is None or self.find_scenario(self.active_scenario_id) is None:
            self.active_scenario_id = self.scenarios[0].scenario_id

    def touch(self) -> None:
        self.updated_at = _timestamp()

    def find_scenario(self, scenario_id: str | None) -> Scenario | None:
        if scenario_id is None:
            return None
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        return None

    @property
    def active_scenario(self) -> Scenario:
        """Return the currently active scenario.

        Returns a live lookup by ID each time; callers must not cache the
        returned object across mutations (e.g. scenario removal), as the
        reference may become stale.
        """
        scenario = self.find_scenario(self.active_scenario_id)
        if scenario is None:
            scenario = self.scenarios[0]
            self.active_scenario_id = scenario.scenario_id
        return scenario

    def set_active_scenario(self, scenario_id: str) -> Scenario:
        scenario = self.find_scenario(scenario_id)
        if scenario is None:
            raise KeyError(f"Unknown scenario_id: {scenario_id}")
        self.active_scenario_id = scenario_id
        return scenario

    def add_scenario(self, scenario: Scenario, *, make_active: bool = True) -> Scenario:
        self.scenarios.append(scenario)
        if make_active:
            self.active_scenario_id = scenario.scenario_id
        self.touch()
        return scenario

    def to_dict(self, *, include_selection: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "metadata": dict(self.metadata),
        }
        if include_selection:
            payload["active_scenario_id"] = self.active_scenario_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Project:
        if "project_id" not in payload:
            raise ValueError(
                "Workspace data is corrupted: 'project_id' is missing from project payload"
            )
        return cls(
            name=str(payload.get("name", "Project 1")),
            project_id=str(payload["project_id"]),
            created_at=str(payload.get("created_at", _timestamp())),
            updated_at=str(payload.get("updated_at", _timestamp())),
            scenarios=[Scenario.from_dict(dict(item)) for item in payload.get("scenarios", [])],
            active_scenario_id=(
                str(payload["active_scenario_id"]) if payload.get("active_scenario_id") else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )


def _default_project() -> Project:
    return Project(name="Project 1")


@dataclass(slots=True)
class Workspace:
    """Top-level GUI workspace containing project and scenario selection."""

    name: str = "Untitled Workspace"
    root_path: str | None = None
    workspace_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=_timestamp)
    updated_at: str = field(default_factory=_timestamp)
    recent_files: list[str] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)
    active_project_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.projects:
            self.projects = [_default_project()]
        if self.active_project_id is None or self.find_project(self.active_project_id) is None:
            self.active_project_id = self.projects[0].project_id

    def touch(self) -> None:
        self.updated_at = _timestamp()

    def find_project(self, project_id: str | None) -> Project | None:
        if project_id is None:
            return None
        for project in self.projects:
            if project.project_id == project_id:
                return project
        return None

    def find_scenario(
        self,
        scenario_id: str,
        *,
        project_id: str | None = None,
    ) -> tuple[Project, Scenario] | None:
        projects = (
            [project for project in self.projects if project.project_id == project_id]
            if project_id is not None
            else self.projects
        )
        for project in projects:
            scenario = project.find_scenario(scenario_id)
            if scenario is not None:
                return project, scenario
        return None

    def iter_scenarios(self) -> Iterator[tuple[Project, Scenario]]:
        for project in self.projects:
            for scenario in project.scenarios:
                yield project, scenario

    @property
    def active_project(self) -> Project:
        project = self.find_project(self.active_project_id)
        if project is None:
            project = self.projects[0]
        return project

    @property
    def active_scenario(self) -> Scenario:
        return self.active_project.active_scenario

    def set_active_project(self, project_id: str) -> Project:
        project = self.find_project(project_id)
        if project is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        self.active_project_id = project_id
        return project

    def set_active_scenario(self, scenario_id: str, *, project_id: str | None = None) -> Scenario:
        match = self.find_scenario(scenario_id, project_id=project_id)
        if match is None:
            raise KeyError(f"Unknown scenario_id: {scenario_id}")
        project, scenario = match
        self.active_project_id = project.project_id
        project.set_active_scenario(scenario.scenario_id)
        return scenario

    def add_project(self, project: Project, *, make_active: bool = True) -> Project:
        with self._lock:
            self.projects.append(project)
            if make_active:
                self.active_project_id = project.project_id
            self.touch()
        return project

    @property
    def active_dataset(self) -> DatasetAsset | None:
        return self.active_scenario.active_dataset

    @active_dataset.setter
    def active_dataset(self, dataset: DatasetAsset | None) -> None:
        self.active_scenario.active_dataset = dataset
        self.active_scenario.dataset_updated_at = _timestamp()
        self.active_scenario.touch()
        self.touch()

    @property
    def active_model_spec(self) -> ModelSpec | None:
        return self.active_scenario.active_model_spec

    @active_model_spec.setter
    def active_model_spec(self, model_spec: ModelSpec | None) -> None:
        self.active_scenario.active_model_spec = model_spec
        self.active_scenario.model_updated_at = _timestamp()
        self.active_scenario.touch()
        self.touch()

    @property
    def runs(self) -> list[RunRecord]:
        return self.active_scenario.runs

    @runs.setter
    def runs(self, runs: list[RunRecord]) -> None:
        self.active_scenario.runs = runs
        self.active_scenario.touch()
        self.touch()

    @property
    def artifacts(self) -> list[ArtifactRecord]:
        return self.active_scenario.artifacts

    @artifacts.setter
    def artifacts(self, artifacts: list[ArtifactRecord]) -> None:
        self.active_scenario.artifacts = artifacts
        self.active_scenario.touch()
        self.touch()

    def add_run(self, run: RunRecord) -> None:
        with self._lock:
            self.active_scenario.add_run(run)
            self.touch()

    def add_artifact(self, artifact: ArtifactRecord) -> None:
        with self._lock:
            self.active_scenario.add_artifact(artifact)
            self.touch()

    def to_dict(self, *, include_selection: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "root_path": self.root_path,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "recent_files": list(self.recent_files),
            "projects": [
                project.to_dict(include_selection=include_selection) for project in self.projects
            ],
            "metadata": dict(self.metadata),
        }
        if include_selection:
            payload["active_project_id"] = self.active_project_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Workspace:
        if "workspace_id" not in payload:
            raise ValueError(
                "Workspace data is corrupted: 'workspace_id' is missing from workspace payload"
            )
        return cls(
            name=str(payload.get("name", "Untitled Workspace")),
            root_path=str(payload["root_path"]) if payload.get("root_path") else None,
            workspace_id=str(payload["workspace_id"]),
            created_at=str(payload.get("created_at", _timestamp())),
            updated_at=str(payload.get("updated_at", _timestamp())),
            recent_files=[str(value) for value in payload.get("recent_files", [])],
            projects=[Project.from_dict(dict(item)) for item in payload.get("projects", [])],
            active_project_id=(
                str(payload["active_project_id"]) if payload.get("active_project_id") else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )
