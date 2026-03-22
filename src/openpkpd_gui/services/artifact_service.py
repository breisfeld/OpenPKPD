"""Helpers for registering and querying workspace artifacts."""

from __future__ import annotations

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.workspace import Workspace


class ArtifactService:
    """Manage artifact records on the selected workspace scenario."""

    def register(self, workspace: Workspace, artifact: ArtifactRecord) -> ArtifactRecord:
        workspace.add_artifact(artifact)
        return artifact

    def list_by_kind(self, workspace: Workspace, kind: str) -> list[ArtifactRecord]:
        return [
            artifact for artifact in workspace.active_scenario.artifacts if artifact.kind == kind
        ]

    def list_for_run(self, workspace: Workspace, run_id: str) -> list[ArtifactRecord]:
        return [
            artifact
            for artifact in workspace.active_scenario.artifacts
            if artifact.source_run_id == run_id
        ]
