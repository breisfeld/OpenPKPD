"""Artifact records produced by GUI workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ArtifactRecord:
    """Metadata for a file or logical artifact produced by a run."""

    kind: str
    label: str
    path: str | None = None
    source_run_id: str | None = None
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    artifact_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=_timestamp)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "label": self.label,
            "path": self.path,
            "source_run_id": self.source_run_id,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ArtifactRecord:
        return cls(
            kind=str(payload["kind"]),
            label=str(payload["label"]),
            path=str(payload["path"]) if payload.get("path") else None,
            source_run_id=(str(payload["source_run_id"]) if payload.get("source_run_id") else None),
            metadata=dict(payload.get("metadata", {})),
            artifact_id=str(payload.get("artifact_id", uuid4().hex)),
            created_at=str(payload.get("created_at", _timestamp())),
        )
