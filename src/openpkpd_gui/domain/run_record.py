"""Run metadata for background workflow executions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


class RunStatus(StrEnum):
    """Lifecycle state for a workflow run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class RunRecord:
    """Serializable record of one workflow execution."""

    workflow: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    status: RunStatus = RunStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    cancel_requested_at: str | None = None
    summary_text: str = ""
    error_text: str = ""
    log_lines: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)

    def mark_running(self) -> None:
        self.status = RunStatus.RUNNING
        self.started_at = _timestamp()
        self.cancel_requested_at = None

    def mark_succeeded(self, summary_text: str = "") -> None:
        self.status = RunStatus.SUCCEEDED
        self.summary_text = summary_text
        self.finished_at = _timestamp()

    def mark_failed(self, error_text: str) -> None:
        self.status = RunStatus.FAILED
        self.error_text = error_text
        self.finished_at = _timestamp()

    def mark_cancel_requested(self) -> None:
        if self.cancel_requested_at is None:
            self.cancel_requested_at = _timestamp()

    def mark_cancelled(self, error_text: str = "Cancelled by user.") -> None:
        self.status = RunStatus.CANCELLED
        self.cancel_requested_at = self.cancel_requested_at or _timestamp()
        self.error_text = error_text
        self.finished_at = _timestamp()

    def add_log(self, line: str) -> None:
        self.log_lines.append(line)

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow": self.workflow,
            "run_id": self.run_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_requested_at": self.cancel_requested_at,
            "summary_text": self.summary_text,
            "error_text": self.error_text,
            "log_lines": list(self.log_lines),
            "artifact_ids": list(self.artifact_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunRecord:
        return cls(
            workflow=str(payload["workflow"]),
            run_id=str(payload.get("run_id", uuid4().hex)),
            status=RunStatus(str(payload.get("status", RunStatus.PENDING.value))),
            started_at=str(payload["started_at"]) if payload.get("started_at") else None,
            finished_at=str(payload["finished_at"]) if payload.get("finished_at") else None,
            cancel_requested_at=(
                str(payload["cancel_requested_at"]) if payload.get("cancel_requested_at") else None
            ),
            summary_text=str(payload.get("summary_text", "")),
            error_text=str(payload.get("error_text", "")),
            log_lines=[str(value) for value in payload.get("log_lines", [])],
            artifact_ids=[str(value) for value in payload.get("artifact_ids", [])],
        )
