"""Base types for background job execution."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


class JobStatus(StrEnum):
    """Status for background job execution."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class JobEvent:
    """One emitted job event."""

    kind: str
    message: str
    progress: float | None = None


class JobContext:
    """Execution context passed to job callables."""

    def __init__(
        self,
        emit_callback: Callable[[JobEvent], None],
        cancel_requested_callback: Callable[[], bool] | None = None,
    ) -> None:
        self._emit_callback = emit_callback
        self._cancel_requested_callback = cancel_requested_callback or (lambda: False)

    def emit(self, message: str, progress: float | None = None, kind: str = "message") -> None:
        self._emit_callback(JobEvent(kind=kind, message=message, progress=progress))

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested_callback())

    def check_cancelled(self) -> None:
        if self.is_cancel_requested():
            raise JobCancelledError()


class JobCancelledError(Exception):
    """Raised by cooperative jobs when cancellation has been requested."""

    def __init__(self, message: str = "Cancelled by user.") -> None:
        super().__init__(message)


JobCallable = Callable[[JobContext], object]


@dataclass(slots=True)
class BackgroundJob:
    """Callable unit of work for the job runner."""

    name: str
    func: JobCallable
    job_id: str = field(default_factory=lambda: uuid4().hex)
    _cancel_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def request_cancel(self) -> None:
        self._cancel_event.set()


@dataclass(slots=True)
class JobOutcome:
    """Normalized result of background execution."""

    job_id: str
    name: str
    status: JobStatus
    value: object | None = None
    error: str | None = None
    events: list[JobEvent] = field(default_factory=list)
