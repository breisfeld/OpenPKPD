"""Simple thread-based job runner for GUI orchestration."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from openpkpd_gui.jobs.base import (
    BackgroundJob,
    JobCancelledError,
    JobContext,
    JobEvent,
    JobOutcome,
    JobStatus,
)


class JobRunner:
    """Run background jobs and normalize their outcomes."""

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        job: BackgroundJob,
        on_event: Callable[[JobEvent], None] | None = None,
    ) -> Future[JobOutcome]:
        return self._executor.submit(self._run_job, job, on_event)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    @staticmethod
    def _run_job(
        job: BackgroundJob,
        on_event: Callable[[JobEvent], None] | None,
    ) -> JobOutcome:
        events: list[JobEvent] = []

        def emit(event: JobEvent) -> None:
            events.append(event)
            if on_event is not None:
                on_event(event)

        if job.cancel_requested:
            emit(JobEvent(kind="status", message=f"Cancelled {job.name}", progress=1.0))
            return JobOutcome(
                job_id=job.job_id,
                name=job.name,
                status=JobStatus.CANCELLED,
                error="Cancelled by user.",
                events=events,
            )

        emit(JobEvent(kind="status", message=f"Started {job.name}", progress=0.0))
        ctx = JobContext(emit, cancel_requested_callback=lambda: job.cancel_requested)
        try:
            value = job.func(ctx)
        except JobCancelledError as exc:
            emit(JobEvent(kind="status", message=f"Cancelled {job.name}", progress=1.0))
            return JobOutcome(
                job_id=job.job_id,
                name=job.name,
                status=JobStatus.CANCELLED,
                error=str(exc),
                events=events,
            )
        except Exception as exc:
            emit(JobEvent(kind="error", message=str(exc), progress=1.0))
            return JobOutcome(
                job_id=job.job_id,
                name=job.name,
                status=JobStatus.FAILED,
                error=str(exc),
                events=events,
            )

        emit(JobEvent(kind="status", message=f"Finished {job.name}", progress=1.0))
        return JobOutcome(
            job_id=job.job_id,
            name=job.name,
            status=JobStatus.SUCCEEDED,
            value=value,
            events=events,
        )
