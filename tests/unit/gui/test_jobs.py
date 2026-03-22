"""Unit tests for the new GUI job runner."""

from __future__ import annotations

import threading
import time

from openpkpd_gui.jobs.base import BackgroundJob, JobStatus
from openpkpd_gui.jobs.runner import JobRunner


def test_job_runner_reports_success_and_progress() -> None:
    events: list[tuple[str, str]] = []

    def work(ctx) -> str:
        ctx.emit("halfway", progress=0.5)
        return "done"

    runner = JobRunner()
    try:
        outcome = runner.submit(
            BackgroundJob(name="success", func=work),
            on_event=lambda event: events.append((event.kind, event.message)),
        ).result(timeout=5)
    finally:
        runner.shutdown()

    assert outcome.status == JobStatus.SUCCEEDED
    assert outcome.value == "done"
    assert ("message", "halfway") in events
    assert events[0][1] == "Started success"
    assert events[-1][1] == "Finished success"


def test_job_runner_normalizes_failure() -> None:
    def work(ctx) -> None:
        ctx.emit("before failure")
        raise ValueError("boom")

    runner = JobRunner()
    try:
        outcome = runner.submit(BackgroundJob(name="failure", func=work)).result(timeout=5)
    finally:
        runner.shutdown()

    assert outcome.status == JobStatus.FAILED
    assert outcome.error == "boom"
    assert outcome.events[-1].kind == "error"


def test_job_runner_returns_cancelled_outcome_for_presubmitted_cancel() -> None:
    called = False

    def work(ctx) -> str:
        nonlocal called
        called = True
        return "done"

    job = BackgroundJob(name="cancel-before-start", func=work)
    job.request_cancel()

    runner = JobRunner()
    try:
        outcome = runner.submit(job).result(timeout=5)
    finally:
        runner.shutdown()

    assert called is False
    assert outcome.status == JobStatus.CANCELLED
    assert outcome.error == "Cancelled by user."
    assert [event.message for event in outcome.events] == ["Cancelled cancel-before-start"]


def test_job_runner_returns_cancelled_outcome_when_job_checks_token() -> None:
    started = threading.Event()

    def work(ctx) -> str:
        started.set()
        while not ctx.is_cancel_requested():
            time.sleep(0.01)
        ctx.check_cancelled()
        return "done"

    job = BackgroundJob(name="cancel-during-run", func=work)
    runner = JobRunner()
    try:
        future = runner.submit(job)
        assert started.wait(timeout=5)
        job.request_cancel()
        outcome = future.result(timeout=5)
    finally:
        runner.shutdown()

    assert outcome.status == JobStatus.CANCELLED
    assert outcome.error == "Cancelled by user."
    assert outcome.events[0].message == "Started cancel-during-run"
    assert outcome.events[-1].message == "Cancelled cancel-during-run"
