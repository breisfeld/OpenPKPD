"""Extended job lifecycle tests — event ordering, exception details, context API."""

from __future__ import annotations

import threading
import time

import pytest

from openpkpd_gui.jobs.base import (
    BackgroundJob,
    JobCancelledError,
    JobContext,
    JobEvent,
    JobStatus,
)
from openpkpd_gui.jobs.runner import JobRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(job: BackgroundJob, on_event=None, timeout: float = 5.0):
    """Submit one job and return its outcome synchronously."""
    runner = JobRunner()
    try:
        return runner.submit(job, on_event=on_event).result(timeout=timeout)
    finally:
        runner.shutdown(wait=False)


# ===========================================================================
# Event ordering and content
# ===========================================================================


def test_events_always_start_with_started_and_end_with_finished() -> None:
    def work(ctx):
        ctx.emit("mid", progress=0.5)
        return "ok"

    outcome = _run(BackgroundJob(name="ordered", func=work))
    messages = [e.message for e in outcome.events]
    assert messages[0] == "Started ordered"
    assert messages[-1] == "Finished ordered"
    assert "mid" in messages


def test_events_accumulated_on_outcome() -> None:
    def work(ctx):
        for i in range(5):
            ctx.emit(f"step-{i}", progress=i / 4)
        return "done"

    outcome = _run(BackgroundJob(name="multi", func=work))
    assert outcome.status == JobStatus.SUCCEEDED
    step_messages = [e.message for e in outcome.events if e.message.startswith("step-")]
    assert len(step_messages) == 5


def test_on_event_callback_receives_all_events() -> None:
    received: list[JobEvent] = []

    def work(ctx):
        ctx.emit("first")
        ctx.emit("second")
        return "x"

    outcome = _run(BackgroundJob(name="cb", func=work), on_event=received.append)

    assert outcome.status == JobStatus.SUCCEEDED
    messages = [e.message for e in received]
    assert "first" in messages
    assert "second" in messages
    assert "Started cb" in messages
    assert "Finished cb" in messages


def test_event_progress_is_forwarded() -> None:
    def work(ctx):
        ctx.emit("halfway", progress=0.5)
        return "done"

    outcome = _run(BackgroundJob(name="progress", func=work))
    halfway = next((e for e in outcome.events if e.message == "halfway"), None)
    assert halfway is not None
    assert halfway.progress == pytest.approx(0.5)


def test_failure_event_has_error_kind() -> None:
    def work(ctx):
        raise RuntimeError("exploded")

    outcome = _run(BackgroundJob(name="explode", func=work))
    assert outcome.status == JobStatus.FAILED
    error_events = [e for e in outcome.events if e.kind == "error"]
    assert len(error_events) == 1
    assert "exploded" in error_events[0].message


# ===========================================================================
# Exception details
# ===========================================================================


def test_error_field_contains_exception_message() -> None:
    def work(ctx):
        raise ValueError("bad input")

    outcome = _run(BackgroundJob(name="val-err", func=work))
    assert outcome.status == JobStatus.FAILED
    assert "bad input" in (outcome.error or "")


def test_custom_exception_type_is_caught() -> None:
    class DomainError(Exception):
        pass

    def work(ctx):
        raise DomainError("domain failure")

    outcome = _run(BackgroundJob(name="domain", func=work))
    assert outcome.status == JobStatus.FAILED
    assert "domain failure" in (outcome.error or "")


def test_failed_outcome_value_is_none() -> None:
    def work(ctx):
        raise RuntimeError("oops")

    outcome = _run(BackgroundJob(name="no-value", func=work))
    assert outcome.value is None


def test_succeeded_outcome_error_is_none() -> None:
    def work(ctx):
        return 42

    outcome = _run(BackgroundJob(name="success", func=work))
    assert outcome.error is None
    assert outcome.value == 42


# ===========================================================================
# JobCancelledError cooperative cancellation
# ===========================================================================


def test_job_raising_job_cancelled_error_produces_cancelled_outcome() -> None:
    def work(ctx):
        raise JobCancelledError("user pressed stop")

    outcome = _run(BackgroundJob(name="cooperative-cancel", func=work))
    assert outcome.status == JobStatus.CANCELLED
    assert "user pressed stop" in (outcome.error or "")


def test_context_check_cancelled_raises_when_requested() -> None:
    started = threading.Event()
    cancel_flag = threading.Event()

    def work(ctx):
        started.set()
        while not cancel_flag.is_set():
            time.sleep(0.01)
        ctx.check_cancelled()
        return "should not reach here"

    job = BackgroundJob(name="check-cancel", func=work)
    runner = JobRunner()
    try:
        future = runner.submit(job)
        started.wait(timeout=5)
        job.request_cancel()
        cancel_flag.set()
        outcome = future.result(timeout=5)
    finally:
        runner.shutdown(wait=False)

    assert outcome.status == JobStatus.CANCELLED


def test_context_check_cancelled_does_not_raise_when_not_requested() -> None:
    """check_cancelled() must be a no-op when cancel has not been requested."""
    checked = threading.Event()

    def work(ctx):
        ctx.check_cancelled()  # must not raise
        checked.set()
        return "ok"

    outcome = _run(BackgroundJob(name="safe-check", func=work))
    assert checked.is_set()
    assert outcome.status == JobStatus.SUCCEEDED


# ===========================================================================
# JobContext.emit with custom kind
# ===========================================================================


def test_context_emit_custom_kind_is_preserved() -> None:
    def work(ctx):
        ctx.emit("custom message", kind="warning")
        return "done"

    outcome = _run(BackgroundJob(name="custom-kind", func=work))
    warning_events = [e for e in outcome.events if e.kind == "warning"]
    assert len(warning_events) == 1
    assert warning_events[0].message == "custom message"


# ===========================================================================
# Multiple concurrent jobs
# ===========================================================================


def test_multiple_jobs_run_independently() -> None:
    """Submit two jobs sequentially; both should complete."""
    results: list[str] = []

    runner = JobRunner(max_workers=2)
    try:
        f1 = runner.submit(BackgroundJob(name="j1", func=lambda ctx: "a"))
        f2 = runner.submit(BackgroundJob(name="j2", func=lambda ctx: "b"))
        o1 = f1.result(timeout=5)
        o2 = f2.result(timeout=5)
    finally:
        runner.shutdown()

    assert o1.value == "a"
    assert o2.value == "b"
    assert o1.status == JobStatus.SUCCEEDED
    assert o2.status == JobStatus.SUCCEEDED


# ===========================================================================
# Pre-cancel: func is never called
# ===========================================================================


def test_pre_cancelled_job_never_calls_func() -> None:
    called = threading.Event()

    def work(ctx):
        called.set()
        return "done"

    job = BackgroundJob(name="never-run", func=work)
    job.request_cancel()
    outcome = _run(job)

    assert not called.is_set()
    assert outcome.status == JobStatus.CANCELLED


def test_pre_cancelled_job_emits_exactly_one_event() -> None:
    job = BackgroundJob(name="one-event", func=lambda ctx: "x")
    job.request_cancel()
    outcome = _run(job)

    assert len(outcome.events) == 1
    assert "Cancelled" in outcome.events[0].message


# ===========================================================================
# BackgroundJob identity and cancel flag
# ===========================================================================


def test_background_job_cancel_flag_initially_false() -> None:
    job = BackgroundJob(name="fresh", func=lambda ctx: None)
    assert job.cancel_requested is False


def test_background_job_request_cancel_sets_flag() -> None:
    job = BackgroundJob(name="flag", func=lambda ctx: None)
    job.request_cancel()
    assert job.cancel_requested is True


def test_background_job_request_cancel_is_idempotent() -> None:
    job = BackgroundJob(name="idem", func=lambda ctx: None)
    job.request_cancel()
    job.request_cancel()
    assert job.cancel_requested is True


def test_two_jobs_have_distinct_ids() -> None:
    j1 = BackgroundJob(name="a", func=lambda ctx: None)
    j2 = BackgroundJob(name="b", func=lambda ctx: None)
    assert j1.job_id != j2.job_id


def test_job_outcome_job_id_matches_job() -> None:
    job = BackgroundJob(name="match-id", func=lambda ctx: "x")
    outcome = _run(job)
    assert outcome.job_id == job.job_id
