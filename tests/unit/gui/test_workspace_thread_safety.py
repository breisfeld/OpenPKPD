"""Thread-safety and data-integrity tests for Workspace domain models."""

from __future__ import annotations

import threading

import pytest

from openpkpd_gui.domain.run_record import RunRecord
from openpkpd_gui.domain.workspace import Project, Scenario, Workspace


# ---------------------------------------------------------------------------
# WS1 — concurrent add_run does not lose runs
# ---------------------------------------------------------------------------


def test_concurrent_add_run_no_loss() -> None:
    """20 threads each add one run; all 20 must be present when done."""
    workspace = Workspace()
    barrier = threading.Barrier(20)

    def _add_one() -> None:
        barrier.wait()  # all threads start together to maximise contention
        workspace.add_run(RunRecord(workflow="fit"))

    threads = [threading.Thread(target=_add_one) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(workspace.active_scenario.runs) == 20


# ---------------------------------------------------------------------------
# WS3 — from_dict raises ValueError for missing ID fields
# ---------------------------------------------------------------------------


def test_scenario_from_dict_missing_id_raises() -> None:
    """Scenario.from_dict({}) must raise ValueError (no scenario_id)."""
    with pytest.raises(ValueError, match="scenario_id"):
        Scenario.from_dict({})


def test_project_from_dict_missing_id_raises() -> None:
    """Project.from_dict({}) must raise ValueError (no project_id)."""
    with pytest.raises(ValueError, match="project_id"):
        Project.from_dict({})


def test_workspace_from_dict_missing_id_raises() -> None:
    """Workspace.from_dict({}) must raise ValueError (no workspace_id)."""
    with pytest.raises(ValueError, match="workspace_id"):
        Workspace.from_dict({})
