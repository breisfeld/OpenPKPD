"""Focused tests for GUI design job orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import JobContext
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.design_service import DesignConfig, DesignRunResult, DesignService
from openpkpd_gui.services.fit_service import FitService


def _cache_fit_context(
    service: FitService, workspace: Workspace, monkeypatch, *, fit_run_id: str
) -> None:
    class _FakeParams:
        theta_specs: list[object] = []
        omega_specs: list[object] = []
        sigma_specs: list[object] = []

    class _FakeSubjectEvents:
        obs_times = np.array([1.0, 2.0, 4.0, 8.0])

    class _FakeIndividualModel:
        subject_events = _FakeSubjectEvents()

    class _FakePopulationModel:
        params = _FakeParams()

        def subject_ids(self):
            return iter([1])

        def individual_model(self, _subject_id):
            return _FakeIndividualModel()

    class _FakeResult:
        warnings: list[str] = []
        converged = True
        ofv = 12.34
        method = "FOCE"
        theta_final = np.array([1.2, 3.4])
        omega_final = np.diag([0.2])
        sigma_final = np.diag([0.05])

    workspace.runs = [RunRecord(workflow="fit", run_id=fit_run_id, status=RunStatus.SUCCEEDED)]
    monkeypatch.setattr(service, "_generate_output_artifacts", lambda *_a, **_k: [])
    service._fit_run_result_from_estimation(
        JobContext(lambda *_args: None),
        workspace,
        "Design demo",
        _FakeResult(),
        "FOCE",
        None,
        fit_run_id,
        project_id=workspace.active_project.project_id,
        scenario_id=workspace.active_scenario.scenario_id,
        dataset_path=str(Path(workspace.root_path or ".") / "theo.csv"),
        population_model=_FakePopulationModel(),
    )


def test_create_job_requires_cached_fit_context() -> None:
    workspace = Workspace(name="Design demo")

    with pytest.raises(ValueError, match="requires a successful fit"):
        DesignService().create_job(workspace, fit_service=FitService())


@pytest.mark.unit
def test_create_job_and_apply_outcome_write_fit_linked_design_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = Workspace(name="Design demo", root_path=str(tmp_path), workspace_id="design-demo")
    fit_service = FitService()
    design_service = DesignService()
    fit_run_id = "fit-run-123"
    design_run_id = "design-run-456"
    _cache_fit_context(fit_service, workspace, monkeypatch, fit_run_id=fit_run_id)

    class _FakeDesignResult:
        sampling_times = np.array([0.5, 2.0, 8.0])
        information_matrix = np.array([[10.0, 1.0], [1.0, 5.0]])
        d_efficiency = float("nan")
        a_efficiency = 0.33
        condition_number = 4.2
        se_theta = np.array([0.1, 0.2])

        def summary(self):
            return "Design Result (3 sampling times)"

    class _FakePFIMEngine:
        def __init__(self, population_model, init_params, sampling_times=None):
            assert population_model is not None
            assert init_params.theta.tolist() == [1.2, 3.4]
            assert sampling_times is None

        def optimize_design(self, n_samples, t_min, t_max, n_subjects, criterion, method, n_starts):
            assert n_samples == 3
            assert t_min == 0.0
            assert t_max == 12.0
            assert n_subjects == 20
            assert criterion == "A"
            assert method == "L-BFGS-B"
            assert n_starts == 4
            return _FakeDesignResult()

        def efficiency(self, times_test, times_reference, criterion, n_subjects):
            np.testing.assert_allclose(times_test, np.array([0.5, 2.0, 8.0]))
            np.testing.assert_allclose(times_reference, np.array([1.0, 2.0, 4.0, 8.0]))
            assert criterion == "A"
            assert n_subjects == 20
            return 1.25

    monkeypatch.setattr("openpkpd_gui.services.design_service.PFIMEngine", _FakePFIMEngine)

    runner = JobRunner()
    try:
        outcome = runner.submit(
            design_service.create_job(
                workspace,
                fit_service=fit_service,
                config=DesignConfig(
                    n_samples=3,
                    t_min=0.0,
                    t_max=12.0,
                    n_subjects=20,
                    criterion="A",
                    method="L-BFGS-B",
                    n_starts=4,
                ),
                run_id=design_run_id,
            )
        ).result(timeout=5)
    finally:
        runner.shutdown()

    run = RunRecord(workflow="design", run_id=design_run_id)
    run.mark_running()
    artifacts = design_service.apply_job_outcome(run, outcome)

    assert outcome.value is not None
    assert isinstance(outcome.value, DesignRunResult)
    assert run.status == RunStatus.SUCCEEDED
    assert len(artifacts) == 5
    assert {artifact.metadata.get("artifact_role") for artifact in artifacts} == {
        "design_summary",
        "design_metrics",
        "design_schedule",
        "design_fim",
        "design_expected_se",
    }

    summary = next(
        artifact
        for artifact in artifacts
        if artifact.metadata.get("artifact_role") == "design_summary"
    )
    schedule = next(
        artifact
        for artifact in artifacts
        if artifact.metadata.get("artifact_role") == "design_schedule"
    )
    assert summary.source_run_id == fit_run_id
    assert summary.metadata.get("fit_run_id") == fit_run_id
    assert summary.metadata.get("design_run_id") == design_run_id
    assert schedule.metadata.get("row_count") == 7
    assert Path(summary.path or "").exists()
    assert Path(schedule.path or "").exists()
