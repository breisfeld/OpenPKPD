"""Focused tests for GUI bootstrap job orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import JobContext
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.bootstrap_service import (
    BootstrapConfig,
    BootstrapRunResult,
    BootstrapService,
)
from openpkpd_gui.services.fit_service import FitService


def _cache_fit_context(
    service: FitService, workspace: Workspace, monkeypatch, *, fit_run_id: str
) -> None:
    class _FakeParams:
        theta_specs: list[object] = []
        omega_specs: list[object] = []
        sigma_specs: list[object] = []

    class _FakePopulationModel:
        params = _FakeParams()

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
        "Bootstrap demo",
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
    workspace = Workspace(name="Bootstrap demo")

    with pytest.raises(ValueError, match="requires a successful fit"):
        BootstrapService().create_job(workspace, fit_service=FitService())


@pytest.mark.unit
def test_create_job_and_apply_outcome_write_fit_linked_bootstrap_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = Workspace(
        name="Bootstrap demo", root_path=str(tmp_path), workspace_id="bootstrap-demo"
    )
    fit_service = FitService()
    bootstrap_service = BootstrapService()
    fit_run_id = "fit-run-123"
    bootstrap_run_id = "boot-run-456"
    _cache_fit_context(fit_service, workspace, monkeypatch, fit_run_id=fit_run_id)

    class _FakeBootstrapResult:
        n_boot = 20
        n_success = 18

        def summary(self):
            return pd.DataFrame({"parameter": ["THETA(1)"], "mean": [1.2]})

        def ci_table(self, _original):
            return pd.DataFrame({"parameter": ["THETA(1)"], "bca_lo": [1.0], "bca_hi": [1.4]})

        theta_samples = np.array([[1.1, 3.3], [1.3, 3.5]])
        omega_diag_samples = np.array([[0.2], [0.25]])
        sigma_diag_samples = np.array([[0.05], [0.06]])

    class _FakeBootstrapEngine:
        def __init__(
            self,
            population_model,
            initial_params,
            estimation_method,
            n_boot,
            n_jobs,
            seed,
            ci_level,
        ):
            assert population_model is not None
            assert initial_params.theta.tolist() == [1.2, 3.4]
            assert estimation_method == "FOCE"
            assert n_boot == 20
            assert n_jobs == 2
            assert seed == 123
            assert ci_level == 0.9

        def run(self):
            return _FakeBootstrapResult()

    monkeypatch.setattr(
        "openpkpd_gui.services.bootstrap_service.BootstrapEngine", _FakeBootstrapEngine
    )

    runner = JobRunner()
    try:
        outcome = runner.submit(
            bootstrap_service.create_job(
                workspace,
                fit_service=fit_service,
                config=BootstrapConfig(n_boot=20, seed=123, n_jobs=2, ci_level=0.9),
                run_id=bootstrap_run_id,
            )
        ).result(timeout=5)
    finally:
        runner.shutdown()

    run = RunRecord(workflow="bootstrap", run_id=bootstrap_run_id)
    run.mark_running()
    artifacts = bootstrap_service.apply_job_outcome(run, outcome)

    assert outcome.value is not None
    assert isinstance(outcome.value, BootstrapRunResult)
    assert run.status == RunStatus.SUCCEEDED
    assert len(artifacts) == 3
    assert {artifact.metadata.get("artifact_role") for artifact in artifacts} == {
        "bootstrap_summary",
        "bootstrap_ci_table",
        "bootstrap_samples",
    }

    summary = next(
        artifact
        for artifact in artifacts
        if artifact.metadata.get("artifact_role") == "bootstrap_summary"
    )
    ci_table = next(
        artifact
        for artifact in artifacts
        if artifact.metadata.get("artifact_role") == "bootstrap_ci_table"
    )
    samples = next(
        artifact
        for artifact in artifacts
        if artifact.metadata.get("artifact_role") == "bootstrap_samples"
    )
    assert summary.source_run_id == fit_run_id
    assert summary.metadata.get("fit_run_id") == fit_run_id
    assert summary.metadata.get("bootstrap_run_id") == bootstrap_run_id
    assert ci_table.metadata.get("row_count") == 1
    assert samples.metadata.get("row_count") == 2
    assert Path(summary.path or "").exists()
    assert Path(ci_table.path or "").exists()
    assert Path(samples.path or "").exists()
