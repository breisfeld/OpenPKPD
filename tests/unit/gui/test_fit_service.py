"""Focused tests for fit preparation and job orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec, ModelSpecMode
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import JobContext, JobOutcome, JobStatus
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.fit_service import FitRunResult, FitService


def _write_dataset(path: Path) -> None:
    path.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")


def test_prepare_run_requires_saved_dataset_and_model() -> None:
    result = FitService().prepare_run(Workspace(name="Demo"))

    assert result.ready is False
    messages = [issue.message for issue in result.validation.issues]
    assert any("Load a dataset" in message for message in messages)
    assert any("Configure a model" in message for message in messages)


def test_prepare_run_builder_mode_success(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    _write_dataset(dataset_path)
    workspace = Workspace(name="Builder demo")
    workspace.active_model_spec = ModelSpec(
        problem_title="Builder demo",
        dataset_path=str(dataset_path),
        pk_code="CL = THETA(1) * EXP(ETA(1))",
        error_code="Y = F * (1 + EPS(1))",
        theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
        omega_values=[[0.3]],
        sigma_values=[[0.1]],
    )

    result = FitService().prepare_run(workspace)

    assert result.ready is True
    assert result.mode == ModelSpecMode.BUILDER
    assert result.problem_title == "Builder demo"
    assert result.dataset_path == str(dataset_path)
    assert result.estimation_method == "FOCE"
    assert result.theta_count == 1


def test_prepare_run_builder_mode_uses_active_dataset_import_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "nmdata.csv"
    dataset_path.write_text("1,0,999,70,5\n", encoding="utf-8")
    workspace = Workspace(name="Builder NMData demo")
    workspace.active_dataset = DatasetAsset(
        source_path=str(dataset_path),
        display_name="nmdata.csv",
        separator=",",
        input_columns=["ID", "TIME", "_DROP_3", "WT", "DV"],
        columns=["ID", "TIME", "WT", "DV", "EVID", "MDV"],
    )
    workspace.active_model_spec = ModelSpec(
        problem_title="Builder demo",
        dataset_path=str(dataset_path),
        pk_code="CL = THETA(1) * EXP(ETA(1))",
        error_code="Y = F * (1 + EPS(1))",
        theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
        omega_values=[[0.3]],
        sigma_values=[[0.1]],
    )

    result = FitService().prepare_run(workspace)

    assert result.ready is True
    assert result.translation is not None
    assert result.translation.builder is not None
    hydrated_dataset = result.translation.builder._dataset
    assert hydrated_dataset is not None
    assert list(hydrated_dataset.df.columns[:4]) == ["ID", "TIME", "WT", "DV"]
    assert float(hydrated_dataset.df.loc[0, "WT"]) == pytest.approx(70.0)


def test_prepare_run_control_stream_success(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    _write_dataset(dataset_path)
    workspace = Workspace(name="Control stream demo")
    workspace.active_model_spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        dataset_path=str(dataset_path),
        control_stream_text="""$PROBLEM Demo
$DATA theo.csv
$INPUT ID TIME AMT DV EVID
$SUBROUTINES ADVAN2 TRANS2
$PK
CL = THETA(1) * EXP(ETA(1))
$ERROR
Y = F * (1 + EPS(1))
$THETA (0, 1.0, 10)
$OMEGA 0.3
$SIGMA 0.1
$ESTIMATION METHOD=FOCE
""",
    )

    result = FitService().prepare_run(workspace)

    assert result.ready is True
    assert result.mode == ModelSpecMode.CONTROL_STREAM
    assert result.dataset_path == str(dataset_path)
    assert result.estimation_method == "FOCE"


def test_estimate_control_stream_forwards_advanced_estimation_options(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / "theo.csv"
    _write_dataset(dataset_path)
    workspace = Workspace(name="Advanced ctl demo")
    workspace.active_model_spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        dataset_path=str(dataset_path),
        control_stream_text="""$PROBLEM Demo
$DATA theo.csv
$INPUT ID TIME AMT DV EVID
$SUBROUTINES ADVAN2 TRANS2
$PK
CL = THETA(1) * EXP(ETA(1))
$ERROR
Y = F * (1 + EPS(1))
$THETA (0, 1.0, 10)
$OMEGA 0.3
$SIGMA 0.1
$ESTIMATION METHOD=COND INTER MAXEVAL=123 OUTEROPT=POWELL FALLBACKOPT=L-BFGS-B FALLBACKMAXEVAL=17 RETAINBEST RETRYONABNORMAL RETRYOMEGASCALE=0.6,0.3
""",
    )
    control_stream = FitService().prepare_run(workspace).translation.control_stream
    assert control_stream is not None

    captured: dict[str, object] = {}

    class _FakeEstimation:
        def estimate(self, population_model, params):
            return type(
                "FakeEstimationResult",
                (),
                {
                    "theta_final": np.asarray(params.theta, dtype=float),
                    "omega_final": np.asarray(params.omega, dtype=float),
                    "sigma_final": np.asarray(params.sigma, dtype=float),
                    "post_hoc_etas": {},
                    "warnings": [],
                    "method": "FOCEI",
                },
            )()

    def _fake_get_estimation_method(method, **kwargs):
        captured["method"] = method
        captured["kwargs"] = kwargs
        return _FakeEstimation()

    monkeypatch.setattr("openpkpd_gui.services.fit_service.get_estimation_method", _fake_get_estimation_method)

    service = FitService()
    service._estimate_control_stream(
        JobContext(lambda *_args, **_kwargs: None),
        control_stream,
        str(dataset_path),
        n_parallel=2,
    )

    assert captured["method"] == "FOCE"
    assert captured["kwargs"] == {
        "interaction": True,
        "maxeval": 123,
        "n_parallel": 2,
        "outer_optimizer": "POWELL",
        "outer_fallback_optimizer": "L-BFGS-B",
        "outer_fallback_maxeval": 17,
        "retain_best_iterate": True,
        "retry_on_abnormal": True,
        "retry_omega_scales": (0.6, 0.3),
    }


def test_create_job_builder_mode_and_apply_outcome(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    _write_dataset(dataset_path)
    service = FitService()
    workspace = Workspace(name="Builder demo")
    workspace.active_model_spec = ModelSpec(
        problem_title="Builder demo",
        dataset_path=str(dataset_path),
        pk_code="CL = THETA(1) * EXP(ETA(1))",
        error_code="Y = F * (1 + EPS(1))",
        theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
        omega_values=[[0.3]],
        sigma_values=[[0.1]],
    )
    preparation = service.prepare_run(workspace)
    assert preparation.translation is not None
    assert preparation.translation.builder is not None

    class _FakeBuiltModel:
        params = object()
        population_model = object()
        estimation_kwargs = {"method": "FOCE"}
        do_covariance = False

    captured: dict[str, int] = {}

    def _fake_estimate(_built_model: object, *, n_parallel: int = 0, ctx=None) -> object:
        captured["n_parallel"] = n_parallel
        return type(
            "FakeEstimationResult",
            (),
            {
                "converged": True,
                "ofv": 12.34,
                "warnings": ["covariance skipped"],
                "method": "FOCE",
            },
        )()

    preparation.translation.builder.build = lambda: _FakeBuiltModel()  # type: ignore[method-assign]
    service._estimate_built_model = _fake_estimate  # type: ignore[method-assign]

    runner = JobRunner()
    try:
        outcome = runner.submit(
            service.create_job(workspace, preparation=preparation, n_parallel=3)
        ).result(timeout=5)
    finally:
        runner.shutdown()

    run = RunRecord(workflow="fit")
    run.mark_running()
    artifacts = service.apply_job_outcome(run, outcome)

    assert outcome.value is not None
    assert isinstance(outcome.value, FitRunResult)
    assert run.status == RunStatus.SUCCEEDED
    assert artifacts == []
    assert captured["n_parallel"] == 3
    assert "Builder demo" in run.summary_text
    assert any("Started fit:Builder demo" in line for line in run.log_lines)
    assert any("Generating fit outputs" in line for line in run.log_lines)
    assert any("covariance skipped" in line for line in run.log_lines)


def test_apply_job_outcome_links_generated_artifacts() -> None:
    service = FitService()
    run = RunRecord(workflow="fit")
    run.mark_running()
    artifact = ArtifactRecord(kind="report", label="Report", path="/tmp/report.html")
    outcome = JobOutcome(
        job_id="job-1",
        name="fit:demo",
        status=JobStatus.SUCCEEDED,
        value=FitRunResult(
            problem_title="Demo",
            estimation_method="FOCE",
            converged=True,
            ofv=12.34,
            summary_text="Demo • FOCE • converged=True • OFV=12.3400",
            artifacts=[artifact],
        ),
    )

    artifacts = service.apply_job_outcome(run, outcome)

    assert artifacts == [artifact]
    assert artifact.source_run_id == run.run_id
    assert run.artifact_ids == [artifact.artifact_id]


def test_apply_job_outcome_marks_cancelled_run() -> None:
    service = FitService()
    run = RunRecord(workflow="fit")
    run.mark_running()
    run.mark_cancel_requested()
    outcome = JobOutcome(
        job_id="job-1",
        name="fit:demo",
        status=JobStatus.CANCELLED,
        error="Cancelled by user.",
    )

    artifacts = service.apply_job_outcome(run, outcome)

    assert artifacts == []
    assert run.status == RunStatus.CANCELLED
    assert run.error_text == "Cancelled by user."
    assert run.cancel_requested_at is not None


@pytest.mark.unit
def test_latest_fit_context_requires_latest_successful_fit(monkeypatch) -> None:
    service = FitService()
    workspace = Workspace(name="Context demo")
    fit_run = RunRecord(workflow="fit", run_id="fit-run-1", status=RunStatus.SUCCEEDED)
    workspace.runs = [fit_run]
    population_model = object()

    class _FakeResult:
        warnings: list[str] = []
        converged = True
        ofv = 12.34
        method = "FOCE"

    monkeypatch.setattr(service, "_generate_output_artifacts", lambda *_a, **_k: [])

    service._fit_run_result_from_estimation(
        JobContext(lambda *_args: None),
        workspace,
        "Context demo",
        _FakeResult(),
        "FOCE",
        None,
        fit_run.run_id,
        project_id=workspace.active_project.project_id,
        scenario_id=workspace.active_scenario.scenario_id,
        dataset_path="/tmp/theo.csv",
        population_model=population_model,
    )

    context = service.latest_fit_context(workspace)

    assert context is not None
    assert context.fit_run_id == fit_run.run_id
    assert context.problem_title == "Context demo"
    assert context.estimation_method == "FOCE"
    assert context.dataset_path == "/tmp/theo.csv"
    assert context.population_model is population_model

    fit_run.status = RunStatus.FAILED
    assert service.latest_fit_context(workspace) is None

    fit_run.status = RunStatus.SUCCEEDED
    workspace.runs.append(RunRecord(workflow="fit", run_id="fit-run-2", status=RunStatus.SUCCEEDED))
    assert service.latest_fit_context(workspace) is None


@pytest.mark.unit
def test_generate_output_artifacts_includes_tables_and_additional_plot_types(
    tmp_path: Path,
    monkeypatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    service = FitService()
    workspace = Workspace(
        name="Artifact demo", root_path=str(tmp_path), workspace_id="artifact-demo"
    )
    dataset_path = tmp_path / "artifact-demo.csv"
    _write_dataset(dataset_path)
    workspace.active_dataset = DatasetAsset(
        source_path=str(dataset_path),
        display_name="artifact-demo.csv",
        columns=["ID", "TIME", "AMT", "DV", "EVID"],
        row_count=2,
        subject_count=1,
        observation_count=1,
    )
    workspace.active_model_spec = ModelSpec(
        problem_title="Artifact demo",
        dataset_path=str(dataset_path),
        pk_code="CL = THETA(1) * EXP(ETA(1))",
        error_code="Y = F * (1 + EPS(1))",
        theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0}],
        omega_values=[[0.2]],
        sigma_values=[[0.1]],
    )

    class _FakeResult:
        theta_final = [1.0]
        omega_final = [[0.2]]
        sigma_final = [[0.1]]
        warnings: list[str] = []
        converged = True
        ofv = 12.34
        method = "FOCE"

    def _make_figure():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        return fig

    captured_provenance: dict[str, object] = {}

    def _fake_write_html_report(path, _result, _params, title, **kwargs) -> None:
        captured_provenance.update(kwargs.get("provenance") or {})
        Path(path).write_text(f"<html><body>{title}</body></html>", encoding="utf-8")

    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.write_html_report", _fake_write_html_report
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.ofv_history", lambda *_a, **_k: _make_figure()
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.parameter_uncertainty_plot",
        lambda *_a, **_k: _make_figure(),
    )
    diag_df = pd.DataFrame(
        {
            "ID": [1],
            "TIME": [1.0],
            "DV": [5.0],
            "PRED": [4.5],
            "IPRED": [4.8],
            "ETA1": [0.1],
        }
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.compute_diagnostics", lambda *_a, **_k: diag_df
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.diagnostic_panel",
        lambda *_a, **_k: _make_figure(),
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.residual_trends_plot",
        lambda *_a, **_k: _make_figure(),
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.spaghetti_plot",
        lambda *_a, **_k: _make_figure(),
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.dv_vs_ipred", lambda *_a, **_k: _make_figure()
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.dv_vs_pred", lambda *_a, **_k: _make_figure()
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.cwres_vs_time", lambda *_a, **_k: _make_figure()
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.cwres_vs_pred", lambda *_a, **_k: _make_figure()
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.cwres_qq", lambda *_a, **_k: _make_figure()
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.abs_iwres_vs_ipred",
        lambda *_a, **_k: _make_figure(),
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.mean_profile", lambda *_a, **_k: _make_figure()
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.eta_histograms", lambda *_a, **_k: _make_figure()
    )
    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.eta_pairs", lambda *_a, **_k: _make_figure()
    )

    artifacts = service._generate_output_artifacts(
        JobContext(lambda *_args: None),
        workspace,
        "Artifact demo",
        "FOCE",
        _FakeResult(),
        object(),
        run_id="run-1234",
        project_id=workspace.active_project.project_id,
        scenario_id=workspace.active_scenario.scenario_id,
        dataset_path=str(dataset_path),
        population_model=object(),
    )

    assert {artifact.kind for artifact in artifacts} == {"report", "plot", "table"}
    assert {
        artifact.metadata.get("artifact_role") for artifact in artifacts if artifact.kind == "table"
    } == {"diagnostics_table"}
    assert {
        artifact.metadata.get("plot_type") for artifact in artifacts if artifact.kind == "plot"
    } == {
        "ofv_history",
        "parameter_uncertainty",
        "gof_panel",
        "residual_trends",
        "spaghetti_plot",
        "dv_vs_ipred",
        "dv_vs_pred",
        "cwres_vs_time",
        "cwres_vs_pred",
        "cwres_qq",
        "abs_iwres_vs_ipred",
        "mean_profile",
        "eta_histograms",
        "eta_pairs",
    }
    assert all(Path(artifact.path or "").exists() for artifact in artifacts)
    assert all(workspace.workspace_id in Path(artifact.path or "").parts for artifact in artifacts)
    assert all(
        workspace.active_project.project_id in Path(artifact.path or "").parts
        for artifact in artifacts
    )
    assert all(
        workspace.active_scenario.scenario_id in Path(artifact.path or "").parts
        for artifact in artifacts
    )
    assert captured_provenance["Run context"]["run_id"] == "run-1234"
    assert captured_provenance["Dataset"]["display_name"] == "artifact-demo.csv"
    assert captured_provenance["Dataset"]["sha256"]
    assert captured_provenance["Model"]["problem_title"] == "Artifact demo"
    assert captured_provenance["Model source"]["pk_code"] == "CL = THETA(1) * EXP(ETA(1))"
    assert captured_provenance["Estimation settings"]["effective_method"] == "FOCE"
    assert captured_provenance["Environment"]["openpkpd_version"]


def test_restore_fit_context_records_error_when_scenario_missing() -> None:
    service = FitService()
    workspace = Workspace(name="Demo")

    ok = service.restore_fit_context(
        workspace,
        {
            "project_id": "missing-project",
            "scenario_id": "missing-scenario",
            "fit_run_id": "fit-1",
            "problem_title": "Demo",
            "estimation_method": "FOCE",
            "dataset_path": None,
            "estimation_result": {
                "theta_final": [1.0],
                "omega_final": [[1.0]],
                "sigma_final": [[1.0]],
                "ofv": 1.0,
                "converged": True,
                "condition_number": None,
                "eta_shrinkage": [],
                "eps_shrinkage": [],
                "post_hoc_etas": {},
                "ofv_history": [],
                "warnings": [],
                "n_function_evals": 0,
                "elapsed_time": 0.0,
                "method": "FOCE",
                "message": "ok",
                "n_observations": 0,
                "n_subjects": 0,
            },
        },
    )

    assert ok is False
    assert service.last_context_error is not None
    assert "was not found" in service.last_context_error


def test_rebuild_population_model_records_translation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FitService()
    workspace = Workspace(name="Demo")
    workspace.active_model_spec = ModelSpec(problem_title="Demo")

    monkeypatch.setattr(
        service._translation_service,
        "translate",
        lambda _spec: type(
            "Translation",
            (),
            {
                "ok": False,
                "validation": type(
                    "Validation",
                    (),
                    {"issues": [type("Issue", (), {"message": "translation failed"})()]},
                )(),
            },
        )(),
    )

    population_model = service._rebuild_population_model(workspace.active_scenario, None)

    assert population_model is None
    assert service.last_context_error is not None
    assert "translation failed" in service.last_context_error


@pytest.mark.unit
def test_restore_fit_context_payloads_enables_reuse_after_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_service = FitService()
    source_workspace = Workspace(name="Context round trip")
    fit_run = RunRecord(workflow="fit", run_id="fit-run-1", status=RunStatus.SUCCEEDED)
    source_workspace.runs = [fit_run]

    class _FakeResult:
        theta_final = np.array([1.0])
        omega_final = np.array([[0.2]])
        sigma_final = np.array([[0.1]])
        condition_number = None
        eta_shrinkage = np.array([])
        eps_shrinkage = np.array([])
        post_hoc_etas = {}
        ofv_history = [12.34]
        warnings: list[str] = []
        n_function_evals = 1
        elapsed_time = 0.1
        n_observations = 1
        n_subjects = 1
        converged = True
        ofv = 12.34
        method = "FOCE"
        message = "ok"

    monkeypatch.setattr(source_service, "_generate_output_artifacts", lambda *_a, **_k: [])
    population_model = object()
    source_service._fit_run_result_from_estimation(
        JobContext(lambda *_args: None),
        source_workspace,
        "Context demo",
        _FakeResult(),
        "FOCE",
        None,
        fit_run.run_id,
        project_id=source_workspace.active_project.project_id,
        scenario_id=source_workspace.active_scenario.scenario_id,
        dataset_path="/tmp/theo.csv",
        population_model=population_model,
    )

    payloads = source_service.all_fit_context_payloads(source_workspace)
    target_workspace = source_workspace
    target_service = FitService()
    restored_model = object()
    monkeypatch.setattr(target_service, "_rebuild_population_model", lambda *_a, **_k: restored_model)

    restored_count, warnings = target_service.restore_fit_context_payloads(target_workspace, payloads)

    assert restored_count == 1
    assert warnings == []
    restored_context = target_service.latest_fit_context(target_workspace)
    assert restored_context is not None
    assert restored_context.fit_run_id == fit_run.run_id
    assert restored_context.population_model is restored_model


@pytest.mark.unit
def test_restore_fit_context_payloads_collects_decode_and_restore_failures() -> None:
    service = FitService()
    workspace = Workspace(name="Restore warnings")

    restored_count, warnings = service.restore_fit_context_payloads(
        workspace,
        {
            ("proj-a", "scen-a"): b"{bad json",
            ("proj-b", "scen-b"): json.dumps(
                {
                    "project_id": "proj-b",
                    "scenario_id": "scen-b",
                    "fit_run_id": "fit-1",
                    "problem_title": "Demo",
                    "estimation_method": "FOCE",
                    "dataset_path": None,
                    "estimation_result": {
                        "theta_final": [1.0],
                        "omega_final": [[1.0]],
                        "sigma_final": [[1.0]],
                        "ofv": 1.0,
                        "converged": True,
                        "condition_number": None,
                        "eta_shrinkage": [],
                        "eps_shrinkage": [],
                        "post_hoc_etas": {},
                        "ofv_history": [],
                        "warnings": [],
                        "n_function_evals": 0,
                        "elapsed_time": 0.0,
                        "method": "FOCE",
                        "message": "ok",
                        "n_observations": 0,
                        "n_subjects": 0,
                    },
                }
            ).encode(),
        },
    )

    assert restored_count == 0
    assert len(warnings) == 2
    assert any("Could not decode saved fit state" in warning for warning in warnings)
    assert any("was not found" in warning for warning in warnings)


# ---------------------------------------------------------------------------
# P2-C: LOQ injection and blq_method routing in FitService
# ---------------------------------------------------------------------------


def _write_dataset_with_lloq(path: Path) -> None:
    path.write_text("ID,TIME,AMT,DV,EVID,LLOQ\n1,0,100,0,1,0.5\n1,1,0,5,0,0.5\n", encoding="utf-8")


def test_apply_dataset_asset_injects_scalar_loq_as_lloq_column(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    _write_dataset(dataset_path)
    dataset_asset = DatasetAsset(
        source_path=str(dataset_path),
        display_name="theo.csv",
        loq=0.5,
    )

    captured_dataset = {}

    class _FakeBuilder:
        def dataset(self, ds):
            captured_dataset["ds"] = ds

    error = FitService._apply_dataset_asset_to_builder(_FakeBuilder(), dataset_asset)

    assert error is None
    ds = captured_dataset["ds"]
    assert "LLOQ" in ds.df.columns
    assert float(ds.df["LLOQ"].iloc[0]) == pytest.approx(0.5)


def test_apply_dataset_asset_does_not_overwrite_existing_lloq(tmp_path: Path) -> None:
    dataset_path = tmp_path / "lloq.csv"
    _write_dataset_with_lloq(dataset_path)
    dataset_asset = DatasetAsset(
        source_path=str(dataset_path),
        display_name="lloq.csv",
        loq=99.0,  # should be ignored — file already has LLOQ
    )

    captured_dataset = {}

    class _FakeBuilder:
        def dataset(self, ds):
            captured_dataset["ds"] = ds

    FitService._apply_dataset_asset_to_builder(_FakeBuilder(), dataset_asset)

    ds = captured_dataset["ds"]
    # Original LLOQ values from file (0.5) must not be overwritten with 99
    assert float(ds.df["LLOQ"].iloc[0]) == pytest.approx(0.5)


def test_apply_dataset_asset_zero_loq_does_not_inject(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    _write_dataset(dataset_path)
    dataset_asset = DatasetAsset(
        source_path=str(dataset_path),
        display_name="theo.csv",
        loq=0.0,
    )

    captured_dataset = {}

    class _FakeBuilder:
        def dataset(self, ds):
            captured_dataset["ds"] = ds

    FitService._apply_dataset_asset_to_builder(_FakeBuilder(), dataset_asset)

    ds = captured_dataset["ds"]
    assert "LLOQ" not in ds.df.columns


def test_estimate_built_model_pops_blq_method_and_applies_to_population_model(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeEstimation:
        def estimate(self, population_model, params):
            return type(
                "FakeResult",
                (),
                {
                    "theta_final": np.asarray([1.0]),
                    "omega_final": np.asarray([[0.3]]),
                    "sigma_final": np.asarray([[0.1]]),
                    "post_hoc_etas": {},
                    "warnings": [],
                    "method": "FOCE",
                    "n_observations": 0,
                    "n_subjects": 0,
                    "compute_n_parameters": lambda *_a, **_kw: None,
                },
            )()

    def _fake_get_estimation_method(method, **kwargs):
        captured["method"] = method
        captured["kwargs"] = kwargs
        return _FakeEstimation()

    monkeypatch.setattr(
        "openpkpd_gui.services.fit_service.get_estimation_method",
        _fake_get_estimation_method,
    )

    class _FakePopulationModel:
        blq_method: str = "M1"

        def n_subjects(self):
            return 1

        @property
        def dataset(self):
            return None

    class _FakeBuiltModel:
        params = type("P", (), {"theta": [1.0], "omega": [[0.3]], "sigma": [[0.1]], "theta_specs": [], "omega_specs": [], "sigma_specs": []})()
        population_model = _FakePopulationModel()
        estimation_kwargs = {"method": "FOCE", "blq_method": "M3"}
        do_covariance = False

    FitService()._estimate_built_model(_FakeBuiltModel())

    assert _FakeBuiltModel.population_model.blq_method == "M3"
    # blq_method must NOT be forwarded to get_estimation_method
    assert "blq_method" not in captured.get("kwargs", {})
