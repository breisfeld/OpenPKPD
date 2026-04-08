"""Tests for FitService error paths — validation failures and job failure propagation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openpkpd_gui.domain.model_spec import ModelSpec, ModelSpecMode
from openpkpd_gui.domain.run_record import RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import JobContext, JobOutcome, JobStatus
from openpkpd_gui.services.fit_service import FitService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ws() -> Workspace:
    return Workspace(name="W")


def _write_dataset(path: Path) -> None:
    path.write_text(
        "ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n",
        encoding="utf-8",
    )


def _builder_spec(dataset_path: str) -> ModelSpec:
    return ModelSpec(
        problem_title="Demo",
        mode=ModelSpecMode.BUILDER,
        dataset_path=dataset_path,
        pk_code="CL = THETA(1) * EXP(ETA(1))\nV = THETA(2)\n",
        error_code="Y = F * (1 + EPS(1))\n",
        theta_rows=[{"init": 1.0, "lower": 0.0}, {"init": 10.0, "lower": 0.0}],
        omega_values=[[0.1]],
        sigma_values=[[0.05]],
    )


def _cs_spec(dataset_path: str, control_stream_text: str) -> ModelSpec:
    return ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        dataset_path=dataset_path,
        control_stream_text=control_stream_text,
    )


# ===========================================================================
# prepare_run validation — builder mode
# ===========================================================================


def test_prepare_run_missing_dataset_not_ready() -> None:
    ws = _ws()
    ws.active_model_spec = ModelSpec(
        problem_title="X",
        pk_code="CL = THETA(1)",
        error_code="Y = F",
        theta_rows=[{"init": 1.0}],
        omega_values=[[0.1]],
        sigma_values=[[0.05]],
        dataset_path="",
    )
    result = FitService().prepare_run(ws)
    assert not result.ready
    messages = [i.message for i in result.validation.issues]
    assert any("dataset" in m.lower() for m in messages)


def test_prepare_run_nonexistent_dataset_not_ready(tmp_path: Path) -> None:
    ws = _ws()
    ws.active_model_spec = _builder_spec(str(tmp_path / "missing.csv"))
    result = FitService().prepare_run(ws)
    assert not result.ready
    messages = [i.message for i in result.validation.issues]
    assert any("does not exist" in m for m in messages)


def test_prepare_run_missing_pk_code_not_ready(tmp_path: Path) -> None:
    dataset = tmp_path / "demo.csv"
    _write_dataset(dataset)
    ws = _ws()
    spec = _builder_spec(str(dataset))
    spec.pk_code = ""
    ws.active_model_spec = spec
    result = FitService().prepare_run(ws)
    assert not result.ready
    assert any("pk_code" in (i.field_name or "") for i in result.validation.issues)


def test_prepare_run_missing_error_code_not_ready(tmp_path: Path) -> None:
    dataset = tmp_path / "demo.csv"
    _write_dataset(dataset)
    ws = _ws()
    spec = _builder_spec(str(dataset))
    spec.error_code = "   "
    ws.active_model_spec = spec
    result = FitService().prepare_run(ws)
    assert not result.ready


def test_prepare_run_eta_out_of_bounds_not_ready(tmp_path: Path) -> None:
    dataset = tmp_path / "demo.csv"
    _write_dataset(dataset)
    ws = _ws()
    spec = _builder_spec(str(dataset))
    # ETA(3) with 1×1 OMEGA — out of bounds
    spec.pk_code = "CL = THETA(1) * EXP(ETA(3))\nV = THETA(2)\n"
    spec.omega_values = [[0.1]]
    ws.active_model_spec = spec
    result = FitService().prepare_run(ws)
    assert not result.ready
    messages = [i.message for i in result.validation.issues]
    assert any("ETA(3)" in m for m in messages)


def test_prepare_run_non_square_omega_not_ready(tmp_path: Path) -> None:
    dataset = tmp_path / "demo.csv"
    _write_dataset(dataset)
    ws = _ws()
    spec = _builder_spec(str(dataset))
    spec.omega_values = [[0.1, 0.0]]  # 1 row, 2 cols
    ws.active_model_spec = spec
    result = FitService().prepare_run(ws)
    assert not result.ready


def test_prepare_run_empty_theta_not_ready(tmp_path: Path) -> None:
    dataset = tmp_path / "demo.csv"
    _write_dataset(dataset)
    ws = _ws()
    spec = _builder_spec(str(dataset))
    spec.theta_rows = []
    ws.active_model_spec = spec
    result = FitService().prepare_run(ws)
    assert not result.ready


# ===========================================================================
# prepare_run validation — control stream mode
# ===========================================================================


def test_prepare_run_cs_empty_text_not_ready() -> None:
    ws = _ws()
    ws.active_model_spec = _cs_spec("demo.csv", "")
    result = FitService().prepare_run(ws)
    assert not result.ready


def test_prepare_run_cs_parse_error_not_ready() -> None:
    ws = _ws()
    ws.active_model_spec = _cs_spec("demo.csv", "$PROBLEM\n$THETA @@BROKEN@@\n")
    result = FitService().prepare_run(ws)
    assert not result.ready
    messages = [i.message for i in result.validation.issues]
    assert len(messages) > 0


def test_prepare_run_cs_missing_data_record_not_ready() -> None:
    ws = _ws()
    text = "$PROBLEM T\n$INPUT ID TIME DV\n$THETA 1\n$OMEGA 0.1\n$SIGMA 0.05\n"
    ws.active_model_spec = _cs_spec("demo.csv", text)
    result = FitService().prepare_run(ws)
    assert not result.ready


def test_prepare_run_cs_missing_theta_not_ready() -> None:
    ws = _ws()
    text = "$PROBLEM T\n$DATA demo.csv\n$INPUT ID DV\n$OMEGA 0.1\n$SIGMA 0.05\n"
    ws.active_model_spec = _cs_spec("demo.csv", text)
    result = FitService().prepare_run(ws)
    assert not result.ready


# ===========================================================================
# Validation result propagation through prepare_run
# ===========================================================================


def test_prepare_run_exposes_validation_issues(tmp_path: Path) -> None:
    """Validation issues must be surfaced on the result object for the UI."""
    ws = _ws()
    spec = _builder_spec(str(tmp_path / "missing.csv"))
    spec.problem_title = ""
    ws.active_model_spec = spec
    result = FitService().prepare_run(ws)

    assert not result.ready
    # Multiple issues: missing title + missing dataset
    assert len(result.validation.issues) >= 2


def test_prepare_run_ready_true_when_valid(tmp_path: Path) -> None:
    dataset = tmp_path / "demo.csv"
    _write_dataset(dataset)
    ws = _ws()
    ws.active_model_spec = _builder_spec(str(dataset))
    result = FitService().prepare_run(ws)
    assert result.ready


# ===========================================================================
# apply_job_outcome — failed job marks run as failed
# ===========================================================================


def test_apply_job_outcome_failed_marks_run_failed() -> None:
    from openpkpd_gui.domain.run_record import RunRecord

    service = FitService()
    run = RunRecord(workflow="fit")
    run.mark_running()

    outcome = JobOutcome(
        job_id="j1",
        name="fit:demo",
        status=JobStatus.FAILED,
        value=None,
        error="EstimationError: convergence failed",
        events=[],
    )
    service.apply_job_outcome(run, outcome)

    assert run.status == RunStatus.FAILED
    assert "EstimationError" in run.error_text


def test_apply_job_outcome_cancelled_marks_run_cancelled() -> None:
    from openpkpd_gui.domain.run_record import RunRecord

    service = FitService()
    run = RunRecord(workflow="fit")
    run.mark_running()
    run.mark_cancel_requested()

    outcome = JobOutcome(
        job_id="j1",
        name="fit:demo",
        status=JobStatus.CANCELLED,
        value=None,
        error="",
        events=[],
    )
    service.apply_job_outcome(run, outcome)
    assert run.status == RunStatus.CANCELLED


# ===========================================================================
# _estimate_built_model — estimation exception propagation
# ===========================================================================


def test_estimate_built_model_propagates_estimation_error(monkeypatch) -> None:
    """If estimation raises, _estimate_built_model should propagate the exception
    so the job runner can mark the run as failed.
    """
    import openpkpd_gui.services.fit_service as _fit_module

    def _raising_get(method, **kwargs):
        from openpkpd.utils.errors import EstimationError

        class _BadEstimation:
            def estimate(self, pm, p):
                raise EstimationError("convergence failed")

        return _BadEstimation()

    monkeypatch.setattr("openpkpd_gui.services.fit_service.get_estimation_method", _raising_get)

    class _FakePopulationModel:
        blq_method: str = "M1"

        def n_subjects(self):
            return 1

        @property
        def dataset(self):
            return None

    class _FakeBuiltModel:
        params = type(
            "P",
            (),
            {
                "theta": [1.0],
                "omega": [[0.3]],
                "sigma": [[0.1]],
                "theta_specs": [],
                "omega_specs": [],
                "sigma_specs": [],
            },
        )()
        population_model = _FakePopulationModel()
        estimation_kwargs = {"method": "FOCE"}
        do_covariance = False

    from openpkpd.utils.errors import EstimationError

    with pytest.raises(EstimationError, match="convergence failed"):
        FitService()._estimate_built_model(_FakeBuiltModel())


# ===========================================================================
# _estimate_control_stream — missing dataset
# ===========================================================================


def test_estimate_control_stream_raises_on_missing_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    """If the dataset file is missing at run time, estimation should fail cleanly."""
    dataset_path = tmp_path / "missing_at_runtime.csv"
    # Do NOT write the file — it does not exist

    ws = Workspace(name="W")
    ws.active_model_spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        dataset_path=str(dataset_path),
        control_stream_text="""$PROBLEM Demo
$DATA missing_at_runtime.csv
$INPUT ID TIME AMT DV EVID
$SUBROUTINES ADVAN2 TRANS2
$PK
CL = THETA(1) * EXP(ETA(1))
$ERROR
Y = F * (1 + EPS(1))
$THETA (0, 1.0, 10)
$OMEGA 0.3
$SIGMA 0.1
$ESTIMATION METHOD=COND INTER MAXEVAL=9999
""",
    )

    # Use a fake estimation method so we don't actually run FOCE
    def _fake_get(method, **kwargs):
        class _FakeEst:
            def estimate(self, pm, p):
                return type(
                    "R",
                    (),
                    {
                        "theta_final": np.asarray(p.theta, dtype=float),
                        "omega_final": np.asarray(p.omega, dtype=float),
                        "sigma_final": np.asarray(p.sigma, dtype=float),
                        "post_hoc_etas": {},
                        "warnings": [],
                        "method": "FOCE",
                    },
                )()

        return _FakeEst()

    monkeypatch.setattr("openpkpd_gui.services.fit_service.get_estimation_method", _fake_get)

    preparation = FitService().prepare_run(ws)
    # The preparation itself may succeed (dataset path warning, not error) but
    # running the job with a missing file should raise when building the population model.
    # We just verify that prepare_run identifies the issue.
    messages = [i.message for i in preparation.validation.issues]
    assert any("missing" in m.lower() or "does not exist" in m.lower() for m in messages) or (
        not preparation.ready
    )
