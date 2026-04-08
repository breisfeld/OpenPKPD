"""Unit tests for ModelTranslationService — validation and builder/control-stream paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from openpkpd_gui.domain.model_spec import EstimationConfig, ModelSpec, ModelSpecMode
from openpkpd_gui.services.model_translation_service import ModelTranslationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svc() -> ModelTranslationService:
    return ModelTranslationService()


def _valid_builder_spec(tmp_path: Path) -> ModelSpec:
    dataset = tmp_path / "demo.csv"
    dataset.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")
    return ModelSpec(
        problem_title="Demo",
        mode=ModelSpecMode.BUILDER,
        dataset_path=str(dataset),
        pk_code="CL = THETA(1) * EXP(ETA(1))\nV = THETA(2)\n",
        error_code="Y = F * (1 + EPS(1))\n",
        theta_rows=[{"init": 1.0, "lower": 0.0}, {"init": 10.0, "lower": 0.0}],
        omega_values=[[0.1]],
        sigma_values=[[0.05]],
    )


def _error_field_names(result) -> list[str]:
    return [
        issue.field_name
        for issue in result.validation.issues
        if issue.severity == "error"
    ]


def _error_messages(result) -> list[str]:
    return [
        issue.message
        for issue in result.validation.issues
        if issue.severity == "error"
    ]


# ===========================================================================
# BUILDER MODE
# ===========================================================================


def test_builder_valid_spec_produces_builder(tmp_path: Path) -> None:
    result = _svc().translate_builder(_valid_builder_spec(tmp_path))
    assert result.ok
    assert result.builder is not None
    assert result.theta_count == 2
    assert result.eta_count == 1
    assert result.eps_count == 1


def test_builder_missing_title_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.problem_title = "   "
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert "problem_title" in _error_field_names(result)


def test_builder_missing_dataset_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.dataset_path = ""
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert "dataset_path" in _error_field_names(result)


def test_builder_nonexistent_dataset_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.dataset_path = str(tmp_path / "nonexistent.csv")
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert any("does not exist" in m for m in _error_messages(result))


def test_builder_dataset_path_is_directory_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.dataset_path = str(tmp_path)
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert any("must point to a file" in m for m in _error_messages(result))


def test_builder_missing_pk_code_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.pk_code = "   "
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert "pk_code" in _error_field_names(result)


def test_builder_missing_error_code_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.error_code = ""
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert "error_code" in _error_field_names(result)


def test_builder_empty_theta_rows_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.theta_rows = []
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert any("THETA" in m for m in _error_messages(result))


def test_builder_invalid_theta_init_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.theta_rows = [{"init": "not-a-number"}]
    result = _svc().translate_builder(spec)
    assert not result.ok


def test_builder_empty_omega_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.omega_values = []
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert any("OMEGA" in m for m in _error_messages(result))


def test_builder_non_square_omega_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.omega_values = [[0.1, 0.0]]  # 1 row, 2 cols → not square
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert any("square" in m.lower() for m in _error_messages(result))


def test_builder_jagged_omega_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.omega_values = [[0.1, 0.0], [0.0]]  # rows have different lengths
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert any("same length" in m for m in _error_messages(result))


def test_builder_non_square_sigma_errors(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.sigma_values = [[0.05, 0.0]]
    result = _svc().translate_builder(spec)
    assert not result.ok


def test_builder_eta_index_exceeds_omega_dimension_errors(tmp_path: Path) -> None:
    """ETA(2) with only a 1×1 OMEGA should produce a validation error."""
    spec = _valid_builder_spec(tmp_path)
    spec.pk_code = "CL = THETA(1) * EXP(ETA(2))\nV = THETA(2)\n"
    spec.omega_values = [[0.1]]  # 1×1 — ETA(2) out of bounds
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert any("ETA(2)" in m for m in _error_messages(result))


def test_builder_eta_within_omega_dimension_is_ok(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.pk_code = "CL = THETA(1) * EXP(ETA(1))\nV = THETA(2)\n"
    spec.omega_values = [[0.1]]
    result = _svc().translate_builder(spec)
    assert result.ok


def test_builder_multiple_errors_collected(tmp_path: Path) -> None:
    """Multiple validation errors should all be reported, not short-circuit."""
    spec = _valid_builder_spec(tmp_path)
    spec.problem_title = ""
    spec.pk_code = ""
    spec.error_code = ""
    result = _svc().translate_builder(spec)
    assert not result.ok
    assert len(result.validation.issues) >= 3


def test_builder_produces_correct_counts(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.theta_rows = [{"init": 1.0}, {"init": 2.0}, {"init": 3.0}]
    spec.omega_values = [[0.1, 0.0], [0.0, 0.2]]
    spec.sigma_values = [[0.05]]
    spec.pk_code = "CL = THETA(1) * EXP(ETA(1))\nV = THETA(2) * EXP(ETA(2))\nKA = THETA(3)\n"
    result = _svc().translate_builder(spec)
    assert result.ok
    assert result.theta_count == 3
    assert result.eta_count == 2
    assert result.eps_count == 1


def test_builder_estimation_method_forwarded(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.estimation = EstimationConfig(method="FOCEI", options={})
    result = _svc().translate_builder(spec)
    assert result.ok
    assert result.estimation_method == "FOCEI"


# ===========================================================================
# CONTROL STREAM MODE
# ===========================================================================


def _valid_cs_text(dataset: str = "demo.csv") -> str:
    return f"""$PROBLEM Test
$DATA {dataset}
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
"""


def test_cs_valid_text_produces_control_stream() -> None:
    spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        control_stream_text=_valid_cs_text(),
    )
    result = _svc().translate_control_stream(spec)
    assert result.ok
    assert result.control_stream is not None
    assert result.theta_count == 1
    assert result.eta_count == 1
    assert result.eps_count == 1


def test_cs_empty_text_errors() -> None:
    spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        control_stream_text="   ",
    )
    result = _svc().translate_control_stream(spec)
    assert not result.ok
    assert any("required" in m.lower() for m in _error_messages(result))


def test_cs_parse_error_produces_validation_errors() -> None:
    """Invalid control stream text is reported as validation errors (not exception)."""
    # Parser is lenient and won't raise, but missing $DATA and zero THETA are errors.
    spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        control_stream_text="$PROBLEM\n$THETA @@INVALID@@\n",
    )
    result = _svc().translate_control_stream(spec)
    assert not result.ok
    # Errors are reported, not exceptions
    assert len(result.validation.issues) > 0


def test_cs_missing_data_record_errors() -> None:
    text = "$PROBLEM Test\n$INPUT ID TIME DV\n$THETA 1\n$OMEGA 0.1\n$SIGMA 0.05\n"
    spec = ModelSpec(mode=ModelSpecMode.CONTROL_STREAM, control_stream_text=text)
    result = _svc().translate_control_stream(spec)
    assert not result.ok
    assert any("$DATA" in m for m in _error_messages(result))


def test_cs_missing_theta_errors() -> None:
    text = "$PROBLEM Test\n$DATA demo.csv\n$INPUT ID TIME DV\n$OMEGA 0.1\n$SIGMA 0.05\n"
    spec = ModelSpec(mode=ModelSpecMode.CONTROL_STREAM, control_stream_text=text)
    result = _svc().translate_control_stream(spec)
    assert not result.ok
    assert any("$THETA" in m for m in _error_messages(result))


def test_cs_extracts_problem_title() -> None:
    text = _valid_cs_text()
    spec = ModelSpec(mode=ModelSpecMode.CONTROL_STREAM, control_stream_text=text)
    result = _svc().translate_control_stream(spec)
    assert result.problem_title == "Test"


def test_cs_extracts_estimation_method() -> None:
    spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        control_stream_text=_valid_cs_text(),
    )
    result = _svc().translate_control_stream(spec)
    # COND INTER maps to FOCE in the parser
    assert result.estimation_method is not None


def test_cs_override_dataset_path_warns_if_missing(tmp_path: Path) -> None:
    spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        control_stream_text=_valid_cs_text(),
        dataset_path=str(tmp_path / "nonexistent.csv"),
    )
    result = _svc().translate_control_stream(spec)
    # Override path given but file missing — should warn, not error
    warnings = [i for i in result.validation.issues if i.severity == "warning"]
    assert any("does not exist" in w.message for w in warnings)


def test_cs_covariance_detected() -> None:
    text = _valid_cs_text() + "$COVARIANCE\n"
    spec = ModelSpec(mode=ModelSpecMode.CONTROL_STREAM, control_stream_text=text)
    result = _svc().translate_control_stream(spec)
    assert result.covariance_enabled


# ===========================================================================
# translate() dispatch
# ===========================================================================


def test_translate_dispatches_to_builder(tmp_path: Path) -> None:
    spec = _valid_builder_spec(tmp_path)
    spec.mode = ModelSpecMode.BUILDER
    result = _svc().translate(spec)
    assert result.builder is not None
    assert result.control_stream is None


def test_translate_dispatches_to_control_stream() -> None:
    spec = ModelSpec(
        mode=ModelSpecMode.CONTROL_STREAM,
        control_stream_text=_valid_cs_text(),
    )
    result = _svc().translate(spec)
    assert result.control_stream is not None
    assert result.builder is None
