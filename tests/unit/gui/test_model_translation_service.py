"""Focused tests for ModelSpec translation and validation behavior."""

from __future__ import annotations

from pathlib import Path

from openpkpd_gui.domain.model_spec import ModelSpec, ModelSpecMode
from openpkpd_gui.services.model_translation_service import ModelTranslationService


def _write_dataset(path: Path) -> None:
    path.write_text("ID,TIME,AMT,DV,EVID\n1,0,100,0,1\n1,1,0,5,0\n", encoding="utf-8")


def test_translate_builder_success(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    _write_dataset(dataset_path)
    service = ModelTranslationService()

    result = service.translate(
        ModelSpec(
            problem_title="One compartment",
            dataset_path=str(dataset_path),
            pk_code="CL = THETA(1) * EXP(ETA(1))",
            error_code="Y = F * (1 + EPS(1))",
            theta_rows=[{"init": 1.0, "lower": 0.0, "upper": 10.0, "label": "CL"}],
            omega_values=[[0.3]],
            sigma_values=[[0.1]],
        )
    )

    assert result.ok is True
    assert result.builder is not None
    assert result.theta_count == 1
    assert result.eta_count == 1
    assert result.eps_count == 1
    assert result.builder._data_path == str(dataset_path)
    assert result.builder._theta_specs[0].label == "CL"


def test_translate_builder_reports_validation_errors(tmp_path: Path) -> None:
    service = ModelTranslationService()

    result = service.translate_builder(
        ModelSpec(
            problem_title="",
            dataset_path=str(tmp_path / "missing.csv"),
            pk_code="",
            error_code="",
            theta_rows=[{"init": 5.0, "lower": 10.0, "upper": 1.0}],
            omega_values=[[0.3, 0.0]],
            sigma_values=[],
        )
    )

    messages = [issue.message for issue in result.validation.issues]

    assert result.ok is False
    assert result.builder is None
    assert any("Problem title is required" in message for message in messages)
    assert any("Dataset path does not exist" in message for message in messages)
    assert any("$PK code is required" in message for message in messages)
    assert any("$ERROR code is required" in message for message in messages)
    assert any("Invalid THETA row 1" in message for message in messages)
    assert any("OMEGA must be a square matrix" in message for message in messages)
    assert any("SIGMA values are required" in message for message in messages)


def test_translate_control_stream_success(tmp_path: Path) -> None:
    dataset_path = tmp_path / "theo.csv"
    _write_dataset(dataset_path)
    service = ModelTranslationService()

    result = service.translate_control_stream(
        ModelSpec(
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
    )

    assert result.ok is True
    assert result.control_stream is not None
    assert result.record_count >= 8
    assert result.problem_title == "Demo"
    assert result.dataset_path == str(dataset_path)
    assert result.theta_count == 1
    assert result.eta_count == 1
    assert result.eps_count == 1
    assert result.estimation_method == "FOCE"


def test_translate_control_stream_reports_parse_error() -> None:
    service = ModelTranslationService()

    result = service.translate_control_stream(
        ModelSpec(
            mode=ModelSpecMode.CONTROL_STREAM,
            control_stream_text="""$PROBLEM Broken
$THETA (0, 1.0, 10
""",
        )
    )

    assert result.ok is False
    assert result.control_stream is None
    assert result.validation.issues
    assert "Unclosed '(' in $THETA" in result.validation.issues[0].message
