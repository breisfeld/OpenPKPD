"""Translate GUI-side model specs into engine-facing objects and validation results."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import inf
from pathlib import Path
from typing import Any

from openpkpd.api.model_builder import ModelBuilder
from openpkpd.model.parameters import ThetaSpec
from openpkpd.parser.control_stream import ControlStream
from openpkpd.utils.errors import ParseError
from openpkpd_gui.domain.model_spec import ModelSpec, ModelSpecMode
from openpkpd_gui.services.validation_service import ValidationResult


@dataclass(slots=True)
class ModelTranslationResult:
    """Normalized outcome for translating a GUI model spec."""

    mode: ModelSpecMode
    validation: ValidationResult = field(default_factory=ValidationResult)
    builder: ModelBuilder | None = None
    control_stream: ControlStream | None = None
    problem_title: str = ""
    dataset_path: str | None = None
    estimation_method: str | None = None
    theta_count: int = 0
    eta_count: int = 0
    eps_count: int = 0
    record_count: int = 0
    covariance_enabled: bool = False

    @property
    def ok(self) -> bool:
        return self.validation.ok and (self.builder is not None or self.control_stream is not None)


class ModelTranslationService:
    """Validate and translate GUI model specs without launching estimation."""

    def translate(self, model_spec: ModelSpec) -> ModelTranslationResult:
        if model_spec.mode == ModelSpecMode.CONTROL_STREAM:
            return self.translate_control_stream(model_spec)
        return self.translate_builder(model_spec)

    def translate_builder(self, model_spec: ModelSpec) -> ModelTranslationResult:
        result = ModelTranslationResult(
            mode=model_spec.mode,
            problem_title=model_spec.problem_title.strip(),
            dataset_path=(model_spec.dataset_path or "").strip() or None,
            estimation_method=model_spec.estimation.method,
            covariance_enabled=model_spec.covariance.enabled,
        )

        if not result.problem_title:
            result.validation.add_error("Problem title is required.", field_name="problem_title")
        dataset_path = self._validate_dataset_path(
            result.validation,
            result.dataset_path,
            required=True,
        )
        if not model_spec.pk_code.strip():
            result.validation.add_error(
                "$PK code is required in builder mode.", field_name="pk_code"
            )
        if not model_spec.error_code.strip():
            result.validation.add_error(
                "$ERROR code is required in builder mode.", field_name="error_code"
            )

        theta_specs = self._translate_theta_rows(model_spec.theta_rows, result.validation)
        omega_matrix = self._translate_square_matrix(
            model_spec.omega_values,
            field_name="omega_values",
            label="OMEGA",
            validation=result.validation,
        )
        sigma_matrix = self._translate_square_matrix(
            model_spec.sigma_values,
            field_name="sigma_values",
            label="SIGMA",
            validation=result.validation,
        )

        result.theta_count = len(theta_specs)
        result.eta_count = len(omega_matrix)
        result.eps_count = len(sigma_matrix)

        # H-12: verify that the maximum ETA(n) index used in $PK does not
        # exceed the OMEGA matrix dimension.  A mismatch means the model will
        # silently read out-of-bounds ETAs at run-time.
        eta_indices = [
            int(m)
            for m in re.findall(r"\bETA\s*\(\s*(\d+)\s*\)", model_spec.pk_code)
        ]
        if eta_indices:
            max_eta = max(eta_indices)
            if max_eta > result.eta_count:
                result.validation.add_error(
                    f"$PK references ETA({max_eta}) but OMEGA is "
                    f"{result.eta_count}\u00d7{result.eta_count} "
                    f"(\u03a9 dimension mismatch: add "
                    f"{max_eta - result.eta_count} OMEGA row(s) or remove "
                    f"the extra ETA call(s)).",
                    field_name="pk_code",
                )

        if not result.validation.ok or dataset_path is None:
            return result

        builder = (
            ModelBuilder()
            .problem(result.problem_title)
            .data(dataset_path)
            .subroutines(advan=model_spec.advan, trans=model_spec.trans)
            .pk(model_spec.pk_code)
            .error(model_spec.error_code)
            .theta(theta_specs)
            .omega(omega_matrix)
            .sigma(sigma_matrix)
            .estimation(method=model_spec.estimation.method, **dict(model_spec.estimation.options))
        )
        if model_spec.des_code.strip():
            builder = builder.des(model_spec.des_code)
        if model_spec.covariance.enabled:
            builder = builder.covariance(
                matrix=model_spec.covariance.matrix,
                **dict(model_spec.covariance.options),
            )
        result.builder = builder
        return result

    def translate_control_stream(self, model_spec: ModelSpec) -> ModelTranslationResult:
        override_dataset_path = (model_spec.dataset_path or "").strip() or None
        result = ModelTranslationResult(
            mode=model_spec.mode,
            problem_title=model_spec.problem_title.strip(),
            dataset_path=override_dataset_path,
            covariance_enabled=False,
        )
        control_stream_text = model_spec.control_stream_text.strip()
        if not control_stream_text:
            result.validation.add_error(
                "Control stream text is required.",
                field_name="control_stream_text",
            )
            return result

        try:
            control_stream = ControlStream.from_string(control_stream_text)
        except ParseError as exc:
            # M-13: surface line number and context snippet as a clean
            # single-line message so the GUI validation list displays it well.
            base_msg = str(exc).split("\n")[0].strip()  # "msg (line N)"
            ctx_snippet = exc.context.strip() if exc.context else ""
            full_msg = f"{base_msg} \u2014 {ctx_snippet}" if ctx_snippet else base_msg
            result.validation.add_error(full_msg, field_name="control_stream_text")
            return result

        result.control_stream = control_stream
        result.record_count = len(control_stream.records)
        result.problem_title = (
            control_stream.problem.title.strip()
            if control_stream.problem is not None and control_stream.problem.title.strip()
            else result.problem_title
        )
        if control_stream.data is None:
            result.validation.add_error(
                "Control stream is missing a $DATA record.",
                field_name="control_stream_text",
            )
        elif override_dataset_path is None:
            result.dataset_path = control_stream.data.filename or None

        result.theta_count = sum(len(record.specs) for record in control_stream.theta_records)
        result.eta_count = sum(
            spec.block_size
            for record in control_stream.omega_records
            for spec in record.specs
            if not spec.same
        )
        result.eps_count = sum(
            spec.block_size for record in control_stream.sigma_records for spec in record.specs
        )
        if control_stream.estimation_records:
            result.estimation_method = str(control_stream.estimation_records[0].method)
        result.covariance_enabled = control_stream.covariance is not None

        if result.theta_count == 0:
            result.validation.add_error(
                "Control stream is missing $THETA parameters.",
                field_name="control_stream_text",
            )
        if override_dataset_path is not None and not Path(override_dataset_path).exists():
            result.validation.add_warning(
                "Dataset path from $DATA does not exist on this machine. "
                "Use the dataset loaded in the Data workflow for fitting.",
                field_name="dataset_path",
            )
        return result

    @staticmethod
    def _validate_dataset_path(
        validation: ValidationResult,
        dataset_path: str | None,
        *,
        required: bool,
    ) -> str | None:
        cleaned_path = (dataset_path or "").strip()
        if not cleaned_path:
            if required:
                validation.add_error("Dataset path is required.", field_name="dataset_path")
            return None
        path = Path(cleaned_path)
        if not path.exists():
            validation.add_error("Dataset path does not exist.", field_name="dataset_path")
            return None
        if path.is_dir():
            validation.add_error("Dataset path must point to a file.", field_name="dataset_path")
            return None
        return str(path)

    def _translate_theta_rows(
        self,
        theta_rows: list[Mapping[str, object]],
        validation: ValidationResult,
    ) -> list[ThetaSpec]:
        if not theta_rows:
            validation.add_error("At least one THETA row is required.", field_name="theta_rows")
            return []

        theta_specs: list[ThetaSpec] = []
        for index, row in enumerate(theta_rows, start=1):
            field_name = f"theta_rows[{index}]"
            try:
                init = self._required_float(row, key="init")
                lower = self._optional_float(row.get("lower"), default=-inf)
                upper = self._optional_float(row.get("upper"), default=inf)
                theta_specs.append(
                    ThetaSpec(
                        init=init,
                        lower=lower,
                        upper=upper,
                        fixed=self._coerce_bool(row.get("fixed", False)),
                        label=self._optional_text(row.get("label")),
                    )
                )
            except (TypeError, ValueError) as exc:
                validation.add_error(f"Invalid THETA row {index}: {exc}", field_name=field_name)
        return theta_specs

    def _translate_square_matrix(
        self,
        values: list[list[float]],
        *,
        field_name: str,
        label: str,
        validation: ValidationResult,
    ) -> list[list[float]]:
        if not values:
            validation.add_error(f"{label} values are required.", field_name=field_name)
            return []

        matrix: list[list[float]] = []
        expected_width: int | None = None
        for row_index, row in enumerate(values, start=1):
            if not row:
                validation.add_error(
                    f"{label} row {row_index} cannot be empty.",
                    field_name=field_name,
                )
                return []
            try:
                parsed_row = [float(value) for value in row]
            except (TypeError, ValueError) as exc:
                validation.add_error(
                    f"Invalid {label} row {row_index}: {exc}",
                    field_name=field_name,
                )
                return []
            expected_width = len(parsed_row) if expected_width is None else expected_width
            if len(parsed_row) != expected_width:
                validation.add_error(
                    f"{label} rows must all be the same length.",
                    field_name=field_name,
                )
                return []
            matrix.append(parsed_row)

        if len(matrix) != expected_width:
            validation.add_error(
                f"{label} must be a square matrix.",
                field_name=field_name,
            )
            return []
        return matrix

    @staticmethod
    def _required_float(row: Mapping[str, object], *, key: str) -> float:
        value = row.get(key)
        if value is None or str(value).strip() == "":
            raise ValueError(f"{key} is required")
        return float(value)

    @staticmethod
    def _optional_float(value: object, *, default: float) -> float:
        if value is None or str(value).strip() == "":
            return default
        return float(value)

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
