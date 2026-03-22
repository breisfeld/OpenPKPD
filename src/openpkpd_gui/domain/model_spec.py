"""Serializable model specification for builder and control-stream modes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum


class ModelSpecMode(StrEnum):
    """Authoring mode for the GUI model workspace."""

    BUILDER = "builder"
    CONTROL_STREAM = "control_stream"


@dataclass(slots=True)
class EstimationConfig:
    """Estimation method and related user options."""

    method: str = "FOCE"
    options: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class CovarianceConfig:
    """Covariance step settings."""

    enabled: bool = False
    matrix: str = "SR"
    options: dict[str, object] = field(default_factory=dict)


def _serialize_model_spec_value(value: object, *, seen: set[int] | None = None) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _serialize_model_spec_value(value.value, seen=seen)
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return "<recursive>"
    if isinstance(value, dict):
        seen.add(value_id)
        try:
            return {
                str(key): _serialize_model_spec_value(item, seen=seen)
                for key, item in value.items()
            }
        finally:
            seen.discard(value_id)
    if isinstance(value, (list, tuple)):
        seen.add(value_id)
        try:
            return [_serialize_model_spec_value(item, seen=seen) for item in value]
        finally:
            seen.discard(value_id)
    if is_dataclass(value):
        seen.add(value_id)
        try:
            return {
                field.name: _serialize_model_spec_value(getattr(value, field.name), seen=seen)
                for field in fields(value)
            }
        finally:
            seen.discard(value_id)
    return str(value)


def _serialize_estimation_config(config: EstimationConfig) -> dict[str, object]:
    return {
        "method": str(config.method),
        "options": _serialize_model_spec_value(dict(config.options)),
    }


def _serialize_covariance_config(config: CovarianceConfig) -> dict[str, object]:
    return {
        "enabled": bool(config.enabled),
        "matrix": str(config.matrix),
        "options": _serialize_model_spec_value(dict(config.options)),
    }


@dataclass(slots=True)
class ModelSpec:
    """GUI-side model definition independent of widgets or engine objects."""

    mode: ModelSpecMode = ModelSpecMode.BUILDER
    problem_title: str = ""
    dataset_path: str | None = None
    control_stream_text: str = ""
    advan: int = 2
    trans: int = 2
    pk_code: str = ""
    error_code: str = ""
    des_code: str = ""
    theta_rows: list[dict[str, object]] = field(default_factory=list)
    omega_values: list[list[float]] = field(default_factory=list)
    sigma_values: list[list[float]] = field(default_factory=list)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "problem_title": self.problem_title,
            "dataset_path": self.dataset_path,
            "control_stream_text": self.control_stream_text,
            "advan": self.advan,
            "trans": self.trans,
            "pk_code": self.pk_code,
            "error_code": self.error_code,
            "des_code": self.des_code,
            "theta_rows": [dict(row) for row in self.theta_rows],
            "omega_values": [list(row) for row in self.omega_values],
            "sigma_values": [list(row) for row in self.sigma_values],
            "estimation": _serialize_estimation_config(self.estimation),
            "covariance": _serialize_covariance_config(self.covariance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ModelSpec:
        return cls(
            mode=ModelSpecMode(str(payload.get("mode", ModelSpecMode.BUILDER.value))),
            problem_title=str(payload.get("problem_title", "")),
            dataset_path=str(payload["dataset_path"]) if payload.get("dataset_path") else None,
            control_stream_text=str(payload.get("control_stream_text", "")),
            advan=int(payload.get("advan", 2)),
            trans=int(payload.get("trans", 2)),
            pk_code=str(payload.get("pk_code", "")),
            error_code=str(payload.get("error_code", "")),
            des_code=str(payload.get("des_code", "")),
            theta_rows=[dict(row) for row in payload.get("theta_rows", [])],
            omega_values=[list(row) for row in payload.get("omega_values", [])],
            sigma_values=[list(row) for row in payload.get("sigma_values", [])],
            estimation=EstimationConfig(**dict(payload.get("estimation", {}))),
            covariance=CovarianceConfig(**dict(payload.get("covariance", {}))),
        )
