"""GUI-facing dataset metadata for project state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(slots=True)
class DatasetAsset:
    """Serializable summary of an imported dataset."""

    source_path: str | None = None
    display_name: str = ""
    separator: str = ","
    treat_as_whitespace: bool = False
    ignore_char: str | None = None
    input_columns: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    subject_count: int | None = None
    observation_count: int | None = None
    preview_rows: list[dict[str, object]] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "display_name": self.display_name,
            "separator": self.separator,
            "treat_as_whitespace": self.treat_as_whitespace,
            "ignore_char": self.ignore_char,
            "input_columns": list(self.input_columns),
            "columns": list(self.columns),
            "row_count": self.row_count,
            "subject_count": self.subject_count,
            "observation_count": self.observation_count,
            "preview_rows": [dict(row) for row in self.preview_rows],
            "validation_errors": list(self.validation_errors),
            "validation_warnings": list(self.validation_warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DatasetAsset:
        return cls(
            source_path=str(payload["source_path"]) if payload.get("source_path") else None,
            display_name=str(payload.get("display_name", "")),
            separator=str(payload.get("separator", ",")),
            treat_as_whitespace=bool(payload.get("treat_as_whitespace", False)),
            ignore_char=str(payload["ignore_char"]) if payload.get("ignore_char") else None,
            input_columns=[str(value) for value in payload.get("input_columns", [])],
            columns=[str(value) for value in payload.get("columns", [])],
            row_count=int(payload.get("row_count", 0)),
            subject_count=(
                int(payload["subject_count"]) if payload.get("subject_count") is not None else None
            ),
            observation_count=(
                int(payload["observation_count"])
                if payload.get("observation_count") is not None
                else None
            ),
            preview_rows=[dict(row) for row in payload.get("preview_rows", [])],
            validation_errors=[str(value) for value in payload.get("validation_errors", [])],
            validation_warnings=[str(value) for value in payload.get("validation_warnings", [])],
        )
