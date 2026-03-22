"""Shared validation result models for GUI workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ValidationSeverity(StrEnum):
    """Severity for a validation issue."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class ValidationIssue:
    """Single validation finding."""

    severity: ValidationSeverity
    message: str
    field_name: str | None = None
    target_workflow: str | None = None
    target_widget: str | None = None


@dataclass(slots=True)
class ValidationResult:
    """Collection of validation issues with convenience helpers."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)

    def add_error(
        self,
        message: str,
        field_name: str | None = None,
        *,
        target_workflow: str | None = None,
        target_widget: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                ValidationSeverity.ERROR,
                message,
                field_name,
                target_workflow,
                target_widget,
            )
        )

    def add_warning(
        self,
        message: str,
        field_name: str | None = None,
        *,
        target_workflow: str | None = None,
        target_widget: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                ValidationSeverity.WARNING,
                message,
                field_name,
                target_workflow,
                target_widget,
            )
        )
