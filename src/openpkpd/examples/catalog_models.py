"""Typed models for the manifest-backed examples catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Category = Literal[
    "pk",
    "pd",
    "pkpd",
    "nca",
    "workflow",
    "diagnostics",
    "simulation",
    "interoperability",
]
PrimaryMode = Literal["control_stream", "dataset", "script", "nca"]
Route = Literal["oral", "iv", "infusion", "extravascular", "mixed"] | None
Difficulty = Literal["starter", "intermediate", "advanced"]
IssueSeverity = Literal["error", "warning"]


@dataclass(slots=True, frozen=True)
class ExampleFiles:
    """Relative file references declared by an example bundle manifest."""

    dataset: str | None = None
    control_stream: str | None = None
    script: str | None = None
    readme: str | None = "README.md"
    preview_image: str | None = None


@dataclass(slots=True, frozen=True)
class ExampleGUI:
    """GUI-facing discovery flags for one example bundle."""

    load_dataset: bool = False
    load_control_stream: bool = False
    featured: bool = False


@dataclass(slots=True, frozen=True)
class ExampleSource:
    """Optional provenance metadata for an example bundle."""

    kind: str = "internal"
    url: str | None = None
    license: str | None = None


@dataclass(slots=True, frozen=True)
class ExampleManifest:
    """Parsed YAML manifest content for one example bundle."""

    manifest_version: int
    id: str
    title: str
    description: str
    category: Category
    primary_mode: PrimaryMode
    route: Route = None
    difficulty: Difficulty = "starter"
    tags: tuple[str, ...] = ()
    sort_order: int = 100
    files: ExampleFiles = field(default_factory=ExampleFiles)
    gui: ExampleGUI = field(default_factory=ExampleGUI)
    source: ExampleSource = field(default_factory=ExampleSource)


@dataclass(slots=True, frozen=True)
class ExampleEntry:
    """Resolved catalog entry with validated filesystem paths."""

    manifest: ExampleManifest
    bundle_dir: Path
    manifest_path: Path
    dataset_path: Path | None = None
    control_stream_path: Path | None = None
    script_path: Path | None = None
    readme_path: Path | None = None
    preview_image_path: Path | None = None


@dataclass(slots=True, frozen=True)
class ManifestIssue:
    """Validation issue emitted while parsing or resolving one manifest."""

    severity: IssueSeverity
    field: str | None
    message: str


@dataclass(slots=True, frozen=True)
class ManifestValidationResult:
    """Outcome of loading one manifest."""

    entry: ExampleEntry | None
    issues: tuple[ManifestIssue, ...]

    @property
    def ok(self) -> bool:
        """Whether the manifest has no validation errors."""
        return not any(issue.severity == "error" for issue in self.issues)
