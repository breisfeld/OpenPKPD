"""Discovery, parsing, and validation for manifest-backed examples."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

from openpkpd.examples.catalog_models import (
    Category,
    Difficulty,
    ExampleEntry,
    ExampleFiles,
    ExampleGUI,
    ExampleManifest,
    ExampleSource,
    ManifestIssue,
    ManifestValidationResult,
    PrimaryMode,
    Route,
)

_REQUIRED_FIELDS = (
    "manifest_version",
    "id",
    "title",
    "description",
    "category",
    "primary_mode",
    "difficulty",
    "files",
    "gui",
)
_ALLOWED_CATEGORIES: set[Category] = {
    "pk",
    "pd",
    "pkpd",
    "nca",
    "workflow",
    "diagnostics",
    "simulation",
    "interoperability",
}
_ALLOWED_PRIMARY_MODES: set[PrimaryMode] = {"control_stream", "dataset", "script", "nca"}
_ALLOWED_DIFFICULTIES: set[Difficulty] = {"starter", "intermediate", "advanced"}
_ALLOWED_ROUTES: set[Route] = {"oral", "iv", "infusion", "extravascular", "mixed", None}


def discover_example_manifests(catalog_root: Path) -> list[Path]:
    """Return all manifest.yaml files under the curated catalog tree."""
    if not catalog_root.exists():
        return []
    return sorted(path for path in catalog_root.rglob("manifest.yaml") if path.is_file())


def load_manifest(
    path: Path,
    *,
    shared_data_root: Path | None = None,
) -> ManifestValidationResult:
    """Load, validate, and resolve one manifest.yaml file."""
    issues: list[ManifestIssue] = []
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ManifestValidationResult(
            entry=None,
            issues=(ManifestIssue("error", "manifest", f"Could not read manifest: {exc}"),),
        )

    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        return ManifestValidationResult(
            entry=None,
            issues=(ManifestIssue("error", "manifest", f"Invalid YAML: {exc}"),),
        )

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return ManifestValidationResult(
            entry=None,
            issues=(ManifestIssue("error", "manifest", "Manifest root must be a mapping/object."),),
        )

    manifest = _parse_manifest(raw, issues)
    entry = None
    if manifest is not None:
        entry = _resolve_entry(
            bundle_dir=path.parent,
            manifest_path=path,
            manifest=manifest,
            issues=issues,
            shared_data_root=shared_data_root,
        )
    return ManifestValidationResult(entry=entry, issues=tuple(issues))


def load_catalog_with_issues(
    catalog_root: Path,
    *,
    shared_data_root: Path | None = None,
) -> list[ManifestValidationResult]:
    """Load every manifest under the catalog root, retaining validation issues."""
    results = [
        load_manifest(path, shared_data_root=shared_data_root)
        for path in discover_example_manifests(catalog_root)
    ]
    _annotate_duplicate_ids(results)
    return results


def load_catalog(
    catalog_root: Path,
    *,
    shared_data_root: Path | None = None,
) -> list[ExampleEntry]:
    """Load the valid resolved catalog entries, sorted for GUI presentation."""
    results = load_catalog_with_issues(catalog_root, shared_data_root=shared_data_root)
    _raise_if_duplicate_ids(results)
    entries = [result.entry for result in results if result.ok and result.entry is not None]
    return sorted(
        entries, key=lambda entry: (entry.manifest.sort_order, entry.manifest.title.casefold())
    )


def _parse_manifest(raw: dict[str, Any], issues: list[ManifestIssue]) -> ExampleManifest | None:
    _validate_required(raw, issues)
    files_raw = _mapping(raw.get("files"), "files", issues)
    gui_raw = _mapping(raw.get("gui"), "gui", issues)
    source_raw = _mapping(raw.get("source", {}), "source", issues)
    if _has_errors(issues):
        return None

    manifest_version = _int_value(raw.get("manifest_version"), "manifest_version", issues)
    identifier = _nonempty_str(raw.get("id"), "id", issues)
    title = _nonempty_str(raw.get("title"), "title", issues)
    description = _nonempty_str(raw.get("description"), "description", issues)
    category = _choice(raw.get("category"), "category", _ALLOWED_CATEGORIES, issues)
    primary_mode = _choice(raw.get("primary_mode"), "primary_mode", _ALLOWED_PRIMARY_MODES, issues)
    route = _choice(raw.get("route"), "route", _ALLOWED_ROUTES, issues, allow_none=True)
    difficulty = _choice(raw.get("difficulty"), "difficulty", _ALLOWED_DIFFICULTIES, issues)
    sort_order = _int_value(raw.get("sort_order", 100), "sort_order", issues)
    tags = _tags(raw.get("tags", ()), issues)

    files = _build_dataclass(ExampleFiles, files_raw or {}, "files", issues)
    gui = _build_dataclass(ExampleGUI, gui_raw or {}, "gui", issues)
    source = _build_dataclass(ExampleSource, source_raw or {}, "source", issues)
    if _has_errors(issues):
        return None

    assert manifest_version is not None
    assert identifier is not None
    assert title is not None
    assert description is not None
    assert category is not None
    assert primary_mode is not None
    assert difficulty is not None
    assert sort_order is not None
    assert files is not None
    assert gui is not None
    assert source is not None

    return ExampleManifest(
        manifest_version=manifest_version,
        id=identifier,
        title=title,
        description=description,
        category=category,
        primary_mode=primary_mode,
        route=route,
        difficulty=difficulty,
        tags=tags,
        sort_order=sort_order,
        files=files,
        gui=gui,
        source=source,
    )


def _resolve_entry(
    *,
    bundle_dir: Path,
    manifest_path: Path,
    manifest: ExampleManifest,
    issues: list[ManifestIssue],
    shared_data_root: Path | None,
) -> ExampleEntry:
    _validate_cross_fields(manifest, issues)
    allowed_roots = [bundle_dir.resolve()]
    if shared_data_root is not None:
        allowed_roots.append(shared_data_root.resolve())

    dataset_path = _resolve_file(
        bundle_dir,
        manifest.files.dataset,
        issues,
        "files.dataset",
        allowed_roots,
        missing_severity="error",
    )
    control_stream_path = _resolve_file(
        bundle_dir,
        manifest.files.control_stream,
        issues,
        "files.control_stream",
        allowed_roots,
        missing_severity="error",
    )
    script_path = _resolve_file(
        bundle_dir,
        manifest.files.script,
        issues,
        "files.script",
        allowed_roots,
        missing_severity="error",
    )
    readme_path = _resolve_file(
        bundle_dir,
        manifest.files.readme,
        issues,
        "files.readme",
        allowed_roots,
        missing_severity="warning",
    )
    preview_image_path = _resolve_file(
        bundle_dir,
        manifest.files.preview_image,
        issues,
        "files.preview_image",
        allowed_roots,
        missing_severity="warning",
    )
    return ExampleEntry(
        manifest=manifest,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        dataset_path=dataset_path,
        control_stream_path=control_stream_path,
        script_path=script_path,
        readme_path=readme_path,
        preview_image_path=preview_image_path,
    )


def _validate_required(raw: dict[str, Any], issues: list[ManifestIssue]) -> None:
    for key in _REQUIRED_FIELDS:
        if key not in raw:
            issues.append(ManifestIssue("error", key, "Missing required field."))


def _validate_cross_fields(manifest: ExampleManifest, issues: list[ManifestIssue]) -> None:
    if manifest.gui.load_dataset and not manifest.files.dataset:
        issues.append(
            ManifestIssue("error", "gui.load_dataset", "Dataset examples require files.dataset.")
        )
    if manifest.gui.load_control_stream and not manifest.files.control_stream:
        issues.append(
            ManifestIssue(
                "error",
                "gui.load_control_stream",
                "Control-stream examples require files.control_stream.",
            )
        )


def _resolve_file(
    bundle_dir: Path,
    rel_path: str | None,
    issues: list[ManifestIssue],
    field: str,
    allowed_roots: list[Path],
    *,
    missing_severity: Literal["error", "warning"],
) -> Path | None:
    if rel_path is None:
        return None
    candidate = Path(rel_path)
    if candidate.is_absolute():
        issues.append(ManifestIssue("error", field, "Absolute paths are not allowed."))
        return None
    resolved = (bundle_dir / candidate).resolve()
    if not any(_is_within(resolved, root) for root in allowed_roots):
        issues.append(ManifestIssue("error", field, "Path escapes allowed example roots."))
        return None
    if not resolved.exists():
        issues.append(ManifestIssue(missing_severity, field, "Referenced file does not exist."))
        return None
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _mapping(value: Any, field_name: str, issues: list[ManifestIssue]) -> dict[str, Any] | None:
    if value is None:
        return {}
    if not isinstance(value, dict):
        issues.append(ManifestIssue("error", field_name, "Expected a mapping/object."))
        return None
    return value


def _nonempty_str(value: Any, field_name: str, issues: list[ManifestIssue]) -> str | None:
    if not isinstance(value, str):
        issues.append(ManifestIssue("error", field_name, "Expected a string."))
        return None
    stripped = value.strip()
    if not stripped:
        issues.append(ManifestIssue("error", field_name, "Value must not be empty."))
        return None
    return stripped


def _int_value(value: Any, field_name: str, issues: list[ManifestIssue]) -> int | None:
    if isinstance(value, bool):
        issues.append(ManifestIssue("error", field_name, "Expected an integer."))
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        issues.append(ManifestIssue("error", field_name, "Expected an integer."))
        return None


def _choice(
    value: Any,
    field_name: str,
    allowed: set[Any],
    issues: list[ManifestIssue],
    *,
    allow_none: bool = False,
) -> Any:
    if value is None and allow_none:
        return None
    if value not in allowed:
        allowed_values = ", ".join(sorted(str(item) for item in allowed if item is not None))
        issues.append(
            ManifestIssue("error", field_name, f"Invalid value. Allowed values: {allowed_values}.")
        )
        return None
    return value


def _tags(value: Any, issues: list[ManifestIssue]) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        issues.append(ManifestIssue("error", "tags", "Expected a list of strings."))
        return ()
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            issues.append(ManifestIssue("error", "tags", "All tags must be strings."))
            return ()
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return tuple(normalized)


def _build_dataclass(
    cls: type[Any], raw: dict[str, Any], field_name: str, issues: list[ManifestIssue]
) -> Any:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(key for key in raw if key not in allowed)
    for key in unknown:
        issues.append(ManifestIssue("error", f"{field_name}.{key}", "Unknown field."))

    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in raw:
            continue
        value = raw[item.name]
        if item.type is bool and not isinstance(value, bool):
            issues.append(
                ManifestIssue("error", f"{field_name}.{item.name}", "Expected a boolean value.")
            )
            continue
        if item.type in (str, str | None) and value is not None and not isinstance(value, str):
            issues.append(
                ManifestIssue("error", f"{field_name}.{item.name}", "Expected a string value.")
            )
            continue
        kwargs[item.name] = value
    if _has_errors(issues):
        return None
    return cls(**kwargs)


def _has_errors(issues: list[ManifestIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)


def _annotate_duplicate_ids(results: list[ManifestValidationResult]) -> None:
    seen: dict[str, int] = {}
    duplicates: set[str] = set()
    for idx, result in enumerate(results):
        if result.entry is None:
            continue
        ex_id = result.entry.manifest.id
        if ex_id in seen:
            duplicates.add(ex_id)
        else:
            seen[ex_id] = idx

    if not duplicates:
        return

    for idx, result in enumerate(results):
        if result.entry is None:
            continue
        ex_id = result.entry.manifest.id
        if ex_id not in duplicates:
            continue
        issue = ManifestIssue("error", "id", f"Duplicate example id '{ex_id}'.")
        results[idx] = ManifestValidationResult(
            entry=result.entry,
            issues=result.issues + (issue,),
        )


def _raise_if_duplicate_ids(results: list[ManifestValidationResult]) -> None:
    duplicates = sorted(
        {
            result.entry.manifest.id
            for result in results
            if result.entry is not None
            and any(
                issue.severity == "error"
                and issue.field == "id"
                and "Duplicate example id" in issue.message
                for issue in result.issues
            )
        }
    )
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate example id(s): {joined}")
