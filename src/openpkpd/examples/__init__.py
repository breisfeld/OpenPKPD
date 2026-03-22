"""Manifest-backed example catalog helpers."""

from openpkpd.examples.catalog_loader import (
    discover_example_manifests,
    load_catalog,
    load_catalog_with_issues,
    load_manifest,
)
from openpkpd.examples.catalog_models import (
    ExampleEntry,
    ExampleFiles,
    ExampleGUI,
    ExampleManifest,
    ExampleSource,
    ManifestIssue,
    ManifestValidationResult,
)
from openpkpd.examples.catalog_service import ExampleCatalogService

__all__ = [
    "ExampleCatalogService",
    "ExampleEntry",
    "ExampleFiles",
    "ExampleGUI",
    "ExampleManifest",
    "ExampleSource",
    "ManifestIssue",
    "ManifestValidationResult",
    "discover_example_manifests",
    "load_catalog",
    "load_catalog_with_issues",
    "load_manifest",
]
