"""Convenience service for querying manifest-backed examples."""

from __future__ import annotations

from pathlib import Path

from openpkpd.examples.catalog_loader import load_catalog
from openpkpd.examples.catalog_models import ExampleEntry


class ExampleCatalogService:
    """Small facade that exposes GUI-friendly catalog queries."""

    def __init__(
        self,
        *,
        catalog_root: Path,
        shared_data_root: Path | None = None,
    ) -> None:
        self.catalog_root = Path(catalog_root)
        self.shared_data_root = Path(shared_data_root) if shared_data_root is not None else None

    def list_dataset_examples(self) -> list[ExampleEntry]:
        """Return catalog entries that provide a loadable dataset."""
        return [
            entry
            for entry in load_catalog(self.catalog_root, shared_data_root=self.shared_data_root)
            if entry.manifest.gui.load_dataset and entry.dataset_path is not None
        ]

    def list_control_stream_examples(self) -> list[ExampleEntry]:
        """Return catalog entries that provide a loadable control stream."""
        return [
            entry
            for entry in load_catalog(self.catalog_root, shared_data_root=self.shared_data_root)
            if entry.manifest.gui.load_control_stream and entry.control_stream_path is not None
        ]

    def get_example_by_id(self, example_id: str) -> ExampleEntry | None:
        """Return one catalog entry by its stable manifest ID."""
        for entry in load_catalog(self.catalog_root, shared_data_root=self.shared_data_root):
            if entry.manifest.id == example_id:
                return entry
        return None
