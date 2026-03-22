"""Dataset loading and summary helpers for the Data workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from openpkpd.data.columns import REQUIRED_COLUMNS
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.examples.catalog_service import ExampleCatalogService
from openpkpd.parser.control_stream import ControlStream
from openpkpd.utils.errors import DataError, ParseError
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.services.validation_service import ValidationResult, ValidationSeverity

OPTIONAL_NONMEM_COLUMNS = ("AMT", "RATE", "EVID", "MDV", "CMT", "ADDL", "II", "SS")


@dataclass(slots=True, frozen=True)
class ExampleDataset:
    """A dataset example exposed through the curated example catalog."""

    key: str
    label: str
    description: str
    dataset_path: str
    source_kind: str = "catalog_manifest"
    category: str = ""
    route: str | None = None
    difficulty: str = "starter"
    tags: tuple[str, ...] = ()
    manifest_path: str = ""
    readme_path: str | None = None
    source_license: str | None = None
    source_url: str | None = None


@dataclass(slots=True)
class DatasetImportOptions:
    """User-configurable import options for dataset loading."""

    separator: str = ","
    treat_as_whitespace: bool = False
    ignore_char: str | None = None
    preview_rows: int = 5

    @property
    def effective_separator(self) -> str:
        return r"\s+" if self.treat_as_whitespace else self.separator

    @property
    def normalized_ignore_char(self) -> str | None:
        cleaned = (self.ignore_char or "").strip()
        return cleaned or None


@dataclass(slots=True)
class DatasetLoadResult:
    """Normalized outcome for a GUI dataset import attempt."""

    dataset_asset: DatasetAsset | None = None
    validation: ValidationResult = field(default_factory=ValidationResult)

    @property
    def ok(self) -> bool:
        return self.dataset_asset is not None and self.validation.ok


class DatasetService:
    """Load datasets through the core engine and produce GUI summaries."""

    def __init__(
        self,
        *,
        catalog_root: str | Path | None = None,
        shared_data_root: str | Path | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._catalog_service = ExampleCatalogService(
            catalog_root=Path(catalog_root)
            if catalog_root is not None
            else repo_root / "examples" / "catalog",
            shared_data_root=(
                Path(shared_data_root)
                if shared_data_root is not None
                else repo_root / "examples" / "shared_data"
            ),
        )

    def load_csv(
        self,
        path: str,
        *,
        options: DatasetImportOptions | None = None,
        input_columns: list[str] | None = None,
    ) -> DatasetLoadResult:
        result = DatasetLoadResult()
        options = options or DatasetImportOptions()
        cleaned_path = path.strip()
        if not cleaned_path:
            result.validation.add_error("Dataset path is required.", field_name="source_path")
            return result
        if not options.treat_as_whitespace and not options.separator:
            result.validation.add_error(
                "Separator is required unless whitespace mode is enabled.",
                field_name="separator",
            )
        if options.normalized_ignore_char is not None and len(options.normalized_ignore_char) != 1:
            result.validation.add_error(
                "Ignore character must be exactly one character.",
                field_name="ignore_char",
            )
        if not result.validation.ok:
            return result

        try:
            dataset = NONMEMDataset.from_csv(
                cleaned_path,
                ignore_char=options.normalized_ignore_char,
                sep=options.effective_separator,
                input_columns=input_columns,
            )
        except DataError as exc:
            result.validation.add_error(str(exc), field_name="source_path")
            return result

        result.dataset_asset = self._build_dataset_asset(
            dataset, options=options, input_columns=input_columns
        )
        self._add_import_warnings(result.validation, dataset, options)
        result.dataset_asset.validation_errors = [
            issue.message
            for issue in result.validation.issues
            if issue.severity == ValidationSeverity.ERROR
        ]
        result.dataset_asset.validation_warnings = [
            issue.message
            for issue in result.validation.issues
            if issue.severity == ValidationSeverity.WARNING
        ]
        return result

    def list_examples(self) -> list[ExampleDataset]:
        """Return example datasets discovered from the curated manifest catalog."""
        examples = [
            self._entry_to_example(entry) for entry in self._catalog_service.list_dataset_examples()
        ]
        return sorted(examples, key=lambda item: (item.label.casefold(), item.key.casefold()))

    def load_example(
        self,
        example_key: str,
        *,
        options: DatasetImportOptions | None = None,
    ) -> DatasetLoadResult:
        """Load one of the curated example datasets by its stable manifest ID."""
        result = DatasetLoadResult()
        entry = self._catalog_service.get_example_by_id(example_key)
        if entry is None or entry.dataset_path is None:
            result.validation.add_error(
                "Selected example dataset could not be found.",
                field_name="example_dataset",
            )
            return result

        example_options = options or DatasetImportOptions()
        input_columns: list[str] | None = None
        if entry.control_stream_path is not None and entry.control_stream_path.exists():
            try:
                control_stream = ControlStream.from_file(str(entry.control_stream_path))
            except (OSError, ParseError):
                control_stream = None
            if control_stream is not None:
                if control_stream.data is not None:
                    example_options = replace(
                        example_options, ignore_char=control_stream.data.ignore_char
                    )
                if control_stream.input is not None:
                    input_columns = list(control_stream.input.columns)

        loaded = self.load_csv(
            str(entry.dataset_path),
            options=example_options,
            input_columns=input_columns,
        )
        if loaded.dataset_asset is not None:
            loaded.dataset_asset.display_name = entry.manifest.title
        return loaded

    @staticmethod
    def required_columns() -> list[str]:
        """Return the minimum columns required for a valid import."""
        return sorted(REQUIRED_COLUMNS)

    @staticmethod
    def validation_from_asset(dataset_asset: DatasetAsset | None) -> ValidationResult:
        """Rebuild a ValidationResult from serialized dataset metadata."""
        result = ValidationResult()
        if dataset_asset is None:
            return result
        for message in dataset_asset.validation_warnings:
            result.add_warning(message)
        for message in dataset_asset.validation_errors:
            result.add_error(message)
        return result

    @staticmethod
    def options_from_asset(dataset_asset: DatasetAsset | None) -> DatasetImportOptions:
        """Rebuild import options from serialized dataset metadata."""
        if dataset_asset is None:
            return DatasetImportOptions()
        return DatasetImportOptions(
            separator=dataset_asset.separator,
            treat_as_whitespace=dataset_asset.treat_as_whitespace,
            ignore_char=dataset_asset.ignore_char,
        )

    def _build_dataset_asset(
        self,
        dataset: NONMEMDataset,
        options: DatasetImportOptions,
        *,
        input_columns: list[str] | None = None,
    ) -> DatasetAsset:
        preview_frame = dataset.df.head(max(1, options.preview_rows)).copy()
        preview_frame = preview_frame.astype(object).where(preview_frame.notna(), None)
        display_name = Path(dataset.source_path).name if dataset.source_path else "Imported dataset"
        return DatasetAsset(
            source_path=dataset.source_path,
            display_name=display_name,
            separator=options.separator,
            treat_as_whitespace=options.treat_as_whitespace,
            ignore_char=options.normalized_ignore_char,
            input_columns=list(input_columns or []),
            columns=[str(column) for column in dataset.df.columns],
            row_count=len(dataset.df),
            subject_count=dataset.n_subjects(),
            observation_count=dataset.n_observations(),
            preview_rows=[dict(row) for row in preview_frame.to_dict(orient="records")],
        )

    def _add_import_warnings(
        self,
        validation: ValidationResult,
        dataset: NONMEMDataset,
        options: DatasetImportOptions,
    ) -> None:
        if len(dataset.df) == 0:
            validation.add_warning("Dataset contains no rows.", field_name="row_count")
        if dataset.n_observations() == 0:
            validation.add_warning(
                "No observation rows detected (EVID=0 and MDV=0).",
                field_name="observation_count",
            )
        if not dataset.source_path:
            return

        source_columns = self._read_source_columns(dataset.source_path, options)
        defaulted_columns = [
            column
            for column in OPTIONAL_NONMEM_COLUMNS
            if column in dataset.df.columns and column not in source_columns
        ]
        if defaulted_columns:
            validation.add_warning(
                "Missing optional NONMEM columns were defaulted during import: "
                + ", ".join(defaulted_columns),
                field_name="columns",
            )

    @staticmethod
    def _read_source_columns(path: str, options: DatasetImportOptions) -> set[str]:
        if options.treat_as_whitespace:
            header = pd.read_csv(path, sep=r"\s+", engine="python", nrows=0)
        else:
            header = pd.read_csv(path, sep=options.separator, nrows=0)
        return {str(column).upper() for column in header.columns}

    @staticmethod
    def _entry_to_example(entry) -> ExampleDataset:
        assert entry.dataset_path is not None
        return ExampleDataset(
            key=entry.manifest.id,
            label=entry.manifest.title,
            description=entry.manifest.description,
            dataset_path=str(entry.dataset_path),
            category=entry.manifest.category,
            route=entry.manifest.route,
            difficulty=entry.manifest.difficulty,
            tags=entry.manifest.tags,
            manifest_path=str(entry.manifest_path),
            readme_path=str(entry.readme_path) if entry.readme_path is not None else None,
            source_license=entry.manifest.source.license,
            source_url=entry.manifest.source.url,
        )
