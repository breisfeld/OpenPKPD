"""Unit tests for dataset import and summary services."""

from __future__ import annotations

from pathlib import Path

from openpkpd_gui.services.data_service import DatasetImportOptions, DatasetService


def test_dataset_service_loads_and_summarizes_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "demo.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n1,1,3.5\n2,0,0\n", encoding="utf-8")

    service = DatasetService()
    result = service.load_csv(
        str(dataset_path),
        options=DatasetImportOptions(preview_rows=2),
    )

    assert result.ok is True
    assert result.dataset_asset is not None
    assert result.dataset_asset.display_name == "demo.csv"
    assert result.dataset_asset.row_count == 3
    assert result.dataset_asset.subject_count == 2
    assert result.dataset_asset.observation_count == 3
    assert result.dataset_asset.columns[:3] == ["ID", "TIME", "DV"]
    assert len(result.dataset_asset.preview_rows) == 2
    assert result.dataset_asset.validation_warnings


def test_dataset_service_reports_validation_errors_for_invalid_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.csv"
    dataset_path.write_text("ID,TIME\n1,0\n", encoding="utf-8")

    result = DatasetService().load_csv(str(dataset_path))

    assert result.ok is False
    assert result.dataset_asset is None
    assert result.validation.ok is False
    assert "required columns" in result.validation.issues[0].message.lower()


def test_dataset_service_supports_whitespace_and_ignore_options(tmp_path: Path) -> None:
    dataset_path = tmp_path / "space.txt"
    dataset_path.write_text(
        "ID TIME DV\n@SKIP 0 0\n1 0 0\n1 1 2.5\n",
        encoding="utf-8",
    )

    result = DatasetService().load_csv(
        str(dataset_path),
        options=DatasetImportOptions(treat_as_whitespace=True, ignore_char="@"),
    )

    assert result.ok is True
    assert result.dataset_asset is not None
    assert result.dataset_asset.treat_as_whitespace is True
    assert result.dataset_asset.ignore_char == "@"
    assert result.dataset_asset.row_count == 2


def test_dataset_service_rejects_invalid_ignore_option(tmp_path: Path) -> None:
    dataset_path = tmp_path / "demo.csv"
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n", encoding="utf-8")

    result = DatasetService().load_csv(
        str(dataset_path),
        options=DatasetImportOptions(ignore_char="##"),
    )

    assert result.ok is False
    assert result.validation.ok is False
    assert "exactly one character" in result.validation.issues[0].message


def _write_catalog_dataset_example(tmp_path: Path) -> tuple[Path, Path]:
    catalog_root = tmp_path / "catalog"
    shared_data_root = tmp_path / "shared_data"
    bundle_dir = catalog_root / "pk" / "oral" / "demo_dataset"
    dataset_path = shared_data_root / "demo" / "demo.csv"
    bundle_dir.mkdir(parents=True)
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("ID,TIME,DV\n1,0,0\n1,1,2.5\n2,0,0\n", encoding="utf-8")
    (bundle_dir / "manifest.yaml").write_text(
        "\n".join(
            [
                "manifest_version: 1",
                "id: demo_dataset",
                "title: Demo dataset",
                "description: Demo curated dataset for GUI tests.",
                "category: pk",
                "primary_mode: nca",
                "route: oral",
                "difficulty: starter",
                "tags: [demo, oral, pk]",
                "files:",
                "  dataset: ../../../../shared_data/demo/demo.csv",
                "  readme: null",
                "  preview_image: null",
                "gui:",
                "  load_dataset: true",
                "  load_control_stream: false",
                "source:",
                "  kind: internal",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return catalog_root, shared_data_root


def _write_catalog_nmdata_example(tmp_path: Path) -> tuple[Path, Path]:
    catalog_root = tmp_path / "catalog"
    shared_data_root = tmp_path / "shared_data"
    bundle_dir = catalog_root / "pk" / "oral" / "nmdata_drop"
    dataset_path = shared_data_root / "nmdata" / "drop.csv"
    bundle_dir.mkdir(parents=True)
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("1,0,999,70,5\n", encoding="utf-8")
    (bundle_dir / "model.ctl").write_text(
        "$PROBLEM Drop demo\n"
        "$DATA ../../../../shared_data/nmdata/drop.csv\n"
        "$INPUT ID TIME SKIP=DROP WT DV\n"
        "$THETA 1\n"
        "$OMEGA 0.1\n"
        "$SIGMA 0.1\n"
        "$PK\nCL=THETA(1)\n"
        "$ERROR\nY=F+EPS(1)\n",
        encoding="utf-8",
    )
    (bundle_dir / "manifest.yaml").write_text(
        "\n".join(
            [
                "manifest_version: 1",
                "id: nmdata_drop",
                "title: NMdata drop example",
                "description: Headerless dataset with INPUT DROP metadata.",
                "category: pk",
                "primary_mode: control_stream",
                "route: oral",
                "difficulty: advanced",
                "tags: [nmdata, drop]",
                "files:",
                "  dataset: ../../../../shared_data/nmdata/drop.csv",
                "  control_stream: model.ctl",
                "gui:",
                "  load_dataset: true",
                "  load_control_stream: true",
                "source:",
                "  kind: imported",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return catalog_root, shared_data_root


def test_dataset_service_lists_manifest_backed_examples(tmp_path: Path) -> None:
    catalog_root, shared_data_root = _write_catalog_dataset_example(tmp_path)

    service = DatasetService(catalog_root=catalog_root, shared_data_root=shared_data_root)

    examples = service.list_examples()

    assert len(examples) == 1
    assert examples[0].key == "demo_dataset"
    assert examples[0].label == "Demo dataset"
    assert examples[0].category == "pk"
    assert examples[0].route == "oral"
    assert examples[0].dataset_path.endswith("shared_data/demo/demo.csv")


def test_dataset_service_loads_manifest_backed_example(tmp_path: Path) -> None:
    catalog_root, shared_data_root = _write_catalog_dataset_example(tmp_path)

    service = DatasetService(catalog_root=catalog_root, shared_data_root=shared_data_root)

    result = service.load_example("demo_dataset")

    assert result.ok is True
    assert result.dataset_asset is not None
    assert result.dataset_asset.display_name == "Demo dataset"
    assert result.dataset_asset.row_count == 3
    assert result.dataset_asset.subject_count == 2
    assert result.dataset_asset.source_path is not None
    assert Path(result.dataset_asset.source_path).exists()


def test_dataset_service_loads_manifest_example_using_control_stream_metadata(
    tmp_path: Path,
) -> None:
    catalog_root, shared_data_root = _write_catalog_nmdata_example(tmp_path)

    service = DatasetService(catalog_root=catalog_root, shared_data_root=shared_data_root)

    result = service.load_example("nmdata_drop")

    assert result.ok is True
    assert result.dataset_asset is not None
    assert result.dataset_asset.display_name == "NMdata drop example"
    assert result.dataset_asset.input_columns == ["ID", "TIME", "_DROP_3", "WT", "DV"]
    assert result.dataset_asset.columns[:4] == ["ID", "TIME", "WT", "DV"]
    assert result.dataset_asset.preview_rows[0]["WT"] == 70
    assert all(not column.startswith("_DROP_") for column in result.dataset_asset.columns)


def test_dataset_service_reports_missing_example_dataset(tmp_path: Path) -> None:
    catalog_root, shared_data_root = _write_catalog_dataset_example(tmp_path)
    service = DatasetService(catalog_root=catalog_root, shared_data_root=shared_data_root)

    result = service.load_example("missing_example")

    assert result.ok is False
    assert result.validation.ok is False
    assert "could not be found" in result.validation.issues[0].message
