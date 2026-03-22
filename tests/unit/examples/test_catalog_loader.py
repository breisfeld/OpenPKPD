"""Tests for the manifest-backed examples catalog loader and service."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from openpkpd.examples import ExampleCatalogService, load_catalog, load_manifest


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")


@pytest.mark.unit
def test_load_manifest_resolves_bundle_and_shared_files(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "examples" / "catalog" / "pk" / "oral" / "demo"
    shared_data_root = tmp_path / "examples" / "shared_data"
    _write_text(
        bundle_dir / "manifest.yaml",
        """
        manifest_version: 1
        id: demo_dataset
        title: Demo dataset example
        description: Demo manifest for testing.
        category: pk
        primary_mode: control_stream
        route: oral
        difficulty: starter
        tags: [demo, oral]
        files:
          dataset: ../../../../shared_data/theophylline/demo.csv
          control_stream: model.ctl
          script: example.py
        gui:
          load_dataset: true
          load_control_stream: true
        """,
    )
    _write_text(bundle_dir / "model.ctl", "$PROBLEM Demo\n")
    _write_text(bundle_dir / "example.py", "print('demo')\n")
    _write_text(shared_data_root / "theophylline" / "demo.csv", "ID,TIME,DV\n1,0,0\n")

    result = load_manifest(bundle_dir / "manifest.yaml", shared_data_root=shared_data_root)

    assert result.ok is True
    assert result.entry is not None
    assert result.entry.manifest.id == "demo_dataset"
    assert result.entry.dataset_path == (shared_data_root / "theophylline" / "demo.csv").resolve()
    assert result.entry.control_stream_path == (bundle_dir / "model.ctl").resolve()
    assert result.entry.script_path == (bundle_dir / "example.py").resolve()


@pytest.mark.unit
def test_load_manifest_reports_missing_required_field(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    _write_text(
        manifest_path,
        """
        id: demo
        title: Demo
        description: Missing required fields.
        files: {}
        gui: {}
        """,
    )

    result = load_manifest(manifest_path)

    assert result.ok is False
    assert any(issue.field == "manifest_version" for issue in result.issues)
    assert any(issue.field == "category" for issue in result.issues)


@pytest.mark.unit
def test_load_manifest_reports_missing_referenced_file(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "examples" / "catalog" / "nca" / "demo"
    _write_text(
        bundle_dir / "manifest.yaml",
        """
        manifest_version: 1
        id: broken_demo
        title: Broken demo
        description: Missing dataset file.
        category: nca
        primary_mode: dataset
        difficulty: starter
        files:
          dataset: dataset.csv
        gui:
          load_dataset: true
        """,
    )

    result = load_manifest(bundle_dir / "manifest.yaml")

    assert result.ok is False
    assert any(
        issue.field == "files.dataset" and "does not exist" in issue.message
        for issue in result.issues
    )


@pytest.mark.unit
def test_load_manifest_rejects_path_outside_allowed_roots(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "examples" / "catalog" / "workflow" / "demo"
    _write_text(
        bundle_dir / "manifest.yaml",
        """
        manifest_version: 1
        id: escaped_demo
        title: Escaped demo
        description: Invalid dataset path.
        category: workflow
        primary_mode: dataset
        difficulty: starter
        files:
          dataset: ../../../../../outside.csv
        gui:
          load_dataset: true
        """,
    )
    _write_text(tmp_path / "outside.csv", "ID,TIME,DV\n1,0,0\n")

    result = load_manifest(bundle_dir / "manifest.yaml")

    assert result.ok is False
    assert any(
        issue.field == "files.dataset" and "escapes allowed" in issue.message
        for issue in result.issues
    )


@pytest.mark.unit
def test_load_catalog_rejects_duplicate_ids(tmp_path: Path) -> None:
    catalog_root = tmp_path / "examples" / "catalog"
    _write_text(
        catalog_root / "pk" / "oral" / "demo_a" / "manifest.yaml",
        """
        manifest_version: 1
        id: duplicate_demo
        title: Demo A
        description: First duplicate manifest.
        category: pk
        primary_mode: control_stream
        difficulty: starter
        files:
          control_stream: model.ctl
        gui:
          load_control_stream: true
        """,
    )
    _write_text(catalog_root / "pk" / "oral" / "demo_a" / "model.ctl", "$PROBLEM A\n")
    _write_text(
        catalog_root / "pk" / "iv" / "demo_b" / "manifest.yaml",
        """
        manifest_version: 1
        id: duplicate_demo
        title: Demo B
        description: Second duplicate manifest.
        category: pk
        primary_mode: control_stream
        difficulty: starter
        files:
          control_stream: model.ctl
        gui:
          load_control_stream: true
        """,
    )
    _write_text(catalog_root / "pk" / "iv" / "demo_b" / "model.ctl", "$PROBLEM B\n")

    with pytest.raises(ValueError, match=r"Duplicate example id\(s\): duplicate_demo"):
        load_catalog(catalog_root)


@pytest.mark.unit
def test_example_catalog_service_filters_dataset_and_control_stream_examples(
    tmp_path: Path,
) -> None:
    catalog_root = tmp_path / "examples" / "catalog"
    shared_data_root = tmp_path / "examples" / "shared_data"
    _write_text(
        catalog_root / "nca" / "basic" / "manifest.yaml",
        """
        manifest_version: 1
        id: basic_nca
        title: Basic NCA
        description: Dataset-backed NCA example.
        category: nca
        primary_mode: nca
        difficulty: starter
        sort_order: 40
        files:
          dataset: ../../../shared_data/theophylline/theo.csv
        gui:
          load_dataset: true
        """,
    )
    _write_text(shared_data_root / "theophylline" / "theo.csv", "ID,TIME,DV\n1,0,0\n")
    _write_text(
        catalog_root / "pk" / "iv" / "two_cmt" / "manifest.yaml",
        """
        manifest_version: 1
        id: two_compartment_iv
        title: Two-compartment IV
        description: Control-stream-backed PK example.
        category: pk
        primary_mode: control_stream
        route: iv
        difficulty: starter
        sort_order: 10
        files:
          control_stream: model.ctl
        gui:
          load_control_stream: true
        """,
    )
    _write_text(catalog_root / "pk" / "iv" / "two_cmt" / "model.ctl", "$PROBLEM Two cmt\n")

    service = ExampleCatalogService(catalog_root=catalog_root, shared_data_root=shared_data_root)

    dataset_examples = service.list_dataset_examples()
    control_examples = service.list_control_stream_examples()

    assert [entry.manifest.id for entry in dataset_examples] == ["basic_nca"]
    assert [entry.manifest.id for entry in control_examples] == ["two_compartment_iv"]
    assert service.get_example_by_id("basic_nca") is not None
    assert service.get_example_by_id("missing") is None


@pytest.mark.unit
def test_example_catalog_service_keeps_control_stream_only_examples_out_of_dataset_list(
    tmp_path: Path,
) -> None:
    catalog_root = tmp_path / "examples" / "catalog"

    _write_text(
        catalog_root / "pk" / "iv" / "two_cmt" / "manifest.yaml",
        """
        manifest_version: 1
        id: two_compartment_iv_focei
        title: Two-compartment IV FOCEI
        description: Control stream without a bundled dataset.
        category: pk
        primary_mode: control_stream
        route: iv
        difficulty: advanced
        files:
          dataset: null
          control_stream: model.ctl
        gui:
          load_dataset: false
          load_control_stream: true
        """,
    )
    _write_text(catalog_root / "pk" / "iv" / "two_cmt" / "model.ctl", "$PROBLEM Two cmt\n")

    service = ExampleCatalogService(catalog_root=catalog_root)

    assert service.list_dataset_examples() == []
    assert [entry.manifest.id for entry in service.list_control_stream_examples()] == [
        "two_compartment_iv_focei"
    ]


@pytest.mark.unit
def test_real_repository_catalog_loads_seeded_examples() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    catalog_root = repo_root / "examples" / "catalog"
    shared_data_root = repo_root / "examples" / "shared_data"

    entries = load_catalog(catalog_root, shared_data_root=shared_data_root)
    entries_by_id = {entry.manifest.id: entry for entry in entries}
    entry_ids = set(entries_by_id)

    assert len(entries) >= 20
    assert {
        "theophylline_oral_fo",
        "basic_nca",
        "minimal_theophylline_focei",
        "two_compartment_iv_focei",
        "covariates_one_cmt_focei",
        "phenobarbital_fo",
        "same_omega_showcase",
        "nmdata_input_drop_and_body_weight_scaling",
        "nmsim_advan13_one_compartment_des",
    }.issubset(entry_ids)
    assert entries_by_id["nmdata_input_drop_and_body_weight_scaling"].readme_path is not None
    assert entries_by_id["nmsim_advan13_one_compartment_des"].readme_path is not None
