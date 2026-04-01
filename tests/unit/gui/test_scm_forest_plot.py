"""Tests for P3-B: SCM step significance plot auto-generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from openpkpd_gui.services.scm_service import SCMRunResult, generate_scm_step_plot


def _make_scm_result(accepted_count: int = 2, total_count: int = 4) -> SCMRunResult:
    step_rows = []
    for i in range(total_count):
        step_rows.append(
            {
                "type": "forward",
                "rel": f"CL~COV{i + 1}(power)",
                "delta_ofv": -(10.0 + i * 3),
                "p_value": 0.001,
                "accepted": i < accepted_count,
            }
        )
    return SCMRunResult(
        summary_text="SCM summary",
        step_rows=step_rows,
        accepted_count=accepted_count,
        final_ofv=100.0,
        base_ofv=140.0,
    )


class TestGenerateScmStepPlot:
    def test_returns_artifact_when_accepted_steps_present(self, tmp_path: Path) -> None:
        result = _make_scm_result(accepted_count=2)
        artifact = generate_scm_step_plot(result, run_id="abc123", output_dir=tmp_path)
        assert artifact is not None

    def test_artifact_kind_is_plot(self, tmp_path: Path) -> None:
        result = _make_scm_result(accepted_count=2)
        artifact = generate_scm_step_plot(result, run_id="abc123", output_dir=tmp_path)
        assert artifact is not None
        assert artifact.kind == "plot"

    def test_artifact_path_is_png_that_exists(self, tmp_path: Path) -> None:
        result = _make_scm_result(accepted_count=2)
        artifact = generate_scm_step_plot(result, run_id="abc123", output_dir=tmp_path)
        assert artifact is not None
        assert artifact.path is not None
        assert Path(artifact.path).exists()
        assert artifact.path.endswith(".png")

    def test_artifact_source_run_id_matches(self, tmp_path: Path) -> None:
        result = _make_scm_result(accepted_count=2)
        artifact = generate_scm_step_plot(result, run_id="testrun99", output_dir=tmp_path)
        assert artifact is not None
        assert artifact.source_run_id == "testrun99"

    def test_artifact_label_mentions_scm(self, tmp_path: Path) -> None:
        result = _make_scm_result(accepted_count=2)
        artifact = generate_scm_step_plot(result, run_id="abc123", output_dir=tmp_path)
        assert artifact is not None
        assert "SCM" in artifact.label or "scm" in artifact.label.lower()

    def test_returns_none_when_no_accepted_steps(self, tmp_path: Path) -> None:
        result = _make_scm_result(accepted_count=0)
        artifact = generate_scm_step_plot(result, run_id="abc123", output_dir=tmp_path)
        assert artifact is None

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "nested" / "artifacts"
        result = _make_scm_result(accepted_count=1)
        artifact = generate_scm_step_plot(result, run_id="abc123", output_dir=output_dir)
        assert artifact is not None
        assert output_dir.exists()

    def test_single_accepted_step(self, tmp_path: Path) -> None:
        result = _make_scm_result(accepted_count=1)
        artifact = generate_scm_step_plot(result, run_id="abc123", output_dir=tmp_path)
        assert artifact is not None

    def test_metadata_has_plot_type_scm(self, tmp_path: Path) -> None:
        result = _make_scm_result(accepted_count=2)
        artifact = generate_scm_step_plot(result, run_id="abc123", output_dir=tmp_path)
        assert artifact is not None
        assert artifact.metadata.get("plot_type") == "scm"

    def test_many_accepted_steps(self, tmp_path: Path) -> None:
        result = _make_scm_result(accepted_count=8, total_count=10)
        artifact = generate_scm_step_plot(result, run_id="abc123", output_dir=tmp_path)
        assert artifact is not None
        assert Path(artifact.path).stat().st_size > 0
