from __future__ import annotations

from pathlib import Path

from openpkpd.release.preflight import release_tree_issues


def test_release_tree_issues_is_empty_for_clean_tree(tmp_path: Path) -> None:
    assert release_tree_issues(tmp_path) == []


def test_release_tree_issues_reports_local_build_artifacts(tmp_path: Path) -> None:
    (tmp_path / "rust" / "target").mkdir(parents=True)
    (tmp_path / "rust" / ".venv").mkdir(parents=True)
    so_path = tmp_path / "src" / "openpkpd" / "_core.cpython-312-x86_64-linux-gnu.so"
    so_path.parent.mkdir(parents=True)
    so_path.write_bytes(b"binary")

    issues = release_tree_issues(tmp_path)

    assert "rust/target" in issues
    assert "rust/.venv" in issues
    assert "src/openpkpd/_core.cpython-312-x86_64-linux-gnu.so" in issues
