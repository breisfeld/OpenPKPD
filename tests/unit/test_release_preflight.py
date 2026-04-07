from __future__ import annotations

from pathlib import Path

import pytest

from openpkpd.release.preflight import release_tree_issues, strict_release_validation_enabled
from tests._release_validation import require_release_fixture


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


def test_strict_release_validation_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENPKPD_STRICT_RELEASE_VALIDATION", raising=False)
    assert strict_release_validation_enabled() is False


def test_strict_release_validation_enabled_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENPKPD_STRICT_RELEASE_VALIDATION", "1")
    assert strict_release_validation_enabled() is True


def test_require_release_fixture_skips_outside_release_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENPKPD_STRICT_RELEASE_VALIDATION", raising=False)
    with pytest.raises(pytest.skip.Exception, match="Reference file not found"):
        require_release_fixture(tmp_path / "missing.json", kind="Reference file")


def test_require_release_fixture_fails_in_release_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENPKPD_STRICT_RELEASE_VALIDATION", "1")
    with pytest.raises(pytest.fail.Exception, match="Data file not found"):
        require_release_fixture(tmp_path / "missing.csv", kind="Data file")
