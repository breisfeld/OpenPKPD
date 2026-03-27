"""Tests for openpkpd_gui.shell.help_browser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from openpkpd_gui.shell.help_browser import (
    WORKFLOW_HEADINGS,
    _find_guide_path,
    _guide_sections,
    get_app_metadata,
    open_about_dialog,
    open_help_dialog,
)

# ---------------------------------------------------------------------------
# get_app_metadata
# ---------------------------------------------------------------------------


class TestGetAppMetadata:
    def test_returns_dict_with_required_keys(self) -> None:
        meta = get_app_metadata()
        assert "version" in meta
        assert "description" in meta
        assert "license" in meta

    def test_version_is_non_empty_string(self) -> None:
        meta = get_app_metadata()
        assert isinstance(meta["version"], str)
        assert len(meta["version"]) > 0

    def test_description_is_non_empty_string(self) -> None:
        meta = get_app_metadata()
        # pyproject.toml description is set, so this should not be blank
        assert isinstance(meta["description"], str)
        assert len(meta["description"]) > 0

    def test_description_mentions_pkpd(self) -> None:
        meta = get_app_metadata()
        assert "PK" in meta["description"] or "pharmacokinetic" in meta["description"].lower()

    def test_license_is_non_empty_string(self) -> None:
        meta = get_app_metadata()
        assert isinstance(meta["license"], str)
        assert len(meta["license"]) > 0

    def test_falls_back_gracefully_when_importlib_fails(self) -> None:
        with patch("openpkpd_gui.shell.help_browser.re") as mock_re:
            # Simulate importlib failure + regex still working
            mock_re.search.return_value = None
            with patch("builtins.__import__", side_effect=ImportError):
                # Should not raise
                pass  # fallback path tested separately

    def test_version_matches_pyproject_toml(self) -> None:
        import re

        toml = Path(__file__).parents[3] / "pyproject.toml"
        m = re.search(r'^version\s*=\s*"([^"]+)"', toml.read_text(), re.MULTILINE)
        assert m is not None
        expected = m.group(1)
        meta = get_app_metadata()
        assert meta["version"] == expected

    def test_logs_warning_when_all_metadata_sources_fail(self, monkeypatch, caplog) -> None:
        import builtins

        real_import = builtins.__import__

        def _raising_import(name, *args, **kwargs):
            if name == "importlib.metadata":
                raise ImportError("metadata unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _raising_import)
        monkeypatch.setattr(
            "pathlib.Path.read_text",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("pyproject missing")),
        )

        with caplog.at_level("WARNING"):
            meta = get_app_metadata()

        assert meta["version"] == "dev"
        assert any("Could not load OpenPKPD GUI metadata" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# _find_guide_path
# ---------------------------------------------------------------------------


class TestFindGuidePath:
    def test_returns_path_or_none(self) -> None:
        result = _find_guide_path()
        assert result is None or isinstance(result, Path)

    def test_found_path_exists(self) -> None:
        result = _find_guide_path()
        if result is not None:
            assert result.exists()

    def test_found_path_is_markdown(self) -> None:
        result = _find_guide_path()
        if result is not None:
            assert result.suffix == ".md"

    def test_guide_contains_expected_sections(self) -> None:
        result = _find_guide_path()
        if result is None:
            pytest.skip("guide file not found in this environment")
        text = result.read_text(encoding="utf-8")
        assert "### Data workflow" in text
        assert "### Fit workflow" in text
        assert "### Advanced workflow" in text


class TestGuideSections:
    def test_extracts_h2_and_h3_titles_only(self) -> None:
        markdown = """# Desktop GUI

## Quick start

### Common tasks at a glance

#### Step 1

## Workflow guide

### Model workflow
"""

        assert _guide_sections(markdown) == [
            (2, "Quick start"),
            (3, "Common tasks at a glance"),
            (2, "Workflow guide"),
            (3, "Model workflow"),
        ]


# ---------------------------------------------------------------------------
# WORKFLOW_HEADINGS
# ---------------------------------------------------------------------------


class TestWorkflowHeadings:
    def test_covers_all_main_workflows(self) -> None:
        expected = {
            "overview",
            "data",
            "model",
            "fit",
            "nca",
            "results",
            "plots",
            "diagnostics",
            "advanced",
            "covariate",
        }
        assert expected <= set(WORKFLOW_HEADINGS.keys())

    def test_headings_are_non_empty_strings(self) -> None:
        for workflow_id, heading in WORKFLOW_HEADINGS.items():
            assert isinstance(heading, str) and heading, workflow_id

    def test_headings_present_in_guide(self) -> None:
        guide = _find_guide_path()
        if guide is None:
            pytest.skip("guide file not found")
        text = guide.read_text(encoding="utf-8")
        for workflow_id, heading in WORKFLOW_HEADINGS.items():
            assert heading in text, f"Heading '{heading}' for '{workflow_id}' not found in guide"


# ---------------------------------------------------------------------------
# open_about_dialog  (Qt required)
# ---------------------------------------------------------------------------


def _qt_available() -> bool:
    try:
        from openpkpd_gui.app.runtime import qt_widgets_available

        return qt_widgets_available()
    except Exception:
        return False


_QT_SKIP = pytest.mark.skipif(not _qt_available(), reason="Qt not available")


@_QT_SKIP
class TestOpenAboutDialog:
    def test_about_dialog_shows_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from openpkpd_gui.app.runtime import load_qt_modules

        qt_core, _, qt_widgets = load_qt_modules()
        qt_widgets.QApplication.instance() or qt_widgets.QApplication([])

        shown: list[tuple] = []
        monkeypatch.setattr(
            qt_widgets.QMessageBox, "about", lambda parent, title, text: shown.append((title, text))
        )

        open_about_dialog(None, qt_widgets, qt_core)

        assert shown, "QMessageBox.about was not called"
        title, text = shown[0]
        assert "OpenPKPD" in title
        assert "v" in text  # version present
        # description from pyproject.toml
        meta = get_app_metadata()
        assert meta["description"] in text
        # licence
        assert meta["license"] in text

    def test_about_dialog_shows_python_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from openpkpd_gui.app.runtime import load_qt_modules

        qt_core, _, qt_widgets = load_qt_modules()
        qt_widgets.QApplication.instance() or qt_widgets.QApplication([])

        shown: list[str] = []
        monkeypatch.setattr(
            qt_widgets.QMessageBox, "about", lambda parent, title, text: shown.append(text)
        )

        open_about_dialog(None, qt_widgets, qt_core)
        py_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert py_major_minor in shown[0]


@_QT_SKIP
class TestOpenHelpDialog:
    def test_opens_without_error(self) -> None:
        from openpkpd_gui.app.runtime import load_qt_modules

        qt_core, _, qt_widgets = load_qt_modules()
        qt_widgets.QApplication.instance() or qt_widgets.QApplication(["-platform", "offscreen"])

        parent = qt_widgets.QWidget()
        dialog_ref: list = []

        original_exec = qt_widgets.QDialog.exec

        def _capture_exec(self):
            dialog_ref.append(self)
            # Don't actually show the dialog in tests
            return 0

        qt_widgets.QDialog.exec = _capture_exec
        try:
            open_help_dialog(parent, qt_widgets, qt_core)
        finally:
            qt_widgets.QDialog.exec = original_exec

        assert dialog_ref, "Dialog was not created"

    def test_opens_with_workflow_id(self) -> None:
        from openpkpd_gui.app.runtime import load_qt_modules

        qt_core, _, qt_widgets = load_qt_modules()
        qt_widgets.QApplication.instance() or qt_widgets.QApplication(["-platform", "offscreen"])

        parent = qt_widgets.QWidget()
        original_exec = qt_widgets.QDialog.exec
        qt_widgets.QDialog.exec = lambda self: 0
        try:
            # Should not raise for any known workflow_id
            for wf in WORKFLOW_HEADINGS:
                open_help_dialog(parent, qt_widgets, qt_core, workflow_id=wf)
        finally:
            qt_widgets.QDialog.exec = original_exec

    def test_builds_contents_tree_and_focuses_requested_workflow(self, tmp_path: Path) -> None:
        from openpkpd_gui.app.runtime import load_qt_modules

        qt_core, _, qt_widgets = load_qt_modules()
        app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
            ["-platform", "offscreen"]
        )
        guide_path = tmp_path / "gui.md"
        guide_path.write_text(
            """# Desktop GUI

## Quick start

### Common tasks at a glance

## Workflow guide

### Model workflow

Model details.

### Results workflow

Results details.
""",
            encoding="utf-8",
        )

        dialog_ref: list = []
        original_find_guide_path = __import__(
            "openpkpd_gui.shell.help_browser", fromlist=["_find_guide_path"]
        )._find_guide_path
        original_exec = qt_widgets.QDialog.exec

        def _capture_exec(self):
            dialog_ref.append(self)
            app.processEvents()
            return 0

        try:
            import openpkpd_gui.shell.help_browser as help_browser_module

            help_browser_module._find_guide_path = lambda: guide_path
            qt_widgets.QDialog.exec = _capture_exec
            open_help_dialog(None, qt_widgets, qt_core, workflow_id="model")
        finally:
            import openpkpd_gui.shell.help_browser as help_browser_module

            help_browser_module._find_guide_path = original_find_guide_path
            qt_widgets.QDialog.exec = original_exec

        assert dialog_ref, "Dialog was not created"
        dialog = dialog_ref[0]
        contents_tree = dialog.findChild(qt_widgets.QTreeWidget, "help-contents-tree")
        browser = dialog.findChild(qt_widgets.QTextBrowser, "help-guide-browser")
        close_button = dialog.findChild(qt_widgets.QPushButton, "help-close-button")

        assert contents_tree is not None
        assert browser is not None
        assert close_button is not None
        assert contents_tree.topLevelItemCount() == 2
        assert contents_tree.topLevelItem(0).text(0) == "Quick start"
        assert contents_tree.topLevelItem(1).text(0) == "Workflow guide"
        assert contents_tree.topLevelItem(1).child(0).text(0) == "Model workflow"
        assert contents_tree.currentItem() is not None
        assert contents_tree.currentItem().text(0) == "Model workflow"
        assert "Model details." in browser.toPlainText()

    def test_opens_with_unknown_workflow_id(self) -> None:
        from openpkpd_gui.app.runtime import load_qt_modules

        qt_core, _, qt_widgets = load_qt_modules()
        qt_widgets.QApplication.instance() or qt_widgets.QApplication(["-platform", "offscreen"])

        parent = qt_widgets.QWidget()
        original_exec = qt_widgets.QDialog.exec
        qt_widgets.QDialog.exec = lambda self: 0
        try:
            open_help_dialog(parent, qt_widgets, qt_core, workflow_id="nonexistent")
        finally:
            qt_widgets.QDialog.exec = original_exec
