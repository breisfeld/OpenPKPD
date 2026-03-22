"""
About dialog and User Guide browser for the OpenPKPD desktop shell.

Provides:
  get_app_metadata()         — version, description, and licence from package metadata
  open_about_dialog()        — versioned About dialog
  open_help_dialog()         — resizable User Guide browser, optionally
                               scrolled to the section for a specific workflow
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_SECTION_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------


def get_app_metadata() -> dict[str, str]:
    """
    Return a dict with keys 'version', 'description', and 'license'
    sourced from importlib.metadata (installed package) with a pyproject.toml
    fallback for editable/source checkouts.
    """
    try:
        from importlib.metadata import metadata

        meta = metadata("openpkpd")
        return {
            "version": meta["Version"] or "dev",
            "description": meta["Summary"] or "",
            "license": meta["License"] or "",
        }
    except Exception:
        pass

    # Fallback: parse pyproject.toml directly
    result: dict[str, str] = {"version": "dev", "description": "", "license": ""}
    try:
        toml_text = (Path(__file__).parents[3] / "pyproject.toml").read_text(encoding="utf-8")
        for key, pattern in (
            ("version", r'^version\s*=\s*"([^"]+)"'),
            ("description", r'^description\s*=\s*"([^"]+)"'),
            ("license", r'^license\s*=\s*\{[^}]*text\s*=\s*"([^"]+)"'),
        ):
            m = re.search(pattern, toml_text, re.MULTILINE)
            if m:
                result[key] = m.group(1)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Guide file location
# ---------------------------------------------------------------------------

# Workflow ID → heading text in gui.md (used by find() to scroll)
WORKFLOW_HEADINGS: dict[str, str] = {
    "overview": "Dashboard workflow",
    "data": "Data workflow",
    "model": "Model workflow",
    "fit": "Fit workflow",
    "nca": "NCA workflow",
    "results": "Results workflow",
    "plots": "Plots workflow",
    "diagnostics": "Diagnostics workflow",
    "advanced": "Advanced workflow",
    "covariate": "Covariate workflow",
}


def _guide_sections(markdown_text: str) -> list[tuple[int, str]]:
    """Return level/title pairs for H2 and H3 guide headings."""
    sections: list[tuple[int, str]] = []
    for heading_markup, raw_title in _SECTION_HEADING_RE.findall(markdown_text):
        title = re.sub(r"[`*_]", "", raw_title).strip()
        if title:
            sections.append((len(heading_markup), title))
    return sections


def _scroll_browser_to_heading(browser: Any, heading: str) -> bool:
    """Move the help browser cursor to the requested heading text."""
    cursor = browser.document().find(heading)
    if cursor.isNull():
        return False
    browser.setTextCursor(cursor)
    browser.ensureCursorVisible()
    return True


def _build_contents_tree(
    tree: Any, markdown_text: str, qt_widgets: Any, qt_core: Any
) -> dict[str, Any]:
    """Populate the help dialog contents tree from guide headings."""
    section_items: dict[str, Any] = {}
    last_top_level_item = None

    for level, title in _guide_sections(markdown_text):
        item = qt_widgets.QTreeWidgetItem([title])
        item.setData(0, qt_core.Qt.ItemDataRole.UserRole, title)
        if level == 2:
            tree.addTopLevelItem(item)
            last_top_level_item = item
        elif level == 3:
            parent = last_top_level_item
            if parent is None:
                tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
        section_items[title] = item

    tree.expandAll()
    return section_items


def _find_guide_path() -> Path | None:
    """Locate gui.md from the development tree or the installed package."""
    # Development / editable install: src/openpkpd_gui/shell/ → root/docs/
    dev_candidate = Path(__file__).parents[3] / "docs" / "user_guide" / "gui.md"
    if dev_candidate.exists():
        return dev_candidate
    # Wheel install: bundled alongside this package
    bundled = Path(__file__).parent.parent / "help" / "gui.md"
    if bundled.exists():
        return bundled
    return None


# ---------------------------------------------------------------------------
# About dialog
# ---------------------------------------------------------------------------


def open_about_dialog(parent: Any, qt_widgets: Any, qt_core: Any) -> None:
    """Show the versioned About dialog."""
    meta = get_app_metadata()
    version = meta["version"]
    description = meta["description"]
    license_id = meta["license"]

    pyside_ver = ""
    try:
        import PySide6

        pyside_ver = f"PySide6 {PySide6.__version__}  ·  "
    except Exception:
        pass
    qt_ver = qt_core.qVersion()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    text = (
        f"<b>OpenPKPD GUI</b> &nbsp; v{version}<br><br>"
        f"{description}<br><br>"
        f"Python {py_ver} &nbsp;·&nbsp; {pyside_ver}Qt {qt_ver}<br><br>"
        f"© 2024–2026 OpenPKPD Contributors<br>"
        f"<small>{license_id}</small>"
    )
    qt_widgets.QMessageBox.about(parent, "About OpenPKPD GUI", text)


# ---------------------------------------------------------------------------
# User Guide browser
# ---------------------------------------------------------------------------


def open_help_dialog(
    parent: Any,
    qt_widgets: Any,
    qt_core: Any,
    workflow_id: str | None = None,
) -> None:
    """
    Open a resizable dialog displaying the User Guide (gui.md).

    Args:
        parent:      Parent QWidget.
        qt_widgets:  The PySide6.QtWidgets module.
        qt_core:     The PySide6.QtCore module.
        workflow_id: If given, the dialog scrolls to that workflow's section.
    """
    guide_path = _find_guide_path()

    dialog = qt_widgets.QDialog(parent)
    dialog.setObjectName("help-dialog")
    dialog.setWindowTitle("OpenPKPD — User Guide")
    dialog.resize(980, 680)
    dialog.setWindowFlags(dialog.windowFlags() | qt_core.Qt.WindowType.WindowMaximizeButtonHint)

    layout = qt_widgets.QVBoxLayout(dialog)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)

    intro_label = qt_widgets.QLabel(
        "Use the table of contents to jump through the guide. "
        "‘Help for this workflow’ opens this dialog focused on the current page."
    )
    intro_label.setObjectName("help-intro-label")
    intro_label.setWordWrap(True)
    layout.addWidget(intro_label)

    splitter = qt_widgets.QSplitter(qt_core.Qt.Orientation.Horizontal, dialog)
    splitter.setObjectName("help-content-splitter")

    contents_panel = qt_widgets.QWidget(splitter)
    contents_panel.setObjectName("help-contents-panel")
    contents_layout = qt_widgets.QVBoxLayout(contents_panel)
    contents_layout.setContentsMargins(0, 0, 0, 0)
    contents_layout.setSpacing(6)

    contents_label = qt_widgets.QLabel("Contents", contents_panel)
    contents_label.setObjectName("help-contents-label")
    contents_layout.addWidget(contents_label)

    contents_tree = qt_widgets.QTreeWidget(contents_panel)
    contents_tree.setObjectName("help-contents-tree")
    contents_tree.setHeaderHidden(True)
    contents_tree.setRootIsDecorated(True)
    contents_tree.setUniformRowHeights(True)
    contents_layout.addWidget(contents_tree, 1)

    browser = qt_widgets.QTextBrowser(splitter)
    browser.setObjectName("help-guide-browser")
    browser.setOpenExternalLinks(True)
    browser.setReadOnly(True)

    guide_text = ""

    if guide_path is not None:
        guide_text = guide_path.read_text(encoding="utf-8")
        browser.setMarkdown(guide_text)
        section_items = _build_contents_tree(contents_tree, guide_text, qt_widgets, qt_core)
    else:
        browser.setHtml(
            "<h2>User Guide not found</h2>"
            "<p>The guide file could not be located in this installation.</p>"
            "<p>Documentation is available in the <code>docs/user_guide/gui.md</code> "
            "file of the source repository.</p>"
        )
        placeholder_item = qt_widgets.QTreeWidgetItem(["Guide file not found"])
        placeholder_item.setFlags(placeholder_item.flags() & ~qt_core.Qt.ItemFlag.ItemIsSelectable)
        contents_tree.addTopLevelItem(placeholder_item)
        section_items = {}

    def _select_heading(heading: str) -> None:
        item = section_items.get(heading)
        if item is not None:
            parent = item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
            contents_tree.setCurrentItem(item)
            contents_tree.scrollToItem(item)
            return
        _scroll_browser_to_heading(browser, heading)

    def _on_contents_item_changed(current, _previous) -> None:
        if current is None:
            return
        heading = current.data(0, qt_core.Qt.ItemDataRole.UserRole)
        if isinstance(heading, str):
            _scroll_browser_to_heading(browser, heading)

    contents_tree.currentItemChanged.connect(_on_contents_item_changed)

    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([260, 700])
    layout.addWidget(splitter, 1)

    def _select_first_contents_item() -> None:
        try:
            first_item = contents_tree.topLevelItem(0)
            if first_item is not None:
                contents_tree.setCurrentItem(first_item)
        except RuntimeError:
            return

    # Scroll to the workflow section after the dialog is shown
    if workflow_id and (heading := WORKFLOW_HEADINGS.get(workflow_id)):

        def _scroll_to_section() -> None:
            try:
                _select_heading(heading)
            except RuntimeError:
                return

        # Use a single-shot timer so the widget is fully laid out first
        qt_core.QTimer.singleShot(0, _scroll_to_section)
    elif contents_tree.topLevelItemCount() > 0:
        qt_core.QTimer.singleShot(0, _select_first_contents_item)

    close_btn = qt_widgets.QPushButton("Close", dialog)
    close_btn.setObjectName("help-close-button")
    close_btn.setDefault(True)
    close_btn.clicked.connect(dialog.accept)
    btn_row = qt_widgets.QHBoxLayout()
    btn_row.addStretch()
    btn_row.addWidget(close_btn)
    layout.addLayout(btn_row)

    dialog.exec()
