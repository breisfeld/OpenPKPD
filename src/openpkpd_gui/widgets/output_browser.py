"""Shared output preview panel widget used by Results, Plots, and Diagnostics."""

from __future__ import annotations

import csv
import io
import shutil
from collections.abc import Callable
from pathlib import Path

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    apply_saved_table_column_widths,
    default_workspace_root_path,
    load_gui_preferences,
    save_gui_preferences,
    with_last_file_dialog_dir,
)
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord
from openpkpd_gui.widgets.table_headers import configure_resizable_table_columns


class OutputPreviewPanel:
    """Stacked preview panel (title / metadata / content area) shared across output workflows.

    Callers create one via :func:`build_output_preview_panel` and wire up their action
    buttons through the ``on_has_path`` callback in :meth:`render`.
    """

    def __init__(
        self,
        title_label,
        metadata_label,
        placeholder,
        browser,
        image_label,
        scroll,
        stack,
        qt_core,
        qt_gui,
        table_widget=None,
    ) -> None:
        self.title_label = title_label
        self.metadata_label = metadata_label
        self.placeholder = placeholder
        self.browser = browser
        self.image_label = image_label
        self.scroll = scroll
        self.stack = stack
        self._qt_core = qt_core
        self._qt_gui = qt_gui
        self.table_widget = table_widget

    def render(
        self,
        artifact: ArtifactRecord | None,
        runs_by_id: dict[str, RunRecord] | None = None,
        *,
        format_metadata_fn: Callable | None = None,
        empty_label: str = "Select an output to preview.",
        on_has_path: Callable[[bool], None] | None = None,
    ) -> None:
        """Update the preview panel for *artifact*.

        Parameters
        ----------
        artifact:
            The artifact to render, or ``None`` to show the empty state.
        runs_by_id:
            Optional run lookup for metadata enrichment.
        format_metadata_fn:
            Callable ``(artifact, runs_by_id) -> str``.  Defaults to the shared
            :func:`~openpkpd_gui.workflows.results_workflow.format_artifact_metadata`.
        empty_label:
            Title text shown when *artifact* is ``None``.
        on_has_path:
            Called with ``True`` when *artifact* has a readable path, ``False``
            otherwise.  Use this to enable/disable action buttons.
        """
        from openpkpd_gui.workflows.results_workflow import (
            artifact_preview_kind,
            format_artifact_metadata,
        )

        _fmt = format_metadata_fn or format_artifact_metadata
        qt_core = self._qt_core
        qt_gui = self._qt_gui

        self.title_label.setText(artifact.label if artifact is not None else empty_label)
        self.metadata_label.setText(_fmt(artifact, runs_by_id))

        has_path = bool(artifact is not None and artifact.path)
        if on_has_path is not None:
            on_has_path(has_path)

        if not has_path:
            self.browser.clear()
            self.image_label.clear()
            self.placeholder.setText(
                "Preview supported for HTML, text, CSV table, and image outputs."
            )
            self.stack.setCurrentWidget(self.placeholder)
            return

        artifact_path = Path(artifact.path)  # type: ignore[arg-type]
        if not artifact_path.exists():
            self.placeholder.setText("File is not available on disk.")
            self.stack.setCurrentWidget(self.placeholder)
            return

        preview_kind = artifact_preview_kind(artifact)
        if preview_kind == "html":
            self.browser.document().setBaseUrl(
                qt_core.QUrl.fromLocalFile(str(artifact_path.parent) + "/")
            )
            self.browser.setHtml(artifact_path.read_text(encoding="utf-8", errors="replace"))
            self.stack.setCurrentWidget(self.browser)
            return
        if preview_kind == "table" and self.table_widget is not None:
            self._render_csv_table(artifact_path)
            return
        if preview_kind == "text":
            self.browser.setPlainText(artifact_path.read_text(encoding="utf-8", errors="replace"))
            self.stack.setCurrentWidget(self.browser)
            return
        if preview_kind == "image":
            pixmap = qt_gui.QPixmap(str(artifact_path))
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap)
                self.stack.setCurrentWidget(self.scroll)
                return

        self.placeholder.setText(
            "Preview unavailable for this output type. Use Open to inspect it."
        )
        self.stack.setCurrentWidget(self.placeholder)

    def _render_csv_table(self, artifact_path: Path) -> None:
        """Parse *artifact_path* as CSV and populate the table widget."""
        try:
            text = artifact_path.read_text(encoding="utf-8", errors="replace")
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
        except Exception:
            self.placeholder.setText("Could not parse CSV file.")
            self.stack.setCurrentWidget(self.placeholder)
            return

        if not rows:
            self.placeholder.setText("CSV file is empty.")
            self.stack.setCurrentWidget(self.placeholder)
            return

        _, _, qt_widgets_mod = load_qt_modules()
        headers = rows[0]
        data_rows = rows[1:]
        tw = self.table_widget
        tw.setUpdatesEnabled(False)
        tw.clearContents()
        tw.setColumnCount(len(headers))
        tw.setRowCount(len(data_rows))
        tw.setHorizontalHeaderLabels(headers)
        for r, row in enumerate(data_rows):
            for c, cell in enumerate(row):
                item = qt_widgets_mod.QTableWidgetItem(cell)
                tw.setItem(r, c, item)
        tw.resizeColumnsToContents()
        apply_saved_table_column_widths(tw)
        tw.setUpdatesEnabled(True)
        self.stack.setCurrentWidget(tw)


def open_output_file(artifact: ArtifactRecord | None) -> None:
    """Open *artifact*'s backing file in the system default viewer."""
    if artifact is None or not artifact.path:
        return
    qt_core, qt_gui, _ = load_qt_modules()
    qt_gui.QDesktopServices.openUrl(qt_core.QUrl.fromLocalFile(artifact.path))


def export_output_file(
    artifact: ArtifactRecord | None,
    *,
    title: str,
    parent,
    fallback_dir: Path | None = None,
) -> bool:
    """Prompt the user to save a copy of *artifact*'s file.

    Returns ``True`` when a copy was successfully written.
    """
    if artifact is None or not artifact.path:
        return False
    source_path = Path(artifact.path)
    if not source_path.exists():
        return False
    _, _, qt_widgets = load_qt_modules()
    preferences = load_gui_preferences()
    default_name = source_path.name or f"{getattr(artifact, 'label', 'output')}.output"
    dest_dir = Path(
        preferences.last_file_dialog_dir or fallback_dir or default_workspace_root_path()
    )
    destination_path, _ = qt_widgets.QFileDialog.getSaveFileName(
        parent,
        title,
        str(dest_dir / default_name),
        "All files (*)",
    )
    if not destination_path:
        return False
    shutil.copy2(source_path, destination_path)
    save_gui_preferences(with_last_file_dialog_dir(preferences, destination_path))
    return True


def build_output_preview_panel(
    parent,
    *,
    object_prefix: str,
    placeholder_text: str = "Preview supported for HTML, text, CSV table, and image outputs.",
) -> OutputPreviewPanel:
    """Build the shared output preview panel.

    Returns an :class:`OutputPreviewPanel` whose individual widgets are also
    accessible via its attributes (``.title_label``, ``.metadata_label``,
    ``.stack``, etc.) so callers can embed them in any layout arrangement.
    """
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    from openpkpd_gui.workflows.results_workflow import format_artifact_metadata

    title_label = qt_widgets.QLabel("Select an output to preview.")
    title_label.setObjectName(f"{object_prefix}-preview-title")
    title_label.setWordWrap(True)

    metadata_label = qt_widgets.QLabel(format_artifact_metadata(None))
    metadata_label.setObjectName(f"{object_prefix}-preview-metadata-label")
    metadata_label.setWordWrap(True)

    stack = qt_widgets.QStackedWidget()
    stack.setObjectName(f"{object_prefix}-preview-stack")

    placeholder = qt_widgets.QLabel(placeholder_text)
    placeholder.setObjectName(f"{object_prefix}-preview-placeholder")
    placeholder.setWordWrap(True)
    placeholder.setAlignment(qt_core.Qt.AlignmentFlag.AlignTop)

    browser = qt_widgets.QTextBrowser()
    browser.setObjectName(f"{object_prefix}-preview-browser")

    image_label = qt_widgets.QLabel()
    image_label.setObjectName(f"{object_prefix}-preview-image")
    image_label.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)

    scroll = qt_widgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(image_label)

    table_widget = qt_widgets.QTableWidget()
    table_widget.setObjectName(f"{object_prefix}-preview-table")
    table_widget.setEditTriggers(qt_widgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table_widget.setSelectionBehavior(qt_widgets.QAbstractItemView.SelectionBehavior.SelectRows)
    configure_resizable_table_columns(table_widget, qt_widgets)
    table_widget.setAlternatingRowColors(True)

    stack.addWidget(placeholder)
    stack.addWidget(browser)
    stack.addWidget(scroll)
    stack.addWidget(table_widget)

    return OutputPreviewPanel(
        title_label=title_label,
        metadata_label=metadata_label,
        placeholder=placeholder,
        browser=browser,
        image_label=image_label,
        scroll=scroll,
        stack=stack,
        qt_core=qt_core,
        qt_gui=qt_gui,
        table_widget=table_widget,
    )
