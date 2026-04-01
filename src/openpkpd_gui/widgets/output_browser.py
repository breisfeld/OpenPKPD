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
        mpl_canvas_container=None,
        mpl_canvas=None,
        mpl_ax=None,
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
        self.mpl_canvas_container = mpl_canvas_container
        self.mpl_canvas = mpl_canvas
        self.mpl_ax = mpl_ax

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
            if self.mpl_canvas is not None:
                try:
                    import matplotlib.image as mpimg

                    img = mpimg.imread(str(artifact_path))
                    self.mpl_ax.clear()
                    self.mpl_ax.imshow(img)
                    self.mpl_ax.axis("off")
                    self.mpl_canvas.figure.tight_layout(pad=0)
                    self.mpl_canvas.draw()
                    self.stack.setCurrentWidget(self.mpl_canvas_container)
                    return
                except Exception:
                    pass
            pixmap = qt_gui.QPixmap(str(artifact_path))
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap)
                self.stack.setCurrentWidget(self.scroll)
                return

        self.placeholder.setText(
            "Preview unavailable for this output type. Use Open to inspect it."
        )
        self.stack.setCurrentWidget(self.placeholder)

    # ------------------------------------------------------------------ #
    # GOF subject-highlighting                                             #
    # ------------------------------------------------------------------ #

    # Maps plot_type → (x_col, y_col | None for special types)
    _GOF_PLOT_COLUMNS: dict[str, tuple[str, str] | None] = {
        "dv_vs_ipred": ("IPRED", "DV"),
        "dv_vs_pred": ("PRED", "DV"),
        "cwres_vs_time": ("TIME", "CWRES"),
        "cwres_vs_pred": ("PRED", "CWRES"),
        "abs_iwres_vs_ipred": ("IPRED", "IWRES"),
    }

    def render_highlighted_artifact(
        self,
        diag_df,
        artifact: ArtifactRecord,
        selected_subject: str,
    ) -> bool:
        """Re-render a GOF artifact into the live canvas with *selected_subject* highlighted.

        Returns ``True`` if the canvas was updated, ``False`` if re-rendering is
        not supported for this artifact / environment.
        """
        if self.mpl_canvas is None or self.mpl_ax is None:
            return False

        plot_type = (artifact.metadata or {}).get("plot_type", "")
        if plot_type not in self._GOF_PLOT_COLUMNS:
            return False

        try:
            from openpkpd.plots import gof as _gof

            ax = self.mpl_ax
            ax.clear()

            _render_fn = {
                "dv_vs_ipred": _gof.dv_vs_ipred,
                "dv_vs_pred": _gof.dv_vs_pred,
                "cwres_vs_time": _gof.cwres_vs_time,
                "cwres_vs_pred": _gof.cwres_vs_pred,
                "abs_iwres_vs_ipred": _gof.abs_iwres_vs_ipred,
            }.get(plot_type)
            if _render_fn is None:
                return False

            _render_fn(diag_df, ax=ax)

            # Overlay the selected subject's points
            xy_cols = self._GOF_PLOT_COLUMNS[plot_type]
            if xy_cols is not None and "ID" in diag_df.columns:
                subject_mask = diag_df["ID"].astype(str) == str(selected_subject)
                subject_df = diag_df[subject_mask]
                if not subject_df.empty:
                    x_col, y_col = xy_cols
                    x_vals = subject_df[x_col].values
                    y_vals = (
                        abs(subject_df[y_col].values)
                        if plot_type == "abs_iwres_vs_ipred"
                        else subject_df[y_col].values
                    )
                    ax.scatter(
                        x_vals,
                        y_vals,
                        color="#ef4444",
                        s=60,
                        zorder=5,
                        edgecolors="white",
                        linewidths=0.8,
                        label=f"ID={selected_subject}",
                    )
                    ax.legend(fontsize=8)

            self.mpl_canvas.figure.tight_layout(pad=0.5)
            self.mpl_canvas.draw()
            self.stack.setCurrentWidget(self.mpl_canvas_container)
            return True
        except Exception:
            return False

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

    # Interactive matplotlib canvas for image artifacts (P2-A)
    mpl_canvas_container = None
    mpl_canvas = None
    mpl_ax = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        mpl_fig, mpl_ax = plt.subplots(figsize=(8, 6))
        plt.close(mpl_fig)  # unregister from pyplot to avoid figure leak warnings
        mpl_fig.patch.set_facecolor("#ffffff")
        mpl_ax.set_facecolor("#ffffff")
        mpl_ax.axis("off")
        mpl_fig.tight_layout(pad=0)

        mpl_canvas = FigureCanvasQTAgg(mpl_fig)
        mpl_canvas.setObjectName(f"{object_prefix}-preview-mpl-canvas")

        mpl_toolbar = NavigationToolbar2QT(mpl_canvas, None)
        mpl_toolbar.setObjectName(f"{object_prefix}-preview-mpl-toolbar")

        mpl_canvas_container = qt_widgets.QWidget()
        mpl_layout = qt_widgets.QVBoxLayout(mpl_canvas_container)
        mpl_layout.setContentsMargins(0, 0, 0, 0)
        mpl_layout.setSpacing(0)
        mpl_layout.addWidget(mpl_toolbar)
        mpl_layout.addWidget(mpl_canvas)
        stack.addWidget(mpl_canvas_container)
    except Exception:
        pass

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
        mpl_canvas_container=mpl_canvas_container,
        mpl_canvas=mpl_canvas,
        mpl_ax=mpl_ax,
    )
