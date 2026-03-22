"""Data workflow widget for importing and summarizing datasets."""

from __future__ import annotations

import html
from pathlib import Path

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    apply_saved_table_column_widths,
    load_gui_preferences,
    save_gui_preferences,
    with_dismissed_hint,
    with_last_file_dialog_dir,
)
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.data_service import DatasetImportOptions, DatasetService, ExampleDataset
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.validation_service import ValidationResult, ValidationSeverity
from openpkpd_gui.widgets.dismissible_hint import build_dismissible_hint
from openpkpd_gui.widgets.link_formatting import (
    copy_link,
    decode_copy_target,
    external_link,
    file_link,
)
from openpkpd_gui.widgets.responsive_layout import (
    install_responsive_box_layouts,
    install_responsive_splitters,
)
from openpkpd_gui.widgets.scrollable_page import build_scrollable_page
from openpkpd_gui.widgets.table_headers import configure_resizable_table_columns

DATA_RESPONSIVE_LAYOUT_BREAKPOINT = 1100


def format_dataset_summary(dataset_asset: DatasetAsset | None) -> str:
    """Return a compact human-readable dataset summary."""
    if dataset_asset is None:
        return "No dataset loaded yet. Import a CSV file to inspect it here."

    parts = [f"{dataset_asset.row_count} rows"]
    if dataset_asset.subject_count is not None:
        parts.append(f"{dataset_asset.subject_count} subjects")
    if dataset_asset.observation_count is not None:
        parts.append(f"{dataset_asset.observation_count} observations")
    name = dataset_asset.display_name or "Imported dataset"
    return f"{name} — " + " • ".join(parts)


def format_dataset_import_summary(dataset_asset: DatasetAsset | None) -> str:
    """Return a compact description of the import settings in use."""
    if dataset_asset is None:
        return "Import settings — delimiter: comma • ignore: none"
    delimiter = "whitespace" if dataset_asset.treat_as_whitespace else repr(dataset_asset.separator)
    ignore_char = dataset_asset.ignore_char or "none"
    return f"Import settings — delimiter: {delimiter} • ignore: {ignore_char}"


def format_data_draft_status(has_unsaved_changes: bool) -> str:
    """Return helper text describing whether the current data inputs are saved."""
    return "Unsaved data import changes" if has_unsaved_changes else ""


def latest_successful_fit_run(project: Workspace) -> RunRecord | None:
    """Return the newest successful fit run for the active scenario."""
    for run in reversed(project.runs):
        if run.workflow == "fit" and run.status == RunStatus.SUCCEEDED:
            return run
    return None


def recommend_data_next_action(
    project: Workspace,
    *,
    has_unsaved_changes: bool,
) -> tuple[str, str, str] | None:
    """Return the primary handoff CTA for the Data workflow."""
    if has_unsaved_changes or project.active_dataset is None:
        return None

    active_dataset_path = (project.active_dataset.source_path or "").strip()
    saved_model = project.active_model_spec
    if saved_model is None:
        return ("Open Model", "model", "A dataset is ready; open Model to configure one next.")

    saved_model_dataset_path = (saved_model.dataset_path or "").strip()
    if active_dataset_path and saved_model_dataset_path != active_dataset_path:
        return (
            "Open Model",
            "model",
            "The active dataset changed — open Model to update the dataset path before fitting.",
        )

    if latest_successful_fit_run(project) is not None:
        return (
            "Open Results",
            "results",
            "A successful fit is already available. Review the latest outputs in Results.",
        )

    return ("Open Fit", "fit", "Dataset and saved model are ready for estimation.")


def format_example_dataset_option(example: ExampleDataset) -> str:
    """Return a readable option label for an example dataset."""
    if example.source_kind == "curated_csv":
        return f"{example.label} [curated CSV]"
    return example.label


def filter_example_datasets(
    example_datasets: list[ExampleDataset],
    filter_text: str,
) -> list[ExampleDataset]:
    """Return example datasets matching a user-entered filter string."""
    normalized_filter = " ".join(filter_text.lower().split())
    if not normalized_filter:
        return example_datasets

    filter_tokens = normalized_filter.split()
    filtered_examples: list[ExampleDataset] = []
    for example in example_datasets:
        haystack = " ".join(
            [
                example.label,
                example.description,
                example.key,
                example.category,
                example.route or "",
                example.difficulty,
                example.source_kind,
                example.source_license or "",
                example.source_url or "",
                example.readme_path or "",
                example.manifest_path,
                " ".join(example.tags),
                example.dataset_path,
            ]
        ).lower()
        if all(token in haystack for token in filter_tokens):
            filtered_examples.append(example)
    return filtered_examples


def format_example_dataset_contents(example: ExampleDataset) -> str:
    """Return a compact summary of the curated dataset source."""
    dataset_name = Path(example.dataset_path).name
    detail_parts = [f"Dataset: {dataset_name}"]
    if example.category:
        detail_parts.append(f"Category: {example.category}")
    if example.route:
        detail_parts.append(f"Route: {example.route}")
    detail_parts.append(f"Difficulty: {example.difficulty}")
    return " • ".join(detail_parts)


def format_example_dataset_details(example: ExampleDataset | None) -> str:
    """Return helper text describing the currently selected example dataset."""
    if example is None:
        return "Select a curated example dataset to preview its description and source before loading it."

    description_lines = [line.strip() for line in example.description.splitlines() if line.strip()]
    if description_lines and description_lines[0].rstrip(".") == example.label:
        description_lines = description_lines[1:]

    detail_parts = [example.label]
    detail_parts.append(format_example_dataset_contents(example))
    if description_lines:
        detail_parts.append(" ".join(description_lines))
    detail_parts.append(f"Source file: {example.dataset_path}")
    provenance_parts = [f"kind: {example.source_kind}"]
    if example.source_license:
        provenance_parts.append(f"license: {example.source_license}")
    if example.source_url:
        provenance_parts.append(f"url: {example.source_url}")
    detail_parts.append("Provenance: " + " • ".join(provenance_parts))
    if example.readme_path:
        detail_parts.append(f"Bundle notes: {example.readme_path}")
    if example.tags:
        detail_parts.append("Tags: " + ", ".join(example.tags))
    detail_parts.append(
        "Loads the catalog dataset directly so later Model/Fit steps can reuse the same file."
    )
    return "\n".join(detail_parts)


def _example_dataset_actions_html(example: ExampleDataset) -> str:
    """Return clickable action links for a curated dataset example."""
    actions = [
        file_link(example.dataset_path, label="Open dataset"),
        file_link(Path(example.manifest_path).resolve().parent, label="Open bundle folder"),
    ]
    if example.readme_path:
        actions.append(file_link(example.readme_path, label="Open bundle notes"))
    if example.source_url:
        actions.append(external_link(example.source_url, label="Open upstream source"))
    return " • ".join(actions)


def format_example_dataset_details_html(example: ExampleDataset | None) -> str:
    """Return rich-text helper text with clickable links for the selected example."""
    if example is None:
        return html.escape(format_example_dataset_details(None))

    description_lines = [line.strip() for line in example.description.splitlines() if line.strip()]
    if description_lines and description_lines[0].rstrip(".") == example.label:
        description_lines = description_lines[1:]

    detail_parts = [f"<b>{html.escape(example.label)}</b>"]
    detail_parts.append(html.escape(format_example_dataset_contents(example)))
    if description_lines:
        detail_parts.append(html.escape(" ".join(description_lines)))
    detail_parts.append(
        "Source file: "
        + file_link(example.dataset_path)
        + " • "
        + copy_link(example.dataset_path, label="Copy path")
    )
    provenance_parts = [f"kind: {html.escape(example.source_kind)}"]
    if example.source_license:
        provenance_parts.append(f"license: {html.escape(example.source_license)}")
    if example.source_url:
        provenance_parts.append(
            "url: "
            + external_link(example.source_url)
            + " • "
            + copy_link(example.source_url, label="Copy URL")
        )
    detail_parts.append("Provenance: " + " • ".join(provenance_parts))
    if example.readme_path:
        detail_parts.append(
            "Bundle notes: "
            + file_link(example.readme_path)
            + " • "
            + copy_link(example.readme_path, label="Copy path")
        )
    if example.tags:
        detail_parts.append("Tags: " + html.escape(", ".join(example.tags)))
    detail_parts.append(
        "Loads the catalog dataset directly so later Model/Fit steps can reuse the same file."
    )
    detail_parts.append("Actions: " + _example_dataset_actions_html(example))
    return "<br/>".join(detail_parts)


def format_example_dataset_hint(
    example_datasets: list[ExampleDataset],
    *,
    visible_count: int | None = None,
    filter_text: str = "",
) -> str:
    """Return helper text for the example-dataset selector."""
    total_count = len(example_datasets)
    if total_count == 0:
        return "No curated example datasets were found in the catalog."
    if visible_count is None:
        visible_count = total_count

    normalized_filter = filter_text.strip()
    if normalized_filter:
        if visible_count == 0:
            return f'No example datasets match "{normalized_filter}". Clear the filter to browse all {total_count} curated examples.'
        noun = "dataset" if visible_count == 1 else "datasets"
        return (
            f'Showing {visible_count} of {total_count} curated example {noun} for "{normalized_filter}". '
            "Inspect details before loading one."
        )

    noun = "dataset" if total_count == 1 else "datasets"
    return (
        f"Curated examples — choose from {total_count} {noun}, filter by keywords like route or method, "
        "and inspect details before loading one."
    )


def build_data_workflow(
    project: Workspace,
    dataset_service: DatasetService | None = None,
    project_service: ProjectService | None = None,
):
    """Build the first real workflow page for dataset import and review."""
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    dataset_service = dataset_service or DatasetService()
    project_service = project_service or ProjectService()

    root, _, layout, scroll_area = build_scrollable_page(
        qt_widgets, root_object_name="data-workflow"
    )
    active_options = dataset_service.options_from_asset(project.active_dataset)
    example_datasets = dataset_service.list_examples()
    example_datasets_by_key = {example.key: example for example in example_datasets}

    title_label = qt_widgets.QLabel("Data workflow")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 4)
    title_font.setBold(True)
    title_label.setFont(title_font)

    _hint_prefs = load_gui_preferences()

    def _save_dismissed_hint_data():
        save_gui_preferences(with_dismissed_hint(load_gui_preferences(), "hint_data"))

    hint_widget, _ = build_dismissible_hint(
        "Import a CSV dataset using the core NONMEMDataset loader. Required columns: "
        + ", ".join(dataset_service.required_columns())
        + ".",
        dismissed="hint_data" in _hint_prefs.dismissed_hints,
        on_dismiss=_save_dismissed_hint_data,
    )

    control_row_widget = qt_widgets.QWidget(root)
    control_row_widget.setObjectName("data-import-row")
    control_row = qt_widgets.QHBoxLayout(control_row_widget)
    control_row.setContentsMargins(0, 0, 0, 0)
    control_row.setSpacing(8)
    path_input = qt_widgets.QLineEdit(
        project.active_dataset.source_path if project.active_dataset else ""
    )
    path_input.setObjectName("data-source-path")
    path_input.setPlaceholderText("/path/to/dataset.csv")

    browse_button = qt_widgets.QPushButton("Import CSV…")
    browse_button.setObjectName("data-import-button")

    control_row.addWidget(path_input, 1)
    control_row.addWidget(browse_button)

    options_row_widget = qt_widgets.QWidget(root)
    options_row_widget.setObjectName("data-options-row")
    options_row = qt_widgets.QHBoxLayout(options_row_widget)
    options_row.setContentsMargins(0, 0, 0, 0)
    options_row.setSpacing(8)
    separator_input = qt_widgets.QLineEdit(active_options.separator)
    separator_input.setObjectName("data-separator-input")
    separator_input.setMaximumWidth(90)
    whitespace_checkbox = qt_widgets.QCheckBox("Whitespace-delimited")
    whitespace_checkbox.setObjectName("data-whitespace-checkbox")
    whitespace_checkbox.setChecked(active_options.treat_as_whitespace)
    ignore_char_input = qt_widgets.QLineEdit(active_options.ignore_char or "")
    ignore_char_input.setObjectName("data-ignore-char-input")
    ignore_char_input.setPlaceholderText("Optional")
    ignore_char_input.setMaximumWidth(90)
    separator_input.setEnabled(not active_options.treat_as_whitespace)

    options_row.addWidget(qt_widgets.QLabel("Separator"))
    options_row.addWidget(separator_input)
    options_row.addWidget(whitespace_checkbox)
    options_row.addSpacing(12)
    options_row.addWidget(qt_widgets.QLabel("IGNORE char"))
    options_row.addWidget(ignore_char_input)
    options_row.addStretch(1)

    summary_label = qt_widgets.QLabel(format_dataset_summary(project.active_dataset))
    summary_label.setObjectName("data-summary-label")
    summary_label.setWordWrap(True)
    options_label = qt_widgets.QLabel(format_dataset_import_summary(project.active_dataset))
    options_label.setObjectName("data-import-summary-label")
    options_label.setWordWrap(True)
    next_action_label = qt_widgets.QLabel("")
    next_action_label.setObjectName("data-next-action-label")
    next_action_label.setWordWrap(True)
    next_action_label.setVisible(False)
    next_action_button = qt_widgets.QPushButton("")
    next_action_button.setObjectName("data-next-action-button")
    next_action_button.setProperty("primaryAction", True)
    next_action_button.setMinimumHeight(36)
    next_action_button.setVisible(False)
    unsaved_label = qt_widgets.QLabel("")
    unsaved_label.setObjectName("data-unsaved-label")
    unsaved_label.setVisible(False)

    content_row_widget = qt_widgets.QSplitter(root)
    content_row_widget.setObjectName("data-content-row")
    content_row_widget.setChildrenCollapsible(False)
    content_row_widget.setHandleWidth(8)
    columns_list = qt_widgets.QListWidget()
    columns_list.setObjectName("data-columns-list")

    preview_table = qt_widgets.QTableWidget()
    preview_table.setObjectName("data-preview-table")
    preview_table.setEditTriggers(qt_widgets.QAbstractItemView.NoEditTriggers)
    preview_table.setAlternatingRowColors(True)
    preview_table.verticalHeader().setVisible(False)
    configure_resizable_table_columns(preview_table, qt_widgets)

    validation_list = qt_widgets.QListWidget()
    validation_list.setObjectName("data-validation-list")

    columns_panel = qt_widgets.QWidget(content_row_widget)
    columns_panel.setObjectName("data-columns-panel")
    columns_layout = qt_widgets.QVBoxLayout(columns_panel)
    columns_layout.setContentsMargins(12, 12, 12, 12)
    columns_layout.setSpacing(8)
    columns_layout.addWidget(qt_widgets.QLabel("Columns"))
    columns_layout.addWidget(columns_list, 1)

    preview_panel = qt_widgets.QWidget(content_row_widget)
    preview_panel.setObjectName("data-preview-panel")
    preview_layout = qt_widgets.QVBoxLayout(preview_panel)
    preview_layout.setContentsMargins(12, 12, 12, 12)
    preview_layout.setSpacing(8)
    preview_layout.addWidget(qt_widgets.QLabel("Preview"))
    preview_layout.addWidget(preview_table, 1)

    validation_panel = qt_widgets.QWidget(content_row_widget)
    validation_panel.setObjectName("data-validation-panel")
    validation_panel_layout = qt_widgets.QVBoxLayout(validation_panel)
    validation_panel_layout.setContentsMargins(12, 12, 12, 12)
    validation_panel_layout.setSpacing(8)
    validation_panel_layout.addWidget(qt_widgets.QLabel("Validation"))
    validation_panel_layout.addWidget(validation_list, 1)

    content_row_widget.addWidget(columns_panel)
    content_row_widget.addWidget(preview_panel)
    content_row_widget.addWidget(validation_panel)
    content_row_widget.setStretchFactor(0, 1)
    content_row_widget.setStretchFactor(1, 3)
    content_row_widget.setStretchFactor(2, 2)
    content_row_widget.setSizes([200, 600, 400])

    # Example datasets group — secondary, collapsed to the bottom
    example_group = qt_widgets.QGroupBox("Or use a built-in example dataset")
    example_group.setObjectName("data-example-group")
    example_group_layout = qt_widgets.QVBoxLayout(example_group)
    example_group_layout.setSpacing(8)

    example_row_widget = qt_widgets.QWidget(example_group)
    example_row_widget.setObjectName("data-example-row")
    example_row = qt_widgets.QHBoxLayout(example_row_widget)
    example_row.setContentsMargins(0, 0, 0, 0)
    example_row.setSpacing(8)
    example_selector = qt_widgets.QComboBox()
    example_selector.setObjectName("data-example-selector")
    example_selector.addItem("Choose an example dataset…", "")
    for example in example_datasets:
        example_selector.addItem(format_example_dataset_option(example), example.key)
        example_selector.setItemData(
            example_selector.count() - 1,
            format_example_dataset_details(example),
            qt_core.Qt.ItemDataRole.ToolTipRole,
        )
    example_selector.setEnabled(bool(example_datasets))
    example_button = qt_widgets.QPushButton("Load example")
    example_button.setObjectName("data-load-example-button")
    example_button.setEnabled(False)
    example_row.addWidget(example_selector, 1)
    example_row.addWidget(example_button)

    example_details_label = qt_widgets.QLabel(
        format_example_dataset_details_html(None)
        if example_datasets
        else "No example details available because no built-in examples were discovered."
    )
    example_details_label.setObjectName("data-example-details")
    example_details_label.setOpenExternalLinks(False)
    example_details_label.setTextInteractionFlags(
        qt_core.Qt.TextInteractionFlag.TextBrowserInteraction
    )
    example_details_label.setWordWrap(True)

    example_group_layout.addWidget(example_row_widget)
    example_group_layout.addWidget(example_details_label)

    layout.addWidget(title_label)
    layout.addWidget(hint_widget)
    layout.addWidget(control_row_widget)
    layout.addWidget(options_row_widget)
    layout.addWidget(unsaved_label)
    layout.addWidget(summary_label)
    layout.addWidget(options_label)
    layout.addWidget(next_action_label)
    layout.addWidget(next_action_button)
    layout.addWidget(content_row_widget, 1)
    layout.addWidget(example_group)

    _apply_responsive_box_layout = install_responsive_box_layouts(
        root,
        breakpoint=DATA_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        layouts=(example_row, control_row, options_row),
    )

    _apply_responsive_splitter = install_responsive_splitters(
        root,
        breakpoint=DATA_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        splitters=(content_row_widget,),
    )

    def _apply_responsive_layout(width: int | None = None) -> None:
        _apply_responsive_box_layout(width)
        _apply_responsive_splitter(width)

    last_synced_project_state: dict[str, object] | None = None
    last_synced_control_state: dict[str, object] | None = None
    last_reported_dirty_state = False
    next_action_target = [""]

    def _populate_columns(dataset_asset: DatasetAsset | None) -> None:
        columns_list.clear()
        for column in dataset_asset.columns if dataset_asset is not None else []:
            columns_list.addItem(column)

    def _populate_preview(dataset_asset: DatasetAsset | None) -> None:
        rows = dataset_asset.preview_rows if dataset_asset is not None else []
        headers = (
            list(rows[0].keys()) if rows else list(dataset_asset.columns if dataset_asset else [])
        )
        preview_table.clear()
        preview_table.setColumnCount(len(headers))
        preview_table.setHorizontalHeaderLabels(headers)
        configure_resizable_table_columns(preview_table, qt_widgets)
        preview_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(headers):
                value = row.get(column)
                preview_table.setItem(
                    row_index,
                    column_index,
                    qt_widgets.QTableWidgetItem("" if value is None else str(value)),
                )
        apply_saved_table_column_widths(preview_table)

    def _render(dataset_asset: DatasetAsset | None, validation: ValidationResult) -> None:
        summary_label.setText(format_dataset_summary(dataset_asset))
        options_label.setText(format_dataset_import_summary(dataset_asset))
        _populate_columns(dataset_asset)
        _populate_preview(dataset_asset)
        validation_list.clear()
        if dataset_asset is None:
            return
        if not validation.issues:
            validation_list.addItem("Dataset looks valid.")
            return
        for issue in validation.issues:
            prefix = "Error" if issue.severity == ValidationSeverity.ERROR else "Warning"
            field_suffix = f" [{issue.field_name}]" if issue.field_name else ""
            validation_list.addItem(f"{prefix}{field_suffix}: {issue.message}")

    def _project_control_state() -> dict[str, object]:
        dataset_asset = project.active_dataset
        options = dataset_service.options_from_asset(dataset_asset)
        return {
            "source_path": dataset_asset.source_path
            if dataset_asset and dataset_asset.source_path
            else "",
            "separator": options.separator,
            "treat_as_whitespace": options.treat_as_whitespace,
            "ignore_char": options.ignore_char or "",
        }

    def _current_control_state() -> dict[str, object]:
        return {
            "source_path": path_input.text(),
            "separator": separator_input.text(),
            "treat_as_whitespace": whitespace_checkbox.isChecked(),
            "ignore_char": ignore_char_input.text(),
        }

    def _sync_from_project() -> None:
        nonlocal last_synced_project_state, last_synced_control_state
        project_state = _project_control_state()
        path_input.setText(str(project_state["source_path"]))
        separator_input.setText(str(project_state["separator"]))
        whitespace_checkbox.setChecked(bool(project_state["treat_as_whitespace"]))
        ignore_char_input.setText(str(project_state["ignore_char"]))
        separator_input.setEnabled(not whitespace_checkbox.isChecked())
        _render(
            project.active_dataset, dataset_service.validation_from_asset(project.active_dataset)
        )
        last_synced_project_state = project_state
        last_synced_control_state = _current_control_state()
        _update_unsaved_indicator(notify_project=False)

    def _refresh_from_project_if_pristine() -> None:
        project_state = _project_control_state()
        if (
            last_synced_control_state is not None
            and _current_control_state() != last_synced_control_state
        ):
            return
        if project_state != last_synced_project_state:
            _sync_from_project()
            return
        _refresh_next_action()

    def _has_unsaved_changes() -> bool:
        return (
            last_synced_control_state is not None
            and _current_control_state() != last_synced_control_state
        )

    def _notify_project_state_changed() -> None:
        callback = getattr(root, "_project_state_changed", None)
        if callable(callback):
            callback()

    def _refresh_next_action() -> None:
        action = recommend_data_next_action(project, has_unsaved_changes=_has_unsaved_changes())
        if action is None:
            next_action_target[0] = ""
            next_action_label.clear()
            next_action_label.setVisible(False)
            next_action_button.setText("")
            next_action_button.setToolTip("")
            next_action_button.setVisible(False)
            return
        button_text, workflow_id, summary = action
        next_action_target[0] = workflow_id
        next_action_label.setText(summary)
        next_action_label.setVisible(True)
        next_action_button.setText(button_text)
        next_action_button.setToolTip(summary)
        next_action_button.setVisible(True)

    def _navigate_to_next_action() -> None:
        target = next_action_target[0]
        if not target:
            return
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(target)

    def _update_unsaved_indicator(*, notify_project: bool = True) -> None:
        nonlocal last_reported_dirty_state
        dirty = _has_unsaved_changes()
        unsaved_label.setText(format_data_draft_status(dirty))
        unsaved_label.setProperty("warningText", dirty)
        unsaved_label.setVisible(dirty)
        unsaved_label.style().unpolish(unsaved_label)
        unsaved_label.style().polish(unsaved_label)
        _refresh_next_action()
        if notify_project and dirty != last_reported_dirty_state:
            _notify_project_state_changed()
        last_reported_dirty_state = dirty

    def _collect_import_options() -> DatasetImportOptions:
        return DatasetImportOptions(
            separator=separator_input.text(),
            treat_as_whitespace=whitespace_checkbox.isChecked(),
            ignore_char=ignore_char_input.text(),
        )

    def _load_from_path() -> None:
        result = dataset_service.load_csv(path_input.text(), options=_collect_import_options())
        if result.dataset_asset is not None:
            project_service.attach_dataset(project, result.dataset_asset)
            path_input.setText(result.dataset_asset.source_path or path_input.text())
            _sync_from_project()
            _notify_project_state_changed()
            return
        dataset_asset = (
            project.active_dataset if result.dataset_asset is None else result.dataset_asset
        )
        _render(dataset_asset, result.validation)

    def _reload_if_path_set() -> None:
        if path_input.text().strip():
            _load_from_path()

    def _browse_for_file() -> None:
        preferences = load_gui_preferences()
        start_dir = preferences.last_file_dialog_dir or project.root_path or ""
        selected_path, _ = qt_widgets.QFileDialog.getOpenFileName(
            root,
            "Select NONMEM dataset",
            start_dir,
            "Delimited data (*.csv *.txt);;All files (*)",
        )
        if not selected_path:
            return
        save_gui_preferences(with_last_file_dialog_dir(preferences, selected_path))
        path_input.setText(selected_path)
        _load_from_path()

    def _load_selected_example() -> None:
        example_key = example_selector.currentData()
        if not example_key:
            return
        separator_input.setText(",")
        whitespace_checkbox.setChecked(False)
        ignore_char_input.setText("")
        result = dataset_service.load_example(str(example_key), options=_collect_import_options())
        if result.dataset_asset is not None:
            project_service.attach_dataset(project, result.dataset_asset)
            path_input.setText(result.dataset_asset.source_path or "")
            _sync_from_project()
            _notify_project_state_changed()
            return
        dataset_asset = (
            project.active_dataset if result.dataset_asset is None else result.dataset_asset
        )
        _render(dataset_asset, result.validation)

    def _update_example_selection() -> None:
        example_key = example_selector.currentData()
        selected_example = example_datasets_by_key.get(str(example_key)) if example_key else None
        example_button.setEnabled(selected_example is not None)
        example_button.setToolTip(
            f"Load {selected_example.label}"
            if selected_example is not None
            else "Select an example first"
        )
        if selected_example is not None:
            example_details_label.setText(format_example_dataset_details_html(selected_example))
        elif example_datasets:
            example_details_label.setText(format_example_dataset_details_html(None))
        else:
            example_details_label.setText(
                "No example details available because no built-in examples were discovered."
            )

    def _handle_example_details_link(href: str) -> None:
        copy_target = decode_copy_target(href)
        if copy_target is not None:
            clipboard = qt_widgets.QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(copy_target)
            return
        qt_gui.QDesktopServices.openUrl(qt_core.QUrl(href))

    browse_button.clicked.connect(_browse_for_file)
    example_button.clicked.connect(_load_selected_example)
    example_selector.currentIndexChanged.connect(lambda _index: _update_example_selection())
    example_details_label.linkActivated.connect(_handle_example_details_link)
    whitespace_checkbox.toggled.connect(lambda checked: separator_input.setEnabled(not checked))
    path_input.textChanged.connect(lambda _text: _update_unsaved_indicator())
    separator_input.textChanged.connect(lambda _text: _update_unsaved_indicator())
    ignore_char_input.textChanged.connect(lambda _text: _update_unsaved_indicator())
    path_input.editingFinished.connect(_load_from_path)
    whitespace_checkbox.toggled.connect(lambda _checked: _reload_if_path_set())
    separator_input.editingFinished.connect(_reload_if_path_set)
    ignore_char_input.editingFinished.connect(_reload_if_path_set)
    next_action_button.clicked.connect(_navigate_to_next_action)
    load_shortcut = qt_gui.QShortcut(qt_gui.QKeySequence("Ctrl+L"), root)
    load_shortcut.activated.connect(_load_from_path)
    root._apply_responsive_layout = _apply_responsive_layout  # type: ignore[attr-defined]
    root._has_unsaved_changes = _has_unsaved_changes  # type: ignore[attr-defined]
    root._refresh_workflow = _refresh_from_project_if_pristine  # type: ignore[attr-defined]
    root._load_project = _sync_from_project  # type: ignore[attr-defined]

    _update_example_selection()
    _sync_from_project()
    _apply_responsive_layout()
    return root
