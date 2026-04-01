"""Deferred-import Qt shell for the OpenPKPD desktop GUI."""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from platformdirs import user_data_dir

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
    GuiPreferences,
    apply_gui_preferences,
    apply_saved_table_column_widths,
    default_font_point_size,
    default_workspace_root_path,
    initialize_gui_preferences,
    load_gui_preferences,
    normalize_directory_path,
    save_gui_preferences,
    with_last_file_dialog_dir,
)
from openpkpd_gui.domain.workspace import Project, Workspace
from openpkpd_gui.services.bootstrap_service import BootstrapService
from openpkpd_gui.services.design_service import DesignService
from openpkpd_gui.services.fit_service import FitService
from openpkpd_gui.services.npde_service import NPDEService
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.report_export_service import ReportExportService
from openpkpd_gui.services.serialization_service import ProjectSnapshotService
from openpkpd_gui.services.vpc_service import VPCService
from openpkpd_gui.services.workflow_state_service import WorkflowStateId, workflow_state_for
from openpkpd_gui.shell.help_browser import get_app_metadata, open_about_dialog, open_help_dialog
from openpkpd_gui.workflows.advanced_workflow import build_advanced_workflow
from openpkpd_gui.workflows.covariate_workflow import build_covariate_workflow
from openpkpd_gui.workflows.dashboard_workflow import build_dashboard_workflow
from openpkpd_gui.workflows.data_workflow import build_data_workflow
from openpkpd_gui.workflows.diagnostics_workflow import build_diagnostics_workflow
from openpkpd_gui.workflows.fit_workflow import build_fit_workflow
from openpkpd_gui.workflows.model_workflow import build_model_workflow
from openpkpd_gui.workflows.nca_workflow import build_nca_workflow
from openpkpd_gui.workflows.registry import DEFAULT_WORKFLOWS
from openpkpd_gui.workflows.results_workflow import build_results_workflow

_PERSISTED_LIST_SELECTION_ROLE = 0x0100


def _default_snapshot_name(project: Workspace) -> str:
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in project.name.strip()
    ).strip("-")
    return f"{safe_name or 'openpkpd-project'}.opkpd"


def _normalize_snapshot_path(selected_path: str) -> str:
    selected = Path(selected_path)
    suffix = selected.suffix.lower()
    if suffix in {".opkpd", ".zip"}:
        return str(selected)
    if suffix == ".pkp":
        return str(selected.with_suffix(".opkpd"))
    return f"{selected_path}.opkpd"


def _replace_project_contents(target: Workspace, source: Workspace) -> None:
    target.name = source.name
    target.root_path = source.root_path
    target.workspace_id = source.workspace_id
    target.created_at = source.created_at
    target.updated_at = source.updated_at
    target.recent_files = list(source.recent_files)
    target.projects = [Project.from_dict(project.to_dict()) for project in source.projects]
    target.active_project_id = source.active_project_id
    target.metadata = dict(source.metadata)


def _metadata_text(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    return value.strip() if isinstance(value, str) else ""


def _notes_from_metadata(metadata: dict[str, object]) -> str:
    return _metadata_text(metadata, "notes")


def _tooltip_with_sections(base_text: str, *, sections: tuple[tuple[str, str], ...]) -> str:
    blocks = [base_text]
    for label, value in sections:
        if value:
            blocks.append(f"{label}:\n{value}")
    return "\n\n".join(blocks)


def _notes_tooltip(base_text: str, metadata: dict[str, object]) -> str:
    return _tooltip_with_sections(base_text, sections=(("Notes", _notes_from_metadata(metadata)),))


def _compute_default_window_bounds(
    available_x: int,
    available_y: int,
    available_width: int,
    available_height: int,
) -> tuple[int, int, int, int]:
    """Return a sensible first-launch geometry for the available screen area."""
    padding = 24
    max_width = max(320, available_width - padding)
    max_height = max(240, available_height - padding)
    target_width = min(max_width, max(960, min(1920, int(available_width * 0.9))))
    target_height = min(max_height, max(700, min(1200, int(available_height * 0.9))))
    centered_x = available_x + max(0, (available_width - target_width) // 2)
    centered_y = available_y + max(0, (available_height - target_height) // 2)
    return centered_x, centered_y, target_width, target_height


def _compute_default_splitter_sizes(total_width: int) -> tuple[int, int]:
    """Return sensible default sidebar/content widths for the shell splitter."""
    usable_width = max(900, total_width)
    sidebar_width = min(max(280, int(usable_width * 0.24)), 420)
    content_width = max(620, usable_width - sidebar_width)
    return sidebar_width, content_width


def _apply_saved_or_default_window_geometry(
    window, qt_core, qt_gui, preferences: GuiPreferences
) -> None:
    """Restore the saved window geometry when possible, otherwise use screen-aware defaults."""
    saved_x = preferences.window_x
    saved_y = preferences.window_y
    saved_width = preferences.window_width
    saved_height = preferences.window_height
    if None not in {saved_x, saved_y, saved_width, saved_height}:
        center_point = qt_core.QPoint(saved_x + saved_width // 2, saved_y + saved_height // 2)
        screen = qt_gui.QGuiApplication.screenAt(center_point)
        if screen is None:
            screen = window.screen() or qt_gui.QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            target_width = max(320, min(saved_width, available.width()))
            target_height = max(240, min(saved_height, available.height()))
            target_x = min(
                max(saved_x, available.x()), available.x() + available.width() - target_width
            )
            target_y = min(
                max(saved_y, available.y()), available.y() + available.height() - target_height
            )
            window.setGeometry(target_x, target_y, target_width, target_height)
            if preferences.window_maximized:
                window.setWindowState(window.windowState() | qt_core.Qt.WindowState.WindowMaximized)
            return

    screen = window.screen() or qt_gui.QGuiApplication.primaryScreen()
    if screen is None:
        window.resize(1280, 860)
        return
    available = screen.availableGeometry()
    target_x, target_y, target_width, target_height = _compute_default_window_bounds(
        available.x(),
        available.y(),
        available.width(),
        available.height(),
    )
    window.setGeometry(target_x, target_y, target_width, target_height)


def _capture_window_geometry_preferences(window, preferences: GuiPreferences) -> GuiPreferences:
    """Return preferences updated with the window's current normal geometry."""
    geometry = window.normalGeometry() if window.isMaximized() else window.geometry()
    if not geometry.isValid():
        geometry = window.geometry()
    return replace(
        preferences,
        window_x=geometry.x(),
        window_y=geometry.y(),
        window_width=geometry.width(),
        window_height=geometry.height(),
        window_maximized=window.isMaximized(),
    )


def _apply_saved_or_default_splitter_sizes(
    splitter, preferences: GuiPreferences, *, total_width: int
) -> None:
    """Restore saved shell splitter sizes or apply a screen-aware default split."""
    sizes = preferences.window_splitter_sizes
    if len(sizes) >= 2 and all(size > 0 for size in sizes[:2]):
        splitter.setSizes([sizes[0], sizes[1]])
        return
    sidebar_width, content_width = _compute_default_splitter_sizes(total_width)
    splitter.setSizes([sidebar_width, content_width])


def _capture_shell_layout_preferences(
    window, splitter, preferences: GuiPreferences
) -> GuiPreferences:
    """Return preferences updated with the current window geometry and splitter layout."""
    window_preferences = _capture_window_geometry_preferences(window, preferences)
    return replace(
        window_preferences,
        window_splitter_sizes=tuple(size for size in splitter.sizes() if size > 0),
    )


def _capture_named_table_column_width_preferences(
    root, qt_widgets, preferences: GuiPreferences
) -> GuiPreferences:
    """Return preferences updated with current widths for named QTableWidget descendants."""
    column_widths = dict(preferences.table_column_widths)
    for table in root.findChildren(qt_widgets.QTableWidget):
        object_name = table.objectName()
        if not object_name or table.columnCount() <= 0:
            continue
        widths = tuple(
            max(1, int(table.columnWidth(index))) for index in range(table.columnCount())
        )
        if widths:
            column_widths[str(object_name)] = widths
    return replace(preferences, table_column_widths=column_widths)


def _apply_saved_table_column_widths_to_root(root, qt_widgets, *, settings_store=None) -> None:
    """Restore saved column widths for named tables already present under *root*."""
    for table in root.findChildren(qt_widgets.QTableWidget):
        if table.objectName():
            apply_saved_table_column_widths(table, settings_store=settings_store)


def _capture_named_combo_box_preferences(
    root, qt_widgets, preferences: GuiPreferences
) -> GuiPreferences:
    """Return preferences updated with current text for opted-in named combo boxes."""
    combo_selections = dict(preferences.combo_box_selections)
    for combo_box in root.findChildren(qt_widgets.QComboBox):
        object_name = combo_box.objectName()
        if not object_name or not bool(combo_box.property("persistComboSelection")):
            continue
        current_text = str(combo_box.currentText()).strip()
        if current_text:
            combo_selections[str(object_name)] = current_text
    return replace(preferences, combo_box_selections=combo_selections)


def _capture_persisted_button_group_preferences(
    root, qt_widgets, preferences: GuiPreferences
) -> GuiPreferences:
    """Return preferences updated with checked opted-in button-group selections."""
    button_group_selections = dict(preferences.button_group_selections)
    for button in root.findChildren(qt_widgets.QAbstractButton):
        group_name = button.property("persistButtonGroupName")
        object_name = button.objectName()
        if (
            not isinstance(group_name, str)
            or not group_name
            or not object_name
            or not button.isChecked()
        ):
            continue
        button_group_selections[group_name] = str(object_name)
    return replace(preferences, button_group_selections=button_group_selections)


def _apply_saved_button_group_selections_to_root(
    root, qt_widgets, preferences: GuiPreferences
) -> None:
    """Restore checked state for opted-in checkable button groups."""
    for button in root.findChildren(qt_widgets.QAbstractButton):
        group_name = button.property("persistButtonGroupName")
        object_name = button.objectName()
        if not isinstance(group_name, str) or not group_name or not object_name:
            continue
        if preferences.button_group_selections.get(group_name) == str(object_name):
            button.setChecked(True)


def _apply_saved_combo_box_selections_to_root(
    root, qt_widgets, preferences: GuiPreferences
) -> None:
    """Restore saved current-text selections for opted-in named combo boxes."""
    for combo_box in root.findChildren(qt_widgets.QComboBox):
        object_name = combo_box.objectName()
        if not object_name or not bool(combo_box.property("persistComboSelection")):
            continue
        saved_text = preferences.combo_box_selections.get(str(object_name))
        if saved_text is None:
            continue
        index = combo_box.findText(saved_text)
        if index >= 0:
            combo_box.setCurrentIndex(index)


def _capture_list_widget_selection_preferences(
    root, qt_widgets, preferences: GuiPreferences
) -> GuiPreferences:
    """Return preferences updated with current selections for opted-in list widgets."""
    list_widget_selections = dict(preferences.list_widget_selections)
    for list_widget in root.findChildren(qt_widgets.QListWidget):
        object_name = list_widget.objectName()
        if not object_name or not bool(list_widget.property("persistListSelection")):
            continue
        item = list_widget.currentItem()
        if item is None:
            continue
        item_key = item.data(_PERSISTED_LIST_SELECTION_ROLE)
        if isinstance(item_key, str) and item_key:
            list_widget_selections[str(object_name)] = item_key
    return replace(preferences, list_widget_selections=list_widget_selections)


def _apply_saved_list_widget_selections_to_root(
    root, qt_widgets, preferences: GuiPreferences
) -> None:
    """Restore current selections for opted-in list widgets by stable item key."""
    for _attempt in range(2):
        for list_widget in root.findChildren(qt_widgets.QListWidget):
            object_name = list_widget.objectName()
            if not object_name or not bool(list_widget.property("persistListSelection")):
                continue
            saved_item_key = preferences.list_widget_selections.get(str(object_name))
            if saved_item_key is None:
                continue
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                if item is None:
                    continue
                item_key = item.data(_PERSISTED_LIST_SELECTION_ROLE)
                if isinstance(item_key, str) and item_key == saved_item_key:
                    list_widget.setCurrentRow(index)
                    break


def _capture_named_tab_selection_preferences(
    root, qt_widgets, preferences: GuiPreferences
) -> GuiPreferences:
    """Return preferences updated with selected indexes for named tab widgets."""
    tab_selections = dict(preferences.tab_selections)
    for tab_widget in root.findChildren(qt_widgets.QTabWidget):
        object_name = tab_widget.objectName()
        if not object_name or tab_widget.count() <= 0:
            continue
        current_index = int(tab_widget.currentIndex())
        if current_index >= 0:
            tab_selections[str(object_name)] = current_index
    return replace(preferences, tab_selections=tab_selections)


def _apply_saved_tab_selections_to_root(root, qt_widgets, preferences: GuiPreferences) -> None:
    """Restore saved selections for named tab widgets already present under *root*."""
    for tab_widget in root.findChildren(qt_widgets.QTabWidget):
        object_name = tab_widget.objectName()
        if not object_name:
            continue
        saved_index = preferences.tab_selections.get(str(object_name))
        if saved_index is None:
            continue
        if 0 <= int(saved_index) < tab_widget.count():
            tab_widget.setCurrentIndex(int(saved_index))


def _capture_collapsible_section_preferences(
    root, qt_widgets, preferences: GuiPreferences
) -> GuiPreferences:
    """Return preferences updated with current expanded state for collapsible sections."""
    section_states = dict(preferences.collapsible_section_states)
    for section in root.findChildren(qt_widgets.QWidget):
        object_name = section.objectName()
        if not object_name or not bool(section.property("collapsibleSection")):
            continue
        section_states[str(object_name)] = bool(section.property("expanded"))
    return replace(preferences, collapsible_section_states=section_states)


def _apply_saved_collapsible_section_states_to_root(
    root, qt_widgets, preferences: GuiPreferences
) -> None:
    """Restore saved expanded/collapsed states for named collapsible sections."""
    for section in root.findChildren(qt_widgets.QWidget):
        object_name = section.objectName()
        if not object_name or not bool(section.property("collapsibleSection")):
            continue
        saved_state = preferences.collapsible_section_states.get(str(object_name))
        if saved_state is None:
            continue
        setter = getattr(section, "_set_expanded", None)
        if callable(setter):
            setter(bool(saved_state))


def _summary_text(label: str, text: str, *, max_length: int = 60) -> str | None:
    normalized = text.strip()
    if not normalized:
        return None
    first_line = next(
        (line.strip() for line in normalized.splitlines() if line.strip()), normalized
    )
    summary = (
        first_line if len(first_line) <= max_length else f"{first_line[: max_length - 1].rstrip()}…"
    )
    if "\n" in normalized:
        summary = f"{summary} …"
    return f"{label}: {summary}"


def _notes_summary(label: str, metadata: dict[str, object], *, max_length: int = 60) -> str | None:
    return _summary_text(label, _notes_from_metadata(metadata), max_length=max_length)


def _metadata_summary(
    label: str, metadata: dict[str, object], *, key: str, max_length: int = 60
) -> str | None:
    return _summary_text(label, _metadata_text(metadata, key), max_length=max_length)


def _project_tooltip(project: Project) -> str:
    return _tooltip_with_sections(
        f"Project: {project.name}",
        sections=(
            ("Description", _metadata_text(project.metadata, "description")),
            ("References", _metadata_text(project.metadata, "references")),
            ("Notes", _notes_from_metadata(project.metadata)),
        ),
    )


def _prompt_for_project_details(
    parent,
    qt_widgets,
    *,
    name: str,
    description: str,
    references: str,
    notes: str,
) -> dict[str, str] | None:
    dialog = qt_widgets.QDialog(parent)
    dialog.setWindowTitle("Edit Project Details")
    dialog.setObjectName("project-details-dialog")

    layout = qt_widgets.QVBoxLayout(dialog)
    form = qt_widgets.QFormLayout()

    name_input = qt_widgets.QLineEdit(name)
    name_input.setObjectName("project-details-name-input")
    description_input = qt_widgets.QPlainTextEdit(description)
    description_input.setObjectName("project-details-description-input")
    references_input = qt_widgets.QPlainTextEdit(references)
    references_input.setObjectName("project-details-references-input")
    notes_input = qt_widgets.QPlainTextEdit(notes)
    notes_input.setObjectName("project-details-notes-input")

    for text_edit in (description_input, references_input, notes_input):
        text_edit.setMinimumHeight(72)

    form.addRow("Project name:", name_input)
    form.addRow("Description:", description_input)
    form.addRow("References:", references_input)
    form.addRow("Notes:", notes_input)
    layout.addLayout(form)

    buttons = qt_widgets.QDialogButtonBox(
        qt_widgets.QDialogButtonBox.StandardButton.Ok
        | qt_widgets.QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    name_input.selectAll()
    name_input.setFocus()

    if dialog.exec() != int(qt_widgets.QDialog.DialogCode.Accepted):
        return None
    return {
        "name": name_input.text(),
        "description": description_input.toPlainText(),
        "references": references_input.toPlainText(),
        "notes": notes_input.toPlainText(),
    }


def _prompt_for_scenario_details(
    parent,
    qt_widgets,
    *,
    name: str,
    description: str,
    references: str,
    notes: str,
) -> dict[str, str] | None:
    dialog = qt_widgets.QDialog(parent)
    dialog.setWindowTitle("Edit Scenario Details")
    dialog.setObjectName("scenario-details-dialog")

    layout = qt_widgets.QVBoxLayout(dialog)
    form = qt_widgets.QFormLayout()

    name_input = qt_widgets.QLineEdit(name)
    name_input.setObjectName("scenario-details-name-input")
    description_input = qt_widgets.QPlainTextEdit(description)
    description_input.setObjectName("scenario-details-description-input")
    references_input = qt_widgets.QPlainTextEdit(references)
    references_input.setObjectName("scenario-details-references-input")
    notes_input = qt_widgets.QPlainTextEdit(notes)
    notes_input.setObjectName("scenario-details-notes-input")

    for text_edit in (description_input, references_input, notes_input):
        text_edit.setMinimumHeight(72)

    form.addRow("Scenario name:", name_input)
    form.addRow("Description:", description_input)
    form.addRow("References:", references_input)
    form.addRow("Notes:", notes_input)
    layout.addLayout(form)

    buttons = qt_widgets.QDialogButtonBox(
        qt_widgets.QDialogButtonBox.StandardButton.Ok
        | qt_widgets.QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    name_input.selectAll()
    name_input.setFocus()

    if dialog.exec() != int(qt_widgets.QDialog.DialogCode.Accepted):
        return None
    return {
        "name": name_input.text(),
        "description": description_input.toPlainText(),
        "references": references_input.toPlainText(),
        "notes": notes_input.toPlainText(),
    }


def _parent_scenario_summary(project: Project, scenario) -> str | None:
    if scenario.parent_scenario_id is None:
        return None
    parent = project.find_scenario(scenario.parent_scenario_id)
    if parent is None:
        return "Parent scenario: Unknown"
    return f"Parent scenario: {parent.name}"


def _scenario_tooltip(project: Project, scenario) -> str:
    lines = [f"Scenario: {scenario.name}"]
    parent_summary = _parent_scenario_summary(project, scenario)
    if parent_summary is not None:
        lines.append(parent_summary)
    return _tooltip_with_sections(
        "\n".join(lines),
        sections=(
            ("Description", _metadata_text(scenario.metadata, "description")),
            ("References", _metadata_text(scenario.metadata, "references")),
            ("Notes", _notes_from_metadata(scenario.metadata)),
        ),
    )


def _build_details_editor_page(
    qt_widgets,
    *,
    page_object_name: str,
    field_prefix: str,
    title: str,
    intro: str,
    context_label_text: str,
    name_label: str,
    save_button_text: str,
):
    root = qt_widgets.QWidget()
    root.setObjectName(page_object_name)
    layout = qt_widgets.QVBoxLayout(root)

    title_label = qt_widgets.QLabel(title)
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 4)
    title_font.setBold(True)
    title_label.setFont(title_font)

    intro_label = qt_widgets.QLabel(intro)
    intro_label.setWordWrap(True)

    context_label = qt_widgets.QLabel(context_label_text)
    context_label.setObjectName(f"{field_prefix}-context-label")
    context_label.setWordWrap(True)

    form = qt_widgets.QFormLayout()
    name_input = qt_widgets.QLineEdit()
    name_input.setObjectName(f"{field_prefix}-name-input")
    description_input = qt_widgets.QPlainTextEdit()
    description_input.setObjectName(f"{field_prefix}-description-input")
    references_input = qt_widgets.QPlainTextEdit()
    references_input.setObjectName(f"{field_prefix}-references-input")
    notes_input = qt_widgets.QPlainTextEdit()
    notes_input.setObjectName(f"{field_prefix}-notes-input")
    for text_edit in (description_input, references_input, notes_input):
        text_edit.setMinimumHeight(96)

    form.addRow(name_label, name_input)
    form.addRow("Description:", description_input)
    form.addRow("References:", references_input)
    form.addRow("Notes:", notes_input)

    button_row = qt_widgets.QHBoxLayout()
    save_button = qt_widgets.QPushButton(save_button_text)
    save_button.setObjectName(f"{field_prefix}-save-button")
    save_button.setProperty("primaryAction", True)
    save_button.setEnabled(False)
    button_row.addStretch(1)
    button_row.addWidget(save_button)

    status_label = qt_widgets.QLabel("No changes to save.")
    status_label.setObjectName(f"{field_prefix}-status-label")
    status_label.setWordWrap(True)

    layout.addWidget(title_label)
    layout.addWidget(intro_label)
    layout.addWidget(context_label)
    layout.addLayout(form)
    layout.addLayout(button_row)
    layout.addWidget(status_label)
    layout.addStretch(1)

    return root, {
        "context_label": context_label,
        "name_input": name_input,
        "description_input": description_input,
        "references_input": references_input,
        "notes_input": notes_input,
        "save_button": save_button,
        "status_label": status_label,
    }


def create_main_window(
    project: Workspace,
    *,
    snapshot_service: ProjectSnapshotService | None = None,
    settings_store=None,
):
    """Build and return the main Qt window for the desktop GUI."""
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    snapshot_service = snapshot_service or ProjectSnapshotService()
    project_service = ProjectService()
    fit_service = FitService()
    npde_service = NPDEService()
    vpc_service = VPCService()
    bootstrap_service = BootstrapService()
    design_service = DesignService()
    report_export_service = ReportExportService()

    current_preferences = [GuiPreferences()]
    workspace_root_from_preferences = [False]
    update_window_title_active = [False]

    def _apply_default_workspace_root(preferences: GuiPreferences) -> bool:
        previous_default_root = normalize_directory_path(
            current_preferences[0].default_workspace_root
        )
        next_default_root = normalize_directory_path(preferences.default_workspace_root)
        current_preferences[0] = preferences

        if next_default_root is None:
            if workspace_root_from_preferences[0] and project.root_path == previous_default_root:
                project.root_path = None
                workspace_root_from_preferences[0] = False
                return True
            return False

        if project.root_path is None or (
            workspace_root_from_preferences[0] and project.root_path == previous_default_root
        ):
            project.root_path = next_default_root
            workspace_root_from_preferences[0] = True
            return True
        return False

    def _preferred_dialog_directory(*fallback_directories: str | Path | None) -> Path:
        persisted_last_dir = load_gui_preferences(
            settings_store=settings_store
        ).last_file_dialog_dir
        candidates: tuple[str | Path | None, ...] = (
            persisted_last_dir,
            current_preferences[0].last_file_dialog_dir,
            *fallback_directories,
            project.root_path,
            default_workspace_root_path(),
        )
        for candidate in candidates:
            normalized = normalize_directory_path(candidate)
            if normalized is not None:
                return Path(normalized)
        return default_workspace_root_path()

    def _remember_last_dialog_selection(
        selected_path: str | Path, *, selection_is_directory: bool = False
    ) -> None:
        updated = with_last_file_dialog_dir(
            current_preferences[0],
            selected_path,
            selection_is_directory=selection_is_directory,
        )
        if updated == current_preferences[0]:
            return
        current_preferences[0] = updated
        save_gui_preferences(updated, settings_store=settings_store)

    def _fallback_workspace_root() -> Path:
        configured_root = normalize_directory_path(current_preferences[0].default_workspace_root)
        if configured_root:
            return Path(configured_root).resolve()
        if project.root_path:
            return Path(project.root_path).resolve()
        return default_workspace_root_path()

    _app_version = get_app_metadata()["version"]
    app = qt_widgets.QApplication.instance()
    if app is not None:
        app.setApplicationName("OpenPKPD")
        app.setApplicationVersion(_app_version)
        app.setOrganizationName("OpenPKPD")
        current_preferences[0] = initialize_gui_preferences(app, settings_store=settings_store)
    _apply_default_workspace_root(current_preferences[0])

    window = qt_widgets.QMainWindow()
    window.setObjectName("main-window")
    window.menuBar().setObjectName("main-menu-bar")

    root = qt_widgets.QWidget()
    layout = qt_widgets.QHBoxLayout(root)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(12)

    shell_splitter = qt_widgets.QSplitter(qt_core.Qt.Orientation.Horizontal, root)
    shell_splitter.setObjectName("shell-main-splitter")
    shell_splitter.setChildrenCollapsible(False)

    sidebar = qt_widgets.QWidget(shell_splitter)
    sidebar.setObjectName("shell-sidebar")
    sidebar.setMinimumWidth(260)
    sidebar_layout = qt_widgets.QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(12, 12, 12, 12)
    sidebar_layout.setSpacing(10)

    app_title_label = qt_widgets.QLabel("OpenPKPD")
    app_title_label.setObjectName("sidebar-app-title")
    project_name_label = qt_widgets.QLabel(project.name)
    project_name_label.setObjectName("sidebar-project-name")
    project_name_label.setWordWrap(True)
    project_path_label = qt_widgets.QLabel("")
    project_path_label.setObjectName("sidebar-project-path")
    project_path_label.setWordWrap(True)

    nav = qt_widgets.QTreeWidget()
    nav.setObjectName("workflow-nav")
    nav.setHeaderHidden(True)
    nav.setRootIsDecorated(True)
    nav.setIndentation(18)
    nav.setUniformRowHeights(True)
    nav.setContextMenuPolicy(qt_core.Qt.ContextMenuPolicy.CustomContextMenu)
    stack = qt_widgets.QStackedWidget(shell_splitter)
    stack.setObjectName("workflow-stack")
    current_snapshot_path: list[str | None] = [None]
    saved_project_snapshot: list[str] = [""]
    workflow_pages: dict[str, int] = {}
    workflow_definitions = {workflow.workflow_id: workflow for workflow in DEFAULT_WORKFLOWS}
    dashboard_workflow = workflow_definitions["dashboard"]
    scenario_workflows = tuple(
        workflow for workflow in DEFAULT_WORKFLOWS if workflow.workflow_id not in {"dashboard"}
    )
    scenario_workflow_sections = tuple(
        dict.fromkeys(workflow.section for workflow in scenario_workflows)
    )
    nav_nodes: dict[tuple[str, str | None, str | None], object] = {}
    nav_items_by_key: dict[str, object] = {}
    nav_workflow_role = qt_core.Qt.ItemDataRole.UserRole + 1
    nav_project_role = qt_core.Qt.ItemDataRole.UserRole + 2
    nav_scenario_role = qt_core.Qt.ItemDataRole.UserRole + 3
    nav_item_key_role = qt_core.Qt.ItemDataRole.UserRole + 4

    def _standard_icon(name: str):
        pixmap = getattr(qt_widgets.QStyle.StandardPixmap, name, None)
        if pixmap is None:
            return qt_gui.QIcon()
        return window.style().standardIcon(pixmap)

    def _workflow_icon(workflow_id: str):
        return _standard_icon(
            {
                "dashboard": "SP_DirHomeIcon",  # house = dashboard
                "home": "SP_DirHomeIcon",  # house = home page (legacy)
                "data": "SP_DriveHDIcon",  # storage = dataset
                "model": "SP_FileDialogNewFolder",  # structure = model definition
                "fit": "SP_MediaPlay",  # play = run fit
                "results": "SP_DialogOkButton",  # checkmark = completed results
                "diagnostics": "SP_MessageBoxInformation",  # info = diagnostics
                "nca": "SP_FileDialogContentsView",  # contents = NCA parameter table
                "advanced": "SP_ToolBarHorizontalExtensionButton",  # extension = advanced tools
            }.get(workflow_id, "SP_FileIcon")
        )

    def _workflow_section_icon(section: str):
        return _standard_icon(
            {
                "Inputs": "SP_DriveHDIcon",  # storage = input data/model
                "Analyses": "SP_MediaPlay",  # play = run analyses
                "Review": "SP_DialogOkButton",  # checkmark = review results
            }.get(section, "SP_DirOpenIcon")
        )

    def _workflow_state_icon(workflow_id: str, state_id: WorkflowStateId):
        return _standard_icon(
            {
                WorkflowStateId.NOT_STARTED: "SP_FileIcon",
                WorkflowStateId.NEEDS_ATTENTION: "SP_MessageBoxWarning",
                WorkflowStateId.READY: "SP_DialogApplyButton",
                WorkflowStateId.RUNNING: "SP_BrowserReload",
                WorkflowStateId.RESULTS_AVAILABLE: "SP_DialogYesButton",
            }.get(state_id, "SP_FileIcon")
        )

    def _apply_workflow_nav_state(item, workflow, *, project_id: str, scenario_id: str) -> None:
        state = workflow_state_for(
            project,
            workflow.workflow_id,
            project_id=project_id,
            scenario_id=scenario_id,
        )
        item.setIcon(0, _workflow_state_icon(workflow.workflow_id, state.state))
        item.setToolTip(
            0,
            f"{workflow.description}\n\nStatus: {state.label} — {state.summary}",
        )

    def _edit_preferences_dialog(current_preferences: GuiPreferences) -> GuiPreferences | None:
        app = qt_widgets.QApplication.instance()
        default_font_size = default_font_point_size(app) if app is not None else 10

        dialog = qt_widgets.QDialog(window)
        dialog.setObjectName("preferences-dialog")
        dialog.setWindowTitle("Preferences")
        dialog.resize(560, 0)

        dialog_layout = qt_widgets.QVBoxLayout(dialog)
        description_label = qt_widgets.QLabel(
            "Adjust app-wide interface preferences. More settings can be added here over time."
        )
        description_label.setObjectName("preferences-description-label")
        description_label.setWordWrap(True)
        dialog_layout.addWidget(description_label)

        form_layout = qt_widgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        theme_combo = qt_widgets.QComboBox(dialog)
        theme_combo.setObjectName("settings-theme-combo")
        theme_combo.addItem("Light", "light")
        theme_combo.addItem("Dark", "dark")
        theme_combo.setCurrentIndex(0 if current_preferences.theme != "dark" else 1)
        form_layout.addRow("Theme", theme_combo)
        font_size_spin = qt_widgets.QSpinBox(dialog)
        font_size_spin.setObjectName("settings-font-size-spinbox")
        font_size_spin.setRange(MIN_FONT_SIZE, MAX_FONT_SIZE)
        font_size_spin.setValue(current_preferences.font_size or default_font_size)
        font_size_spin.setSuffix(" pt")
        form_layout.addRow("Interface font size", font_size_spin)

        default_workspace_root_input = qt_widgets.QLineEdit(dialog)
        default_workspace_root_input.setObjectName("settings-workspace-root-lineedit")
        default_workspace_root_input.setReadOnly(True)
        default_workspace_root_input.setPlaceholderText(
            f"Defaults to {default_workspace_root_path()} when no dataset/model path is available"
        )
        default_workspace_root_input.setText(current_preferences.default_workspace_root or "")

        default_workspace_root_row = qt_widgets.QWidget(dialog)
        default_workspace_root_layout = qt_widgets.QHBoxLayout(default_workspace_root_row)
        default_workspace_root_layout.setContentsMargins(0, 0, 0, 0)
        default_workspace_root_layout.addWidget(default_workspace_root_input, 1)

        browse_workspace_root_button = qt_widgets.QPushButton("Browse…", dialog)
        browse_workspace_root_button.setObjectName("settings-workspace-root-browse-button")
        clear_workspace_root_button = qt_widgets.QPushButton("Clear", dialog)
        clear_workspace_root_button.setObjectName("settings-workspace-root-clear-button")
        default_workspace_root_layout.addWidget(browse_workspace_root_button)
        default_workspace_root_layout.addWidget(clear_workspace_root_button)

        def _choose_default_workspace_root() -> None:
            start_dir = str(_preferred_dialog_directory(default_workspace_root_input.text()))
            selected_dir = qt_widgets.QFileDialog.getExistingDirectory(
                dialog,
                "Default workspace files location",
                start_dir,
            )
            if selected_dir:
                _remember_last_dialog_selection(selected_dir, selection_is_directory=True)
                default_workspace_root_input.setText(
                    normalize_directory_path(selected_dir) or selected_dir
                )

        browse_workspace_root_button.clicked.connect(_choose_default_workspace_root)
        clear_workspace_root_button.clicked.connect(default_workspace_root_input.clear)
        form_layout.addRow("Default workspace files location", default_workspace_root_row)

        import os as _os

        n_cpu = _os.cpu_count() or 1
        n_parallel_spin = qt_widgets.QSpinBox(dialog)
        n_parallel_spin.setObjectName("settings-n-parallel-spinbox")
        n_parallel_spin.setRange(0, n_cpu * 2)
        n_parallel_spin.setValue(current_preferences.n_parallel)
        n_parallel_spin.setSpecialValueText(f"Auto ({n_cpu} cores detected)")
        n_parallel_spin.setToolTip(
            "Number of parallel workers for estimation and simulation.\n"
            "0 = auto (use all available cores). 1 = serial (no parallelism)."
        )
        form_layout.addRow("CPU cores for estimation/simulation", n_parallel_spin)

        autosave_spin = qt_widgets.QSpinBox(dialog)
        autosave_spin.setObjectName("settings-autosave-spinbox")
        autosave_spin.setRange(0, 60)
        autosave_spin.setValue(current_preferences.autosave_interval_minutes)
        autosave_spin.setSuffix(" min")
        autosave_spin.setSpecialValueText("Disabled")
        autosave_spin.setToolTip(
            "How often to auto-save a recovery snapshot in the background.\n"
            "0 = disabled. Recovery file is deleted on clean exit."
        )
        form_layout.addRow("Autosave interval", autosave_spin)

        dialog_layout.addLayout(form_layout)

        dialog_buttons = qt_widgets.QDialogButtonBox(
            qt_widgets.QDialogButtonBox.StandardButton.Ok
            | qt_widgets.QDialogButtonBox.StandardButton.Cancel
        )
        dialog_buttons.setObjectName("preferences-dialog-buttons")
        restore_defaults_button = dialog_buttons.addButton(
            "Restore default",
            qt_widgets.QDialogButtonBox.ButtonRole.ResetRole,
        )
        restore_defaults_button.setObjectName("settings-restore-default-button")

        def _restore_defaults() -> None:
            theme_combo.setCurrentIndex(0)
            font_size_spin.setValue(default_font_size)
            default_workspace_root_input.clear()
            n_parallel_spin.setValue(0)
            autosave_spin.setValue(5)

        restore_defaults_button.clicked.connect(_restore_defaults)
        dialog_buttons.accepted.connect(dialog.accept)
        dialog_buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(dialog_buttons)

        if dialog.exec() != int(qt_widgets.QDialog.DialogCode.Accepted):
            return None

        selected_font_size = font_size_spin.value()
        selected_theme = theme_combo.currentData() or "light"
        return replace(
            current_preferences,
            font_size=None if selected_font_size == default_font_size else selected_font_size,
            theme=selected_theme,
            default_workspace_root=normalize_directory_path(default_workspace_root_input.text()),
            n_parallel=n_parallel_spin.value(),
            autosave_interval_minutes=autosave_spin.value(),
        )

    def _open_preferences_dialog() -> None:
        app = qt_widgets.QApplication.instance()
        if app is None:
            return
        persisted_preferences = load_gui_preferences(settings_store=settings_store)
        edit_preferences_dialog = getattr(
            window, "_edit_preferences_dialog", _edit_preferences_dialog
        )
        updated_preferences = edit_preferences_dialog(persisted_preferences)
        if updated_preferences is None:
            return
        save_gui_preferences(updated_preferences, settings_store=settings_store)
        apply_gui_preferences(app, updated_preferences)
        _apply_default_workspace_root(updated_preferences)
        _configure_autosave(updated_preferences.autosave_interval_minutes)
        _update_window_title()
        _status_bar().showMessage("Updated interface preferences")

    for workflow in DEFAULT_WORKFLOWS:
        if workflow.workflow_id == "dashboard":
            page = build_dashboard_workflow(project)
        elif workflow.workflow_id == "data":
            page = build_data_workflow(project)
        elif workflow.workflow_id == "model":
            page = build_model_workflow(project)
        elif workflow.workflow_id == "fit":
            page = build_fit_workflow(
                project,
                fit_service=fit_service,
                project_service=project_service,
                preferences=current_preferences,
            )
        elif workflow.workflow_id == "nca":
            page = build_nca_workflow(project)
        elif workflow.workflow_id == "results":
            page = build_results_workflow(project)
        elif workflow.workflow_id == "diagnostics":
            page = build_diagnostics_workflow(
                project,
                fit_service=fit_service,
                npde_service=npde_service,
                project_service=project_service,
            )
        elif workflow.workflow_id == "advanced":
            page = build_advanced_workflow(
                project,
                fit_service=fit_service,
                vpc_service=vpc_service,
                bootstrap_service=bootstrap_service,
                design_service=design_service,
                project_service=project_service,
                preferences=current_preferences,
            )
        elif workflow.workflow_id == "covariate":
            page = build_covariate_workflow(project)
        else:
            raise ValueError(f"Unsupported workflow: {workflow.workflow_id}")
        page._dirty_label = workflow.label
        workflow_pages[workflow.workflow_id] = stack.count()
        stack.addWidget(page)

    home_page_index = workflow_pages["dashboard"]
    project_details_page, project_details_controls = _build_details_editor_page(
        qt_widgets,
        page_object_name="project-details-page",
        field_prefix="project-details-pane",
        title="Project details",
        intro="Edit the selected project's name, description, references, and notes.",
        context_label_text="Select a project to review and update its details.",
        name_label="Project name:",
        save_button_text="Save Project Details",
    )
    project_details_page._dirty_label = "Project details"
    project_details_page_index = stack.count()
    stack.addWidget(project_details_page)

    scenario_details_page, scenario_details_controls = _build_details_editor_page(
        qt_widgets,
        page_object_name="scenario-details-page",
        field_prefix="scenario-details-pane",
        title="Scenario details",
        intro="Edit the selected scenario's name, description, references, and notes.",
        context_label_text="Select a scenario to review and update its details.",
        name_label="Scenario name:",
        save_button_text="Save Scenario Details",
    )
    scenario_details_page._dirty_label = "Scenario details"
    scenario_details_page_index = stack.count()
    stack.addWidget(scenario_details_page)

    sidebar_layout.addWidget(app_title_label)
    sidebar_layout.addWidget(project_name_label)
    sidebar_layout.addWidget(project_path_label)
    sidebar_layout.addWidget(nav, 1)

    def _set_nav_context(
        item,
        *,
        item_key: str,
        workflow_id: str,
        project_id: str | None = None,
        scenario_id: str | None = None,
        register_nav_node: bool = True,
    ):
        item.setData(0, nav_workflow_role, workflow_id)
        item.setData(0, nav_project_role, project_id)
        item.setData(0, nav_scenario_role, scenario_id)
        item.setData(0, nav_item_key_role, item_key)
        if register_nav_node:
            nav_nodes[(workflow_id, project_id, scenario_id)] = item
        nav_items_by_key[item_key] = item

    def _make_nav_item_key(
        kind: str,
        *,
        workflow_id: str | None = None,
        project_id: str | None = None,
        scenario_id: str | None = None,
        section: str | None = None,
    ) -> str:
        payload: dict[str, str] = {"kind": kind}
        if workflow_id is not None:
            payload["workflow_id"] = workflow_id
        if project_id is not None:
            payload["project_id"] = project_id
        if scenario_id is not None:
            payload["scenario_id"] = scenario_id
        if section is not None:
            payload["section"] = section
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _parse_nav_item_key(item_key: str | None) -> dict[str, str] | None:
        if not item_key:
            return None
        try:
            payload = json.loads(item_key)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return {
            str(key): str(value)
            for key, value in payload.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    def _current_nav_key() -> tuple[str | None, str | None, str | None]:
        item = nav.currentItem()
        if item is None:
            return None, None, None
        return (
            item.data(0, nav_workflow_role),
            item.data(0, nav_project_role),
            item.data(0, nav_scenario_role),
        )

    def _current_nav_item_key() -> str | None:
        item = nav.currentItem()
        if item is None:
            return None
        item_key = item.data(0, nav_item_key_role)
        return str(item_key) if isinstance(item_key, str) else None

    def _expanded_nav_item_keys() -> tuple[str, ...]:
        expanded_keys: list[str] = []
        stack_items = [nav.topLevelItem(index) for index in range(nav.topLevelItemCount())]
        while stack_items:
            item = stack_items.pop()
            if item is None:
                continue
            if item.childCount() > 0 and item.isExpanded():
                item_key = item.data(0, nav_item_key_role)
                if isinstance(item_key, str) and item_key:
                    expanded_keys.append(item_key)
            for child_index in range(item.childCount()):
                stack_items.append(item.child(child_index))
        return tuple(expanded_keys)

    def _apply_expanded_nav_item_keys(expanded_item_keys: tuple[str, ...]) -> None:
        expanded_set = set(expanded_item_keys)
        stack_items = [nav.topLevelItem(index) for index in range(nav.topLevelItemCount())]
        while stack_items:
            item = stack_items.pop()
            if item is None:
                continue
            if item.childCount() > 0:
                item_key = item.data(0, nav_item_key_role)
                if isinstance(item_key, str):
                    item.setExpanded(item_key in expanded_set)
            for child_index in range(item.childCount()):
                stack_items.append(item.child(child_index))

    def _set_line_edit_text(widget, value: str) -> None:
        blocked = widget.blockSignals(True)
        widget.setText(value)
        widget.blockSignals(blocked)

    def _set_plain_text(widget, value: str) -> None:
        blocked = widget.blockSignals(True)
        widget.setPlainText(value)
        widget.blockSignals(blocked)

    def _details_form_state(controls: dict[str, object]) -> dict[str, str]:
        return {
            "name": controls["name_input"].text(),
            "description": controls["description_input"].toPlainText(),
            "references": controls["references_input"].toPlainText(),
            "notes": controls["notes_input"].toPlainText(),
        }

    def _apply_details_form_state(controls: dict[str, object], values: dict[str, str]) -> None:
        _set_line_edit_text(controls["name_input"], values["name"])
        _set_plain_text(controls["description_input"], values["description"])
        _set_plain_text(controls["references_input"], values["references"])
        _set_plain_text(controls["notes_input"], values["notes"])

    project_details_synced_state: list[dict[str, str] | None] = [None]
    scenario_details_synced_state: list[dict[str, str] | None] = [None]

    def _project_details_values(selected_project: Project) -> dict[str, str]:
        return {
            "name": selected_project.name,
            "description": _metadata_text(selected_project.metadata, "description"),
            "references": _metadata_text(selected_project.metadata, "references"),
            "notes": _notes_from_metadata(selected_project.metadata),
        }

    def _scenario_details_values(selected_project: Project) -> dict[str, str]:
        selected_scenario = selected_project.active_scenario
        return {
            "name": selected_scenario.name,
            "description": _metadata_text(selected_scenario.metadata, "description"),
            "references": _metadata_text(selected_scenario.metadata, "references"),
            "notes": _notes_from_metadata(selected_scenario.metadata),
        }

    def _project_details_dirty() -> bool:
        synced = project_details_synced_state[0]
        return synced is not None and _details_form_state(project_details_controls) != synced

    def _scenario_details_dirty() -> bool:
        synced = scenario_details_synced_state[0]
        return synced is not None and _details_form_state(scenario_details_controls) != synced

    def _update_project_details_editor_state(*_args) -> None:
        dirty = _project_details_dirty()
        project_details_controls["save_button"].setEnabled(dirty)
        project_details_controls["status_label"].setText(
            "Project details have unsaved changes." if dirty else "No changes to save."
        )
        _update_window_title()

    def _update_scenario_details_editor_state(*_args) -> None:
        dirty = _scenario_details_dirty()
        scenario_details_controls["save_button"].setEnabled(dirty)
        scenario_details_controls["status_label"].setText(
            "Scenario details have unsaved changes." if dirty else "No changes to save."
        )
        _update_window_title()

    def _sync_project_details_editor() -> None:
        selected_project = project.active_project
        values = _project_details_values(selected_project)
        _apply_details_form_state(project_details_controls, values)
        project_details_synced_state[0] = values
        project_details_controls["context_label"].setText(
            f"Editing project: {selected_project.name}\nActive scenario: {selected_project.active_scenario.name}"
        )
        _update_project_details_editor_state()

    def _sync_scenario_details_editor() -> None:
        selected_project = project.active_project
        selected_scenario = selected_project.active_scenario
        values = _scenario_details_values(selected_project)
        _apply_details_form_state(scenario_details_controls, values)
        scenario_details_synced_state[0] = values
        context_lines = [
            f"Editing scenario: {selected_scenario.name}",
            f"Project: {selected_project.name}",
        ]
        parent_summary = _parent_scenario_summary(selected_project, selected_scenario)
        if parent_summary is not None:
            context_lines.append(parent_summary)
        scenario_details_controls["context_label"].setText("\n".join(context_lines))
        _update_scenario_details_editor_state()

    def _refresh_project_details_editor_if_pristine() -> None:
        if not _project_details_dirty():
            _sync_project_details_editor()

    def _refresh_scenario_details_editor_if_pristine() -> None:
        if not _scenario_details_dirty():
            _sync_scenario_details_editor()

    def _save_project_details_editor() -> None:
        current_snapshot = _project_snapshot()
        updated_project = project_service.update_project_details(
            project, **_details_form_state(project_details_controls)
        )
        _sync_project_details_editor()
        if _project_snapshot() == current_snapshot:
            _status_bar().showMessage(f"Project details already up to date: {updated_project.name}")
            return
        _rebuild_navigation_tree(workflow_id="dashboard", project_id=updated_project.project_id)
        _update_window_title()
        _status_bar().showMessage(f"Updated project details: {updated_project.name}")

    def _save_scenario_details_editor() -> None:
        current_snapshot = _project_snapshot()
        updated_scenario = project_service.update_scenario_details(
            project, **_details_form_state(scenario_details_controls)
        )
        _sync_scenario_details_editor()
        if _project_snapshot() == current_snapshot:
            _status_bar().showMessage(
                f"Scenario details already up to date: {updated_scenario.name}"
            )
            return
        _rebuild_navigation_tree(
            workflow_id="dashboard",
            project_id=project.active_project.project_id,
            scenario_id=updated_scenario.scenario_id,
        )
        _update_window_title()
        _status_bar().showMessage(f"Updated scenario details: {updated_scenario.name}")

    project_details_controls["name_input"].textChanged.connect(_update_project_details_editor_state)
    project_details_controls["description_input"].textChanged.connect(
        _update_project_details_editor_state
    )
    project_details_controls["references_input"].textChanged.connect(
        _update_project_details_editor_state
    )
    project_details_controls["notes_input"].textChanged.connect(
        _update_project_details_editor_state
    )
    project_details_controls["save_button"].clicked.connect(_save_project_details_editor)
    project_details_page._has_unsaved_changes = _project_details_dirty  # type: ignore[attr-defined]
    project_details_page._refresh_workflow = _refresh_project_details_editor_if_pristine  # type: ignore[attr-defined]
    project_details_page._load_project = _sync_project_details_editor  # type: ignore[attr-defined]

    scenario_details_controls["name_input"].textChanged.connect(
        _update_scenario_details_editor_state
    )
    scenario_details_controls["description_input"].textChanged.connect(
        _update_scenario_details_editor_state
    )
    scenario_details_controls["references_input"].textChanged.connect(
        _update_scenario_details_editor_state
    )
    scenario_details_controls["notes_input"].textChanged.connect(
        _update_scenario_details_editor_state
    )
    scenario_details_controls["save_button"].clicked.connect(_save_scenario_details_editor)
    scenario_details_page._has_unsaved_changes = _scenario_details_dirty  # type: ignore[attr-defined]
    scenario_details_page._refresh_workflow = _refresh_scenario_details_editor_if_pristine  # type: ignore[attr-defined]
    scenario_details_page._load_project = _sync_scenario_details_editor  # type: ignore[attr-defined]

    def _rebuild_navigation_tree(
        *,
        workflow_id: str | None = None,
        project_id: str | None = None,
        scenario_id: str | None = None,
        item_key: str | None = None,
        expanded_item_keys: tuple[str, ...] | None = None,
    ) -> None:
        selected_workflow_id, selected_project_id, selected_scenario_id = _current_nav_key()
        selected_item_key = _current_nav_item_key()
        current_expanded_item_keys = _expanded_nav_item_keys()
        if workflow_id is not None:
            selected_workflow_id = workflow_id
        if project_id is not None:
            selected_project_id = project_id
        if scenario_id is not None:
            selected_scenario_id = scenario_id
        if item_key is not None:
            selected_item_key = item_key
        elif workflow_id is not None or project_id is not None or scenario_id is not None:
            selected_item_key = None
        if expanded_item_keys is None:
            expanded_item_keys = current_expanded_item_keys

        nav.blockSignals(True)
        nav.clear()
        nav_nodes.clear()
        nav_items_by_key.clear()
        workspace_item = qt_widgets.QTreeWidgetItem(nav, ["Workspace"])
        workspace_item.setIcon(0, _standard_icon("SP_DirHomeIcon"))
        workspace_item.setToolTip(
            0, "Browse projects, scenarios, and grouped workflow destinations."
        )
        workspace_item.setExpanded(True)
        _set_nav_context(
            workspace_item,
            item_key=_make_nav_item_key("workspace"),
            workflow_id="dashboard",
        )

        fallback_item = workspace_item
        for project_model in project.projects:
            project_item = qt_widgets.QTreeWidgetItem(workspace_item, [project_model.name])
            project_item.setIcon(0, _standard_icon("SP_DirIcon"))
            project_item.setExpanded(True)
            project_item.setToolTip(
                0,
                _project_tooltip(project_model),
            )
            _set_nav_context(
                project_item,
                item_key=_make_nav_item_key("project", project_id=project_model.project_id),
                workflow_id="dashboard",
                project_id=project_model.project_id,
            )
            if project_model.project_id == project.active_project_id:
                fallback_item = project_item
            for scenario in project_model.scenarios:
                scenario_item = qt_widgets.QTreeWidgetItem(project_item, [scenario.name])
                scenario_item.setIcon(0, _standard_icon("SP_FileDialogInfoView"))
                scenario_item.setExpanded(True)
                scenario_item.setToolTip(0, _scenario_tooltip(project_model, scenario))
                _set_nav_context(
                    scenario_item,
                    item_key=_make_nav_item_key(
                        "scenario",
                        project_id=project_model.project_id,
                        scenario_id=scenario.scenario_id,
                    ),
                    workflow_id="dashboard",
                    project_id=project_model.project_id,
                    scenario_id=scenario.scenario_id,
                )

                overview_item = qt_widgets.QTreeWidgetItem(
                    scenario_item, [dashboard_workflow.label]
                )
                _apply_workflow_nav_state(
                    overview_item,
                    dashboard_workflow,
                    project_id=project_model.project_id,
                    scenario_id=scenario.scenario_id,
                )
                _set_nav_context(
                    overview_item,
                    item_key=_make_nav_item_key(
                        "overview",
                        workflow_id=dashboard_workflow.workflow_id,
                        project_id=project_model.project_id,
                        scenario_id=scenario.scenario_id,
                    ),
                    workflow_id=dashboard_workflow.workflow_id,
                    project_id=project_model.project_id,
                    scenario_id=scenario.scenario_id,
                )

                section_items: dict[str, object] = {}
                for section in scenario_workflow_sections:
                    section_item = qt_widgets.QTreeWidgetItem(scenario_item, [section])
                    section_item.setIcon(0, _workflow_section_icon(section))
                    section_item.setExpanded(True)
                    section_item.setToolTip(
                        0, f"Browse {section.lower()} workflows for this scenario."
                    )
                    _set_nav_context(
                        section_item,
                        item_key=_make_nav_item_key(
                            "section",
                            project_id=project_model.project_id,
                            scenario_id=scenario.scenario_id,
                            section=section,
                        ),
                        workflow_id="dashboard",
                        project_id=project_model.project_id,
                        scenario_id=scenario.scenario_id,
                        register_nav_node=False,
                    )
                    section_items[section] = section_item

                if (
                    project_model.project_id == project.active_project_id
                    and scenario.scenario_id == project.active_scenario.scenario_id
                ):
                    fallback_item = overview_item
                for workflow in scenario_workflows:
                    workflow_parent = section_items[workflow.section]
                    workflow_item = qt_widgets.QTreeWidgetItem(workflow_parent, [workflow.label])
                    _apply_workflow_nav_state(
                        workflow_item,
                        workflow,
                        project_id=project_model.project_id,
                        scenario_id=scenario.scenario_id,
                    )
                    _set_nav_context(
                        workflow_item,
                        item_key=_make_nav_item_key(
                            "workflow",
                            workflow_id=workflow.workflow_id,
                            project_id=project_model.project_id,
                            scenario_id=scenario.scenario_id,
                        ),
                        workflow_id=workflow.workflow_id,
                        project_id=project_model.project_id,
                        scenario_id=scenario.scenario_id,
                    )

        selected_item = None
        if selected_item_key is not None:
            selected_item = nav_items_by_key.get(selected_item_key)
        if selected_item is None:
            selected_item = nav_nodes.get(
                (selected_workflow_id, selected_project_id, selected_scenario_id)
            )
        if expanded_item_keys is not None:
            _apply_expanded_nav_item_keys(expanded_item_keys)
        nav.setCurrentItem(selected_item or fallback_item)
        nav.blockSignals(False)

    def _refresh_navigation_tree_states() -> None:
        for project_model in project.projects:
            project_item = nav_nodes.get(("dashboard", project_model.project_id, None))
            if project_item is not None:
                project_item.setToolTip(0, _project_tooltip(project_model))
            for scenario in project_model.scenarios:
                scenario_item = nav_nodes.get(
                    ("dashboard", project_model.project_id, scenario.scenario_id)
                )
                if scenario_item is not None:
                    scenario_item.setToolTip(0, _scenario_tooltip(project_model, scenario))
                overview_item = nav_nodes.get(
                    (dashboard_workflow.workflow_id, project_model.project_id, scenario.scenario_id)
                )
                if overview_item is not None:
                    _apply_workflow_nav_state(
                        overview_item,
                        dashboard_workflow,
                        project_id=project_model.project_id,
                        scenario_id=scenario.scenario_id,
                    )
                for workflow in scenario_workflows:
                    workflow_item = nav_nodes.get(
                        (workflow.workflow_id, project_model.project_id, scenario.scenario_id)
                    )
                    if workflow_item is not None:
                        _apply_workflow_nav_state(
                            workflow_item,
                            workflow,
                            project_id=project_model.project_id,
                            scenario_id=scenario.scenario_id,
                        )

    def _update_sidebar_summary() -> None:
        project_name_label.setText(project.name or "Untitled Project")
        context_lines = [
            f"Project: {project.active_project.name}",
            f"Scenario: {project.active_scenario.name}",
        ]
        parent_scenario = _parent_scenario_summary(project.active_project, project.active_scenario)
        if parent_scenario is not None:
            context_lines.append(parent_scenario)
        project_description = _metadata_summary(
            "Project description", project.active_project.metadata, key="description"
        )
        if project_description is not None:
            context_lines.append(project_description)
        project_references = _metadata_summary(
            "Project references", project.active_project.metadata, key="references"
        )
        if project_references is not None:
            context_lines.append(project_references)
        project_notes = _notes_summary("Project notes", project.active_project.metadata)
        if project_notes is not None:
            context_lines.append(project_notes)
        scenario_description = _metadata_summary(
            "Scenario description", project.active_scenario.metadata, key="description"
        )
        if scenario_description is not None:
            context_lines.append(scenario_description)
        scenario_references = _metadata_summary(
            "Scenario references", project.active_scenario.metadata, key="references"
        )
        if scenario_references is not None:
            context_lines.append(scenario_references)
        scenario_notes = _notes_summary("Scenario notes", project.active_scenario.metadata)
        if scenario_notes is not None:
            context_lines.append(scenario_notes)
        if current_snapshot_path[0]:
            snapshot_path = Path(current_snapshot_path[0])
            project_path_label.setText(
                f"Snapshot: {snapshot_path.name}\n{snapshot_path.parent}\n"
                + "\n".join(context_lines)
            )
            return
        if project.root_path:
            project_path_label.setText(
                f"Workspace: {project.root_path}\n" + "\n".join(context_lines)
            )
            return
        project_path_label.setText("Unsaved project snapshot\n" + "\n".join(context_lines))

    def _project_snapshot() -> str:
        return json.dumps(project.to_dict(), sort_keys=True, separators=(",", ":"))

    def _dirty_workflow_labels() -> list[str]:
        labels: list[str] = []
        for index in range(stack.count()):
            widget = stack.widget(index)
            has_unsaved_changes = getattr(widget, "_has_unsaved_changes", None)
            if callable(has_unsaved_changes) and bool(has_unsaved_changes()):
                label = getattr(widget, "_dirty_label", None)
                if not isinstance(label, str) or not label:
                    label = (
                        DEFAULT_WORKFLOWS[index].label
                        if index < len(DEFAULT_WORKFLOWS)
                        else f"Page {index + 1}"
                    )
                labels.append(label)
        return labels

    def _is_project_dirty() -> bool:
        return bool(_dirty_workflow_labels()) or _project_snapshot() != saved_project_snapshot[0]

    def _update_window_title() -> None:
        if update_window_title_active[0]:
            return
        update_window_title_active[0] = True
        try:
            dirty_suffix = " *" if _is_project_dirty() else ""
            window.setWindowTitle(f"OpenPKPD — {project.name}{dirty_suffix}")
            _update_sidebar_summary()
            _refresh_navigation_tree_states()
            for index in range(stack.count()):
                refresh_context_header = getattr(
                    stack.widget(index), "_refresh_context_header", None
                )
                if callable(refresh_context_header):
                    refresh_context_header()
            current = stack.currentWidget()
            refresh = getattr(current, "_refresh_workflow", None)
            if callable(refresh):
                refresh()
        finally:
            update_window_title_active[0] = False

    for index in range(stack.count()):
        stack.widget(index)._project_state_changed = _update_window_title
        stack.widget(index)._project_dirty = _is_project_dirty

    def _refresh_workflow(index: int) -> None:
        widget = stack.widget(index)
        refresh = getattr(widget, "_refresh_workflow", None)
        if callable(refresh):
            refresh()

    def _reload_workflows_from_project() -> None:
        for index in range(stack.count()):
            widget = stack.widget(index)
            refresh = getattr(widget, "_load_project", None)
            if not callable(refresh):
                refresh = getattr(widget, "_refresh_workflow", None)
            if callable(refresh):
                refresh()

    def _status_bar():
        return window.statusBar()

    def _latest_artifact(
        *,
        kind: str | None = None,
        role: str | None = None,
        plot_type: str | None = None,
    ) -> object | None:
        for artifact in reversed(project.artifacts):
            if kind is not None and artifact.kind != kind:
                continue
            artifact_role = str(
                artifact.metadata.get("artifact_role") or artifact.kind or ""
            ).strip()
            if role is not None and artifact_role != role:
                continue
            artifact_plot_type = artifact.metadata.get("plot_type")
            if plot_type is not None and str(artifact_plot_type or "") != plot_type:
                continue
            return artifact
        return None

    def _open_artifact_record(artifact, *, empty_message: str) -> bool:
        if artifact is None:
            _status_bar().showMessage(empty_message)
            return False
        if not getattr(artifact, "path", None):
            _status_bar().showMessage("Selected output is not backed by a file.")
            return False
        artifact_path = Path(str(artifact.path))
        if not artifact_path.exists():
            qt_widgets.QMessageBox.information(
                window,
                "Open output",
                f"File is not available on disk:\n{artifact_path}",
            )
            return False
        return qt_gui.QDesktopServices.openUrl(qt_core.QUrl.fromLocalFile(str(artifact_path)))

    def _open_latest_report() -> None:
        _open_artifact_record(
            _latest_artifact(kind="report"), empty_message="No report available yet."
        )

    def _open_latest_plot() -> None:
        _open_artifact_record(_latest_artifact(kind="plot"), empty_message="No plot available yet.")

    def _export_artifact_record(artifact, *, title: str, empty_message: str) -> bool:
        if artifact is None:
            _status_bar().showMessage(empty_message)
            return False
        artifact_path = Path(str(getattr(artifact, "path", "") or "")).expanduser()
        if not artifact_path.exists():
            _status_bar().showMessage(f"File is not available: {artifact_path}")
            return False
        default_name = artifact_path.name or f"{getattr(artifact, 'label', 'artifact')}.artifact"
        destination_path, _ = qt_widgets.QFileDialog.getSaveFileName(
            window,
            title,
            str(_preferred_dialog_directory(_fallback_workspace_root()) / default_name),
            "All files (*)",
        )
        if not destination_path:
            return False
        shutil.copy2(artifact_path, destination_path)
        _remember_last_dialog_selection(destination_path)
        _status_bar().showMessage(f"Saved output copy to {destination_path}")
        return True

    def _save_latest_plot_copy() -> None:
        _export_artifact_record(
            _latest_artifact(kind="plot"),
            title="Save latest plot copy",
            empty_message="No plot available yet.",
        )

    def _save_latest_report_copy() -> None:
        _export_artifact_record(
            _latest_artifact(kind="report"),
            title="Save latest report copy",
            empty_message="No report available yet.",
        )

    def _export_report_record_to_pdf(artifact, *, title: str, empty_message: str) -> bool:
        if artifact is None:
            _status_bar().showMessage(empty_message)
            return False
        artifact_path = Path(str(getattr(artifact, "path", "") or "")).expanduser()
        if not artifact_path.exists():
            _status_bar().showMessage(f"Report file is not available: {artifact_path}")
            return False
        default_pdf_name = (
            artifact_path.with_suffix(".pdf").name or f"{getattr(artifact, 'label', 'report')}.pdf"
        )
        destination_path, _ = qt_widgets.QFileDialog.getSaveFileName(
            window,
            title,
            str(_preferred_dialog_directory(_fallback_workspace_root()) / default_pdf_name),
            "PDF files (*.pdf)",
        )
        if not destination_path:
            return False
        destination = Path(destination_path)
        if destination.suffix.lower() != ".pdf":
            destination = destination.with_suffix(".pdf")
        success, message = report_export_service.export_html_report_to_pdf(
            parent=window,
            source_path=artifact_path,
            destination_path=destination,
        )
        if not success:
            _status_bar().showMessage(
                message or f"Failed to export PDF report from {artifact_path.name}"
            )
            return False
        _remember_last_dialog_selection(destination)
        _status_bar().showMessage(f"Saved PDF report to {destination}")
        return True

    def _export_latest_report_pdf() -> None:
        _export_report_record_to_pdf(
            _latest_artifact(kind="report"),
            title="Export latest report as PDF",
            empty_message="No report available yet.",
        )

    def _open_latest_diagnostics_plot(plot_type: str, *, empty_message: str) -> None:
        _open_artifact_record(
            _latest_artifact(kind="plot", role="plot", plot_type=plot_type),
            empty_message=empty_message,
        )

    def _open_artifact_folder() -> None:
        latest_artifact = _latest_artifact()
        if latest_artifact is not None and getattr(latest_artifact, "path", None):
            target = Path(str(latest_artifact.path)).resolve().parent
        elif project.root_path:
            target = Path(project.root_path).resolve()
        else:
            target = _fallback_workspace_root()
        qt_gui.QDesktopServices.openUrl(qt_core.QUrl.fromLocalFile(str(target)))

    def _clear_recent_projects() -> None:
        project_service.clear_recent_files(project)
        _update_recent_menu()
        _update_window_title()

    def _discard_dirty_workflows() -> None:
        for index in range(stack.count()):
            widget = stack.widget(index)
            has_unsaved_changes = getattr(widget, "_has_unsaved_changes", None)
            if not callable(has_unsaved_changes) or not bool(has_unsaved_changes()):
                continue
            reload = getattr(widget, "_load_project", None)
            if not callable(reload):
                reload = getattr(widget, "_refresh_workflow", None)
            if callable(reload):
                reload()

    def _open_snapshot_with_prompt(source: str | Path) -> bool:
        if not _confirm_discard_unsaved_changes("opening another project snapshot"):
            return False
        return _load_snapshot_from(source)

    def _open_recent_snapshot(source: str | Path) -> bool:
        recent_path = Path(source).resolve()
        if not recent_path.exists():
            project.recent_files = [
                path for path in project.recent_files if path != str(recent_path)
            ]
            project.touch()
            _update_recent_menu()
            qt_widgets.QMessageBox.information(
                window,
                "Open Recent",
                f"The recent project snapshot is no longer available:\n{recent_path}",
            )
            _update_window_title()
            return False
        return _open_snapshot_with_prompt(recent_path)

    def _update_recent_menu() -> None:
        open_recent_menu.clear()
        if not project.recent_files:
            placeholder_action = open_recent_menu.addAction("No recent projects")
            placeholder_action.setObjectName("file-open-recent-empty-action")
            placeholder_action.setEnabled(False)
            return
        for index, recent_path in enumerate(project.recent_files):
            resolved_path = str(Path(recent_path).resolve())
            recent_snapshot = Path(resolved_path)
            action = open_recent_menu.addAction(
                f"{recent_snapshot.name} — {recent_snapshot.parent}"
            )
            action.setObjectName(f"file-open-recent-action-{index}")
            action.setStatusTip(resolved_path)
            action.setToolTip(resolved_path)
            action.triggered.connect(
                lambda _checked=False, path=resolved_path: _open_recent_snapshot(path)
            )
        open_recent_menu.addSeparator()
        clear_recent_action = open_recent_menu.addAction("Clear Recent")
        clear_recent_action.setObjectName("file-clear-recent-projects-action")
        clear_recent_action.triggered.connect(_clear_recent_projects)

    def _mark_project_saved() -> None:
        saved_project_snapshot[0] = _project_snapshot()
        _update_window_title()

    def _can_serialize_snapshot(action_title: str, action_label: str) -> bool:
        dirty_workflows = _dirty_workflow_labels()
        if not dirty_workflows:
            return True
        qt_widgets.QMessageBox.warning(
            window,
            action_title,
            f"Save pending workflow edits before {action_label}.\n\n"
            f"Pending editor changes: {', '.join(dirty_workflows)}",
        )
        return False

    def _save_current_project() -> bool:
        if not _can_serialize_snapshot("Save Project Snapshot", "saving the project snapshot"):
            return False
        return (
            _save_snapshot_to(current_snapshot_path[0])
            if current_snapshot_path[0]
            else _choose_snapshot_to_save()
        )

    def _save_current_project_as() -> bool:
        if not _can_serialize_snapshot("Save Project Snapshot", "saving the project snapshot"):
            return False
        return _choose_snapshot_to_save()

    def _confirm_discard_unsaved_changes(action_label: str) -> bool:
        dirty_workflows = _dirty_workflow_labels()
        if dirty_workflows:
            discard_editor_changes = qt_widgets.QMessageBox.warning(
                window,
                "Unsaved workflow edits",
                "Some workflow editors contain unapplied changes that are not yet part of the project state.\n\n"
                f"Pending editor changes: {', '.join(dirty_workflows)}\n\n"
                f"Continue {action_label} and discard those editor-only changes?",
                qt_widgets.QMessageBox.StandardButton.Discard
                | qt_widgets.QMessageBox.StandardButton.Cancel,
                qt_widgets.QMessageBox.StandardButton.Cancel,
            )
            if discard_editor_changes != qt_widgets.QMessageBox.StandardButton.Discard:
                return False
            _discard_dirty_workflows()
        if _project_snapshot() == saved_project_snapshot[0]:
            _update_window_title()
            return True
        save_changes = qt_widgets.QMessageBox.warning(
            window,
            "Unsaved project changes",
            f"Save project changes before {action_label}?",
            qt_widgets.QMessageBox.StandardButton.Save
            | qt_widgets.QMessageBox.StandardButton.Discard
            | qt_widgets.QMessageBox.StandardButton.Cancel,
            qt_widgets.QMessageBox.StandardButton.Save,
        )
        if save_changes == qt_widgets.QMessageBox.StandardButton.Save:
            return _save_current_project()
        return save_changes == qt_widgets.QMessageBox.StandardButton.Discard

    def _save_snapshot_to(destination: str | Path) -> bool:
        snapshot_path = Path(destination).resolve()
        previous_recent_files = list(project.recent_files)
        previous_updated_at = project.updated_at
        project_service.remember_recent_file(project, snapshot_path)
        try:
            fit_state_payloads = fit_service.all_fit_context_payloads(project)
            snapshot_service.save_snapshot(
                project,
                snapshot_path,
                fit_state_payloads=fit_state_payloads or None,
            )
        except Exception as exc:
            project.recent_files = previous_recent_files
            project.updated_at = previous_updated_at
            qt_widgets.QMessageBox.critical(
                window,
                "Save Project Snapshot",
                f"Failed to save project snapshot:\n{exc}",
            )
            return False
        current_snapshot_path[0] = str(snapshot_path)
        project.root_path = str(snapshot_path.parent)
        _remember_last_dialog_selection(snapshot_path)
        _update_recent_menu()
        _mark_project_saved()
        _status_bar().showMessage(f"Saved project snapshot: {snapshot_path.name}")
        return True

    def _load_snapshot_from(source: str | Path) -> bool:
        snapshot_path = Path(source).resolve()
        selected_workflow_id, _, _ = _current_nav_key()
        try:
            loaded_snapshot = snapshot_service.load_snapshot(snapshot_path)
        except Exception as exc:
            qt_widgets.QMessageBox.critical(
                window,
                "Open Project Snapshot",
                f"Failed to open project snapshot:\n{exc}",
            )
            return False
        _replace_project_contents(project, loaded_snapshot.project)
        current_snapshot_path[0] = str(snapshot_path)
        project.root_path = str(snapshot_path.parent)
        _remember_last_dialog_selection(snapshot_path)
        project_service.remember_recent_file(project, snapshot_path)
        restored_fit_states, restore_warnings = fit_service.restore_fit_context_payloads(
            project,
            loaded_snapshot.fit_state_payloads,
        )
        _rebuild_navigation_tree(
            workflow_id=selected_workflow_id or "dashboard",
            project_id=project.active_project_id,
            scenario_id=project.active_scenario.scenario_id,
        )
        _update_recent_menu()
        _reload_workflows_from_project()
        _apply_saved_table_column_widths_to_root(window, qt_widgets, settings_store=settings_store)
        _mark_project_saved()
        status_message = f"Opened project snapshot: {snapshot_path.name}"
        if restored_fit_states:
            suffix = "fit state" if restored_fit_states == 1 else "fit states"
            status_message += f" • restored {restored_fit_states} {suffix}"
        if restore_warnings:
            suffix = "warning" if len(restore_warnings) == 1 else "warnings"
            status_message += f" • {len(restore_warnings)} restore {suffix}"
            qt_widgets.QMessageBox.warning(
                window,
                "Open Project Snapshot",
                "Some saved fit state could not be restored:\n\n"
                + "\n".join(f"- {warning}" for warning in restore_warnings[:5]),
            )
        _status_bar().showMessage(status_message)
        return True

    def _default_snapshot_path() -> str:
        snapshot_name = (
            Path(current_snapshot_path[0]).name
            if current_snapshot_path[0]
            else _default_snapshot_name(project)
        )
        return str(
            _preferred_dialog_directory(
                Path(current_snapshot_path[0]).parent if current_snapshot_path[0] else None
            )
            / snapshot_name
        )

    def _default_export_snapshot_path(export_project: Workspace) -> str:
        return str(
            _preferred_dialog_directory(
                Path(current_snapshot_path[0]).parent if current_snapshot_path[0] else None
            )
            / _default_snapshot_name(export_project)
        )

    def _default_snapshot_open_dir() -> str:
        return str(
            _preferred_dialog_directory(
                Path(current_snapshot_path[0]).parent if current_snapshot_path[0] else None
            )
        )

    def _choose_snapshot_destination(*, title: str, default_path: str) -> str | None:
        selected_path, _ = qt_widgets.QFileDialog.getSaveFileName(
            window,
            title,
            default_path,
            "OpenPKPD project snapshots (*.opkpd);;ZIP archives (*.zip);;All files (*)",
        )
        if not selected_path:
            return None
        normalized_path = _normalize_snapshot_path(selected_path)
        _remember_last_dialog_selection(normalized_path)
        return normalized_path

    def _choose_snapshot_source(*, title: str) -> str | None:
        selected_path, _ = qt_widgets.QFileDialog.getOpenFileName(
            window,
            title,
            _default_snapshot_open_dir(),
            "OpenPKPD project snapshots (*.opkpd *.pkp *.zip);;All files (*)",
        )
        if not selected_path:
            return None
        resolved_path = str(Path(selected_path).resolve())
        _remember_last_dialog_selection(resolved_path)
        return resolved_path

    def _workflow_after_import() -> str:
        workflow_id, _, _ = _current_nav_key()
        return workflow_id if workflow_id not in {None, "dashboard"} else "dashboard"

    def _choose_snapshot_to_open() -> bool:
        selected_path = _choose_snapshot_source(title="Open Project Snapshot")
        if selected_path is None:
            return False
        return _open_snapshot_with_prompt(selected_path)

    def _choose_snapshot_to_save() -> bool:
        selected_path = _choose_snapshot_destination(
            title="Save Project Snapshot",
            default_path=_default_snapshot_path(),
        )
        if selected_path is None:
            return False
        return _save_snapshot_to(selected_path)

    def _save_export_snapshot(
        export_project: Workspace,
        destination: str | Path,
        *,
        action_title: str,
        success_message: str,
    ) -> bool:
        snapshot_path = Path(destination).resolve()
        try:
            snapshot_service.save_snapshot(export_project, snapshot_path)
        except Exception as exc:
            qt_widgets.QMessageBox.critical(
                window,
                action_title,
                f"Failed to save snapshot:\n{exc}",
            )
            return False
        _status_bar().showMessage(success_message.format(snapshot_name=snapshot_path.name))
        return True

    def _save_active_project_snapshot(destination: str | Path) -> bool:
        if not _can_serialize_snapshot(
            "Save Project Snapshot",
            "exporting the project snapshot",
        ):
            return False
        export_project = snapshot_service.export_workspace_for_project(project)
        return _save_export_snapshot(
            export_project,
            destination,
            action_title="Save Project Snapshot",
            success_message="Saved project snapshot: {snapshot_name}",
        )

    def _save_active_scenario_snapshot(destination: str | Path) -> bool:
        if not _can_serialize_snapshot(
            "Save Scenario Snapshot",
            "exporting the scenario snapshot",
        ):
            return False
        export_project = snapshot_service.export_workspace_for_scenario(project)
        return _save_export_snapshot(
            export_project,
            destination,
            action_title="Save Scenario Snapshot",
            success_message="Saved scenario snapshot: {snapshot_name}",
        )

    def _choose_project_snapshot_to_save() -> bool:
        if not _can_serialize_snapshot(
            "Save Project Snapshot",
            "exporting the project snapshot",
        ):
            return False
        export_project = snapshot_service.export_workspace_for_project(project)
        selected_path = _choose_snapshot_destination(
            title="Save Project Snapshot",
            default_path=_default_export_snapshot_path(export_project),
        )
        if selected_path is None:
            return False
        return _save_active_project_snapshot(selected_path)

    def _choose_scenario_snapshot_to_save() -> bool:
        if not _can_serialize_snapshot(
            "Save Scenario Snapshot",
            "exporting the scenario snapshot",
        ):
            return False
        export_project = snapshot_service.export_workspace_for_scenario(project)
        selected_path = _choose_snapshot_destination(
            title="Save Scenario Snapshot",
            default_path=_default_export_snapshot_path(export_project),
        )
        if selected_path is None:
            return False
        return _save_active_scenario_snapshot(selected_path)

    def _load_project_snapshot_from(source: str | Path) -> bool:
        if not _confirm_discard_unsaved_changes("loading a project snapshot"):
            return False
        snapshot_path = Path(source).resolve()
        try:
            loaded_snapshot = snapshot_service.load_snapshot(snapshot_path)
            loaded_project = project_service.import_project(project, loaded_snapshot.project)
        except Exception as exc:
            qt_widgets.QMessageBox.critical(
                window,
                "Load Project Snapshot",
                f"Failed to load project snapshot:\n{exc}",
            )
            return False
        _rebuild_navigation_tree(
            workflow_id=_workflow_after_import(),
            project_id=loaded_project.project_id,
            scenario_id=loaded_project.active_scenario.scenario_id,
        )
        _reload_workflows_from_project()
        if nav.currentItem() is not None:
            _on_item_changed(nav.currentItem(), None)
        _update_window_title()
        _status_bar().showMessage(f"Loaded project snapshot: {snapshot_path.name}")
        return True

    def _load_scenario_snapshot_from(source: str | Path) -> bool:
        if not _confirm_discard_unsaved_changes("loading a scenario snapshot"):
            return False
        snapshot_path = Path(source).resolve()
        try:
            loaded_snapshot = snapshot_service.load_snapshot(snapshot_path)
            scenario = project_service.import_scenario(project, loaded_snapshot.project)
        except Exception as exc:
            qt_widgets.QMessageBox.critical(
                window,
                "Load Scenario Snapshot",
                f"Failed to load scenario snapshot:\n{exc}",
            )
            return False
        _rebuild_navigation_tree(
            workflow_id=_workflow_after_import(),
            project_id=project.active_project.project_id,
            scenario_id=scenario.scenario_id,
        )
        _reload_workflows_from_project()
        if nav.currentItem() is not None:
            _on_item_changed(nav.currentItem(), None)
        _update_window_title()
        _status_bar().showMessage(f"Loaded scenario snapshot: {snapshot_path.name}")
        return True

    def _choose_project_snapshot_to_load() -> bool:
        selected_path = _choose_snapshot_source(title="Load Project Snapshot")
        if selected_path is None:
            return False
        return _load_project_snapshot_from(selected_path)

    def _choose_scenario_snapshot_to_load() -> bool:
        selected_path = _choose_snapshot_source(title="Load Scenario Snapshot")
        if selected_path is None:
            return False
        return _load_scenario_snapshot_from(selected_path)

    def _prompt_for_name(*, title: str, label: str, default_value: str) -> str | None:
        prompt_override = getattr(window, "_prompt_for_name_override", None)
        if callable(prompt_override):
            overridden = prompt_override(title=title, label=label, default_value=default_value)
            if overridden is None:
                return None
            normalized_override = str(overridden).strip()
            return normalized_override or default_value
        value, accepted = qt_widgets.QInputDialog.getText(
            window,
            title,
            label,
            text=default_value,
        )
        if not accepted:
            return None
        normalized = value.strip()
        return normalized or default_value

    def _prompt_for_notes(*, title: str, label: str, default_value: str) -> str | None:
        value, accepted = qt_widgets.QInputDialog.getMultiLineText(
            window,
            title,
            label,
            default_value,
        )
        if not accepted:
            return None
        return value

    def _confirm_delete(title: str, message: str) -> bool:
        response = qt_widgets.QMessageBox.warning(
            window,
            title,
            message,
            qt_widgets.QMessageBox.StandardButton.Yes
            | qt_widgets.QMessageBox.StandardButton.Cancel,
            qt_widgets.QMessageBox.StandardButton.Cancel,
        )
        return response == qt_widgets.QMessageBox.StandardButton.Yes

    def _selection_after_context_change() -> tuple[str, str | None, str | None]:
        workflow_id, project_id, scenario_id = _current_nav_key()
        if workflow_id != "dashboard":
            return (
                workflow_id or "dashboard",
                project.active_project_id,
                project.active_scenario.scenario_id,
            )
        if project_id is None:
            return "dashboard", None, None
        if scenario_id is None:
            return "dashboard", project.active_project_id, None
        return "dashboard", project.active_project_id, project.active_scenario.scenario_id

    def _create_project() -> bool:
        if not _confirm_discard_unsaved_changes("creating a new project"):
            return False
        default_name = f"Project {len(project.projects) + 1}"
        name = _prompt_for_name(
            title="New Project",
            label="Project name:",
            default_value=default_name,
        )
        if name is None:
            return False
        created_project = project_service.create_project(project, name=name)
        _rebuild_navigation_tree(
            workflow_id="dashboard",
            project_id=created_project.project_id,
            scenario_id=created_project.active_scenario.scenario_id,
        )
        _reload_workflows_from_project()
        if nav.currentItem() is not None:
            _on_item_changed(nav.currentItem(), None)
        _update_window_title()
        _status_bar().showMessage(f"Created project: {created_project.name}")
        return True

    def _create_scenario() -> bool:
        if not _confirm_discard_unsaved_changes("creating a new scenario"):
            return False
        default_name = f"Scenario {len(project.active_project.scenarios) + 1}"
        name = _prompt_for_name(
            title="New Scenario",
            label=f"Scenario name for {project.active_project.name}:",
            default_value=default_name,
        )
        if name is None:
            return False
        scenario = project_service.create_scenario(project, name=name)
        _rebuild_navigation_tree(
            workflow_id="dashboard",
            project_id=project.active_project.project_id,
            scenario_id=scenario.scenario_id,
        )
        _reload_workflows_from_project()
        if nav.currentItem() is not None:
            _on_item_changed(nav.currentItem(), None)
        _update_window_title()
        _status_bar().showMessage(
            f"Created scenario: {scenario.name} in {project.active_project.name}"
        )
        return True

    def _duplicate_project() -> bool:
        source_project = project.active_project
        if not _confirm_discard_unsaved_changes(f"duplicating project '{source_project.name}'"):
            return False
        name = _prompt_for_name(
            title="Duplicate Project",
            label="New project name:",
            default_value=f"{source_project.name} Copy",
        )
        if name is None:
            return False
        duplicate_project = project_service.duplicate_project(project, name=name)
        _rebuild_navigation_tree(
            workflow_id="dashboard",
            project_id=duplicate_project.project_id,
            scenario_id=duplicate_project.active_scenario.scenario_id,
        )
        _reload_workflows_from_project()
        if nav.currentItem() is not None:
            _on_item_changed(nav.currentItem(), None)
        _update_window_title()
        _status_bar().showMessage(f"Duplicated project: {duplicate_project.name}")
        return True

    def _duplicate_scenario() -> bool:
        source_scenario = project.active_scenario
        if not _confirm_discard_unsaved_changes(f"duplicating scenario '{source_scenario.name}'"):
            return False
        name = _prompt_for_name(
            title="Duplicate Scenario",
            label=f"New scenario name for {project.active_project.name}:",
            default_value=f"{source_scenario.name} Copy",
        )
        if name is None:
            return False
        scenario = project_service.duplicate_scenario(project, name=name)
        _rebuild_navigation_tree(
            workflow_id="dashboard",
            project_id=project.active_project.project_id,
            scenario_id=scenario.scenario_id,
        )
        _reload_workflows_from_project()
        if nav.currentItem() is not None:
            _on_item_changed(nav.currentItem(), None)
        _update_window_title()
        _status_bar().showMessage(
            f"Duplicated scenario: {scenario.name} in {project.active_project.name}"
        )
        return True

    def _rename_project() -> bool:
        current_name = project.active_project.name
        name = _prompt_for_name(
            title="Rename Project",
            label="Project name:",
            default_value=current_name,
        )
        if name is None or name == current_name:
            return False
        renamed_project = project_service.rename_project(project, name=name)
        _rebuild_navigation_tree()
        _update_window_title()
        _status_bar().showMessage(f"Renamed project to: {renamed_project.name}")
        return True

    def _rename_scenario() -> bool:
        current_name = project.active_scenario.name
        name = _prompt_for_name(
            title="Rename Scenario",
            label=f"Scenario name for {project.active_project.name}:",
            default_value=current_name,
        )
        if name is None or name == current_name:
            return False
        scenario = project_service.rename_scenario(project, name=name)
        _rebuild_navigation_tree()
        _update_window_title()
        _status_bar().showMessage(f"Renamed scenario to: {scenario.name}")
        return True

    def _edit_project_details() -> bool:
        selected_project = project.active_project
        current_snapshot = _project_snapshot()
        details = _prompt_for_project_details(
            window,
            qt_widgets,
            name=selected_project.name,
            description=_metadata_text(selected_project.metadata, "description"),
            references=_metadata_text(selected_project.metadata, "references"),
            notes=_notes_from_metadata(selected_project.metadata),
        )
        if details is None:
            return False
        updated_project = project_service.update_project_details(project, **details)
        if _project_snapshot() == current_snapshot:
            return False
        _rebuild_navigation_tree()
        _update_window_title()
        _status_bar().showMessage(f"Updated project details: {updated_project.name}")
        return True

    def _edit_scenario_details() -> bool:
        _refresh_scenario_details_editor_if_pristine()
        stack.setCurrentIndex(scenario_details_page_index)
        _update_window_title()
        _status_bar().showMessage(f"Scenario details ready: {project.active_scenario.name}")
        return True

    def _delete_project() -> bool:
        selected_project = project.active_project
        if len(project.projects) <= 1:
            qt_widgets.QMessageBox.information(
                window,
                "Delete Project",
                "At least one project must remain in the workspace.",
            )
            return False
        if not _confirm_delete(
            "Delete Project",
            f"Delete project '{selected_project.name}' and all of its scenarios?\n\n"
            "This removes that project's saved inputs, runs, and outputs from the workspace.",
        ):
            return False
        if not _confirm_discard_unsaved_changes(f"deleting project '{selected_project.name}'"):
            return False
        removed = project_service.delete_project(project)
        workflow_id, project_id, scenario_id = _selection_after_context_change()
        _rebuild_navigation_tree(
            workflow_id=workflow_id,
            project_id=project_id,
            scenario_id=scenario_id,
        )
        _reload_workflows_from_project()
        if nav.currentItem() is not None:
            _on_item_changed(nav.currentItem(), None)
        _update_window_title()
        _status_bar().showMessage(f"Deleted project: {removed.name}")
        return True

    def _delete_scenario() -> bool:
        scenario = project.active_scenario
        selected_project = project.active_project
        if len(selected_project.scenarios) <= 1:
            qt_widgets.QMessageBox.information(
                window,
                "Delete Scenario",
                f"At least one scenario must remain in {selected_project.name}.",
            )
            return False
        if not _confirm_delete(
            "Delete Scenario",
            f"Delete scenario '{scenario.name}' from {selected_project.name}?\n\n"
            "This removes the scenario's saved inputs, runs, and outputs from the workspace.",
        ):
            return False
        if not _confirm_discard_unsaved_changes(f"deleting scenario '{scenario.name}'"):
            return False
        removed = project_service.delete_scenario(project)
        workflow_id, project_id, scenario_id = _selection_after_context_change()
        _rebuild_navigation_tree(
            workflow_id=workflow_id,
            project_id=project_id,
            scenario_id=scenario_id,
        )
        _reload_workflows_from_project()
        if nav.currentItem() is not None:
            _on_item_changed(nav.currentItem(), None)
        _update_window_title()
        _status_bar().showMessage(f"Deleted scenario: {removed.name}")
        return True

    file_menu = window.menuBar().addMenu("&File")
    file_new_project_action = file_menu.addAction("New Project…")
    file_new_project_action.setObjectName("file-new-project-action")
    file_new_project_action.setShortcut(qt_gui.QKeySequence.New)
    file_new_project_action.triggered.connect(_create_project)
    file_menu.addSeparator()

    open_action = file_menu.addAction("Open Project Snapshot…")
    open_action.setObjectName("file-open-project-action")
    open_action.setShortcut(qt_gui.QKeySequence.Open)
    open_action.triggered.connect(_choose_snapshot_to_open)

    open_recent_menu = file_menu.addMenu("Open Recent")
    open_recent_menu.setObjectName("file-open-recent-menu")

    save_action = file_menu.addAction("Save Project Snapshot")
    save_action.setObjectName("file-save-project-action")
    save_action.setShortcut(qt_gui.QKeySequence.Save)
    save_action.triggered.connect(_save_current_project)

    save_as_action = file_menu.addAction("Save Project Snapshot As…")
    save_as_action.setObjectName("file-save-project-as-action")
    save_as_action.setShortcut(qt_gui.QKeySequence.SaveAs)
    save_as_action.triggered.connect(_save_current_project_as)

    file_menu.addSeparator()
    close_action = file_menu.addAction("Close")
    close_action.setObjectName("file-close-action")
    close_action.setShortcut(qt_gui.QKeySequence.Close)
    close_action.triggered.connect(window.close)

    exit_action = file_menu.addAction("Quit" if sys.platform == "darwin" else "Exit")
    exit_action.setObjectName("file-exit-action")
    exit_action.setShortcut(qt_gui.QKeySequence.Quit)
    exit_action.triggered.connect(window.close)

    workspace_menu = window.menuBar().addMenu("&Workspace")
    new_project_action = workspace_menu.addAction("New Project…")
    new_project_action.setObjectName("workspace-new-project-action")
    new_project_action.triggered.connect(_create_project)
    new_scenario_action = workspace_menu.addAction("New Scenario…")
    new_scenario_action.setObjectName("workspace-new-scenario-action")
    new_scenario_action.triggered.connect(_create_scenario)
    workspace_menu.addSeparator()
    duplicate_project_action = workspace_menu.addAction("Duplicate Project…")
    duplicate_project_action.setObjectName("workspace-duplicate-project-action")
    duplicate_project_action.triggered.connect(_duplicate_project)
    duplicate_scenario_action = workspace_menu.addAction("Duplicate Scenario…")
    duplicate_scenario_action.setObjectName("workspace-duplicate-scenario-action")
    duplicate_scenario_action.triggered.connect(_duplicate_scenario)
    workspace_menu.addSeparator()
    save_project_snapshot_action = workspace_menu.addAction("Save Project Snapshot…")
    save_project_snapshot_action.setObjectName("workspace-save-project-snapshot-action")
    save_project_snapshot_action.triggered.connect(_choose_project_snapshot_to_save)
    save_scenario_snapshot_action = workspace_menu.addAction("Save Scenario Snapshot…")
    save_scenario_snapshot_action.setObjectName("workspace-save-scenario-snapshot-action")
    save_scenario_snapshot_action.triggered.connect(_choose_scenario_snapshot_to_save)
    load_project_snapshot_action = workspace_menu.addAction("Load Project Snapshot…")
    load_project_snapshot_action.setObjectName("workspace-load-project-snapshot-action")
    load_project_snapshot_action.triggered.connect(_choose_project_snapshot_to_load)
    load_scenario_snapshot_action = workspace_menu.addAction("Load Scenario Snapshot…")
    load_scenario_snapshot_action.setObjectName("workspace-load-scenario-snapshot-action")
    load_scenario_snapshot_action.triggered.connect(_choose_scenario_snapshot_to_load)
    workspace_menu.addSeparator()
    rename_project_action = workspace_menu.addAction("Rename Project…")
    rename_project_action.setObjectName("workspace-rename-project-action")
    rename_project_action.triggered.connect(_rename_project)
    rename_scenario_action = workspace_menu.addAction("Rename Scenario…")
    rename_scenario_action.setObjectName("workspace-rename-scenario-action")
    rename_scenario_action.triggered.connect(_rename_scenario)
    edit_project_details_action = workspace_menu.addAction("Edit Project Details…")
    edit_project_details_action.setObjectName("workspace-edit-project-details-action")
    edit_project_details_action.triggered.connect(_edit_project_details)
    edit_scenario_details_action = workspace_menu.addAction("Edit Scenario Details…")
    edit_scenario_details_action.setObjectName("workspace-edit-scenario-details-action")
    edit_scenario_details_action.triggered.connect(_edit_scenario_details)
    workspace_menu.addSeparator()
    delete_project_action = workspace_menu.addAction("Delete Project…")
    delete_project_action.setObjectName("workspace-delete-project-action")
    delete_project_action.triggered.connect(_delete_project)
    delete_scenario_action = workspace_menu.addAction("Delete Scenario…")
    delete_scenario_action.setObjectName("workspace-delete-scenario-action")
    delete_scenario_action.triggered.connect(_delete_scenario)

    navigate_menu = window.menuBar().addMenu("&Navigate")

    def _select_navigation_item(
        workflow_id: str,
        *,
        project_id: str | None = None,
        scenario_id: str | None = None,
    ) -> None:
        key = (
            workflow_id,
            (project_id or project.active_project_id) if workflow_id != "dashboard" else None,
            (scenario_id or project.active_scenario.scenario_id)
            if workflow_id != "dashboard"
            else None,
        )
        item = nav_nodes.get(key)
        if item is None and workflow_id != "dashboard":
            item = next(
                (
                    candidate
                    for (
                        candidate_workflow,
                        _project_id,
                        _scenario_id,
                    ), candidate in nav_nodes.items()
                    if candidate_workflow == workflow_id
                ),
                None,
            )
        if item is None:
            item = nav_nodes.get(("dashboard", None, None))
        if item is not None:
            nav.setCurrentItem(item)

    def _workflow_widget(workflow_id: str):
        workflow_index = workflow_pages.get(workflow_id)
        if workflow_index is None:
            return None
        return stack.widget(workflow_index)

    def _trigger_workflow_button(workflow_id: str, button_object_name: str) -> bool:
        workflow_widget = _workflow_widget(workflow_id)
        if workflow_widget is None:
            return False
        _select_navigation_item(workflow_id)
        button = workflow_widget.findChild(qt_widgets.QPushButton, button_object_name)
        if button is None:
            return False
        button.click()
        return True

    def _focus_workflow_widget(workflow_id: str, widget_object_name: str | None = None) -> bool:
        workflow_widget = _workflow_widget(workflow_id)
        if workflow_widget is None:
            return False
        _select_navigation_item(workflow_id)
        if not widget_object_name:
            workflow_widget.setFocus()
            qt_core.QTimer.singleShot(0, workflow_widget.setFocus)
            return True
        widget = workflow_widget.findChild(qt_widgets.QWidget, widget_object_name)
        if widget is None:
            return False
        if hasattr(widget, "setFocus"):
            widget.setFocus()
            qt_core.QTimer.singleShot(0, widget.setFocus)
        return True

    for index in range(stack.count()):
        stack.widget(index)._navigate_to_workflow = _select_navigation_item
        stack.widget(index)._focus_workflow_widget = _focus_workflow_widget
        stack.widget(index)._create_project = _create_project
        stack.widget(index)._choose_project_snapshot_to_open = _choose_snapshot_to_open
        stack.widget(index)._open_recent_snapshot = _open_recent_snapshot
        stack.widget(index)._current_snapshot_path = lambda: current_snapshot_path[0]
        stack.widget(index)._navigate_to_results = lambda: _select_navigation_item("results")
        stack.widget(index)._project_open_latest_report = _open_latest_report
        stack.widget(index)._project_export_latest_report_pdf = _export_latest_report_pdf
        stack.widget(index)._project_open_latest_plot = _open_latest_plot
        stack.widget(index)._duplicate_scenario = _duplicate_scenario

    def _import_dataset() -> bool:
        return _trigger_workflow_button("data", "data-import-button")

    def _open_control_stream() -> bool:
        return _trigger_workflow_button("model", "model-control-stream-open-button")

    current_section = None
    for workflow in DEFAULT_WORKFLOWS:
        if workflow.section != current_section:
            current_section = workflow.section
            navigate_menu.addSection(current_section)
        action = navigate_menu.addAction(_workflow_icon(workflow.workflow_id), workflow.label)
        action.setObjectName(f"navigate-{workflow.workflow_id}-action")
        action.triggered.connect(
            lambda _checked=False, workflow_id=workflow.workflow_id: _select_navigation_item(
                workflow_id
            )
        )

    for _shortcut_index, _shortcut_workflow in enumerate(DEFAULT_WORKFLOWS[:9], start=1):
        _shortcut = qt_gui.QShortcut(qt_gui.QKeySequence(f"Alt+{_shortcut_index}"), window)
        _shortcut.activated.connect(
            lambda wid=_shortcut_workflow.workflow_id: _select_navigation_item(wid)
        )

    def _step_font_size(delta: int) -> None:
        _app = qt_widgets.QApplication.instance()
        if _app is None:
            return
        prefs = load_gui_preferences(settings_store=settings_store)
        default_size = default_font_point_size(_app)
        base_size = prefs.font_size or default_size
        new_size = (
            default_size
            if delta == 0
            else max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, base_size + delta))
        )
        updated = replace(prefs, font_size=None if new_size == default_size else new_size)
        save_gui_preferences(updated, settings_store=settings_store)
        apply_gui_preferences(_app, updated)
        current_preferences[0] = updated
        _status_bar().showMessage(f"Interface font size: {new_size}pt")

    _zoom_in_shortcut = qt_gui.QShortcut(qt_gui.QKeySequence("Ctrl+="), window)
    _zoom_in_shortcut.activated.connect(lambda: _step_font_size(+1))
    _zoom_out_shortcut = qt_gui.QShortcut(qt_gui.QKeySequence("Ctrl+-"), window)
    _zoom_out_shortcut.activated.connect(lambda: _step_font_size(-1))
    _zoom_reset_shortcut = qt_gui.QShortcut(qt_gui.QKeySequence("Ctrl+0"), window)
    _zoom_reset_shortcut.activated.connect(lambda: _step_font_size(0))

    inputs_menu = window.menuBar().addMenu("&Inputs")
    import_dataset_action = inputs_menu.addAction("Import CSV Dataset…")
    import_dataset_action.setObjectName("inputs-import-dataset-action")
    import_dataset_action.triggered.connect(_import_dataset)
    open_control_stream_action = inputs_menu.addAction("Open NONMEM File…")
    open_control_stream_action.setObjectName("inputs-open-control-stream-action")
    open_control_stream_action.triggered.connect(_open_control_stream)

    results_menu = window.menuBar().addMenu("&Results")
    reports_menu = results_menu.addMenu("Reports")
    reports_menu.setObjectName("results-reports-menu")
    latest_report_action = reports_menu.addAction("Open latest report")
    latest_report_action.setObjectName("results-open-latest-report-action")
    latest_report_action.triggered.connect(_open_latest_report)
    save_latest_report_copy_action = reports_menu.addAction("Save latest report copy…")
    save_latest_report_copy_action.setObjectName("results-save-latest-report-copy-action")
    save_latest_report_copy_action.triggered.connect(_save_latest_report_copy)
    export_latest_report_pdf_action = reports_menu.addAction("Export latest report as PDF…")
    export_latest_report_pdf_action.setObjectName("results-export-latest-report-pdf-action")
    export_latest_report_pdf_action.triggered.connect(_export_latest_report_pdf)

    results_plot_menu = results_menu.addMenu("Plots")
    results_plot_menu.setObjectName("results-plots-menu")
    latest_plot_action = results_plot_menu.addAction("Open latest plot")
    latest_plot_action.setObjectName("results-open-latest-plot-action")
    latest_plot_action.triggered.connect(_open_latest_plot)
    save_latest_plot_copy_action = results_plot_menu.addAction("Save latest plot copy…")
    save_latest_plot_copy_action.setObjectName("results-save-latest-plot-copy-action")
    save_latest_plot_copy_action.triggered.connect(_save_latest_plot_copy)

    diagnostics_menu = results_menu.addMenu("Diagnostics")
    diagnostics_menu.setObjectName("results-diagnostics-menu")
    open_gof_panel_action = diagnostics_menu.addAction("Open GOF panel")
    open_gof_panel_action.setObjectName("diagnostics-open-gof-panel-action")
    open_gof_panel_action.triggered.connect(
        lambda: _open_latest_diagnostics_plot(
            "gof_panel",
            empty_message="No GOF panel output is available yet.",
        )
    )
    open_residual_trends_action = diagnostics_menu.addAction("Open residual trends")
    open_residual_trends_action.setObjectName("diagnostics-open-residual-trends-action")
    open_residual_trends_action.triggered.connect(
        lambda: _open_latest_diagnostics_plot(
            "residual_trends",
            empty_message="No residual trends output is available yet.",
        )
    )
    diagnostics_menu.addSeparator()
    diagnostics_open_artifact_folder_action = diagnostics_menu.addAction("Open outputs folder")
    diagnostics_open_artifact_folder_action.setObjectName("diagnostics-open-artifact-folder-action")
    diagnostics_open_artifact_folder_action.triggered.connect(_open_artifact_folder)

    results_menu.addSeparator()
    open_artifact_folder_action = results_menu.addAction("Open outputs folder")
    open_artifact_folder_action.setObjectName("results-open-artifact-folder-action")
    open_artifact_folder_action.triggered.connect(_open_artifact_folder)

    settings_menu = window.menuBar().addMenu("&Settings")
    preferences_action = settings_menu.addAction("Preferences…")
    preferences_action.setObjectName("settings-preferences-action")
    preferences_action.triggered.connect(_open_preferences_dialog)

    def _tree_context_menu_actions(item) -> tuple[object | None, ...]:
        if item is None:
            return ()
        workflow_id = item.data(0, nav_workflow_role)
        project_id = item.data(0, nav_project_role)
        scenario_id = item.data(0, nav_scenario_role)
        if project_id is None:
            return (
                new_project_action,
                None,
                load_project_snapshot_action,
            )
        if scenario_id is None:
            return (
                new_scenario_action,
                None,
                duplicate_project_action,
                save_project_snapshot_action,
                load_scenario_snapshot_action,
                None,
                rename_project_action,
                edit_project_details_action,
                None,
                delete_project_action,
            )

        actions: list[object | None] = [
            new_scenario_action,
            None,
            duplicate_scenario_action,
            save_scenario_snapshot_action,
            load_scenario_snapshot_action,
        ]
        if str(workflow_id) == "data":
            actions.extend((None, import_dataset_action))
        elif str(workflow_id) == "model":
            actions.extend((None, open_control_stream_action))
        elif str(workflow_id) == "results":
            actions.extend(
                (
                    None,
                    latest_report_action,
                    save_latest_report_copy_action,
                    export_latest_report_pdf_action,
                    latest_plot_action,
                    save_latest_plot_copy_action,
                    open_artifact_folder_action,
                )
            )
        elif str(workflow_id) == "diagnostics":
            actions.extend(
                (
                    None,
                    open_gof_panel_action,
                    open_residual_trends_action,
                    diagnostics_open_artifact_folder_action,
                )
            )
        actions.extend(
            (
                None,
                rename_scenario_action,
                edit_scenario_details_action,
                None,
                delete_scenario_action,
            )
        )
        return tuple(actions)

    def _tree_context_menu_action_names(item) -> tuple[str, ...]:
        return tuple(
            "separator" if action is None else action.objectName() or action.text()
            for action in _tree_context_menu_actions(item)
        )

    def _prepare_tree_context_item(item) -> bool:
        if item is None:
            return False
        if nav.currentItem() is not item:
            nav.setCurrentItem(item)
        return True

    def _show_tree_context_menu(position) -> None:
        item = nav.itemAt(position)
        if not _prepare_tree_context_item(item):
            return
        menu = qt_widgets.QMenu(nav)
        for action in _tree_context_menu_actions(item):
            if action is None:
                menu.addSeparator()
                continue
            menu.addAction(action)
        if not menu.actions():
            return
        menu.exec(nav.viewport().mapToGlobal(position))

    # Track the workflow currently visible so context-sensitive help works.
    _active_workflow_id: list[str | None] = [None]

    def _show_sidebar_details_page(
        *, page_index: int, project_id: str, scenario_id: str | None, status_message: str
    ) -> bool:
        departing = stack.currentWidget()
        on_leave = getattr(departing, "_on_leave", None)
        if callable(on_leave):
            on_leave()
        item = nav_nodes.get(("dashboard", project_id, scenario_id))
        if item is not None:
            blocked = nav.blockSignals(True)
            nav.setCurrentItem(item)
            nav.blockSignals(blocked)
        project.set_active_project(project_id)
        if scenario_id is not None:
            project.set_active_scenario(scenario_id, project_id=project_id)
        _active_workflow_id[0] = "dashboard"
        stack.setCurrentIndex(page_index)
        _refresh_workflow(page_index)
        widget = stack.widget(page_index)
        apply_responsive_layout = getattr(widget, "_apply_responsive_layout", None)
        if callable(apply_responsive_layout):
            apply_responsive_layout(stack.width())
        _update_window_title()
        _status_bar().showMessage(status_message)
        return True

    help_menu = window.menuBar().addMenu("&Help")

    user_guide_action = help_menu.addAction("User Guide")
    user_guide_action.setObjectName("help-user-guide-action")
    user_guide_action.setShortcut("F1")
    user_guide_action.triggered.connect(lambda: open_help_dialog(window, qt_widgets, qt_core))

    workflow_help_action = help_menu.addAction("Help for this workflow")
    workflow_help_action.setObjectName("help-workflow-action")
    workflow_help_action.triggered.connect(
        lambda: open_help_dialog(window, qt_widgets, qt_core, _active_workflow_id[0])
    )

    help_menu.addSeparator()

    about_action = help_menu.addAction("About OpenPKPD GUI…")
    about_action.setObjectName("help-about-action")
    about_action.triggered.connect(lambda: open_about_dialog(window, qt_widgets, qt_core))

    style = window.style()
    file_new_project_action.setIcon(_standard_icon("SP_FileDialogNewFolder"))
    open_action.setIcon(style.standardIcon(qt_widgets.QStyle.StandardPixmap.SP_DialogOpenButton))
    save_action.setIcon(style.standardIcon(qt_widgets.QStyle.StandardPixmap.SP_DialogSaveButton))
    save_as_action.setIcon(style.standardIcon(qt_widgets.QStyle.StandardPixmap.SP_DriveHDIcon))
    close_action.setIcon(style.standardIcon(qt_widgets.QStyle.StandardPixmap.SP_DialogCloseButton))
    new_project_action.setIcon(_standard_icon("SP_FileDialogNewFolder"))
    new_scenario_action.setIcon(_standard_icon("SP_FileDialogNewFolder"))
    import_dataset_action.setIcon(
        style.standardIcon(qt_widgets.QStyle.StandardPixmap.SP_DialogOpenButton)
    )
    open_control_stream_action.setIcon(
        style.standardIcon(qt_widgets.QStyle.StandardPixmap.SP_FileIcon)
    )
    latest_report_action.setIcon(_standard_icon("SP_FileIcon"))
    save_latest_report_copy_action.setIcon(
        style.standardIcon(qt_widgets.QStyle.StandardPixmap.SP_DialogSaveButton)
    )
    export_latest_report_pdf_action.setIcon(
        style.standardIcon(qt_widgets.QStyle.StandardPixmap.SP_DialogSaveButton)
    )
    latest_plot_action.setIcon(_standard_icon("SP_DesktopIcon"))
    save_latest_plot_copy_action.setIcon(
        style.standardIcon(qt_widgets.QStyle.StandardPixmap.SP_DialogSaveButton)
    )
    open_artifact_folder_action.setIcon(_standard_icon("SP_DirOpenIcon"))
    open_gof_panel_action.setIcon(_standard_icon("SP_MessageBoxInformation"))
    open_residual_trends_action.setIcon(_standard_icon("SP_MessageBoxInformation"))
    diagnostics_open_artifact_folder_action.setIcon(_standard_icon("SP_DirOpenIcon"))
    edit_project_details_action.setIcon(_standard_icon("SP_FileDialogDetailedView"))
    edit_scenario_details_action.setIcon(_standard_icon("SP_FileDialogDetailedView"))
    preferences_action.setIcon(_standard_icon("SP_FileDialogContentsView"))

    def _on_item_changed(current, _previous) -> None:
        if current is None:
            return
        departing = stack.currentWidget()
        on_leave = getattr(departing, "_on_leave", None)
        if callable(on_leave):
            on_leave()
        workflow_id = current.data(0, nav_workflow_role)
        project_id = current.data(0, nav_project_role)
        scenario_id = current.data(0, nav_scenario_role)
        if project_id is not None:
            project.set_active_project(str(project_id))
        if scenario_id is not None:
            project.set_active_scenario(
                str(scenario_id), project_id=str(project_id) if project_id else None
            )
        if workflow_id is None:
            return
        if str(workflow_id) == "dashboard":
            if project_id is None:
                workflow_index = home_page_index
            elif scenario_id is None:
                workflow_index = project_details_page_index
            else:
                workflow_index = workflow_pages["dashboard"]
            ready_label = current.text(0)
        else:
            workflow = workflow_definitions.get(str(workflow_id))
            if workflow is None:
                return
            workflow_index = workflow_pages[str(workflow_id)]
            ready_label = workflow.label
        _active_workflow_id[0] = str(workflow_id)
        stack.setCurrentIndex(workflow_index)
        _refresh_workflow(workflow_index)
        widget = stack.widget(workflow_index)
        apply_responsive_layout = getattr(widget, "_apply_responsive_layout", None)
        if callable(apply_responsive_layout):
            apply_responsive_layout(stack.width())
        _update_window_title()
        _status_bar().showMessage(f"{ready_label} ready")

    def _current_active_page_id() -> str | None:
        widget = stack.currentWidget()
        if widget is None:
            return None
        name = widget.objectName()
        return str(name) if name else None

    def _capture_navigation_preferences(preferences: GuiPreferences) -> GuiPreferences:
        return replace(
            preferences,
            nav_selected_item_key=_current_nav_item_key(),
            nav_active_page=_current_active_page_id(),
            nav_expanded_item_keys=_expanded_nav_item_keys(),
        )

    def _restore_saved_navigation_destination(preferences: GuiPreferences) -> None:
        saved_page = preferences.nav_active_page
        saved_item_payload = _parse_nav_item_key(preferences.nav_selected_item_key)
        if saved_page == "project-details-page" and saved_item_payload is not None:
            project_id = saved_item_payload.get("project_id")
            if (
                project_id
                and nav_nodes.get(("dashboard", project_id, None)) is not None
                and _show_sidebar_details_page(
                    page_index=project_details_page_index,
                    project_id=project_id,
                    scenario_id=None,
                    status_message="Project details ready",
                )
            ):
                return
        if saved_page == "scenario-details-page" and saved_item_payload is not None:
            project_id = saved_item_payload.get("project_id")
            scenario_id = saved_item_payload.get("scenario_id")
            if (
                project_id
                and scenario_id
                and nav_nodes.get(("dashboard", project_id, scenario_id)) is not None
                and _show_sidebar_details_page(
                    page_index=scenario_details_page_index,
                    project_id=project_id,
                    scenario_id=scenario_id,
                    status_message="Scenario details ready",
                )
            ):
                return
        if nav.currentItem() is not None:
            _on_item_changed(nav.currentItem(), None)

    nav.currentItemChanged.connect(_on_item_changed)
    nav.customContextMenuRequested.connect(_show_tree_context_menu)

    _autosave_path = Path(user_data_dir("OpenPKPD")) / "autosave.pkp"
    _autosave_timer = qt_core.QTimer(window)

    def _autosave() -> None:
        with contextlib.suppress(Exception):
            snapshot_service.save_snapshot(project, str(_autosave_path))

    def _configure_autosave(interval_minutes: int) -> None:
        if interval_minutes > 0:
            _autosave_timer.setInterval(interval_minutes * 60_000)
            _autosave_timer.start()
        else:
            _autosave_timer.stop()

    _autosave_timer.timeout.connect(_autosave)
    _configure_autosave(current_preferences[0].autosave_interval_minutes)

    def _handle_close_event(event) -> None:
        if _confirm_discard_unsaved_changes("closing the current project"):
            _autosave_timer.stop()
            _autosave_path.unlink(missing_ok=True)
            current_preferences[0] = load_gui_preferences(settings_store=settings_store)
            current_preferences[0] = _capture_navigation_preferences(current_preferences[0])
            current_preferences[0] = _capture_persisted_button_group_preferences(
                window,
                qt_widgets,
                current_preferences[0],
            )
            current_preferences[0] = _capture_named_combo_box_preferences(
                window,
                qt_widgets,
                current_preferences[0],
            )
            current_preferences[0] = _capture_list_widget_selection_preferences(
                window,
                qt_widgets,
                current_preferences[0],
            )
            current_preferences[0] = _capture_named_tab_selection_preferences(
                window,
                qt_widgets,
                current_preferences[0],
            )
            current_preferences[0] = _capture_collapsible_section_preferences(
                window,
                qt_widgets,
                current_preferences[0],
            )
            current_preferences[0] = _capture_named_table_column_width_preferences(
                window,
                qt_widgets,
                current_preferences[0],
            )
            persisted_preferences = _capture_shell_layout_preferences(
                window,
                shell_splitter,
                current_preferences[0],
            )
            save_gui_preferences(persisted_preferences, settings_store=settings_store)
            current_preferences[0] = persisted_preferences
            event.accept()
            return
        event.ignore()

    shell_splitter.setStretchFactor(0, 0)
    shell_splitter.setStretchFactor(1, 1)
    layout.addWidget(shell_splitter, 1)
    window.setCentralWidget(root)
    window.closeEvent = _handle_close_event  # type: ignore[method-assign]
    window._save_project_snapshot = _save_snapshot_to  # type: ignore[attr-defined]
    window._open_project_snapshot = _load_snapshot_from  # type: ignore[attr-defined]
    window._open_project_snapshot_with_prompt = _open_snapshot_with_prompt  # type: ignore[attr-defined]
    window._choose_project_snapshot_to_open = _choose_snapshot_to_open  # type: ignore[attr-defined]
    window._open_recent_snapshot = _open_recent_snapshot  # type: ignore[attr-defined]
    window._save_project_snapshot_as = _choose_snapshot_to_save  # type: ignore[attr-defined]
    window._save_active_project_snapshot = _save_active_project_snapshot  # type: ignore[attr-defined]
    window._save_active_scenario_snapshot = _save_active_scenario_snapshot  # type: ignore[attr-defined]
    window._load_project_snapshot = _load_project_snapshot_from  # type: ignore[attr-defined]
    window._load_scenario_snapshot = _load_scenario_snapshot_from  # type: ignore[attr-defined]
    window._open_latest_report = _open_latest_report  # type: ignore[attr-defined]
    window._export_latest_report_pdf = _export_latest_report_pdf  # type: ignore[attr-defined]
    window._open_latest_plot = _open_latest_plot  # type: ignore[attr-defined]
    window._open_artifact_folder = _open_artifact_folder  # type: ignore[attr-defined]
    window._create_project = _create_project  # type: ignore[attr-defined]
    window._create_scenario = _create_scenario  # type: ignore[attr-defined]
    window._import_dataset = _import_dataset  # type: ignore[attr-defined]
    window._open_control_stream = _open_control_stream  # type: ignore[attr-defined]
    window._rename_project = _rename_project  # type: ignore[attr-defined]
    window._edit_project_details = _edit_project_details  # type: ignore[attr-defined]
    window._rename_scenario = _rename_scenario  # type: ignore[attr-defined]
    window._edit_scenario_details = _edit_scenario_details  # type: ignore[attr-defined]
    window._edit_preferences_dialog = _edit_preferences_dialog  # type: ignore[attr-defined]
    window._open_preferences_dialog = _open_preferences_dialog  # type: ignore[attr-defined]
    window._prepare_tree_context_item = _prepare_tree_context_item  # type: ignore[attr-defined]
    window._show_tree_context_menu = _show_tree_context_menu  # type: ignore[attr-defined]
    window._tree_context_menu_action_names = _tree_context_menu_action_names  # type: ignore[attr-defined]
    window._is_project_dirty = _is_project_dirty  # type: ignore[attr-defined]
    window._refresh_recent_projects_menu = _update_recent_menu  # type: ignore[attr-defined]
    window._select_navigation_item = _select_navigation_item  # type: ignore[attr-defined]
    if _autosave_path.exists():
        app = qt_widgets.QApplication.instance()
        platform_name = app.platformName().lower() if app is not None else ""
        if platform_name in {"offscreen", "minimal"}:
            _autosave_path.unlink(missing_ok=True)
        else:
            reply = qt_widgets.QMessageBox.question(
                window,
                "Restore autosave?",
                "An autosave from a previous session was found.\nRestore it?",
                qt_widgets.QMessageBox.StandardButton.Yes
                | qt_widgets.QMessageBox.StandardButton.No,
            )
            if reply == qt_widgets.QMessageBox.StandardButton.Yes:
                with contextlib.suppress(Exception):
                    _load_snapshot_from(_autosave_path)
            _autosave_path.unlink(missing_ok=True)

    saved_project_snapshot[0] = _project_snapshot()
    _update_recent_menu()
    has_saved_nav_state = any(
        (
            current_preferences[0].nav_selected_item_key,
            current_preferences[0].nav_active_page,
            current_preferences[0].nav_expanded_item_keys,
        )
    )
    _rebuild_navigation_tree(
        workflow_id=None if has_saved_nav_state else "dashboard",
        project_id=None if has_saved_nav_state else project.active_project_id,
        scenario_id=None if has_saved_nav_state else project.active_scenario.scenario_id,
        item_key=current_preferences[0].nav_selected_item_key if has_saved_nav_state else None,
        expanded_item_keys=current_preferences[0].nav_expanded_item_keys
        if has_saved_nav_state
        else None,
    )
    _update_window_title()
    _apply_saved_or_default_window_geometry(window, qt_core, qt_gui, current_preferences[0])
    _apply_saved_or_default_splitter_sizes(
        shell_splitter, current_preferences[0], total_width=window.width()
    )
    _apply_saved_button_group_selections_to_root(window, qt_widgets, current_preferences[0])
    _apply_saved_combo_box_selections_to_root(window, qt_widgets, current_preferences[0])
    _apply_saved_list_widget_selections_to_root(window, qt_widgets, current_preferences[0])
    _apply_saved_tab_selections_to_root(window, qt_widgets, current_preferences[0])
    _apply_saved_collapsible_section_states_to_root(window, qt_widgets, current_preferences[0])
    _apply_saved_table_column_widths_to_root(window, qt_widgets, settings_store=settings_store)
    _restore_saved_navigation_destination(current_preferences[0])
    return window
