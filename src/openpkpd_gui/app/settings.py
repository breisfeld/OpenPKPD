"""Persistent application-level GUI preferences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from platformdirs import user_data_dir

from openpkpd_gui.app.runtime import load_qt_modules

FONT_SIZE_KEY = "appearance/font_point_size"
THEME_KEY = "appearance/theme"
DEFAULT_WORKSPACE_ROOT_KEY = "paths/default_workspace_root"
N_PARALLEL_KEY = "performance/n_parallel"
AUTOSAVE_INTERVAL_KEY = "performance/autosave_interval_minutes"
DISMISSED_HINTS_KEY = "ui/dismissed_hints"
WINDOW_X_KEY = "window/x"
WINDOW_Y_KEY = "window/y"
WINDOW_WIDTH_KEY = "window/width"
WINDOW_HEIGHT_KEY = "window/height"
WINDOW_MAXIMIZED_KEY = "window/maximized"
WINDOW_SPLITTER_SIZES_KEY = "window/splitter_sizes"
NAV_SELECTED_ITEM_KEY = "nav/selected_item_key"
NAV_ACTIVE_PAGE_KEY = "nav/active_page"
NAV_EXPANDED_ITEM_KEYS_KEY = "nav/expanded_item_keys"
LAST_FILE_DIALOG_DIR_KEY = "paths/last_file_dialog_dir"
TABLE_COLUMN_WIDTHS_KEY = "ui/table_column_widths"
TAB_SELECTIONS_KEY = "ui/tab_selections"
COLLAPSIBLE_SECTION_STATES_KEY = "ui/collapsible_section_states"
COMBO_BOX_SELECTIONS_KEY = "ui/combo_box_selections"
LIST_WIDGET_SELECTIONS_KEY = "ui/list_widget_selections"
BUTTON_GROUP_SELECTIONS_KEY = "ui/button_group_selections"
DEFAULT_FONT_SIZE_PROPERTY = "openpkpd_default_font_point_size"
SETTINGS_STORE_PROPERTY = "openpkpd_settings_store"
DEFAULT_WORKSPACE_DIRNAME = "OpenPKPD"
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 24


@dataclass(frozen=True)
class GuiPreferences:
    """Stored user preferences for the desktop GUI."""

    font_size: int | None = None
    theme: str = "light"
    default_workspace_root: str | None = None
    n_parallel: int = 0
    autosave_interval_minutes: int = 5
    dismissed_hints: frozenset[str] = frozenset()
    window_x: int | None = None
    window_y: int | None = None
    window_width: int | None = None
    window_height: int | None = None
    window_maximized: bool = False
    window_splitter_sizes: tuple[int, ...] = ()
    nav_selected_item_key: str | None = None
    nav_active_page: str | None = None
    nav_expanded_item_keys: tuple[str, ...] = ()
    last_file_dialog_dir: str | None = None
    table_column_widths: dict[str, tuple[int, ...]] = field(default_factory=dict)
    tab_selections: dict[str, int] = field(default_factory=dict)
    collapsible_section_states: dict[str, bool] = field(default_factory=dict)
    combo_box_selections: dict[str, str] = field(default_factory=dict)
    list_widget_selections: dict[str, str] = field(default_factory=dict)
    button_group_selections: dict[str, str] = field(default_factory=dict)


def with_dismissed_hint(preferences: GuiPreferences, hint_id: str) -> GuiPreferences:
    """Return a copy of *preferences* with *hint_id* added to dismissed_hints."""
    return replace(preferences, dismissed_hints=preferences.dismissed_hints | {hint_id})


def _coerce_font_size(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        size = int(value)
    except (TypeError, ValueError):
        return None
    return max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, size))


def _coerce_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_window_extent(value: object, *, minimum: int) -> int | None:
    extent = _coerce_int(value)
    if extent is None:
        return None
    return max(minimum, extent)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_splitter_sizes(value: object) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",")]
    elif isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        return ()
    sizes: list[int] = []
    for token in tokens:
        try:
            size = int(token)
        except (TypeError, ValueError):
            continue
        if size > 0:
            sizes.append(size)
    return tuple(sizes)


def _coerce_string(value: object) -> str | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    return text or None


def _coerce_string_tuple_json(value: object) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (list, tuple)):
        values = value
    else:
        try:
            values = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return ()
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(str(item) for item in values if str(item).strip())


def _coerce_table_column_widths(value: object) -> dict[str, tuple[int, ...]]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(payload, dict):
        return {}
    restored: dict[str, tuple[int, ...]] = {}
    for key, raw_sizes in payload.items():
        if not isinstance(key, str) or not key.strip():
            continue
        restored_sizes = _coerce_splitter_sizes(raw_sizes)
        if restored_sizes:
            restored[key] = restored_sizes
    return restored


def _coerce_tab_selections(value: object) -> dict[str, int]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(payload, dict):
        return {}
    restored: dict[str, int] = {}
    for key, raw_index in payload.items():
        if not isinstance(key, str) or not key.strip():
            continue
        index = _coerce_int(raw_index)
        if index is not None and index >= 0:
            restored[key] = index
    return restored


def _coerce_collapsible_section_states(value: object) -> dict[str, bool]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(payload, dict):
        return {}
    restored: dict[str, bool] = {}
    for key, raw_state in payload.items():
        if not isinstance(key, str) or not key.strip():
            continue
        restored[key] = _coerce_bool(raw_state)
    return restored


def _coerce_combo_box_selections(value: object) -> dict[str, str]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(payload, dict):
        return {}
    restored: dict[str, str] = {}
    for key, raw_text in payload.items():
        if not isinstance(key, str) or not key.strip():
            continue
        text = _coerce_string(raw_text)
        if text is not None:
            restored[key] = text
    return restored


def _coerce_string_mapping(value: object) -> dict[str, str]:
    return _coerce_combo_box_selections(value)


def normalize_directory_path(value: object) -> str | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def default_workspace_root_path() -> Path:
    """Return the platform-specific user-data location for rootless GUI workspaces."""

    return Path(user_data_dir(DEFAULT_WORKSPACE_DIRNAME)).resolve()


def default_settings_store():
    qt_core, _, _ = load_qt_modules()
    return qt_core.QSettings()


def resolve_settings_store(*, settings_store=None):
    if settings_store is not None:
        return settings_store
    try:
        _, _, qt_widgets = load_qt_modules()
        app = qt_widgets.QApplication.instance()
    except Exception:
        app = None
    if app is not None:
        # Use a plain Python attribute instead of a Qt dynamic property here.
        # Storing a live QSettings wrapper as a Qt property can crash under
        # PySide6 during startup on some environments.
        app_settings_store = getattr(app, SETTINGS_STORE_PROPERTY, None)
        if app_settings_store is not None:
            return app_settings_store
    return default_settings_store()


def load_gui_preferences(*, settings_store=None) -> GuiPreferences:
    """Load persisted GUI preferences from the Qt settings store."""

    store = resolve_settings_store(settings_store=settings_store)
    try:
        n_parallel = int(store.value(N_PARALLEL_KEY, 0))
    except (TypeError, ValueError):
        n_parallel = 0
    try:
        autosave_interval_minutes = int(store.value(AUTOSAVE_INTERVAL_KEY, 5))
    except (TypeError, ValueError):
        autosave_interval_minutes = 5
    raw_hints = store.value(DISMISSED_HINTS_KEY, "") or ""
    dismissed_hints: frozenset[str] = frozenset(
        h.strip() for h in str(raw_hints).split(",") if h.strip()
    )
    from openpkpd_gui.app.theme import VALID_THEMES

    raw_theme = str(store.value(THEME_KEY, "light") or "light").strip()
    theme = raw_theme if raw_theme in VALID_THEMES else "light"
    return GuiPreferences(
        font_size=_coerce_font_size(store.value(FONT_SIZE_KEY, None)),
        theme=theme,
        default_workspace_root=normalize_directory_path(
            store.value(DEFAULT_WORKSPACE_ROOT_KEY, None)
        ),
        n_parallel=max(0, n_parallel),
        autosave_interval_minutes=max(0, min(60, autosave_interval_minutes)),
        dismissed_hints=dismissed_hints,
        window_x=_coerce_int(store.value(WINDOW_X_KEY, None)),
        window_y=_coerce_int(store.value(WINDOW_Y_KEY, None)),
        window_width=_coerce_window_extent(store.value(WINDOW_WIDTH_KEY, None), minimum=320),
        window_height=_coerce_window_extent(store.value(WINDOW_HEIGHT_KEY, None), minimum=240),
        window_maximized=_coerce_bool(store.value(WINDOW_MAXIMIZED_KEY, False)),
        window_splitter_sizes=_coerce_splitter_sizes(store.value(WINDOW_SPLITTER_SIZES_KEY, "")),
        nav_selected_item_key=_coerce_string(store.value(NAV_SELECTED_ITEM_KEY, None)),
        nav_active_page=_coerce_string(store.value(NAV_ACTIVE_PAGE_KEY, None)),
        nav_expanded_item_keys=_coerce_string_tuple_json(
            store.value(NAV_EXPANDED_ITEM_KEYS_KEY, "")
        ),
        last_file_dialog_dir=normalize_directory_path(store.value(LAST_FILE_DIALOG_DIR_KEY, None)),
        table_column_widths=_coerce_table_column_widths(store.value(TABLE_COLUMN_WIDTHS_KEY, "")),
        tab_selections=_coerce_tab_selections(store.value(TAB_SELECTIONS_KEY, "")),
        collapsible_section_states=_coerce_collapsible_section_states(
            store.value(COLLAPSIBLE_SECTION_STATES_KEY, "")
        ),
        combo_box_selections=_coerce_combo_box_selections(
            store.value(COMBO_BOX_SELECTIONS_KEY, "")
        ),
        list_widget_selections=_coerce_string_mapping(store.value(LIST_WIDGET_SELECTIONS_KEY, "")),
        button_group_selections=_coerce_string_mapping(
            store.value(BUTTON_GROUP_SELECTIONS_KEY, "")
        ),
    )


def save_gui_preferences(preferences: GuiPreferences, *, settings_store=None) -> None:
    """Persist GUI preferences into the Qt settings store."""

    from openpkpd_gui.app.theme import VALID_THEMES

    store = resolve_settings_store(settings_store=settings_store)
    font_size = _coerce_font_size(preferences.font_size)
    default_workspace_root = normalize_directory_path(preferences.default_workspace_root)
    last_file_dialog_dir = normalize_directory_path(preferences.last_file_dialog_dir)
    if font_size is None:
        store.remove(FONT_SIZE_KEY)
    else:
        store.setValue(FONT_SIZE_KEY, font_size)
    theme = preferences.theme if preferences.theme in VALID_THEMES else "light"
    store.setValue(THEME_KEY, theme)
    if default_workspace_root is None:
        store.remove(DEFAULT_WORKSPACE_ROOT_KEY)
    else:
        store.setValue(DEFAULT_WORKSPACE_ROOT_KEY, default_workspace_root)
    store.setValue(N_PARALLEL_KEY, max(0, preferences.n_parallel))
    store.setValue(AUTOSAVE_INTERVAL_KEY, max(0, min(60, preferences.autosave_interval_minutes)))
    store.setValue(DISMISSED_HINTS_KEY, ",".join(sorted(preferences.dismissed_hints)))
    if preferences.window_x is None:
        store.remove(WINDOW_X_KEY)
    else:
        store.setValue(WINDOW_X_KEY, preferences.window_x)
    if preferences.window_y is None:
        store.remove(WINDOW_Y_KEY)
    else:
        store.setValue(WINDOW_Y_KEY, preferences.window_y)
    if preferences.window_width is None:
        store.remove(WINDOW_WIDTH_KEY)
    else:
        store.setValue(WINDOW_WIDTH_KEY, max(320, preferences.window_width))
    if preferences.window_height is None:
        store.remove(WINDOW_HEIGHT_KEY)
    else:
        store.setValue(WINDOW_HEIGHT_KEY, max(240, preferences.window_height))
    store.setValue(WINDOW_MAXIMIZED_KEY, bool(preferences.window_maximized))
    if preferences.window_splitter_sizes:
        store.setValue(
            WINDOW_SPLITTER_SIZES_KEY,
            ",".join(str(max(1, int(size))) for size in preferences.window_splitter_sizes),
        )
    else:
        store.remove(WINDOW_SPLITTER_SIZES_KEY)
    if preferences.nav_selected_item_key is None:
        store.remove(NAV_SELECTED_ITEM_KEY)
    else:
        store.setValue(NAV_SELECTED_ITEM_KEY, preferences.nav_selected_item_key)
    if preferences.nav_active_page is None:
        store.remove(NAV_ACTIVE_PAGE_KEY)
    else:
        store.setValue(NAV_ACTIVE_PAGE_KEY, preferences.nav_active_page)
    if preferences.nav_expanded_item_keys:
        store.setValue(
            NAV_EXPANDED_ITEM_KEYS_KEY, json.dumps(list(preferences.nav_expanded_item_keys))
        )
    else:
        store.remove(NAV_EXPANDED_ITEM_KEYS_KEY)
    if last_file_dialog_dir is None:
        store.remove(LAST_FILE_DIALOG_DIR_KEY)
    else:
        store.setValue(LAST_FILE_DIALOG_DIR_KEY, last_file_dialog_dir)
    if preferences.table_column_widths:
        store.setValue(
            TABLE_COLUMN_WIDTHS_KEY,
            json.dumps(
                {
                    key: [max(1, int(size)) for size in sizes if int(size) > 0]
                    for key, sizes in sorted(preferences.table_column_widths.items())
                    if key.strip() and sizes
                },
                sort_keys=True,
            ),
        )
    else:
        store.remove(TABLE_COLUMN_WIDTHS_KEY)
    if preferences.tab_selections:
        store.setValue(
            TAB_SELECTIONS_KEY,
            json.dumps(
                {
                    key: int(index)
                    for key, index in sorted(preferences.tab_selections.items())
                    if key.strip() and int(index) >= 0
                },
                sort_keys=True,
            ),
        )
    else:
        store.remove(TAB_SELECTIONS_KEY)
    if preferences.collapsible_section_states:
        store.setValue(
            COLLAPSIBLE_SECTION_STATES_KEY,
            json.dumps(
                {
                    key: bool(is_expanded)
                    for key, is_expanded in sorted(preferences.collapsible_section_states.items())
                    if key.strip()
                },
                sort_keys=True,
            ),
        )
    else:
        store.remove(COLLAPSIBLE_SECTION_STATES_KEY)
    if preferences.combo_box_selections:
        store.setValue(
            COMBO_BOX_SELECTIONS_KEY,
            json.dumps(
                {
                    key: text
                    for key, text in sorted(preferences.combo_box_selections.items())
                    if key.strip() and str(text).strip()
                },
                sort_keys=True,
            ),
        )
    else:
        store.remove(COMBO_BOX_SELECTIONS_KEY)
    if preferences.list_widget_selections:
        store.setValue(
            LIST_WIDGET_SELECTIONS_KEY,
            json.dumps(
                {
                    key: text
                    for key, text in sorted(preferences.list_widget_selections.items())
                    if key.strip() and str(text).strip()
                },
                sort_keys=True,
            ),
        )
    else:
        store.remove(LIST_WIDGET_SELECTIONS_KEY)
    if preferences.button_group_selections:
        store.setValue(
            BUTTON_GROUP_SELECTIONS_KEY,
            json.dumps(
                {
                    key: text
                    for key, text in sorted(preferences.button_group_selections.items())
                    if key.strip() and str(text).strip()
                },
                sort_keys=True,
            ),
        )
    else:
        store.remove(BUTTON_GROUP_SELECTIONS_KEY)
    sync = getattr(store, "sync", None)
    if callable(sync):
        sync()


def with_last_file_dialog_dir(
    preferences: GuiPreferences,
    selected_path: str | Path | None,
    *,
    selection_is_directory: bool = False,
) -> GuiPreferences:
    if selected_path is None or selected_path == "":
        return preferences
    selected = Path(str(selected_path)).expanduser()
    target_dir = selected if selection_is_directory else selected.parent
    normalized = normalize_directory_path(target_dir)
    if normalized is None:
        return preferences
    return replace(preferences, last_file_dialog_dir=normalized)


def apply_saved_table_column_widths(table, *, settings_store=None) -> None:
    object_name = getattr(table, "objectName", lambda: "")()
    if not object_name:
        return
    preferences = load_gui_preferences(settings_store=settings_store)
    widths = preferences.table_column_widths.get(str(object_name), ())
    if not widths:
        return
    column_count = getattr(table, "columnCount", lambda: 0)()
    if column_count <= 0:
        return
    for index, width in enumerate(widths[:column_count]):
        if int(width) > 0:
            table.setColumnWidth(index, int(width))


def default_font_point_size(app) -> int:
    """Return the app's original default font size for reset behavior."""

    stored_size = app.property(DEFAULT_FONT_SIZE_PROPERTY)
    if isinstance(stored_size, int) and stored_size > 0:
        return stored_size
    font_size = app.font().pointSize()
    if font_size <= 0:
        font_size = 10
    app.setProperty(DEFAULT_FONT_SIZE_PROPERTY, font_size)
    return font_size


def apply_gui_preferences(app, preferences: GuiPreferences) -> int:
    """Apply the persisted preferences to the running application."""
    from openpkpd_gui.app.theme import VALID_THEMES, build_palette, build_stylesheet

    default_size = default_font_point_size(app)
    font_size = _coerce_font_size(preferences.font_size) or default_size
    font = app.font()
    if font.pointSize() != font_size:
        font.setPointSize(font_size)
        app.setFont(font)

    theme = preferences.theme if preferences.theme in VALID_THEMES else "light"
    qt_core, qt_gui, _ = load_qt_modules()
    app.setPalette(build_palette(theme, qt_gui))
    base_ss = build_stylesheet(theme)
    app.setProperty("openpkpd_base_stylesheet", base_ss)
    if font_size != default_size:
        app.setStyleSheet(base_ss + f"\nQWidget {{ font-size: {font_size}pt; }}")
    else:
        app.setStyleSheet(base_ss)
    return font_size


def initialize_gui_preferences(app, *, settings_store=None) -> GuiPreferences:
    """Load and apply persisted preferences to the running application."""

    setattr(app, SETTINGS_STORE_PROPERTY, resolve_settings_store(settings_store=settings_store))
    preferences = load_gui_preferences(settings_store=settings_store)
    apply_gui_preferences(app, preferences)
    return preferences
