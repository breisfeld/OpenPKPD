"""Unit tests for GUI settings persistence."""

from __future__ import annotations

from unittest.mock import MagicMock

from openpkpd_gui.app.settings import (
    AUTOSAVE_INTERVAL_KEY,
    BUTTON_GROUP_SELECTIONS_KEY,
    COLLAPSIBLE_SECTION_STATES_KEY,
    COMBO_BOX_SELECTIONS_KEY,
    DISMISSED_HINTS_KEY,
    LAST_FILE_DIALOG_DIR_KEY,
    LIST_WIDGET_SELECTIONS_KEY,
    NAV_ACTIVE_PAGE_KEY,
    NAV_EXPANDED_ITEM_KEYS_KEY,
    NAV_SELECTED_ITEM_KEY,
    SETTINGS_STORE_PROPERTY,
    TAB_SELECTIONS_KEY,
    TABLE_COLUMN_WIDTHS_KEY,
    WINDOW_HEIGHT_KEY,
    WINDOW_MAXIMIZED_KEY,
    WINDOW_SPLITTER_SIZES_KEY,
    WINDOW_WIDTH_KEY,
    WINDOW_X_KEY,
    WINDOW_Y_KEY,
    GuiPreferences,
    initialize_gui_preferences,
    load_gui_preferences,
    resolve_settings_store,
    save_gui_preferences,
    with_dismissed_hint,
)


def _mock_store(values: dict | None = None) -> MagicMock:
    """Return a mock QSettings-like store with optional preset values."""
    stored: dict = {}
    if values:
        stored.update(values)

    store = MagicMock()
    store.value.side_effect = lambda key, default=None: stored.get(key, default)
    store.setValue.side_effect = lambda key, value: stored.__setitem__(key, value)
    store.remove.side_effect = lambda key: stored.pop(key, None)
    return store


def test_dismissed_hints_round_trip():
    store = _mock_store()
    prefs = GuiPreferences(dismissed_hints=frozenset({"hint_model", "hint_fit"}))
    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.dismissed_hints == frozenset({"hint_model", "hint_fit"})


def test_dismissed_hints_empty_by_default():
    store = _mock_store()
    loaded = load_gui_preferences(settings_store=store)
    assert loaded.dismissed_hints == frozenset()


def test_dismissed_hints_ignores_blank_tokens():
    store = _mock_store({DISMISSED_HINTS_KEY: ",hint_nca,,hint_data,"})
    loaded = load_gui_preferences(settings_store=store)
    assert loaded.dismissed_hints == frozenset({"hint_nca", "hint_data"})


def test_with_dismissed_hint_adds_id():
    prefs = GuiPreferences(dismissed_hints=frozenset({"hint_model"}))
    updated = with_dismissed_hint(prefs, "hint_fit")
    assert "hint_fit" in updated.dismissed_hints
    assert "hint_model" in updated.dismissed_hints


def test_with_dismissed_hint_is_idempotent():
    prefs = GuiPreferences(dismissed_hints=frozenset({"hint_model"}))
    updated = with_dismissed_hint(prefs, "hint_model")
    assert updated.dismissed_hints == frozenset({"hint_model"})


def test_other_prefs_preserved_across_dismissed_hint_save_load():
    store = _mock_store()
    prefs = GuiPreferences(font_size=14, n_parallel=4, dismissed_hints=frozenset({"hint_plots"}))
    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.font_size == 14
    assert loaded.n_parallel == 4
    assert loaded.dismissed_hints == frozenset({"hint_plots"})


def test_autosave_interval_round_trip():
    store = _mock_store()
    prefs = GuiPreferences(autosave_interval_minutes=10)
    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.autosave_interval_minutes == 10


def test_autosave_interval_defaults_to_five():
    store = _mock_store()
    loaded = load_gui_preferences(settings_store=store)
    assert loaded.autosave_interval_minutes == 5


def test_autosave_interval_clamped_to_range():
    store = _mock_store({AUTOSAVE_INTERVAL_KEY: "999"})
    loaded = load_gui_preferences(settings_store=store)
    assert loaded.autosave_interval_minutes == 60


def test_window_geometry_round_trip() -> None:
    store = _mock_store()
    prefs = GuiPreferences(
        window_x=120,
        window_y=80,
        window_width=1440,
        window_height=900,
        window_maximized=True,
    )

    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.window_x == 120
    assert loaded.window_y == 80
    assert loaded.window_width == 1440
    assert loaded.window_height == 900
    assert loaded.window_maximized is True


def test_window_geometry_keys_removed_when_unset() -> None:
    store = _mock_store(
        {
            WINDOW_X_KEY: 10,
            WINDOW_Y_KEY: 20,
            WINDOW_WIDTH_KEY: 800,
            WINDOW_HEIGHT_KEY: 600,
            WINDOW_MAXIMIZED_KEY: True,
        }
    )

    save_gui_preferences(GuiPreferences(), settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.window_x is None
    assert loaded.window_y is None
    assert loaded.window_width is None
    assert loaded.window_height is None
    assert loaded.window_maximized is False


def test_window_splitter_sizes_round_trip() -> None:
    store = _mock_store()
    prefs = GuiPreferences(window_splitter_sizes=(320, 1180))

    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.window_splitter_sizes == (320, 1180)


def test_window_splitter_sizes_removed_when_unset() -> None:
    store = _mock_store({WINDOW_SPLITTER_SIZES_KEY: "300,1200"})

    save_gui_preferences(GuiPreferences(), settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.window_splitter_sizes == ()


def test_navigation_state_round_trip() -> None:
    store = _mock_store()
    prefs = GuiPreferences(
        nav_selected_item_key='{"kind":"workflow","project_id":"p1","scenario_id":"s1","workflow_id":"model"}',
        nav_active_page="model-workflow",
        nav_expanded_item_keys=(
            '{"kind":"workspace"}',
            '{"kind":"project","project_id":"p1"}',
        ),
    )

    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.nav_selected_item_key == prefs.nav_selected_item_key
    assert loaded.nav_active_page == "model-workflow"
    assert loaded.nav_expanded_item_keys == prefs.nav_expanded_item_keys


def test_navigation_state_keys_removed_when_unset() -> None:
    store = _mock_store(
        {
            NAV_SELECTED_ITEM_KEY: '{"kind":"workspace"}',
            NAV_ACTIVE_PAGE_KEY: "home-workflow",
            NAV_EXPANDED_ITEM_KEYS_KEY: '["{\\"kind\\":\\"workspace\\"}"]',
        }
    )

    save_gui_preferences(GuiPreferences(), settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.nav_selected_item_key is None
    assert loaded.nav_active_page is None
    assert loaded.nav_expanded_item_keys == ()


def test_last_file_dialog_directory_round_trip() -> None:
    from pathlib import Path

    store = _mock_store()
    prefs = GuiPreferences(last_file_dialog_dir="/tmp/openpkpd/dialogs")

    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    # normalize_directory_path calls Path.resolve() which expands symlinks
    # (e.g. /tmp → /private/tmp on macOS). Compare against the resolved form.
    expected = str(Path("/tmp/openpkpd/dialogs").resolve())
    assert loaded.last_file_dialog_dir == expected


def test_table_column_widths_round_trip() -> None:
    store = _mock_store()
    prefs = GuiPreferences(table_column_widths={"model-theta-table": (220, 110, 110, 110, 84)})

    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.table_column_widths == {"model-theta-table": (220, 110, 110, 110, 84)}


def test_tab_selections_round_trip() -> None:
    store = _mock_store()
    prefs = GuiPreferences(tab_selections={"advanced-tab-widget": 2})

    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.tab_selections == {"advanced-tab-widget": 2}


def test_collapsible_section_states_round_trip() -> None:
    store = _mock_store()
    prefs = GuiPreferences(
        collapsible_section_states={
            "advanced-vpc-log-section": True,
            "results-artifact-preview-section": False,
        }
    )

    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.collapsible_section_states == {
        "advanced-vpc-log-section": True,
        "results-artifact-preview-section": False,
    }


def test_combo_box_selections_round_trip() -> None:
    store = _mock_store()
    prefs = GuiPreferences(combo_box_selections={"advanced-artifact-scope-combo": "Bootstrap only"})

    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.combo_box_selections == {"advanced-artifact-scope-combo": "Bootstrap only"}


def test_list_widget_selections_round_trip() -> None:
    store = _mock_store()
    prefs = GuiPreferences(
        list_widget_selections={
            "results-runs-list": "run-2",
            "results-artifacts-list": "artifact-2",
        }
    )

    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.list_widget_selections == {
        "results-runs-list": "run-2",
        "results-artifacts-list": "artifact-2",
    }


def test_button_group_selections_round_trip() -> None:
    store = _mock_store()
    prefs = GuiPreferences(
        button_group_selections={"results-kind-filter": "results-kind-filter-plot"}
    )

    save_gui_preferences(prefs, settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.button_group_selections == {"results-kind-filter": "results-kind-filter-plot"}


def test_resolve_settings_store_uses_cached_app_attribute(monkeypatch) -> None:
    cached_store = object()

    class _DummyApp:
        pass

    app = _DummyApp()
    setattr(app, SETTINGS_STORE_PROPERTY, cached_store)

    class _DummyApplication:
        @staticmethod
        def instance():
            return app

    class _DummyQtWidgets:
        QApplication = _DummyApplication

    monkeypatch.setattr(
        "openpkpd_gui.app.settings.load_qt_modules",
        lambda: (None, None, _DummyQtWidgets),
    )

    assert resolve_settings_store() is cached_store


def test_initialize_gui_preferences_caches_store_on_app_attribute(monkeypatch) -> None:
    class _DummyApp:
        pass

    app = _DummyApp()
    store = object()
    preferences = GuiPreferences(theme="light")

    monkeypatch.setattr("openpkpd_gui.app.settings.load_gui_preferences", lambda **_: preferences)
    monkeypatch.setattr(
        "openpkpd_gui.app.settings.apply_gui_preferences", lambda *_args, **_kwargs: 10
    )

    loaded = initialize_gui_preferences(app, settings_store=store)

    assert loaded == preferences
    assert getattr(app, SETTINGS_STORE_PROPERTY) is store


def test_last_dialog_directory_and_table_widths_removed_when_unset() -> None:
    store = _mock_store(
        {
            LAST_FILE_DIALOG_DIR_KEY: "/tmp/openpkpd/dialogs",
            BUTTON_GROUP_SELECTIONS_KEY: '{"results-kind-filter":"results-kind-filter-plot"}',
            COLLAPSIBLE_SECTION_STATES_KEY: '{"advanced-vpc-log-section":true}',
            COMBO_BOX_SELECTIONS_KEY: '{"advanced-artifact-scope-combo":"Bootstrap only"}',
            LIST_WIDGET_SELECTIONS_KEY: '{"results-runs-list":"run-2"}',
            TAB_SELECTIONS_KEY: '{"advanced-tab-widget":2}',
            TABLE_COLUMN_WIDTHS_KEY: '{"model-theta-table":[220,110,110,110,84]}',
        }
    )

    save_gui_preferences(GuiPreferences(), settings_store=store)

    loaded = load_gui_preferences(settings_store=store)
    assert loaded.last_file_dialog_dir is None
    assert loaded.button_group_selections == {}
    assert loaded.collapsible_section_states == {}
    assert loaded.combo_box_selections == {}
    assert loaded.list_widget_selections == {}
    assert loaded.tab_selections == {}
    assert loaded.table_column_widths == {}
