"""Theme definitions (light / dark) for the OpenPKPD desktop GUI."""

from __future__ import annotations

LIGHT = "light"
DARK = "dark"
VALID_THEMES = (LIGHT, DARK)


def _light_colors() -> dict[str, str]:
    return {
        "window": "#f3f6fb",
        "sidebar_bg": "#ffffff",
        "border": "#d6deeb",
        "text": "#0f172a",
        "text_muted": "#64748b",
        "text_accent": "#1d4ed8",
        "input_bg": "#ffffff",
        "panel_bg": "#ffffff",
        "nav_item_selected_bg": "#e0ebff",
        "nav_item_selected_text": "#1d4ed8",
        "tab_bg": "#e8eef9",
        "tab_text": "#334155",
        "tab_selected_bg": "#ffffff",
        "tab_selected_text": "#1d4ed8",
        "button_bg": "#eff6ff",
        "button_text": "#1e3a8a",
        "button_border": "#93c5fd",
        "button_hover_bg": "#dbeafe",
        "button_hover_border": "#60a5fa",
        "button_hover_text": "#1e40af",
        "button_focus_bg": "#e0ebff",
        "button_focus_border": "#2563eb",
        "button_pressed_bg": "#bfdbfe",
        "button_pressed_border": "#1d4ed8",
        "button_disabled_bg": "#e2e8f0",
        "button_disabled_text": "#64748b",
        "button_disabled_border": "#cbd5e1",
        "primary_bg": "#2563eb",
        "primary_text": "#ffffff",
        "primary_border": "#1d4ed8",
        "primary_hover_bg": "#1d4ed8",
        "primary_hover_border": "#1e40af",
        "primary_pressed_bg": "#1e40af",
        "primary_pressed_border": "#1e3a8a",
        "status_bar_bg": "#ffffff",
        "menu_bg": "#ffffff",
        "menu_item_selected_bg": "#eff6ff",
        "menu_item_selected_text": "#1d4ed8",
        "alt_base": "#eef2ff",
        "highlight": "#2563eb",
        "highlight_text": "#ffffff",
        "collapsible_header_hover": "#f8fafc",
        "collapsible_header_focus": "#f1f5f9",
        "collapsible_header_focus_border": "#cbd5e1",
        "collapsible_header_pressed": "#e2e8f0",
        "collapsible_content_border": "#e2e8f0",
    }


def _dark_colors() -> dict[str, str]:
    return {
        "window": "#0f172a",
        "sidebar_bg": "#1e293b",
        "border": "#334155",
        "text": "#e2e8f0",
        "text_muted": "#94a3b8",
        "text_accent": "#60a5fa",
        "input_bg": "#1e293b",
        "panel_bg": "#1e293b",
        "nav_item_selected_bg": "#1e3a5f",
        "nav_item_selected_text": "#93c5fd",
        "tab_bg": "#1a2744",
        "tab_text": "#94a3b8",
        "tab_selected_bg": "#1e293b",
        "tab_selected_text": "#60a5fa",
        "button_bg": "#1e3a5f",
        "button_text": "#93c5fd",
        "button_border": "#2563eb",
        "button_hover_bg": "#1e40af",
        "button_hover_border": "#3b82f6",
        "button_hover_text": "#bfdbfe",
        "button_focus_bg": "#1e3a5f",
        "button_focus_border": "#60a5fa",
        "button_pressed_bg": "#1e3a8a",
        "button_pressed_border": "#93c5fd",
        "button_disabled_bg": "#1e293b",
        "button_disabled_text": "#475569",
        "button_disabled_border": "#334155",
        "primary_bg": "#2563eb",
        "primary_text": "#ffffff",
        "primary_border": "#1d4ed8",
        "primary_hover_bg": "#1d4ed8",
        "primary_hover_border": "#1e40af",
        "primary_pressed_bg": "#1e40af",
        "primary_pressed_border": "#1e3a8a",
        "status_bar_bg": "#1e293b",
        "menu_bg": "#1e293b",
        "menu_item_selected_bg": "#1e3a5f",
        "menu_item_selected_text": "#93c5fd",
        "alt_base": "#1a2744",
        "highlight": "#3b82f6",
        "highlight_text": "#ffffff",
        "collapsible_header_hover": "#263344",
        "collapsible_header_focus": "#2d4a6e",
        "collapsible_header_focus_border": "#334155",
        "collapsible_header_pressed": "#1e3a5f",
        "collapsible_content_border": "#334155",
    }


def build_palette(theme: str, qt_gui):
    """Build a QPalette for *theme*."""
    c = _dark_colors() if theme == DARK else _light_colors()
    palette = qt_gui.QPalette()
    cr = qt_gui.QPalette.ColorRole

    def qcolor(key: str):
        return qt_gui.QColor(c[key])

    palette.setColor(cr.Window, qcolor("window"))
    palette.setColor(cr.WindowText, qcolor("text"))
    palette.setColor(cr.Base, qcolor("input_bg"))
    palette.setColor(cr.AlternateBase, qcolor("alt_base"))
    palette.setColor(cr.ToolTipBase, qcolor("panel_bg"))
    palette.setColor(cr.ToolTipText, qcolor("text"))
    palette.setColor(cr.Text, qcolor("text"))
    palette.setColor(cr.Button, qcolor("button_bg"))
    palette.setColor(cr.ButtonText, qcolor("text_accent"))
    palette.setColor(cr.Highlight, qcolor("highlight"))
    palette.setColor(cr.HighlightedText, qcolor("highlight_text"))
    return palette


def build_stylesheet(theme: str) -> str:
    """Build the full application CSS stylesheet for *theme*."""
    c = _dark_colors() if theme == DARK else _light_colors()
    return f"""
        QMainWindow {{ background: {c["window"]}; }}
        QMenuBar#main-menu-bar {{
            background: {c["sidebar_bg"]};
            border-bottom: 1px solid {c["border"]};
            padding: 4px 8px;
        }}
        QMenuBar#main-menu-bar::item {{
            background: transparent;
            padding: 6px 10px;
            border-radius: 8px;
        }}
        QMenuBar#main-menu-bar::item:selected {{ background: {c["menu_item_selected_bg"]}; }}
        QMenu {{
            background: {c["menu_bg"]};
            border: 1px solid {c["border"]};
            padding: 6px;
        }}
        QMenu::item {{
            padding: 7px 20px 7px 12px;
            border-radius: 8px;
            color: {c["text"]};
        }}
        QMenu::item:selected {{ background: {c["menu_item_selected_bg"]}; color: {c["menu_item_selected_text"]}; }}
        QWidget#shell-sidebar {{
            background: {c["sidebar_bg"]};
            border: 1px solid {c["border"]};
            border-radius: 18px;
        }}
        QLabel#sidebar-app-title {{
            color: {c["text_accent"]};
            font-size: 18px;
            font-weight: 700;
        }}
        QLabel#sidebar-project-name {{
            color: {c["text"]};
            font-size: 16px;
            font-weight: 700;
        }}
        QLabel#sidebar-project-path,
        QLabel#workflow-nav-section-header {{
            color: {c["text_muted"]};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        QListWidget#workflow-nav,
        QStackedWidget#workflow-stack,
        QListWidget#results-runs-list,
        QListWidget#results-artifacts-list,
        QTableWidget,
        QPlainTextEdit,
        QTextBrowser,
        QLineEdit,
        QComboBox,
        QAbstractSpinBox {{
            background: {c["input_bg"]};
            border: 1px solid {c["border"]};
            border-radius: 10px;
            padding: 6px;
            color: {c["text"]};
        }}
        QListWidget#workflow-nav {{
            padding: 4px;
            border: none;
            background: transparent;
        }}
        QListWidget#workflow-nav::item {{
            margin: 2px 0;
            padding: 10px 12px;
            border-radius: 12px;
        }}
        QListWidget#workflow-nav::item:selected {{
            background: {c["nav_item_selected_bg"]};
            color: {c["nav_item_selected_text"]};
            font-weight: 700;
        }}
        QStackedWidget#workflow-stack {{
            border-radius: 18px;
            padding: 8px;
        }}
        QPushButton {{
            background: {c["button_bg"]};
            color: {c["button_text"]};
            border: 2px solid {c["button_border"]};
            border-radius: 10px;
            padding: 8px 16px;
            font-weight: 700;
            min-height: 18px;
        }}
        QPushButton:hover:!disabled {{
            background: {c["button_hover_bg"]};
            border-color: {c["button_hover_border"]};
            color: {c["button_hover_text"]};
        }}
        QPushButton:focus:!disabled {{
            background: {c["button_focus_bg"]};
            border-color: {c["button_focus_border"]};
        }}
        QPushButton:pressed:!disabled,
        QPushButton:checked:!disabled {{
            background: {c["button_pressed_bg"]};
            border-color: {c["button_pressed_border"]};
            color: {c["button_text"]};
            padding-top: 9px;
            padding-bottom: 7px;
        }}
        QPushButton:default:!disabled {{
            background: {c["primary_bg"]};
            color: {c["primary_text"]};
            border-color: {c["primary_border"]};
        }}
        QPushButton:default:hover:!disabled {{
            background: {c["primary_hover_bg"]};
            border-color: {c["primary_hover_border"]};
        }}
        QPushButton:default:pressed:!disabled,
        QPushButton:default:checked:!disabled {{
            background: {c["primary_pressed_bg"]};
            border-color: {c["primary_pressed_border"]};
        }}
        QPushButton[primaryAction="true"] {{
            background: {c["primary_bg"]};
            color: {c["primary_text"]};
            border-color: {c["primary_border"]};
        }}
        QPushButton[primaryAction="true"]:hover:!disabled {{
            background: {c["primary_hover_bg"]};
            border-color: {c["primary_hover_border"]};
            color: {c["primary_text"]};
        }}
        QPushButton[primaryAction="true"]:focus:!disabled {{
            background: {c["primary_bg"]};
            border-color: {c["primary_pressed_border"]};
        }}
        QPushButton[primaryAction="true"]:pressed:!disabled,
        QPushButton[primaryAction="true"]:checked:!disabled {{
            background: {c["primary_pressed_bg"]};
            border-color: {c["primary_pressed_border"]};
            color: {c["primary_text"]};
        }}
        QPushButton:disabled {{
            background: {c["button_disabled_bg"]};
            color: {c["button_disabled_text"]};
            border-color: {c["button_disabled_border"]};
        }}
        QWidget[surfaceRole="workflow-header"],
        QWidget[surfaceRole="workflow-status-strip"],
        QWidget#data-columns-panel,
        QWidget#data-preview-panel,
        QWidget#data-validation-panel,
        QWidget#fit-preparation-panel,
        QWidget#fit-run-panel,
        QWidget#model-configuration-panel,
        QWidget#model-translation-panel,
        QWidget#nca-readiness-panel,
        QWidget#nca-results-panel,
        QWidget#results-detail-panel,
        QWidget#results-artifact-panel,
        QWidget#diagnostics-artifact-list-panel,
        QWidget#diagnostics-artifact-preview-panel,
        QWidget#plots-list-panel,
        QWidget#plots-preview-panel,
        QWidget#covariate-configuration-panel,
        QWidget#covariate-results-panel,
        QWidget#advanced-artifact-list-panel,
        QWidget#advanced-artifact-preview-panel,
        QFrame#overview-hero-panel,
        QWidget#overview-content QGroupBox,
        QTabWidget#advanced-tab-widget::pane {{
            background: {c["panel_bg"]};
            border: 1px solid {c["border"]};
            border-radius: 14px;
        }}
        QTabWidget#advanced-tab-widget::pane {{
            margin-top: 8px;
            padding: 8px;
        }}
        QTabBar::tab {{
            background: {c["tab_bg"]};
            color: {c["tab_text"]};
            border: 1px solid {c["border"]};
            border-bottom: none;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            padding: 7px 12px;
            margin-right: 4px;
        }}
        QTabBar::tab:selected {{
            background: {c["tab_selected_bg"]};
            color: {c["tab_selected_text"]};
            font-weight: 700;
        }}
        QTabBar::tab:!selected {{
            margin-top: 2px;
        }}
        QGroupBox {{
            font-weight: 700;
            margin-top: 10px;
            padding-top: 6px;
            color: {c["text"]};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
            color: {c["text"]};
        }}
        QToolButton {{
            background: {c["button_bg"]};
            color: {c["button_text"]};
            border: 2px solid {c["button_border"]};
            border-radius: 10px;
            padding: 8px 14px;
            font-weight: 700;
            min-height: 18px;
        }}
        QToolButton:hover:!disabled {{
            background: {c["button_hover_bg"]};
            border-color: {c["button_hover_border"]};
            color: {c["button_hover_text"]};
        }}
        QToolButton:focus:!disabled {{
            background: {c["button_focus_bg"]};
            border-color: {c["button_focus_border"]};
        }}
        QToolButton:pressed:!disabled,
        QToolButton:checked:!disabled {{
            background: {c["button_pressed_bg"]};
            border-color: {c["button_pressed_border"]};
            color: {c["button_text"]};
            padding-top: 9px;
            padding-bottom: 7px;
        }}
        QToolButton:disabled {{
            background: {c["button_disabled_bg"]};
            color: {c["button_disabled_text"]};
            border-color: {c["button_disabled_border"]};
        }}
        QToolButton#dismissible-hint-dismiss {{
            padding: 0px 2px;
            border-radius: 5px;
            min-height: 0;
            font-size: 13px;
        }}
        QWidget[collapsibleSection="true"][collapsibleFrame="true"] {{
            background: {c["panel_bg"]};
            border: 1px solid {c["border"]};
            border-radius: 14px;
        }}
        QToolButton[collapsibleHeader="true"] {{
            background: transparent;
            border: none;
            color: {c["text"]};
            font-weight: 700;
            padding: 10px 12px;
            text-align: left;
        }}
        QToolButton[collapsibleHeader="true"]:hover {{
            background: {c["collapsible_header_hover"]};
        }}
        QToolButton[collapsibleHeader="true"]:focus {{
            background: {c["collapsible_header_focus"]};
            border: 1px solid {c["collapsible_header_focus_border"]};
        }}
        QToolButton[collapsibleHeader="true"]:pressed,
        QToolButton[collapsibleHeader="true"]:checked {{
            background: {c["collapsible_header_pressed"]};
            color: {c["text"]};
        }}
        QWidget[collapsibleContent="true"] {{
            border-top: 1px solid {c["collapsible_content_border"]};
        }}
        QLabel#results-overview-label,
        QLabel#diagnostics-overview-label,
        QLabel#plots-overview-label,
        QLabel#plots-status-label,
        QLabel#data-example-details,
        QLabel#data-import-summary-label,
        QLabel#fit-preparation-summary,
        QLabel#fit-run-summary,
        QLabel#model-parameter-summary,
        QLabel#model-translation-summary,
        QLabel#nca-preparation-summary,
        QLabel#nca-run-summary,
        QLabel#nca-results-summary,
        QLabel#diagnostics-status-label,
        QLabel#diagnostics-next-steps-label,
        QLabel#diagnostics-npde-status-label,
        QLabel#covariate-status-label,
        QLabel#advanced-vpc-status-label,
        QLabel#advanced-bootstrap-status-label,
        QLabel#advanced-design-status-label,
        QLabel#model-summary-label,
        QLabel#data-summary-label {{
            background: {c["panel_bg"]};
            border: 1px solid {c["border"]};
            border-radius: 14px;
            padding: 10px 12px;
            color: {c["text"]};
        }}
        QLabel#overview-eyebrow-label,
        QLabel#sidebar-project-path,
        QLabel#workflow-nav-section-header {{
            color: {c["text_muted"]};
        }}
        QStatusBar {{ background: {c["status_bar_bg"]}; color: {c["text"]}; }}
        QLabel {{ color: {c["text"]}; }}
        QCheckBox {{ color: {c["text"]}; }}
        QRadioButton {{ color: {c["text"]}; }}
    """
