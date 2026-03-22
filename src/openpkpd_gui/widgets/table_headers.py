"""Helpers for consistent table header resizing behavior."""

from __future__ import annotations


def configure_resizable_table_columns(
    table, qt_widgets, *, stretch_last_section: bool = False
) -> None:
    """Allow manual column resizing for *table* while preserving optional last-column stretch."""
    header = table.horizontalHeader()
    header.setSectionResizeMode(qt_widgets.QHeaderView.ResizeMode.Interactive)
    for index in range(getattr(table, "columnCount", lambda: 0)()):
        header.setSectionResizeMode(index, qt_widgets.QHeaderView.ResizeMode.Interactive)
    header.setStretchLastSection(stretch_last_section)
