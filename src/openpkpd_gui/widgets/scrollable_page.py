"""Helpers for building scrollable workflow pages."""

from __future__ import annotations


def build_scrollable_page(qt_widgets, *, root_object_name: str):
    """Return a root widget with content hosted in a resizable scroll area.

    Returns (root, content, layout, scroll_area, outer_layout).

    ``outer_layout`` is the QVBoxLayout of *root* that sits above the scroll
    area.  Widgets added to ``outer_layout`` (e.g. a sticky header) are pinned
    outside the scrollable region and will not be overlapped by the scroll bar.
    Use ``outer_layout.insertWidget(0, header_widget)`` before
    ``outer_layout.addWidget(scroll_area)`` — or call
    ``outer_layout.insertWidget(index, widget)`` after construction.
    """
    root = qt_widgets.QWidget()
    root.setObjectName(root_object_name)

    outer_layout = qt_widgets.QVBoxLayout(root)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    outer_layout.setSpacing(0)

    scroll_area = qt_widgets.QScrollArea(root)
    scroll_area.setObjectName(f"{root_object_name}-scroll-area")
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(qt_widgets.QFrame.Shape.NoFrame)

    content = qt_widgets.QWidget(scroll_area)
    content.setObjectName(f"{root_object_name}-content")
    layout = qt_widgets.QVBoxLayout(content)

    scroll_area.setWidget(content)
    outer_layout.addWidget(scroll_area)
    return root, content, layout, scroll_area, outer_layout
