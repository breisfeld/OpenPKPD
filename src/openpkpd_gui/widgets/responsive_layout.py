"""Helpers for workflow pages with narrow-width responsive box layouts."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from openpkpd_gui.app.runtime import load_qt_modules


def install_responsive_box_layouts(
    root,
    *,
    breakpoint: int,
    width_provider: Callable[[], int],
    layouts: Iterable[object],
    on_mode_changed: Callable[[bool], None] | None = None,
):
    """Install a resize-driven responsive direction toggle for box layouts."""
    _qt_core, _, qt_widgets = load_qt_modules()
    tracked_layouts = tuple(layouts)

    def _apply_responsive_layout(width: int | None = None) -> None:
        available_width = width if width is not None else width_provider()
        compact = available_width < breakpoint
        direction = (
            qt_widgets.QBoxLayout.Direction.TopToBottom
            if compact
            else qt_widgets.QBoxLayout.Direction.LeftToRight
        )
        for layout in tracked_layouts:
            layout.setDirection(direction)
        if on_mode_changed is not None:
            on_mode_changed(compact)

    original_resize_event = root.resizeEvent

    def _resize_event(event) -> None:
        original_resize_event(event)
        _apply_responsive_layout()

    root.resizeEvent = _resize_event  # type: ignore[method-assign]
    return _apply_responsive_layout


def install_responsive_splitters(
    root,
    *,
    breakpoint: int,
    width_provider: Callable[[], int],
    splitters: Iterable[object],
    on_mode_changed: Callable[[bool], None] | None = None,
):
    """Install a resize-driven responsive orientation toggle for splitters."""
    qt_core, _, _qt_widgets = load_qt_modules()
    tracked_splitters = tuple(splitters)
    compact_state: dict[str, bool | None] = {"value": None}

    def _apply_responsive_splitter(width: int | None = None) -> None:
        available_width = width if width is not None else width_provider()
        compact = available_width < breakpoint
        if compact_state["value"] != compact:
            orientation = (
                qt_core.Qt.Orientation.Vertical if compact else qt_core.Qt.Orientation.Horizontal
            )
            for splitter in tracked_splitters:
                splitter.setOrientation(orientation)
            compact_state["value"] = compact
        if on_mode_changed is not None:
            on_mode_changed(compact)

    original_resize_event = root.resizeEvent

    def _resize_event(event) -> None:
        original_resize_event(event)
        _apply_responsive_splitter()

    root.resizeEvent = _resize_event  # type: ignore[method-assign]
    return _apply_responsive_splitter
