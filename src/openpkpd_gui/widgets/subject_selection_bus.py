"""Lightweight subject-selection observer bus for cross-plot subject highlighting."""

from __future__ import annotations

from collections.abc import Callable


class SubjectSelectionBus:
    """Broadcast selected subject ID to all registered listeners.

    Usage::

        bus = SubjectSelectionBus()
        bus.connect(lambda sid: print(f"selected: {sid}"))
        bus.notify("001")
    """

    def __init__(self) -> None:
        self._callbacks: list[Callable[[str], None]] = []
        self._current: str = ""

    @property
    def current_subject(self) -> str:
        """Most recently broadcast subject ID (empty string = all subjects)."""
        return self._current

    def connect(self, callback: Callable[[str], None]) -> None:
        """Register *callback* to be called whenever the selection changes."""
        self._callbacks.append(callback)

    def disconnect(self, callback: Callable[[str], None]) -> None:
        """Remove a previously registered *callback* (no-op if not found)."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def notify(self, subject_id: str) -> None:
        """Broadcast *subject_id* to all connected callbacks."""
        self._current = subject_id
        for cb in self._callbacks:
            try:
                cb(subject_id)
            except Exception:
                pass
