"""Tests for SubjectSelectionBus (P2-B: linked graphics)."""

from __future__ import annotations

import pytest

from openpkpd_gui.widgets.subject_selection_bus import SubjectSelectionBus


class TestSubjectSelectionBus:
    def test_initial_subject_is_empty(self) -> None:
        bus = SubjectSelectionBus()
        assert bus.current_subject == ""

    def test_notify_updates_current_subject(self) -> None:
        bus = SubjectSelectionBus()
        bus.notify("001")
        assert bus.current_subject == "001"

    def test_notify_calls_connected_callback(self) -> None:
        bus = SubjectSelectionBus()
        received: list[str] = []
        bus.connect(received.append)
        bus.notify("42")
        assert received == ["42"]

    def test_notify_calls_multiple_callbacks(self) -> None:
        bus = SubjectSelectionBus()
        results: list[str] = []
        bus.connect(lambda sid: results.append(f"a:{sid}"))
        bus.connect(lambda sid: results.append(f"b:{sid}"))
        bus.notify("007")
        assert results == ["a:007", "b:007"]

    def test_notify_empty_subject_broadcasts_empty(self) -> None:
        bus = SubjectSelectionBus()
        received: list[str] = []
        bus.connect(received.append)
        bus.notify("001")
        bus.notify("")
        assert received[-1] == ""
        assert bus.current_subject == ""

    def test_no_callbacks_notify_does_not_raise(self) -> None:
        bus = SubjectSelectionBus()
        bus.notify("999")  # Should not raise

    def test_disconnect_removes_callback(self) -> None:
        bus = SubjectSelectionBus()
        received: list[str] = []
        cb = received.append
        bus.connect(cb)
        bus.disconnect(cb)
        bus.notify("abc")
        assert received == []

    def test_disconnect_nonexistent_is_noop(self) -> None:
        bus = SubjectSelectionBus()
        bus.disconnect(lambda sid: None)  # Should not raise

    def test_callback_exception_does_not_propagate(self) -> None:
        bus = SubjectSelectionBus()

        def bad_cb(sid: str) -> None:
            raise RuntimeError("boom")

        bus.connect(bad_cb)
        bus.notify("123")  # Should not raise
        assert bus.current_subject == "123"

    def test_multiple_notifies_updates_current(self) -> None:
        bus = SubjectSelectionBus()
        bus.notify("A")
        bus.notify("B")
        bus.notify("C")
        assert bus.current_subject == "C"
