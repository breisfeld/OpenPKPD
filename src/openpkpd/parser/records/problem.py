"""$PROBLEM record parser."""

from __future__ import annotations

from .base import BaseRecord


class ProblemRecord(BaseRecord):
    """
    $PROBLEM <title text>

    The entire body (excluding the $PROBLEM header) is the problem title.
    """

    record_name = "PROBLEM"

    def _parse(self, text: str) -> None:
        self.title: str = text.strip()

    def to_string(self) -> str:
        return f"$PROBLEM {self.title}\n"

    def __repr__(self) -> str:
        return f"ProblemRecord(title={self.title!r})"
