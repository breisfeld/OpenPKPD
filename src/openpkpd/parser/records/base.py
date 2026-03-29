"""Abstract base class for all NONMEM record types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRecord(ABC):
    """
    Abstract base for all $-record parsers.

    Each subclass handles one NONMEM record type (e.g., $THETA, $OMEGA, $DATA).
    """

    record_name: str = ""  # Override in subclasses

    def __init__(self, raw_text: str, header_line: int = 0) -> None:
        self.raw_text = raw_text
        self.header_line = header_line
        self._parse(raw_text)

    @abstractmethod
    def _parse(self, text: str) -> None:
        """Parse the record body text and populate instance attributes."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to a plain dict (for debugging / round-trip)."""
        return {"record": self.record_name, "raw_text": self.raw_text}

    def to_string(self) -> str:
        """Serialize this record to NONMEM control stream text.

        The default implementation emits::

            $RECORDNAME
            <raw_text>

        Subclasses may override to reconstruct from their parsed attributes so
        that programmatic modifications are reflected in the output.
        """
        return f"${self.record_name}\n{self.raw_text.rstrip()}\n"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(header_line={self.header_line})"
