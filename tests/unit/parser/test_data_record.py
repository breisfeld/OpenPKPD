"""Tests for $DATA record parser — PR2: file-existence warning."""

from __future__ import annotations

import os
import warnings

import pytest

from openpkpd.parser.records.data import DataRecord
from openpkpd.utils.errors import ParseWarning


def _make_record(body: str) -> DataRecord:
    """Create a DataRecord from a body string (the part after $DATA).

    The raw_text passed to the record is the body only (without '$DATA'),
    matching how the lexer provides it from a full control stream.
    """
    return DataRecord(raw_text=body, header_line=1)


class TestDataRecordFileExistenceWarning:
    def test_existing_file_no_warning(self, tmp_path):
        """Parsing $DATA with an existing file emits no warning."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("ID,TIME,DV\n1,0,1.0\n")

        with warnings.catch_warnings():
            warnings.simplefilter("error", ParseWarning)
            rec = _make_record(str(data_file))

        assert rec.filename == str(data_file)

    def test_nonexistent_file_emits_warning(self):
        """Parsing $DATA with a nonexistent absolute path emits ParseWarning."""
        path = "/nonexistent/path/to/data.csv"
        with pytest.warns(ParseWarning, match=r"/nonexistent/path/to/data\.csv"):
            rec = _make_record(path)

        assert rec.filename == path

    def test_relative_nonexistent_path_warning_message(self):
        """Warning for a nonexistent relative path mentions 'relative to a run directory'."""
        path = "./relative/path.csv"
        with pytest.warns(ParseWarning, match="relative to a run directory"):
            rec = _make_record(path)

        assert rec.filename == path

    def test_warning_does_not_raise(self):
        """Warning must not raise; parsing must succeed and filename must be set."""
        path = "/does/not/exist.csv"
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            rec = _make_record(path)

        assert rec.filename == path

    def test_warning_filename_in_message(self):
        """The warning message must contain the filename."""
        path = "/some/missing/file.csv"
        with pytest.warns(ParseWarning) as record:
            _make_record(path)

        assert any(path in str(w.message) for w in record.list)

    def test_options_still_parsed_with_warning(self):
        """Other $DATA options parse correctly even when a warning is emitted."""
        path = "/missing.csv"
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            rec = _make_record(f"{path} IGNORE=@ RECORDS=100")

        assert rec.filename == path
        assert rec.ignore_char == "@"
        assert rec.records == 100
