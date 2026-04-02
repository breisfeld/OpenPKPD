"""Tests for unknown $RECORD warning — PR5."""

from __future__ import annotations

import warnings

import pytest

from openpkpd.parser.control_stream import ControlStream
from openpkpd.utils.errors import ParseWarning

_BASE_CTL = """\
$PROBLEM Test problem
$DATA ./data.csv
$INPUT ID TIME DV AMT
$THETA (0, 1.5, 10)
$OMEGA 0.1
$SIGMA 0.05
$ESTIMATION METHOD=1 MAXEVAL=9999
"""


class TestUnknownRecordWarning:
    def test_known_records_no_warning(self):
        """Parsing a control stream with all known records emits no ParseWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", ParseWarning)
            # $DATA warning for missing file is not ParseWarning from unknown record
            # so we selectively ignore it by checking only unknown-record-related messages
            warnings.filterwarnings("ignore", message=".*does not exist.*", category=ParseWarning)
            cs = ControlStream.from_string(_BASE_CTL)

        # Should have parsed all known records
        assert cs.get("PROBLEM") is not None

    def test_unknown_record_emits_warning(self):
        """Control stream with an unknown record (e.g. $SCATTER) emits ParseWarning."""
        ctl = _BASE_CTL + "$SCATTER MATRIX=S\n"
        with pytest.warns(ParseWarning, match="SCATTER"):
            cs = ControlStream.from_string(ctl)
        # Parsing still succeeds
        assert cs.get("PROBLEM") is not None

    def test_unknown_record_warning_message_contains_name(self):
        """The warning message mentions the record name."""
        ctl = _BASE_CTL + "$FOOBAR whatever\n"
        with pytest.warns(ParseWarning) as record:
            ControlStream.from_string(ctl)

        # Filter to only unknown-record warnings
        unknown_warnings = [
            w for w in record.list if "FOOBAR" in str(w.message)
        ]
        assert unknown_warnings, "Expected a warning mentioning 'FOOBAR'"

    def test_multiple_unknown_records_each_warn(self):
        """Multiple unknown records each produce at least one warning."""
        ctl = _BASE_CTL + "$FOOBAR x\n$BAZZQUUX y\n"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ControlStream.from_string(ctl)

        unknown_names = {"FOOBAR", "BAZZQUUX"}
        warned_names = set()
        for warning in w:
            if issubclass(warning.category, ParseWarning):
                for name in unknown_names:
                    if name in str(warning.message):
                        warned_names.add(name)

        assert unknown_names <= warned_names, (
            f"Expected warnings for {unknown_names - warned_names}"
        )

    def test_parsing_succeeds_after_unknown_records(self):
        """Parsing succeeds and returns a valid ControlStream after unknown records."""
        ctl = _BASE_CTL + "$COVARIANCE MATRIX=R\n"
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            cs = ControlStream.from_string(ctl)

        assert isinstance(cs, ControlStream)
        assert cs.get("THETA") is not None
        assert cs.get("OMEGA") is not None
