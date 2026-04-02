"""Tests for $OMEGA BLOCK(SAME) source validation — PR1."""

from __future__ import annotations

import pytest

from openpkpd.parser.records.omega import OmegaRecord
from openpkpd.utils.errors import ParseError


def _make_omega(body: str) -> OmegaRecord:
    """Create an OmegaRecord from body text (the part after $OMEGA)."""
    return OmegaRecord(raw_text=body, header_line=1)


class TestOmegaBlockSameValidation:
    def test_valid_block_same_parses(self):
        """Valid BLOCK(2) SAME after a matching BLOCK(2) parses successfully."""
        rec = _make_omega("BLOCK(2) 0.1 0.05 0.1 BLOCK(2) SAME")
        assert len(rec.specs) == 2
        assert not rec.specs[0].same
        assert rec.specs[1].same
        assert rec.specs[1].block_size == 2

    def test_same_standalone_record_no_raise(self):
        """Bare SAME as the only item in a record (cross-record reference) does not raise.

        In NONMEM, $OMEGA BLOCK(1) SAME as its own record is the standard IOV pattern
        referencing the preceding $OMEGA record.  We cannot validate this cross-record
        at parse time, so we accept it.
        """
        # Should not raise — standalone-record SAME is a valid IOV pattern
        rec = _make_omega("SAME")
        assert len(rec.specs) == 1
        assert rec.specs[0].same

    def test_same_with_size_mismatch_in_same_record_raises(self):
        """BLOCK(3) SAME after a BLOCK(2) in the same record raises ParseError."""
        with pytest.raises(ParseError, match="size mismatch"):
            _make_omega("BLOCK(2) 0.1 0.05 0.1 BLOCK(3) SAME")

    def test_block_same_values_match_prior(self):
        """$OMEGA BLOCK(2) ... BLOCK(2) SAME produces two specs with matching block_size."""
        body = "BLOCK(2)\n  0.1\n  0.05 0.1\nBLOCK(2) SAME"
        rec = _make_omega(body)
        assert len(rec.specs) == 2

        prior_spec = rec.specs[0]
        same_spec = rec.specs[1]

        assert prior_spec.block_size == 2
        assert prior_spec.values == [0.1, 0.05, 0.1]

        assert same_spec.same is True
        assert same_spec.block_size == 2

        # Verify the prior block expands to the expected matrix
        mat = prior_spec.to_matrix()
        assert mat[0, 0] == pytest.approx(0.1)
        assert mat[1, 0] == pytest.approx(0.05)
        assert mat[1, 1] == pytest.approx(0.1)

    def test_same_without_block_n_inherits_prior_size(self):
        """Bare SAME after BLOCK(2) produces a SAME spec with block_size == 2."""
        rec = _make_omega("BLOCK(2) 0.1 0.05 0.1 SAME")
        assert len(rec.specs) == 2
        assert rec.specs[1].same
        assert rec.specs[1].block_size == 2

    def test_diagonal_then_block_same_raises_mismatch(self):
        """BLOCK(2) SAME after a scalar diagonal (block_size=1) in the same record raises ParseError."""
        with pytest.raises(ParseError, match="size mismatch"):
            _make_omega("0.1 BLOCK(2) SAME")

    def test_standalone_block_n_same_record_does_not_raise(self):
        """$OMEGA BLOCK(1) SAME as its own record (cross-record IOV pattern) does not raise."""
        rec = _make_omega("BLOCK(1) SAME")
        assert len(rec.specs) == 1
        assert rec.specs[0].same
        assert rec.specs[0].block_size == 1

    def test_multiple_valid_same_blocks(self):
        """Multiple BLOCK(2) SAME blocks after one BLOCK(2) all parse successfully."""
        rec = _make_omega("BLOCK(2) 0.1 0.05 0.1 BLOCK(2) SAME BLOCK(2) SAME")
        # Two SAME specs
        assert len(rec.specs) == 3
        assert all(s.same for s in rec.specs[1:])
