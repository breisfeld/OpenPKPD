"""Tests for CM5: AST-based _detect_uses_amounts in CompiledErrorCallable.

Verifies that the AST-based detector correctly identifies subscript access
on the 'a' compartment array and avoids comment false-positives.
"""

from __future__ import annotations

import pytest

from openpkpd.parser.code_compiler import CompiledErrorCallable, _detect_uses_amounts


class TestDetectUsesAmounts:
    """Tests for the _detect_uses_amounts helper function."""

    def test_direct_subscript_returns_true(self):
        """Code with a[0] → True."""
        assert _detect_uses_amounts("y = a[0] * theta[0]") is True

    def test_comment_only_returns_false(self):
        """Comment containing a[1] does not trigger detection → False."""
        # The raw string passed to _detect_uses_amounts is the *translated* Python
        # code, not NM-TRAN. Python comments start with #; the regex would match
        # 'a[1]' in the comment, but AST ignores comments entirely.
        code = "# a[1] is the central compartment\ny = theta[0] * f"
        assert _detect_uses_amounts(code) is False

    def test_no_a_subscript_returns_false(self):
        """Code with no a[...] references → False."""
        code = "cl = theta[0]\nv = theta[1]\nf = dose * cl / v"
        assert _detect_uses_amounts(code) is False

    def test_alias_access_returns_false_known_limitation(self):
        """Aliased access 'amounts = a; amounts[1]' → False (documented limitation)."""
        code = "amounts = a\ny = amounts[1] * theta[0]"
        # This is a known limitation: aliased subscripts are not detected.
        # The test documents the current (expected) behaviour.
        assert _detect_uses_amounts(code) is False

    def test_unparseable_code_returns_true_conservative(self):
        """SyntaxError in input → True (conservative fallback)."""
        bad_code = "this is not valid python !!!! @@@@"
        assert _detect_uses_amounts(bad_code) is True

    def test_multiple_a_subscripts(self):
        """Multiple a[...] accesses still detected → True."""
        code = "dadt0 = -k * a[0]\ndadt1 = k * a[0] - k2 * a[1]"
        assert _detect_uses_amounts(code) is True


class TestCompiledErrorCallableUsesAmounts:
    """Tests that CompiledErrorCallable uses the AST-based detector."""

    def test_code_with_a_subscript_sets_uses_amounts_true(self):
        """$ERROR code referencing a[0] → _uses_amounts is True."""
        code = "y = a[0] / theta[0] + eps[0]"
        err = CompiledErrorCallable(code)
        assert err._uses_amounts is True

    def test_code_with_comment_only_a_subscript_sets_false(self):
        """$ERROR code with a[1] only in a comment → _uses_amounts is False."""
        code = "# a[1] is central compartment\ny = f + eps[0]"
        err = CompiledErrorCallable(code)
        assert err._uses_amounts is False

    def test_code_without_a_subscript_sets_false(self):
        """$ERROR code without a[...] → _uses_amounts is False."""
        code = "w = theta[0]\niwres = (dv - ipred) / w\ny = ipred + w * eps[0]"
        err = CompiledErrorCallable(code)
        assert err._uses_amounts is False
