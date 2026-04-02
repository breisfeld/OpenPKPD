"""Tests for CM4: Inline-IF indentation tracking in _translate_block.

Verifies that nested inline-IFs produce correctly indented Python code
at all nesting levels.
"""

from __future__ import annotations

import ast

import pytest

from openpkpd.parser.code_compiler import NMTRANCompiler, _translate_block, _INTRINSICS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def translate(nmtran_code: str) -> str:
    """Translate a NM-TRAN snippet to Python using the module-level function."""
    return _translate_block(nmtran_code, _INTRINSICS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInlineIfIndentation:
    """Indentation correctness for inline-IF translation."""

    def test_single_level_if_body_has_4_space_indent(self):
        """Single-level inline-IF body is prefixed with exactly 4 spaces."""
        code = "IF(X>1) Z=1"
        result = translate(code)
        lines = result.splitlines()
        # Expect: "if (X>1):" and "    Z=1" (or "    z=1" after name translation)
        assert len(lines) == 2, f"Expected 2 lines, got: {lines!r}"
        assert lines[0] == "if (X>1):", f"Header wrong: {lines[0]!r}"
        assert lines[1].startswith("    "), f"Body lacks 4-space indent: {lines[1]!r}"
        assert lines[1] == "    Z=1", f"Body wrong: {lines[1]!r}"

    def test_nested_inline_if_body_has_8_space_indent(self):
        """Nested inline-IF (IF inside IF body) produces 8-space inner body."""
        # Note: 'Y' and 'F' are NONMEM reserved names that get translated to
        # lowercase 'y' and 'f'. Use 'YY' to avoid the reserved-name translation.
        code = "IF(X>1) IF(YY>2) Z=1"
        result = translate(code)
        lines = result.splitlines()
        # Expected:
        #   if (X>1):
        #       if (YY>2):
        #           Z=1
        assert len(lines) == 3, f"Expected 3 lines, got {lines!r}"
        assert lines[0] == "if (X>1):", f"Outer header wrong: {lines[0]!r}"
        assert lines[1] == "    if (YY>2):", f"Inner header wrong: {lines[1]!r}"
        assert lines[2] == "        Z=1", f"Inner body wrong: {lines[2]!r}"

    def test_generated_python_is_syntactically_valid(self):
        """All translated NM-TRAN snippets parse without SyntaxError."""
        snippets = [
            "IF(X>1) Z=1",
            "IF(X>1) IF(YY>2) Z=1",
            "IF(X.GT.0) XX=X+1",
            "IF(A>0) THEN\n  B=A*2\nENDIF",
        ]
        for snippet in snippets:
            result = translate(snippet)
            try:
                ast.parse(result)
            except SyntaxError as e:
                pytest.fail(f"SyntaxError for snippet {snippet!r}: {e}\nTranslated:\n{result}")

    def test_nested_if_correct_execution(self):
        """Nested inline-IF: IF(X>1) IF(Y>2) Z=1 executes correctly."""
        compiler = NMTRANCompiler()
        # Use $PK compilation so we can call the result
        pk_code = "IF(THETA(1).GT.1) IF(THETA(2).GT.2) Z=1"
        pk_fn = compiler.compile_pk(pk_code)

        # X=2, Y=3 → both conditions true → Z=1
        result = pk_fn(theta=[2.0, 3.0], eta=[])
        assert result.get("Z", 0.0) == pytest.approx(1.0), (
            f"Expected Z=1 for (X=2,Y=3), got {result}"
        )

        # X=2, Y=1 → outer true, inner false → Z unchanged (absent from result)
        result2 = pk_fn(theta=[2.0, 1.0], eta=[])
        assert result2.get("Z", float("nan")) != pytest.approx(1.0), (
            f"Expected Z unchanged for (X=2,Y=1), got {result2}"
        )

        # X=0, Y=3 → outer false → Z unchanged
        result3 = pk_fn(theta=[0.0, 3.0], eta=[])
        assert result3.get("Z", float("nan")) != pytest.approx(1.0), (
            f"Expected Z unchanged for (X=0,Y=3), got {result3}"
        )

    def test_inline_if_inside_block_if_is_doubly_indented(self):
        """Inline-IF inside an IF-THEN block uses compound indentation."""
        # Use YY to avoid the NONMEM reserved name 'Y' (translated to 'y')
        code = "IF(X>0) THEN\n  IF(YY>0) Z=1\nENDIF"
        result = translate(code)
        lines = result.splitlines()
        # Expected:
        #   if (X>0):
        #       if (YY>0):
        #           Z=1
        assert len(lines) == 3, f"Expected 3 lines, got {lines!r}"
        assert lines[0] == "if (X>0):", f"Outer block header wrong: {lines[0]!r}"
        assert lines[1] == "    if (YY>0):", f"Inner if wrong: {lines[1]!r}"
        assert lines[2] == "        Z=1", f"Inner body wrong: {lines[2]!r}"
