"""Tests for NmtranHighlighter — NMTRAN syntax highlighter widget.

All tests require a QApplication context.  PySide6 is only available when the
``gui`` extra is installed; the package-level conftest skips this entire module
when PySide6 is absent.
"""

from __future__ import annotations

import pytest

from openpkpd_gui.app.runtime import load_qt_modules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_app():
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    return qt_widgets.QApplication.instance() or qt_widgets.QApplication([])


def _make_editor(text: str = ""):
    """Return a QPlainTextEdit with *text* pre-loaded."""
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    _get_app()
    editor = qt_widgets.QPlainTextEdit(text)
    return editor


def _pump_events():
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    qt_widgets.QApplication.processEvents()


def _formats_in_line(editor, line_index: int) -> list[tuple[int, int, str]]:
    """Return [(start, length, foreground_hex)] for every coloured region on
    *line_index* (0-based) after pumping the event loop so highlighting runs.

    Reads from ``QTextLayout.formats()`` which is where ``QSyntaxHighlighter``
    stores its overlay formats (not in the QTextFragment character data).
    Only regions with a non-default foreground colour are returned.
    """
    _pump_events()
    doc = editor.document()
    block = doc.findBlockByLineNumber(line_index)
    results = []
    for fmt_range in block.layout().formats():
        color = fmt_range.format.foreground().color()
        if color.isValid() and color.alpha() > 0 and color.name() not in ("#000000", "#ffffff"):
            results.append((fmt_range.start, fmt_range.length, color.name()))
    return results


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def editor_with_highlighter():
    """Return a fresh QPlainTextEdit with the NmtranHighlighter attached."""
    from openpkpd_gui.widgets.nmtran_highlighter import NmtranHighlighter

    editor = _make_editor()
    NmtranHighlighter.attach(editor)
    return editor


# ---------------------------------------------------------------------------
# Attachment tests
# ---------------------------------------------------------------------------


class TestAttach:
    def test_attach_stores_reference_on_editor(self, editor_with_highlighter):
        """The highlighter instance must be reachable via the editor attribute."""
        from openpkpd_gui.widgets.nmtran_highlighter import NmtranHighlighter

        assert hasattr(editor_with_highlighter, NmtranHighlighter._ATTR)

    def test_attach_twice_replaces_reference(self):
        """Calling attach twice on the same editor replaces the stored reference."""
        from openpkpd_gui.widgets.nmtran_highlighter import NmtranHighlighter

        editor = _make_editor()
        h1 = NmtranHighlighter.attach(editor)
        h2 = NmtranHighlighter.attach(editor)
        assert h1 is not h2
        assert getattr(editor, NmtranHighlighter._ATTR) is h2

    def test_attach_returns_highlighter_instance(self):
        from openpkpd_gui.widgets.nmtran_highlighter import NmtranHighlighter

        editor = _make_editor()
        result = NmtranHighlighter.attach(editor)
        assert isinstance(result, NmtranHighlighter)


# ---------------------------------------------------------------------------
# Colour tests — verify the highlighter colours specific token types
# ---------------------------------------------------------------------------


def _colours_on_line(text: str, line: int = 0) -> set[str]:
    """Attach a highlighter to an editor with *text*, pump events, return colour set."""
    from openpkpd_gui.widgets.nmtran_highlighter import NmtranHighlighter

    editor = _make_editor(text)
    NmtranHighlighter.attach(editor)
    _pump_events()
    return {c for _, _, c in _formats_in_line(editor, line)}


class TestBlockKeywords:
    """$RECORD keywords should be coloured blue (#1a56db)."""

    BLUE = "#1a56db"

    @pytest.mark.parametrize(
        "record",
        ["$PK", "$ERROR", "$THETA", "$OMEGA", "$SIGMA", "$ESTIMATION", "$PROBLEM", "$DATA"],
    )
    def test_block_keyword_coloured(self, record):
        colours = _colours_on_line(record)
        assert self.BLUE in colours, f"Expected blue for {record!r}, got {colours}"

    def test_block_keyword_case_insensitive(self):
        colours = _colours_on_line("$pk")
        assert self.BLUE in colours


class TestComments:
    """Semicolon comments should be coloured green (#16a34a)."""

    GREEN = "#16a34a"

    def test_full_comment_line(self):
        colours = _colours_on_line("; This is a comment")
        assert self.GREEN in colours

    def test_inline_comment(self):
        colours = _colours_on_line("CL = THETA(1) ; clearance")
        assert self.GREEN in colours


class TestNumericLiterals:
    """Numeric literals should be coloured amber (#b45309)."""

    AMBER = "#b45309"

    @pytest.mark.parametrize("text", ["42", "3.14", "1.5E-3"])
    def test_numeric_coloured(self, text):
        colours = _colours_on_line(text)
        assert self.AMBER in colours, f"Expected amber for {text!r}, got {colours}"


class TestBuiltinFunctions:
    """NMTRAN built-in functions should be coloured purple (#7c3aed)."""

    PURPLE = "#7c3aed"

    @pytest.mark.parametrize("func", ["EXP", "LOG", "SQRT", "PHI", "ABS"])
    def test_builtin_coloured(self, func):
        colours = _colours_on_line(f"Y = {func}(F)")
        assert self.PURPLE in colours, f"Expected purple for {func!r}, got {colours}"

    def test_builtin_case_insensitive(self):
        colours = _colours_on_line("exp(CL)")
        assert self.PURPLE in colours


class TestSpecialVariables:
    """NMTRAN special variables should be coloured teal (#0891b2)."""

    TEAL = "#0891b2"

    @pytest.mark.parametrize(
        "text",
        [
            "CL = THETA(1) * EXP(ETA(1))",
            "Y = F + EPS(1)",
            "ERR(1)",
            "A(1)",
        ],
    )
    def test_special_var_coloured(self, text):
        colours = _colours_on_line(text)
        assert self.TEAL in colours, f"Expected teal for {text!r}, got {colours}"


# ---------------------------------------------------------------------------
# Multi-line test
# ---------------------------------------------------------------------------


class TestMultiLine:
    def test_each_line_highlighted_independently(self):
        """A multi-line document highlights each line according to its content."""
        from openpkpd_gui.widgets.nmtran_highlighter import NmtranHighlighter

        lines = [
            "$PK",
            "CL = THETA(1) * EXP(ETA(1))",
            "; comment line",
            "Y = F + EPS(1) * W",
        ]
        editor = _make_editor("\n".join(lines))
        NmtranHighlighter.attach(editor)
        _pump_events()

        # Line 0: $PK → blue
        assert "#1a56db" in {c for _, _, c in _formats_in_line(editor, 0)}
        # Line 2: comment → green
        assert "#16a34a" in {c for _, _, c in _formats_in_line(editor, 2)}
        # Line 1: THETA/ETA → teal
        line1_colours = {c for _, _, c in _formats_in_line(editor, 1)}
        assert "#0891b2" in line1_colours


# ---------------------------------------------------------------------------
# Integration with model_workflow: verify highlighters are attached
# ---------------------------------------------------------------------------


class TestModelWorkflowIntegration:
    """Smoke tests: build_model_workflow attaches NmtranHighlighter to editors."""

    def test_pk_editor_has_highlighter(self):
        from openpkpd_gui.widgets.nmtran_highlighter import NmtranHighlighter
        from openpkpd_gui.workflows.model_workflow import build_model_workflow
        from openpkpd_gui.domain.workspace import Workspace

        qt_core, qt_gui, qt_widgets = load_qt_modules()
        _get_app()

        ws = Workspace()
        widget = build_model_workflow(ws)

        pk_edit = widget.findChild(qt_widgets.QPlainTextEdit, "model-pk-code")
        assert pk_edit is not None, "model-pk-code editor not found"
        assert hasattr(pk_edit, NmtranHighlighter._ATTR), (
            "NmtranHighlighter not attached to model-pk-code editor"
        )

    def test_control_stream_editor_has_highlighter(self):
        from openpkpd_gui.widgets.nmtran_highlighter import NmtranHighlighter
        from openpkpd_gui.workflows.model_workflow import build_model_workflow
        from openpkpd_gui.domain.workspace import Workspace

        qt_core, qt_gui, qt_widgets = load_qt_modules()
        _get_app()

        ws = Workspace()
        widget = build_model_workflow(ws)

        ctl_edit = widget.findChild(qt_widgets.QPlainTextEdit, "model-control-stream-text")
        assert ctl_edit is not None, "model-control-stream-text editor not found"
        assert hasattr(ctl_edit, NmtranHighlighter._ATTR), (
            "NmtranHighlighter not attached to control stream editor"
        )

    def test_error_editor_has_highlighter(self):
        from openpkpd_gui.widgets.nmtran_highlighter import NmtranHighlighter
        from openpkpd_gui.workflows.model_workflow import build_model_workflow
        from openpkpd_gui.domain.workspace import Workspace

        qt_core, qt_gui, qt_widgets = load_qt_modules()
        _get_app()

        ws = Workspace()
        widget = build_model_workflow(ws)

        error_edit = widget.findChild(qt_widgets.QPlainTextEdit, "model-error-code")
        assert error_edit is not None, "model-error-code editor not found"
        assert hasattr(error_edit, NmtranHighlighter._ATTR), (
            "NmtranHighlighter not attached to model-error-code editor"
        )
