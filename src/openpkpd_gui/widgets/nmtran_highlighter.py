"""NMTRAN syntax highlighter for QPlainTextEdit / QTextEdit widgets.

Provides ``NmtranHighlighter``, a ``QSyntaxHighlighter`` subclass that colours
NONMEM control-stream syntax in the model workflow's code editors.

Colour scheme (system-palette-agnostic — all colours are opaque foreground
values that work on both light and dark backgrounds):

    Block records  ($PK, $THETA …)  — bold blue  (#1a56db)
    Comments       (; …)            — italic green (#16a34a)
    Numeric literals                — amber (#b45309)
    Built-in functions (EXP, LOG …) — purple (#7c3aed)
    Special variables (ETA, EPS …)  — teal (#0891b2)
"""

from __future__ import annotations

import re

from openpkpd_gui.app.runtime import load_qt_modules

# ---------------------------------------------------------------------------
# Token patterns
# ---------------------------------------------------------------------------

# Complete $RECORD names (must appear at the start of a record line or after
# whitespace in multi-record files).  We match the dollar sign as part of the
# token so the entire "$PK" sequence gets coloured.
_BLOCK_KEYWORDS: list[str] = [
    "ABBREVIATED",
    "BIND",
    "CONTR",
    "COVARIANCE",
    "DATA",
    "DES",
    "ERROR",
    "ESTIMATION",
    "INFN",
    "INPUT",
    "LEVEL",
    "MIX",
    "MODEL",
    "MSFI",
    "NONPARAMETRIC",
    "OMEGAP",
    "OMEGAPD",
    "OMIT",
    "OMEGA",
    "PK",
    "PRED",
    "PRIOR",
    "PROBLEM",
    "SCATTER",
    "SIGMA",
    "SIGMAP",
    "SIGMAPD",
    "SIMULATION",
    "SIZES",
    "SUPER",
    "SUBROUTINE",
    "TABLE",
    "THETA",
    "THETAP",
    "THETAPV",
    "TOL",
]

# Built-in NMTRAN functions
_BUILTIN_FUNCS: list[str] = [
    "ABS",
    "ATAN",
    "COS",
    "EXP",
    "GAMMA",
    "INT",
    "LOG10",
    "LOG",
    "MOD",
    "NORDF",
    "PHI",
    "SIN",
    "SQRT",
    "TAN",
    "ERF",
    "ERFC",
    "GAMLN",
    "DPHI",
]

# Special NMTRAN variables / reserved names that appear in $PK / $ERROR / $DES
_SPECIAL_VARS: list[str] = [
    r"THETA\s*\(\s*\d+\s*\)",  # THETA(n)
    r"ETA\s*\(\s*\d+\s*\)",    # ETA(n)
    r"EPS\s*\(\s*\d+\s*\)",    # EPS(n)
    r"ERR\s*\(\s*\d+\s*\)",    # ERR(n)
    r"\bA\s*\(\s*\d+\s*\)",    # A(n)  — compartment amounts
    r"\bF\b",
    r"\bY\b",
    r"\bW\b",
    r"\bIPRED\b",
    r"\bIWRES\b",
    r"\bCWRES\b",
    r"\bPRED\b",
    r"\bALAG\d*\b",
    r"\bS\d+\b",
    r"\bR\d+\b",
    r"\bD\d+\b",
    r"\bT\b",
    r"\bTIME\b",
    r"\bDV\b",
    r"\bID\b",
    r"\bAMT\b",
    r"\bEVID\b",
    r"\bMDV\b",
    r"\bCMT\b",
]


def _compile_rules(
    qt_gui,  # PySide6.QtGui module
) -> list[tuple[re.Pattern[str], object]]:
    """Build (compiled-pattern, QTextCharFormat) pairs for each token type."""
    QColor = qt_gui.QColor
    QFont = qt_gui.QFont
    QTextCharFormat = qt_gui.QTextCharFormat

    def _fmt(
        color: str,
        *,
        bold: bool = False,
        italic: bool = False,
    ) -> object:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(QFont.Weight.Bold)
        if italic:
            fmt.setFontItalic(True)
        return fmt

    block_fmt = _fmt("#1a56db", bold=True)
    comment_fmt = _fmt("#16a34a", italic=True)
    number_fmt = _fmt("#b45309")
    func_fmt = _fmt("#7c3aed")
    var_fmt = _fmt("#0891b2")

    rules: list[tuple[re.Pattern[str], object]] = []

    # Block record keywords — match $KEYWORD (case-insensitive)
    block_pat = r"\$(?:" + "|".join(_BLOCK_KEYWORDS) + r")\b"
    rules.append((re.compile(block_pat, re.IGNORECASE), block_fmt))

    # Comments — from ; to end of line (highest priority — override others)
    rules.append((re.compile(r";[^\n]*"), comment_fmt))

    # Numeric literals (integers and floats, including scientific notation)
    rules.append(
        (re.compile(r"\b\d+(?:\.\d*)?(?:[eEdD][+\-]?\d+)?\b"), number_fmt)
    )

    # Built-in functions — match word boundary to avoid partial matches
    func_pat = r"\b(?:" + "|".join(_BUILTIN_FUNCS) + r")\b"
    rules.append((re.compile(func_pat, re.IGNORECASE), func_fmt))

    # Special NMTRAN variables
    for pat in _SPECIAL_VARS:
        rules.append((re.compile(pat, re.IGNORECASE), var_fmt))

    return rules


class NmtranHighlighter:
    """Attach NMTRAN syntax highlighting to a ``QPlainTextEdit`` or ``QTextEdit``.

    Usage::

        editor = QPlainTextEdit()
        NmtranHighlighter.attach(editor)

    The highlighter object is stored as an attribute on the editor widget so it
    is not garbage-collected while the editor is alive.
    """

    _ATTR = "_nmtran_highlighter"

    def __init__(self, document, qt_gui) -> None:
        _, QtGui, _ = load_qt_modules()

        class _Impl(QtGui.QSyntaxHighlighter):
            def __init__(self_inner, parent) -> None:  # noqa: N805
                super().__init__(parent)
                self_inner._rules = _compile_rules(qt_gui)

            def highlightBlock(self_inner, text: str) -> None:  # noqa: N802, N805
                for pattern, fmt in self_inner._rules:
                    for m in pattern.finditer(text):
                        self_inner.setFormat(m.start(), m.end() - m.start(), fmt)

        self._impl = _Impl(document)

    @classmethod
    def attach(cls, editor) -> "NmtranHighlighter":
        """Attach a highlighter to *editor* and return it.

        The instance is stored on the editor as ``editor._nmtran_highlighter``
        to keep it alive for the lifetime of the editor widget.
        """
        _, QtGui, _ = load_qt_modules()
        hl = cls(editor.document(), QtGui)
        setattr(editor, cls._ATTR, hl)
        return hl
