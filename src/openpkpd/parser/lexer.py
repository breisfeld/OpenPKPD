"""
Tokenizer for NM-TRAN (NONMEM control stream) syntax.

NM-TRAN grammar overview:
  - Lines starting with '$' introduce records: $PROBLEM, $DATA, $THETA, etc.
  - Lines starting with ';' are comments (ignored)
  - Everything after ';' on a line is a comment
  - Record headers may have abbreviated names: $EST → $ESTIMATION
  - Records continue until the next '$' record header
  - Inline FORTRAN-style code blocks ($PK, $DES, $ERROR, $PRED) are preserved verbatim
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    RECORD_HEADER = auto()  # $PROBLEM, $DATA, $THETA, etc.
    TEXT = auto()  # raw text within a record
    NUMBER = auto()  # numeric literal
    WORD = auto()  # identifier / keyword
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    COMMA = auto()  # ,
    EQUALS = auto()  # =
    SLASH = auto()  # / (separates init values in some records)
    SEMICOLON = auto()  # ; (comment start, usually stripped)
    NEWLINE = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


# Map of abbreviated → canonical record names (NM-TRAN abbreviation rules)
_RECORD_ABBREVS: dict[str, str] = {
    "PROB": "PROBLEM",
    "SUBROUT": "SUBROUTINES",
    "SUBR": "SUBROUTINES",
    "EST": "ESTIMATION",
    "ESTIM": "ESTIMATION",
    "COV": "COVARIANCE",
    "COVAR": "COVARIANCE",
    "SIM": "SIMULATION",
    "SIMUL": "SIMULATION",
    "THETA": "THETA",
    "OMEGA": "OMEGA",
    "SIGMA": "SIGMA",
    "DATA": "DATA",
    "INPUT": "INPUT",
    "PK": "PK",
    "DES": "DES",
    "ERROR": "ERROR",
    "PRED": "PRED",
    "TABLE": "TABLE",
    "PRIOR": "PRIOR",
    "THETAP": "THETAP",
    "THETAPV": "THETAPV",
    "OMEGAP": "OMEGAP",
    "OMEGAPD": "OMEGAPD",
    "SIGMAP": "SIGMAP",
    "SIGMAPD": "SIGMAPD",
    "MIX": "MIXTURE",
    "MIXTURE": "MIXTURE",
    "ABBR": "ABBREVIATED",
    "ABBREVIATED": "ABBREVIATED",
    "NONPAR": "NONPARAMETRIC",
    "NONPARAMETRIC": "NONPARAMETRIC",
    "SIZES": "SIZES",
    "DESIGN": "DESIGN",
    "CONTR": "CONTR",
    "PROBLEM": "PROBLEM",
    "SUBROUTINES": "SUBROUTINES",
    "ESTIMATION": "ESTIMATION",
    "COVARIANCE": "COVARIANCE",
    "SIMULATION": "SIMULATION",
}

# Code block record types: content is preserved verbatim as FORTRAN/NMTRAN
CODE_BLOCK_RECORDS: frozenset[str] = frozenset({"PK", "DES", "ERROR", "PRED"})

# Token pattern for number (scientific notation, negatives, fractions)
_NUMBER_PAT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?")
_WORD_PAT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class RawRecord:
    """A raw parsed record before record-type-specific parsing."""

    name: str  # Canonical record name, e.g. "THETA"
    header_line: int  # Source line of the $RECORD header
    raw_text: str  # Full text of the record body (stripped of comments)
    is_code_block: bool = False  # True for $PK/$DES/$ERROR/$PRED


def tokenize_record_body(raw_text: str, start_line: int = 1) -> list[Token]:
    """
    Tokenize the body of a non-code record into a flat token stream.

    Numbers, words, parens, commas, equals signs, and slashes are returned.
    Comments (;...) are stripped. Newlines are emitted.
    """
    tokens: list[Token] = []
    lines = raw_text.split("\n")
    for line_offset, line in enumerate(lines):
        line_no = start_line + line_offset
        # Strip comments
        comment_pos = line.find(";")
        if comment_pos >= 0:
            line = line[:comment_pos]
        line = line.rstrip()
        col = 0
        while col < len(line):
            ch = line[col]
            if ch in " \t":
                col += 1
                continue
            if ch == "(":
                tokens.append(Token(TokenType.LPAREN, "(", line_no, col))
                col += 1
            elif ch == ")":
                tokens.append(Token(TokenType.RPAREN, ")", line_no, col))
                col += 1
            elif ch == ",":
                tokens.append(Token(TokenType.COMMA, ",", line_no, col))
                col += 1
            elif ch == "=":
                tokens.append(Token(TokenType.EQUALS, "=", line_no, col))
                col += 1
            elif ch == "/":
                tokens.append(Token(TokenType.SLASH, "/", line_no, col))
                col += 1
            else:
                # Try number first (handles negatives starting with - when not ambiguous)
                m = _NUMBER_PAT.match(line, col)
                if m and (ch.isdigit() or ch == "." or (ch in "+-" and col == 0)):
                    tokens.append(Token(TokenType.NUMBER, m.group(), line_no, col))
                    col = m.end()
                elif _WORD_PAT.match(line, col):
                    m2 = _WORD_PAT.match(line, col)
                    assert m2 is not None
                    tokens.append(Token(TokenType.WORD, m2.group(), line_no, col))
                    col = m2.end()
                else:
                    # Standalone sign or unknown character — emit as text
                    tokens.append(Token(TokenType.TEXT, ch, line_no, col))
                    col += 1
        tokens.append(Token(TokenType.NEWLINE, "\n", line_no, len(line)))
    return tokens


def split_into_raw_records(text: str) -> list[RawRecord]:
    """
    Split a full NONMEM control stream into RawRecord objects.

    Each RawRecord contains the canonical record name and the verbatim body text.
    """
    records: list[RawRecord] = []
    lines = text.splitlines()

    # Pattern: optional whitespace, $, record name, optional rest
    header_re = re.compile(r"^\s*\$([A-Za-z][A-Za-z0-9_]*)(.*)", re.DOTALL)

    current_name: str | None = None
    current_lines: list[str] = []
    current_header_line: int = 0

    def flush() -> None:
        if current_name is not None:
            body = "\n".join(current_lines)
            canonical = _canonicalize_record_name(current_name)
            records.append(
                RawRecord(
                    name=canonical,
                    header_line=current_header_line,
                    raw_text=body,
                    is_code_block=canonical in CODE_BLOCK_RECORDS,
                )
            )

    for line_no, line in enumerate(lines, start=1):
        # Skip blank lines and comment-only lines
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            if current_name is not None:
                current_lines.append(line)
            continue

        m = header_re.match(line)
        if m:
            flush()
            current_name = m.group(1).upper()
            rest = m.group(2).strip()
            current_lines = [rest] if rest else []
            current_header_line = line_no
        else:
            if current_name is not None:
                current_lines.append(line)

    flush()
    return records


def _canonicalize_record_name(name: str) -> str:
    """
    Resolve NM-TRAN abbreviated record name to canonical form.

    Matching is done by trying the full name, then checking if it is a
    unique prefix of any known canonical name.
    """
    upper = name.upper()
    # Exact match
    if upper in _RECORD_ABBREVS:
        return _RECORD_ABBREVS[upper]
    # Prefix match
    matches = [canon for abbr, canon in _RECORD_ABBREVS.items() if canon.startswith(upper)]
    unique = list(set(matches))
    if len(unique) == 1:
        return unique[0]
    # Return as-is (unknown records are kept for forward compatibility)
    return upper
