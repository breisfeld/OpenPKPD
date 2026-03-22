"""Unit tests for the NM-TRAN lexer."""

import pytest

from openpkpd.parser.lexer import _canonicalize_record_name, split_into_raw_records

SIMPLE_CTL = """\
$PROBLEM Theophylline 1-compartment oral
$DATA theo.csv IGNORE=@
$INPUT ID TIME AMT DV EVID MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  KA = THETA(1)*EXP(ETA(1))
  CL = THETA(2)*EXP(ETA(2))
  V  = THETA(3)*EXP(ETA(3))
$ERROR
  Y = F*(1 + EPS(1))
$THETA
  (0, 1.5, 10)
  (0, 0.08, 5)
  (0, 30, 500)
$OMEGA
  0.5
  0.3
  0.3
$SIGMA
  0.1
$ESTIMATION METHOD=COND INTER MAXEVAL=9999
$COVARIANCE
"""


@pytest.mark.unit
def test_split_records_count():
    records = split_into_raw_records(SIMPLE_CTL)
    names = [r.name for r in records]
    assert "PROBLEM" in names
    assert "DATA" in names
    assert "INPUT" in names
    assert "SUBROUTINES" in names
    assert "PK" in names
    assert "ERROR" in names
    assert "THETA" in names
    assert "OMEGA" in names
    assert "SIGMA" in names
    assert "ESTIMATION" in names
    assert "COVARIANCE" in names


@pytest.mark.unit
def test_canonicalize_abbreviations():
    assert _canonicalize_record_name("PROB") == "PROBLEM"
    assert _canonicalize_record_name("EST") == "ESTIMATION"
    assert _canonicalize_record_name("COV") == "COVARIANCE"
    assert _canonicalize_record_name("SUBR") == "SUBROUTINES"
    assert _canonicalize_record_name("SIM") == "SIMULATION"


@pytest.mark.unit
def test_problem_body():
    records = split_into_raw_records(SIMPLE_CTL)
    prob = next(r for r in records if r.name == "PROBLEM")
    assert "Theophylline" in prob.raw_text


@pytest.mark.unit
def test_code_block_preserved():
    records = split_into_raw_records(SIMPLE_CTL)
    pk = next(r for r in records if r.name == "PK")
    assert pk.is_code_block is True
    assert "THETA(1)" in pk.raw_text or "KA" in pk.raw_text


@pytest.mark.unit
def test_comment_handling():
    text = """\
; This is a comment
$PROBLEM Test ; inline comment
; another comment
$THETA
  1.5 ; theta 1
"""
    records = split_into_raw_records(text)
    assert len(records) == 2
