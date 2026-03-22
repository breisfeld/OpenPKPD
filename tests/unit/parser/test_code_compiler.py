"""Unit tests for the NMTRANCompiler."""

import math

import pytest

from openpkpd.parser.code_compiler import _INTRINSICS, NMTRANCompiler, _translate_line


@pytest.mark.unit
class TestTranslateLine:
    def test_theta_substitution(self):
        line = "CL = THETA(1)*EXP(ETA(1))"
        result = _translate_line(line, _INTRINSICS)
        assert "theta[0]" in result
        assert "eta[0]" in result

    def test_theta_indexing(self):
        line = "KA = THETA(1)"
        result = _translate_line(line, _INTRINSICS)
        assert "theta[0]" in result

        line2 = "V = THETA(3)"
        result2 = _translate_line(line2, _INTRINSICS)
        assert "theta[2]" in result2

    def test_exp_mapping(self):
        line = "X = EXP(1.0)"
        result = _translate_line(line, _INTRINSICS)
        assert "math.exp" in result

    def test_fortran_logical_operators(self):
        line = "IF (A .GT. B) X = 1"
        result = _translate_line(line, _INTRINSICS)
        assert ".GT." not in result
        assert ">" in result

    def test_fortran_and(self):
        line = "IF (A .EQ. 1 .AND. B .EQ. 2) X = 3"
        result = _translate_line(line, _INTRINSICS)
        assert ".AND." not in result
        assert "and" in result.lower()

    def test_dadt_substitution(self):
        line = "DADT(1) = -K*A(1)"
        result = _translate_line(line, _INTRINSICS)
        assert "dadt[0]" in result
        assert "a[0]" in result

    def test_err_and_sigma_substitution(self):
        line = "Y = F + F*ERR(1) + SQRT(SIGMA(2,2))"
        result = _translate_line(line, _INTRINSICS)
        assert "eps[0]" in result
        assert "sigma[1][1]" in result

    def test_double_precision_literals(self):
        line = "X = 1.5D0 + 2.0D-3"
        result = _translate_line(line, _INTRINSICS)
        assert "D0" not in result
        assert "e0" in result.lower() or "e" in result.lower()


@pytest.mark.unit
class TestNMTRANCompiler:
    def test_compile_pk_basic(self):
        compiler = NMTRANCompiler()
        pk_code = """
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
"""
        fn = compiler.compile_pk(pk_code)
        theta = [1.5, 0.08, 30.0]
        eta = [0.0, 0.0, 0.0]
        result = fn(theta, eta)
        assert "KA" in result
        assert result["KA"] == pytest.approx(1.5)
        assert result["CL"] == pytest.approx(0.08)
        assert result["V"] == pytest.approx(30.0)

    def test_compile_pk_with_eta(self):
        compiler = NMTRANCompiler()
        pk_code = "CL = THETA(1)*EXP(ETA(1))"
        fn = compiler.compile_pk(pk_code)
        theta = [5.0]
        # With ETA(1) = log(2), CL should be 5*2 = 10
        eta = [math.log(2.0)]
        result = fn(theta, eta)
        assert result.get("CL", 0) == pytest.approx(10.0, rel=1e-5)

    def test_compile_pk_exposes_covariates_by_name(self):
        compiler = NMTRANCompiler()
        pk_code = "CL = THETA(1) * (WT/70) ** THETA(2)"
        fn = compiler.compile_pk(pk_code)
        result = fn([5.0, 0.75], [], covariates={"WT": 84.0})
        assert result.get("CL", 0) == pytest.approx(5.0 * (84.0 / 70.0) ** 0.75)

    def test_compile_pk_preserves_numeric_covariates_in_result(self):
        compiler = NMTRANCompiler()
        fn = compiler.compile_pk("CL = THETA(1) * (WT/70) ** THETA(2)")

        result = fn([5.0, 0.75], [], covariates={"WT": 84.0})

        assert result["WT"] == pytest.approx(84.0)

    def test_compile_error_basic(self):
        compiler = NMTRANCompiler()
        error_code = """
IPRED = F
W = THETA(1)*F
Y = F + W*EPS(1)
"""
        fn = compiler.compile_error(error_code)
        theta = [0.1]
        eta = []
        eps = [0.0]
        result = fn(theta, eta, eps, f=10.0)
        assert result.get("Y", None) == pytest.approx(10.0)

    def test_compile_error_proportional(self):
        compiler = NMTRANCompiler()
        error_code = "Y = F*(1 + EPS(1))"
        fn = compiler.compile_error(error_code)
        result = fn([], [], [0.0], f=5.0)
        assert result.get("Y", None) == pytest.approx(5.0)

    def test_compile_error_with_eps(self):
        compiler = NMTRANCompiler()
        error_code = "Y = F*(1 + EPS(1))"
        fn = compiler.compile_error(error_code)
        result = fn([], [], [0.1], f=10.0)
        assert result.get("Y", None) == pytest.approx(11.0)

    def test_compile_error_exposes_ipred_and_explicit_w(self):
        compiler = NMTRANCompiler()
        error_code = """
IPRED = F + 1
W = THETA(1)
Y = IPRED + W*EPS(1)
"""
        fn = compiler.compile_error(error_code)
        result = fn([0.25], [], [0.0], f=5.0)
        assert result.get("IPRED") == pytest.approx(6.0)
        assert result.get("W") == pytest.approx(0.25)
        assert result.get("Y") == pytest.approx(6.0)

    def test_compile_error_without_explicit_w_does_not_invent_w(self):
        compiler = NMTRANCompiler()
        error_code = """
IPRED = F + 1
Y = IPRED*(1 + EPS(1))
"""
        fn = compiler.compile_error(error_code)
        result = fn([], [], [0.0], f=5.0)
        assert result.get("IPRED") == pytest.approx(6.0)
        assert "W" not in result
        assert result.get("Y") == pytest.approx(6.0)

    def test_compile_error_supports_err_alias_and_sigma_references(self):
        compiler = NMTRANCompiler()
        error_code = """
W = SQRT(F**2*SIGMA(1,1) + SIGMA(2,2))
Y = F + F*ERR(1) + ERR(2)
"""
        fn = compiler.compile_error(error_code)
        result = fn([], [], [0.0, 0.0], f=5.0, sigma=[[0.04, 0.0], [0.0, 0.25]])
        assert result.get("W") == pytest.approx(math.sqrt(5.0**2 * 0.04 + 0.25))
        assert result.get("Y") == pytest.approx(5.0)

    def test_compile_error_exposes_covariates_by_name(self):
        compiler = NMTRANCompiler()
        error_code = "Y = F + WT*EPS(1)"
        fn = compiler.compile_error(error_code)
        result = fn([], [], [0.5], f=10.0, covariates={"WT": 4.0})
        assert result.get("Y") == pytest.approx(12.0)

    def test_compile_error_raw_call_preserves_internal_lowercase_values(self):
        compiler = NMTRANCompiler()
        error_code = "IPRED = F + 1\nW = THETA(1)\nY = IPRED + W*EPS(1)"
        fn = compiler.compile_error(error_code)

        raw = fn._call_raw([0.25], [], [0.0], f=5.0)

        assert raw["ipred"] == pytest.approx(6.0)
        assert raw["w"] == pytest.approx(0.25)
        assert raw["y"] == pytest.approx(6.0)

    def test_compile_error_tracks_when_amounts_are_used(self):
        compiler = NMTRANCompiler()

        no_amounts = compiler.compile_error("Y = F*(1 + EPS(1))")
        with_amounts = compiler.compile_error("Y = F + A(1)*EPS(1)")

        assert no_amounts._uses_amounts is False
        assert with_amounts._uses_amounts is True
