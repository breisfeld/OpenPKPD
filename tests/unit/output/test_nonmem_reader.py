"""
Tests for the NONMEM result file reader (output/nonmem_reader.py).

Uses temporary files containing representative NONMEM output to verify
correct parsing of .ext, .lst, .phi, and .cov formats.
"""

from __future__ import annotations

import math
import textwrap

import numpy as np
import pytest

from openpkpd.output.nonmem_reader import (
    NonmemResult,
    _parse_cov,
    _parse_ext,
    _parse_lst,
    _parse_phi,
    _vector_to_matrix,
    read_nonmem_results,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal NONMEM output files
# ---------------------------------------------------------------------------

EXT_CONTENT = textwrap.dedent("""\
TABLE NO.  1: ESTIMATION METHOD: FOCE INTERACTION
ITERATION   THETA1   THETA2   THETA3   OMEGA(1,1)   SIGMA(1,1)   OBJ
0           1.500    3.000   35.000   0.090   0.010   220.000
1           1.499    3.001   35.001   0.090   0.010   215.000
-1000000000 1.4999   3.0001  34.9999  0.0901  0.0101  214.400
""")

LST_CONTENT = textwrap.dedent("""\
$PROBLEM Theophylline 1-cmt oral PK
 ESTIMATION METHOD USED: FOCE
 MINIMIZATION SUCCESSFUL
 #OBJV:  214.4000
""")

PHI_CONTENT = textwrap.dedent("""\
TABLE NO. 1
SUBJECT_NO  ID  ETA1
1           1   -0.120
2           2    0.200
3           3    0.050
""")

COV_CONTENT = textwrap.dedent("""\
TABLE NO.  1: FOCE COVARIANCE
NAME      THETA1    THETA2
THETA1    0.010000  0.002000
THETA2    0.002000  0.050000
""")


@pytest.fixture()
def ext_file(tmp_path):
    p = tmp_path / "run.ext"
    p.write_text(EXT_CONTENT)
    return str(p)


@pytest.fixture()
def lst_file(tmp_path):
    p = tmp_path / "run.lst"
    p.write_text(LST_CONTENT)
    return str(p)


@pytest.fixture()
def phi_file(tmp_path):
    p = tmp_path / "run.phi"
    p.write_text(PHI_CONTENT)
    return str(p)


@pytest.fixture()
def cov_file(tmp_path):
    p = tmp_path / "run.cov"
    p.write_text(COV_CONTENT)
    return str(p)


# ---------------------------------------------------------------------------
# _vector_to_matrix
# ---------------------------------------------------------------------------


class TestVectorToMatrix:
    def test_1x1(self):
        mat = _vector_to_matrix(np.array([4.0]))
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(4.0)

    def test_2x2_symmetric(self):
        # lower triangle: (1,1), (2,1), (2,2)
        vec = np.array([1.0, 0.5, 2.0])
        mat = _vector_to_matrix(vec)
        assert mat.shape == (2, 2)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 0] == pytest.approx(0.5)
        assert mat[0, 1] == pytest.approx(0.5)  # symmetrised
        assert mat[1, 1] == pytest.approx(2.0)

    def test_3x3(self):
        vec = np.array([1.0, 0.1, 2.0, 0.2, 0.3, 3.0])
        mat = _vector_to_matrix(vec)
        assert mat.shape == (3, 3)
        assert mat[2, 0] == pytest.approx(0.2)
        assert mat[0, 2] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# _parse_ext
# ---------------------------------------------------------------------------


class TestParseExt:
    def test_theta_final(self, ext_file):
        result = _parse_ext(ext_file)
        theta = result["theta_final"]
        assert len(theta) == 3
        assert theta[0] == pytest.approx(1.4999, rel=1e-3)

    def test_theta_names(self, ext_file):
        result = _parse_ext(ext_file)
        assert result["theta_names"] == ["THETA1", "THETA2", "THETA3"]

    def test_final_ofv(self, ext_file):
        result = _parse_ext(ext_file)
        assert result["ofv"] == pytest.approx(214.4, rel=1e-3)

    def test_ofv_history_length(self, ext_file):
        result = _parse_ext(ext_file)
        # 3 rows: iter 0, 1, -1000000000
        assert len(result["ofv_history"]) == 3

    def test_ofv_history_final_marker(self, ext_file):
        result = _parse_ext(ext_file)
        iters = [it for it, _ in result["ofv_history"]]
        assert -1000000000 in iters

    def test_omega_vector_extracted(self, ext_file):
        result = _parse_ext(ext_file)
        omega_vec = result["omega_vector"]
        assert len(omega_vec) == 1
        assert omega_vec[0] == pytest.approx(0.0901, rel=1e-2)

    def test_sigma_vector_extracted(self, ext_file):
        result = _parse_ext(ext_file)
        sigma_vec = result["sigma_vector"]
        assert len(sigma_vec) == 1
        assert sigma_vec[0] == pytest.approx(0.0101, rel=1e-2)


# ---------------------------------------------------------------------------
# _parse_lst
# ---------------------------------------------------------------------------


class TestParseLst:
    def test_converged_true(self, lst_file):
        result = _parse_lst(lst_file)
        assert result["converged"] is True

    def test_termination_message(self, lst_file):
        result = _parse_lst(lst_file)
        assert "MINIMIZATION SUCCESSFUL" in result["termination_message"].upper()

    def test_ofv_extracted(self, lst_file):
        result = _parse_lst(lst_file)
        assert result["ofv"] == pytest.approx(214.4, rel=1e-3)

    def test_method_extracted(self, lst_file):
        result = _parse_lst(lst_file)
        assert "FOCE" in result["method"].upper()

    def test_failed_minimization(self, tmp_path):
        content = "MINIMIZATION TERMINATED\n#OBJV: 999.0\n"
        p = tmp_path / "fail.lst"
        p.write_text(content)
        result = _parse_lst(str(p))
        assert result["converged"] is False


# ---------------------------------------------------------------------------
# _parse_phi
# ---------------------------------------------------------------------------


class TestParsePhi:
    def test_subject_count(self, phi_file):
        data = _parse_phi(phi_file)
        assert len(data) == 3

    def test_subject_eta_values(self, phi_file):
        data = _parse_phi(phi_file)
        assert 1 in data
        assert data[1][0] == pytest.approx(-0.120, rel=1e-2)

    def test_all_subjects_have_eta(self, phi_file):
        data = _parse_phi(phi_file)
        for _sid, eta in data.items():
            assert isinstance(eta, np.ndarray)
            assert len(eta) > 0


# ---------------------------------------------------------------------------
# _parse_cov
# ---------------------------------------------------------------------------


class TestParseCov:
    def test_matrix_shape(self, cov_file):
        result = _parse_cov(cov_file)
        mat = result["matrix"]
        assert mat.shape == (2, 2)

    def test_diagonal_values(self, cov_file):
        result = _parse_cov(cov_file)
        mat = result["matrix"]
        assert mat[0, 0] == pytest.approx(0.01, rel=1e-3)
        assert mat[1, 1] == pytest.approx(0.05, rel=1e-3)

    def test_symmetric(self, cov_file):
        result = _parse_cov(cov_file)
        mat = result["matrix"]
        np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_names(self, cov_file):
        result = _parse_cov(cov_file)
        assert result["names"] == ["THETA1", "THETA2"]


# ---------------------------------------------------------------------------
# read_nonmem_results (integration)
# ---------------------------------------------------------------------------


class TestReadNonmemResults:
    def test_all_files_parsed(self, ext_file, lst_file, phi_file, cov_file):
        result = read_nonmem_results(ext_file, lst_file, phi_file, cov_file)
        assert isinstance(result, NonmemResult)
        assert len(result.source_files) == 4

    def test_theta_final(self, ext_file, lst_file, phi_file, cov_file):
        result = read_nonmem_results(ext_file, lst_file, phi_file, cov_file)
        assert len(result.theta_final) == 3
        assert result.theta_final[0] == pytest.approx(1.4999, rel=1e-3)

    def test_ofv(self, ext_file, lst_file, phi_file, cov_file):
        result = read_nonmem_results(ext_file, lst_file, phi_file, cov_file)
        assert result.ofv == pytest.approx(214.4, rel=1e-3)

    def test_converged(self, ext_file, lst_file, phi_file, cov_file):
        result = read_nonmem_results(ext_file, lst_file, phi_file, cov_file)
        assert result.converged is True

    def test_post_hoc_etas(self, ext_file, lst_file, phi_file, cov_file):
        result = read_nonmem_results(ext_file, lst_file, phi_file, cov_file)
        assert len(result.post_hoc_etas) == 3

    def test_se_theta_from_cov(self, ext_file, lst_file, phi_file, cov_file):
        result = read_nonmem_results(ext_file, lst_file, phi_file, cov_file)
        assert len(result.se_theta) >= 1
        assert result.se_theta[0] == pytest.approx(0.1, rel=1e-2)

    def test_none_paths_skipped(self):
        result = read_nonmem_results(None, None, None, None)
        assert isinstance(result, NonmemResult)
        assert math.isnan(result.ofv)

    def test_missing_files_skipped(self, tmp_path):
        result = read_nonmem_results(
            str(tmp_path / "no.ext"),
            str(tmp_path / "no.lst"),
        )
        assert isinstance(result, NonmemResult)
        assert len(result.source_files) == 0

    def test_summary_string(self, ext_file, lst_file, phi_file, cov_file):
        result = read_nonmem_results(ext_file, lst_file, phi_file, cov_file)
        s = result.summary()
        assert "OFV" in s
        assert "Converged" in s

    def test_omega_matrix_shape(self, ext_file, lst_file, phi_file, cov_file):
        result = read_nonmem_results(ext_file, lst_file, phi_file, cov_file)
        assert result.omega_final.shape == (1, 1)
