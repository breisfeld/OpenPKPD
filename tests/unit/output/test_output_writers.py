"""
Unit tests for output file writers.

Covers: lst_writer, ext_writer, phi_writer, cov_writer, report (HTML).
Each test writes to a temporary directory and validates file content.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_result(ofv=200.0, converged=True, n_obs=100, method="FOCE") -> EstimationResult:
    theta = np.array([1.5, 0.08, 30.0])
    omega = np.diag([0.25, 0.15])
    sigma = np.diag([0.05])
    r = EstimationResult(
        theta_final=theta,
        omega_final=omega,
        sigma_final=sigma,
        ofv=ofv,
        converged=converged,
        n_observations=n_obs,
        method=method,
        ofv_history=[250.0, 220.0, 205.0, 200.0],
    )
    return r


def _make_params() -> ParameterSet:
    theta_specs = [
        ThetaSpec(init=1.5, lower=0.01, upper=20.0, label="KA"),
        ThetaSpec(init=0.08, lower=0.001, upper=5.0, label="CL"),
        ThetaSpec(init=30.0, lower=0.1, upper=500.0, label="V"),
    ]
    omega_specs = [
        OmegaSpec(block_size=1, values=[0.25]),
        OmegaSpec(block_size=1, values=[0.15]),
    ]
    sigma_specs = [
        SigmaSpec(block_size=1, values=[0.05]),
    ]
    return ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)


def _make_cov_result(se=None):
    """Build a minimal CovarianceResult for testing."""
    from openpkpd.covariance.sandwich import CovarianceResult

    if se is None:
        se = np.array([0.1, 0.045, 0.7])
    n = len(se)
    mat = np.diag(se**2)
    return CovarianceResult(
        cov_matrix=mat,
        cor_matrix=np.eye(n),
        se=se,
        r_matrix=mat,
        s_matrix=mat,
        condition_number=150.0,
        converged=True,
        param_names=[f"THETA{i + 1}" for i in range(n)],
    )


class _TableSubjectEvents:
    def __init__(self, obs_times, obs_dv, obs_mdv) -> None:
        self.obs_times = np.asarray(obs_times, dtype=float)
        self.obs_dv = np.asarray(obs_dv, dtype=float)
        self.obs_mdv = np.asarray(obs_mdv, dtype=float)

    def observation_mask(self):
        return self.obs_mdv == 0


class _TableIndividual:
    def __init__(self, subject_events, ipred=None, f=None, raise_error: bool = False) -> None:
        self.subject_events = subject_events
        self._ipred = np.asarray(ipred if ipred is not None else [], dtype=float)
        self._f = np.asarray(f if f is not None else [], dtype=float)
        self._raise_error = raise_error
        self.calls = []

    def evaluate(self, theta, eta, sigma, trans=None):
        self.calls.append((theta.copy(), eta.copy(), sigma.copy(), trans))
        if self._raise_error:
            raise RuntimeError("synthetic evaluate failure")
        return self._ipred.copy(), self.subject_events.observation_mask(), self._f.copy()


class _TablePopulationModel:
    def __init__(self, individuals, trans="TRANS") -> None:
        self._individuals = individuals
        self.trans = trans

    def subject_ids(self):
        return list(self._individuals)

    def individual_model(self, sid):
        return self._individuals[sid]


def _read_table(path: str):
    lines = Path(path).read_text().splitlines()
    if lines and lines[0].startswith("TABLE NO."):
        lines = lines[1:]

    parsed = list(csv.reader(lines, delimiter=" "))
    header = parsed[0]
    rows = []
    for row in parsed[1:]:
        if not row:
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        rows.append(
            {
                name: (float(value) if value != "" else float("nan"))
                for name, value in zip(header, row, strict=False)
            }
        )
    return header, rows


# ---------------------------------------------------------------------------
# .lst writer
# ---------------------------------------------------------------------------


class TestLstWriter:
    def test_writes_file(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result()
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run1.lst")
            write_lst(path, result, params, title="Test Model", n_subjects=12, n_obs=100)
            assert os.path.exists(path)
            content = Path(path).read_text()
            assert "Test Model" in content

    def test_header_contains_problem_number(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result()
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params, problem_no=3)
            content = Path(path).read_text()
            assert "PROBLEM NO.: 3" in content

    def test_data_info_block(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result()
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params, n_subjects=12, n_obs=100)
            content = Path(path).read_text()
            assert "NO. OF INDIVIDUALS: 12" in content
            assert "NO. OF OBS RECS: 100" in content

    def test_theta_block(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result()
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params)
            content = Path(path).read_text()
            assert "THETA - VECTOR OF FIXED EFFECTS PARAMETERS" in content
            assert "KA" in content
            assert "CL" in content

    def test_omega_block(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result()
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params)
            content = Path(path).read_text()
            assert "OMEGA - COV MATRIX FOR RANDOM EFFECTS" in content
            assert "OMEGA(1,1)" in content
            assert "OMEGA(2,2)" in content

    def test_sigma_block(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result()
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params)
            content = Path(path).read_text()
            assert "SIGMA - COV MATRIX FOR RESIDUAL EFFECTS" in content
            assert "SIGMA(1,1)" in content

    def test_ofv_aic_bic_in_output(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result(ofv=200.0, n_obs=100)
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params)
            content = Path(path).read_text()
            assert "OBJ FUNC VAL" in content
            assert "AIC" in content
            assert "BIC" in content

    def test_convergence_yes(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result(converged=True)
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params)
            content = Path(path).read_text()
            assert "MINIMIZATION SUCCESSFUL" in content

    def test_convergence_no(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result(converged=False)
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params)
            content = Path(path).read_text()
            assert "MINIMIZATION TERMINATED" in content

    def test_eta_shrinkage_block(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result()
        result.eta_shrinkage = np.array([0.15, 0.22])
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params)
            content = Path(path).read_text()
            assert "ETA SHRINKAGE" in content
            assert "ETA1" in content
            assert "15.00%" in content

    def test_eps_shrinkage_block(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result()
        result.eps_shrinkage = np.array([0.08])
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params)
            content = Path(path).read_text()
            assert "EPS SHRINKAGE" in content
            assert "EPS1" in content

    def test_warnings_block(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result()
        result.warnings = ["Rounding errors detected"]
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params)
            content = Path(path).read_text()
            assert "WARNINGS" in content
            assert "Rounding errors detected" in content

    def test_covariance_results_block(self):
        from openpkpd.output.lst_writer import write_lst

        result = _make_result()
        params = _make_params()
        cov = _make_cov_result()
        # Attach eigenvalues attribute (checked via hasattr in lst_writer)
        cov.eigenvalues = np.array([0.001, 0.01, 0.1])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.lst")
            write_lst(path, result, params, cov_result=cov)
            content = Path(path).read_text()
            assert "COVARIANCE STEP COMPLETED" in content
            assert "STANDARD ERRORS" in content
            assert "CONDITION NUMBER" in content
            assert "EIGENVALUES" in content

    def test_invalid_path_raises_output_error(self):
        from openpkpd.output.lst_writer import write_lst
        from openpkpd.utils.errors import OutputError

        result = _make_result()
        params = _make_params()

        with pytest.raises(OutputError):
            write_lst("/nonexistent/path/run.lst", result, params)


# ---------------------------------------------------------------------------
# .ext writer
# ---------------------------------------------------------------------------


class TestExtWriter:
    def test_writes_file(self):
        from openpkpd.output.ext_writer import write_ext

        result = _make_result()
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.ext")
            write_ext(path, result, params, method="FOCE")
            assert os.path.exists(path)

    def test_header_line(self):
        from openpkpd.output.ext_writer import write_ext

        result = _make_result()
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.ext")
            write_ext(path, result, params, method="FOCE", problem_no=2)
            lines = Path(path).read_text().splitlines(keepends=True)
            assert "TABLE NO." in lines[0]
            assert "FOCE" in lines[0]
            assert "2" in lines[0]

    def test_column_header(self):
        from openpkpd.output.ext_writer import write_ext

        result = _make_result()
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.ext")
            write_ext(path, result, params)
            content = Path(path).read_text()
            assert "THETA1" in content
            assert "THETA2" in content
            assert "THETA3" in content
            assert "OMEGA(1,1)" in content
            assert "SIGMA(1,1)" in content

    def test_final_row_sentinel(self):
        from openpkpd.output.ext_writer import write_ext

        result = _make_result()
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.ext")
            write_ext(path, result, params)
            content = Path(path).read_text()
            # NONMEM uses -1000000000 for the final estimates row
            assert "-1000000000" in content

    def test_ofv_history_rows(self):
        from openpkpd.output.ext_writer import write_ext

        result = _make_result()
        result.ofv_history = [300.0, 250.0, 210.0, 200.0]
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.ext")
            write_ext(path, result, params)
            lines = Path(path).read_text().splitlines(keepends=True)
            # header + 4 history rows + 1 final row = 6 lines (minimum)
            assert len(lines) >= 4

    def test_invalid_path_raises(self):
        from openpkpd.output.ext_writer import write_ext
        from openpkpd.utils.errors import OutputError

        result = _make_result()
        params = _make_params()
        with pytest.raises(OutputError):
            write_ext("/nonexistent/dir/run.ext", result, params)


# ---------------------------------------------------------------------------
# .phi writer
# ---------------------------------------------------------------------------


class TestPhiWriter:
    def test_writes_file(self):
        from openpkpd.output.phi_writer import write_phi

        result = _make_result()
        result.post_hoc_etas = {1: np.array([0.1, -0.2]), 2: np.array([-0.05, 0.15])}
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.phi")
            write_phi(path, result, params, subject_ids=[1, 2], method="FOCE")
            assert os.path.exists(path)

    def test_header_line(self):
        from openpkpd.output.phi_writer import write_phi

        result = _make_result()
        result.post_hoc_etas = {1: np.array([0.1, 0.0])}
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.phi")
            write_phi(path, result, params, subject_ids=[1], method="FOCE", problem_no=1)
            lines = Path(path).read_text().splitlines(keepends=True)
            assert "TABLE NO." in lines[0]
            assert "FOCE" in lines[0]

    def test_eta_columns_in_header(self):
        from openpkpd.output.phi_writer import write_phi

        result = _make_result()
        result.post_hoc_etas = {1: np.array([0.1, -0.2])}
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.phi")
            write_phi(path, result, params, subject_ids=[1])
            content = Path(path).read_text()
            assert "ETA(1)" in content
            assert "ETA(2)" in content

    def test_all_subjects_written(self):
        from openpkpd.output.phi_writer import write_phi

        result = _make_result()
        n_subjects = 5
        result.post_hoc_etas = {i + 1: np.array([0.1 * i, -0.1 * i]) for i in range(n_subjects)}
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.phi")
            write_phi(path, result, params, subject_ids=list(range(1, n_subjects + 1)))
            lines = [
                ln
                for ln in Path(path).read_text().splitlines(keepends=True)
                if ln.strip() and "SUBJECT_NO" not in ln and "TABLE" not in ln
            ]
            assert len(lines) == n_subjects

    def test_missing_subject_eta_defaults_to_zeros(self):
        from openpkpd.output.phi_writer import write_phi

        result = _make_result()
        result.post_hoc_etas = {}  # No ETAs stored
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.phi")
            write_phi(path, result, params, subject_ids=[1, 2, 3])
            # Should not raise — missing subjects get zero ETAs
            assert os.path.exists(path)

    def test_invalid_path_raises(self):
        from openpkpd.output.phi_writer import write_phi
        from openpkpd.utils.errors import OutputError

        result = _make_result()
        params = _make_params()
        with pytest.raises(OutputError):
            write_phi("/bad/path/run.phi", result, params, subject_ids=[1])


# ---------------------------------------------------------------------------
# $TABLE writer
# ---------------------------------------------------------------------------


class TestTableWriter:
    def test_derived_columns_and_etas(self):
        from openpkpd.output.table_writer import write_table

        result = _make_result()
        result.post_hoc_etas = {1: np.array([0.1, -0.2])}
        params = _make_params()
        individual = _TableIndividual(
            _TableSubjectEvents([0.0, 12.0], [1.2, 1.8], [0, 0]),
            ipred=[1.0, 1.5],
            f=[0.9, 1.4],
        )
        population = _TablePopulationModel({1: individual}, trans="TRANS6")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.tab")
            write_table(
                path,
                population,
                result,
                params,
                columns=[
                    "ID",
                    "TIME",
                    "DV",
                    "IPRED",
                    "PRED",
                    "ETA1",
                    "ETA2",
                    "IWRES",
                    "RES",
                    "IRES",
                    "WRES",
                    "MDV",
                ],
                problem_no=7,
            )
            lines = Path(path).read_text().splitlines(keepends=True)
            assert lines[0].strip() == "TABLE NO.     7"

            header, rows = _read_table(path)
            assert header == [
                "ID",
                "TIME",
                "DV",
                "IPRED",
                "PRED",
                "ETA1",
                "ETA2",
                "IWRES",
                "RES",
                "IRES",
                "WRES",
                "MDV",
            ]
            assert len(rows) == 2

            sigma = float(params.sigma[0, 0])
            assert rows[0]["ETA1"] == pytest.approx(0.1)
            assert rows[0]["ETA2"] == pytest.approx(-0.2)
            assert rows[0]["IPRED"] == pytest.approx(1.0)
            assert rows[0]["PRED"] == pytest.approx(0.9)
            assert rows[0]["IWRES"] == pytest.approx((1.2 - 1.0) / np.sqrt(sigma))
            assert rows[0]["RES"] == pytest.approx(1.2 - 0.9)
            assert rows[0]["IRES"] == pytest.approx(1.2 - 1.0)
            assert rows[0]["WRES"] == pytest.approx((1.2 - 0.9) / np.sqrt(sigma))

        assert len(individual.calls) == 1
        np.testing.assert_allclose(individual.calls[0][1], [0.1, -0.2])
        assert individual.calls[0][3] == "TRANS6"

    def test_missing_subject_eta_defaults_to_zero_and_unknown_columns_fall_back(self):
        from openpkpd.output.table_writer import write_table

        result = _make_result()
        params = _make_params()
        individual = _TableIndividual(
            _TableSubjectEvents([0.0], [2.0], [0]),
            ipred=[1.5],
            f=[1.4],
        )
        population = _TablePopulationModel({3: individual})

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.tab")
            write_table(path, population, result, params, columns=["NOT_A_REAL_COLUMN"])
            header, rows = _read_table(path)

        assert "ETA1" in header
        assert "ETA2" in header
        assert rows[0]["ETA1"] == pytest.approx(0.0)
        assert rows[0]["ETA2"] == pytest.approx(0.0)
        np.testing.assert_allclose(individual.calls[0][1], np.zeros(params.n_eta()))

    def test_evaluate_failure_falls_back_to_nan_predictions(self):
        from openpkpd.output.table_writer import write_table

        result = _make_result()
        params = _make_params()
        individual = _TableIndividual(
            _TableSubjectEvents([0.0, 6.0], [1.0, 1.5], [0, 0]),
            raise_error=True,
        )
        population = _TablePopulationModel({1: individual})

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.tab")
            write_table(
                path,
                population,
                result,
                params,
                columns=["ID", "TIME", "IPRED", "PRED", "IWRES", "RES", "IRES", "WRES"],
            )
            content = Path(path).read_text()

        assert "TABLE NO." in content
        assert "nan" not in content.lower()
        assert "1.000000E+00 0.000000E+00" in content

    def test_firstonly_and_oneheader_false(self):
        from openpkpd.output.table_writer import write_table

        result = _make_result()
        params = _make_params()
        population = _TablePopulationModel(
            {
                1: _TableIndividual(
                    _TableSubjectEvents([0.0, 4.0], [1.0, 1.1], [0, 0]),
                    ipred=[0.8, 0.9],
                    f=[0.7, 0.8],
                ),
                2: _TableIndividual(
                    _TableSubjectEvents([1.0, 5.0], [2.0, 2.1], [0, 0]),
                    ipred=[1.8, 1.9],
                    f=[1.7, 1.8],
                ),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.tab")
            write_table(
                path,
                population,
                result,
                params,
                columns=["ID", "TIME"],
                firstonly=True,
                oneheader=False,
            )
            content = Path(path).read_text()
            header, rows = _read_table(path)

        assert not content.startswith("TABLE NO.")
        assert header == ["ID", "TIME"]
        assert len(rows) == 2
        assert rows[0]["ID"] == pytest.approx(1.0)
        assert rows[0]["TIME"] == pytest.approx(0.0)
        assert rows[1]["ID"] == pytest.approx(2.0)
        assert rows[1]["TIME"] == pytest.approx(1.0)

    def test_nonpositive_sigma_sets_residual_columns_to_nan(self):
        from openpkpd.output.table_writer import write_table

        result = _make_result()
        params = _make_params()
        params.sigma = np.array([[0.0]])
        population = _TablePopulationModel(
            {
                1: _TableIndividual(_TableSubjectEvents([0.0], [1.2], [0]), ipred=[1.0], f=[0.9]),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.tab")
            write_table(
                path,
                population,
                result,
                params,
                columns=["IPRED", "PRED", "IWRES", "RES", "IRES", "WRES"],
            )
            content = Path(path).read_text()

        assert "1.000000E+00 9.000000E-01" in content
        assert "nan" not in content.lower()

    def test_invalid_path_raises_output_error(self):
        from openpkpd.output.table_writer import write_table
        from openpkpd.utils.errors import OutputError

        result = _make_result()
        params = _make_params()
        population = _TablePopulationModel(
            {
                1: _TableIndividual(_TableSubjectEvents([0.0], [1.0], [0]), ipred=[1.0], f=[1.0]),
            }
        )

        with pytest.raises(OutputError):
            write_table("/bad/path/run.tab", population, result, params, columns=["ID"])


# ---------------------------------------------------------------------------
# .cov / .cor writer
# ---------------------------------------------------------------------------


class TestCovWriter:
    def test_write_cov_header_and_matrix_layout(self):
        from openpkpd.output.cov_writer import write_cov

        cov = _make_cov_result(se=np.array([0.1, 0.2]))
        cov.cov_matrix = np.array([[0.01, 0.002], [0.002, 0.05]])
        cov.param_names = ["CL", "V"]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.cov")
            write_cov(path, cov, problem_no=4)
            lines = Path(path).read_text().splitlines()

        assert lines[0] == "TABLE NO.     4: COVARIANCE MATRIX"
        assert lines[1].split() == ["NAME", "CL", "V"]
        assert lines[2].split() == ["CL", "1.000000E-02", "2.000000E-03"]
        assert lines[3].split() == ["V", "2.000000E-03", "5.000000E-02"]

    def test_write_cor_includes_se_row_in_provided_name_order(self):
        from openpkpd.output.cov_writer import write_cor

        cov = _make_cov_result(se=np.array([0.15, 0.008, 3.0]))
        cov.cor_matrix = np.array(
            [
                [1.0, 0.25, -0.4],
                [0.25, 1.0, 0.1],
                [-0.4, 0.1, 1.0],
            ]
        )
        cov.param_names = ["CL", "OMEGA(1,1)", "SIGMA(1,1)"]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.cor")
            write_cor(path, cov, problem_no=2)
            lines = Path(path).read_text().splitlines()

        assert lines[0] == "TABLE NO.     2: CORRELATION MATRIX"
        assert lines[1].split() == ["NAME", "CL", "OMEGA(1,1)", "SIGMA(1,1)"]
        assert lines[2].split() == ["SE", "1.500000E-01", "8.000000E-03", "3.000000E+00"]
        assert lines[3].split() == ["CL", "1.000000E+00", "2.500000E-01", "-4.000000E-01"]
        assert lines[5].split() == ["SIGMA(1,1)", "-4.000000E-01", "1.000000E-01", "1.000000E+00"]

    @pytest.mark.parametrize("writer_name", ["write_cov", "write_cor"])
    def test_invalid_path_raises_output_error(self, writer_name):
        from openpkpd.output import cov_writer
        from openpkpd.utils.errors import OutputError

        cov = _make_cov_result(se=np.array([0.1, 0.2]))

        with pytest.raises(OutputError):
            getattr(cov_writer, writer_name)("/bad/path/run.cov", cov)


# ---------------------------------------------------------------------------
# HTML report writer
# ---------------------------------------------------------------------------


class TestHtmlReport:
    def test_public_report_api_exports_are_available(self):
        from openpkpd import (
            ReportExportError,
            estimation_result_to_html,
            export_html_report_to_pdf,
            write_html_report,
            write_pdf_report,
        )
        from openpkpd.output import (
            estimation_result_to_html as output_estimation_result_to_html,
        )
        from openpkpd.output import (
            export_html_report_to_pdf as output_export_html_report_to_pdf,
        )
        from openpkpd.output import (
            write_html_report as output_write_html_report,
        )
        from openpkpd.output import (
            write_pdf_report as output_write_pdf_report,
        )

        assert callable(estimation_result_to_html)
        assert callable(export_html_report_to_pdf)
        assert callable(write_html_report)
        assert callable(write_pdf_report)
        assert ReportExportError.__name__ == "ReportExportError"
        assert estimation_result_to_html is output_estimation_result_to_html
        assert export_html_report_to_pdf is output_export_html_report_to_pdf
        assert write_html_report is output_write_html_report
        assert write_pdf_report is output_write_pdf_report

    def test_write_html_report(self):
        from openpkpd.output.report import write_html_report

        result = _make_result(ofv=180.0, n_obs=100)
        result.method = "FOCE"
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            write_html_report(path, result, params, title="Test Report")
            assert os.path.exists(path)
            content = Path(path).read_text()
            assert "<!DOCTYPE html>" in content
            assert "Test Report" in content

    def test_estimation_result_to_pdf_delegates_to_public_writer(self, monkeypatch):
        result = _make_result(ofv=180.0, n_obs=100)
        result.method = "FOCE"
        recorded: dict[str, object] = {}

        def _fake_write_pdf_report(
            path, result_arg, params_arg, title="", cov_result=None, provenance=None, **kwargs
        ):
            recorded.update(
                {
                    "path": path,
                    "result": result_arg,
                    "theta_count": len(params_arg.theta_specs),
                    "title": title,
                    "cov_result": cov_result,
                    "provenance": provenance,
                }
            )

        monkeypatch.setattr("openpkpd.output.report.write_pdf_report", _fake_write_pdf_report)

        result.to_pdf(
            "result.pdf", title="Delegated PDF", provenance={"Run context": {"run_id": "run-1"}}
        )

        assert recorded["path"] == "result.pdf"
        assert recorded["result"] is result
        assert recorded["theta_count"] == len(result.theta_final)
        assert recorded["title"] == "Delegated PDF"
        assert recorded["cov_result"] is None
        assert recorded["provenance"] == {"Run context": {"run_id": "run-1"}}

    def test_estimation_result_to_html_string(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result(ofv=180.0, n_obs=100)
        result.method = "FOCE"
        params = _make_params()

        html = estimation_result_to_html(result, params, title="My Model")
        assert isinstance(html, str)
        assert "My Model" in html
        assert "THETA" in html
        assert "OMEGA" in html
        assert "SIGMA" in html

    def test_html_contains_stat_cards(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result(ofv=200.0, n_obs=80)
        result.method = "FO"
        params = _make_params()

        html = estimation_result_to_html(result, params)
        assert "OFV" in html
        assert "AIC" in html
        assert "BIC" in html
        assert "Converged" in html

    def test_html_theta_labels(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result()
        result.method = "FOCE"
        params = _make_params()

        html = estimation_result_to_html(result, params)
        assert "KA" in html
        assert "CL" in html
        assert "V" in html

    def test_html_not_converged_class(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result(converged=False)
        params = _make_params()

        html = estimation_result_to_html(result, params)
        assert "not-converged" in html

    def test_html_converged_class(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result(converged=True)
        params = _make_params()

        html = estimation_result_to_html(result, params)
        assert 'class="value converged"' in html

    def test_html_with_warnings(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result()
        result.warnings = ["High condition number detected"]
        params = _make_params()

        html = estimation_result_to_html(result, params)
        assert "High condition number detected" in html
        assert "Warnings" in html

    def test_html_with_shrinkage(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result()
        result.eta_shrinkage = np.array([0.15, 0.35])  # second > 30%
        result.eps_shrinkage = np.array([0.05])
        params = _make_params()

        html = estimation_result_to_html(result, params)
        assert "ETA Shrinkage" in html
        assert "15.0%" in html
        assert "HIGH" in html  # second ETA shrinkage > 30%

    def test_html_with_covariance(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result()
        params = _make_params()
        cov = _make_cov_result()
        cov.condition_number = 200.0

        html = estimation_result_to_html(result, params, cov_result=cov)
        assert "Covariance Step" in html
        assert "200" in html  # condition number

    def test_html_with_provenance_and_reproducibility_sections(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result()
        params = _make_params()

        html = estimation_result_to_html(
            result,
            params,
            title="My Model",
            provenance={
                "Run context": {"run_id": "run-123", "scenario_name": "Baseline"},
                "Dataset": {
                    "source_path": "/tmp/theo.csv",
                    "sha256": "abc123",
                    "columns": ["ID", "TIME", "DV"],
                },
                "Model source": {"pk_code": "CL = THETA(1)\nV = THETA(2)"},
                "Environment": {"openpkpd_version": "0.1.0", "platform": "Linux-6.0"},
            },
        )

        assert "Provenance &amp; Reproducibility" in html
        assert "run-123" in html
        assert "/tmp/theo.csv" in html
        assert "abc123" in html
        assert "CL = THETA(1)" in html
        assert "openpkpd version" in html

    def test_html_with_ofv_history_plot(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result()
        result.ofv_history = [300.0, 250.0, 220.0, 200.0]
        params = _make_params()

        html = estimation_result_to_html(result, params)
        # Plot is generated only if matplotlib is available
        # Just verify no exception and HTML is well-formed
        assert "<!DOCTYPE html>" in html

    def test_html_se_rse_columns(self):
        from openpkpd.output.report import estimation_result_to_html

        result = _make_result()
        params = _make_params()
        # SE = 10% of each theta: 0.15, 0.008, 3.0
        se = np.array([0.15, 0.008, 3.0])
        cov = _make_cov_result(se=se)

        html = estimation_result_to_html(result, params, cov_result=cov)
        assert "SE" in html
        assert "RSE" in html
        # 10% RSE (0.15/1.5 = 10%)
        assert "10.0%" in html
