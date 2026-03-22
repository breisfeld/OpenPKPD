"""Tests for to_cdisc_pp() CDISC PP domain output."""

import math

import pandas as pd
import pytest

from openpkpd.nca.cdisc_pp import to_cdisc_pp


@pytest.fixture()
def nca_df():
    """Minimal NCA results DataFrame (two subjects)."""
    return pd.DataFrame(
        [
            {
                "subject_id": "SUBJ01",
                "c0": 150.0,
                "cmax": 120.5,
                "tmax": 1.5,
                "auc_last": 500.0,
                "auc_inf": 550.0,
                "t_half": 8.0,
                "lambda_z": math.log(2) / 8.0,
                "cl_f": 10.0 / 550.0,
                "vz_f": 1.2,
                "mrt": 12.0,
            },
            {
                "subject_id": "SUBJ02",
                "c0": 110.0,
                "cmax": 95.0,
                "tmax": 2.0,
                "auc_last": 420.0,
                "auc_inf": 460.0,
                "t_half": 9.0,
                "lambda_z": math.log(2) / 9.0,
                "cl_f": 10.0 / 460.0,
                "vz_f": 1.1,
                "mrt": 13.0,
            },
        ]
    )


class TestToCdiscPP:
    def test_required_columns_present(self, nca_df):
        result = to_cdisc_pp(nca_df, study_id="STUDY99")
        required = {"STUDYID", "USUBJID", "DOMAIN", "PARAMCD", "PARAM", "AVAL", "DTYPE"}
        assert required.issubset(set(result.columns))

    def test_study_id_propagated(self, nca_df):
        result = to_cdisc_pp(nca_df, study_id="MYSTUDY")
        assert (result["STUDYID"] == "MYSTUDY").all()

    def test_domain_column(self, nca_df):
        result = to_cdisc_pp(nca_df, domain="PP")
        assert (result["DOMAIN"] == "PP").all()

    def test_paramcd_values(self, nca_df):
        result = to_cdisc_pp(nca_df)
        expected_codes = {
            "C0",
            "CMAX",
            "TMAX",
            "AUCLST",
            "AUCIFO",
            "THALF",
            "LAMZ",
            "CLF",
            "VZF",
            "MRT",
        }
        actual_codes = set(result["PARAMCD"].unique())
        assert expected_codes == actual_codes

    def test_c0_exported_when_present(self, nca_df):
        result = to_cdisc_pp(nca_df)
        subj1_c0 = result[(result["USUBJID"] == "SUBJ01") & (result["PARAMCD"] == "C0")]
        assert len(subj1_c0) == 1
        assert subj1_c0["AVAL"].iloc[0] == pytest.approx(150.0)

    def test_aval_matches_source(self, nca_df):
        result = to_cdisc_pp(nca_df)
        subj1_cmax = result[(result["USUBJID"] == "SUBJ01") & (result["PARAMCD"] == "CMAX")]
        assert len(subj1_cmax) == 1
        assert subj1_cmax["AVAL"].iloc[0] == pytest.approx(120.5)

    def test_row_count(self, nca_df):
        result = to_cdisc_pp(nca_df)
        # 10 PARAMCD codes × 2 subjects = 20 rows
        assert len(result) == 20

    def test_subject_ids_present(self, nca_df):
        result = to_cdisc_pp(nca_df)
        assert set(result["USUBJID"].unique()) == {"SUBJ01", "SUBJ02"}

    def test_empty_dataframe(self):
        empty = pd.DataFrame()
        result = to_cdisc_pp(empty)
        assert len(result) == 0
        assert "PARAMCD" in result.columns

    def test_custom_usubjid_col(self):
        df = pd.DataFrame([{"SUBJ": "X01", "cmax": 10.0, "tmax": 1.0}])
        result = to_cdisc_pp(df, usubjid_col="SUBJ")
        assert "X01" in result["USUBJID"].values

    def test_missing_param_skipped(self):
        """If a key is absent from a row it should be silently skipped."""
        df = pd.DataFrame([{"subject_id": "S1", "cmax": 5.0}])
        result = to_cdisc_pp(df)
        assert set(result["PARAMCD"].unique()) == {"CMAX"}
