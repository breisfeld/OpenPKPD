"""Smoke tests for write_cdisc_adppk(), write_sdtm_pc(), and write_sdtm_adsl()."""

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.output.cdisc_writer import write_cdisc_adppk, write_sdtm_adsl, write_sdtm_pc

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


class _FakeResult:
    theta_final = np.array([2.0, 10.0])
    omega_final = np.diag([0.1, 0.2])
    sigma_final = np.diag([0.05])
    post_hoc_etas = {1: np.array([0.1, -0.2]), 2: np.array([-0.05, 0.15])}


class _FakeDataset:
    """Minimal NONMEMDataset stub."""

    def observation_rows(self):
        return pd.DataFrame(
            {
                "ID": [1, 1, 2, 2],
                "TIME": [1.0, 2.0, 1.0, 2.0],
                "DV": [45.0, 30.0, 55.0, 35.0],
            }
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteCdiscAdppk:
    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "adppk.csv")
        write_cdisc_adppk(_FakeResult(), _FakeDataset(), path)
        assert (tmp_path / "adppk.csv").exists()

    def test_required_columns(self, tmp_path):
        path = str(tmp_path / "adppk.csv")
        write_cdisc_adppk(_FakeResult(), _FakeDataset(), path)
        df = pd.read_csv(path)
        required = {"STUDYID", "USUBJID", "PARAMCD", "PARAM", "AVAL", "AVALU", "DTYPE"}
        assert required.issubset(set(df.columns))

    def test_observation_rows_present(self, tmp_path):
        path = str(tmp_path / "adppk.csv")
        write_cdisc_adppk(_FakeResult(), _FakeDataset(), path)
        df = pd.read_csv(path)
        obs = df[df["DTYPE"] == "OBSERVATION"]
        assert len(obs) == 4
        assert (obs["PARAMCD"] == "CONC").all()

    def test_theta_rows(self, tmp_path):
        path = str(tmp_path / "adppk.csv")
        write_cdisc_adppk(_FakeResult(), _FakeDataset(), path)
        df = pd.read_csv(path)
        thetas = df[df["DTYPE"] == "THETA"]
        assert set(thetas["PARAMCD"].unique()) == {"THETA1", "THETA2"}
        assert thetas[thetas["PARAMCD"] == "THETA1"]["AVAL"].iloc[0] == pytest.approx(2.0)

    def test_omega_rows(self, tmp_path):
        path = str(tmp_path / "adppk.csv")
        write_cdisc_adppk(_FakeResult(), _FakeDataset(), path)
        df = pd.read_csv(path)
        omegas = df[df["DTYPE"] == "OMEGA"]
        # 2×2 lower-triangular: (1,1), (2,1), (2,2) = 3 rows
        assert len(omegas) == 3

    def test_sigma_rows(self, tmp_path):
        path = str(tmp_path / "adppk.csv")
        write_cdisc_adppk(_FakeResult(), _FakeDataset(), path)
        df = pd.read_csv(path)
        sigmas = df[df["DTYPE"] == "SIGMA"]
        assert len(sigmas) == 1
        assert sigmas["PARAMCD"].iloc[0] == "SIGMA(1,1)"

    def test_eta_rows(self, tmp_path):
        path = str(tmp_path / "adppk.csv")
        write_cdisc_adppk(_FakeResult(), _FakeDataset(), path)
        df = pd.read_csv(path)
        etas = df[df["DTYPE"] == "ETA"]
        # 2 subjects × 2 ETAs = 4 rows
        assert len(etas) == 4

    def test_study_id_propagated(self, tmp_path):
        path = str(tmp_path / "adppk.csv")
        write_cdisc_adppk(_FakeResult(), _FakeDataset(), path, study_id="TESTSDY")
        df = pd.read_csv(path)
        assert (df["STUDYID"] == "TESTSDY").all()

    def test_exported_from_output_init(self):
        from openpkpd.output import write_cdisc_adppk as wca

        assert callable(wca)


# ── Helpers for SDTM tests ─────────────────────────────────────────────────────


def _make_dataset(with_demographics: bool = False) -> NONMEMDataset:
    """Build a minimal NONMEMDataset for SDTM tests."""
    data = {
        "ID": [1, 1, 1, 2, 2, 2],
        "TIME": [0.0, 1.0, 4.0, 0.0, 1.0, 4.0],
        "DV": [np.nan, 45.0, 30.0, np.nan, 55.0, 35.0],
        "AMT": [100.0, 0.0, 0.0, 100.0, 0.0, 0.0],
        "EVID": [1, 0, 0, 1, 0, 0],
        "MDV": [1, 0, 0, 1, 0, 0],
    }
    if with_demographics:
        data["AGE"] = [34.0, 34.0, 34.0, 52.0, 52.0, 52.0]
        data["SEX"] = ["M", "M", "M", "F", "F", "F"]
        data["WT"] = [70.0, 70.0, 70.0, 65.0, 65.0, 65.0]
    return NONMEMDataset.from_dataframe(pd.DataFrame(data))


# ── write_sdtm_pc tests ────────────────────────────────────────────────────────


class TestWriteSDTMPC:
    def test_creates_file(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_pc(ds, str(tmp_path / "pc.csv"))
        assert (tmp_path / "pc.csv").exists()

    def test_domain_column_is_PC(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_pc(ds, str(tmp_path / "pc.csv"))
        df = pd.read_csv(tmp_path / "pc.csv")
        assert (df["DOMAIN"] == "PC").all()

    def test_required_sdtm_columns(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_pc(ds, str(tmp_path / "pc.csv"))
        df = pd.read_csv(tmp_path / "pc.csv")
        required = {
            "STUDYID",
            "DOMAIN",
            "USUBJID",
            "PCSEQ",
            "PCTESTCD",
            "PCTEST",
            "PCORRES",
            "PCORRESU",
            "PCSTRESC",
            "PCSTRESN",
            "PCSTRESU",
            "NFRLT",
        }
        assert required.issubset(set(df.columns))

    def test_one_row_per_observation(self, tmp_path):
        ds = _make_dataset()
        obs_count = len(ds.observation_rows())
        write_sdtm_pc(ds, str(tmp_path / "pc.csv"))
        df = pd.read_csv(tmp_path / "pc.csv")
        assert len(df) == obs_count

    def test_pcseq_increments_per_subject(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_pc(ds, str(tmp_path / "pc.csv"))
        df = pd.read_csv(tmp_path / "pc.csv")
        for _subj, grp in df.groupby("USUBJID"):
            assert list(grp["PCSEQ"]) == list(range(1, len(grp) + 1))

    def test_study_id_propagated(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_pc(ds, str(tmp_path / "pc.csv"), study_id="ABC123")
        df = pd.read_csv(tmp_path / "pc.csv")
        assert (df["STUDYID"] == "ABC123").all()

    def test_pcstresn_numeric_values(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_pc(ds, str(tmp_path / "pc.csv"))
        df = pd.read_csv(tmp_path / "pc.csv")
        numeric = pd.to_numeric(df["PCSTRESN"], errors="coerce")
        assert numeric.notna().all()

    def test_unit_columns(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_pc(ds, str(tmp_path / "pc.csv"), unit="ug/mL")
        df = pd.read_csv(tmp_path / "pc.csv")
        assert (df["PCORRESU"] == "ug/mL").all()
        assert (df["PCSTRESU"] == "ug/mL").all()

    def test_nfrlt_matches_observation_times(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_pc(ds, str(tmp_path / "pc.csv"))
        df = pd.read_csv(tmp_path / "pc.csv")
        obs = ds.observation_rows()
        assert sorted(df["NFRLT"].tolist()) == sorted(obs["TIME"].tolist())

    def test_pctestcd_max_8_chars(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_pc(ds, str(tmp_path / "pc.csv"), test_code="TOOLONGCODE")
        df = pd.read_csv(tmp_path / "pc.csv")
        assert all(len(c) <= 8 for c in df["PCTESTCD"])

    def test_exported_from_output_init(self):
        from openpkpd.output import write_sdtm_pc as wpc

        assert callable(wpc)


# ── write_sdtm_adsl tests ─────────────────────────────────────────────────────


class TestWriteSDTMAdsl:
    def test_creates_file(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_adsl(ds, str(tmp_path / "adsl.csv"))
        assert (tmp_path / "adsl.csv").exists()

    def test_one_row_per_subject(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_adsl(ds, str(tmp_path / "adsl.csv"))
        df = pd.read_csv(tmp_path / "adsl.csv")
        assert len(df) == ds.n_subjects()

    def test_required_adsl_columns(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_adsl(ds, str(tmp_path / "adsl.csv"))
        df = pd.read_csv(tmp_path / "adsl.csv")
        required = {"STUDYID", "USUBJID", "SUBJID", "SAFFL", "ITTFL"}
        assert required.issubset(set(df.columns))

    def test_all_subjects_flagged_safe(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_adsl(ds, str(tmp_path / "adsl.csv"))
        df = pd.read_csv(tmp_path / "adsl.csv")
        assert (df["SAFFL"] == "Y").all()
        assert (df["ITTFL"] == "Y").all()

    def test_demographic_columns_included_when_present(self, tmp_path):
        ds = _make_dataset(with_demographics=True)
        write_sdtm_adsl(ds, str(tmp_path / "adsl.csv"))
        df = pd.read_csv(tmp_path / "adsl.csv")
        assert "AGE" in df.columns
        assert "SEX" in df.columns
        assert "WT" in df.columns

    def test_demographic_values_from_first_record(self, tmp_path):
        ds = _make_dataset(with_demographics=True)
        write_sdtm_adsl(ds, str(tmp_path / "adsl.csv"))
        df = pd.read_csv(tmp_path / "adsl.csv")
        df["SUBJID"] = df["SUBJID"].astype(str)
        subj1 = df[df["SUBJID"] == "1"].iloc[0]
        assert subj1["AGE"] == pytest.approx(34.0)
        assert subj1["SEX"] == "M"

    def test_no_demographic_cols_when_absent(self, tmp_path):
        ds = _make_dataset(with_demographics=False)
        write_sdtm_adsl(ds, str(tmp_path / "adsl.csv"))
        df = pd.read_csv(tmp_path / "adsl.csv")
        for col in ("AGE", "SEX", "RACE"):
            assert col not in df.columns

    def test_explicit_demographic_columns(self, tmp_path):
        ds = _make_dataset(with_demographics=True)
        write_sdtm_adsl(ds, str(tmp_path / "adsl.csv"), demographic_columns=["AGE"])
        df = pd.read_csv(tmp_path / "adsl.csv")
        assert "AGE" in df.columns
        assert "SEX" not in df.columns

    def test_study_id_propagated(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_adsl(ds, str(tmp_path / "adsl.csv"), study_id="XYZ")
        df = pd.read_csv(tmp_path / "adsl.csv")
        assert (df["STUDYID"] == "XYZ").all()

    def test_usubjid_includes_study_id(self, tmp_path):
        ds = _make_dataset()
        write_sdtm_adsl(ds, str(tmp_path / "adsl.csv"), study_id="S01")
        df = pd.read_csv(tmp_path / "adsl.csv")
        assert df["USUBJID"].str.startswith("S01").all()

    def test_exported_from_output_init(self):
        from openpkpd.output import write_sdtm_adsl as wsa

        assert callable(wsa)
