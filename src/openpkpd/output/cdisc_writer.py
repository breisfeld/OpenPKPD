"""
CDISC output writers: ADPPK (ADaM-style), SDTM PC, and SDTM ADSL.

All writers produce CSV files.  Scope is intentionally limited: CSV
format only (no SAS XPT transport file, no full SDTM validator).
Suitable for downstream processing and as a standard exchange format.

Functions
---------
write_cdisc_adppk   — ADPPK-style summary (parameters + observations)
write_sdtm_pc       — SDTM PC domain (one row per PK concentration record)
write_sdtm_adsl     — SDTM ADSL domain (one row per subject, demographics)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from openpkpd.data.dataset import NONMEMDataset
    from openpkpd.estimation.base import EstimationResult


def write_cdisc_adppk(
    result: EstimationResult,
    dataset: NONMEMDataset,
    path: str,
    study_id: str = "STUDY1",
    avalu: str = "ng/mL",
) -> None:
    """
    Write a CDISC ADPPK-style CSV file from estimation results.

    Parameters
    ----------
    result:
        Completed estimation result (theta, omega, sigma, post-hoc ETAs).
    dataset:
        The NONMEM dataset used during estimation (provides DV and IDs).
    path:
        Output file path (will be created or overwritten).
    study_id:
        Value for the ``STUDYID`` column.
    avalu:
        Concentration unit string for observation rows.
    """
    rows: list[dict[str, object]] = []

    # ── Observation rows ──────────────────────────────────────────────────
    obs_df = dataset.observation_rows()
    for _, rec in obs_df.iterrows():
        rows.append(
            {
                "STUDYID": study_id,
                "USUBJID": rec["ID"],
                "PARAMCD": "CONC",
                "PARAM": "Observed Concentration",
                "AVAL": rec["DV"],
                "AVALU": avalu,
                "DTYPE": "OBSERVATION",
            }
        )

    # ── Population fixed effects (THETAs) ─────────────────────────────────
    for i, val in enumerate(result.theta_final, start=1):
        rows.append(
            {
                "STUDYID": study_id,
                "USUBJID": "",
                "PARAMCD": f"THETA{i}",
                "PARAM": f"Fixed Effect THETA{i}",
                "AVAL": float(val),
                "AVALU": "",
                "DTYPE": "THETA",
            }
        )

    # ── OMEGA matrix ──────────────────────────────────────────────────────
    n_eta = result.omega_final.shape[0]
    for i in range(n_eta):
        for j in range(i + 1):
            rows.append(
                {
                    "STUDYID": study_id,
                    "USUBJID": "",
                    "PARAMCD": f"OMEGA({i + 1},{j + 1})",
                    "PARAM": f"Random Effect Variance OMEGA({i + 1},{j + 1})",
                    "AVAL": float(result.omega_final[i, j]),
                    "AVALU": "",
                    "DTYPE": "OMEGA",
                }
            )

    # ── SIGMA matrix ──────────────────────────────────────────────────────
    n_eps = result.sigma_final.shape[0]
    for i in range(n_eps):
        for j in range(i + 1):
            rows.append(
                {
                    "STUDYID": study_id,
                    "USUBJID": "",
                    "PARAMCD": f"SIGMA({i + 1},{j + 1})",
                    "PARAM": f"Residual Variance SIGMA({i + 1},{j + 1})",
                    "AVAL": float(result.sigma_final[i, j]),
                    "AVALU": "",
                    "DTYPE": "SIGMA",
                }
            )

    # ── Post-hoc ETAs ─────────────────────────────────────────────────────
    for subj_id, eta_vec in result.post_hoc_etas.items():
        for k, eta_val in enumerate(eta_vec, start=1):
            rows.append(
                {
                    "STUDYID": study_id,
                    "USUBJID": subj_id,
                    "PARAMCD": f"ETA{k}",
                    "PARAM": f"Empirical Bayes Estimate ETA{k}",
                    "AVAL": float(eta_val),
                    "AVALU": "",
                    "DTYPE": "ETA",
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


# ── SDTM PC domain ────────────────────────────────────────────────────────────


def write_sdtm_pc(
    dataset: NONMEMDataset,
    path: str,
    study_id: str = "STUDY1",
    test_code: str = "CONC",
    test_name: str = "Analyte Concentration",
    unit: str = "ng/mL",
) -> None:
    """
    Write a SDTM PC domain CSV from a NONMEM dataset.

    One row is written per observation record (EVID=0, MDV=0).  Columns
    follow the CDISC SDTM PC domain specification (CDISC SDTM v1.8):
    STUDYID, DOMAIN, USUBJID, PCSEQ, PCTESTCD, PCTEST, PCORRES,
    PCORRESU, PCSTRESC, PCSTRESN, PCSTRESU, NFRLT.

    Args:
        dataset:   NONMEMDataset whose observation rows will be written.
        path:      Output CSV file path.
        study_id:  Value for the STUDYID column.
        test_code: Short test code string for PCTESTCD (≤8 chars).
        test_name: Full test name for PCTEST.
        unit:      Concentration unit written to PCORRESU and PCSTRESU.
    """
    obs_df = dataset.observation_rows()
    rows: list[dict[str, object]] = []
    seq_counters: dict[object, int] = {}

    for _, rec in obs_df.iterrows():
        subj = rec["ID"]
        seq_counters[subj] = seq_counters.get(subj, 0) + 1
        dv_val = rec["DV"]
        dv_str = "" if pd.isna(dv_val) else str(round(float(dv_val), 6))

        rows.append(
            {
                "STUDYID": study_id,
                "DOMAIN": "PC",
                "USUBJID": subj,
                "PCSEQ": seq_counters[subj],
                "PCTESTCD": test_code[:8],
                "PCTEST": test_name,
                "PCORRES": dv_str,
                "PCORRESU": unit,
                "PCSTRESC": dv_str,
                "PCSTRESN": float(dv_val) if not pd.isna(dv_val) else "",
                "PCSTRESU": unit,
                "NFRLT": float(rec["TIME"]),
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


# ── SDTM ADSL domain ──────────────────────────────────────────────────────────


def write_sdtm_adsl(
    dataset: NONMEMDataset,
    path: str,
    study_id: str = "STUDY1",
    demographic_columns: list[str] | None = None,
) -> None:
    """
    Write a SDTM ADSL (Subject-Level Analysis Dataset) CSV.

    One row is written per unique subject.  Demographic covariates
    (AGE, SEX, RACE, etc.) are extracted from the first record of each
    subject in the dataset when the corresponding columns are present.

    Standard ADSL columns always written:
    STUDYID, USUBJID, SUBJID, SAFFL, ITTFL.

    Additional demographic columns (from ``demographic_columns`` or the
    auto-detected set ``{AGE, SEX, RACE, COUNTRY, WT, HT, BMI}``) are
    appended when present in the dataset.

    Args:
        dataset:              NONMEMDataset from which subjects are extracted.
        path:                 Output CSV file path.
        study_id:             Value for the STUDYID column.
        demographic_columns:  Explicit list of dataset columns to include as
                              demographics.  When ``None``, common demographic
                              column names are detected automatically.
    """
    _AUTO_DEMO_COLS = {"AGE", "SEX", "RACE", "COUNTRY", "WT", "HT", "BMI"}
    df = dataset.df

    if demographic_columns is None:
        demo_cols = sorted(_AUTO_DEMO_COLS & set(df.columns))
    else:
        demo_cols = [c for c in demographic_columns if c in df.columns]

    rows: list[dict[str, object]] = []
    for subj_id in dataset.subject_ids():
        subj_df = dataset.subject_data(subj_id)
        first_row = subj_df.iloc[0]

        record: dict[str, object] = {
            "STUDYID": study_id,
            "USUBJID": f"{study_id}-{subj_id:04d}",
            "SUBJID": str(subj_id),
            "SAFFL": "Y",
            "ITTFL": "Y",
        }
        for col in demo_cols:
            val = first_row.get(col, "")
            record[col] = "" if pd.isna(val) else val  # type: ignore[arg-type]

        rows.append(record)

    pd.DataFrame(rows).to_csv(path, index=False)
