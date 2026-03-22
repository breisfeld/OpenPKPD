"""
CDISC PP (Pharmacokinetics Parameters) domain output.

Converts OpenPKPD NCA results to the CDISC SEND/SDTM PP domain format,
which is the standard for reporting non-compartmental PK parameters in
regulatory submissions.

Reference: CDISC SEND Implementation Guide, PP domain specification.
"""

from __future__ import annotations

import pandas as pd

# Mapping from OpenPKPD internal NCA keys → (PARAMCD, PARAM label)
_PARAM_MAP: dict[str, tuple[str, str]] = {
    "c0": ("C0", "Concentration at Time Zero"),
    "cmax": ("CMAX", "Maximum Observed Concentration"),
    "tmax": ("TMAX", "Time of Cmax"),
    "auc_last": ("AUCLST", "AUC from Time Zero to Last"),
    "auc_inf": ("AUCIFO", "AUC from Time Zero to Infinity"),
    "t_half": ("THALF", "Half-Life"),
    "lambda_z": ("LAMZ", "Terminal Elimination Rate Constant"),
    "cl_f": ("CLF", "Apparent Clearance"),
    "vz_f": ("VZF", "Apparent Volume of Distribution"),
    "mrt": ("MRT", "Mean Residence Time"),
}


def to_cdisc_pp(
    nca_results: pd.DataFrame,
    study_id: str = "STUDY1",
    domain: str = "PP",
    usubjid_col: str = "subject_id",
) -> pd.DataFrame:
    """
    Convert a DataFrame of NCA results to CDISC PP domain format.

    Parameters
    ----------
    nca_results:
        DataFrame where each row is one subject's NCA parameters.
        Expected columns include those produced by
        :meth:`~openpkpd.nca.NCAParameters.to_dict`.
    study_id:
        Value for the ``STUDYID`` column (default ``"STUDY1"``).
    domain:
        Value for the ``DOMAIN`` column (default ``"PP"``).
    usubjid_col:
        Column in *nca_results* that contains the subject identifier
        (default ``"subject_id"``).

    Returns
    -------
    pd.DataFrame
        Long-format CDISC PP domain with columns:
        ``STUDYID``, ``USUBJID``, ``DOMAIN``, ``PARAMCD``, ``PARAM``,
        ``AVAL``, ``DTYPE``.
    """
    records: list[dict[str, object]] = []

    for _, row in nca_results.iterrows():
        subj = row.get(usubjid_col, "UNKNOWN")
        for key, (paramcd, param) in _PARAM_MAP.items():
            if key not in row:
                continue
            val = row[key]
            records.append(
                {
                    "STUDYID": study_id,
                    "USUBJID": subj,
                    "DOMAIN": domain,
                    "PARAMCD": paramcd,
                    "PARAM": param,
                    "AVAL": val,
                    "DTYPE": "",
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["STUDYID", "USUBJID", "DOMAIN", "PARAMCD", "PARAM", "AVAL", "DTYPE"]
        )

    return pd.DataFrame(records).reset_index(drop=True)
