"""
$TABLE output file writer.

Generates NONMEM-style table files (sdtab, patab, cotab, catab, etc.)
with per-observation or per-subject data columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.errors import OutputError


def write_table(
    path: str,
    population_model: Any,
    result: EstimationResult,
    params: ParameterSet,
    columns: list[str],
    noprint: bool = False,
    oneheader: bool = True,
    firstonly: bool = False,
    problem_no: int = 1,
) -> None:
    """
    Write a $TABLE output file.

    Args:
        path:             Output file path.
        population_model: Assembled PopulationModel.
        result:           EstimationResult with post-hoc ETAs.
        params:           Final ParameterSet.
        columns:          List of column names to include.
        noprint:          If True, skip writing (NOPRINT option).
        oneheader:        Write header only once (not per-subject).
        firstonly:        Write only first observation per subject.
        problem_no:       Problem number.
    """
    if noprint:
        return

    try:
        rows: list[dict] = []
        for sid in population_model.subject_ids():
            indiv = population_model.individual_model(sid)
            subj_ev = indiv.subject_events
            eta_i = result.post_hoc_etas.get(sid, np.zeros(params.n_eta()))

            try:
                ipred, obs_mask, f = indiv.evaluate(
                    params.theta,
                    eta_i,
                    params.sigma,
                    trans=population_model.trans,
                )
            except Exception:
                ipred = np.full(len(subj_ev.obs_times), float("nan"))
                f = ipred.copy()
                subj_ev.observation_mask()

            dv = subj_ev.obs_dv
            times = subj_ev.obs_times

            for i in range(len(times)):
                if firstonly and i > 0:
                    break
                row: dict[str, float] = {
                    "ID": float(sid),
                    "TIME": float(times[i]),
                    "DV": float(dv[i]) if not np.isnan(dv[i]) else float("nan"),
                    "IPRED": float(ipred[i]) if i < len(ipred) else float("nan"),
                    "PRED": float(f[i]) if i < len(f) else float("nan"),
                    "MDV": float(subj_ev.obs_mdv[i]),
                }
                # ETAs
                for k, eta_val in enumerate(eta_i):
                    row[f"ETA{k + 1}"] = float(eta_val)

                # IWRES (simple version)
                sigma_diag = float(params.sigma[0, 0]) if params.sigma.size > 0 else 1.0
                if not np.isnan(dv[i]) and i < len(ipred) and sigma_diag > 0:
                    row["IWRES"] = (float(dv[i]) - float(ipred[i])) / np.sqrt(sigma_diag)
                    row["RES"] = float(dv[i]) - float(f[i])
                    row["IRES"] = float(dv[i]) - float(ipred[i])
                    row["WRES"] = row["RES"] / np.sqrt(sigma_diag)
                else:
                    row["IWRES"] = float("nan")
                    row["RES"] = float("nan")
                    row["IRES"] = float("nan")
                    row["WRES"] = float("nan")

                rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows)
        # Select only requested columns (if available)
        output_cols = [c for c in columns if c in df.columns]
        if not output_cols:
            output_cols = list(df.columns)
        df = df[output_cols]

        with open(path, "w") as fh:
            if not noprint:
                if oneheader:
                    fh.write(f"TABLE NO.     {problem_no}\n")
                fh.write(df.to_csv(index=False, sep=" ", float_format="%.6E"))

    except OSError as exc:
        raise OutputError(f"Failed to write table file {path!r}: {exc}") from exc
