"""Example 28 — Indometh NCA against WinNonlin-backed reference data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from openpkpd.nca import NCAEngine


DATA_FILE = Path("tests/external_validation/data/indometh.csv")


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    zero_rows = pd.DataFrame(
        {
            "Subject": sorted(df["Subject"].unique()),
            "time": 0.0,
            "conc": 0.0,
        }
    )
    df = pd.concat([zero_rows, df], ignore_index=True).sort_values(
        ["Subject", "time"], kind="stable"
    )

    engine = NCAEngine(auc_method="linear-log", exclude_cmax=True)
    rows: list[dict[str, float]] = []
    for subject_id, group in df.groupby("Subject", sort=True):
        params = engine.compute_subject(
            times=group["time"].to_numpy(float),
            conc=group["conc"].to_numpy(float),
            dose=25.0,
            subject_id=int(subject_id),
            route="oral",
        )
        rows.append(
            {
                "Subject": int(subject_id),
                "Cmax": float(params.cmax),
                "Tmax": float(params.tmax),
                "AUClast": float(params.auc_last),
                "AUCinf": float(params.auc_inf),
                "t_half": float(params.t_half),
                "CL/F": float(params.cl_f),
            }
        )

    result_df = pd.DataFrame(rows)
    print("Example 28: Indometh NCA")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nMean AUCinf: {result_df['AUCinf'].mean():.4f}")
    print(f"Mean t_half: {result_df['t_half'].mean():.4f}")


if __name__ == "__main__":
    main()
