"""
OpenPKPD — Notebook 02: Data Handling

Covers:
  - NONMEMDataset: creating, validating, and inspecting datasets
  - EventProcessor: how dosing events are parsed
  - BLQ (Below Limit of Quantification) handling flags
  - Covariate imputation
  - Loading from CSV and from DataFrames
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — Data Handling")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Data Handling in OpenPKPD

        OpenPKPD reads data in the **NONMEM format**: a flat rectangular CSV where
        each row is either a dose event or an observation.  The `NONMEMDataset` class
        validates and pre-processes this data before it reaches the model.

        ## Standard Column Conventions

        | Column | Required | Description |
        |--------|----------|-------------|
        | `ID`   | Yes | Subject/individual identifier (numeric) |
        | `TIME` | Yes | Time (hours, or any consistent unit) |
        | `AMT`  | Yes | Dose amount — `0` on observation rows |
        | `DV`   | Yes | Observed dependent variable (e.g., concentration) |
        | `EVID` | Yes | Event ID: `0` = observation, `1` = dose |
        | `MDV`  | Yes | Missing DV: `1` on dose rows or censored observations |
        | `CMT`  | No  | Compartment number for the dose |
        | `RATE` | No  | Infusion rate (0 = bolus, −1 = duration-specified) |
        | `SS`   | No  | Steady-state flag |
        | `II`   | No  | Interdose interval (for steady-state) |
        | `BLQ`  | No  | Below-limit-of-quantification flag |
        | `LLOQ` | No  | Lower limit of quantification (subject/row-specific) |
        """
    )
    return


@app.cell
def _():
    import io
    import numpy as np
    import pandas as pd
    from openpkpd.data.dataset import NONMEMDataset

    return io, np, pd, NONMEMDataset


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Creating a Dataset from a DataFrame

        The simplest path is `NONMEMDataset.from_dataframe(df)`.  OpenPKPD expects
        the standard NONMEM column names; if your DataFrame already uses them you
        need no extra arguments.
        """
    )
    return


@app.cell
def _(io, pd, NONMEMDataset):
    WARFARIN_CSV = """\
ID,TIME,AMT,DV,EVID,MDV,WT,AGE,SEX
1,0,70,0,1,1,66.7,50,1
1,24,0,0.87,0,0,66.7,50,1
1,36,0,1.21,0,0,66.7,50,1
1,48,0,1.44,0,0,66.7,50,1
1,72,0,1.52,0,0,66.7,50,1
1,96,0,1.19,0,0,66.7,50,1
1,120,0,0.93,0,0,66.7,50,1
2,0,70,0,1,1,78.5,45,1
2,24,0,0.51,0,0,78.5,45,1
2,36,0,0.73,0,0,78.5,45,1
2,48,0,0.88,0,0,78.5,45,1
2,72,0,1.04,0,0,78.5,45,1
2,96,0,0.95,0,0,78.5,45,1
2,120,0,0.77,0,0,78.5,45,1
3,0,70,0,1,1,54.3,62,0
3,24,0,1.18,0,0,54.3,62,0
3,36,0,1.73,0,0,54.3,62,0
3,48,0,2.01,0,0,54.3,62,0
3,72,0,2.11,0,0,54.3,62,0
3,96,0,1.87,0,0,54.3,62,0
3,120,0,1.54,0,0,54.3,62,0
"""

    warf_df = pd.read_csv(io.StringIO(WARFARIN_CSV))
    warf_ds = NONMEMDataset.from_dataframe(warf_df)
    covariate_cols = [
        col
        for col in warf_ds.df.columns
        if col not in {"ID", "TIME", "AMT", "DV", "EVID", "MDV", "CMT", "RATE", "ADDL", "II", "SS"}
    ]

    print(f"Subjects: {warf_ds.n_subjects()}")
    print(f"Observations: {warf_ds.n_observations()}")
    print(f"Covariates: {covariate_cols}")
    warf_df
    return WARFARIN_CSV, covariate_cols, warf_df, warf_ds


@app.cell
def _(covariate_cols, mo, warf_ds):
    mo.md(
        f"""
        ## 2. Dataset Properties

        After loading, inspect the dataset:

        | Property | Value |
        |----------|-------|
        | `.n_subjects()` | `{warf_ds.n_subjects()}` |
        | `.n_observations()` | `{warf_ds.n_observations()}` |
        | covariate columns | `{covariate_cols}` |
        | `.df.columns` | `{list(warf_ds.df.columns)}` |
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Loading from a CSV File

        If you have an actual file on disk, pass the path to
        `NONMEMDataset.from_csv()`:

        ```python
        from openpkpd.data.dataset import NONMEMDataset

        ds = NONMEMDataset.from_csv("path/to/data.csv")
        ```

        OpenPKPD accepts any NONMEM-compatible CSV (including those with
        `ignore=@` comment markers or `IGNORE=(...)` filters).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. BLQ (Below-Limit-of-Quantification) Data

        When observations fall below the assay detection limit they are
        **censored** — we know only that `DV < LLOQ`.  OpenPKPD supports
        multiple BLQ handling strategies via the `BLQMethod` constants:

        | Method | Description |
        |--------|-------------|
        | `M1`   | Discard BLQ observations entirely |
        | `M3`   | Likelihood-based censored regression (Beal 2001) — most rigorous |
        | `M5`   | Replace BLQ with `LLOQ/2` |
        | `M7`   | Replace BLQ with `0` |

        Set the method in `.estimation(blq_method="M3")` or specify
        `LLOQ` column in your dataset for per-row LLOQ values.

        ```python
        built = (
            ModelBuilder()
            ...
            .estimation(method="FOCE", interaction=True, blq_method="M3")
            .build()
        )
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Multiple Dose Events and Infusions

        For **multiple dosing**, add a row for each dose with `EVID=1`.
        For **IV infusions**, set `RATE > 0` (infusion rate in amount/time)
        or `RATE=-1` and provide the infusion duration in the `DUR` column.

        ```
        ID,TIME,AMT,RATE,DV,EVID,MDV
        1,0,500,50,0,1,1        ← 10-hour infusion at 50 mg/hr
        1,12,500,50,0,1,1       ← second dose at 12 h
        1,24,0,0,4.2,0,0        ← observation
        ```

        Steady-state dosing uses `SS=1` and `II` (interdose interval):

        ```
        ID,TIME,AMT,SS,II,DV,EVID,MDV
        1,0,100,1,24,0,1,1     ← steady-state 100 mg q24h
        1,4,0,0,0,3.7,0,0      ← observation 4h post-dose
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. Occasion Variability (IOV)

        For designs with multiple study periods (e.g., crossover), mark
        the occasion with an `OCC` column.  OpenPKPD uses this to build
        a **block-diagonal IOV omega matrix** in the model.

        ```
        ID,TIME,AMT,DV,EVID,MDV,OCC
        1,0,100,0,1,1,1          ← Period 1 dose
        1,2,0,1.4,0,0,1
        1,168,100,0,1,1,2        ← Period 2 dose (one week later)
        1,170,0,1.3,0,0,2
        ```

        Then add IOV variances in your model:

        ```python
        .pk(\"\"\"
        KA = THETA(1)*EXP(ETA(1) + KAPPA(1))   ; KAPPA = IOV eta
        CL = THETA(2)*EXP(ETA(2))
        V  = THETA(3)*EXP(ETA(3))
        \"\"\")
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. Covariate Imputation

        By default, `NONMEMDataset.from_dataframe()` carries covariates forward
        (LOCF — Last Observation Carried Forward) and backward (BOCF) to fill
        any missing values.  The `impute_covariates` parameter controls this:

        ```python
        ds = NONMEMDataset.from_dataframe(
            df,
            impute_covariates=True,   # default: True
            impute_method="locf",     # "locf" | "bocf" | "mean"
        )
        ```

        After loading, inspect `ds.df.columns` and select the covariate columns
        you want to use in modeling or imputation.
        """
    )
    return


@app.cell
def _(covariate_cols, mo, warf_df, warf_ds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs_df = warf_df[warf_df["EVID"] == 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for sid in obs_df["ID"].unique():
        sub = obs_df[obs_df["ID"] == sid]
        axes[0].plot(sub["TIME"], sub["DV"], marker="o", label=f"ID={sid}")

    axes[0].set_xlabel("Time (h)")
    axes[0].set_ylabel("Warfarin concentration (mg/L)")
    axes[0].set_title("Individual Concentration–Time Profiles")
    axes[0].legend()

    axes[1].scatter(obs_df["WT"], obs_df["DV"], c=obs_df["SEX"], cmap="bwr", alpha=0.7)
    axes[1].set_xlabel("Body weight (kg)")
    axes[1].set_ylabel("DV (mg/L)")
    axes[1].set_title("DV vs Body Weight (colour = SEX)")

    fig.tight_layout()
    mo.md(
        f"**Dataset summary:** {warf_ds.n_subjects()} subjects, "
        f"{warf_ds.n_observations()} observations, "
        f"covariates: {covariate_cols}"
    )
    return axes, fig, matplotlib, obs_df, plt, sid, sub


@app.cell
def _(fig):
    fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        | Task | API |
        |------|-----|
        | Load from DataFrame | `NONMEMDataset.from_dataframe(df)` |
        | Load from CSV file | `NONMEMDataset.from_csv("file.csv")` |
        | Check subjects/observations | `.n_subjects()`, `.n_observations()` |
        | Inspect covariates | `ds.df.columns` |
        | BLQ handling | `.estimation(blq_method="M3")` |
        | IOV | `OCC` column + IOV omega block |

        **Next:** `03_estimation_methods.py` — FO, FOCE, SAEM, and Bayesian estimation.
        """
    )
    return


if __name__ == "__main__":
    app.run()
