"""
OpenPKPD — Notebook 01: Quickstart

Covers:
  - ModelBuilder fluent API
  - Theophylline 1-compartment oral model (FO)
  - Inspecting estimation results
  - Concentration-time and spaghetti plots
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — Quickstart")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # OpenPKPD — Quickstart

        **OpenPKPD** is an open-source Python toolkit for population PK/PD analyses,
        modelled on the NONMEM control stream workflow.

        This notebook walks through the core workflow:

        1. Embed or load pharmacokinetic data
        2. Define a model with the `ModelBuilder` fluent API
        3. Fit the model (FO estimation)
        4. Inspect parameter estimates and model summary
        5. Visualise concentration-time profiles

        ## Installation

        ```bash
        pip install openpkpd[plots]
        # or, in the project directory:
        uv sync --extra plots --extra dev
        ```
        """
    )
    return


@app.cell
def _():
    import io
    import numpy as np
    import pandas as pd
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    return io, np, pd, NONMEMDataset, ModelBuilder


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. The Dataset

        We embed the classic **theophylline** dataset (Boeckmann et al., 1994) directly
        in the notebook.  Each row is either a dose event (`EVID=1`) or an observation
        (`EVID=0`).  The `MDV` flag marks rows that are *missing dependent variable*
        (i.e. dose rows where DV is not observed).

        | Column | Meaning |
        |--------|---------|
        | `ID`   | Subject identifier |
        | `TIME` | Hours post-first-dose |
        | `AMT`  | Dose amount (mg) — non-zero on dose rows |
        | `DV`   | Measured plasma concentration (mg/L) |
        | `EVID` | Event ID: 0 = observation, 1 = dose |
        | `MDV`  | Missing DV flag: 1 on dose rows |
        | `WT`   | Body weight (kg) |
        """
    )
    return


@app.cell
def _(io, pd):
    THEO_CSV = """\
ID,TIME,AMT,DV,EVID,MDV,WT
1,0,4.02,0,1,1,79.6
1,0.27,0,0.74,0,0,79.6
1,0.57,0,1.72,0,0,79.6
1,1.02,0,7.91,0,0,79.6
1,1.92,0,8.31,0,0,79.6
1,3.5,0,8.33,0,0,79.6
1,5.02,0,6.85,0,0,79.6
1,7.03,0,6.08,0,0,79.6
1,9.0,0,5.4,0,0,79.6
1,12.05,0,4.55,0,0,79.6
1,24.37,0,1.25,0,0,79.6
2,0,4.4,0,1,1,72.4
2,0.35,0,0.96,0,0,72.4
2,0.6,0,2.33,0,0,72.4
2,1.07,0,4.71,0,0,72.4
2,2.13,0,8.33,0,0,72.4
2,3.5,0,9.02,0,0,72.4
2,5.02,0,7.14,0,0,72.4
2,7.02,0,5.68,0,0,72.4
2,9.1,0,4.55,0,0,72.4
2,12.1,0,3.01,0,0,72.4
2,25.0,0,0.9,0,0,72.4
3,0,4.95,0,1,1,70.5
3,0.27,0,0.64,0,0,70.5
3,0.58,0,1.92,0,0,70.5
3,1.02,0,4.44,0,0,70.5
3,1.92,0,7.03,0,0,70.5
3,3.5,0,9.07,0,0,70.5
3,5.02,0,7.56,0,0,70.5
3,7.02,0,6.59,0,0,70.5
3,9.0,0,5.88,0,0,70.5
3,12.15,0,4.73,0,0,70.5
3,24.17,0,1.25,0,0,70.5
4,0,4.53,0,1,1,72.7
4,0.3,0,1.03,0,0,72.7
4,0.52,0,2.02,0,0,72.7
4,1.0,0,5.63,0,0,72.7
4,1.92,0,8.6,0,0,72.7
4,3.5,0,8.38,0,0,72.7
4,5.02,0,7.54,0,0,72.7
4,7.07,0,6.88,0,0,72.7
4,9.0,0,5.78,0,0,72.7
4,12.12,0,3.99,0,0,72.7
4,24.08,0,1.17,0,0,72.7
5,0,5.68,0,1,1,54.6
5,0.27,0,2.24,0,0,54.6
5,0.58,0,4.57,0,0,54.6
5,1.02,0,9.29,0,0,54.6
5,2.02,0,9.66,0,0,54.6
5,3.62,0,8.58,0,0,54.6
5,5.08,0,8.36,0,0,54.6
5,7.07,0,7.47,0,0,54.6
5,9.0,0,6.89,0,0,54.6
5,12.15,0,5.94,0,0,54.6
5,24.17,0,3.28,0,0,54.6
6,0,4.0,0,1,1,80.0
6,0.35,0,0.4,0,0,80.0
6,0.6,0,1.15,0,0,80.0
6,1.07,0,4.0,0,0,80.0
6,2.13,0,6.6,0,0,80.0
6,3.5,0,7.99,0,0,80.0
6,5.02,0,7.31,0,0,80.0
6,7.02,0,6.72,0,0,80.0
6,9.1,0,5.76,0,0,80.0
6,12.1,0,4.06,0,0,80.0
6,25.0,0,1.25,0,0,80.0
"""

    df = pd.read_csv(io.StringIO(THEO_CSV))
    ds = NONMEMDataset.from_dataframe(df)
    df
    return THEO_CSV, df, ds


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Build the Model

        The `ModelBuilder` uses a **fluent (method-chaining) API** modelled on the
        NONMEM control stream:

        | Builder method | Equivalent `$RECORD` | Purpose |
        |----------------|----------------------|---------|
        | `.problem()`   | `$PROBLEM`           | Model title |
        | `.dataset()`   | `$DATA`              | Attach a `NONMEMDataset` |
        | `.subroutines()` | `$SUBROUTINES`     | Choose ADVAN/TRANS |
        | `.pk()`        | `$PK`                | PK parameter equations |
        | `.error()`     | `$ERROR`             | Residual error model |
        | `.theta()`     | `$THETA`             | Fixed-effect initial estimates |
        | `.omega()`     | `$OMEGA`             | IIV variance initial estimates |
        | `.sigma()`     | `$SIGMA`             | Residual variance initial estimates |
        | `.estimation()` | `$ESTIMATION`       | Estimation method and options |
        | `.covariance()` | `$COVARIANCE`       | Request standard errors |

        ### PK model: ADVAN2 TRANS2 (1-compartment oral, absorption)

        ADVAN2 (one-compartment with first-order absorption) parameterised via TRANS2
        (CL/V/KA):

        $$
        \\frac{dA_{gut}}{dt} = -K_A \\cdot A_{gut},
        \\quad
        \\frac{dA_{central}}{dt} = K_A \\cdot A_{gut} - \\frac{CL}{V} \\cdot A_{central}
        $$

        $$
        C(t) = \\frac{A_{central}(t)}{V}
        $$

        Individual parameters include log-normal IIV on KA, CL, and V.
        """
    )
    return


@app.cell
def _(ModelBuilder, ds):
    built = (
        ModelBuilder()
        .problem("Theophylline 1-cmt oral — FO")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk("""
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
""")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)])
        .omega([0.5, 0.3, 0.3])
        .sigma(0.1)
        .estimation(method="FO", maxeval=500)
        .build()
    )
    built
    return (built,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Fit the Model

        Calling `.fit()` runs the selected estimation algorithm (here FO — First-Order)
        and returns an `EstimationResult`.

        > **Tip:** For better accuracy use `method="FOCE"` with `interaction=True`.
        > FO is faster and useful for initial exploration.
        """
    )
    return


@app.cell
def _(built):
    result = built.fit()
    result
    return (result,)


@app.cell
def _(mo, result):
    mo.md(
        f"""
        ## 4. Results

        ```
        {result.summary()}
        ```
        """
    )
    return


@app.cell
def _(mo, pd, result):
    theta_df = pd.DataFrame(
        {
            "Parameter": ["KA (hr⁻¹)", "CL (L/hr)", "V (L)"],
            "Estimate": result.theta_final,
            "SE": result.theta_se if result.theta_se is not None else [None] * 3,
        }
    )

    omega_vals = result.omega_final.diagonal() if result.omega_final is not None else []
    omega_df = pd.DataFrame(
        {
            "Parameter": ["ω²(KA)", "ω²(CL)", "ω²(V)"],
            "Estimate": list(omega_vals),
        }
    )

    mo.vstack(
        [
            mo.md("### Fixed Effects (THETA)"),
            mo.ui.table(theta_df),
            mo.md("### Random Effects — IIV variances (OMEGA diagonal)"),
            mo.ui.table(omega_df),
            mo.md(f"**OFV:** `{result.ofv:.4f}`  |  **Converged:** `{result.converged}`"),
        ]
    )
    return omega_df, omega_vals, theta_df


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Diagnostic Plots

        `compute_diagnostics` assembles a tidy DataFrame with population/individual
        predictions and residuals. Standard plots are then generated from that frame.
        """
    )
    return


@app.cell
def _(built, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openpkpd.plots.diagnostics import compute_diagnostics
    from openpkpd.plots.pk import concentration_time, spaghetti_plot

    diag_df = compute_diagnostics(built.population_model, result)
    return diag_df, concentration_time, matplotlib, plt, spaghetti_plot


@app.cell
def _(concentration_time, diag_df):
    fig_ct = concentration_time(
        diag_df,
        log_y=False,
        title="Theophylline — Population and Individual Predictions",
    )
    fig_ct
    return (fig_ct,)


@app.cell
def _(diag_df, spaghetti_plot):
    fig_sp = spaghetti_plot(diag_df, title="Theophylline — Spaghetti Plot")
    fig_sp
    return (fig_sp,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        | Notebook | Topic |
        |----------|-------|
        | `02_data_handling.py` | Loading, validating, and exploring NONMEM datasets |
        | `03_estimation_methods.py` | FO, FOCE, SAEM, IMP, Bayesian, nonparametric |
        | `04_simulation_vpc_npde.py` | Simulation, VPC, NPDE |
        | `05_nca.py` | Non-compartmental analysis and bioequivalence |
        | `08_diagnostics_plots.py` | Full GOF diagnostic panel |
        """
    )
    return


if __name__ == "__main__":
    app.run()
