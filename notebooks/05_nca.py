"""
OpenPKPD — Notebook 05: Non-Compartmental Analysis (NCA)

Covers:
  - NCAEngine: computing PK parameters from concentration-time data
  - NCAParameters: the result dataclass and all computed metrics
  - Urine NCA (UrineNCAEngine)
  - Bioequivalence: average BE (ABE) and reference-scaled ABE (RSABE)
  - Crossover study analysis
  - NCA plots: profile, boxplot, dose proportionality
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — NCA")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Non-Compartmental Analysis (NCA)

        NCA derives pharmacokinetic parameters directly from observed
        concentration-time data without assuming a compartment model.
        It uses the **linear-up/log-down trapezoidal rule** for AUC and
        log-linear regression over the terminal phase for $\lambda_z$ and $t_{1/2}$.

        ## Parameters Computed

        | Parameter | Symbol | Description |
        |-----------|--------|-------------|
        | `cmax` | $C_{\max}$ | Peak concentration |
        | `tmax` | $t_{\max}$ | Time of peak |
        | `auc_last` | $\text{AUC}_{0-t}$ | AUC to last measurable concentration |
        | `auc_inf` | $\text{AUC}_{0-\infty}$ | AUC extrapolated to infinity |
        | `aumc_last` | $\text{AUMC}_{0-t}$ | Area under first moment curve |
        | `half_life` | $t_{1/2}$ | Elimination half-life |
        | `lambda_z` | $\lambda_z$ | Terminal elimination rate constant |
        | `cl_obs` | $CL/F$ | Apparent oral clearance |
        | `vd_obs` | $V_d/F$ | Apparent volume of distribution |
        | `mrt` | $\text{MRT}$ | Mean residence time |
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from openpkpd.nca import NCAEngine, NCAParameters

    return np, pd, NCAEngine, NCAParameters


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Basic NCA on a Single Subject

        Provide arrays of time points and concentrations.  The dose is required
        to compute clearance and volume.
        """
    )
    return


@app.cell
def _(NCAEngine, np):
    # Simulated 1-compartment PK: C(t) = (D/V) * KA/(KA-CL/V) * (exp(-CL/V*t) - exp(-KA*t))
    times = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24], dtype=float)

    ka, cl, v, dose = 1.2, 3.5, 40.0, 100.0
    conc = (dose / v) * (ka / (ka - cl / v)) * (np.exp(-cl / v * times) - np.exp(-ka * times))
    conc[0] = 0.0  # pre-dose = 0

    engine = NCAEngine()
    params = engine.compute_subject(
        times=times,
        conc=conc,
        dose=dose,
        route="oral",
        subject_id="demo-1",
    )
    print(params.summary())
    return cl, conc, dose, engine, ka, params, times, v


@app.cell
def _(mo, params):
    import pandas as _pd

    param_df = _pd.DataFrame(
        {
            "Parameter": [
                "Cmax (ng/mL)",
                "Tmax (h)",
                "AUC_last (ng·h/mL)",
                "AUC_inf (ng·h/mL)",
                "t½ (h)",
                "λz (h⁻¹)",
                "CL/F (L/h)",
                "Vd/F (L)",
                "MRT (h)",
            ],
            "Value": [
                params.cmax,
                params.tmax,
                params.auc_last,
                params.auc_inf,
                params.t_half,
                params.lambda_z,
                params.cl_f,
                params.vz_f,
                params.mrt,
            ],
        }
    ).round(4)

    mo.vstack(
        [
            mo.md("### NCA Parameters — Single Subject"),
            mo.ui.table(param_df),
        ]
    )
    return param_df, _pd


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. NCA on a Full Dataset (Multiple Subjects)

        Pass a `DataFrame` with `ID`, `TIME`, and `DV` columns (plus optionally
        `DOSE`).  `NCAEngine.compute_population()` returns a per-subject
        summary DataFrame.
        """
    )
    return


@app.cell
def _(NCAEngine, np, pd):
    rng = np.random.default_rng(42)
    _times = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24], dtype=float)

    rows = []
    for _sid in range(1, 9):
        _ka = rng.lognormal(np.log(1.2), 0.3)
        _cl = rng.lognormal(np.log(3.5), 0.25)
        _v = rng.lognormal(np.log(40), 0.2)
        _dose = 100.0
        _c = (
            (_dose / _v)
            * (_ka / (_ka - _cl / _v))
            * (np.exp(-_cl / _v * _times) - np.exp(-_ka * _times))
        )
        _c = np.maximum(_c + rng.normal(0, 0.05 * _c), 0)
        _c[0] = 0.0
        rows.append({"ID": _sid, "TIME": 0.0, "DV": 0.0, "AMT": _dose, "EVID": 1, "MDV": 1})
        for t, c in zip(_times, _c):
            rows.append({"ID": _sid, "TIME": t, "DV": c, "AMT": 0.0, "EVID": 0, "MDV": 0})

    pop_df = pd.DataFrame(rows)

    engine_pop = NCAEngine()
    nca_summary = engine_pop.compute_dataset(
        df=pop_df,
        id_col="ID",
        time_col="TIME",
        conc_col="DV",
        dose_col="AMT",
        dose_row_col="EVID",
        route="oral",
    )
    nca_summary.round(3)
    return engine_pop, nca_summary, pop_df, rows, rng


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openpkpd.plots.nca import nca_profile_plot, nca_boxplot, nca_distributions

    return matplotlib, nca_boxplot, nca_distributions, nca_profile_plot, plt


@app.cell
def _(conc, nca_profile_plot, params, times):
    fig_profile = nca_profile_plot(
        times,
        conc,
        params,
        log_y=True,
        title="Individual PK Profiles (semi-log)",
    )
    fig_profile
    return (fig_profile,)


@app.cell
def _(nca_boxplot, nca_summary):
    fig_box = nca_boxplot(
        nca_summary,
        params=["cmax", "auc_last", "half_life"],
        title="NCA Summary — Boxplots",
    )
    fig_box
    return (fig_box,)


@app.cell
def _(nca_distributions, nca_summary):
    fig_dist = nca_distributions(
        nca_summary,
        params=["cmax", "auc_last", "half_life"],
        title="NCA Parameter Distributions",
    )
    fig_dist
    return (fig_dist,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Dose Proportionality

        For multi-dose studies, assess whether AUC and Cmax scale linearly
        with dose using a power model:

        $$\ln(\text{AUC}) = \alpha + \beta \cdot \ln(\text{Dose})$$

        $\beta \approx 1$ indicates dose proportionality.
        """
    )
    return


@app.cell
def _(NCAEngine, np, pd):
    _rng = np.random.default_rng(0)
    _doses = [50, 100, 200, 400]
    _dp_rows = []

    for _dose in _doses:
        for _sid in range(1, 7):
            _t = np.array([0, 0.5, 1, 2, 4, 8, 12, 24], dtype=float)
            _ka = _rng.lognormal(np.log(1.2), 0.2)
            _cl = _rng.lognormal(np.log(3.5), 0.2)
            _v = _rng.lognormal(np.log(40), 0.15)
        _c = (
            (_dose / _v)
            * (_ka / (_ka - _cl / _v))
            * (np.exp(-_cl / _v * _t) - np.exp(-_ka * _t))
        )
        _c[0] = 0.0
        _dp_rows.append(
            {"ID": f"D{_dose}_S{_sid}", "TIME": 0.0, "DV": 0.0, "AMT": float(_dose), "EVID": 1, "MDV": 1}
        )
        for _time, _conc in zip(_t, _c):
            _dp_rows.append(
                {
                    "ID": f"D{_dose}_S{_sid}",
                    "TIME": _time,
                    "DV": max(_conc, 0),
                    "AMT": 0.0,
                    "EVID": 0,
                    "MDV": 0,
                }
            )

    dp_df = pd.DataFrame(_dp_rows)

    _engine_dp = NCAEngine()
    dp_nca = _engine_dp.compute_dataset(
        df=dp_df,
        id_col="ID",
        time_col="TIME",
        conc_col="DV",
        dose_col="AMT",
        dose_row_col="EVID",
        route="oral",
    )
    dp_nca.head()
    return dp_df, dp_nca


@app.cell
def _():
    from openpkpd.plots.nca import dose_proportionality_plot

    return (dose_proportionality_plot,)


@app.cell
def _(dose_proportionality_plot, dp_nca):
    fig_dp = dose_proportionality_plot(
        dp_nca,
        dose_col="dose",
        metric="auc_last",
        title="Dose Proportionality — AUC₀₋ₜ vs Dose",
    )
    fig_dp
    return (fig_dp,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Bioequivalence (BE)

        ### Average Bioequivalence (ABE)

        ABE tests whether the 90% CI of the geometric mean ratio
        (Test/Reference) falls within [80%, 125%] for Cmax and AUC.

        ```python
        from openpkpd.nca import average_bioequivalence, BEResult

        auc_be = average_bioequivalence(test_auc, ref_auc, metric="AUC")
        cmax_be = average_bioequivalence(test_cmax, ref_cmax, metric="Cmax")
        print(auc_be.summary())
        print(cmax_be.summary())
        ```

        ### Reference-Scaled ABE (RSABE)

        For highly variable drugs (HVD), FDA guidance allows wider limits
        scaled to the intra-subject variability of the reference:

        ```python
        from openpkpd.nca import reference_scaled_abe

        rsabe = reference_scaled_abe(
            test_values=[..],
            ref_values=[..],
            sequence=["TR", "RT", ...],
            period=[1, 2, ...],
        )
        ```

        ### Crossover Study Analysis

        ```python
        from openpkpd.nca import crossover_be_analysis, be_sample_size

        result = crossover_be_analysis(df, metric_col="log_metric", treatment_col="TRT")
        n = be_sample_size(cv=0.25, true_ratio=1.0, alpha=0.05, power=0.80)
        print(f"Required N per sequence: {n}")
        ```
        """
    )
    return


@app.cell
def _(np):
    from openpkpd.nca import average_bioequivalence, be_sample_size

    rng_be = np.random.default_rng(0)
    n_sub = 24
    ref_auc = rng_be.lognormal(np.log(80), 0.2, n_sub)
    test_auc = ref_auc * rng_be.lognormal(np.log(1.02), 0.15, n_sub)

    ref_cmax = rng_be.lognormal(np.log(4.5), 0.2, n_sub)
    test_cmax = ref_cmax * rng_be.lognormal(np.log(1.01), 0.15, n_sub)

    auc_be_result = average_bioequivalence(test_auc, ref_auc, metric="AUC")
    cmax_be_result = average_bioequivalence(test_cmax, ref_cmax, metric="Cmax")
    print(auc_be_result.summary())
    print()
    print(cmax_be_result.summary())

    n_required = be_sample_size(cv=0.25, true_ratio=1.0)
    print(f"\nRequired N per sequence (CV=25%): {n_required}")
    return (
        average_bioequivalence,
        auc_be_result,
        be_sample_size,
        cmax_be_result,
        n_required,
        n_sub,
        ref_auc,
        ref_cmax,
        rng_be,
        test_auc,
        test_cmax,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Urine NCA

        For renal elimination studies, `UrineNCAEngine` computes renal
        clearance and amount excreted:

        ```python
        from openpkpd.nca import UrineNCAEngine

        urine_engine = UrineNCAEngine()
        urine_params = urine_engine.compute(
            time_mid=times_mid,   # midpoint of collection interval
            amount_excreted=ae,   # cumulative amount in urine
            dose=dose,
            plasma_auc=params.auc_inf,
        )
        print(urine_params.clr)   # renal clearance
        print(urine_params.fe)    # fraction excreted unchanged
        ```

        ## Summary

        | Task | API |
        |------|-----|
        | Single-subject NCA | `NCAEngine().compute_subject(times, conc, dose, route)` |
        | Population NCA | `NCAEngine().compute_dataset(df, ...)` |
        | Average BE | `average_bioequivalence(test, reference, metric=...)` |
        | RSABE | `reference_scaled_abe(test, ref, sequence, period)` |
        | Sample size | `be_sample_size(cv, true_ratio)` |
        | Crossover | `crossover_be_analysis(df, metric_col, treatment_col)` |
        | Urine NCA | `UrineNCAEngine().compute(time_mid, amount_excreted, ...)` |
        """
    )
    return


if __name__ == "__main__":
    app.run()
