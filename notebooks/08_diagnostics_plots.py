"""
OpenPKPD — Notebook 08: Diagnostics and Plots

Covers:
  - compute_diagnostics: building the GOF DataFrame
  - GOF panel: DV vs PRED/IPRED, CWRES vs TIME/PRED, Q-Q
  - ETA diagnostics: histograms, pairs, shrinkage, covariate plots
  - OFV history, parameter uncertainty
  - PK plots: spaghetti, mean_profile, individual_fit_grid
  - PD plots: effect_time, emax_curve, hysteresis_loop
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — Diagnostics & Plots")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Diagnostic Plots

        OpenPKPD's `openpkpd.plots` package provides a comprehensive set of
        standard PK/PD diagnostic and exploratory plots built on matplotlib.

        ## Workflow

        ```
        EstimationResult
              │
              ▼
        compute_diagnostics(population_model, result)
              │         → diag_df: DV, PRED, IPRED, CWRES, IWRES, ETA, ...
              │
        ├── GOF plots (dv_vs_ipred, cwres_vs_time, ...)
        ├── ETA plots (eta_histograms, eta_pairs, eta_vs_covariate)
        └── PK plots  (spaghetti_plot, individual_fit_grid)
        ```
        """
    )
    return


@app.cell
def _():
    import io
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    return io, np, pd, matplotlib, plt, ModelBuilder, NONMEMDataset


@app.cell
def _(io, pd, NONMEMDataset, ModelBuilder):
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

    ds = NONMEMDataset.from_dataframe(pd.read_csv(io.StringIO(THEO_CSV)))

    built = (
        ModelBuilder()
        .problem("Theophylline FOCEI for diagnostics")
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
        .estimation(method="FOCE", interaction=True, maxeval=600)
        .covariance()
        .build()
    )

    result = built.fit()
    print(result.summary())
    return THEO_CSV, built, ds, result


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. The Diagnostic DataFrame

        `compute_diagnostics` evaluates the fitted model at each observation
        time and assembles a tidy DataFrame with all the information needed
        for GOF plots:

        | Column | Description |
        |--------|-------------|
        | `ID`, `TIME` | Subject identifier and time |
        | `DV` | Observed dependent variable |
        | `PRED` | Population prediction (η=0) |
        | `IPRED` | Individual prediction (at η̂ᵢ) |
        | `RES` | Residual: DV − PRED |
        | `WRES` | Weighted residual |
        | `CWRES` | Conditional weighted residual |
        | `IWRES` | Individual weighted residual |
        | `ETA_1..n` | Empirical Bayes estimates of η |
        """
    )
    return


@app.cell
def _(built, result):
    from openpkpd.plots.diagnostics import compute_diagnostics

    diag_df = compute_diagnostics(built.population_model, result)
    diag_df.head(10)
    return compute_diagnostics, diag_df


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Full GOF Diagnostic Panel

        `diagnostic_panel` generates the standard 4 (or 6)-panel GOF figure
        used in population PK/PD reports:

        1. DV vs PRED (population predictions)
        2. DV vs IPRED (individual predictions)
        3. CWRES vs TIME
        4. CWRES vs PRED
        5. CWRES Q-Q plot
        6. |IWRES| vs IPRED
        """
    )
    return


@app.cell
def _(diag_df):
    from openpkpd.plots.gof import diagnostic_panel

    fig_gof = diagnostic_panel(diag_df, title="Theophylline GOF — FOCE+I")
    fig_gof
    return diagnostic_panel, fig_gof


@app.cell
def _(mo):
    mo.md("## 3. Individual GOF Plots")
    return


@app.cell
def _(diag_df):
    from openpkpd.plots.gof import (
        dv_vs_ipred,
        dv_vs_pred,
        cwres_vs_time,
        cwres_vs_pred,
        cwres_qq,
        cwres_histogram,
        abs_iwres_vs_ipred,
    )

    fig_dv_ipred = dv_vs_ipred(diag_df, title="DV vs IPRED")
    fig_dv_ipred
    return (
        abs_iwres_vs_ipred,
        cwres_histogram,
        cwres_qq,
        cwres_vs_pred,
        cwres_vs_time,
        dv_vs_ipred,
        dv_vs_pred,
        fig_dv_ipred,
    )


@app.cell
def _(cwres_vs_time, diag_df):
    fig_cwres_t = cwres_vs_time(diag_df, title="CWRES vs TIME")
    fig_cwres_t
    return (fig_cwres_t,)


@app.cell
def _(cwres_qq, diag_df):
    fig_qq = cwres_qq(diag_df, title="CWRES Normal Q-Q")
    fig_qq
    return (fig_qq,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. ETA (IIV) Diagnostics

        Plots of the empirical Bayes estimates (EBEs) help diagnose:
        - Shrinkage: are ETAs dominated by the prior?
        - Non-normality: asymmetry or multi-modality in ETA distributions
        - Covariate relationships: can a covariate explain IIV?
        """
    )
    return


@app.cell
def _(diag_df, result):
    from openpkpd.plots.eta import (
        eta_histograms,
        eta_pairs,
        eta_shrinkage_plot,
        iiv_cv_plot,
        omega_heatmap,
    )

    fig_eta_hist = eta_histograms(diag_df, result.omega_final, title="ETA Distributions")
    fig_eta_hist
    return (
        eta_histograms,
        eta_pairs,
        eta_shrinkage_plot,
        fig_eta_hist,
        iiv_cv_plot,
        omega_heatmap,
    )


@app.cell
def _(diag_df, eta_pairs):
    fig_eta_pairs = eta_pairs(diag_df, title="ETA Pairwise Scatter")
    fig_eta_pairs
    return (fig_eta_pairs,)


@app.cell
def _(diag_df, eta_shrinkage_plot, result):
    try:
        result.compute_shrinkage(iwres=diag_df["IWRES"].to_numpy())
        fig_shrink = eta_shrinkage_plot(result, title="ETA Shrinkage")
        fig_shrink
    except Exception as e:
        print(f"Shrinkage plot: {e}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. ETA vs Covariates

        Plot individual ETAs against observed covariates to identify
        covariate relationships that could explain IIV.  A systematic trend
        suggests the covariate should be added to the model.
        """
    )
    return


@app.cell
def _(diag_df):
    from openpkpd.plots.eta import eta_vs_covariate

    # diag_df has WT if it was in the original dataset
    if "WT" in diag_df.columns and "ETA1" in diag_df.columns:
        fig_eta_cov = eta_vs_covariate(
            diag_df,
            covariate="WT",
            eta_col="ETA1",
            title="ETA vs Body Weight",
        )
        fig_eta_cov
    else:
        print("Covariate WT or ETA1 not available in the diagnostic DataFrame.")
    return (eta_vs_covariate,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. Concentration-Time Plots
        """
    )
    return


@app.cell
def _(diag_df):
    from openpkpd.plots.pk import (
        spaghetti_plot,
        concentration_time,
        individual_fit_grid,
        mean_profile,
    )

    fig_sp = spaghetti_plot(diag_df, title="Spaghetti Plot — Theophylline FOCEI")
    fig_sp
    return (
        concentration_time,
        fig_sp,
        individual_fit_grid,
        mean_profile,
        spaghetti_plot,
    )


@app.cell
def _(diag_df, individual_fit_grid):
    fig_grid = individual_fit_grid(
        diag_df,
        n_cols=3,
        title="Individual Fits (Theophylline FOCEI)",
    )
    fig_grid
    return (fig_grid,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. Parameter Uncertainty and OFV History
        """
    )
    return


@app.cell
def _(result):
    from openpkpd.plots.model_perf import ofv_history, parameter_uncertainty_plot

    try:
        fig_ofv = ofv_history(result, title="OFV History")
        fig_ofv
    except Exception as e:
        print(f"OFV history: {e}")
    return ofv_history, parameter_uncertainty_plot


@app.cell
def _(parameter_uncertainty_plot, result):
    try:
        fig_pu = parameter_uncertainty_plot(result, title="Parameter Uncertainty")
        fig_pu
    except Exception as e:
        print(f"Parameter uncertainty: {e}")
    return (fig_pu,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary — Plot Gallery

        | Plot | Function | Module |
        |------|----------|--------|
        | Full GOF panel | `diagnostic_panel(diag_df)` | `plots.gof` |
        | DV vs PRED | `dv_vs_pred(diag_df)` | `plots.gof` |
        | DV vs IPRED | `dv_vs_ipred(diag_df)` | `plots.gof` |
        | CWRES vs TIME | `cwres_vs_time(diag_df)` | `plots.gof` |
        | CWRES Q-Q | `cwres_qq(diag_df)` | `plots.gof` |
        | ETA histograms | `eta_histograms(diag_df, result.omega_final)` | `plots.eta` |
        | ETA pairs | `eta_pairs(diag_df)` | `plots.eta` |
        | ETA vs covariate | `eta_vs_covariate(diag_df, cov)` | `plots.eta` |
        | Spaghetti | `spaghetti_plot(diag_df)` | `plots.pk` |
        | Individual fits | `individual_fit_grid(diag_df)` | `plots.pk` |
        | OFV history | `ofv_history(result)` | `plots.model_perf` |
        | VPC | `vpc_plot(vpc_result)` | `plots.simulation` |
        | NPDE | `npde_plot(npde_result)` | `plots.simulation` |
        | MCMC trace (by chain) | `mcmc_trace_by_chain_plot(chains)` | `plots.bayesian` |
        | R-hat | `rhat_plot(result.r_hat)` | `plots.bayesian` |
        | ESS | `ess_plot(result.n_effective, n_total)` | `plots.bayesian` |
        | Posterior density | `posterior_density_plot(result)` | `plots.bayesian` |
        | Posterior forest | `posterior_forest_plot(result)` | `plots.bayesian` |

        **Next:** `09_covariate_modeling.py` — SCM and covariate effects.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. MCMC Diagnostic Plots (Bayesian Estimation)

        When a run uses `.estimation(method="BAYES")`, the result is a
        `BayesianResult` that carries per-chain posterior samples, R-hat,
        and effective sample size (ESS).  Three dedicated plot functions
        give a full convergence picture.

        ### Data layout

        ```python
        # result.posterior_samples_by_chain["theta"]
        # shape: (n_chains, n_draws, n_params)
        chains = result.posterior_samples_by_chain["theta"]  # e.g. (4, 1000, 5)

        # Flattened (all chains together)
        flat = result.posterior_samples["theta"]             # (4000, 5)

        param_names = ["KA", "V", "CL", "a_err", "b_err"]
        ```

        ### Per-chain trace plot

        ```python
        from openpkpd.plots import mcmc_trace_by_chain_plot

        fig = mcmc_trace_by_chain_plot(
            chains,
            param_names=param_names,
            burnin=0,     # shade warm-up iterations if kept in samples
            n_cols=2,
        )
        ```

        Each subplot shows all chains overlaid in distinct colours.
        Well-mixed chains look like overlapping "hairy caterpillars".
        Separated or drifting chains indicate poor convergence.

        ### R-hat bar chart

        ```python
        from openpkpd.plots import rhat_plot

        fig = rhat_plot(
            result.r_hat,
            param_names=param_names,
            threshold=1.01,   # strict threshold; 1.1 is the standard minimum
        )
        ```

        Bars are coloured **green** (R-hat ≤ threshold) or **red** (> threshold).
        The annotation shows how many parameters are not yet converged.

        | R-hat | Interpretation |
        |-------|----------------|
        | ≤ 1.01 | Excellent convergence |
        | 1.01 – 1.05 | Acceptable for most purposes |
        | 1.05 – 1.1 | Marginal — run more iterations |
        | > 1.1 | Not converged — do not use estimates |

        ### ESS bar chart

        ```python
        from openpkpd.plots import ess_plot

        n_total = 4 * 1000   # n_chains × n_draws
        fig = ess_plot(
            result.n_effective,
            n_total,
            param_names=param_names,
            target_fraction=0.1,   # flag if ESS < 10% of total
        )
        ```

        Bars below the target line (dashed) indicate high autocorrelation
        — increase `nsamples` or thin the chain.

        ### Standalone diagnostic functions

        ```python
        from openpkpd.estimation.mcmc_diagnostics import (
            compute_rhat,
            compute_ess,
            compute_autocorr,
        )

        # chains: shape (n_chains, n_draws, n_params)
        rhat  = compute_rhat(chains)          # shape (n_params,)
        ess   = compute_ess(chains)            # shape (n_params,)

        # Autocorrelation at each lag for chain 0, parameter 0
        acf = compute_autocorr(chains[0, :, 0], max_lag=50)  # shape (51,)
        ```

        Both `compute_rhat` and `compute_ess` implement the **split-R-hat**
        method of Vehtari et al. (2021) with rank normalisation, so they work
        without ArviZ on any backend (NumPyro, PyMC, or custom chains).

        ### Marginal posteriors and forest plot

        ```python
        from openpkpd.plots import posterior_density_plot, posterior_forest_plot

        # Violin + 95% credible interval per parameter
        fig_density = posterior_density_plot(
            result, param_names=param_names,
            ci_lo=result.posterior_ci_lo, ci_hi=result.posterior_ci_hi,
        )

        # Forest plot — population θ with credible intervals
        fig_forest = posterior_forest_plot(result, param_names=param_names)
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
