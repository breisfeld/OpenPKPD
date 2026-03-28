"""
OpenPKPD — Notebook 10: Model Comparison, LRT, and Bootstrap

Covers:
  - lrt(): likelihood ratio test between nested models
  - compare_models(): AIC/BIC table and Akaike weights for non-nested models
  - aic_weights(): model averaging weights
  - Bootstrap confidence intervals for parameter estimates
  - model_comparison_plot and bootstrap_ci_plot
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — Inference & Bootstrap")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Model Comparison, Inference, and Bootstrap

        After fitting candidate models, statistical inference is used to decide
        which model best describes the data.

        ## Tools Available

        | Tool | Purpose |
        |------|---------|
        | `lrt()` | Likelihood ratio test for **nested** models |
        | `compare_models()` | AIC/BIC table for nested or non-nested models |
        | `aic_weights()` | Akaike model weights (probability each model is best) |
        | Bootstrap | Non-parametric CI for parameters, robust to distributional assumptions |

        ## Likelihood Ratio Test (LRT)

        For nested models (base model is a special case of the full model):

        $$\chi^2 = -2 \cdot (\ell_{\text{base}} - \ell_{\text{full}}) = \text{OFV}_{\text{base}} - \text{OFV}_{\text{full}}$$

        Under $H_0$ (no additional effect), $\chi^2 \sim \chi^2_{df}$ where
        $df$ = number of additional parameters.

        | Significance | 1 df threshold |
        |-------------|----------------|
        | $p < 0.05$  | ΔOFV > 3.84    |
        | $p < 0.01$  | ΔOFV > 6.63    |
        | $p < 0.001$ | ΔOFV > 10.83   |
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
    from openpkpd.inference import lrt, compare_models, aic_weights, LRTResult

    return (
        io,
        np,
        pd,
        NONMEMDataset,
        ModelBuilder,
        lrt,
        compare_models,
        aic_weights,
        LRTResult,
    )


@app.cell
def _(io, pd, NONMEMDataset, ModelBuilder):
    THEO_CSV = """\
ID,TIME,AMT,DV,EVID,MDV,WT
1,0,4.02,0,1,1,79.6
1,1.02,0,7.91,0,0,79.6
1,3.5,0,8.33,0,0,79.6
1,7.03,0,6.08,0,0,79.6
1,12.05,0,4.55,0,0,79.6
1,24.37,0,1.25,0,0,79.6
2,0,4.4,0,1,1,72.4
2,1.07,0,4.71,0,0,72.4
2,3.5,0,9.02,0,0,72.4
2,7.02,0,5.68,0,0,72.4
2,12.1,0,3.01,0,0,72.4
2,25.0,0,0.9,0,0,72.4
3,0,4.95,0,1,1,70.5
3,1.02,0,4.44,0,0,70.5
3,3.5,0,9.07,0,0,70.5
3,7.02,0,6.59,0,0,70.5
3,12.15,0,4.73,0,0,70.5
3,24.17,0,1.25,0,0,70.5
4,0,4.53,0,1,1,72.7
4,1.0,0,5.63,0,0,72.7
4,3.5,0,8.38,0,0,72.7
4,7.07,0,6.88,0,0,72.7
4,12.12,0,3.99,0,0,72.7
4,24.08,0,1.17,0,0,72.7
5,0,5.68,0,1,1,54.6
5,1.02,0,9.29,0,0,54.6
5,3.62,0,8.58,0,0,54.6
5,7.07,0,7.47,0,0,54.6
5,12.15,0,5.94,0,0,54.6
5,24.17,0,3.28,0,0,54.6
"""
    ds = NONMEMDataset.from_dataframe(pd.read_csv(io.StringIO(THEO_CSV)))

    def make_model(method="FO", omega_size=3, add_wt=False):
        pk = """
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
"""
        if add_wt:
            pk += "\nCL = CL * (WT/70)**THETA(4)"

        theta_specs = [(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)]
        if add_wt:
            theta_specs.append((0.01, 1.0, 2.0))

        builder = (
            ModelBuilder()
            .problem("Theophylline comparison")
            .dataset(ds)
            .subroutines(advan=2, trans=2)
            .pk(pk)
            .error("Y = F*(1 + EPS(1))")
            .theta(theta_specs)
            .omega([0.5, 0.3, 0.3] if omega_size == 3 else [0.5])
            .sigma(0.1)
            .estimation(method=method, maxeval=400)
        )
        return builder.build()

    result_fo = make_model(method="FO").fit()
    result_fo_wt = make_model(method="FO", add_wt=True).fit()
    result_foce = make_model(method="FOCE").fit()

    print(f"1-cmt FO   OFV = {result_fo.ofv:.3f}")
    print(f"1-cmt FO+WT OFV = {result_fo_wt.ofv:.3f}")
    print(f"1-cmt FOCE OFV = {result_foce.ofv:.3f}")
    return THEO_CSV, ds, make_model, result_fo, result_fo_wt, result_foce


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Likelihood Ratio Test
        """
    )
    return


@app.cell
def _(lrt, mo, result_fo, result_fo_wt):
    lrt_result = lrt(result_fo_wt, result_fo)
    print(lrt_result)

    mo.md(
        f"""
        **LRT Result:**

        - Reduced OFV (FO):      `{result_fo.ofv:.3f}`
        - Full OFV (FO + WT):    `{result_fo_wt.ofv:.3f}`
        - ΔOFV:                  `{lrt_result.delta_ofv:.3f}`
        - Degrees of freedom:    `{lrt_result.df}`
        - p-value:               `{lrt_result.p_value:.4f}`

        *This comparison uses the same estimation method and differs by one
        additional covariate effect, so the LRT assumptions are met.*
        """
    )
    return (lrt_result,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. AIC / BIC Comparison Table

        For non-nested models or as a complement to LRT, use information criteria:

        $$\text{AIC} = \text{OFV} + 2p, \qquad \text{BIC} = \text{OFV} + p \ln N$$

        where $p$ is the number of estimated parameters and $N$ is the number
        of observations.  **Lower AIC/BIC is better.**
        """
    )
    return


@app.cell
def _(compare_models, mo, result_fo, result_fo_wt, result_foce):
    comparison_table = compare_models(
        results=[result_fo, result_fo_wt, result_foce],
        labels=["1-cmt FO", "1-cmt FO + WT", "1-cmt FOCE"],
    )
    print(comparison_table)

    mo.vstack(
        [
            mo.md("### Model Comparison Table"),
            mo.ui.table(comparison_table)
            if hasattr(comparison_table, "columns")
            else mo.md(str(comparison_table)),
        ]
    )
    return (comparison_table,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Akaike Weights

        Akaike weights convert the AIC differences into probabilities that
        each model is the best approximating model in the candidate set:

        $$w_i = \frac{e^{-\Delta_i / 2}}{\sum_j e^{-\Delta_j / 2}}$$

        where $\Delta_i = \text{AIC}_i - \min_j \text{AIC}_j$.
        """
    )
    return


@app.cell
def _(aic_weights, mo, result_fo, result_fo_wt, result_foce):
    weights = aic_weights([result_fo, result_fo_wt, result_foce])
    mo.md(
        f"""
        **Akaike Weights:**

        - 1-cmt FO:       `{weights[0]:.3f}`
        - 1-cmt FO + WT:  `{weights[1]:.3f}`
        - 1-cmt FOCE:     `{weights[2]:.3f}`
        """
    )
    return (weights,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Bootstrap Confidence Intervals

        Bootstrap provides non-parametric confidence intervals for parameter
        estimates.  The procedure:

        1. Sample subjects with replacement (stratified by, e.g., dose group)
        2. Refit the model to each bootstrap dataset
        3. Compute the 2.5th–97.5th percentile of the bootstrap distribution

        ```python
        from openpkpd.inference import BootstrapEngine

        boot = BootstrapEngine(built, n_bootstrap=1000, seed=42)
        boot_result = boot.run()

        print(boot_result.summary())
        # Parameter  Estimate  Bootstrap_Mean  Bootstrap_2.5  Bootstrap_97.5
        ```

        Bootstrap is computationally intensive.  Use `just run-notebook` or a
        parallel back-end (see `just run-tests` → parallel).
        """
    )
    return


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openpkpd.plots.model_perf import model_comparison_plot

    return matplotlib, model_comparison_plot, plt


@app.cell
def _(model_comparison_plot, result_fo, result_fo_wt, result_foce):
    try:
        fig_comp = model_comparison_plot(
            results=[result_fo, result_fo_wt, result_foce],
            labels=["1-cmt FO", "1-cmt FO + WT", "1-cmt FOCE"],
            title="Model Comparison",
        )
        fig_comp
    except Exception as e:
        print(f"model_comparison_plot: {e}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Likelihood Profile

        A likelihood profile plots OFV vs a single parameter value, scanning
        the parameter over a range while re-optimising the remaining parameters.
        The 95% CI is where ΔOFV < 3.84.

        ```python
        from openpkpd.plots.model_perf import likelihood_profile_plot
        from openpkpd.inference import likelihood_profile

        profile = likelihood_profile(built, param_index=0, n_points=21, width=0.5)
        fig = likelihood_profile_plot(profile, param_name="KA")
        ```

        ## Summary

        | Task | API |
        |------|-----|
        | LRT (nested models) | `lrt(result_full, result_reduced)` |
        | AIC/BIC table | `compare_models([r1, r2, ...], labels=[...])` |
        | Akaike weights | `aic_weights([r1, r2, ...])` |
        | Bootstrap CI | `BootstrapEngine(built, n_bootstrap=1000).run()` |
        | Likelihood profile | `likelihood_profile(built, param_index)` |
        | Comparison plot | `model_comparison_plot(results, labels)` |
        | Bootstrap plot | `bootstrap_ci_plot(boot_result)` |
        """
    )
    return


if __name__ == "__main__":
    app.run()
