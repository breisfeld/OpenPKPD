"""
OpenPKPD — Notebook 03: Estimation Methods

Covers:
  - FO (First-Order)
  - FOCE / FOCEI (First-Order Conditional Estimation with Interaction)
  - SAEM (Stochastic Approximation EM)
  - IMP (Importance Sampling)
  - Nonparametric estimation
  - Laplacian approximation
  - Comparison of OFV across methods
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — Estimation Methods")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Estimation Methods

        Population PK/PD models are fitted by maximising the likelihood (or
        equivalently minimising the **Objective Function Value**, OFV) over the
        population parameters $\Theta$, $\Omega$, $\Sigma$.

        Because the marginal likelihood requires integrating over the random
        effects $\eta_i$, various approximations are used:

        | Method | Approximation | Speed | Accuracy |
        |--------|--------------|-------|----------|
        | **FO** | First-order Taylor around η=0 | Fastest | Lowest |
        | **FOCE** | First-order Taylor around η̂ᵢ | Fast | Good |
        | **FOCEI** | FOCE + ε-η interaction | Fast | Better |
        | **Laplacian** | Second-order Taylor | Moderate | High |
        | **SAEM** | Stochastic EM on latent η | Slow | Very high |
        | **IMP** | Monte Carlo importance sampling | Slow | Very high |
        | **Nonparametric** | NPML / NPMLE | Moderate | Distribution-free |

        OpenPKPD also supports **Bayesian estimation** via `method="BAYES"`,
        using either the **PyMC** back-end (install `openpkpd[bayes]`) or the
        built-in pure-NumPy **NUTS** sampler (no extra dependency required).
        The built-in NUTS backend currently samples THETA only; use PyMC or
        `BAYES(backend="laplace")` for faster Bayesian summaries.
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
def _(io, pd, NONMEMDataset):
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
"""

    df = pd.read_csv(io.StringIO(THEO_CSV))
    ds = NONMEMDataset.from_dataframe(df)
    ds
    return THEO_CSV, df, ds


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. FO — First-Order Estimation

        FO linearises the model around $\hat{\eta}_i = 0$ for every subject.
        It is the fastest method but biased when IIV is large or the model
        is nonlinear in $\eta$.

        ```python
        .estimation(method="FO", maxeval=500)
        ```

        **Use when:** Initial exploration, very large datasets, or as a warm
        start before FOCE.
        """
    )
    return


@app.cell
def _(ModelBuilder, ds):
    def build_model(method: str, interaction: bool = False, maxeval: int = 500):
        return (
            ModelBuilder()
            .problem(f"Theophylline 1-cmt — {method}")
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
            .estimation(method=method, interaction=interaction, maxeval=maxeval)
            .build()
        )

    built_fo = build_model("FO")
    result_fo = built_fo.fit()
    print(f"FO  OFV = {result_fo.ofv:.4f}  converged={result_fo.converged}")
    return build_model, built_fo, result_fo


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. FOCE / FOCEI — First-Order Conditional Estimation

        FOCE evaluates the likelihood at the *conditional* mode $\hat{\eta}_i$
        for each subject.  This is more accurate than FO for nonlinear models.

        Adding `interaction=True` accounts for the $\varepsilon$-$\eta$ interaction
        (i.e. the residual variance depends on individual predictions), giving **FOCEI**.
        FOCEI is the default recommended method for most PK/PD analyses.

        ```python
        .estimation(method="FOCE", interaction=True, maxeval=9999)
        ```

        **FOCEI OFV formula** (per-subject contribution):

        $$
        \text{OFV}_i = (\mathbf{y}_i - \mathbf{f}_i)^\top \mathbf{C}_i^{-1}
                       (\mathbf{y}_i - \mathbf{f}_i)
                     + \ln|\mathbf{C}_i|
                     + \hat{\boldsymbol{\eta}}_i^\top \boldsymbol{\Omega}^{-1} \hat{\boldsymbol{\eta}}_i
                     + \ln|\boldsymbol{\Omega}|
                     + n_i \ln(2\pi)
        $$

        where $\mathbf{C}_i = G_i \Omega G_i^\top + R_i$ is the marginal
        covariance of the data.

        FOCEI also exposes advanced robustness controls:

        ```python
        .estimation(
            method="FOCEI",
            maxeval=600,
            outer_optimizer="L-BFGS-B",
            outer_fallback_optimizer="Powell",
            outer_fallback_maxeval=40,
            retain_best_iterate=True,
            retry_on_abnormal=True,
            retry_omega_scales=[0.5, 0.25],
        )
        ```
        """
    )
    return


@app.cell
def _(build_model):
    built_foce = build_model("FOCE", interaction=True, maxeval=600)
    result_foce = built_foce.fit()
    print(f"FOCE OFV = {result_foce.ofv:.4f}  converged={result_foce.converged}")
    return built_foce, result_foce


@app.cell
def _(ModelBuilder, ds):
    built_focei_robust = (
        ModelBuilder()
        .problem("Theophylline 1-cmt oral — FOCEI robust options")
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
        .estimation(
            method="FOCEI",
            maxeval=600,
            outer_optimizer="L-BFGS-B",
            outer_fallback_optimizer="Powell",
            outer_fallback_maxeval=40,
            retain_best_iterate=True,
            retry_on_abnormal=True,
            retry_omega_scales=[0.5, 0.25],
        )
        .build()
    )
    options = built_focei_robust.estimation_kwargs
    print("FOCEI advanced options")
    print(
        "  "
        f"outer={options['outer_optimizer']} "
        f"fallback={options['outer_fallback_optimizer']} "
        f"retain_best={options['retain_best_iterate']} "
        f"retry={options['retry_on_abnormal']} "
        f"scales={options['retry_omega_scales']}"
    )
    return built_focei_robust, options


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. SAEM — Stochastic Approximation EM

        SAEM is a global stochastic optimisation algorithm that avoids local
        minima and scales well to complex models.  It is particularly useful
        when:

        - The model is highly nonlinear
        - Good initial estimates are unavailable
        - The IIV structure is complex (correlated, large)

        ```python
        .estimation(
            method="SAEM",
            n_iter_phase1=200,    # stochastic exploration iterations
            n_iter_phase2=100,    # smoothing / convergence iterations
            seed=42,
        )
        ```

        SAEM may be followed by IMP for an accurate OFV estimate:

        ```python
        .estimation(method="SAEM", n_iter_phase1=300, n_iter_phase2=100)
        .estimation(method="IMP", isample=1000)  # chained estimation
        ```
        """
    )
    return


@app.cell
def _(ModelBuilder, NONMEMDataset, build_model, io, pd):
    _THEO = """\
ID,TIME,AMT,DV,EVID,MDV
1,0,4.02,0,1,1
1,1.02,0,7.91,0,0
1,3.5,0,8.33,0,0
1,7.03,0,6.08,0,0
1,12.05,0,4.55,0,0
1,24.37,0,1.25,0,0
2,0,4.4,0,1,1
2,1.07,0,4.71,0,0
2,3.5,0,9.02,0,0
2,7.02,0,5.68,0,0
2,12.1,0,3.01,0,0
2,25.0,0,0.9,0,0
3,0,4.95,0,1,1
3,1.02,0,4.44,0,0
3,3.5,0,9.07,0,0
3,7.02,0,6.59,0,0
3,12.15,0,4.73,0,0
3,24.17,0,1.25,0,0
"""
    _ds_saem = NONMEMDataset.from_dataframe(pd.read_csv(io.StringIO(_THEO)))

    built_saem_model = (
        ModelBuilder()
        .problem("Theophylline SAEM")
        .dataset(_ds_saem)
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
        .estimation(method="SAEM", n_iter_phase1=100, n_iter_phase2=50, seed=42)
        .build()
    )
    result_saem = built_saem_model.fit()
    print(f"SAEM OFV = {result_saem.ofv:.4f}  converged={result_saem.converged}")
    return (
        _THEO,
        _ds_saem,
        built_saem_model,
        result_saem,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Comparing Methods

        We compare the OFV and parameter estimates across methods.
        A **lower OFV is better** (= higher likelihood).

        Note: FO OFV is not directly comparable to FOCE/SAEM OFV because
        different approximations are used.  Always compare models fitted
        with the *same* method. The FOCEI robust controls above affect search
        behavior rather than the likelihood definition itself.
        """
    )
    return


@app.cell
def _(mo, pd, result_fo, result_foce, result_saem):
    comparison = pd.DataFrame(
        {
            "Method": ["FO", "FOCE+I", "SAEM"],
            "OFV": [result_fo.ofv, result_foce.ofv, result_saem.ofv],
            "KA": [r.theta_final[0] for r in [result_fo, result_foce, result_saem]],
            "CL": [r.theta_final[1] for r in [result_fo, result_foce, result_saem]],
            "V": [r.theta_final[2] for r in [result_fo, result_foce, result_saem]],
            "Converged": [r.converged for r in [result_fo, result_foce, result_saem]],
        }
    ).round(4)

    mo.vstack(
        [
            mo.md("### Parameter Estimates by Method"),
            mo.ui.table(comparison),
        ]
    )
    return (comparison,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. OFV History Plot

        When available, the OFV trajectory over optimisation iterations
        helps diagnose convergence:
        """
    )
    return


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openpkpd.plots.model_perf import ofv_history

    return matplotlib, ofv_history, plt


@app.cell
def _(ofv_history, result_foce):
    try:
        fig_ofv = ofv_history(result_foce, title="FOCE OFV history")
        fig_ofv
    except Exception as e:
        print(f"OFV history plot not available: {e}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. ETA Shrinkage and De-shrinkage

        EBEs from FOCE are shrunk toward zero relative to the true individual
        values. When shrinkage is high (> 30%), covariate plots and ETA
        histograms based on raw EBEs underestimate between-subject variability.

        ```python
        # Check shrinkage after fitting
        result.compute_shrinkage()
        print(result.eta_shrinkage)   # fractions, e.g. [0.62, 0.41, 0.55]

        # Obtain de-shrunken EBEs (Combes 2013 rescaling)
        adj = result.compute_deshrinkage_etas()
        # adj is dict[subject_id → adjusted_eta_vector]
        # SD(adj_etas_k) == sqrt(omega_kk) for each k
        ```

        The correction factor per random effect k is `1 / (1 − shrinkage_k)`.
        Subject ordering is preserved; only the dispersion is corrected.

        The HTML report (generated by `result.to_html(...)`) automatically
        computes and displays de-shrunken ETAs — including an SD summary table
        and a per-subject collapsible table — whenever any ETA shrinkage
        exceeds 30%.

        ## 7. Nonparametric Estimation

        The nonparametric estimator (NPMLE) makes no distributional assumption
        about $\eta$.  Instead, it finds a discrete support that maximises the
        likelihood.

        ```python
        .estimation(method="NONPARAMETRIC", npde_options={"support_size": 50})
        ```

        The result contains a support grid and probability weights instead
        of an $\Omega$ matrix.

        ## 7. Bayesian Estimation (NUTS / MCMC)

        OpenPKPD supports Bayesian posterior sampling via the No-U-Turn Sampler
        (NUTS) through two paths:

        - **Built-in NumPy NUTS** — no extra dependency; `backend="nuts"`.
          Currently samples **THETA only** (OMEGA and SIGMA remain fixed).
          Best for lightweight checks and models with analytical PK subroutines.
        - **PyMC** — install `openpkpd[bayes]`; `backend="pymc"`. Recommended
          for primary MCMC workflows needing full posterior coverage.
        - **Laplace** — `backend="laplace"`. Fast Gaussian approximation at the
          MAP; no MCMC required. Good for quick Bayesian summaries.

        ```bash
        pip install "openpkpd[bayes]"   # PyMC backend (recommended for full MCMC)
        ```

        ### Configuring the sampler

        ```python
        model = (
            ModelBuilder()
            .problem("Theophylline — Bayesian")
            .data("examples/shared_data/theophylline/theophylline.csv")
            .subroutines(advan=2, trans=2)
            .pk(\"\"\"
                KA = THETA(1) * EXP(ETA(1))
                V  = THETA(2) * EXP(ETA(2)) * WT
                CL = THETA(3) * EXP(ETA(3)) * WT
            \"\"\")
            .error("Y = F + (THETA(4) + THETA(5)*F)*EPS(1)")
            .theta([(0, 1.53, None), (0, 0.456, None), (0, 0.0402, None),
                    (0, 0.3, None), (0, 0.1, None)])
            .omega([0.45**2, 0.26**2, 0.33**2])
            .sigma([1.0])
            .estimation(
                method="BAYES",
                nsamples=1000,   # posterior draws per chain (after warm-up)
                nchains=4,       # independent Markov chains
                nwarmup=500,     # NUTS warm-up / step-size adaptation
                backend="pymc",  # or "nuts" (built-in) or "laplace"
                seed=42,
            )
        )

        result = fit(model)   # returns BayesianResult
        ```

        ### Inspecting the posterior

        ```python
        # Posterior summary — mean, SD, 95% CrI, R-hat, ESS
        print(result.posterior_summary())

        # Point estimates (posterior means)
        print(result.theta_final)

        # Full posterior samples  (n_total × n_theta)
        samples = result.posterior_samples["theta"]   # shape (4000, 5)

        # Per-chain samples for convergence diagnostics  (chains × draws × params)
        chains = result.posterior_samples_by_chain["theta"]  # shape (4, 1000, 5)
        ```

        ### MCMC convergence diagnostics

        ```python
        from openpkpd.plots import (
            mcmc_trace_by_chain_plot,
            rhat_plot,
            ess_plot,
            posterior_density_plot,
            posterior_forest_plot,
        )

        param_names = ["KA", "V", "CL", "a_err", "b_err"]

        # 1. Trace plots — coloured by chain; look for overlapping caterpillars
        fig_trace = mcmc_trace_by_chain_plot(
            chains, param_names=param_names, burnin=0
        )

        # 2. R-hat bar chart — all bars should be < 1.01 (green)
        fig_rhat = rhat_plot(result.r_hat, param_names=param_names, threshold=1.01)

        # 3. ESS bar chart — each bar should exceed 10% of total draws
        total_draws = 4 * 1000
        fig_ess = ess_plot(result.n_effective, total_draws, param_names=param_names)

        # 4. Marginal posterior densities (violin + 95% CrI)
        fig_density = posterior_density_plot(
            result, param_names=param_names,
            ci_lo=result.posterior_ci_lo, ci_hi=result.posterior_ci_hi
        )

        # 5. Forest plot — population parameters with credible intervals
        fig_forest = posterior_forest_plot(result, param_names=param_names)
        ```

        ### Interpreting convergence

        | Diagnostic | Good | Investigate | Poor (re-run) |
        |------------|------|-------------|---------------|
        | R-hat | ≤ 1.01 | 1.01 – 1.1 | > 1.1 |
        | ESS | > 400 | 100 – 400 | < 100 |
        | Trace plot | Chains overlap, stationary | One chain drifts | Chains separated |

        If chains diverge, try:
        - More warm-up (`nwarmup=1000`)
        - Tighter initial values (`.theta(...)`)
        - `target_accept=0.95` for NUTS step-size tuning
        - Switch to a more informative prior on `omega`

        > **Tip**: `result.converged` is `True` when all R-hat values ≤ 1.1.
        > Check `result.r_hat` for per-parameter details.

        ## Summary

        | Method | `.estimation(method=...)` | Key options |
        |--------|--------------------------|-------------|
        | FO | `"FO"` | `maxeval` |
        | FOCE | `"FOCE"` | `interaction=True`, `maxeval` |
        | SAEM | `"SAEM"` | `n_iter_phase1`, `n_iter_phase2`, `seed` |
        | IMP | `"IMP"` | `isample`, `seed` |
        | Laplacian | `"LAPLACIAN"` | `maxeval` |
        | Nonparametric | `"NONPARAMETRIC"` | `support_size` |
        | Bayesian / NUTS | `"BAYES"` | `nsamples`, `nchains`, `nwarmup`, `backend`, `seed` |
        """
    )
    return


if __name__ == "__main__":
    app.run()
