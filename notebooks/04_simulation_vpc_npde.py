"""
OpenPKPD — Notebook 04: Simulation, VPC, and NPDE

Covers:
  - SimulationEngine: generating replicate datasets
  - VPCEngine: computing percentile bands for visual predictive checks
  - NPDEEngine: normalised prediction distribution errors (Brendel 2006)
  - NPCEngine: numerical predictive check
  - SSEEngine: stochastic simulation and re-estimation
  - Plotting: vpc_plot, npde_plot, simulation_panel
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — Simulation, VPC & NPDE")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Simulation, VPC, and NPDE

        After fitting a population model, simulation-based diagnostics let us
        assess whether the model adequately describes the data.

        ## Workflow

        ```
        EstimationResult
              │
              ▼
        SimulationEngine.simulate(n_replicates=K)
              │
              ├──► VPCEngine  → prediction interval bands (VPC)
              ├──► NPDEEngine → Normalised Prediction Distribution Errors
              ├──► NPCEngine  → numerical predictive check (NPC)
              └──► SSEEngine  → stochastic simulation and re-estimation (SSE)
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
    from openpkpd.simulation import (
        SimulationEngine,
        VPCEngine,
        NPDEEngine,
        NPCEngine,
    )

    return (
        io,
        np,
        pd,
        NONMEMDataset,
        ModelBuilder,
        SimulationEngine,
        VPCEngine,
        NPDEEngine,
        NPCEngine,
    )


@app.cell
def _(io, pd, NONMEMDataset, ModelBuilder):
    THEO_CSV = """\
ID,TIME,AMT,DV,EVID,MDV
1,0,4.02,0,1,1
1,0.27,0,0.74,0,0
1,0.57,0,1.72,0,0
1,1.02,0,7.91,0,0
1,1.92,0,8.31,0,0
1,3.5,0,8.33,0,0
1,5.02,0,6.85,0,0
1,7.03,0,6.08,0,0
1,9.0,0,5.4,0,0
1,12.05,0,4.55,0,0
1,24.37,0,1.25,0,0
2,0,4.4,0,1,1
2,0.35,0,0.96,0,0
2,0.6,0,2.33,0,0
2,1.07,0,4.71,0,0
2,2.13,0,8.33,0,0
2,3.5,0,9.02,0,0
2,5.02,0,7.14,0,0
2,7.02,0,5.68,0,0
2,9.1,0,4.55,0,0
2,12.1,0,3.01,0,0
2,25.0,0,0.9,0,0
3,0,4.95,0,1,1
3,0.27,0,0.64,0,0
3,0.58,0,1.92,0,0
3,1.02,0,4.44,0,0
3,1.92,0,7.03,0,0
3,3.5,0,9.07,0,0
3,5.02,0,7.56,0,0
3,7.02,0,6.59,0,0
3,9.0,0,5.88,0,0
3,12.15,0,4.73,0,0
3,24.17,0,1.25,0,0
4,0,4.53,0,1,1
4,0.3,0,1.03,0,0
4,0.52,0,2.02,0,0
4,1.0,0,5.63,0,0
4,1.92,0,8.6,0,0
4,3.5,0,8.38,0,0
4,5.02,0,7.54,0,0
4,7.07,0,6.88,0,0
4,9.0,0,5.78,0,0
4,12.12,0,3.99,0,0
4,24.08,0,1.17,0,0
"""

    ds = NONMEMDataset.from_dataframe(pd.read_csv(io.StringIO(THEO_CSV)))

    built = (
        ModelBuilder()
        .problem("Theophylline FOCE for simulation")
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
        .build()
    )

    result = built.fit()
    print(result.summary())
    return THEO_CSV, built, ds, result


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. SimulationEngine

        `SimulationEngine` generates $K$ replicate datasets by drawing
        $\eta_i \sim \mathcal{N}(0, \hat{\Omega})$ and
        $\varepsilon_{ij} \sim \mathcal{N}(0, \hat{\Sigma})$ for each
        replicate.

        The output `SimulationResult.simulated_df` is a single long-format
        DataFrame with a `REP` column:
        - `REP=0`: the original observed data
        - `REP=1..K`: the simulated replicates
        """
    )
    return


@app.cell
def _(SimulationEngine, built, result):
    sim_engine = SimulationEngine(built.population_model, result, seed=42)
    sim_result = sim_engine.simulate(n_replicates=200)

    full_df = sim_result.simulated_df
    print(f"Total rows: {len(full_df)}")
    print(f"REP=0 (observed): {(full_df['REP'] == 0).sum()}")
    print(f"REP>=1 (simulated): {(full_df['REP'] >= 1).sum()}")
    full_df.head(20)
    return full_df, sim_engine, sim_result


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. VPCEngine — Visual Predictive Check

        `VPCEngine` bins observations by time, computes the 5th, 50th, and 95th
        percentiles of the observed and simulated distributions, and returns
        a `VPCResult` ready to plot.

        Key options:
        - `n_replicates`: number of Monte Carlo datasets (≥500 recommended)
        - `n_bins`: number of time bins for binning
        - `pi_lower/pi_upper`: percentile bounds (default 5th/95th)
        """
    )
    return


@app.cell
def _(SimulationEngine, VPCEngine, built, result):
    _sim_engine = SimulationEngine(built.population_model, result, seed=42)
    vpc_engine = VPCEngine(_sim_engine)
    vpc_result = vpc_engine.compute(n_replicates=200, n_bins=8)
    print(vpc_result.summary())
    return vpc_engine, vpc_result


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openpkpd.plots.simulation import vpc_plot, npde_plot, simulation_panel

    return matplotlib, npde_plot, plt, simulation_panel, vpc_plot


@app.cell
def _(vpc_plot, vpc_result):
    fig_vpc = vpc_plot(vpc_result, title="Theophylline VPC (200 replicates)")
    fig_vpc
    return (fig_vpc,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Stratified VPC

        For datasets with multiple dose groups or covariates, stratify the VPC
        by a grouping variable:

        ```python
        from openpkpd.plots.simulation import stratified_vpc_plot

        fig = stratified_vpc_plot(
            vpc_result,
            strata_col="DOSE",
            title="VPC stratified by dose",
        )
        ```

        ## 4. pc-VPC (Prediction-Corrected VPC)

        The pc-VPC normalises observations by the population prediction to
        remove the dose or concentration gradient across time bins:

        ```python
        from openpkpd.plots.simulation import pcvpc_plot

        fig = pcvpc_plot(vpc_result, title="pcVPC")
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. NPDEEngine — Normalised Prediction Distribution Errors

        NPDE (Brendel et al., 2006) is a more rigorous simulation-based
        diagnostic than VPC.  For each observed value $y_{ij}$, the predictive
        CDF is evaluated from $K$ simulated replicates:

        $$
        \mathrm{pd}_{ij} = \frac{\#\{r : \tilde{y}_{ij}^r < y_{ij}\}
                           + 0.5 \cdot \#\{\text{ties}\}}{K}
        $$

        Then transformed to the normal scale and decorrelated within subjects:

        $$
        \mathrm{NPDE}_{ij} = L_i^{-\top} \Phi^{-1}(\mathrm{pd}_{ij})
        $$

        Under a correct model, NPDE ~ N(0,1) independently across all observations.
        """
    )
    return


@app.cell
def _(NPDEEngine, SimulationEngine, built, result):
    _sim_engine_npde = SimulationEngine(built.population_model, result, seed=99)
    npde_engine = NPDEEngine(_sim_engine_npde)
    npde_result = npde_engine.compute(n_replicates=500, seed=99)
    print(npde_result.summary())
    return npde_engine, npde_result


@app.cell
def _(npde_plot, npde_result):
    fig_npde = npde_plot(npde_result, title="Theophylline NPDE")
    fig_npde
    return (fig_npde,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. NPCEngine — Numerical Predictive Check

        NPC quantifies how well prediction intervals contain the observed data.
        It counts the fraction of observations falling below each simulated
        percentile:

        ```python
        from openpkpd.simulation import NPCEngine

        npc_engine = NPCEngine(sim_engine)
        npc_result = npc_engine.compute(n_replicates=1000, percentiles=[5, 10, 25, 50, 75, 90, 95])
        print(npc_result.summary())
        ```

        Under a correct model, the fraction below the $p$-th percentile should
        equal $p/100$.
        """
    )
    return


@app.cell
def _(NPCEngine, SimulationEngine, built, result):
    _sim_engine_npc = SimulationEngine(built.population_model, result, seed=7)
    npc_engine = NPCEngine(_sim_engine_npc)
    npc_result = npc_engine.compute(n_replicates=200)
    print(npc_result.summary())
    return npc_engine, npc_result


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 7. Individual Simulation Panel

        Plot a grid of individual profiles (observed vs. simulated) to spot
        systematic misfits in specific subjects:
        """
    )
    return


@app.cell
def _(full_df, simulation_panel):
    sim_df = full_df[full_df["REP"] >= 1]
    obs_df = full_df[(full_df["REP"] == 0) & (full_df["MDV"] == 0)][["ID", "TIME", "DV"]].copy()

    fig_panel = simulation_panel(
        simulated_df=sim_df,
        observed_df=obs_df,
        n_subjects=4,
        title="Theophylline — Simulated vs Observed",
    )
    fig_panel
    return fig_panel, obs_df, sim_df


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 8. SSEEngine — Stochastic Simulation and Re-Estimation

        SSE assesses parameter estimation performance by:
        1. Simulating $K$ datasets from the fitted model
        2. Re-fitting the model to each simulated dataset
        3. Computing bias, precision, and coverage of confidence intervals

        ```python
        from openpkpd.simulation import SSEEngine

        sse = SSEEngine(sim_engine, built)
        sse_result = sse.run(n_replicates=100, seed=0)
        print(sse_result.summary())
        ```

        SSE is computationally intensive (requires $K$ full fits) and is best
        run with parallelism enabled.

        ## Summary

        | Engine | Purpose | Key method |
        |--------|---------|------------|
        | `SimulationEngine` | Generate replicate datasets | `.simulate(n_replicates=K)` |
        | `VPCEngine` | Prediction interval bands | `.compute(n_replicates, n_bins)` |
        | `NPDEEngine` | Normalised prediction errors | `.compute(n_replicates, seed)` |
        | `NPCEngine` | Numerical predictive check | `.compute(n_replicates)` |
        | `SSEEngine` | Estimation performance | `.run(n_replicates)` |
        """
    )
    return


if __name__ == "__main__":
    app.run()
