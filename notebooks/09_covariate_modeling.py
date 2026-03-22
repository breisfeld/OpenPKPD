"""
OpenPKPD — Notebook 09: Covariate Modeling

Covers:
  - CovariateEffect types: power, linear, exponential, categorical
  - CovariateRelationship: defining candidate relationships
  - SCMEngine: stepwise covariate model building (forward + backward)
  - SCMResult: reading the final model and dropped covariates
  - covariate_forest_plot: visualising retained effects
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — Covariate Modeling")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Covariate Modeling

        Covariates (body weight, age, renal function, sex, …) can explain part
        of the between-subject variability (IIV) in PK parameters.  Adding them
        to the model reduces IIV and enables dose individualisation.

        ## Stepwise Covariate Model (SCM) Building

        SCM is a hypothesis-testing approach that:

        1. **Forward addition**: adds covariates one at a time, retaining those
           that reduce OFV by more than a critical χ² threshold
           (p < 0.05 → ΔOFV < −3.84 for 1 df).
        2. **Backward elimination**: removes covariates one at a time from the
           final forward model, dropping those that increase OFV by less than
           a more stringent threshold (p < 0.001 → ΔOFV < 10.83 for 1 df).

        ## Effect Types

        | Effect | Formula | Use for |
        |--------|---------|---------|
        | `POWER` | $P = \hat{P} \cdot (COV/\text{ref})^\theta$ | Allometric scaling, continuous |
        | `LINEAR` | $P = \hat{P} \cdot (1 + \theta \cdot (COV - \text{ref}))$ | Additive continuous |
        | `EXPONENTIAL` | $P = \hat{P} \cdot e^{\theta \cdot (COV - \text{ref})}$ | Log-scale continuous |
        | `CATEGORICAL` | $P = \hat{P} \cdot e^{\theta \cdot \mathbb{1}[\text{CAT}=k]}$ | Binary/categorical |
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
    from openpkpd.covariate import (
        CovariateEffect,
        CovariateRelationship,
        SCMEngine,
    )

    return (
        io,
        np,
        pd,
        NONMEMDataset,
        ModelBuilder,
        CovariateEffect,
        CovariateRelationship,
        SCMEngine,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Covariate Effect Functions

        Before running SCM, you can evaluate what a specific covariate effect
        looks like over the covariate range:
        """
    )
    return


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openpkpd.covariate import power_effect, linear_effect, exponential_effect

    return exponential_effect, linear_effect, matplotlib, plt, power_effect


@app.cell
def _(exponential_effect, linear_effect, np, plt, power_effect):
    wt = np.linspace(40, 120, 200)
    ref_wt = 70.0

    # Power (allometric) effect on CL: CL = CLhat * (WT/70)^0.75
    cl_power = power_effect(wt, ref=ref_wt, theta=0.75, base_param=5.0)
    # Linear effect on V
    v_linear = linear_effect(wt, ref=ref_wt, theta=0.015, base_param=30.0)
    # Exponential effect on CL
    cl_exp = exponential_effect(wt, ref=ref_wt, theta=0.01, base_param=5.0)

    fig_cov, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(wt, cl_power, lw=2, color="steelblue")
    axes[0].axvline(ref_wt, color="grey", ls=":", lw=1)
    axes[0].set_xlabel("Body Weight (kg)")
    axes[0].set_ylabel("CL (L/h)")
    axes[0].set_title("Power Effect\nCL = 5 × (WT/70)^0.75")

    axes[1].plot(wt, v_linear, lw=2, color="darkorange")
    axes[1].axvline(ref_wt, color="grey", ls=":", lw=1)
    axes[1].set_xlabel("Body Weight (kg)")
    axes[1].set_ylabel("V (L)")
    axes[1].set_title("Linear Effect\nV = 30 × (1 + 0.015×(WT−70))")

    axes[2].plot(wt, cl_exp, lw=2, color="seagreen")
    axes[2].axvline(ref_wt, color="grey", ls=":", lw=1)
    axes[2].set_xlabel("Body Weight (kg)")
    axes[2].set_ylabel("CL (L/h)")
    axes[2].set_title("Exponential Effect\nCL = 5 × exp(0.01×(WT−70))")

    fig_cov.tight_layout()
    fig_cov
    return (
        axes,
        cl_exp,
        cl_power,
        fig_cov,
        ref_wt,
        v_linear,
        wt,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Setting Up Candidate Relationships

        Define which covariate-parameter relationships to test.
        Each `CovariateRelationship` specifies:
        - `parameter`: the PK parameter to modify (e.g. `"CL"`)
        - `covariate`: the column name in the dataset
        - `effect`: the functional form (`CovariateEffect.POWER`, etc.)
        - `reference`: the reference/centering value
        """
    )
    return


@app.cell
def _(CovariateEffect, CovariateRelationship):
    candidates = [
        CovariateRelationship(
            parameter="CL",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        ),
        CovariateRelationship(
            parameter="V",
            covariate="WT",
            effect=CovariateEffect.POWER,
            reference=70.0,
        ),
        CovariateRelationship(
            parameter="CL",
            covariate="AGE",
            effect=CovariateEffect.LINEAR,
            reference=40.0,
        ),
        CovariateRelationship(
            parameter="CL",
            covariate="SEX",
            effect=CovariateEffect.CATEGORICAL,
            reference=0,
        ),
    ]
    print(f"Candidate relationships: {len(candidates)}")
    for c in candidates:
        print(f"  {c.parameter} ~ {c.covariate} ({c.effect.name})")
    return (candidates,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Building the Base Model

        The SCM needs a base model (without covariates) as its starting point.
        We use the warfarin dataset which includes WT, AGE, and SEX covariates.
        """
    )
    return


@app.cell
def _(io, pd, NONMEMDataset, ModelBuilder):
    WARF_CSV = """\
ID,TIME,AMT,DV,EVID,MDV,WT,AGE,SEX
1,0,70,0,1,1,66.7,50,1
1,24,0,0.87,0,0,66.7,50,1
1,36,0,1.21,0,0,66.7,50,1
1,48,0,1.44,0,0,66.7,50,1
1,72,0,1.52,0,0,66.7,50,1
1,96,0,1.19,0,0,66.7,50,1
2,0,70,0,1,1,78.5,45,1
2,24,0,0.51,0,0,78.5,45,1
2,36,0,0.73,0,0,78.5,45,1
2,48,0,0.88,0,0,78.5,45,1
2,72,0,1.04,0,0,78.5,45,1
2,96,0,0.95,0,0,78.5,45,1
3,0,70,0,1,1,54.3,62,0
3,24,0,1.18,0,0,54.3,62,0
3,36,0,1.73,0,0,54.3,62,0
3,48,0,2.01,0,0,54.3,62,0
3,72,0,2.11,0,0,54.3,62,0
3,96,0,1.87,0,0,54.3,62,0
4,0,70,0,1,1,85.2,38,1
4,24,0,0.44,0,0,85.2,38,1
4,36,0,0.61,0,0,85.2,38,1
4,48,0,0.79,0,0,85.2,38,1
4,72,0,0.92,0,0,85.2,38,1
4,96,0,0.84,0,0,85.2,38,1
5,0,70,0,1,1,60.1,55,0
5,24,0,1.05,0,0,60.1,55,0
5,36,0,1.42,0,0,60.1,55,0
5,48,0,1.69,0,0,60.1,55,0
5,72,0,1.74,0,0,60.1,55,0
5,96,0,1.51,0,0,60.1,55,0
"""
    warf_ds = NONMEMDataset.from_dataframe(pd.read_csv(io.StringIO(WARF_CSV)))

    base_pk = """
CL = THETA(1)*EXP(ETA(1))
V  = THETA(2)*EXP(ETA(2))
KA = THETA(3)*EXP(ETA(3))
"""

    base_builder = (
        ModelBuilder()
        .problem("Warfarin base model for SCM")
        .dataset(warf_ds)
        .subroutines(advan=2, trans=2)
        .pk(base_pk)
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.001, 0.15, 5), (1, 8, 100), (0.01, 1.5, 20)])
        .omega([0.3, 0.2, 0.4])
        .sigma(0.1)
        .estimation(method="FO", maxeval=300)
    )

    base_result = base_builder.build().fit()
    print(f"Base model OFV: {base_result.ofv:.3f}")
    return WARF_CSV, base_builder, base_pk, base_result, warf_ds


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Running the SCM

        `SCMEngine` runs forward selection followed by backward elimination.
        Each step fits a model with one additional (or removed) covariate
        and compares the OFV change to the significance threshold.
        """
    )
    return


@app.cell
def _(SCMEngine, base_builder, base_pk, candidates, warf_ds):
    scm = SCMEngine(
        base_model_builder=base_builder,
        base_pk_code=base_pk,
        candidates=candidates,
        dataset=warf_ds,
        forward_pvalue=0.05,
        backward_pvalue=0.001,
    )

    scm_result = scm.run()
    print(scm_result.summary())
    return scm, scm_result


@app.cell
def _(mo, scm_result):
    import pandas as _pd

    steps_data = []
    for step in scm_result.steps:
        steps_data.append(
            {
                "Phase": step.phase,
                "Covariate": f"{step.relationship.parameter} ~ {step.relationship.covariate}",
                "ΔOFV": round(step.delta_ofv, 3),
                "p-value": round(step.pvalue, 4),
                "Action": step.action,
            }
        )

    steps_df = _pd.DataFrame(steps_data)

    mo.vstack(
        [
            mo.md("### SCM Steps"),
            mo.ui.table(steps_df) if not steps_df.empty else mo.md("*(no steps completed)*"),
            mo.md(f"**Final model covariates:** {scm_result.retained_covariates}"),
        ]
    )
    return _pd, steps_data, steps_df


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openpkpd.plots.covariate import covariate_forest_plot

    return covariate_forest_plot, matplotlib, plt


@app.cell
def _(covariate_forest_plot, scm_result):
    try:
        fig_forest = covariate_forest_plot(
            scm_result,
            title="SCM — Covariate Forest Plot",
        )
        fig_forest
    except Exception as e:
        print(f"Forest plot: {e}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Manually Adding a Covariate

        Once you identify a covariate relationship, add it directly to the
        `$PK` block:

        ```python
        # Allometric weight scaling on CL and V
        pk_with_wt = \"\"\"
        TVCL = THETA(1) * (WT/70)**0.75
        CL   = TVCL * EXP(ETA(1))

        TVV  = THETA(2) * (WT/70)
        V    = TVV * EXP(ETA(2))

        KA   = THETA(3) * EXP(ETA(3))
        \"\"\"

        built_wt = (
            ModelBuilder()
            ...
            .pk(pk_with_wt)
            .estimation(method="FOCE", interaction=True)
            .build()
        )
        result_wt = built_wt.fit()
        ```

        ## 6. Model Comparison

        Use the likelihood ratio test to formally compare the base vs covariate
        model:

        ```python
        from openpkpd.inference import lrt

        lrt_result = lrt(base_result, covariate_result, df=1)
        print(lrt_result)
        # ΔOFV = x.xx, p = 0.xxx → significant / not significant
        ```

        See `10_inference_bootstrap.py` for full model comparison workflow.

        ## Summary

        | Task | API |
        |------|-----|
        | Define candidate effects | `CovariateRelationship(param, cov, effect, ref)` |
        | Power effect | `CovariateEffect.POWER` |
        | Linear effect | `CovariateEffect.LINEAR` |
        | Exponential effect | `CovariateEffect.EXPONENTIAL` |
        | Categorical effect | `CovariateEffect.CATEGORICAL` |
        | Run SCM | `SCMEngine(base_builder, ..., candidates).run()` |
        | Read results | `scm_result.retained_covariates`, `.steps` |
        | Forest plot | `covariate_forest_plot(scm_result)` |
        """
    )
    return


if __name__ == "__main__":
    app.run()
