"""
OpenPKPD — Notebook 11: Advanced Topics

Covers:
  - Prior information (PriorAugmentedModel)
  - Optimal experimental design (PFIMEngine)
  - Mixture models
  - PBPK (physiologically based PK)
  - Parallel computation back-ends
  - SBML import/export
  - Control stream workflow (ControlStream, run_model)
  - HTML/PDF report generation
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — Advanced Topics")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Advanced Topics

        This notebook covers the more specialised capabilities of OpenPKPD
        that go beyond the standard population PK fitting workflow.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Prior Information

        Bayesian prior information can be added to any model via
        `PriorAugmentedModel`.  This adds a penalty term to the OFV
        that penalises parameters deviating from the prior means.

        The penalty uses the $\Omega_p$ (prior covariance) for $\theta$
        and $\Omega_p^{-1}$ for $\Omega$ (from $\Omega \sim \text{IW}$):

        $$\text{OFV}_{\text{aug}} = \text{OFV}_{\text{data}}
          + (\theta - \theta_p)^\top \Sigma_p^{-1} (\theta - \theta_p)
          + \text{tr}[\Omega_p \Omega^{-1}] - \ln|\Omega_p \Omega^{-1}|$$

        ```python
        from openpkpd.prior import PriorSpec, PriorAugmentedModel

        prior = PriorSpec(
            theta_prior=np.array([1.5, 0.08, 30.0]),   # prior means for THETA
            theta_prior_cov=np.diag([0.5, 0.01, 100]), # prior covariance
            omega_prior=np.diag([0.5, 0.3, 0.3]),      # prior mean for OMEGA
            omega_prior_cov=np.diag([0.1, 0.1, 0.1]),
            nwpri=20,                                   # informational NWPRI count
        )

        augmented = PriorAugmentedModel(population_model, prior)
        result = built_with_augmented.fit()
        ```
        """
    )
    return


@app.cell
def _():
    from openpkpd.prior import PriorSpec

    prior = PriorSpec(
        theta_prior=np.array([1.5, 0.08, 30.0]),
        theta_prior_cov=np.diag([0.25, 0.004, 225.0]),
        omega_prior=np.array([0.5, 0.3, 0.3]),
        omega_prior_cov=np.diag([0.05, 0.05, 0.05]),
        nwpri=15,
    )
    print("Prior specification:")
    print(f"  θ prior means:  {prior.theta_prior}")
    print(f"  θ prior std:    {np.sqrt(np.diag(prior.theta_prior_cov)).round(3)}")
    print(f"  Ω prior diag:   {prior.omega_prior}")
    print(f"  NWPRI count:    {prior.nwpri}")
    return (prior,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Optimal Experimental Design (OED)

        `PFIMEngine` implements the PFIM (Population Fisher Information Matrix)
        approach for designing optimal sampling schedules that maximise
        information about the population parameters.

        The Fisher Information Matrix (FIM) is:

        $$\mathcal{I}(\Psi) = \sum_{i=1}^{N} \left[\frac{\partial^2 \ell_i}{\partial \Psi^2}\right]$$

        Common design criteria:
        - **D-optimal**: maximise $|\mathcal{I}|$ → minimise joint ellipsoid volume
        - **ED-optimal**: expected D-optimality, integrating over parameter uncertainty
        """
    )
    return


@app.cell
def _(np):
    import io
    import pandas as pd
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

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
"""
    _ds = NONMEMDataset.from_dataframe(pd.read_csv(io.StringIO(_THEO)))

    built_oed = (
        ModelBuilder()
        .problem("Theophylline OED")
        .dataset(_ds)
        .subroutines(advan=2, trans=2)
        .pk("""
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
""")
        .error("Y = F*(1 + EPS(1))")
        .theta([1.5, 0.08, 30.0])
        .omega([0.3, 0.2, 0.2])
        .sigma(0.1)
        .estimation(method="FO", maxeval=200)
        .build()
    )

    result_oed = built_oed.fit()
    return NONMEMDataset, ModelBuilder, _THEO, _ds, built_oed, io, pd, result_oed


@app.cell
def _(built_oed, np):
    # Access the PFIMEngine through BuiltModel.design()
    pfim_engine = built_oed.design(sampling_times=np.linspace(0.5, 24, 10))

    # Evaluate FIM at a specific design
    sampling_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    fim = pfim_engine.compute_fim(sampling_times)
    print("FIM trace (sum of diagonal):", round(float(np.trace(fim)), 4))
    print("FIM det (D-criterion):", f"{float(np.linalg.det(fim)):.4e}")
    return fim, pfim_engine, sampling_times


@app.cell
def _(pfim_engine):
    # Optimise design: find the best 6 sampling times in [0, 24]
    try:
        opt_design = pfim_engine.optimize_design(n_samples=6, t_min=0.0, t_max=24.0)
        print(opt_design.summary())
    except Exception as e:
        print(f"Design optimisation: {e}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Mixture Models

        Mixture models describe populations with distinct sub-populations
        (e.g., poor vs. extensive metabolisers):

        $$p(y_i \mid \Psi) = \sum_{k=1}^{K} \pi_k \cdot p_k(y_i \mid \Psi_k)$$

        ```python
        from openpkpd.mixture import MixtureModel

        mixture = MixtureModel(
            n_components=2,
            base_builder=base_builder,
        )
        result = mixture.fit(dataset)
        print(result.mixing_proportions)   # e.g. [0.75, 0.25]
        print(result.component_params)     # params for each component
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. PBPK — Physiologically Based PK

        `PBPKModel` implements a multi-organ physiologically based PK model.
        Compartments are defined by measured physiological volumes and blood
        flows, with tissue-to-plasma partition coefficients ($K_p$).

        `FiveOrganPBPK` provides a ready-to-use template with liver, kidney,
        gut, lung, and rest-of-body compartments:

        ```python
        from openpkpd.pk.pbpk import FiveOrganPBPK

        pbpk = FiveOrganPBPK()

        params = {
            "Kp_liver":  5.0,
            "Kp_kidney": 3.0,
            "Kp_gut":    2.0,
            "Kp_lung":   4.0,
            "Kp_rest":   1.5,
            "CLint":     20.0,  # hepatic intrinsic clearance (L/h)
            "BP":        0.9,   # blood:plasma ratio
            "fu":        0.05,  # fraction unbound in plasma
        }

        sol = pbpk.solve(params, dose_events, obs_times)
        ```

        PBPK models are useful for:
        - Predicting PK in special populations (paediatric, renal impairment)
        - First-in-human dose prediction from preclinical data
        - Drug-drug interaction mechanistic modelling
        """
    )
    return


@app.cell
def _(np, plt):
    from openpkpd.pk.pbpk import FiveOrganPBPK
    from openpkpd.data.event_processor import DoseEvent

    pbpk = FiveOrganPBPK()

    def des_callable(t, a, pk_params, theta=None, eta=None):
        q_lung = pk_params.get("Q_lung", 350.0)
        q_liver = pk_params.get("Q_liver", 90.0)
        q_kidney = pk_params.get("Q_kidney", 72.0)
        q_gut = pk_params.get("Q_gut", 60.0)

        v_lung = max(pk_params.get("V_lung", 0.5), 1e-12)
        v_liver = max(pk_params.get("V_liver", 1.8), 1e-12)
        v_kidney = max(pk_params.get("V_kidney", 0.3), 1e-12)
        v_gut = max(pk_params.get("V_gut", 1.0), 1e-12)
        v_central = max(pk_params.get("V_central", 5.0), 1e-12)

        kp_lung = pk_params.get("Kp_lung", 2.5)
        kp_liver = pk_params.get("Kp_liver", 8.0)
        kp_kidney = pk_params.get("Kp_kidney", 4.0)
        kp_gut = pk_params.get("Kp_gut", 3.0)

        cl_liver = pk_params.get("CL_liver", 15.0)
        cl_kidney = pk_params.get("CL_kidney", 8.0)

        c_lung = a[0] / v_lung
        c_liver = a[1] / v_liver
        c_kidney = a[2] / v_kidney
        c_gut = a[3] / v_gut
        c_central = a[4] / v_central

        ret_lung = q_lung * c_lung / kp_lung
        ret_liver = q_liver * c_liver / kp_liver
        ret_kidney = q_kidney * c_kidney / kp_kidney
        ret_gut = q_gut * c_gut / kp_gut
        q_total = q_lung + q_liver + q_kidney + q_gut

        return [
            q_lung * (c_central - c_lung / kp_lung),
            q_liver * (c_central - c_liver / kp_liver) - cl_liver * c_liver,
            q_kidney * (c_central - c_kidney / kp_kidney) - cl_kidney * c_kidney,
            q_gut * (c_central - c_gut / kp_gut),
            -q_total * c_central + ret_lung + ret_liver + ret_kidney + ret_gut,
        ]

    pbpk_params = {
        "Kp_liver": 5.0,
        "Kp_kidney": 3.0,
        "Kp_gut": 2.0,
        "Kp_lung": 4.0,
        "Kp_rest": 1.5,
        "CLint": 20.0,
        "BP": 0.9,
        "fu": 0.05,
    }
    t_pbpk = np.linspace(0.01, 24, 200)
    dose_iv = [DoseEvent(time=0.0, amount=100.0, compartment=1, rate=0.0)]

    sol_pbpk = pbpk.solve(pbpk_params, dose_iv, t_pbpk, des_callable=des_callable)

    fig_pbpk, ax_pbpk = plt.subplots(figsize=(9, 4))
    ax_pbpk.semilogy(t_pbpk, sol_pbpk.ipred, lw=2, label="Plasma (PBPK)")
    ax_pbpk.set_xlabel("Time (h)")
    ax_pbpk.set_ylabel("Concentration (mg/L)")
    ax_pbpk.set_title("Five-Organ PBPK — Predicted Plasma Concentration")
    ax_pbpk.legend()
    fig_pbpk.tight_layout()
    fig_pbpk
    return (
        DoseEvent,
        FiveOrganPBPK,
        ax_pbpk,
        dose_iv,
        des_callable,
        fig_pbpk,
        pbpk,
        pbpk_params,
        sol_pbpk,
        t_pbpk,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Parallel Computation

        For computationally intensive tasks (bootstrap, SSE, SAEM with many
        chains), OpenPKPD supports parallel back-ends:

        ```python
        from openpkpd.parallel import ParallelBackend

        # Dask (recommended for local clusters)
        with ParallelBackend("dask", n_workers=8) as backend:
            result = sse_engine.run(n_replicates=500, backend=backend)

        # Ray (for distributed clusters)
        with ParallelBackend("ray") as backend:
            result = boot_engine.run(n_bootstrap=1000, backend=backend)
        ```

        Install extras:
        ```bash
        uv sync --extra cluster   # Dask
        pip install ray           # Ray
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 6. SBML Import / Export

        OpenPKPD can import SBML (Systems Biology Markup Language) models and
        convert them to OpenPKPD ODE models:

        ```python
        from openpkpd.io import SBMLReader, SBMLWriter

        # Import from SBML
        reader = SBMLReader()
        pop_model = reader.load("model.xml")

        # Export a fitted model to SBML
        writer = SBMLWriter()
        writer.write(pop_model, result, "fitted_model.xml")
        ```

        ## 7. Control Stream Workflow

        For users migrating from NONMEM, OpenPKPD can run `.ctl` files directly:

        ```python
        from openpkpd.parser.control_stream import ControlStream
        from openpkpd.cli.runner import run_model

        # Parse a control stream
        cs = ControlStream.from_file("model.ctl")
        print(cs.records)

        # Run a control stream file
        result = run_model("model.ctl", verbose=True)
        ```

        ## 8. HTML / PDF Report Generation

        Generate publication-ready HTML or PDF reports from estimation results:

        ```python
        from openpkpd import write_html_report, write_pdf_report

        write_html_report(result, output="report.html")
        write_pdf_report(result, output="report.pdf")  # requires LaTeX or weasyprint
        ```

        The HTML report includes:
        - Parameter table with SE and 95% CI
        - GOF plots (if matplotlib is available)
        - Covariance matrix (if `.covariance()` was called)
        - OFV history plot
        - ETA shrinkage table
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 9. HTML Report Example
        """
    )
    return


@app.cell
def _(built_oed, result_oed):
    from openpkpd import estimation_result_to_html

    html_str = estimation_result_to_html(result_oed, built_oed.params)
    print(f"Report HTML length: {len(html_str)} chars")
    print(html_str[:500] + "...")
    return (estimation_result_to_html, html_str)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        | Feature | Module / API |
        |---------|-------------|
        | Prior information | `prior.PriorSpec`, `prior.PriorAugmentedModel` |
        | Optimal design | `built.design()`, `PFIMEngine.optimize_design()` |
        | Mixture models | `mixture.MixtureModel` |
        | PBPK | `pk.pbpk.FiveOrganPBPK` |
        | Parallel | `parallel.ParallelBackend("dask")` |
        | SBML I/O | `io.SBMLReader`, `io.SBMLWriter` |
        | Control stream | `ControlStream.from_file()`, `run_model()` |
        | HTML report | `write_html_report(result, output)` |
        | PDF report | `write_pdf_report(result, output)` |

        ---

        *This completes the OpenPKPD marimo notebook library.  Return to the
        [quickstart](01_quickstart.py) for a compact end-to-end example.*
        """
    )
    return


if __name__ == "__main__":
    app.run()
