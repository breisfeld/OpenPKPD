"""
OpenPKPD — Notebook Library Index

Navigate to any notebook to explore a specific capability.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — Notebook Library")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # OpenPKPD — Notebook Library

        An interactive, self-contained documentation library for the
        OpenPKPD population PK/PD toolkit.

        ## Quick Start

        ```bash
        # Launch a notebook
        just notebook notebooks/01_quickstart.py

        # Launch the full library
        just notebooks
        ```

        ## Notebooks

        | # | Notebook | Topics |
        |---|----------|--------|
        | 01 | [Quickstart](01_quickstart.py) | ModelBuilder, FO fitting, spaghetti plot |
        | 02 | [Data Handling](02_data_handling.py) | NONMEMDataset, BLQ, IOV, covariates |
        | 03 | [Estimation Methods](03_estimation_methods.py) | FO, FOCE, SAEM, IMP, Bayesian |
        | 04 | [Simulation, VPC & NPDE](04_simulation_vpc_npde.py) | SimulationEngine, VPC, NPDE, NPC |
        | 05 | [NCA](05_nca.py) | NCAEngine, bioequivalence, crossover |
        | 06 | [PK Subroutines](06_pk_subroutines.py) | ADVAN1–13, transit, parallel, DDE |
        | 07 | [PK/PD Models](07_pkpd_models.py) | Emax, IDR, Ce, TMDD, TTE, Markov |
        | 08 | [Diagnostics & Plots](08_diagnostics_plots.py) | GOF, ETA, spaghetti, VPC plots |
        | 09 | [Covariate Modeling](09_covariate_modeling.py) | SCM, power/linear/categorical effects |
        | 10 | [Inference & Bootstrap](10_inference_bootstrap.py) | LRT, AIC/BIC, bootstrap CI |
        | 11 | [Advanced Topics](11_advanced.py) | Prior, OED, PBPK, mixture, SBML, reports |

        ## Installation

        ```bash
        # Install all dependencies (including marimo and matplotlib)
        uv sync --extra notebooks

        # Or install individual extras
        pip install openpkpd[plots]   # matplotlib only
        pip install openpkpd[bayes]   # PyMC Bayesian (recommended for full MCMC)
        ```

        ## About OpenPKPD

        OpenPKPD is an open-source Python reimplementation of the NONMEM population
        PK/PD modelling framework.  It implements:

        - **All ADVAN subroutines** (analytical: ADVAN1–4, 10–12; ODE: ADVAN6, 8, 13)
        - **All estimation methods** (FO, FOCE/FOCEI, SAEM, IMP, Laplacian, nonparametric, Bayesian)
        - **Simulation-based diagnostics** (VPC, NPDE, NPC, SSE)
        - **Non-compartmental analysis** (NCA, bioequivalence)
        - **Covariate modeling** (SCM with power/linear/exponential/categorical effects)
        - **Extended model library** (PKPD, TMDD, DDI, Markov, TTE, count, categorical)
        - **Optimal experimental design** (PFIM-based, D-optimal)
        - **PBPK** (five-organ physiologically based PK)
        - **Publication-ready reports** (HTML/PDF)
        - **SBML import/export**
        - **NONMEM control stream** compatibility
        """
    )
    return


@app.cell
def _():
    import openpkpd

    print(f"OpenPKPD version: {openpkpd.__version__}")
    return (openpkpd,)


if __name__ == "__main__":
    app.run()
