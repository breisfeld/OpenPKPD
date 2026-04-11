# OpenPKPD

```{image} OpenPKPD_logo.*
:alt: OpenPKPD
:width: 320px
:align: center
```

**Open-source Python toolkit for population PK/PD analysis.**

OpenPKPD provides a pure-Python estimation engine that reads NONMEM control
streams, accepts the same data format, and produces NONMEM 7.x-compatible output
files — with no commercial license required.


::::{grid} 3
:::{grid-item-card} 🚀 Getting Started
:link: getting_started/index
:link-type: doc

Install OpenPKPD and run your first model in 5 minutes.
:::
:::{grid-item-card} 📖 User Guide
:link: user_guide/index
:link-type: doc

In-depth coverage of models, estimation methods, and diagnostic plots.
:::
:::{grid-item-card} 🔧 API Reference
:link: api/index
:link-type: doc

Full autodoc reference for all public classes and functions.
:::
::::

---

## Highlights

- **Model setup and execution**
  - NONMEM-style datasets, BLQ handling, covariates, control streams, and `ModelBuilder`
  - Python API, CLI, and desktop GUI workflows for the same core engine
- **Estimation and inference**
  - FO, FOCE, FOCEI, and Laplacian as the main mixed-effects workflows
  - SAEM, IMP, IMPMAP, `BAYES(Laplace)`, and nonparametric support-point estimation
- **Model families**
  - analytical PK subroutines, numerical ODE routes, and DDE support
  - population PK, PK/PD, PBPK, TMDD, tumor growth, TTE, and categorical/count PD
- **Simulation, diagnostics, and analysis**
  - GOF plots, residual diagnostics, ETA panels, VPC/pcVPC, NPC, NPDE
  - NCA, sparse-sampling NCA, bioequivalence, bootstrap, and optimal design
- **Outputs and integrations**
  - NONMEM-compatible `.lst/.ext/.phi/.cov/.cor` outputs and `$TABLE`
  - HTML/PDF reporting, SBML import, and optional distributed/JIT/Bayesian extras
- **Examples and validation**
  - 34 shipped examples, marimo notebook support, and broad automated test coverage

For the current support classification across estimation, analysis, and workflow
surfaces, start with [`user_guide/validation_matrix.md`](user_guide/validation_matrix.md).

```{toctree}
:maxdepth: 2
:hidden:
:caption: Contents

getting_started/index
user_guide/index
examples/index
api/index
contributing
building_installers
developer_performance
changelog
```
