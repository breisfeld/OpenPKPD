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

```{image} _static/openpkpd_capabilities.svg
:alt: OpenPKPD capability overview
:width: 100%
:align: center
```


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

- Python API, control streams, CLI, and desktop GUI on one estimation engine
- FO/FOCE/FOCEI/Laplacian plus SAEM, IMP/IMPMAP, `BAYES(Laplace)`, NUTS, and nonparametric estimation
- analytical ADVAN routes, numerical ODE solvers, DDE support, and broad PK/PD model families
- simulation, diagnostics, NCA, bootstrap, optimal design, and NONMEM-compatible reporting
- 34 shipped examples, marimo notebooks, and broad automated validation against exact references and external tools

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
