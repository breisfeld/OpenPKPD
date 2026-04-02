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

- **Estimation methods**: primary FO/FOCE/FOCEI/Laplacian workflows plus secondary SAEM, IMP/IMPMAP, `BAYES(Laplace)`, and nonparametric paths
- **PK subroutines**: ADVAN1–4, ADVAN11/12, ODE routes, and DDE support
- **NM-TRAN compiler**: `$PK` / `$ERROR` / `$DES` blocks → Python callables
- **NONMEM-compatible output**: `.lst`, `.ext`, `.phi`, `.cov`, `.cor`, `$TABLE`
- **Python API**: `ModelBuilder` — no `.ctl` file required
- **Diagnostic plots**: GOF, PK, PD, ETA panels (matplotlib, optional)
- **Desktop GUI**: a workspace / project / scenario shell with Dashboard, Data, Model, Fit, NCA, Results, Plots, Diagnostics, Covariate, and Advanced workflows; BLQ/M3 handling, interactive GOF subject highlighting, VPC stratification and pcVPC
- **Examples and tests**: runnable examples plus unit/integration/regression coverage

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
