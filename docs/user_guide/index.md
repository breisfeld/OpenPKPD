# User Guide

This guide documents the current end-user surface of OpenPKPD: the Python API,
the NONMEM-style control-stream workflow, the CLI, and the desktop GUI.

The pages below reflect the current source tree as of the 2026-03 documentation
audit. Some advanced capabilities are still covered more deeply in `examples/`
and API docstrings than in narrative documentation, especially for PBPK, DDE,
Bayesian workflows, bootstrap, and IOV-heavy models.

## What this guide covers

- dataset expectations and NONMEM-style inputs
- programmatic modelling with `ModelBuilder`
- control-stream parsing and execution
- currently implemented PK subroutines and estimation methods
- covariance/output/reporting workflows
- the current GUI shell, menus, and workflow map
- comparison and functionality-gap tables updated to match the codebase

```{toctree}
:maxdepth: 2

gui
data_format
model_builder
control_stream
pk_subroutines
estimation_methods
covariance_step
output_files
plots
analysis_tools
analysis_validation_gaps
validation
testing
advanced_pk
nca
comparison
external_validation_benchmarks
citations
```
