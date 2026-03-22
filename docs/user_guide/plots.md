# Diagnostic Plots

OpenPKPD includes a `plots` subpackage with publication-quality diagnostic
figures. All plots require `matplotlib`.

## Installation

```bash
uv add "OpenPKPD[plots]"
# or
pip install "OpenPKPD[plots]"
```

## Quick example

```python
from openpkpd.plots.diagnostics import compute_diagnostics
from openpkpd.plots.gof import diagnostic_panel
from openpkpd.plots.pk import spaghetti_plot
from openpkpd.plots.eta import eta_histograms

# Run your model first
result = built.fit()

# Build the diagnostics DataFrame (all residuals, predictions, ETAs)
diag_df = compute_diagnostics(built.population_model, result)

# 2×3 GOF panel
fig = diagnostic_panel(diag_df, title="Theophylline FOCE — GOF")
fig.savefig("gof_panel.png", dpi=150)

# Individual concentration-time profiles
fig2 = spaghetti_plot(diag_df, log_y=True)

# ETA histograms with N(0, ω) overlay
fig3 = eta_histograms(diag_df, result.omega_final)
```

## `compute_diagnostics()`

All plot functions take a diagnostics `DataFrame` produced by:

```python
from openpkpd.plots.diagnostics import compute_diagnostics

diag_df = compute_diagnostics(population_model, result)
```

### Columns returned

| Column | Description |
|--------|-------------|
| `ID` | Subject identifier |
| `TIME` | Observation time |
| `DV` | Observed value |
| `PRED` | Population prediction (ETA = 0) |
| `IPRED` | Individual prediction (ETA = η̂ᵢ) |
| `RES` | `DV − PRED` |
| `IRES` | `DV − IPRED` |
| `WRES` | Weighted residual |
| `IWRES` | Individual weighted residual |
| `CWRES` | Conditional weighted residual (via C_i Cholesky) |
| `ETA1`, `ETA2`, … | Subject-level empirical Bayes estimates |
| Any covariates | Passed through from the dataset |

Only rows with `EVID=0` and `MDV=0` are included.

---

## GOF plots (`plots/gof.py`)

### `diagnostic_panel()`

Combined 2×3 panel of all six GOF plots:

```python
from openpkpd.plots.gof import diagnostic_panel

fig = diagnostic_panel(diag_df, title="Model 1 — GOF", figsize=(14, 10))
fig.savefig("gof.png", dpi=150)
```

### Individual GOF functions

```python
from openpkpd.plots.gof import (
    dv_vs_ipred, dv_vs_pred,
    cwres_vs_time, cwres_vs_pred,
    cwres_qq, abs_iwres_vs_ipred,
)

fig = dv_vs_ipred(diag_df, log_scale=True)
fig = cwres_vs_time(diag_df)
fig = cwres_qq(diag_df)
```

All six functions accept an optional `ax` argument and return a
`matplotlib.figure.Figure`.

---

## PK plots (`plots/pk.py`)

```python
from openpkpd.plots.pk import concentration_time, spaghetti_plot, mean_profile

# Individual profiles (one line per subject)
fig = spaghetti_plot(diag_df, log_y=True, mean_overlay=True)

# Single subject
fig = concentration_time(diag_df, individual=3, log_y=True)

# Population mean ± SD band
fig = mean_profile(diag_df, log_y=False, sd_band=True)
```

---

## PD plots (`plots/pd.py`)

```python
from openpkpd.plots.pd import (
    effect_time, emax_curve, hysteresis_loop, pd_individual
)

# Effect vs time
fig = effect_time(diag_df, effect_col="DV")

# E vs C with Emax overlay
fig = emax_curve(diag_df, conc_col="IPRED", effect_col="DV",
                 emax=result.theta_final[4],
                 ec50=result.theta_final[5])

# Clockwise/counter-clockwise hysteresis loop
fig = hysteresis_loop(diag_df, conc_col="IPRED", effect_col="DV",
                      color_by_time=True)

# Per-subject dual PK + PD panels
fig = pd_individual(diag_df, conc_col="IPRED", effect_col="DV", n_cols=3)
```

---

## ETA plots (`plots/eta.py`)

```python
from openpkpd.plots.eta import eta_histograms, eta_pairs, eta_vs_covariate

# Histograms with N(0, ω_kk) overlay
fig = eta_histograms(diag_df, result.omega_final, overlay_normal=True)

# Pairwise scatter matrix
fig = eta_pairs(diag_df)

# ETA vs continuous covariate
fig = eta_vs_covariate(diag_df, covariate="WT", eta_col="ETA1")

# ETA vs categorical covariate (box plots)
fig = eta_vs_covariate(diag_df, covariate="SEX", eta_col="ETA1",
                       categorical=True)
```

---

## Model performance plots (`plots/model_perf.py`)

```python
from openpkpd.plots.model_perf import ofv_history, vpc

# OFV convergence plot
fig = ofv_history(result, log_scale=False)

# Visual predictive check (Monte Carlo simulation)
fig = vpc(diag_df, built.population_model, result, n_sim=200,
          percentiles=(5, 50, 95))
```

---

## NCA plots (`plots/nca.py`)

```python
from openpkpd.plots.nca import (
    nca_distributions,
    nca_boxplot,
    nca_profile_plot,
    dose_proportionality_plot,
)
```

| Function | Input | Description |
|---|---|---|
| `nca_distributions(nca_df)` | NCA results DataFrame | Histogram grid for C0 (when available), Cmax, AUC_last, AUC_inf, t½, CL/F, Vz/F |
| `nca_boxplot(nca_df)` | NCA results DataFrame | Side-by-side boxplots, optional `group_col` stratification |
| `nca_profile_plot(times, conc, nca_params)` | Raw C-t data + `NCAParameters` | C-t profile annotated with Cmax, Tmax, AUC fill, and λz regression |
| `dose_proportionality_plot(nca_df)` | NCA results DataFrame with dose column | Log-log exposure vs dose with slope-1 reference and power model |

See the [NCA guide](nca.md#nca-visualization) for detailed usage examples.

---

## Saving figures

Every plot function returns a `matplotlib.figure.Figure`. Save with:

```python
fig.savefig("plot.png", dpi=150, bbox_inches="tight")
fig.savefig("plot.pdf")          # Vector PDF for publication
```

## Style

All plots use a consistent IBM colorblind-friendly palette and minimal grid
style. To override:

```python
import matplotlib.pyplot as plt
with plt.rc_context({"figure.figsize": (8, 6)}):
    fig = diagnostic_panel(diag_df)
```
