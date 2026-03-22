# Non-Compartmental Analysis (NCA)

OpenPKPD provides a full NCA engine for dense and sparse sampling designs,
plus CDISC PP domain export for regulatory submissions.

Current validation status for the dense-profile NCA path is summarized in
[Validation notes](validation.md), including the analytic and
reference-workflow checks used in the D2 milestone.

---

## Standard NCA

`NCAEngine.compute_subject()` accepts a concentration–time profile and returns
an `NCAParameters` object with all standard endpoints.

For IV infusion profiles, pass `route="infusion"` and the known
`infusion_duration=...` so `MRT` is reported using the standard
infusion-adjusted definition (`AUMC_inf / AUC_inf - Tinf / 2`).

```python
import numpy as np
from openpkpd.nca import NCAEngine

engine = NCAEngine()

times = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 12, 24])
conc  = np.array([0, 0.74, 1.72, 2.54, 2.54, 2.02, 1.44, 0.88, 0.35])

params = engine.compute_subject(
    times=times,
    conc=conc,
    dose=4.02,
    subject_id=1,
    route="oral",
)

print(f"Cmax  = {params.cmax:.3f}")
print(f"Tmax  = {params.tmax:.2f} h")
print(f"AUCt  = {params.auc_last:.2f} h·mg/L")
print(f"AUCinf= {params.auc_inf:.2f} h·mg/L")
print(f"t½    = {params.t_half:.2f} h")
print(f"CL/F  = {params.cl_f:.4f} L/h")
```

For a full dataset, use `compute_dataset()`:

```python
nca_df = engine.compute_dataset(
    df,
    dose_col="AMT",
    time_col="TIME",
    conc_col="DV",
    id_col="ID",
    route="oral",
)

# For IV infusions with a known 0.25 h infusion duration:
# nca_df = engine.compute_dataset(df, route="infusion", infusion_duration=0.25)
```

---

## Sparse Sampling NCA

When subjects have only 2–5 blood samples per occasion, classical NCA
produces unreliable AUC estimates because the concentration–time profile
is poorly characterised.  `SparseNCAEngine` uses a fitted population model
to reconstruct a dense predicted profile for each subject, then applies
standard NCA to the predicted profile.

### Algorithm

1. **ETA optimisation** — minimise the penalised individual log-likelihood
   over the sparse observations using `scipy.optimize.minimize` (L-BFGS-B).
2. **Dense profile prediction** — predict concentrations at a 200-point
   time grid (0 → last observation time) using the optimised ETAs.
3. **Standard NCA** — pass the dense predicted profile to `NCAEngine`.

### Usage

```python
from openpkpd.nca.sparse import SparseNCAEngine

# population_model must have .theta, .omega, .sigma attributes
engine = SparseNCAEngine(population_model, dense_times=None)  # auto 200-pt grid

# Single subject
params = engine.compute_subject(
    subject_id=1,
    sparse_times=np.array([1.0, 4.0, 12.0]),
    sparse_conc=np.array([1.81, 1.34, 0.60]),
    dose=100.0,
    route="IV",
)

# Whole dataset (long-format DataFrame)
results_df = engine.compute_dataset(sparse_df, dose=100.0, route="IV")
```

### Custom dense time grid

```python
import numpy as np
dense_t = np.linspace(0, 24, 500)
engine = SparseNCAEngine(population_model, dense_times=dense_t)
```

---

## CDISC PP Domain Export

`to_cdisc_pp()` converts an NCA results DataFrame to the CDISC SEND/SDTM
**PP domain** format, which is required for regulatory submissions.

### Supported PARAMCD codes

| Internal key | PARAMCD | PARAM |
|-------------|---------|-------|
| `c0` | `C0` | Concentration at Time Zero |
| `cmax` | `CMAX` | Maximum Observed Concentration |
| `tmax` | `TMAX` | Time of Cmax |
| `auc_last` | `AUCLST` | AUC from Time Zero to Last |
| `auc_inf` | `AUCIFO` | AUC from Time Zero to Infinity |
| `t_half` | `THALF` | Half-Life |
| `lambda_z` | `LAMZ` | Terminal Elimination Rate Constant |
| `cl_f` | `CLF` | Apparent Clearance |
| `vz_f` | `VZF` | Apparent Volume of Distribution |
| `mrt` | `MRT` | Mean Residence Time |

`c0` is populated for IV bolus profiles. When a positive time-zero sample is not
observed, OpenPKPD back-extrapolates `c0` from the first two positive samples
when the initial decline is log-linear.

### Usage

```python
from openpkpd.nca import NCAEngine
from openpkpd.nca.cdisc_pp import to_cdisc_pp

# 1. Compute NCA for all subjects
nca_df = NCAEngine().compute_dataset(df, dose_col="AMT", route="oral")

# 2. Export to CDISC PP domain
pp_df = to_cdisc_pp(
    nca_df,
    study_id="STUDY001",
    domain="PP",
    usubjid_col="subject_id",   # column in nca_df that holds the subject ID
)

# 3. Write to CSV
pp_df.to_csv("pp_domain.csv", index=False)
```

The output DataFrame has columns:
`STUDYID`, `USUBJID`, `DOMAIN`, `PARAMCD`, `PARAM`, `AVAL`, `DTYPE`.

Parameters absent from a row are silently skipped (e.g. `auc_inf` is omitted
when the terminal phase cannot be estimated).

---

## NCA Visualization

`plots/nca.py` provides four ready-made figures for NCA results.  All require
`matplotlib` (`openpkpd[plots]`).

### Distribution histograms

Grid of histograms for key NCA parameters across subjects, with median overlay:

```python
from openpkpd.plots.nca import nca_distributions

nca_df = NCAEngine().compute_dataset(df, route="oral")
fig = nca_distributions(nca_df, title="Study A — NCA distributions")
fig.savefig("nca_distributions.png", dpi=150)
```

Default parameters plotted: C0 (when available), Cmax, AUC_last, AUC_inf, t½,
CL/F, Vz/F.
Pass `params=[...]` to customise.

### Boxplots (optionally stratified)

```python
from openpkpd.plots.nca import nca_boxplot

# All subjects combined
fig = nca_boxplot(nca_df)

# Stratified by dose group
fig = nca_boxplot(nca_df, group_col="DOSE")
```

### Individual C-t profile with NCA annotations

Plots observed concentrations with Cmax line, Tmax line, AUC fill, and the
terminal λz regression overlaid:

```python
from openpkpd.plots.nca import nca_profile_plot

params = NCAEngine().compute_subject(times, conc, dose=100.0, route="oral")
fig = nca_profile_plot(times, conc, params, log_y=True)
```

### Dose proportionality (log-log)

```python
from openpkpd.plots.nca import dose_proportionality_plot

# nca_df must contain a dose column (e.g. "AMT" or "DOSE")
fig = dose_proportionality_plot(nca_df, metric="auc_inf", dose_col="DOSE")
```

Overlays a slope-1 reference line and, when ≥ 3 dose levels are present, a
fitted power model with slope and R² annotation.

### GUI NCA workflow

When NCA is run via the desktop GUI (**NCA workflow → Run NCA**), the service
automatically generates and saves the distributions plot and boxplot as PNG
artifacts alongside the CSV summary.  These appear in the **Results** artifact
panel with `plot_type` = `nca_distributions` and `nca_boxplot`.

The GUI results preview mirrors the saved CSV summary, so newly populated
fields such as `c0` appear there automatically when present.
