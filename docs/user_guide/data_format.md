# Data Format

OpenPKPD reads data from a CSV file (or a pandas DataFrame) in the standard NONMEM format.

## Required columns

| Column | Default name | Description |
|--------|-------------|-------------|
| Subject ID | `ID` | Unique integer identifier per subject |
| Time | `TIME` | Observation or dose time |
| Dependent variable | `DV` | Measured concentration or effect; `0` on dose rows |
| Dose amount | `AMT` | Dose amount on dose rows; `0` on observation rows |
| Event ID | `EVID` | Event type (see table below) |

## Optional columns

| Column | Description |
|--------|-------------|
| `MDV` | Missing DV flag (`1` = ignore this row in likelihood) |
| `RATE` | Infusion rate (`AMT/TIME`); zero means bolus |
| `CMT` | Compartment number for the dose or observation |
| `ADDL` | Additional doses after this one |
| `II` | Interdose interval for `ADDL` doses |
| `SS` | Steady-state flag (`1` = dose achieves SS before this event) |
| Any covariate | e.g. `WT`, `AGE`, `SEX`; accessible in `$PK` as user-defined variables |

## EVID codes

| EVID | Meaning |
|------|---------|
| `0` | Observation (DV is measured) |
| `1` | Dose event |
| `4` | Reset + dose (compartment amounts zeroed before dose) |

## Loading data

```python
from openpkpd.data.dataset import NONMEMDataset

# From a CSV file
ds = NONMEMDataset.from_csv(
    "theo.csv",
    id_col="ID",          # default
    time_col="TIME",      # default
    dv_col="DV",          # default
    amt_col="AMT",        # default
    evid_col="EVID",      # default; set None to auto-generate from AMT
    missing_value=-99,    # rows with DV==-99 set MDV=1
    ignore_char="#",      # skip rows starting with this character
)

# From a pandas DataFrame
import pandas as pd
df = pd.read_csv("theo.csv")
ds = NONMEMDataset.from_dataframe(df)

# Access the underlying DataFrame
ds.df
```

## Column auto-generation

- If `EVID` is absent or `evid_col=None`, OpenPKPD auto-generates it: rows with `AMT > 0` get `EVID=1`, all others `EVID=0`.
- If `MDV` is absent, it is set to `1` on all dose rows (`EVID != 0`).

## Pre-dose convention

An observation at exactly the same time as a dose is treated as a **pre-dose** measurement (it reflects concentrations before the dose is absorbed). Internally this is implemented as `dt > 0` (strict inequality) when computing the ADVAN solution.

## Example dataset (Theophylline)

```
ID,TIME,AMT,DV,EVID,MDV,WT
1,0,4.02,0,1,1,79.6
1,0.27,0,0.74,0,0,79.6
1,0.57,0,1.72,0,0,79.6
...
```

Covariates (`WT` here) are passed through to the model unchanged. You can reference them inside `$PK` blocks:

```
$PK
CL = THETA(1) * (WT/70)**THETA(4) * EXP(ETA(1))
```

## Covariate imputation

Real datasets often have missing covariate values between scheduled visits.
OpenPKPD provides `CovariateImputer` with five strategies:

| Method | Description |
|--------|-------------|
| `locf` | Last-observation-carried-forward per subject (default) |
| `nocb` | Next-observation-carried-backward per subject |
| `mean` | Column mean across all subjects |
| `median` | Column median across all subjects |
| `knn` | k-nearest-neighbours (requires `scikit-learn`) |

```python
from openpkpd.data.impute import CovariateImputer

# Standalone usage
df_imputed = CovariateImputer.fit_transform(df, columns=["WT", "AGE"], method="locf")

# Via NONMEMDataset (returns a new dataset; original unchanged)
ds_imputed = ds.impute_covariates(["WT", "AGE"], method="locf")
ds_imputed.df["WT"].isna().sum()   # 0
```

LOCF is the most common approach for time-varying covariates (e.g. body weight
measured at each visit). For baseline covariates recorded only once, `mean` or
`median` imputation is typically more appropriate.
