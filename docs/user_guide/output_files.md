# Output Files

OpenPKPD writes the same output files as NONMEM 7.x. All files are created in
the current working directory (or the directory of the `.ctl` file when using
the CLI).

## File overview

| File | Description |
|------|-------------|
| `.lst` | Estimation log — mirrors the NONMEM `.lst` report |
| `.ext` | Parameter estimates at each outer iteration |
| `.phi` | Individual empirical Bayes ETAs (post-hoc estimates) |
| `.cov` | Covariance matrix of parameter estimates |
| `.cor` | Correlation matrix |
| `sdtab` | Standard diagnostic table (`$TABLE` default) |
| `patab` | Parameter table |

## `.lst` file

The list file contains:
- Parsed control stream echo
- Estimation method and options
- OFV history (one line per outer iteration)
- Final parameter estimates with standard errors
- Covariance step output (if enabled)

```
 **ESTIMATION STEP OMITTED:  NO
 **COVARIANCE STEP OMITTED:  NO

 #TBLN:      1
 #METH: First Order Conditional Estimation with Interaction

 ESTIMATION STEP COMPLETED

 FINAL PARAMETER ESTIMATE
 THETA - VECTOR OF FIXED EFFECTS PARAMETERS   *TH:
    TH 1      TH 2      TH 3
    1.50E+00  8.00E-02  3.00E+01
```

## `.ext` file

Tab-delimited file with parameter values at each iteration. Columns:

```
ITERATION  THETA1  THETA2  ...  SIGMA(1,1)  OBJ
```

Final estimates are on the row with `ITERATION = -1000000000`.

```python
import pandas as pd
ext = pd.read_csv("run001.ext", sep=r"\s+", skiprows=1)
final = ext[ext["ITERATION"] == -1000000000]
```

## `.phi` file

Individual EBE (empirical Bayes estimate) file. One row per subject:

```
ID  ETA1       ETA2       ETA3       OBJ_i
1   0.1234    -0.0456     0.0789    12.345
```

```python
import pandas as pd
phi = pd.read_csv("run001.phi", sep=r"\s+", skiprows=1)
```

Or access directly from Python:

```python
result.post_hoc_etas   # dict {subject_id: np.ndarray}
```

## `.cov` and `.cor` files

Symmetric matrices of parameter covariance / correlation written in
NONMEM-compatible space-delimited format.

```python
result.covariance_result.cov_matrix   # np.ndarray
result.covariance_result.cor_matrix   # np.ndarray
```

## `$TABLE` output

`$TABLE` blocks request specific columns in a diagnostic table file:

```
$TABLE ID TIME DV PRED IPRED CWRES IWRES NOAPPEND NOPRINT FILE=sdtab
```

In OpenPKPD:

```python
# Generated automatically after fitting; access via diagnostics:
from openpkpd.plots.diagnostics import compute_diagnostics

diag_df = compute_diagnostics(built.population_model, result)
diag_df.to_csv("sdtab.csv", index=False)
```

## CDISC ADPPK output

`write_cdisc_adppk()` writes a narrow-scope CDISC ADPPK-style CSV suitable for
downstream processing and regulatory data exchange.  The file contains four row
types, identified by the `DTYPE` column:

| DTYPE | Content |
|-------|---------|
| `OBSERVATION` | One row per observed concentration (PARAMCD=CONC, AVAL=DV) |
| `THETA` | Fixed-effect estimates (THETA1, THETA2, …) |
| `OMEGA` | Variance/covariance elements, lower-triangular (OMEGA(i,j)) |
| `SIGMA` | Residual-variance elements (SIGMA(i,j)) |
| `ETA` | Post-hoc EBEs per subject (ETA1, ETA2, …) |

```python
from openpkpd.output import write_cdisc_adppk

write_cdisc_adppk(
    result,          # EstimationResult
    dataset,         # NONMEMDataset used during fitting
    "adppk.csv",
    study_id="STUDY001",
    avalu="ng/mL",
)
```

Output columns: `STUDYID, USUBJID, PARAMCD, PARAM, AVAL, AVALU, DTYPE`.

> **Note:** This is a CSV-format ADPPK file. Full SDTM/ADaM validation and
> XPT transport format are outside the scope of OpenPKPD.

## Accessing results programmatically

All output data is available on the `EstimationResult` object without
touching any files:

```python
result.ofv              # Scalar OFV
result.theta_final      # np.ndarray
result.omega_final      # np.ndarray (full matrix)
result.sigma_final      # np.ndarray (full matrix)
result.ofv_history      # list[float]
result.post_hoc_etas    # dict {id: np.ndarray}
result.eta_shrinkage    # np.ndarray (after compute_shrinkage())
result.converged        # bool
result.warnings         # list[str]
print(result.summary()) # One-line text summary
```
