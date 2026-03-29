# Migrating from NONMEM

This page maps familiar NONMEM control stream concepts to their OpenPKPD equivalents.

## Control stream → Python API

| NONMEM record | OpenPKPD equivalent |
|---------------|---------------------|
| `$PROBLEM title` | `.problem("title")` |
| `$DATA file.csv IGNORE=#` | `.data("file.csv")` |
| `$INPUT ID TIME AMT DV EVID` | Inferred from column names (auto-mapping) |
| `$SUBROUTINES ADVAN2 TRANS2` | `.subroutines(advan=2, trans=2)` |
| `$PK … $ERROR …` | `.pk("…")` / `.error("…")` |
| `$THETA (0,1.5,20)` | `.theta([(0, 1.5, 20)])` |
| `$OMEGA 0.5 0.3` | `.omega([0.5, 0.3])` |
| `$OMEGA BLOCK(2) 0.4 0.1 0.3` | `.omega([[0.4,0.1],[0.1,0.3]])` |
| `$SIGMA 0.1` | `.sigma(0.1)` |
| `$ESTIMATION METHOD=COND INTER` | `.estimation(method="FOCE", interaction=True)` |
| `$ESTIMATION METHOD=ZERO` | `.estimation(method="FO")` |
| `$ESTIMATION METHOD=SAEM` | `.estimation(method="SAEM")` |
| `$COVARIANCE` | `.covariance()` |

OpenPKPD also supports non-NONMEM FOCE/FOCEI robustness options in the Python
API and control-stream parser, including outer-optimizer selection, fallback
polish optimizers, best-iterate retention, and structured retry controls.
These are OpenPKPD extensions rather than direct NONMEM mappings.

## Estimation method names

| NONMEM `METHOD=` | OpenPKPD `method=` |
|-----------------|-------------------|
| `ZERO` | `"FO"` |
| `COND` | `"FOCE"` |
| `COND INTER` | `"FOCE"` + `interaction=True` |
| `COND LAPLACE` | `"LAPLACIAN"` |
| `SAEM` | `"SAEM"` |
| `IMP` | `"IMP"` |
| `IMPMAP` | `"IMP"` |

## NM-TRAN code blocks

`$PK` and `$ERROR` blocks are written identically to NONMEM. The compiler handles:

| NM-TRAN | Python translation |
|---------|-------------------|
| `THETA(n)` | `theta[n-1]` |
| `ETA(n)` | `eta[n-1]` |
| `EPS(n)` | `eps[n-1]` |
| `F` | Predicted value from PK model |
| `Y` | Predicted observation (set in `$ERROR`) |
| `T` | Current observation time |
| `IPRED` | Individual prediction |
| `EXP(x)` | `math.exp(x)` |
| `LOG(x)` | `math.log(x)` |
| `SQRT(x)` | `math.sqrt(x)` |
| `DADT(n)` | `dadt[n-1]` (in `$DES`) |
| `A(n)` | `a[n-1]` (compartment amount) |

## Using a `.ctl` file directly

OpenPKPD can parse real NONMEM control streams:

```python
from openpkpd.parser.control_stream import ControlStream

cs = ControlStream.from_file("run001.ctl")
# Inspect parsed records
cs.problem.title
cs.theta_records[0].specs        # List[ThetaSpec]
cs.estimation_records[0].method  # "FOCE"
```

Use the CLI to parse and inspect without fitting:

```bash
openpkpd parse run001.ctl
openpkpd parse run001.ctl --json   # Machine-readable JSON
```

## Output files

OpenPKPD writes the same output files as NONMEM 7.x:

| NONMEM file | Description |
|-------------|-------------|
| `.lst` | Estimation log + parameter table |
| `.ext` | Parameter estimates by iteration |
| `.phi` | Empirical Bayes estimates (post-hoc ETAs) |
| `.cov` | Covariance matrix |
| `.cor` | Correlation matrix |
| `sdtab` / `patab` | `$TABLE` output |

## Current limitations

The following NONMEM features are not yet fully supported:

- **$MIXTURE** — finite mixture models — not yet implemented
- **$PRIOR** — MAP estimation with informative priors — not yet implemented
- **$SIMULATION** — Monte Carlo simulation block — partial; use `openpkpd.simulation` directly

Previously listed as limitations but now implemented:

| Feature | Status |
|---------|--------|
| ADVAN5/7 (general linear, matrix exponential) | ✅ Implemented |
| ADVAN6/8 (general nonlinear ODE, stiff/nonstiff) | ✅ Implemented |
| ADVAN11/12 (3-compartment IV/oral) | ✅ Implemented |
| ADVAN13 (stiff ODE + adjoint sensitivity) | ✅ Implemented (partial) |
| IOV (inter-occasion variability) | ✅ Implemented |
| NUTS/BAYES (MCMC via PyMC/NumPyro) | ✅ Implemented |
| Control stream round-trip write (`to_nmtran()`) | ✅ Implemented |
