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

The following NONMEM features are planned for future phases:

- **ADVAN5/7** — general linear (matrix exponential) — Phase 3
- **ADVAN6/8/13** — general nonlinear ODE — Phase 3
- **ADVAN11/12** — 3-compartment IV/oral — Phase 3
- **$MIXTURE** — finite mixture models — Phase 3
- **$PRIOR** — MAP estimation with informative priors — Phase 3
- **IOV** (inter-occasion variability) — Phase 3
- **$SIMULATION** — Monte Carlo simulation — Phase 3
- **NUTS/BAYES** — MCMC via NumPyro or PyMC — Phase 3
