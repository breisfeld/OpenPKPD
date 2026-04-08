# Migrating from NONMEM

This page maps familiar NONMEM control stream concepts to their OpenPKPD equivalents.

For the current support classification behind these mappings, see
[`../user_guide/validation_matrix.md`](../user_guide/validation_matrix.md).
Some mappings are exact at the syntax layer but still have narrower runtime or
validation support than full NONMEM.

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
| `IMPMAP` | `"IMPMAP"` |

Practical migration guidance:

- `FO`, `FOCE`, `FOCEI`, and `LAPLACIAN` are the strongest current migration
  targets from a validation standpoint
- `SAEM`, `IMP` / `IMPMAP`, `BAYES(Laplace)`, and `NONPARAMETRIC` are real
  native surfaces, but their empirical validation breadth is still narrower
  than the core FO/FOCEI path
- native `BAYES(NUTS)` should currently be treated as an experimental /
  second-tier backend rather than as blanket NONMEM-parity MCMC support

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

OpenPKPD now supports useful runtime subsets for several records that were
previously listed as simply "not implemented". Treat the control-stream user
guide as the source of truth for the current parser/runtime/export contract:

- [`docs/user_guide/control_stream.md`](../user_guide/control_stream.md)

The most important current limitations are:

- **`$MIXTURE`** — partial runtime subset only. The native runner currently
  supports `NSPOP=n` with dedicated `.mix.json` and `.mix_assignments.csv`
  artifacts, but not the full NONMEM mixture workflow surface.
- **`$PRIOR`** — partial runtime subset only. The native runner supports the
  Gaussian/NWPRI-oriented subset documented in the control-stream guide, not
  the full NONMEM prior semantics.
- **`$SIMULATION`** — partial runtime subset only. `ONLYSIMULATION`,
  `SUBPROBLEMS=n`, the first parsed seed, and `.sim.csv` artifact generation are
  supported; broader NONMEM simulation semantics are not.
- **Round-trip write support** — `ControlStream.to_string()` / `.write()` can
  serialize the current parsed control-stream representation, but this should be
  treated as **supported for the documented subset**, not as a blanket guarantee
  for every NONMEM record combination.

Previously listed as limitations but now implemented:

| Feature | Status |
|---------|--------|
| ADVAN5 (general linear, matrix exponential) | ✅ Implemented |
| ADVAN7 | ✅ Implemented; overlaps ADVAN5 and is usually not needed as a separate selector |
| ADVAN6/8 (general nonlinear ODE, stiff/nonstiff) | ✅ Implemented |
| ADVAN11/12 (3-compartment IV/oral) | ✅ Implemented |
| ADVAN13 (stiff ODE + forward sensitivity) | ✅ Implemented (partial) |
| IOV (inter-occasion variability) | ✅ Implemented |
| NUTS/BAYES (MCMC via PyMC/native backend) | ✅ Implemented, with backend-specific support boundaries |
| Control stream serialization (`ControlStream.to_string()` / `.write()`) | ✅ Implemented for the documented subset |
