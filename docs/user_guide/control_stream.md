# Control Streams

OpenPKPD can parse real NONMEM control-stream files (`.ctl`, `.mod`, `.txt`)
and execute the parts of that workflow that are currently wired into the native
runner.

## Parsing a control stream

```python
from openpkpd.parser.control_stream import ControlStream

cs = ControlStream.from_file("run001.ctl")
```

Or from a string:

```python
cs = ControlStream.from_string("""
$PROBLEM Theophylline 1-cmt oral FOCE
$DATA theo.csv IGNORE=#
$INPUT ID TIME AMT DV EVID
$SUBROUTINES ADVAN2 TRANS2
$PK
  KA = THETA(1)*EXP(ETA(1))
  CL = THETA(2)*EXP(ETA(2))
  V  = THETA(3)*EXP(ETA(3))
$ERROR
  Y = F*(1+EPS(1))
$THETA (0.01,1.5,20) (0.001,0.08,5) (0.1,30,500)
$OMEGA 0.5 0.3 0.3
$SIGMA 0.1
$ESTIMATION METHOD=COND INTER MAXEVAL=9999
$COVARIANCE
""")
```

## Inspecting parsed records

```python
cs.problem.title            # "Theophylline 1-cmt oral FOCE"

cs.data.path                # "theo.csv"
cs.data.ignore_char         # "#"

cs.subroutines.advan        # 2
cs.subroutines.trans        # 2

cs.theta_records[0].specs[0]
cs.omega_records[0].values
cs.sigma_records[0].values

cs.estimation_records[0].method       # "FOCE"
cs.estimation_records[0].interaction  # True
cs.estimation_records[0].maxeval      # 9999
```

## Running from the CLI

```bash
openpkpd run model.ctl
openpkpd run model.ctl --method FOCE --verbose
openpkpd parse model.ctl
openpkpd parse model.ctl --json
```

## Running from Python

```python
from openpkpd.cli.runner import run_model

result = run_model("model.ctl")
print(result.summary())
```

## Supported records

The parser recognizes more record types than the current runner executes.

| Record | Current status | Notes |
|--------|----------------|-------|
| `$PROBLEM` | runtime | parsed and used |
| `$DATA` | runtime | `IGNORE=` supported |
| `$INPUT` | runtime | includes `DROP` / `SKIP` handling |
| `$SUBROUTINES` | runtime | `ADVAN1–4/6/8/10/11/12/13/16`; `TRANS1–6`, plus OpenPKPD `TRANS7/8` |
| `$PK` | runtime | compiled by the NM-TRAN compiler |
| `$ERROR` | runtime | compiled by the NM-TRAN compiler |
| `$DES` | runtime | used by ODE/DDE workflows |
| `$THETA` | runtime | bounds and `FIXED` supported |
| `$OMEGA` | runtime | diagonal, `BLOCK`, `SAME`, `FIXED` |
| `$SIGMA` | runtime | diagonal, `BLOCK`, `FIXED` |
| `$ESTIMATION` | runtime | FO/FOCE/FOCEI/Laplacian/SAEM/IMP/IMPMAP/BAYES keywords parsed |
| `$COVARIANCE` | runtime | covariance step available |
| `$TABLE` | runtime | column selection and export |
| `$SIMULATION` | runtime/partial | runner supports a first subset: first seed, `ONLYSIMULATION`, `SUBPROBLEMS=n`, and a default `.sim.csv` output artifact |
| `$MIXTURE` | runtime/partial | runner supports an `NSPOP=n` subset via dedicated `.mix.json` and `.mix_assignments.csv` artifacts |
| `$PRIOR` and prior blocks | runtime/partial | `$THETAP`/`$THETAPV` and `$OMEGAP`/`$OMEGAPD` are wired into a supported Gaussian-prior subset; `$SIGMAP*` remains parse-only |

Current prior-runtime subset notes:

- `$THETAP` requires `$THETAPV`
- `$OMEGAP` requires `$OMEGAPD`
- `$OMEGAP` currently supports either diagonal-only values or the full lower triangle
- `$OMEGAPD` is currently interpreted as Gaussian penalty weights, not full NONMEM Wishart semantics

Current simulation-runtime subset notes:

- the runner uses the **first** parsed `$SIMULATION` seed, defaulting to `42` when no seed is provided
- `SUBPROBLEMS=n` maps to `n` Monte Carlo replicates in the written simulation artifact
- `ONLYSIMULATION` runs from the current parameter state in the control stream without an estimation step
- when `$SIMULATION` is present alongside estimation, the runner writes an additional `<run>.sim.csv` artifact after fitting
- `TRUE=FINAL` is parsed but not yet given distinct runtime semantics in this subset

Current mixture-runtime subset notes:

- the runner supports `$MIXTURE NSPOP=n` using the existing EM-style `MixtureModel`
- the current runtime subset supports only inner estimation methods `FO`, `FOCE`/`FOCEI`, and `LAPLACIAN`
- outputs are written as dedicated `<run>.mix.json` and `<run>.mix_assignments.csv` artifacts rather than standard NONMEM-style `.ext/.phi/.tab` files
- `PMIX=THETA(n)` is parsed but not yet given runtime semantics in this subset
- `$MIXTURE` cannot currently be combined with `$SIMULATION`, `$COVARIANCE`, or `$TABLE` in the runner subset

## NM-TRAN code blocks

`$PK`, `$DES`, and `$ERROR` are written in NONMEM-style syntax and translated
to Python callables.

| NM-TRAN | Python |
|---------|--------|
| `THETA(n)` | `theta[n-1]` |
| `ETA(n)` | `eta[n-1]` |
| `EPS(n)` | `eps[n-1]` |
| `F` | predicted compartment output |
| `Y` | predicted observation |
| `T` | current time |
| `IPRED` | individual prediction |
| `DADT(n)` | `dadt[n-1]` (inside `$DES`) |
| `A(n)` | `a[n-1]` |

Common FORTRAN intrinsics such as `EXP`, `LOG`, `SQRT`, `ABS`, `SIN`, `COS`,
`MIN`, and `MAX` are also translated.

## BLOCK OMEGA example

```
$OMEGA BLOCK(2)
0.4
0.1 0.3
```

```python
omega_rec = cs.omega_records[0]
omega_rec.block_size
omega_rec.values
omega_rec.to_matrix()
```

## SAME OMEGA (IOV-style blocks)

```
$OMEGA BLOCK(2)
0.4
0.1 0.3
$OMEGA BLOCK(2) SAME
```

`SAME` blocks are parsed into repeated `OmegaSpec` entries and can be used in
IOV-oriented workflows. Support is broader than simple parsing, but still less
polished than the core FO/FOCE control-stream path.
