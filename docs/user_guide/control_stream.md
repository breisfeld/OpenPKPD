# Control Streams

OpenPKPD can parse real NONMEM control-stream files (`.ctl`, `.mod`, `.txt`)
and execute the parts of that workflow that are currently wired into the native
runner.

For the broader support classification behind the runtime paths below, see
[`validation_matrix.md`](validation_matrix.md). In particular, "the runner
accepts this record" does **not** imply full NONMEM parity or FOCE-level
validation breadth for every estimator that can be named in `$ESTIMATION`.

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

Additional OpenPKPD-native estimation extensions are also parsed from
`$ESTIMATION` when present:

```text
$ESTIMATION METHOD=COND INTER MAXEVAL=200 OUTEROPT=L-BFGS-B \
  FALLBACKOPT=POWELL FALLBACKMAXEVAL=40 RETAINBEST \
  RETRYONABNORMAL RETRYOMEGASCALE=0.5,0.25,0.1
```

These extensions control FOCE/FOCEI outer optimization and retry behavior.
They are OpenPKPD additions rather than standard NONMEM keywords.

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

## Supported records and serialization contract

The parser recognizes more record types than the current runner executes. Treat
the table below as the executable support contract for the native runner and
the supported serialization contract for `ControlStream.to_string()` /
`ControlStream.write()`.

Status meanings:

- `runtime` = parsed and executed by the runner
- `runtime/partial` = parsed and executed only for a documented subset
- `parse-only` = parsed into typed records but not executed by the runner
- `round-trip` = the parsed record is serialized back into `.ctl` text by the
  current `ControlStream` object model

Important interpretation:

- parser acceptance is broader than runtime support
- runtime support is broader than high-confidence validation support
- the strongest current control-stream workflows remain the core
  `FO` / `FOCE` / `FOCEI` / `LAPLACIAN` estimation paths

| Record | Runtime status | Round-trip | Runner contract |
|--------|----------------|------------|-----------------|
| `$PROBLEM` | `runtime` | `yes` | parsed and used |
| `$DATA` | `runtime` | `yes` | `IGNORE=` supported |
| `$INPUT` | `runtime` | `yes` | includes `DROP` / `SKIP` handling |
| `$SUBROUTINES` | `runtime` | `yes` | `ADVAN1–6/8/10/11/12/13/16`; `TRANS1–6`, plus OpenPKPD `TRANS7/8` |
| `$PK` | `runtime` | `yes` | compiled by the NM-TRAN compiler |
| `$ERROR` | `runtime` | `yes` | compiled by the NM-TRAN compiler |
| `$DES` | `runtime` | `yes` | used by ODE/DDE workflows |
| `$THETA` | `runtime` | `yes` | bounds and `FIXED` supported |
| `$OMEGA` | `runtime` | `yes` | diagonal, `BLOCK`, `SAME`, `FIXED` |
| `$SIGMA` | `runtime` | `yes` | diagonal, `BLOCK`, `FIXED` |
| `$ESTIMATION` | `runtime` | `yes` | FO/FOCE/FOCEI/Laplacian/SAEM/IMP/IMPMAP/BAYES keywords parsed; estimator maturity still follows the validation matrix |
| `$COVARIANCE` | `runtime` | `yes` | covariance step available in the standard estimation path |
| `$TABLE` | `runtime` | `yes` | column selection and export in the standard estimation path |
| `$SIMULATION` | `runtime/partial` | `yes` | supports first seed, `ONLYSIMULATION`, `SUBPROBLEMS=n`, and `.sim.csv` output |
| `$MIXTURE` | `runtime/partial` | `yes` | supports `NSPOP=n` with dedicated `.mix.json` and `.mix_assignments.csv` artifacts |
| `$PRIOR` | `runtime/partial` | `yes` | NWPRI-oriented Gaussian-prior subset only |
| `$THETAP` / `$THETAPV` | `runtime/partial` | `yes` | supported together as THETA prior mean/variance |
| `$OMEGAP` / `$OMEGAPD` | `runtime/partial` | `yes` | supported as Gaussian penalty subset, not full Wishart semantics |
| `$SIGMAP` / `$SIGMAPD` | `parse-only` | `yes` | parsed but not executed by the native runner |
| `$ABBREVIATED` | `parse-only` | `yes` | parsed but not executed by the native runner |
| `$NONPARAMETRIC` | `parse-only` | `yes` | parsed but not yet exposed as a dedicated control-stream runtime path |
| `$SIZES` | `parse-only` | `yes` | parsed but not executed by the native runner |
| `$DESIGN` | `parse-only` | `yes` | parsed but not executed by the native runner |
| `$CONTR` | `parse-only` | `yes` | parsed but not executed by the native runner |

## Round-trip example

```python
from openpkpd.parser.control_stream import ControlStream

cs = ControlStream.from_file("model.ctl")
rendered = cs.to_string()
ControlStream.from_string(rendered)   # reparses the current object model
cs.write("roundtrip/model.ctl")
```

Round-trip support means OpenPKPD can serialize the current parsed representation
back to `.ctl` text for the documented subset above. It does **not** mean every
NONMEM feature combination is guaranteed to execute or to preserve semantics
beyond the current parser/runtime contract.

Practical recommendation:

- use control streams most confidently today for `FO`, `FOCE`, `FOCEI`, and
  `LAPLACIAN` workflows
- treat control-stream `SAEM`, `IMP` / `IMPMAP`, and `BAYES` usage as real but
  subject to the same secondary or experimental support boundaries documented
  for the Python API
- treat `$NONPARAMETRIC` as parser-visible but not yet a native control-stream
  runtime workflow

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

Current FOCE/FOCEI `$ESTIMATION` extension notes:

- `OUTEROPT=` selects the primary outer optimizer
- `FALLBACKOPT=` selects an optional follow-up polish optimizer
- `FALLBACKMAXEVAL=` limits that fallback budget
- `RETAINBEST` / `NORETAINBEST` toggle best-iterate retention
- `RETRYONABNORMAL` / `NORETRYONABNORMAL` toggle structured retry after abnormal termination
- `RETRYOMEGASCALE=a,b,c` supplies OMEGA scaling factors for those retries
- the GUI model builder does not currently expose these advanced controls; use control streams or the Python API when needed

## Unsupported runtime combinations

The current native runner rejects the following combinations explicitly:

| Combination | Current behavior |
|------------|------------------|
| `$SIMULATION ONLYSIMULATION` with CLI `--method` override | rejected |
| `$SIMULATION SUBPROBLEMS < 1` | rejected |
| `$MIXTURE` + `$SIMULATION` | rejected |
| `$MIXTURE` + `$COVARIANCE` | rejected |
| `$MIXTURE` + `$TABLE` | rejected |
| `$MIXTURE` + more than one `$ESTIMATION` record | rejected |
| `$MIXTURE` with inner method outside `FO`, `FOCE`, `FOCEI`, `LAPLACIAN` | rejected |

These are intentional contract failures, not undefined behavior. The integration
tests pin these cases so that unsupported workflows fail clearly rather than
silently drifting.

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
