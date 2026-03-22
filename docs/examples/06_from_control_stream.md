# Example 6 — Running a Control Stream File

**Script:** `examples/06_from_control_stream.py`

Demonstrates parsing and running an existing NONMEM `.ctl` file directly,
without any Python model building.

## Output

```{literalinclude} ../_static/examples/06_output.txt
:language: text
```

## From Python

```python
from openpkpd.parser.control_stream import ControlStream
from openpkpd.cli.runner import run_model

# Parse only — inspect records without fitting
cs = ControlStream.from_file("run001.ctl")
print(cs.problem.title)
print(cs.estimation_records[0].method)
print(cs.theta_records[0].specs)

# Parse + fit
result = run_model("run001.ctl")
print(result.summary())
```

## From the CLI

```bash
# Run with default settings
openpkpd run run001.ctl

# Override method
openpkpd run run001.ctl --method FOCE --verbose

# Inspect records only (no estimation)
openpkpd parse run001.ctl
openpkpd parse run001.ctl --json   # Machine-readable JSON
```

## Minimal `.ctl` file

```
$PROBLEM Theophylline 1-compartment oral FO
$DATA theo.csv IGNORE=#
$INPUT ID TIME AMT DV EVID
$SUBROUTINES ADVAN2 TRANS2
$PK
  KA = THETA(1)*EXP(ETA(1))
  CL = THETA(2)*EXP(ETA(2))
  V  = THETA(3)*EXP(ETA(3))
$ERROR
  IPRED = F
  W = THETA(4) * IPRED
  Y = IPRED + W * EPS(1)
$THETA (0.01,1.5,20) (0,0.04,2) (0,0.50,5) (0.01,0.10,0.50)
$OMEGA 0.48 0.07 0.02
$SIGMA 1 FIXED
$ESTIMATION METHOD=ZERO MAXEVAL=500
$COVARIANCE
$TABLE ID TIME DV PRED IPRED CWRES NOAPPEND NOPRINT FILE=sdtab
```

## Supported ESTIMATION keywords

| NM-TRAN `METHOD=` | OpenPKPD internal |
|--------------------|---------------------|
| `ZERO` | `"FO"` |
| `COND` | `"FOCE"` |
| `COND INTER` | `"FOCE"` + interaction |
| `COND LAPLACE` | `"LAPLACIAN"` |
| `SAEM` | `"SAEM"` |
| `IMP` | `"IMP"` |

## Notes

- Column auto-mapping reads `$INPUT` and matches names to the dataset.
- If `EVID` is absent from `$INPUT`, it is auto-generated from `AMT`.
- `IGNORE=#` causes rows starting with `#` to be skipped.
- OpenPKPD writes `.lst`, `.ext`, `.phi`, `.cov`, `.cor` to the same
  directory as the `.ctl` file.
