# ModelBuilder API

`ModelBuilder` is the primary OpenPKPD interface for defining and fitting models
without writing a NONMEM control stream file. Methods are chained in order.

## Minimal working example

```python
from openpkpd import ModelBuilder

result = (
    ModelBuilder()
    .problem("My model")
    .data("data.csv")
    .subroutines(advan=2, trans=2)
    .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV = THETA(3)*EXP(ETA(3))")
    .error("Y = F*(1 + EPS(1))")
    .theta([(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)])
    .omega([0.5, 0.3, 0.3])
    .sigma(0.1)
    .estimation(method="FOCE", interaction=True)
    .build()
    .fit()
)
```

## Method reference

### `.problem(title)`

Sets the problem description string (equivalent to `$PROBLEM`).

```python
.problem("Theophylline 1-compartment oral FOCE")
```

---

### `.data(path, ...)` / `.dataset(ds)`

Specifies the dataset.

```python
# From a file
.data("theo.csv",
      id="ID",      # column name for subject ID
      time="TIME",  # column name for time
      dv="DV",      # column name for observed value
      amt="AMT",    # column name for dose amount
      evid="EVID")  # column name for event ID; None = auto-generate

# From a pre-loaded NONMEMDataset
from openpkpd.data.dataset import NONMEMDataset
ds = NONMEMDataset.from_csv("theo.csv")
.dataset(ds)
```

---

### `.subroutines(advan, trans)`

Selects the PK subroutine and parameter transformation.

```python
.subroutines(advan=2, trans=2)   # 1-cmt oral, CL/V
.subroutines(advan=3, trans=4)   # 2-cmt IV, CL/V1/Q/V2
.subroutines(advan=1, trans=2)   # 1-cmt IV, CL/V
```

See {doc}`pk_subroutines` for the full ADVAN/TRANS combination table.

---

### `.pk(code)`

NM-TRAN `$PK` block as a Python string. Defines individual PK parameters from
`THETA`, `ETA`, and covariates.

```python
.pk("""
    KA = THETA(1) * EXP(ETA(1))
    CL = THETA(2) * (WT/70)**0.75 * EXP(ETA(2))
    V  = THETA(3) * EXP(ETA(3))
""")
```

---

### `.error(code)`

NM-TRAN `$ERROR` block. Defines `Y` (predicted observation), `IPRED`, `W`,
`IRES`, and `IWRES`.

```python
# Proportional error
.error("Y = F * (1 + EPS(1))")

# Additive error
.error("Y = F + EPS(1)")

# Combined additive + proportional
.error("""
    W     = SQRT(THETA(4)**2 + (F*THETA(5))**2)
    Y     = F + W*EPS(1)
    IRES  = DV - F
    IWRES = IRES / W
""")

# Direct Emax PD model
.error("""
    E0    = THETA(4)
    EMAX  = THETA(5)
    EC50  = THETA(6)
    IPRED = E0 + EMAX*F / (EC50 + F)
    W     = THETA(7)
    Y     = IPRED + W*EPS(1)
    IRES  = DV - IPRED
    IWRES = IRES / W
""")
```

---

### `.theta(specs)`

Defines THETA initial values and bounds.

Each element may be:

| Format | Meaning |
|--------|---------|
| `1.5` | Initial value, unbounded |
| `(0, 1.5)` | Lower-bounded: `lower=0, init=1.5` |
| `(0, 1.5, 20)` | Fully bounded: `lower=0, init=1.5, upper=20` |
| `ThetaSpec(init=1.0, fixed=True)` | Fixed parameter |

```python
.theta([
    (0.01, 1.5, 20),    # THETA(1): KA
    (0.001, 0.08, 5),   # THETA(2): CL
    (0.1, 30, 500),     # THETA(3): V
])
```

Optional labels:

```python
from openpkpd.model.parameters import ThetaSpec
.theta([
    ThetaSpec(lower=0.01, init=1.5, upper=20, label="KA"),
    ThetaSpec(lower=0.001, init=0.08, upper=5,  label="CL"),
])
```

---

### `.omega(values, fixed=False)`

Defines the OMEGA (between-subject variability) matrix.

```python
# Diagonal (separate 1×1 blocks — most common)
.omega([0.5, 0.3, 0.3])

# Full 2×2 block
.omega([[0.5, 0.1],
        [0.1, 0.3]])

# Fixed diagonal
.omega([0.1, 0.1], fixed=True)
```

:::{note}
For BLOCK OMEGA, provide the full symmetric matrix. OpenPKPD stores it as a
lower-triangular Cholesky factor internally.
:::

---

### `.sigma(values, fixed=False)`

Defines SIGMA (residual error variance).

```python
.sigma(0.1)            # Scalar → 1×1 SIGMA
.sigma([0.1, 0.05])    # Two EPS terms (additive + proportional)
.sigma(1.0, fixed=True) # Fixed at 1 (W absorbs scaling)
```

---

### `.estimation(method, **kwargs)`

Selects the estimation method and options.

```python
.estimation(method="FOCE", interaction=True, maxeval=9999)
.estimation(method="FO",   maxeval=500)
.estimation(method="SAEM", maxeval=300)
```

| Argument | Default | Description |
|----------|---------|-------------|
| `method` | `"FOCE"` | `"FO"`, `"FOCE"`, `"FOCEI"`, `"LAPLACIAN"`, `"SAEM"`, `"IMP"` |
| `interaction` | `False` | FOCEI — ETA-EPS interaction |
| `maxeval` | `9999` | Maximum objective function evaluations |

---

### `.covariance(matrix="SR")`

Enables the covariance step (sandwich R/S estimator) after estimation.

```python
.covariance()              # Default R/S sandwich
.covariance(matrix="R")    # R matrix only (inverse Hessian)
```

---

### `.covariates(columns)`

Declares time-varying covariate columns so they appear in diagnostic output.

```python
.covariates(["WT", "AGE", "SEX"])
```

---

### `.build()`

Assembles and validates all components, returning a `BuiltModel` object.

```python
built = (
    ModelBuilder()
    ...
    .build()
)
```

---

### `.fit()` (on `BuiltModel`)

Runs estimation and returns an `EstimationResult`.

```python
result = built.fit()
```

See {doc}`estimation_methods` for details on `EstimationResult` fields.
