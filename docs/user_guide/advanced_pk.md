# Advanced PK Features

This page documents OpenPKPD's advanced pharmacokinetic modelling features:
delay differential equations (DDE), SBML model import, inter-occasion
variability (IOV), and parallel execution.

---

## Delay Differential Equations (DDE)

### Motivation

Standard ODEs assume the rate of change at time *t* depends only on the
current state **A**(*t*).  Some PK/PD processes involve explicit delay:

- **Transit absorption**: drug must traverse *n* transit compartments before
  reaching the central compartment.  The transit compartment ODE chain is
  equivalent to a delay.
- **Target-mediated drug disposition (TMDD)**: receptor-occupancy feedback.
- **Tumour cell-cycle models**: mitosis occurs after a fixed maturation delay τ.

OpenPKPD supports these via `DDESubroutine` (registered as ADVAN16):

```python
from openpkpd.pk.ode.dde import DDESubroutine
# or equivalently:
from openpkpd.pk import get_advan
solver = get_advan(16)
```

### How it works

The history function `_AHISTORY` is injected into `pk_params` before each call
to your `$DES` callable.  Query it at any past time:

```python
def my_des(t, A, pk_params, theta, eta):
    hist = pk_params["_AHISTORY"]        # callable: t_past -> list[float]
    tau  = pk_params["TAU"]              # delay parameter from $PK
    A_lag = hist(max(t - tau, 0.0))     # A at time t - tau (zeros before dose)

    CL, V = pk_params["CL"], pk_params["V"]
    return [-(CL / V) * A_lag[0]]
```

The delay value **τ** is read from `pk_params` under the key `"TAU"` or
`"DELAY"`.  If neither is present, the solver degenerates to a plain ODE.

### Dose events and history

Dose events (bolus and infusion) are handled identically to ADVAN6.
The history returns `0` for any query before the first dose, consistent with
the assumption that all compartments start empty.

Initial non-zero amounts can be set via `pk_params["A0_1"]`, `"A0_2"`, etc.

### Full example

```python
import numpy as np
from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.ode.dde import DDESubroutine

def elimination_delay_des(t, A, pk_params, theta, eta):
    """Elimination rate driven by A(t - TAU) instead of A(t)."""
    hist = pk_params.get("_AHISTORY")
    tau  = pk_params.get("TAU", 0.0)
    ke   = pk_params["CL"] / pk_params["V"]
    if hist and tau > 0:
        A_lag = hist(max(t - tau, 0.0))
        return [-ke * A_lag[0]]
    return [-ke * A[0]]

dose_events = [DoseEvent(time=0.0, amount=100.0, rate=0.0, duration=0.0, compartment=1)]
obs_times   = np.linspace(0.5, 12.0, 48)

solver = DDESubroutine(n_compartments=1, rtol=1e-6, atol=1e-8)
sol = solver.solve(
    pk_params={"CL": 2.0, "V": 10.0, "TAU": 0.5},
    dose_events=dose_events,
    obs_times=obs_times,
    des_callable=elimination_delay_des,
)

# sol.ipred — concentration at each obs time (A/V)
# sol.amounts — shape (n_obs, n_compartments)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_compartments` | 10 | Number of ODE compartments |
| `rtol` | 1e-6 | Relative ODE tolerance |
| `atol` | 1e-8 | Absolute ODE tolerance |
| `method` | `"RK45"` | scipy `solve_ivp` integration method |

### Limitations

- The current implementation uses dense-output piecewise RK45 with a linear
  interpolated history buffer.  For systems requiring very long history queries
  relative to the observation window, ensure `rtol` and `atol` are tight
  enough.
- ADVAN13 forward sensitivity is separate from DDE support.

---

## SBML Model Import

### Motivation

Systems Biology Markup Language (SBML) is the standard format for QSP
(quantitative systems pharmacology) and mechanistic PK/PD models.  OpenPKPD
can import SBML files and automatically generate a `des_callable` compatible
with the ADVAN6/DDESubroutine integration engine.

### Requirements

```bash
pip install python-libsbml
```

### Loading an SBML file

```python
from openpkpd.io import load_sbml

model = load_sbml("tumor_growth.xml")

print(model.species_names)       # ['A_tumor', 'A_drug']
print(model.parameter_names)     # ['kgrow', 'kdrug', 'V']
print(model.default_pk_params)   # {'kgrow': 0.1, 'kdrug': 0.5, 'V': 1.0}
print(model.n_compartments)      # 2
```

The `SBMLModel` object provides:

| Attribute | Description |
|-----------|-------------|
| `species_names` | Ordered list of species IDs (index n → A[n]) |
| `parameter_names` | Parameter IDs in THETA order |
| `default_pk_params` | Initial parameter values from SBML |
| `initial_amounts` | Initial species amounts |
| `n_compartments` | Number of ODE compartments |
| `des_callable` | DES function compatible with ADVAN6 |

### Simulation

```python
from openpkpd.pk.ode.advan6 import ADVAN6
from openpkpd.data.event_processor import DoseEvent
import numpy as np

dose_events = [DoseEvent(time=0.0, amount=100.0, rate=0.0, duration=0.0, compartment=1)]
obs_times   = np.linspace(0.0, 24.0, 100)

advan = ADVAN6(n_compartments=model.n_compartments)
sol   = advan.solve(
    pk_params=model.default_pk_params,
    dose_events=dose_events,
    obs_times=obs_times,
    des_callable=model.des_callable,
)
```

### Estimation

Convert SBML parameters to `ThetaSpec` for use with any estimation method:

```python
from openpkpd.model.parameters import ParameterSet

theta_specs = model.to_theta_specs()
params = ParameterSet.from_specs(theta_specs, [], [])

# Map estimated THETA back to pk_params dict:
pk_params = model.pk_callable_from_theta(result.theta_final.tolist())
```

### Supported SBML features

- Species (→ compartment amounts A[n])
- Parameters (→ ThetaSpec initial estimates)
- Compartment volumes (used for concentration/amount conversion)
- Reactions with MathML kinetic laws (evaluated via `libsbml.formulaToL3String`)

**Not yet supported**: SBML events, rules, constraints (silently warned and skipped).

### Building models programmatically

Without an SBML file, you can construct an `SBMLModel` directly using the
internal `_build_des_callable` helper:

```python
from openpkpd.io.sbml import SBMLModel, _build_des_callable

species_names  = ["A_central", "A_peripheral"]
parameter_names = ["CL", "V1", "Q", "V2"]
species_index  = {"A_central": 0, "A_peripheral": 1}

dadt_exprs = {
    0: ["-(pk_params['CL']/pk_params['V1'])*A[0]",
        "-(pk_params['Q']/pk_params['V1'])*A[0]",
        "+(pk_params['Q']/pk_params['V2'])*A[1]"],
    1: ["+(pk_params['Q']/pk_params['V1'])*A[0]",
        "-(pk_params['Q']/pk_params['V2'])*A[1]"],
}

des_callable = _build_des_callable(
    dadt_exprs, species_names, species_index, parameter_names, 2, []
)

model = SBMLModel(
    species_names=species_names,
    parameter_names=parameter_names,
    default_pk_params={"CL": 3.0, "V1": 10.0, "Q": 1.5, "V2": 30.0},
    initial_amounts={"A_central": 0.0, "A_peripheral": 0.0},
    n_compartments=2,
    des_callable=des_callable,
)
```

---

## Inter-Occasion Variability (IOV)

### Background

IOV models separate random effects into:
- **IIV** (inter-individual variability) — differences *between* subjects
- **IOV** (inter-occasion variability) — differences *within* the same subject
  across study occasions (e.g., Period 1 vs Period 2 of a cross-over study)

### Data requirements

The dataset must contain an **`OCC`** column (occasion indicator, integer ≥ 1).
OpenPKPD's `EventProcessor` reads this column automatically and stores it
per subject as `occasion_indices`.

### Model specification

In your `$PK` block, use conditional logic on the `OCC` covariate:

```
$PK
ETAKA = ETA(1)
IF (OCC.EQ.2) ETAKA = ETA(4)    ; Occasion-specific ETA
KA = THETA(1) * EXP(ETAKA)
CL = THETA(2) * EXP(ETA(2))
V  = THETA(3) * EXP(ETA(3))
```

Define the occasion OMEGA block with `same=True` to link occasions:

```python
omega_specs = [
    OmegaSpec(init=0.3),            # IIV on KA
    OmegaSpec(init=0.2),            # IIV on CL
    OmegaSpec(init=0.15),           # IIV on V
    OmegaSpec(init=0.3, same=True), # IOV on KA (same variance as occasion 1)
]
```

The compiled `$PK` callable receives `covariates={"OCC": float}` for each
occasion and the results are stitched together across occasions during
`IndividualModel.evaluate()`.

---

## Parallel Execution

### Overview

OpenPKPD's `parallel` module provides a unified `map()` interface across three
backends:

| Backend | When to use | Install |
|---------|-------------|---------|
| `multiprocessing` | Single machine, always available | — |
| `dask` | HPC clusters, large-scale work | `pip install dask[distributed]` |
| `ray` | Cloud, mixed CPU/GPU | `pip install ray` |

### Usage

```python
from openpkpd.parallel import get_backend

# Auto-select best available backend
backend = get_backend(n_jobs=8)

# Apply a function to each element of a list in parallel
results = backend.map(my_function, list_of_arguments)

# Context manager (cleanly shuts down Dask/Ray on exit)
with get_backend(n_jobs=4) as backend:
    results = backend.map(fit_bootstrap, replicate_datasets)
```

### Backends in detail

#### Multiprocessing (always available)

```python
from openpkpd.parallel import _MultiprocessingBackend

b = _MultiprocessingBackend(n_jobs=4)  # or n_jobs=-1 for all CPUs
results = b.map(fit_function, args_list)
```

Uses `concurrent.futures.ProcessPoolExecutor`.  `n_jobs=1` executes inline
(no subprocess overhead — useful for debugging).

#### Dask

```python
backend = get_backend(n_jobs=8, backend="dask")
# Optionally connect to an existing Dask scheduler:
backend = get_backend(backend="dask", scheduler_address="tcp://scheduler:8786")
```

#### Ray

```python
backend = get_backend(n_jobs=8, backend="ray")
# Optionally connect to an existing Ray cluster:
backend = get_backend(backend="ray", address="auto")
```

### Bootstrap example

```python
import numpy as np
from openpkpd.parallel import get_backend

def fit_replicate(replicate_df):
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset
    ds  = NONMEMDataset.from_dataframe(replicate_df)
    res = (ModelBuilder().dataset(ds).subroutines(advan=2)
           .pk("CL=THETA(1); V=THETA(2)").error("Y=F*(1+EPS(1))")
           .theta([(0.1,3,10),(1,30,200)]).omega([0.1,0.1]).sigma(0.05)
           .estimation("FO").build().fit())
    return res.theta_final.tolist()

n_jobs = 4
replicates = [...]  # list of DataFrames
backend = get_backend(n_jobs=n_jobs, backend="multiprocessing")
with backend:
    boot_thetas = backend.map(fit_replicate, replicates)

# Percentile CI
boot_arr = np.array(boot_thetas)
ci_lo = np.percentile(boot_arr, 2.5, axis=0)
ci_hi = np.percentile(boot_arr, 97.5, axis=0)
```

### Timeout support

```python
# Abort workers that take longer than 60 seconds
results = backend.map(fit_function, args_list, timeout=60.0)
```

Timeout is supported by the multiprocessing backend.  For Dask, use
`client.gather(futures, timeout=60)` directly.
