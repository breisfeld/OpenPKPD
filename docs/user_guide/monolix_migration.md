# Migrating from Monolix

This page maps Monolix 2024R1 / Mlxtran concepts to their OpenPKPD equivalents.
It also highlights the key data-format and parameterisation differences you will
encounter when converting an existing Monolix project.

> **No Mlxtran parser yet.** OpenPKPD does not currently parse `.mlxtran` project
> files automatically. Migration is done by re-expressing the model in the Python
> `ModelBuilder` API (or a NONMEM-style control stream). A Mlxtran parser is on
> the roadmap. The [NONMEM migration guide](../getting_started/migrating_from_nonmem.md)
> covers parsing existing `.ctl` files if you prefer to go via NONMEM syntax.

---

## Concept mapping

| Monolix concept | OpenPKPD equivalent |
|-----------------|---------------------|
| `[LONGITUDINAL]` model block | `.pk()` / `.des()` / `.error()` |
| `[INDIVIDUAL]` prior block | `.omega()` + distribution tag |
| `[PARAMETER]` fixed effects | `.theta()` |
| `<MODEL>` section | `ModelBuilder` chain |
| Monolix project `.mlxtran` | `ModelBuilder` Python object |
| Structural model library | `ModelBuilder.subroutines(advan=…)` |
| `SAEM` (Monolix default) | `.estimation(method="SAEM")` |
| `linearization` | `.estimation(method="FOCE")` |
| `importanceSampling` | `.estimation(method="IMP")` |
| Diagnostic plots panel | `openpkpd.plots.*` |
| Simulation tab | `openpkpd.simulation.*` |

---

## Data format

### Monolix CSV (semicolon-delimited)

Monolix project data typically uses semicolons and Mlxtran column names:

```
id;time;amt;y;wt;evid
1;0;4.02;;70;1
1;0.25;;2.84;;0
1;0.5;;6.57;;0
```

### NONMEM-style CSV (OpenPKPD)

OpenPKPD uses comma-delimited NONMEM format with standardised column names:

```
ID,TIME,AMT,DV,WT,EVID,MDV
1,0,4.02,.,70,1,1
1,0.25,.,2.84,.,0,0
1,0.5,.,6.57,.,0,0
```

Key differences:

| Monolix convention | OpenPKPD / NONMEM convention |
|--------------------|------------------------------|
| Semicolon (`;`) separator | Comma (`,`) separator |
| `y` for observation | `DV` for dependent variable |
| `amt` for dose | `AMT` for dose amount |
| `evid` (0 = obs, 1 = dose) | `EVID` (same coding) |
| Blank / `.` for missing | `.` or empty for missing; `MDV=1` masks observations |
| Dose in **mg/kg** (some models) | Dose in absolute units; divide `AMT/WT` in `$PK` |
| No `MDV` column required | `MDV=1` on dose rows recommended |

Convert a Monolix CSV file:

```python
import pandas as pd

df = pd.read_csv("monolix_data.csv", sep=";")
df = df.rename(columns={"id": "ID", "time": "TIME", "amt": "AMT",
                         "y": "DV", "evid": "EVID"})
df["MDV"] = (df["EVID"] == 1).astype(int)
df["DV"] = df["DV"].fillna(".")
df.to_csv("openpkpd_data.csv", index=False)
```

---

## Parameter conventions

### Fixed effects — natural scale vs log scale

Monolix reports fixed effects on the **natural scale** with `_pop` suffix:

| Monolix | Meaning | NONMEM / OpenPKPD |
|---------|---------|-------------------|
| `ka_pop = 1.53` | Population absorption rate (h⁻¹) | `THETA(1) = 1.53` |
| `V_pop = 0.456` | Population volume (L/kg or L) | `THETA(2) = 0.456` |
| `Cl_pop = 0.0402` | Population clearance (L/h) | `THETA(3) = 0.0402` |

For log-normal IIV (`logNormal` distribution in Mlxtran), the NONMEM / OpenPKPD
`$PK` block exponentiates:

```
; Monolix: ka ~ logNormal(ka_pop, omega_ka)
CL = THETA(3) * EXP(ETA(3))
V  = THETA(2) * EXP(ETA(2))
KA = THETA(1) * EXP(ETA(1))
```

### Random effects — SD vs variance

Monolix `omega_X` is a **standard deviation**; NONMEM/OpenPKPD `$OMEGA` stores
**variance**. Square the Monolix value when initialising:

```python
# Monolix: omega_ka = 0.45, omega_V = 0.26, omega_Cl = 0.33
model.omega([0.45**2, 0.26**2, 0.33**2])   # variances
```

### Residual error

Monolix combined error `a + b * f`:

| Monolix parameter | Meaning | NONMEM / OpenPKPD |
|-------------------|---------|-------------------|
| `a` | Additive SD | `THETA(n)` used as `EPS(1)` SD in `$SIGMA` |
| `b` | Proportional CV (fraction) | `THETA(m)` used as `EPS(2)` SD in `$SIGMA` |

```
; Monolix: y = f + (a + b*f)*eps,  a=0.3, b=0.1
Y = F + (THETA(4) + THETA(5)*F) * EPS(1)
```

OpenPKPD `$ERROR` / `.error()`:

```python
model.error("Y = F + (A_ERR + B_ERR*F)*EPS(1)")
model.theta([(0, 0.3, None), (0, 0.1, None)], labels=["A_ERR", "B_ERR"])
model.sigma([1.0])   # EPS(1) is N(0,1); SD carried by THETA(4,5)
```

---

## Estimation settings

| Monolix setting | OpenPKPD equivalent |
|-----------------|---------------------|
| `nbSAEMiterations = 500` | `SAEMMethod(n_iterations=500)` |
| `nbBurningIterations = 150` | `SAEMMethod(n_burn=150)` |
| `nbExploratoryIterations = 200` | Controlled by `phase1` parameter |
| `nbSimulatedIndividuals` | `SAEMMethod(n_chains=5)` (multi-chain) |
| `linearization` | `method="FOCE"` |
| `importanceSampling` | `method="IMP"` |


---

## Worked example — Theophylline 1-compartment oral (SAEM)

This example reproduces the public Monolix 2024R1 theophylline project
(available at monolixsuite.slp-software.com) using OpenPKPD.

### Reference Monolix parameters (natural scale)

| Parameter | Monolix value | Source |
|-----------|:------------:|--------|
| `ka_pop` | 1.533 h⁻¹ | monolix2rx article |
| `V_pop` | 0.456 L/kg | monolix2rx article |
| `Cl_pop` | 0.0402 L/h/kg | monolix2rx article |

> **Dose normalisation**: the Monolix theophylline project encodes `AMT` in
> mg/kg. In the OpenPKPD dataset use absolute mg; divide by body weight in
> `$PK` (see `CL = THETA(3)/WT`).

### OpenPKPD model

```python
from openpkpd import ModelBuilder, fit

model = (
    ModelBuilder()
    .problem("Theophylline 1-cmt oral — from Monolix")
    .data("examples/shared_data/theophylline/theophylline.csv")
    .subroutines(advan=2, trans=2)          # 1-cmt oral, CL/V/KA
    .pk("""
        KA = THETA(1) * EXP(ETA(1))
        V  = THETA(2) * EXP(ETA(2)) * WT   # V in L; Monolix reports L/kg
        CL = THETA(3) * EXP(ETA(3)) * WT   # CL in L/h; Monolix reports L/h/kg
    """)
    .error("Y = F + (THETA(4) + THETA(5)*F) * EPS(1)")
    # Monolix natural-scale starting values
    .theta([(0, 1.53, None),    # KA
            (0, 0.456, None),   # V  (L/kg, scaled by WT in $PK)
            (0, 0.0402, None),  # CL (L/h/kg, scaled by WT)
            (0, 0.3, None),     # Additive error SD
            (0, 0.1, None)])    # Proportional error CV
    # Monolix omega values squared (SD→variance)
    .omega([0.45**2, 0.26**2, 0.33**2])
    .sigma([1.0])
    .estimation(method="SAEM")
)

result = fit(model)
print(result.summary())
```

Expected final estimates (within ±15% of Monolix):

```
THETA  1 (KA):  ~1.5   h⁻¹
THETA  2 (V):   ~0.48  L/kg
THETA  3 (CL):  ~0.040 L/h/kg
OMEGA  1 (KA):  ~0.20  (SD ~0.45)
OMEGA  2 (V):   ~0.069 (SD ~0.26)
OMEGA  3 (CL):  ~0.11  (SD ~0.33)
```

The external validation suite (`tests/external_validation/test_vs_monolix.py`)
runs this comparison automatically and checks agreement within 15% on all
population parameters.

---

## Output file mapping

| Monolix output | OpenPKPD equivalent |
|----------------|---------------------|
| `FinalEstimatedLogLikelihood.txt` | `.lst` (OFV section) |
| `EstimatedPopulationParameters.txt` | `.ext` (final row) |
| `IndividualParameters/` | `.phi` |
| `CovarianceMatrix.txt` | `.cov` |
| `CorrelationMatrix.txt` | `.cor` |
| `ObservationParameters/` (`PRED`, `IPRED`) | `sdtab` / `$TABLE` |
| Plots (in `Charts/`) | `openpkpd.plots.goodness_of_fit()` |

---

## Covariate effects

| Monolix syntax | OpenPKPD `$PK` equivalent |
|----------------|--------------------------|
| `Cl = Cl_pop * (WT/70)^beta_Cl_WT` | `CL = THETA(3) * (WT/70)**THETA(6)` |
| `V = V_pop * (WT/70)` (fixed exponent 1) | `V = THETA(2) * WT/70` |
| `ka ~ logNormal(ka_pop, omega_ka) {wt}` | `KA = THETA(1) * EXP(ETA(1)) * WT**THETA(7)` |
| Categorical: `SEX = {M:0, F:dCl_F}` | `IF (SEX.EQ.2) CL = CL * (1 + THETA(8))` |

Automated covariate search (equivalent to Monolix's COSSAC):

```python
from openpkpd.covariate import AMD

amd = AMD(base_model=model, covariates=["WT", "SEX", "AGE"],
          parameters=["CL", "V"], method="SCM")
result = amd.run()
print(result.selected_covariates)
```

---

## Current limitations

| Feature | Status |
|---------|--------|
| Mlxtran `.mlxtran` file parser | Not yet implemented; on roadmap |
| COSSAC covariate method | Not implemented; SCM + FREM available |
| Monolix `mixture` models | Not yet implemented |
| Stochastic approximation EM with Rao-Blackwellisation | Multi-chain MH SAEM available; full Rao-Blackwell pending |
| Monolix `BayesianInformation` (MCMC-based) | NUTS via PyMC available (see Bayesian estimation guide) |
| Monolix Charts auto-layout | Individual plots available; layout wizard not yet |

---

## See also

- [Migrating from NONMEM](../getting_started/migrating_from_nonmem.md)
- [PK subroutines and ODE solver](pk_subroutines.md)
- [Estimation methods](estimation_methods.md)
- [Covariate modelling](analysis_tools.md)
- [External validation benchmarks](external_validation_benchmarks.md)
