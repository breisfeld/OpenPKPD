# PK Subroutines

OpenPKPD currently ships both analytical and numerical PK subroutines. The
core router in `openpkpd.pk` resolves the requested `ADVAN` and, for selected
cases, also uses `TRANS` to activate specialized absorption models.

## Implemented ADVAN selectors

| ADVAN | Type | Model |
|-------|------|-------|
| `ADVAN1` | analytical | 1-compartment IV bolus |
| `ADVAN2` | analytical | 1-compartment first-order oral |
| `ADVAN3` | analytical | 2-compartment IV bolus |
| `ADVAN4` | analytical | 2-compartment first-order oral |
| `ADVAN5` | analytical | N-compartment general linear (arbitrary Kij rate constants) |
| `ADVAN6` | numerical | general ODE system |
| `ADVAN8` | numerical | stiff ODE system |
| `ADVAN10` | numerical | Michaelis-Menten / nonlinear elimination workflows |
| `ADVAN11` | analytical | 3-compartment IV |
| `ADVAN12` | analytical | 3-compartment oral |
| `ADVAN13` | numerical | general ODE system with advanced sensitivity hooks |
| `ADVAN16` | numerical | delay differential equation extension |

## TRANS codes

| TRANS | Meaning |
|-------|---------|
| `TRANS1` | micro rate constants directly (`K`, `K12`, `K21`, …, `Kij`, `Ki0`) — required for ADVAN5 |
| `TRANS2` | `CL`, `V`-style one-compartment parameterisation |
| `TRANS3` | `CL`, `VSS`, `Q` |
| `TRANS4` | `CL`, `V1`, `Q`, `V2` |
| `TRANS5` | three-compartment macro-constant parameterisation |
| `TRANS6` | alternative three-compartment parameterisation |
| `TRANS7` | OpenPKPD transit-absorption extension |
| `TRANS8` | OpenPKPD parallel-absorption extension |

`TRANS7` and `TRANS8` are OpenPKPD-specific extensions rather than canonical
NONMEM PREDPP TRANS codes.

## Selecting a subroutine

```python
# 1-compartment IV — ADVAN1 TRANS2
.subroutines(advan=1, trans=2)

# 1-compartment oral — ADVAN2 TRANS2
.subroutines(advan=2, trans=2)

# 2-compartment IV — ADVAN3 TRANS4
.subroutines(advan=3, trans=4)

# 3-compartment oral — ADVAN12
.subroutines(advan=12, trans=6)

# General ODE / $DES workflow
.subroutines(advan=6)

# Delay differential equation workflow
.subroutines(advan=16)
```

## Specialized absorption helpers

When you need non-standard absorption without writing a full bespoke ODE model,
OpenPKPD can route selected `TRANS` values to dedicated absorption classes:

- `TRANS=7` → transit absorption
- `TRANS=8` → parallel absorption

For larger mechanistic models such as PBPK or custom absorption chains, prefer
`ADVAN6`, `ADVAN8`, or `ADVAN13` with a `$DES` block.

## Multiple dosing

Implemented subroutines support the usual NONMEM-style event handling:

- bolus doses
- zero-order infusions
- `ADDL` + `II`
- `SS=1`
- compartment selection via `CMT`

## Pre-dose observations

An observation recorded at exactly the same time as a dose is treated as a
pre-dose measurement, matching NONMEM convention.

## Covariate effects

Covariates are available inside `$PK` by column name:

```python
.pk("""
    CL = THETA(1) * (WT/70)**0.75 * EXP(ETA(1))
    V  = THETA(2) * (WT/70)       * EXP(ETA(2))
    KA = THETA(3) * EXP(ETA(3))
""")
.covariates(["WT"])
```

## Not yet available as named ADVAN selectors

The current router does **not** expose the following as built-in selectors:

| ADVAN | Status |
|-------|--------|
| `ADVAN7` | not currently implemented |
| `ADVAN9` | not currently implemented |

For many use cases that would otherwise require those selectors, the practical
alternative today is a custom `$DES` model through `ADVAN6/8/13`.

## ADVAN5 — General N-Compartment Linear Model

ADVAN5 analytically solves an arbitrary N-compartment linear system of ODEs
via eigendecomposition of the N×N rate matrix. It generalises ADVAN3 (N=2) and
ADVAN11 (N=3) to any number of compartments without writing a `$DES` block.
TRANS1 (micro rate constants) is the only supported TRANS code.

### Parameter naming convention

| Key pattern | Meaning |
|-------------|---------|
| `K{i}{j}` | Transfer rate FROM compartment *i* TO compartment *j* (i, j ∈ 1–9, i ≠ j) |
| `K{i}0` | Elimination rate FROM compartment *i* (i ∈ 1–9) |
| `K` | Alias for `K10` — elimination from compartment 1 |

**N inference rule:** N = max compartment index found across all `Kij` / `Ki0`
keys. Compartment indices are limited to 1–9; use ADVAN6/8 for N > 9.

### Volume and output compartment

- Default output compartment = 1 (central). Override via
  `ADVAN5(output_compartment=n)` or the `PCMT` key in `pk_params`.
- Volume is looked up as `V{output_cmt}` first, then `V` as a fallback.

### Example — 4-compartment linear model

```python
# Central (1) ↔ three peripherals (2, 3, 4)
.subroutines(advan=5, trans=1)
.pk("""
    K   = CL  / V1   ; elimination from central
    K12 = Q2  / V1   ; central  → peripheral 1
    K21 = Q2  / V2   ; peripheral 1 → central
    K13 = Q3  / V1   ; central  → peripheral 2
    K31 = Q3  / V3   ; peripheral 2 → central
    K14 = Q4  / V1   ; central  → peripheral 3
    K41 = Q4  / V4   ; peripheral 3 → central
""")
