# Comparison with Other PK/PD Software

This page compares OpenPKPD against six widely used population PK/PD and
NCA tools: NONMEM 7.5 (ICON plc), WinNonLin/Phoenix 8.4 (Certara),
mrgsolve 1.4 (open-source R), Monolix 2024R1 (Lixoft), Pumas.jl 2.5 (PumasAI),
and Pharmpy 1.x (Uppsala University).

> **Note on Pharmpy:** Pharmpy is a Python model-manipulation and workflow-automation
> library. It does not include a native estimation engine; instead it reads, transforms,
> and writes NONMEM control streams and dispatches estimation runs to NONMEM or
> nlmixr2/rxode2. Capabilities marked **via NM** require a NONMEM licence.

**Legend:**
- **Y** — fully supported and tested
- **P** — implemented but still partial, narrower in scope, or less validated than mature alternatives
- **—** — not supported
- **via NM** — supported by dispatching to an external NONMEM installation

---

## Estimation methods

| Method | OpenPKPD | NONMEM | Monolix | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|--------|:--------:|:------:|:-------:|:---------:|:--------:|:--------:|:-------:|
| First-Order (FO) | Y | Y | Y | Y | — | Y | via NM |
| FOCE / FOCEI | Y | Y | Y | Y | — | Y | via NM |
| Laplacian approximation | Y | Y | — | — | — | Y | via NM |
| SAEM | P | Y | Y | — | — | Y | via NM |
| Importance sampling (IMP/IMPMAP) | P | Y | — | — | — | Y | via NM |
| Nonparametric (NPML/NPEM) | P | Y | — | — | — | — | via NM |
| Full Bayesian / NUTS | P | Y | — | — | — | Y | — |
| MCMC diagnostics (R-hat, ESS) | P | Y | — | — | — | Y | — |

OpenPKPD implements the FO, FOCE, and Laplacian methods with the full NONMEM-compatible
estimation loop. SAEM uses a single-chain Metropolis-Hastings sampler (multi-chain
Rao-Blackwellisation is a known gap). Bayesian estimation dispatches to PyMC or
NumPyro when installed, otherwise falls back to a Laplace approximation.

---

## PK subroutines

### Analytical solutions

| ADVAN | Description | OpenPKPD | NONMEM | Monolix | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|-------|-------------|:--------:|:------:|:-------:|:---------:|:--------:|:--------:|:-------:|
| ADVAN1 | 1-cmt IV bolus | Y | Y | Y | Y | Y | Y | via NM |
| ADVAN2 | 1-cmt oral | Y | Y | Y | Y | Y | Y | via NM |
| ADVAN3 | 2-cmt IV | Y | Y | Y | Y | Y | Y | via NM |
| ADVAN4 | 2-cmt oral | Y | Y | Y | Y | Y | Y | via NM |
| ADVAN11 | 3-cmt IV | Y | Y | Y | Y | Y | Y | via NM |
| ADVAN12 | 3-cmt oral | Y | Y | Y | Y | Y | Y | via NM |
| TRANS1–6 | Parameterisation variants | Y | Y | Y | Y | Y | Y | via NM |

### ODE-based

| ADVAN | Description | OpenPKPD | NONMEM | Monolix | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|-------|-------------|:--------:|:------:|:-------:|:---------:|:--------:|:--------:|:-------:|
| ADVAN6 | Non-stiff ODE ($DES) | Y | Y | Y | — | Y | Y | via NM |
| ADVAN8 | Stiff ODE (LSODA) | Y | Y | Y | — | Y | Y | via NM |
| ADVAN10 | 1-cmt Michaelis-Menten | Y | Y | Y | Y | Y | Y | via NM |
| ADVAN13 | Stiff ODE + adjoint sensitivity | P | Y | — | — | — | Y | via NM |
| ADVAN16-style DDE | Delay differential equations | Y | Y | — | — | — | Y | via NM |
| PBPK | Physiologically-based PK | P | — | — | — | Y | Y | — |
| Transit absorption | Savic 2007 n-transit model | P | via $DES | Y | — | Y | Y | via NM |
| Parallel absorption | Multiple absorption pathways | P | via $DES | Y | — | Y | Y | via NM |
| Enterohepatic circulation | EHC re-absorption loop | P | via $DES | — | — | Y | — | via NM |

---

## PD and PK/PD models

| Model | OpenPKPD | NONMEM | Monolix | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|-------|:--------:|:------:|:-------:|:---------:|:--------:|:--------:|:-------:|
| Direct Emax / sigmoidal Hill | Y | via $PK | Y | Y | Y | Y | via NM |
| Indirect response (IDR I–IV) | Y | via $DES | Y | Y | Y | Y | via NM |
| Effect compartment | Y | via $PK | Y | Y | Y | Y | via NM |
| TMDD (full / QSSA / MM) | Y | via $DES | — | — | Y | Y | via NM |
| Tumor growth inhibition (Simeoni) | Y | via $DES | — | — | Y | — | via NM |
| Turnover / transit PD | Y | via $DES | Y | Y | Y | Y | via NM |
| Placebo response | Y | via $ERROR | — | — | — | — | via NM |
| DDI (competitive / TDI / induction) | Y | via $DES | — | — | Y | — | via NM |
| Count data (Poisson / NegBin / ZIP) | Y | — | Y | — | — | Y | via NM |
| Ordered categorical / proportional odds | Y | — | Y | — | — | Y | via NM |
| Markov chain PD | Y | — | Y | — | — | Y | via NM |
| Time-to-event (TTE) / survival | Y | via $DES | Y | — | — | Y | via NM |
| Mixed-effects PD (IIV on PD params) | P | Y | Y | — | — | Y | via NM |

---

## Data handling

| Feature | OpenPKPD | NONMEM | Monolix | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|---------|:--------:|:------:|:-------:|:---------:|:--------:|:--------:|:-------:|
| NONMEM CSV format | Y | Y | P | Y | Y | P | Y |
| EVID 0–4 event records | Y | Y | P | Y | Y | Y | Y |
| ADDL / II (additional doses) | Y | Y | P | Y | Y | Y | Y |
| SS (steady-state dosing) | Y | Y | P | Y | Y | Y | Y |
| Infusion (RATE / DURATION) | Y | Y | Y | Y | Y | Y | Y |
| BLQ M1 (exclusion) | Y | Y | Y | Y | Y | Y | via NM |
| BLQ M3/M4 (censored likelihood) | Y | Y | Y | Y | — | Y | via NM |
| BLQ M5/M7 (imputation) | Y | Y | Y | Y | — | Y | via NM |
| IOV (inter-occasion variability) | Y | Y | Y | Y | — | Y | Y |
| LLOQ column | Y | Y | Y | — | Y | Y | Y |
| Time-varying covariates | P | Y | Y | Y | Y | Y | Y |
| Missing covariate imputation | P | Y | Y | Y | — | Y | P |

---

## Output and reporting

| Feature | OpenPKPD | NONMEM | Monolix | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|---------|:--------:|:------:|:-------:|:---------:|:--------:|:--------:|:-------:|
| .lst run summary | Y | Y | — | — | — | — | Y |
| .ext parameter history | Y | Y | — | — | — | — | Y |
| .phi post-hoc ETAs | Y | Y | — | — | — | — | Y |
| .cov / .cor matrices | Y | Y | — | — | — | — | Y |
| $TABLE CSV export | Y | Y | Y | — | — | — | Y |
| AIC / BIC | Y | Y | Y | — | Y | Y | Y |
| Likelihood ratio test | Y | Y | Y | — | Y | Y | Y |
| Condition number | Y | Y | Y | — | Y | Y | Y |
| ETA shrinkage | Y | Y | Y | — | Y | Y | Y |
| EPS shrinkage | Y | Y | Y | — | Y | Y | Y |
| HTML report | Y | — | Y | Y | — | Y | P |
| Diagnostic plots (GOF, VPC, ETA panels) | Y | via Xpose | Y | Y | — | Y | Y |
| CDISC-formatted output | P | — | — | Y | — | — | — |
| R-hat / ESS for MCMC | P | Y | — | — | — | Y | — |

---

## Non-compartmental analysis (NCA)

| Feature | OpenPKPD | NONMEM | Monolix (PKanalix) | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|---------|:--------:|:------:|:------------------:|:---------:|:--------:|:--------:|:-------:|
| AUC (linear-log trapezoidal) | Y | — | Y | Y | — | Y | Y |
| Cmax, Tmax, t½, CLF, Vz | Y | — | Y | Y | — | Y | Y |
| Multiple-dose NCA | Y | — | Y | Y | — | Y | Y |
| Urine NCA (Ae, fe, CLR) | Y | — | Y | Y | — | Y | P |
| Average bioequivalence (ABE) | Y | — | Y | Y | — | Y | P |
| Reference-scaled ABE (RSABE) | Y | — | Y | Y | — | Y | P |
| BE sample size estimation | Y | — | Y | Y | — | Y | — |
| CDISC PP domain output | P | — | Y | Y | — | — | — |
| Sparse sampling NCA | P | — | Y | Y | — | Y | — |

---

## Simulation and model evaluation

| Feature | OpenPKPD | NONMEM | Monolix | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|---------|:--------:|:------:|:-------:|:---------:|:--------:|:--------:|:-------:|
| Replicate dataset simulation | Y | Y | Y | Y | Y | Y | via NM |
| New-design simulation | P | Y | Y | Y | Y | Y | via NM |
| Visual Predictive Check (VPC) | P | via PsN | Y | Y | — | Y | Y |
| pcVPC | P | via PsN | Y | Y | — | Y | Y |
| NPC (Numerical Predictive Check) | Y | via PsN | Y | Y | — | Y | via NM |
| NPDE (Normalised Prediction Distribution Errors) | P | via PsN | Y | Y | — | Y | Y |
| Bootstrap CI | P | via PsN | Y | Y | — | Y | Y |
| Stochastic simulation & estimation (SSE) | P | via PsN | — | — | — | Y | Y |

---

## Covariate modelling

| Feature | OpenPKPD | NONMEM | Monolix | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|---------|:--------:|:------:|:-------:|:---------:|:--------:|:--------:|:-------:|
| Manual covariate coding | Y | Y | Y | Y | Y | Y | Y |
| Linear, power, exponential effects | Y | Y | Y | — | Y | Y | Y |
| Categorical covariate effects | Y | Y | Y | Y | — | Y | Y |
| Stepwise SCM (forward/backward) | P | via PsN | Y (COSSAC) | — | — | Y | Y |
| Parallel SCM candidate evaluation | P | — | Y | — | — | Y | Y |
| FREM (full random effects model) | — | — | — | — | — | — | Y |
| Automated model development (AMD) | — | — | — | — | — | — | Y |

---

## Parallel computing

| Feature | OpenPKPD | NONMEM | Monolix | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|---------|:--------:|:------:|:-------:|:---------:|:--------:|:--------:|:-------:|
| Multi-core (ProcessPool) | Y | Y | Y | — | Y | Y | Y |
| Dask distributed cluster | P | — | — | — | — | Y | — |
| Ray cluster | P | — | — | — | — | — | — |
| MPI / HPC grid | P | Y | — | — | — | Y | via NM |
| GPU acceleration | — | — | — | — | — | Y | — |

---

## Ecosystem and usability

| Feature | OpenPKPD | NONMEM | Monolix | WinNonLin | mrgsolve | Pumas.jl | Pharmpy |
|---------|:--------:|:------:|:-------:|:---------:|:--------:|:--------:|:-------:|
| Open source (no licence fee) | Y | — | — | — | Y | partial | Y |
| Native Python API | Y | — | — | — | — | — | Y |
| NONMEM .ctl file parsing | Y | — | — | — | — | — | Y |
| SBML / QSP model import | Y | — | — | — | — | — | — |
| Delay differential equations | Y | Y | — | — | — | Y | via NM |
| R integration | — | via PsN | — | Y | Y | — | Y |
| GUI | P | — | Y | Y | — | partial | — |
| Sphinx documentation | Y | Y | Y | Y | Y | Y | Y |
| GxP regulatory validation | — | Y | Y | Y | — | — | — |

---

## Summary: when to choose OpenPKPD

**Choose OpenPKPD when you want to:**
- Work entirely in Python without NONMEM or Julia installations
- Migrate existing NONMEM `.ctl` control streams to an open-source platform
- Import SBML/QSP models from systems biology databases and fit them to PK data
- Use delay differential equations for mechanistic transit or feedback models
- Reproduce NONMEM-format `.lst`, `.ext`, `.phi`, `.cov` output for Xpose/PsN interoperability
- Integrate PK/PD modelling into Python data science workflows (pandas, numpy, scipy, matplotlib)
- Use the bundled `openpkpd_gui` desktop interface for dataset loading, model running,
  NCA, and diagnostic plots without writing code

**Consider alternatives when you need:**
- GxP-validated, regulatory-grade software (NONMEM, WinNonLin)
- Best-in-class SAEM convergence with Rao-Blackwellisation (Monolix)
- Fast compiled ODE simulation for 10,000+ subject VPCs (mrgsolve)
- GPU-accelerated estimation and adjoint sensitivity (Pumas.jl)
- A full GUI-only modelling environment (WinNonLin, Monolix) — OpenPKPD's GUI covers
  common workflows but lacks the model-building canvas of those tools
- Highly polished NPDE/VPC/automation pipelines for production workflows — OpenPKPD now includes NPDE/NPC/VPC building blocks, but the surrounding workflow polish still lags mature toolchains
- NONMEM model manipulation, FREM covariate search, or automated model development
  workflows built around an existing NONMEM infrastructure (Pharmpy)

---

> **Code audit note (2026-03-09):** The comparison tables above were reviewed
> against the current source tree. Key corrections in this pass:
> - **GUI** remains `P`, not `—`: `openpkpd_gui` provides a working desktop
>   interface for data, model, fit, NCA, results, plots, diagnostics, and SCM,
>   but the Advanced page is still mostly a placeholder.
> - **NPDE** is now marked `P`: the repository contains a dedicated
>   `simulation/npde.py` implementation and GUI-facing NPDE services, but this
>   area is still less validated and less polished than mature external stacks.
> - Parser-vs-runner gaps matter: records such as `$SIMULATION`, `$MIXTURE`, and
>   `$PRIOR` are parsed into typed records, but their end-to-end execution support
>   is still more limited than the core FO/FOCE-style workflow.
