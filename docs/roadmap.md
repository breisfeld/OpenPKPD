# OpenPKPD Development Roadmap

**Audit basis:** 2026-03 source-tree review and documentation refresh  
**Last updated:** 2026-03-26

This roadmap highlights the **remaining high-value follow-up work** after the
2026-03 documentation audit. It intentionally avoids listing features that are
already present in the codebase, even when those features are still partial or
need polish.

## What is no longer a roadmap item

The following are already implemented and therefore should **not** be described
as missing roadmap goals:

- desktop GUI support
- NPDE support in the simulation layer
- PK coverage beyond `ADVAN1–4` (including `ADVAN11`, `ADVAN12`, ODE routes,
  and `ADVAN16`-style DDE support)
- time-varying covariate handling in the core model/evaluation path
- MPI backend hooks in the parallel layer
- PFIM/optimal-design entry from `BuiltModel.design()`
- GUI covariate workflow scaffolding
- SBML model import (`io/sbml.py`, example 17)
- R interoperability bridge (`r_bridge/`, `rpy2` optional extra)
- `$MIXTURE` as a runtime path (`mixture/mixture.py`, discrete EM algorithm,
  subpopulation output via `mixture_writer.py`)
- `$PRIOR` / prior-driven estimation across API and control-stream entry points
  (`prior/prior.py`, MAP penalty, NWPRI-compatible covariance priors)
- native NumPy/SciPy NUTS/HMC sampler requiring neither JAX nor PyMC
  (`estimation/nuts.py`)
- IOV (inter-occasion variability) with OCC column and block-diagonal OMEGA
  (example 23)
- PBPK model template (`pk/pbpk/`, `FiveOrganPBPK`, five named organs, example 22)
- Ray parallel backend in addition to multiprocessing, Dask, and MPI
- GUI bootstrap, VPC, NPDE, and SCM services

These areas may still need validation, documentation, or workflow expansion, but
they are no longer “feature absent” gaps.

---

## Priority 1 — Highest-value near-term work

### 1.1 Harden advanced estimator validation

**Current state:** SAEM, IMP/IMPMAP, `BAYES`, NUTS, and nonparametric estimation
all exist.

**Gap:** These paths still need broader regression and benchmark coverage than
the core FO/FOCE workflow.

**What’s needed:**

- add reference/regression tests for SAEM, IMP/IMPMAP, `BAYES`, NUTS, and
  nonparametric estimation
- verify posterior diagnostics and backend fallback behavior for `BAYES`
- expand benchmark comparisons beyond theophylline-style base cases

### 1.2 Close remaining parser-versus-runner gaps

**Current state:** `$MIXTURE` and `$PRIOR` now have full runtime paths.
`$SIMULATION` has a simulation engine. The remaining gap is narrower than
before.

**Gap:** Some less-common NONMEM records (e.g. `$SIMULATION` end-to-end via
control-stream entry, multi-problem `$PROB` runs, `$ABBREVIATED`) are
recognized at parse time but not fully exercised as polished runtime workflows.

**What’s needed:**

- validate `$SIMULATION` end-to-end from `.ctl` entry
- confirm multi-problem workflow handling
- test less-common record types that the parser accepts

### 1.3 Improve simulation-diagnostics maturity

**Current state:** VPC, NPC, NPDE, and bootstrap-related building blocks exist.

**Gap:** Validation depth and workflow polish still lag mature external stacks.

**What’s needed:**

- validate NPDE/VPC behavior against known reference workflows where possible
- tighten pcVPC normalization behavior
- improve bootstrap confidence-interval reporting and related summaries

### 1.4 Promote advanced workflows to first-class UX

**Current state:** Library support exists for design, advanced diagnostics, and
simulation-driven workflows. The GUI exposes bootstrap, VPC, NPDE, and SCM
as services, but the Advanced page is still mostly a placeholder.

**Gap:** Some functionality is still easier to reach from Python than from the
CLI or GUI.

**What’s needed:**

- turn the Advanced GUI page into a real workflow hub
- add user-guide coverage for recently added features (NUTS, PBPK, IOV, HMM)
- improve CLI discoverability where advanced capabilities are already available

---

## Priority 2 — Competitive maturity improvements

### 2.1 Harden newer-feature test and documentation coverage

**Current state:** Many features added recently (NUTS, PBPK, IOV, HMM, TTE,
SBML) have working examples but limited test depth and user-guide coverage.

**Gap:** Newer features need regression tests, API reference polish, and
user-guide chapters comparable to the core FO/FOCE workflow.

**What’s needed:**

- add regression and integration tests for NUTS, PBPK, IOV, HMM, SAEM, IMP
- expand user-guide chapters for recently added features
- ensure API reference auto-docs cover all public classes in new modules

### 2.2 Improve new-design and simulation ergonomics

**Current state:** Simulation support exists, including new-design-oriented code
paths.

**Gap:** New-design workflows need stronger end-to-end coverage and clearer user
surface area.

**What’s needed:**

- strengthen validation of new-design simulation behavior
- add examples and tests for design/simulation handoff
- ensure outputs match the standard `SimulationResult` workflow expectations

### 2.3 Deepen parallel backend utilization

**Current state:** Multiprocessing, Dask, Ray, and MPI backends are all
available. Bootstrap and SAEM use subject-level parallelism. SCM uses
`ThreadPoolExecutor` for candidate evaluation.

**Gap:** Not every expensive workflow makes full use of the best available
backend. Backend-selection ergonomics could be clearer.

**What’s needed:**

- extend SCM to use process-level parallelism under Dask/Ray/multiprocessing
  backends (not just threads)
- expand validation for bootstrap/SSE/SCM under alternate backends
- improve backend-selection ergonomics and error reporting

### 2.4 Broaden reporting and interchange

**Current state:** OpenPKPD writes NONMEM-like outputs, HTML/PDF reports, and
NCA/CDISC SDTM helpers. A NONMEM output reader (`nonmem_reader.py`) exists.

**Gap:** Interchange and reporting remain narrower than commercial stacks.

**What’s needed:**

- expand CDISC exports beyond current SDTM helpers (e.g. ADaM domains)
- improve NONMEM result-file import/reuse workflows
- improve downstream interoperability for external analysis/reporting tools

---

## Priority 3 — Medium-term extensions

### 3.1 Deepen external interoperability

SBML import and an R bridge exist. Potential directions for further work:

- richer SBML round-trip (export as well as import)
- deeper nlmixr2/Monolix result interchange
- more reusable conversion paths for diagnostics/reporting ecosystems

### 3.2 Promote the native NUTS sampler to primary Bayesian fallback

**Current state:** A native NumPy/SciPy NUTS/HMC sampler (`estimation/nuts.py`)
exists that requires neither JAX nor PyMC. The `BAYES` estimator currently
prefers PyMC/NumPyro when available and otherwise falls back to a Laplace
approximation, bypassing the native sampler.

**Gap:** The native NUTS sampler is not yet wired as the primary `BAYES`
fallback, and its diagnostics (R-hat, ESS, trace plots) are not yet integrated
with the standard output layer.

**What's needed:**

- promote the native NUTS sampler as the default `BAYES` fallback (ahead of
  Laplace) when PyMC/NumPyro are absent
- integrate NUTS convergence diagnostics (R-hat, ESS) into the reporting layer

### 3.3 Full nonparametric support-point optimization

**Current state:** Nonparametric estimation (NPML) is present.

**Gap:** Support-point location optimization is still less complete than a full
NPEM-style implementation.

---

## Priority 4 — Long-term / strategic work

### 4.1 GPU-accelerated workflows

Likely requires deeper JAX-first or equivalent solver/objective integration.

### 4.2 Formal GxP validation package

Would require IQ/OQ/PQ-style validation artifacts, locked environments, and
external QA process support.

---

## Summary table

| Area | Current state | Target improvement | Effort | Impact |
|------|---------------|--------------------|:------:|:------:|
| Advanced estimator validation | P | broader regression and benchmark coverage | M | High |
| Parser-versus-runner parity | P | end-to-end for remaining records | S | High |
| Simulation diagnostics maturity | P | validated and better-polished workflows | M | High |
| GUI advanced workflow hub | P | turn Advanced page into real workflow hub | M | High |
| Newer feature test/doc coverage | P | regression tests + user-guide chapters | M | High |
| New-design simulation ergonomics | P | clearer, better-tested workflow | S | Medium |
| Parallel backend utilization | P | process-level SCM parallelism, backend UX | S | Medium |
| Reporting and interchange | P | richer import/export/report paths | M | Medium |
| External interoperability depth | P | richer SBML + nlmixr2/Monolix bridges | M | Low |
| Native NUTS as primary Bayes fallback | P | wire NUTS as default fallback, add diagnostics | S | Medium |
| Full nonparametric support-point optimization | P | closer to full NPEM behavior | L | Low |
| GPU acceleration | N | accelerated solver/objective stack | XL | Low |
| GxP validation package | N | formal validation program | XL | Low |

**Effort key:** S = days · M = 1–2 weeks · L = 1 month+ · XL = multi-month / strategic
