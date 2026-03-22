# OpenPKPD Development Roadmap

**Audit basis:** 2026-03 source-tree review and documentation refresh  
**Last updated:** 2026-03-09

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

These areas may still need validation, documentation, or workflow expansion, but
they are no longer “feature absent” gaps.

---

## Priority 1 — Highest-value near-term work

### 1.1 Harden advanced estimator validation

**Current state:** SAEM, IMP/IMPMAP, `BAYES`, and nonparametric estimation all
exist.

**Gap:** These paths still need broader regression and benchmark coverage than
the core FO/FOCE workflow.

**What’s needed:**

- add reference/regression tests for SAEM, IMP/IMPMAP, `BAYES`, and
  nonparametric estimation
- verify posterior diagnostics and backend fallback behavior for `BAYES`
- expand benchmark comparisons beyond theophylline-style base cases

### 1.2 Close parser-versus-runner gaps

**Current state:** The parser recognizes more NONMEM-style records than the
runner currently exposes as equally mature end-to-end workflows.

**Gap:** `$SIMULATION`, `$MIXTURE`, and `$PRIOR` support is stronger at parse
time than in polished runtime workflows.

**What’s needed:**

- validate `$SIMULATION` end-to-end
- make `$MIXTURE` a clearer runtime path
- confirm prior-driven workflows consistently across API and control-stream entry points

### 1.3 Improve simulation-diagnostics maturity

**Current state:** VPC, NPC, NPDE, and bootstrap-related building blocks exist.

**Gap:** Validation depth and workflow polish still lag mature external stacks.

**What’s needed:**

- validate NPDE/VPC behavior against known reference workflows where possible
- tighten pcVPC normalization behavior
- improve bootstrap confidence-interval reporting and related summaries

### 1.4 Promote advanced workflows to first-class UX

**Current state:** Library support exists for design, advanced diagnostics, and
simulation-driven workflows.

**Gap:** Some of that functionality is still easier to reach from Python than
from the CLI or GUI.

**What’s needed:**

- add clearer examples and user-guide coverage for advanced workflows
- expose VPC/bootstrap/design workflows more directly in the GUI
- improve CLI discoverability where advanced capabilities are already available

---

## Priority 2 — Competitive maturity improvements

### 2.1 Expand GUI completeness

**Current state:** The GUI covers data, model authoring, fit execution, NCA,
results, plots, diagnostics, and covariates.

**Gap:** The Advanced area is still mostly a placeholder, and advanced artifact
navigation can be improved.

**What’s needed:**

- turn the Advanced page into a real workflow hub
- surface simulation/diagnostic pipelines more directly in the UI
- improve run management and artifact navigation for scenario-heavy work

### 2.2 Improve new-design and simulation ergonomics

**Current state:** Simulation support exists, including new-design-oriented code
paths.

**Gap:** New-design workflows need stronger end-to-end coverage and clearer user
surface area.

**What’s needed:**

- strengthen validation of new-design simulation behavior
- add examples and tests for design/simulation handoff
- ensure outputs match the standard `SimulationResult` workflow expectations

### 2.3 Parallelize more expensive search workflows

**Current state:** multiprocessing, Dask, Ray, and MPI backend hooks are
available.

**Gap:** Not every expensive workflow makes full use of the available parallel
infrastructure.

**What’s needed:**

- parallelize SCM candidate evaluation more aggressively
- expand validation for bootstrap/SSE/SCM under alternate backends
- improve backend-selection ergonomics and error reporting

### 2.4 Broaden reporting and interchange

**Current state:** OpenPKPD writes NONMEM-like outputs, HTML reports, and NCA/CDISC helpers.

**Gap:** Interchange and reporting remain narrower than commercial stacks.

**What’s needed:**

- expand CDISC exports beyond current helpers
- add NONMEM result-file import/reuse workflows
- improve downstream interoperability for external analysis/reporting tools

---

## Priority 3 — Medium-term extensions

### 3.1 Broader external interoperability

Potential directions:

- R integration via `rpy2`
- richer import/export bridges for external pharmacometrics tools
- more reusable conversion paths for diagnostics/reporting ecosystems

### 3.2 Bayesian independence from optional backends

**Current state:** `BAYES` works best with PyMC or NumPyro/JAX and otherwise
falls back to a Laplace approximation.

**Gap:** A native posterior sampler without optional ecosystem dependencies does
not yet exist.

### 3.3 Full nonparametric support-point optimization

**Current state:** nonparametric estimation is present.

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
| Parser-versus-runner parity | P | stronger end-to-end runtime support | M | High |
| Simulation diagnostics maturity | P | validated and better-polished workflows | M | High |
| GUI advanced workflow exposure | P | first-class advanced workflows | M | High |
| New-design simulation ergonomics | P | clearer, better-tested workflow | S | Medium |
| Parallel search/SCM utilization | P | fuller backend usage | S | Medium |
| Reporting and interchange | P | richer import/export/report paths | M | Medium |
| External interoperability | P | better cross-tool integration | M | Medium |
| Native backend-independent Bayesian sampling | P | true sampler without optional backends | L | Low |
| Full nonparametric support-point optimization | P | closer to full NPEM behavior | L | Low |
| GPU acceleration | N | accelerated solver/objective stack | XL | Low |
| GxP validation package | N | formal validation program | XL | Low |

**Effort key:** S = days · M = 1–2 weeks · L = 1 month+ · XL = multi-month / strategic
