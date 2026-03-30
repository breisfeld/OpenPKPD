# OpenPKPD — Remaining Priorities
*Last updated: 2026-03-29*
*Priority snapshot date: 2026-03-29*

This document reflects the current source tree, test suite, docs, and examples.
It is not a historical backlog dump. It focuses on what most limits OpenPKPD
relative to the tools users will compare it against:

- `NONMEM`
- `Phoenix WinNonlin`
- `Monolix`
- `nlmixr2`
- `Pharmpy`
- `Pumas`

The main theme is no longer "large features are missing everywhere". The core
library is broad. The highest-value remaining work is now:

1. hardening partially implemented workflows
2. widening empirical validation and benchmark coverage
3. tightening parser/runtime/export consistency
4. improving performance visibility and operational guidance
5. making the GUI and docs feel credible against mature competitor workflows

---

## What Is Already Real

These should not be treated as open capability gaps:

- FO / FOCE / FOCEI / Laplacian estimation
- SAEM, BAYES(Laplace), pure-NumPy NUTS, IMP/IMPMAP, and nonparametric paths
- analytical PK plus ODE PK, PBPK, DDE, TMDD, TTE, categorical/count PD
- NONMEM-style control-stream parsing and runtime execution for the supported subset
- desktop GUI across data, model, fit, diagnostics, plots, NCA, covariates, and advanced workflows
- broad external-validation coverage against `NONMEM`, `Monolix`, `nlmixr2`,
  `WinNonlin`, `PKNCA`, `Pharmpy`, and live R-backed checks
- optimal-design / PFIM entry points in API and GUI
- SBML import, HTML reports, NONMEM-like output files, and CDISC helper exports

OpenPKPD is already a serious pharmacometrics workbench. The current gap is
competitive maturity, not feature existence.

---

## Priority Summary

### P0 — Highest-value work

| Area | Why it matters competitively | Current state | Next action |
|------|-------------------------------|---------------|-------------|
| **Advanced-estimator credibility** | `Monolix`, `Pumas`, and `nlmixr2` win on trust in stochastic estimation workflows | NUTS, SAEM, IMP, and Bayes exist, but empirical validation depth is uneven | expand empirical benchmark matrix and document operational envelopes |
| **Parser/runtime/export parity** | `NONMEM` and `Pharmpy` users care about predictable control-stream workflows and round-trip interoperability | parser support is broader than full runtime/export support; writer/round-trip story is still incomplete | publish executable support matrix and finish high-value writer/export gaps |
| **Performance measurement and profiling** | `Pumas`, `Monolix`, and compiled toolchains win on transparent speed expectations | JIT work exists, but performance guidance is fragmented and benchmark breadth is thin | add repeatable benchmark suites and profiling reports for estimation/simulation paths |
| **GUI workflow polish** | `WinNonlin` and `Monolix` compete on coherence, not just features | GUI exists and is tested, but result comparison, simulation-first work, and review UX still lag | improve artifact comparison, run review, and scenario navigation |
| **Reference-grade docs and examples** | users judge migration viability against `NONMEM`, `Monolix`, `nlmixr2`, and `Pharmpy` from docs first | docs are much better, but some advanced areas still read like technical notes rather than polished user workflows | promote benchmark-backed workflows and tighten support/limitation statements |

### P1 — Important but second-order

| Area | Current state | Main gap |
|------|---------------|----------|
| PFIM / optimal design | implemented and initially validated | validation breadth and end-user workflow polish still trail specialist tools |
| CDISC / reporting | helpers exist | not close to a submission-grade or WinNonlin-like reporting stack |
| HPC / parallel workflows | multiprocessing, Dask, Ray, MPI hooks exist | too little validation and operational documentation for cluster-grade use |
| Advanced model-family examples | model breadth is strong | PBPK, TMDD, and advanced PD need more polished end-to-end tutorials |

---

## P0 — Immediate Priorities

### 1. Expand empirical validation for advanced estimators

This is the single biggest remaining credibility gap.

The FO/FOCE/FOCEI surface is no longer the weakest area. The thinner areas are:

- `IMP` / `IMPMAP` on empirical datasets
- `SAEM` beyond current benchmark slices
- `BAYES` / `NUTS` on realistic population models
- nonparametric estimation outside narrow reference cases

Required work:

1. add at least one second empirical dataset for `SAEM`, `IMP`, `BAYES`, and nonparametric methods
2. expand multi-endpoint or mixed PK/PD empirical coverage beyond the current reduced-runtime slices
3. publish method-by-method "validated on / not yet validated on" notes in docs
4. keep reference fixtures tied to external tools or literature, not only internal regression baselines

Why now:

- `Monolix`, `Pumas`, and `nlmixr2` are trusted because users know where their stochastic workflows hold up
- OpenPKPD already has the methods; it now needs the validation matrix

### 2. Fix the Bayesian/NUTS maturity gaps

This is real technical debt in shipped functionality.

Current issues:

- the standalone `nuts_estimate()` helper still returns `r_hat = ones`
- NUTS is still theta-only rather than full joint population posterior sampling
- finite-difference gradients remain expensive for larger models
- primary Bayesian support boundaries are still narrower than the commercial-tool surface

Priority order:

1. decide and document the supported NUTS surface
2. either improve the native path or add a stronger backend with clear support boundaries
3. add empirical NUTS benchmark runs with explicit runtime + diagnostic expectations
4. document Bayesian limitations versus `Monolix` / `Pumas` rather than implying parity

### 3. Publish a clear parser/runtime/export support contract

This matters for `NONMEM` migration and for `Pharmpy`-style workflows.

The repo already supports useful runtime subsets for records such as:

- `$SIMULATION`
- `$MIXTURE`
- `$PRIOR`

But users still need a sharper answer to:

- what parses
- what runs
- what round-trips
- what exports in NONMEM-like form
- what combinations are intentionally rejected

Required work:

1. document the control-stream support matrix as a maintained source of truth
2. finish or explicitly defer the control-stream writer / round-trip story
3. add tests for the highest-value record combinations, not only single-record support
4. align migration docs with actual runtime behavior

### 4. Establish benchmark and profiling suites as first-class deliverables

If OpenPKPD wants to compete with `NONMEM`, `Monolix`, `nlmixr2`, and `Pumas`,
performance claims need repeatable evidence.

Needed additions:

1. estimation benchmarks for FOCEI, SAEM, IMP, BAYES, and NUTS on shared datasets
2. simulation/VPC benchmarks across analytical PK, ODE PK, PBPK, and mixed workflows
3. profiling reports that identify current hotspots in:
   - FOCE outer loop
   - ODE sensitivity path
   - simulation/VPC pipelines
   - GUI result loading and artifact rendering
4. benchmark baselines checked into the repo or docs with exact environment notes

Priority comparison targets:

- `NONMEM` / `Monolix` / `nlmixr2` for estimation workflows
- `WinNonlin` / `PKNCA` for NCA/report semantics
- `Pumas` for ODE-heavy and sensitivity-aware workloads

### 5. Improve GUI competitiveness instead of just feature presence

The GUI is no longer a missing feature. The gap is that it still feels more like
a capable engineering shell than a polished analyst environment.

Highest-value work:

1. strengthen run-to-run result comparison and artifact diff workflows
2. make simulation-first and review-first workflows more obvious
3. improve scenario navigation and output inspection for multi-run projects
4. keep growing workflow-level GUI regression tests around persistence and recovery

Competitive lens:

- `WinNonlin` and `Monolix` win here on coherence and reviewability
- OpenPKPD should target "confident daily use", not only "feature exists"

---

## P1 — Functional Gaps Worth Closing Next

### 6. Deepen PFIM / optimal-design maturity

Current state:

- design API exists
- GUI path exists
- example exists
- initial PFIM-backed validation exists

Main gaps:

- too few reference families
- too little coverage for nonlinear or covariate-rich design scenarios
- not enough docs telling users where the design engine is reliable

Why it matters:

- this is a place where `PFIM`, `NONMEM`, and `Monolix`-adjacent workflows set expectations

### 7. Harden ADVAN13 sensitivity and ODE-gradient performance

Current state:

- `solve_with_sensitivity()` exists
- sensitivity tests exist
- some FOCE sensitivity wiring exists

Main gaps:

- sensitivity is not yet the default high-confidence path everywhere users expect it
- ODE-heavy estimation still pays too much finite-difference cost
- this directly affects competitiveness against `Pumas` and mature compiled tools

This is both a correctness-adjacent and performance-adjacent priority.

### 8. Clarify the status of FREM and AMD

These are implemented enough that they should not be described as empty
scaffolds, but they are not yet competitive with the strongest `Pharmpy` /
`NONMEM` workflow story.

FREM:

- currently behaves more like a practical approximation than a full polished FREM platform
- needs stronger external validation and clearer documentation of scope

AMD:

- structural search and covariate screening paths exist
- still needs more validation, richer search strategy coverage, and clearer success criteria

Priority:

1. document current scope honestly
2. add at least one benchmark-backed end-to-end example
3. avoid overstating parity with `Pharmpy`

### 9. Expand CDISC / report-output maturity

Current state:

- CDISC helpers exist
- HTML reports exist
- NONMEM-like outputs exist

Main gaps:

- ADPPK is helper-grade rather than full-domain grade
- submission-style completeness and validation are not there
- report polish and metadata completeness still lag `WinNonlin`

This is not the top engineering priority, but it is an important user-trust gap.

---

## P1 — Testing Priorities

These are specific test expansions with high return on effort.

1. Add exact analytical checks for `SimulationEngine` trajectories and noise semantics.
2. Add stored regression baselines for SSE outputs.
3. Convert more plot tests from smoke checks to data-bearing assertions.
4. Add fixed-dataset regression fits for PD and PK/PD model families.
5. Add broader PBPK / advanced PD / TMDD empirical or literature-backed benchmarks.
6. Add more multi-record control-stream runtime tests, especially around supported/rejected combinations.
7. Add stronger validation for Dask / Ray / MPI execution paths.
8. Add API-vs-GUI consistency tests for optimal-design, diagnostics, and reporting workflows.

---

## P1 — Documentation Priorities

### 10. Tighten support statements and migration docs

The docs should answer "can I rely on this?" as directly as the code does.

Needed work:

1. remove any remaining doc statements that imply round-trip or runtime support beyond the current implementation
2. add a maintained support matrix for control-stream records and record combinations
3. document which benchmark pages are release-gated versus illustrative
4. make competitive limitations explicit where `NONMEM`, `Monolix`, `WinNonlin`, `nlmixr2`, `Pharmpy`, or `Pumas` are still materially stronger

### 11. Add polished benchmark-backed workflow examples

Highest-value examples to add or improve:

1. one covariate-rich NONMEM-style workflow
2. one joint PK/PD or mixed-endpoint workflow
3. one PBPK or advanced ODE walkthrough
4. one "migration from NONMEM to OpenPKPD" example with explicit supported subset notes

These examples should read like production user guidance, not just developer demos.

---

## Suggested Execution Order

If engineering capacity is limited, the best order is:

1. advanced-estimator empirical validation matrix
2. parser/runtime/export support contract plus writer/round-trip clarification
3. benchmark and profiling suite establishment
4. Bayesian/NUTS maturity improvements
5. GUI result comparison and review workflow improvements
6. PFIM/design validation expansion
7. FREM/AMD scope clarification plus validation-backed examples
8. CDISC/report polish

---

## What Not To Misclassify

These are important, but should not be treated as the top blockers right now:

- adding more raw feature checkboxes without validation
- broad GUI redesign before workflow review/comparison problems are solved
- regulatory-pack claims beyond the current research-grade scope
- performance claims without reproducible benchmark evidence

---

## Working Rule For Future Updates

This document should be updated only from:

- current source code
- current tests
- current docs
- current benchmark/profiling evidence

Avoid carrying forward stale backlog statements such as:

- "exists only as scaffolding" when a working implementation is present
- "no external validation" when the actual gap is limited breadth or depth
- "not merged" when the code is already in-tree but still under-tested

---

## Top 5 Execution Tickets

These are the next concrete tickets to execute from this priority document.
Each ticket is timestamped so later reviews can distinguish fresh priorities
from stale carryover items.

### RP-001 — Advanced-estimator empirical benchmark expansion
*Created: 2026-03-29*
*Priority: P0*

Objective:

- expand empirical benchmark coverage for `SAEM`, `IMP` / `IMPMAP`, `BAYES`, and nonparametric estimation beyond the current narrow slice of datasets and method-specific anchors

Scope:

- add at least one additional empirical dataset for advanced estimators
- add at least one mixed-endpoint or PK/PD benchmark path
- add method-by-method benchmark expectations and tolerances in docs
- keep external anchors tied to `NONMEM`, `Monolix`, `nlmixr2`, `Pharmpy`, literature, or published reference outputs where feasible

Acceptance criteria:

1. New tests land under `tests/external_validation/` or `tests/regression/` for at least two advanced estimator families.
2. Each added benchmark has an explicit external or literature-backed reference.
3. Docs state which advanced estimators have empirical external validation and on which datasets.
4. Release-facing docs stop implying parity outside the validated envelope.

Suggested file areas:

- `tests/external_validation/`
- `tests/regression/`
- `docs/user_guide/external_validation_benchmarks.md`
- `docs/user_guide/validation.md`

Status note (2026-03-29): Empirical benchmark coverage and docs were expanded for
advanced estimators, including an IMP empirical reference suite, measured
theophylline IMP budget calibration, and a documented validated-estimator
envelope. The ticket remains open because mixed-endpoint/PKPD and broader
nonparametric empirical anchors are still missing.
Status note (2026-03-29): Added one more empirical advanced-estimator anchor by
promoting BAYES(Laplace) on the NONMEM 402 two-compartment IV benchmark into
the documented external-validation matrix (`4 passed` in `129.43s`). This gives
BAYES(Laplace) a second model-family surface beyond the 1-compartment oral PK
datasets, though nonparametric empirical coverage is still thin.
Status note (2026-03-29): Warfarin IMP `KA` diagnostics now indicate a basin /
initialization problem more than an ESS-collapse problem. Raw `IMP` converges
almost immediately with negligible OFV improvement and misses `KA`, while a
FOCEI warm start lands near the validated basin but can still hit the outer
budget. `IMPMAP` is no longer only a label variant: it now uses a FOCEI warm
start before the IMP outer optimization, which gives it a real basin-capture
distinction from raw `IMP`.
Status note (2026-03-29): Warfarin empirical validation is now routed through
`IMPMAP(isample=60, maxeval=12)`, and the recommendation test proving
`IMPMAP` > raw `IMP` on warfarin passed (`3 passed` in `259.34s`). This closes
the old warfarin `KA` xfail as a validation-contract issue, but it also makes
the runtime cost of the FOCEI warm start part of the real supported path.
Status note (2026-03-29): The paired benchmark artifact for `IMP` vs `IMPMAP`
now exists at `artifacts/profiling/estimation_baseline.json`. On the current
small-PK benchmark, `IMP` took `38.74s` and did not converge, while `IMPMAP`
took `65.89s` and did converge; the incremental warm-start cost was about
`25.18s`. This means the repo now has a measured accuracy/runtime tradeoff for
the recommended warfarin MAP-style path.
Status note (2026-03-29): Nonparametric estimation now has a real empirical
Pharmpy-backed benchmark on the bundled `pheno` dataset. The new external
validation class passed (`5 passed` in `30.93s`) and documents a converged
support-point fit that stays near the Pharmpy fixed-effects / residual-scale
basin without collapsing the support weights. This closes the old "no
empirical nonparametric anchor" gap, though coverage is still only one dataset.
Status note (2026-03-30): Four new external-validation benchmark classes added
(`34 passed` in `219s`):
- `TestWarfarinSAEMvsNlmixr2` — SAEM on Warfarin PK (32 subjects) vs existing
  `warfarin_pk_saem.json` nlmixr2 reference; KA, V, and sigma are release-gated;
  CL has a documented systematic SAEM gap and is tracked with a wide safety-net
  assertion rather than hidden.
- `TestPhenobarbitalSAEMvsLiterature` — SAEM on neonatal phenobarbital (59
  subjects) vs Grasela & Donn (1985) published FO NONMEM reference; CL/kg within
  35%, V/kg within 25%, half-life within 40% of literature. Provides a second
  published-literature SAEM anchor beyond theophylline/Monolix.
- `TestPhenobarbitalBayesLaplaceEmpirical` — BAYES(Laplace) on phenobarbital vs
  Grasela 1985; CL/kg within 25%; V/kg test documents known FO-vs-FOCEI basin
  difference on sparse IV data (FOCEI MAP finds V/kg ≈ 2.5 vs FO 0.96) and
  uses a physiological-range assertion instead of a literature-match assertion.
  Extends BAYES(Laplace) coverage to a second dataset (beyond theophylline and
  NONMEM Run 402).
- `TestTheophyllineNonparametricEmpirical` — Nonparametric NPML on Boeckmann
  theophylline (12 subjects, oral) vs nlmixr2 FOCEI basin; CL and V within 20%;
  KA physiologically plausible. Provides a second nonparametric dataset anchor
  (oral vs pheno IV) and confirms support-point non-collapse.
The remaining gap for RP-001 is mixed-endpoint PK/PD benchmark coverage and a
second IMP/IMPMAP dataset with an external (NONMEM/Monolix) reference.

### RP-002 — Control-stream support matrix and round-trip contract
*Created: 2026-03-29*
*Priority: P0*

Objective:

- make the parser/runtime/export contract explicit for users migrating from `NONMEM` or using `Pharmpy`-style workflows

Scope:

- document what parses, what runs, what round-trips, and what combinations are intentionally unsupported
- close the highest-value gaps in the control-stream writer / round-trip story or explicitly defer them
- add tests for supported and rejected multi-record combinations

Acceptance criteria:

1. One source-of-truth support matrix exists in docs and is referenced from migration pages.
2. Round-trip claims are backed by tests for the supported subset.
3. Unsupported combinations fail with stable, documented runtime messages.
4. README and migration docs use the same support language as the matrix.

Suggested file areas:

- `docs/user_guide/control_stream.md`
- `docs/getting_started/migrating_from_nonmem.md`
- `README.md`
- `tests/integration/`
- writer/export implementation files under `src/openpkpd/`

Status note (2026-03-29): Control-stream support language was tightened in the
user guide and migration docs, supported round-trip behavior is now backed by
unit tests, and curated `$PRIOR` / `$SIMULATION` examples were added. The ticket
is materially advanced; remaining work is mostly broader runtime rejection
coverage and README alignment.
Status note (2026-03-29): `$PRIOR` now also has runner-level integration
coverage in `tests/integration/test_control_stream_prior.py`. The native runner
is explicitly pinned to (1) wrap parsed prior records into a
`PriorAugmentedModel` before estimation, (2) write the standard `.lst/.ext/.phi`
artifacts for a prior-augmented run, and (3) fail clearly when an incomplete
runtime prior pair such as `$THETAP` without `$THETAPV` is supplied. That closes
an end-to-end validation gap between the documented prior subset and actual
runner behavior.
Status note (2026-03-29): The same integration coverage now also pins the
parse-only side of the contract: `$SIGMAP` / `$SIGMAPD` records are parsed but
do not activate runtime priors in `run_model()`. That reduces the risk of the
control-stream docs drifting away from actual runner semantics for the prior
record family.
Status note (2026-03-29): Parse-only runner semantics are now also pinned for
`$NONPARAMETRIC`: a control stream containing `$NONPARAMETRIC NPSUPP=... MCETA=...`
still follows the declared `$ESTIMATION` runtime path unchanged. That gives the
control-stream support matrix a concrete regression test for one more
high-migration parse-only record family.

### RP-003 — Benchmark and profiling suite establishment
*Created: 2026-03-29*
*Priority: P0*

Objective:

- make performance claims reproducible and actionable against the comparison tools

Scope:

- add repeatable benchmark entry points for estimation and simulation workflows
- add profiling workflows for the main hot paths
- publish environment assumptions and benchmark methodology

Acceptance criteria:

1. A documented benchmark suite exists for FOCEI, SAEM, IMP, BAYES, NUTS, and simulation/VPC-heavy paths.
2. At least one profiling report identifies the dominant hotspots in estimation and simulation.
3. Benchmarks are runnable through a stable repo entry point such as `just`, `pytest -m benchmark`, or documented scripts.
4. Benchmark docs clearly separate measured results from aspirational targets.

Suggested file areas:

- `tests/benchmarks/`
- `justfile`
- `docs/user_guide/performance_analysis_simulation_report.md`
- `docs/user_guide/performance_profiling_report.md`
- new benchmark/profiling docs as needed

Status note (2026-03-29): Stable benchmark entry points now exist through
`just benchmark-estimation` and `just profile-analysis`, the benchmark harness
covers FO/FOCE/FOCEI/SAEM/IMP/BAYES/NUTS, and the profiling docs now describe
how to reproduce measured runs. The ticket remains open for stored baselines
and broader simulation/VPC benchmarking.

### RP-004 — Bayesian/NUTS maturity hardening
*Created: 2026-03-29*
*Priority: P0*

Objective:

- turn the Bayesian/NUTS surface from "promising and real" into "narrow but trustworthy"

Scope:

- define the supported NUTS surface explicitly
- remove misleading diagnostics behavior
- improve performance and validation on realistic models
- document limitations relative to `Monolix` and `Pumas`

Acceptance criteria:

1. `nuts_estimate()` no longer silently reports meaningless all-ones `r_hat`, or its limitations are made impossible to miss.
2. Bayesian docs clearly distinguish Laplace, NumPy NUTS, and optional backends by support level.
3. At least one empirical Bayesian/NUTS benchmark with runtime and diagnostic expectations is added.
4. User-facing docs no longer imply full Bayesian NLME parity where it does not exist.

Suggested file areas:

- `src/openpkpd/estimation/nuts.py`
- `src/openpkpd/estimation/bayes.py`
- `tests/unit/estimation/`
- `tests/external_validation/`
- `docs/user_guide/estimation_methods.md`

Status note (2026-03-29): Standalone `nuts_estimate()` no longer reports fake
all-ones `r_hat`, unit coverage was updated accordingly, and Bayesian docs now
differentiate built-in NUTS, Laplace, and optional backends with explicit
limitations. The ticket remains open for broader empirical Bayesian benchmark
coverage.
Status note (2026-03-29): The built-in `BAYES(NUTS)` path is now explicitly
classified as a second-tier empirical route rather than a primary release-gated
benchmark surface. A bounded 6-subject oral-PK probe with `n_samples=12`,
`tune=8`, and `n_chains=2` took `26.43s` and still failed convergence
(`R-hat=[1.10, 2.31, 1.28]`), which is consistent with the current theta-only,
finite-difference FOCE-backed design. The current recommendation is to prefer
`BAYES(Laplace)` for fast weak-prior summaries and `PyMC`/stronger backends for
primary MCMC workflows.
Status note (2026-03-29): Native NUTS now emits run diagnostics through
`result.diagnostics["nuts"]`, including per-chain step-size / tree-depth /
acceptance summaries plus log-probability and FOCE call counters. A bounded
benchmark run on the shared 6-subject oral-PK workload
(`n_samples=12`, `tune=8`, `n_chains=2`) took `48.68s` with
`log_prob_calls=2200`, `foce_inner_calls=2105`, and `foce_inner_seconds=47.13`.
That is strong evidence that the main hardening lever is cheaper posterior
evaluation and gradients, not a sampler-only rewrite.
Status note (2026-03-29): Exact `theta`-value caching is now active inside the
native NUTS sampler, which cuts repeated FOCE-backed posterior evaluations
within finite-difference gradient and tree-building paths. Rerunning the same
bounded 6-subject oral-PK probe after that change reduced runtime to `18.99s`
with `log_prob_calls=872`, `foce_inner_calls=796`, and
`foce_inner_seconds=18.37`; sampler diagnostics now also expose
`log_prob_cache_hits` / `log_prob_cache_misses` per chain. The path is still
second-tier and not yet converged at that budget, but the dominant cost was
reduced materially without loosening the support claim.
Status note (2026-03-29): The FOCE-backed NUTS log-posterior now also keeps a
bounded recent-theta warm-start cache for `eta_hat`, so unique nearby theta
proposals seed the inner loop from the closest previously solved conditional
mode instead of a cold zero start. On the same bounded 6-subject oral-PK probe,
that reduced runtime further to `17.33s` and `foce_inner_seconds` to `16.51s`
without changing the number of unique posterior evaluations
(`log_prob_calls=872`, `foce_inner_calls=796`). Diagnostics now expose
`warm_start_exact_hits`, `warm_start_nearest_hits`, and
`warm_start_cold_starts`; in the measured probe they were `4`, `791`, and `1`
respectively. The next remaining lever is cheaper gradients or a cheaper FOCE
objective per unique theta, not more warm-start plumbing alone.
Status note (2026-03-29): The symbolic analytical derivative path is now part
of the tested dev workflow, can load from prewarmed source caches even when
live `sympy` is unavailable, and the shared estimation benchmark helper now
builds the standard oral-PK workload through compiled `ModelBuilder`
callables instead of a bare `PopulationModel`. On the same bounded
6-subject oral-PK `BAYES(NUTS)` benchmark (`n_samples=12`, `tune=8`,
`n_chains=2`), that switched the run onto the analytic theta-gradient path
(`used_analytic_theta_gradient=true`, `used_fd_gradient=false` on both chains)
and cut runtime to `8.53s` with `log_prob_calls=327`,
`foce_inner_calls=326`, and `foce_inner_seconds=6.98`. That materially narrows
the cost gap for the supported analytical subset while keeping the overall NUTS
support statement narrow rather than overstated.
Status note (2026-03-29): `BAYES(NUTS)` now also has a second-tier empirical
theophylline benchmark in `tests/external_validation/`. On the measured budget
(`n_samples=24`, `tune=16`, `n_chains=2`) it stayed close to the validated
`nlmixr2` FOCEI basin (`KA/CL/V` relative errors about `1.33% / 0.50% / 1.16%`)
while using the analytic symbolic path
(`used_analytic_theta_gradient=true`, `used_fd_gradient=false` on both chains),
but it still required about `28.08s` and only reduced max `R-hat` to `1.19`.
That is enough for a real empirical benchmark, but it still supports the
current positioning of native NUTS as a dependency-free second-tier workflow
rather than the default Bayesian release path.

### RP-005 — GUI result review and comparison workflow polish
*Created: 2026-03-29*
*Priority: P0*

Objective:

- improve the parts of the GUI that most affect analyst confidence during repeated model-review cycles

Scope:

- strengthen run-to-run artifact comparison
- improve result review for multi-run projects
- make simulation-first and review-first navigation more obvious
- keep regression coverage aligned with workflow behavior

Acceptance criteria:

1. Users can compare two or more result sets with clear artifact-level summaries.
2. Multi-run project navigation supports practical review workflows without hidden state traps.
3. GUI tests cover the main comparison, persistence, and recovery paths.
4. The GUI help/docs describe the intended review workflow instead of only the screen layout.

Suggested file areas:

- `src/openpkpd_gui/workflows/results_workflow.py`
- related GUI services/widgets
- `tests/unit/gui/`
- `docs/user_guide/gui.md`

Status note (2026-03-29): The result-review docs now describe comparison and
artifact-delta workflows, and the in-tree GUI workflow/tests already cover the
comparison surface added in this pass. The ticket remains open for additional
multi-run workflow polish and recovery-path breadth.

---

## Ticket Maintenance Rule

When a ticket above materially changes state, update:

1. its timestamp block
2. its acceptance criteria if scope changed
3. this document's `Last updated` date

Prefer appending status notes in the form:

- `Status note (YYYY-MM-DD): ...`
