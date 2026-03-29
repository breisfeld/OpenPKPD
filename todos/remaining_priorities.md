# OpenPKPD — Remaining Priorities
*Last updated: 2026-03-29*

This document reflects what has shipped and what remains, based on the
`gap_analysis_auggie.md` backlog and the work completed in the March 2026
development sprint.

---

## What Shipped (March 2026 Sprint)

| Task | Notes |
|------|-------|
| P1.15 ODE JIT (LLC/Numba tier) | 10–30× speedup; stiff ODE fallback to Radau |
| P1.17 Monolix migration guide | Full docs/user_guide/monolix_migration.md |
| P1.10 MCMC diagnostics | split-R-hat, ESS, autocorr; `mcmc_diagnostics.py` |
| P1.8 Bayesian NUTS (pure-NumPy) | NUTSSampler wired as default backend |
| P1.8 FOCE marginal likelihood | Replaces eta=0 approximation; warm-started |
| GUI: BAYES in method selector | ESTIMATION_METHODS + results display |
| Notebooks 03 + 08 | Bayesian NUTS worked example; MCMC diagnostic plots |
| Comprehensive NUTS tests | 41 tests covering leapfrog, tree, accuracy, ESS |

---

## Known Technical Debt (introduced this sprint)

These are limitations deliberately deferred for speed; they should be
addressed before calling P1.8 production-ready.

| Item | Detail | Effort |
|------|--------|--------|
| **NUTS: theta-only sampling** | Omega and Sigma held fixed during NUTS. Full joint posterior requires sampling Omega/Sigma too (Wishart/InvWishart priors on Omega, Half-Normal on Sigma diag). | M |
| **NUTS: no autodiff gradient** | FD gradient costs `2·n_theta` FOCE evaluations per leapfrog step. For n_theta>10 this is slow. Options: (a) evaluate `nutpie` standalone API, (b) implement adjoint gradient through `_outer_ofv`. | L |
| **`nuts_estimate()` R-hat=ones** | The single-chain wrapper always returns `r_hat=ones`. It should either run multiple chains or document that callers must use `BAYESMethod._estimate_nuts()` for real R-hat. | S |
| **NumPyro backend untested** | Demoted from auto-selection due to Intel macOS jaxlib incompatibility. Still callable via `backend="numpyro"` but has no CI coverage on supported platforms. | S |
| **ADVAN13 adjoint marked done in gap analysis but not integrated with NUTS** | The FD gradient in NUTS calls `_outer_ofv` which internally uses FD for `_compute_G_i`. True adjoint gradient would require wiring `solve_with_sensitivity` into `_outer_ofv`. | L |

---

## P0 — Still Outstanding

These were in the original P0 backlog and remain incomplete despite the
gap analysis marking them "done" (the implementations were stubs or had
test coverage gaps).

| # | Task | Honest Status | Effort |
|---|------|--------------|--------|
| P0.3 | **ADVAN13 adjoint-sensitivity gradients** | Stub exists; sensitivity ODE not wired into FOCE outer gradient. `_compute_G_i` uses FD, not true adjoint. | L |
| P0.6 | **IOV gradient through ODE models (ADVAN6/8)** | IOV block-sparse structure implemented for analytical ADVANs. Not validated for ODE-based models where `_compute_G_i` spans occasion boundaries. | M |
| P0 | **Comprehensive tests** | Regression tests exist for Theophylline/Warfarin. Moxonidine dataset and external-validation comparisons (vs. NONMEM reference output) are incomplete. | M |

---

## P1 — Still Outstanding

| # | Task | Status | Effort |
|---|------|--------|--------|
| P1.7 | **ADVAN7 (matrix-exponential)** | ADVAN5 done. ADVAN7 `scipy.linalg.expm` path marked in-progress but not merged. | S |
| P1.11 | **NONMEM control-stream writer (round-trip)** | Parser reads `.ctl`; writer was added but `to_nmtran()` coverage incomplete for all record types. | M |
| P1.12 | **FREM covariate method** | `frem.py` scaffolded; not externally validated against Pharmpy FREM output. | L |
| P1.14 | **AMD pipeline** | `amd.py` scaffolded; structural + covariate search loop not end-to-end. | XL |
| P1.16 | **CDISC ADPPK full domain export** | PP domain partial; required variables per CDISC ADPPK IG missing. | M |
| P1 | **Comprehensive tests** | Integration/regression tests across all P1 features incomplete. No external-validation suite comparing SAEM/FOCE output to Monolix/NONMEM on shared datasets. | L |

---

## Suggested Priority Order (next 4–6 weeks)

### Immediate (correctness gaps in shipped work)

1. **NUTS: add Omega/Sigma sampling** (P1.8 completeness)
   Wire Wishart prior on Omega and Half-Normal prior on Sigma diagonal into
   the NUTS state vector. This is the most significant missing piece for
   true Bayesian NLME inference.

2. **Evaluate `nutpie` as NUTS backend** (performance)
   `nutpie` is a Rust NUTS with a standalone Python log-density API.
   Evaluate whether its standalone interface works without PyMC, and if so,
   add as `backend="nutpie"` in `BAYESMethod`. Potentially eliminates the
   FD gradient bottleneck.

3. **P1.7 ADVAN7 — close out** (quick win, in-progress)
   The `scipy.linalg.expm` path is nearly done. Complete, test, merge.

### Short-term (2–3 weeks each)

4. **P0 external validation suite**
   Run NONMEM 7.6 and Monolix 2024R1 on Theophylline, Warfarin, and
   Moxonidine datasets. Assert OpenPKPD FOCE/SAEM OFV and parameter
   estimates agree within tolerance. This is the hardest test of
   estimation correctness and the most credible thing to show regulators.

5. **P1.11 control-stream writer completion**
   Full round-trip for all record types. Enables the Pharmpy/NONMEM
   interop story and is prerequisite for AMD.

6. **P1.12 FREM external validation**
   Run against Pharmpy FREM on a published dataset. This is the only
   credible way to validate the implementation.

### Medium-term (1–2 months each)

7. **P0.3 ADVAN13 true adjoint gradient**
   Replace FD in `_compute_G_i` with the ODE sensitivity system.
   Prerequisite for competitive FOCE performance on large ODE models.

8. **P1.14 AMD pipeline end-to-end**
   Connect structural search + stepwise covariate search + model scoring.
   High user-facing value; requires P1.11 and P1.12 first.

9. **P1 comprehensive tests + external validation**
   Before calling P1 done: automated comparison against NONMEM/Monolix
   reference runs for every estimation method shipped.

### Deferred (P2/P3 — see gap_analysis_auggie.md)

- GUI polish (real-time OFV trace, interactive VPC, SCM visualisation)
- CDISC ADPPK domain completion
- Regulatory tooling (GxP audit trail, IQ/OQ/PQ validation pack)
- Rust NUTS extension (`openpkpd-nuts-rs`) for maximum sampling speed
- JAX/GPU end-to-end FOCE/SAEM

---

## Reference

Full competitive gap analysis: `todos/gap_analysis_auggie.md`
