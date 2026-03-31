# External validation benchmarks

This note records the current **cross-tool benchmark fixtures** that are bundled
with the repository and exercised by `tests/external_validation/`.

For the broader support classification across estimators and workflows, see
[`validation_matrix.md`](validation_matrix.md).

## Validated estimator envelope

The table below is the quickest summary of what currently has **empirical**
external validation in-tree. Treat it as a release-facing support summary, not
as a claim of blanket parity with commercial tools.

| Estimator family | Empirical external anchors in-tree | Current interpretation |
|---|---|---|
| FO / FOCE / FOCEI | `nlmixr2`, `NONMEM`, `Pharmpy`, Bauer tutorial workflows, covariate models | strongest cross-tool evidence in the repository |
| SAEM | `Monolix` theophylline; `nlmixr2` warfarin PK; reduced `nlmixr2` warfarin PK/PD | credible on current benchmark slices, but not yet broad enough for blanket parity claims |
| BAYES(Laplace) | `nlmixr2` theophylline, warfarin PK, and reduced warfarin PK/PD basins; `NONMEM` 402 empirical basin | useful weak-prior Bayesian summary path; still an approximation, not full MCMC parity |
| BAYES(NUTS) | internal statistical validation, bounded synthetic benchmark, and second-tier empirical theophylline benchmark | real implementation, but currently a second-tier empirical path rather than a primary release-gated benchmark surface |
| IMP / IMPMAP | theophylline and warfarin PK empirical basin checks; analytic Gaussian-reference tests | numerically credible on the current empirical slices, but still narrower than FO/FOCEI validation; benchmark budgets matter materially |
| Nonparametric | Pharmpy `pheno` empirical benchmark plus regression/exact reference tests | credible on one real dataset now, but still thin relative to FO/FOCEI coverage |

When adding new claims to README or migration docs, prefer the wording in this
table over generic statements like "validated" or "production-ready".

## Monolix: theophylline SAEM

- **Reference source:** public Monolix theophylline project parameters exposed by
  `monolix2rx`'s conversion article/README.
- **Fixture:** `tests/external_validation/reference/monolix_theophylline_saem.json`
- **Test:** `tests/external_validation/test_vs_monolix.py`
- **Comparison mode:** OpenPKPD fits the Boeckmann theophylline dataset with
  `ADVAN2` + SAEM after converting dose to the Monolix project convention
  (`AMT / WT`, i.e. mg/kg).

The dose normalization step is now tested explicitly: using the raw NONMEM-style
`AMT` values without conversion produces materially worse agreement with the
public Monolix fixed effects, so the benchmark guards both the estimator and the
benchmark setup itself.

Reference values on the natural scale are approximately:

| Parameter | Monolix reference |
|---|---:|
| `ka_pop` | 1.533 h⁻¹ |
| `Cl_pop` | 0.0402 L/h/kg |
| `V_pop` | 0.4555 L/kg |

Current OpenPKPD results land within a few percent of those public Monolix
values with a fixed seed and a moderate SAEM schedule (`200 + 100` iterations).

## Advanced estimators: empirical benchmark notes

### SAEM

- **Anchors:** public Monolix theophylline SAEM reference; `nlmixr2` warfarin
  PK SAEM reference; reduced `nlmixr2` warfarin PK/PD mixed-endpoint basin;
  Grasela & Donn (1985) neonatal phenobarbital FO NONMEM reference
- **Tests:** `tests/external_validation/test_vs_monolix.py`,
  `tests/external_validation/test_saem_reference.py`,
  `tests/external_validation/test_vs_nlmixr2.py`
- **Current interpretation:** SAEM now has three external anchors — Monolix
  theophylline, nlmixr2 warfarin PK (`TestWarfarinSAEMvsNlmixr2`: KA, V, and
  σ within 30% of the reference and CL now within the same tracked acceptance
  envelope after the current direct M-step update), and Grasela & Donn (1985)
  phenobarbital (`TestPhenobarbitalSAEMvsLiterature`: CL/kg within 35%, V/kg
  within 25%, half-life within 40% of the published FO NONMEM values). Broader
  PBPK-style empirical coverage is still missing.
- **Reduced mixed-endpoint benchmark (2026-03-31):**
  - `TestSAEMWarfarinPKPDReducedVsNlmixr2` now validates a short-schedule SAEM
    run on the 4-subject reduced warfarin PK/PD mixed-endpoint fixture with
    `jit="llc"` and a `20 + 10` iteration schedule
  - measured run time on the current local machine is about `3.5s`
  - structural terms stay essentially on top of the bundled reduced `nlmixr2`
    reference basin, while `PK_PROP_ERR` remains the loosest tracked term at
    about `+16%` under the short speed-gating schedule
  - this promotes SAEM into the reduced mixed-endpoint external-validation
    surface, while still leaving broader mixed-endpoint and PBPK-style breadth
    as open work

### BAYES(Laplace)

- **Anchors:** `nlmixr2` FOCEI basins for theophylline and warfarin PK, the
  reduced `nlmixr2` warfarin PK/PD mixed-endpoint basin, plus the `NONMEM` 402
  two-compartment IV empirical basin
- **Tests:** `tests/external_validation/test_bayes_empirical_reference.py`,
  `tests/external_validation/test_bayes_nonmem_402_diagnostics.py`
- **Current interpretation:** the weak-prior Laplace path stays near validated
  empirical basins and produces stable posterior summaries, but it remains a
  Gaussian approximation rather than a full MCMC benchmark.
- **Reduced mixed-endpoint benchmark (2026-03-31):**
  - `TestWarfarinPKPDReducedBayesLaplaceEmpirical` now validates the
    4-subject reduced warfarin PK/PD mixed-endpoint path against the bundled
    `nlmixr2` reference basin with `jit="llc"`, `maxeval=1`, `n_samples=10`,
    and `n_parallel=4`
  - measured run time on the current local machine is about `3.7s`
  - the covariance path now stays on `optimizer_inverse_hessian` rather than
    falling back to the `265`-probe finite-difference Hessian path
  - measured relative errors on the promoted slice are very small:
    `KTR +0.02%`, `KA +0.02%`, `CL +0.06%`, `V -0.18%`, `EC50 -0.02%`,
    `KOUT -0.05%`, `E0 +0.07%`, `PK_PROP_ERR +0.01%`, `PK_ADD_ERR +0.20%`,
    `PD_ADD_ERR +0.04%`
  - this closes the earlier "reduced mixed-endpoint Laplace is still too slow
    to promote" note; the current reduced warfarin PK/PD Laplace path is now a
    real external benchmark rather than only a profiling target
- **Measured benchmark expansion (2026-03-29):**
  - the `NONMEM` 402 BAYES(Laplace) benchmark passes on the current local 402
    fixture (`4 passed` in `129.43s`)
  - this adds a second model family beyond the 1-compartment oral PK empirical
    slices and makes BAYES(Laplace) less dependent on one narrow benchmark shape
- **Warfarin BAYES(Laplace) benchmark (2026-03-30):**
  - `TestWarfarinBayesLaplaceEmpirical` validates BAYES(Laplace) against the
    `nlmixr2` FOCEI basin on 32-subject warfarin PK: OFV ≈ −219 (real, not a
    penalty sentinel), KA within 20%, CL within 15%, and V within 15% of the
    nlmixr2 reference (`KA −9.4%`, `CL +0.4%`, `V −8.2%` on the measured run)
  - runtime ≈ 20 s; the `test_ofv_is_finite_and_not_penalty` guard checks both
    `np.isfinite(ofv)` and `ofv < 1e6` to catch any recurrence of the warm-start
    OFV sentinel bug that was fixed in the same pass (see note below)
  - **FOCE warm-start corruption fix (2026-03-30):** a three-part fix in
    `foce.py` eliminates the η̂ warm-start contamination that caused the FOCE
    outer optimizer to emit a `1e10` penalty OFV when the search passed through
    a near-singular Ω region and the cached η̂ values became stale. The fix adds
    a cold-start retry in `objective()`, a η̂ reset before final OFV evaluation
    in `_run_single()`, and a η̂ reset before best-iterate re-evaluation in
    `_maybe_promote_best_iterate()`.

### BAYES(NUTS)

- **Anchors:** exact Gaussian-reference tests, bounded synthetic runtime /
  diagnostic probes on small oral-PK workloads, and an empirical theophylline
  benchmark against the `nlmixr2` FOCEI basin
- **Tests:** `tests/external_validation/test_advanced_estimators_reference.py`,
  `tests/external_validation/test_bayes_empirical_reference.py`,
  `tests/unit/estimation/test_nuts.py`, `tests/unit/estimation/test_bayes.py`
- **Current interpretation:** the built-in NUTS backend is real and scientifically
  useful, but today it is a second-tier empirical path. It is not yet broad or
  cheap enough to serve as a primary release-gated cross-tool benchmark on
  realistic population models.
- **Measured probes (2026-03-29):**
  - on the bounded synthetic 6-subject oral-PK workload with `n_samples=12`,
    `tune=8`, and `n_chains=2`, the compiled symbolic path took about `8.53s`,
    with `used_analytic_theta_gradient=true` and `used_fd_gradient=false` on
    both chains
  - on the empirical theophylline benchmark with `n_samples=24`, `tune=16`,
    and `n_chains=2`, built-in `BAYES(NUTS)` took about `28.08s`, landed close
    to the `nlmixr2` FOCEI basin (`KA/CL/V` relative errors about
    `1.33% / 0.50% / 1.16%`), and reduced max `R-hat` to about `1.19`
  - `OMEGA` and `SIGMA` remained fixed at their starting values, which is the
    current documented theta-only contract
  - practical recommendation: use `BAYES(Laplace)` for fast weak-prior Bayesian
    summaries and prefer `PyMC` or another stronger backend for primary MCMC
    workflows when full posterior credibility matters

### IMP

- **Anchors:** `nlmixr2` FOCEI basins for theophylline and warfarin PK plus
  exact Gaussian marginal references for the IMP objective
- **Tests:** `tests/external_validation/test_imp_empirical_reference.py`,
  `tests/external_validation/test_saem_reference.py`
- **Current interpretation:** IMP now has both analytic validation and a
  two-dataset empirical sanity envelope. It is no longer "toy only", but its
  empirical coverage is still thinner than the core FO/FOCEI path.
- **Measured benchmark envelope (2026-03-29):**
  - theophylline reaches the validated `nlmixr2` basin with a fixed-seed
    budget of `isample=150, maxeval=12`
  - the earlier short budget `isample=40, maxeval=4` is numerically stable but
    does not reliably converge into the acceptance envelope
  - warfarin is now release-gated through the MAP-style path:
    `IMPMAP(isample=60, maxeval=12)` converges and stays in the validated
    basin for `KA`/`CL`/`V`
  - raw `IMP` remains more basin-sensitive on warfarin and is not the
    recommended practical path there
  - measured recommendation-test runtime for the warfarin comparison
    (`raw IMP` plus `IMPMAP`) is about `259 s` on the current test machine, so
    the warm-started path should be treated as a higher-confidence, higher-cost
    validation surface rather than a quick smoke check
  - **Reduced mixed-endpoint probe (2026-03-31):**
    - reduced 4-subject warfarin PK/PD `IMPMAP(isample=30, maxeval=4, jit="llc")`
      converges in about `50.8s` and lands close to the bundled reduced
      `nlmixr2` basin
    - however, the current run emits repeated native `CVODES mxstep` warnings,
      and disabling the narrow native warfarin PK/PD seam caused the same probe
      to time out at `90s`
    - interpretation: this is strong evidence that the mixed-endpoint IMPMAP
      path is scientifically plausible, but it is not yet quiet or cheap enough
      to promote into the stable external-validation matrix beside the reduced
      `BAYES(Laplace)` and `SAEM` tests

### Nonparametric

- **Anchors:** Pharmpy's bundled `pheno` empirical dataset and fit results;
  `nlmixr2` FOCEI basin for Boeckmann theophylline (12-subject oral PK);
  plus exact EM-weight references for the core NPML optimizer
- **Tests:** `tests/external_validation/test_vs_pharmpy.py`,
  `tests/external_validation/test_advanced_estimators_reference.py`
- **Current interpretation:** the nonparametric path is no longer validated
  only on exact toy likelihood matrices. On Pharmpy's empirical
  phenobarbital dataset it converges, preserves a non-degenerate support
  distribution, and keeps the fixed-effects / residual-scale surface close to
  the Pharmpy benchmark basin.
- **Measured benchmark envelope (2026-03-29):**
  - `NONPARAMETRIC(base_method="FOCEI", maxeval=300, max_iter=80)` on Pharmpy's
    `pheno` dataset converged successfully
  - fixed effects landed near the Pharmpy fit:
    `POP_CL=0.00469994`, `POP_VC=0.985239`, `COVAPGR=0.157775`
  - residual variance remained close to the Pharmpy reference:
    `sigma=0.013416` vs `0.013241`
  - the final support distribution did not collapse:
    `59` support points retained, largest weight about `0.086`, and `27`
    support points above `1%` mass
  - this is a real empirical operational benchmark, but it is still anchored
    to Pharmpy's FOCEI basin rather than a separate external nonparametric
    reference implementation
- **Theophylline oral NPML benchmark (2026-03-30):**
  - `TestTheophyllineNonparametricEmpirical` validates NPML on the Boeckmann
    theophylline dataset (12 subjects, oral) against the `nlmixr2` FOCEI basin:
    CL within 20% (`−0.1%` on the measured run), V within 20% (`+0.2%`), KA
    physiologically plausible (1.47 h⁻¹ vs nlmixr2 1.44 h⁻¹)
  - provides a second nonparametric dataset anchor (oral PK vs the IV-type pheno
    dataset) and confirms support-point non-collapse on a different absorption
    route

## PKNCA / Phoenix-style NCA: theophylline

- **Reference source:** the rendered PKNCA vignette *Computing NCA Parameters for
  Theophylline* plus Phoenix WinNonlin formula documentation.
- **Fixture:** `tests/external_validation/reference/pknca_theophylline_summary.json`
- **Test:** `tests/external_validation/test_vs_pknca.py`
- **Comparison mode:** OpenPKPD runs `NCAEngine(auc_method="linear-up-log-down",
  exclude_cmax=True)` on the Boeckmann theophylline profiles and compares the
  summary statistics against the vignette's published output.

Benchmarked summary targets include:

| Interval | Metric | Reference |
|---|---|---:|
| `0-24 h` | `AUClast` geometric mean [gCV%] | `74.6 [24.3]` |
| `0-Inf` | `Cmax` geometric mean [gCV%] | `8.65 [17.0]` |
| `0-Inf` | `Tmax` median [min, max] | `1.14 [0.63, 3.55]` |
| `0-Inf` | `half.life` mean [SD] | `8.18 [2.12]` |
| `0-Inf` | `AUCinf.obs` geometric mean [gCV%] | `115 [28.4]` |

Current OpenPKPD results match the public PKNCA summary closely. Separately,
OpenPKPD now has dedicated unit coverage for interpolated user-supplied
`t_last` cutoffs, which is a stricter boundary-handling check than the public
vignette summary surface itself.

## WinNonlin-backed NCA: Indometh

- **Reference source:** Han (2018), *Validation of Noncompartmental Analysis
  Performed by NonCompart R package*, Appendix A (Indometh tables validated
  against Phoenix WinNonlin).
- **Fixture:** `tests/external_validation/reference/winnonlin_indometh_nca.json`
- **Data:** `tests/external_validation/data/indometh.csv`
- **Test:** `tests/external_validation/test_vs_winnonlin_indometh.py`
- **Comparison mode:** the repository now exercises four public Indometh
  surfaces:
  - a zero-start comparison that prepends an explicit zero-concentration row at
    dose time and checks the shared dense-profile core endpoints
  - an IV bolus comparison that back-extrapolates `C0`, includes the `0 → first
    sample` contribution in AUC/AUMC, and checks `C0`, `AUClast`, `AUMClast`,
    `AUCinf`, `AUMCinf`, `CL`, `Vz`, and `MRT` against the published WinNonlin
    tables
  - an IV infusion (0.25 h) comparison that checks the published exposure,
    terminal, AUMC, and infusion-adjusted `MRT` endpoints
  - an extravascular comparison that checks the published zero-start exposure,
    AUMC, and `MRT` endpoints

This gives exact or near-exact parity for the asserted dense-profile Indometh
endpoints on all 6 subjects across the published linear and log tables.

### IV bolus implementation note

To match the published IV bolus tables, OpenPKPD now:

- reconstructs missing `C0` from the first two positive observations using
  log-linear back-extrapolation
- uses that reconstructed `C0` in AUC/AUMC aggregation and terminal-regression
  window selection
- uses the exact exponential-decline AUMC segment formula for log-down
  intervals, which is required for exact log-bolus `AUMClast`/`MRT` parity

## WinNonlin status

Phoenix WinNonlin itself is proprietary, so the repository does **not** bundle a
redistributable Phoenix project/output file. Instead, the benchmark uses:

- Phoenix documentation for parameter formulas and terminal-phase conventions
- PKNCA's public theophylline vignette as an executable open surrogate for the
  same lin-up/log-down workflow family
- Han (2018)'s public NonCompart validation paper, which republishes
  WinNonlin-backed Indometh reference tables

This gives public, reviewable parity evidence without introducing license or
reproducibility problems.

## nlmixr2 FOCEI: theophylline and warfarin

- **Reference source:** nlmixr2 v5.0.0 FOCEI fits generated by the bundled R
  scripts in `tests/external_validation/nlmixr2/`.
- **Fixtures:** `tests/external_validation/nlmixr2/reference/theophylline_foce.json`,
  `warfarin_pk_foce.json`, `warfarin_pkpd_foce.json`.
- **Test:** `tests/external_validation/test_vs_nlmixr2.py`

### Theophylline 1-compartment FOCEI

| Parameter | nlmixr2 v5.0 | OpenPKPD | Δ% |
|-----------|-------------|----------|-----|
| KA (h⁻¹) | 1.438 | ~1.44 | <1% ✅ |
| CL (L/h) | 2.793 | ~2.79 | <1% ✅ |
| V (L) | 32.193 | ~32.2 | <1% ✅ |
| IIV KA | 0.413 | ~0.41 | ~1% ✅ |
| IIV CL | 0.0604 | ~0.060 | <1% ✅ |
| Prop. σ² | 0.0242 | ~0.024 | <1% ✅ |

### Warfarin PK 1-compartment FOCEI

| Parameter | nlmixr2 v5.0 | OpenPKPD | Δ% |
|-----------|-------------|----------|-----|
| KA (h⁻¹) | 0.648 | ~0.65 | <2% ✅ |
| CL (L/h) | 0.136 | ~0.14 | <3% ✅ |
| V (L) | 8.168 | ~8.2 | <1% ✅ |
| Prop. σ² | 0.0505 | ~0.050 | <1% ✅ |
| OFV | 474.61 | ~474 | <0.2% ✅ |

**Assessment:** Excellent agreement with nlmixr2 on both datasets. Parameter
estimates and OFV are within 5% across all metrics.

---

## NONMEM 7.x: multi-compartment and covariate models

- **Reference source:** NONMEM 7.4.3/7.5.0 output files (`temp/nonmem/*.res`),
  from the Bauer (2019) NONMEM tutorial dataset.
- **Fixtures:** `tests/external_validation/reference/nonmem_402_focei.json`,
  `nonmem_504_focei.json`, `nonmem_504f_focei.json`.
- **Test:** `tests/external_validation/test_vs_nonmem.py`

### OFV convention

NONMEM reports OFV **without** N·log(2π). OpenPKPD now matches that convention
for the validated FOCE/FOCEI paths, but raw OFV parity can still differ on some
benchmarks because the tools do not necessarily report identical approximated
objectives after interaction/Hessian corrections. For run 402, parameter parity
is therefore the primary acceptance surface.

### Run 402 — Two-Compartment IV (ADVAN3 TRANS4), 30 subjects

| Parameter | NONMEM 7.4.3 | OpenPKPD | Status |
|-----------|-------------|----------|--------|
| V1 (L) | 9.76 | ~9.76 | ✅ |
| CL (L/h) | 3.88 | ~3.88 | ✅ |
| V2 (L) | 30.8 | ~30.8 | ✅ |
| Q (L/h) | 8.77 | ~8.77 | ✅ |
| OFV | **196.0** | tool-dependent offset | ⚠️ convention/approximation |

**Current interpretation:** OpenPKPD now reaches the NONMEM-like parameter
basin on run 402. The remaining benchmark note is about raw OFV comparability,
not the previous V2/Q-swapped local minimum.

### Run 504 — 1-Compartment + Covariates (ADVAN1 TRANS2), 60 subjects

| Parameter | NONMEM 7.5.0 | OpenPKPD | Δ% |
|-----------|-------------|----------|-----|
| CL (L/h) | 3.031 | 3.057 | +0.8% ✅ |
| V (L) | 32.384 | 32.740 | +1.1% ✅ |
| CL~WT exp | 0.660 | 0.670 | +1.6% ✅ |
| V~WT exp | 1.322 | 1.323 | +0.1% ✅ |
| CL~AGE exp | −0.534 | −0.537 | +0.5% ✅ |
| OFV | **1058.3** | 1271.0 | +20.1% ⚠️ |

**Current interpretation:** the maintained Python-API FOCEI benchmark now lands
very close to the NONMEM 7.5.0 parameter basin on the full 60-subject
covariate-rich dataset, even at `maxeval=1`, when started from physiologic
covariate priors (`WT^0.8`, `V~WT^0.8`, `CL~AGE≈-0.5`, sex multipliers near
0.9–0.95). The remaining issue on this case is a raw OFV gap, not parameter
recovery.

**Important scope note:** the older control-stream stress case (`temp/nonmem/504.ctl`)
still uses rougher cold starts and remains useful as a parser/runtime benchmark
for difficult covariate landscapes. That is a different support question from
the maintained API benchmark above.

### Run 504f — Fixed covariate exponents

| | NONMEM 7.5.0 | OpenPKPD | Δ% |
|--|--|--|--|
| OFV | **1065.4** | 1179.2 | +10.7% ⚠️ |
| CL~AGE exp | −0.529 | −0.099 | −81% ❌ |

---

## Phenobarbital — Neonatal Population PK (1-Compartment FO)

- **Reference source:** Grasela & Donn (1985). NONMEM FO analysis of 59 preterm neonates.
- **Fixture:** `tests/external_validation/reference/grasela1985_phenobarbital_fo.json`
- **Data:** `tests/external_validation/data/phenobarbital_simulated.csv`
  (simulated with published parameters; seed=42)
- **Test:** `tests/external_validation/test_phenobarbital.py`
- **Example control stream:** `examples/control_streams/12_phenobarbital_fo.ctl`

Published NONMEM FO estimates (Grasela & Donn 1985):

| Parameter | Published | Units |
|-----------|-----------|-------|
| CL | 0.0047 | L/h/kg |
| V | 0.96 | L/kg |
| BSV CL | ~19% | CV% |
| BSV V | ~16% | CV% |
| t½ | ~141 | h |

This dataset demonstrates weight-based allometric scaling and is the
classic example from NONMEM Users Guide Part V. OpenPKPD should recover
the simulation-truth parameters within ±30% on the simulated dataset.

---

## New Example Control Streams

Four new annotated control stream examples have been added:

| File | Description | Reference |
|------|-------------|-----------|
| `10_warfarin_pk_focei.ctl` | Warfarin 1-cmt oral FOCEI | nlmixr2 v5.0 |
| `11_two_compartment_iv_focei.ctl` | 2-cmt IV FOCEI with init guidance | NONMEM 7.4.3 Run 402 |
| `12_phenobarbital_fo.ctl` | Neonatal phenobarbital FO | Grasela & Donn 1985 |
| `13_covariates_one_cmt_focei.ctl` | 1-cmt + WT/AGE/SEX covariates | NONMEM 7.5.0 Run 504 |

---

## Priority Improvement Areas

### P1 — Two-Compartment FOCEI parity (Run 402) — **Improved**

The old V2/Q-swapped local-minimum failure mode is no longer the defining
behavior on this benchmark. The current remaining gap is documenting and
defending the residual OFV difference after OpenPKPD reaches the correct basin.

**Implemented in v0.3:**
- `FOCEMethod(n_starts=N, perturbation_scale=σ, seed=S)` runs N independent
  starts from perturbed initial values and returns the best OFV.
- `perturbation_scale` sets the Gaussian σ in the transformed (log/logit)
  parameter space; σ=1.0 ≈ factor-of-e perturbation.

**Operational guidance for 402:** provide near-correct initial values. For a
2-compartment IV model, ensure V2 > V1 and Q ≈ CL:
```
$THETA
  (0, 9.8)    ; V1
  (0, 3.7)    ; CL
  (0, 30.0)   ; V2 — initialise near peripheral volume (>> central)
  (0,  9.0)   ; Q  — initialise near inter-compartmental CL
```
This lands in the validated basin reliably. Multi-start still helps on models
with shallower alternative basins, but 402 is no longer primarily tracked as a
“stuck local minimum” case.

### P2 — Covariate Exponent Estimation (Run 504/504f) — **Refined**

Run 504 is no longer primarily a “can OpenPKPD recover the covariate basin?”
problem. The maintained Python-API FOCEI benchmark now reaches the NONMEM
parameter basin tightly on the full 60-subject dataset.

What remains difficult is narrower:
- cold-start control-stream execution for `504.ctl` / `504f.ctl`
- raw OFV comparability on the same converged basin
- broader coverage of covariate-rich datasets beyond this single NONMEM family

Operational guidance still stands:
- initialize covariate exponents near physiological priors
- tighten `GTOL` on flatter covariate landscapes
- use multi-start when exploring unfamiliar covariate structures

### P3 — Block OMEGA with Covariates — **Medium**

Block OMEGA for correlated IIV in the presence of multiple power-law covariates
produces inflated IIV estimates. Likely requires profile likelihood or adaptive
OMEGA update step size. No fix yet.

---

## Current conclusions

- **Monolix SAEM:** OpenPKPD is numerically consistent with the public Monolix
  theophylline project after matching dose units.
- **nlmixr2 SAEM (warfarin):** KA, V, and σ are within 30% of the nlmixr2
  reference. CL has a documented ~28% systematic gap (tracked in
  `test_cl_documented_gap_from_nlmixr2`); root cause under investigation.
- **SAEM (phenobarbital, Grasela 1985):** CL/kg within 35%, V/kg within 25%,
  and half-life within 40% of the published FO NONMEM reference values.
- **BAYES(Laplace) (warfarin):** OFV ≈ −219 (not a penalty value); KA −9.4%,
  CL +0.4%, V −8.2% from the nlmixr2 FOCEI basin. FOCE warm-start corruption
  fix eliminates the previous 1e10 OFV sentinel.
- **Nonparametric (theophylline NPML):** CL −0.1%, V +0.2%, KA +1.9% from the
  nlmixr2 FOCEI basin on the 12-subject Boeckmann theophylline dataset.
- **PKNCA / Phoenix-style NCA:** OpenPKPD reproduces the published theophylline
  summary metrics very closely, and the repository now separately tests
  interpolated user-specified cutoff handling in the NCA unit suite.
- **WinNonlin-backed Indometh NCA:** OpenPKPD now matches the published
  Indometh zero-start, IV bolus, IV infusion shared-endpoint, and
  extravascular reference tables for both linear and log rules.
- **nlmixr2 FOCEI:** Excellent agreement (< 5% on all parameters) for
  theophylline and warfarin 1-compartment models.
- **NONMEM 2-compartment (Run 402):** OpenPKPD now reaches the correct
  NONMEM-like parameter basin. The remaining note is a raw OFV offset due to
  differing Hessian-correction conventions, not a local-minimum failure.
- **NONMEM covariate model (Run 504):** maintained FOCEI benchmark now reaches
  the NONMEM parameter basin closely on the full 60-subject dataset; the
  remaining note is an OFV gap rather than parameter recovery.
- **NONMEM covariate model cold-start stress cases (Run 504/504f control
  streams):** still useful runner-level benchmarks for difficult covariate
  initialization and raw OFV comparability.

---

## References

1. Boeckmann AJ, Sheiner LB, Beal SL (1992). NONMEM Users Guide — Part V.
2. Grasela TH Jr, Donn SM (1985). Dev Pharmacol Ther, 8(6):374-83.
3. Bauer RJ (2019). NONMEM Tutorial Part II. CPT Pharmacometrics Syst Pharmacol, 8(8):538-556. [PMC6709422]
4. Schoemaker R et al. (2019). nlmixr: An R package for nonlinear mixed-effects model building. CPT Pharmacometrics Syst Pharmacol, 8(9):641-654. [PMC6930853]
5. Han S (2018). Validation of Noncompartmental Analysis Performed by NonCompart. Transl Clin Pharmacol, 26(1):10-17.
6. Bae KS, Yim DS (2016). FOCEI and the R implementation. Transl Clin Pharmacol, 24(4):161.
