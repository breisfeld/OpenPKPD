# OpenPKPD Full Code Review — 2026-04-02

Scope: estimation, model layer, native ODE path, parser/data, PK subroutines,
GUI, test suite, documentation, and misc modules (VPC, NPDE, bootstrap, NCA,
TMDD, TTE, PFIM, sandwich covariance).

Each item carries a status field:
  `[ ]` open · `[~]` not necessary / not prudent · `[x]` fixed

---

## P1 — CRITICAL (correctness bugs; must fix before next release)

### C-01 · FOCEI log-det formula sign error
**File:** `src/openpkpd/estimation/foce.py:1216`
The Woodbury identity is applied with the **wrong sign** for the matrix
determinant lemma: `log_det_M` is added rather than subtracted.  Result:
FOCEI OFV is biased upward; all FOCEI-based population estimates are
incorrect.
Status: `[ ]`

### C-02 · Eta penalty missing factor of 2
**File:** `src/openpkpd/model/individual.py:2398` (`_eta_penalty_value`)
The inner-loop objective must be `−2 log L + 2 η^T Ω^{-1} η` (−2LL scale)
but the penalty is computed without the factor 2.  The **gradient** at
line 2135 correctly uses `2 * omega_inv @ eta`, so objective and gradient
are inconsistent, biasing EBE estimates toward zero.
Status: `[ ]`

### C-03 · Observation interpolation bug in `as_multidose_probe`
**File:** `src/openpkpd/parser/code_compiler.py:817–824`
When a segment's `seg_t` list is empty, the RK45 integrator is invoked
with `t_eval=[bp]` only.  Observations that fall in that segment but are
not in `t_eval` receive `seg_states[0]` (the initial state) instead of an
interpolated value.  Affects all models using the P1.4 native Numba path.
Status: `[ ]`

### C-04 · Infusion silent NaN propagation in `DoseEvent`
**File:** `src/openpkpd/data/event_processor.py:57–62`
`infusion_end_time` executes `time + amount / rate` without guarding
against NaN amounts.  A NaN in the AMT column propagates silently to all
subsequent dose-event calculations.
Status: `[ ]`

### C-05 · Missing DUR validation for RATE = −1 infusions
**File:** `src/openpkpd/data/event_processor.py:265`
When `RATE = −1` (duration-specified infusion), the code silently defaults
`duration = 0.0` if the DUR column is absent or NaN.  This converts the
infusion to a bolus with infinite rate; no warning is raised.
Status: `[ ]`

### C-06 · User ODE template lazy rebuild unsafe in parallel workers
**File:** `src/openpkpd/model/individual.py:468–470` / `579`
`__getstate__` drops `_user_ode_template`; the rebuild is triggered lazily
on the first prediction call inside the worker, which may trigger Numba
JIT compilation on the critical path (unexpected latency or silent failure).
Fix: rebuild eagerly in `__setstate__`.
Status: `[ ]`

### C-07 · SS dosing with II = 0 not rejected
**File:** `src/openpkpd/data/event_processor.py:282`
The guard `II > 0 when ADDL > 0` is correct, but the same check is not
applied when `SS = 1`.  Steady-state calculations with II = 0 produce
silently incorrect PK profiles.
Status: `[ ]`

### C-08 · `FOCEI` constant term sign / convention inconsistency
**File:** `src/openpkpd/estimation/foce.py:1227`
The term `n_eta * LOG2PI` is subtracted; per the Beal–Sheiner convention
on −2LL scale it should be added.  Compounds the bias from C-01.
Status: `[ ]`

---

## P2 — HIGH (important correctness / usability issues)

### H-01 · ADVAN5 eigendecomposition unstable for near-repeated eigenvalues
**File:** `src/openpkpd/pk/analytical/advan5.py:131–153`
For ill-conditioned eigenvector matrix P, the pseudo-inverse fallback
(`np.linalg.pinv`) introduces modal errors.  Recommend auto-fallback to
ADVAN7 (matrix exponential) when `cond(P) > threshold`, with a logged
warning.
Status: `[ ]`

### H-02 · FO residual variance double-counted for proportional error
**File:** `src/openpkpd/estimation/fo.py:184`
`C_i = R @ omega @ R^T + diag(var_obs)` where `var_obs` already encodes
σ² dependence through the proportional error model.  Adding `diag(var_obs)`
a second time inflates the residual term and biases FO OFV.
Status: `[ ]`

### H-03 · FOCEI G_i assumes diagonal R_i (no off-diagonal IIV support)
**File:** `src/openpkpd/estimation/foce.py:1209`
`G_T_Rinv @ G` is computed with `diag(1/var)` assumed; block-diagonal or
full residual covariance structures produce incorrect interaction terms.
Status: `[ ]`

### H-04 · F1 / ALAG not extracted in ADVAN1, ADVAN3, ADVAN5, ADVAN7, ADVAN11
**Files:** `src/openpkpd/pk/analytical/advan1.py`, `advan3.py`, `advan5.py`,
`advan7.py`, `advan11.py`
Oral ADVANs 2, 4, 12 extract F1 internally; the IV/linear ADVANs rely on
upstream application.  This hidden contract is undocumented and fragile; a
model that bypasses `IndividualModel._apply_alag` silently receives no lag.
Status: `[ ]`

### H-05 · D-literal regex misses `1.D0` and `.5D-3`
**File:** `src/openpkpd/parser/code_compiler.py:106`
Regex `(\d)D([+-]?\d)` requires a digit before `D`; edge cases like `1.D0`
(decimal before D) or `.5D-3` (no leading digit) are not converted and
cause Python syntax errors at compile time.
Fix: use `([0-9.])D([+-]?\d)`.
Status: `[ ]`

### H-06 · FD step size in sensitivity probes not parameter-scaled
**File:** `src/openpkpd/parser/code_compiler.py:745, 856`
Default `fd_eps = 1e-5` is absolute.  For parameters like `ALAG = 0.001`
this is a 1 000 % perturbation; for `V = 500` it may fall below machine
precision.  Should use `h = fd_eps * max(abs(x), 1.0)`.
Status: `[ ]`

### H-07 · Global RNG seed in NUTS breaks multi-chain reproducibility
**File:** `src/openpkpd/estimation/nuts.py:265`
`np.random.seed(seed)` resets the **global** RNG.  Concurrent NUTSSampler
instances in `bayes.py` overwrite each other's seeds, breaking
within-chain reproducibility.  Use `np.random.default_rng(seed)`.
Status: `[ ]`

### H-08 · Laplacian Hessian correction incomplete
**File:** `src/openpkpd/estimation/laplacian.py:111–146`
True Laplace approximation requires the Hessian correction w.r.t. THETA
and OMEGA; only the ETA Hessian is computed.  When the Hessian is
non-PD, log-det is set to 0 with no warning, silently dropping the
correction term.
Status: `[ ]`

### H-09 · `can_start_fit_run` does not validate `dataset_path`
**File:** `src/openpkpd_gui/workflows/fit_workflow.py:103–109`
The readiness check gates on `preparation.ready` and run status only.
If the dataset file is deleted after translation succeeds, the "Run"
button stays enabled but the fit fails immediately with a cryptic error.
Status: `[ ]`

### H-10 · Silent fallback to stale dataset on CSV load failure
**File:** `src/openpkpd_gui/workflows/data_workflow.py:691–711`
When CSV loading fails, the UI renders validation errors alongside the
**previous** dataset's column preview, making it unclear which dataset
is active.
Status: `[ ]`

### H-11 · Mode switch in Model workflow loses edits without warning
**File:** `src/openpkpd_gui/workflows/model_workflow.py:738–755`
Switching Builder ↔ Control Stream mode discards THETA/OMEGA/SIGMA tables
and PK/ERROR code without any "unsaved changes" warning.
Status: `[ ]`

### H-12 · No PK ↔ OMEGA dimension consistency check
**File:** `src/openpkpd_gui/services/model_translation_service.py:74–86`
The translator does not verify that the number of ETA() calls in the
PK block matches the OMEGA matrix dimension.  Dimension mismatches are
only caught at fit time with a cryptic error.
Status: `[ ]`

### H-13 · RATE = −2 infusion mode silently ignored
**File:** `src/openpkpd/data/event_processor.py:262–267`
NONMEM RATE = −2 (rate specified in PK code) falls through all branches
without error or warning and is treated as a zero-rate event.
Status: `[ ]`

---

## P3 — MEDIUM (design, performance, completeness)

### M-01 · `IndividualModel` "God Object" anti-pattern
**File:** `src/openpkpd/model/individual.py` (~2 455 lines)
The class conflates PK solving, error models, likelihood, derivatives,
caching, IOV, BLQ, and serialisation.  Recommend splitting into functional
mixins: `PKSolutionMixin`, `ObservationModelMixin`, `LikelihoodMixin`,
`DerivativesMixin`.
Status: `[ ]`

### M-02 · 4× redundant template-matching loops
**File:** `src/openpkpd/model/individual.py:685, 826, 970, 1156`
The same template eligibility logic is repeated verbatim in four dispatch
methods.  Extract to a single `_select_template(pk_params) → _NativeOdeTemplate | None`
and cache the result per pk_params signature.
Status: `[ ]`

### M-03 · Covariate time-constancy check silently disables native path
**File:** `src/openpkpd/model/individual.py:618`
A broad `except Exception: return None` treats any non-numeric column as
a time-varying covariate and falls back from the native path.  Categorical
covariates unused in the PK block should not disqualify the native path.
Status: `[ ]`

### M-04 · Power residual variance discontinuous at f = 0
**File:** `src/openpkpd/model/residual_models.py:217–219`
Returns `np.finfo(float).tiny` for `f ≤ 0` instead of evaluating
`σ² * |f|^(2θ)`, introducing a discontinuity that distorts numerical
gradients near zero predictions.
Status: `[ ]`

### M-05 · IF-THEN-ELSE block indentation fragility in NM-TRAN compiler
**File:** `src/openpkpd/parser/code_compiler.py:198–222`
Nested single-line IFs inside THEN blocks, or unmatched ENDIF, can produce
invalid Python without raising a CompilerError.  Add THEN/ENDIF pairing
validation.
Status: `[ ]`

### M-06 · ADVAN12 degenerate eigenvalue tolerance is absolute
**File:** `src/openpkpd/pk/analytical/advan12.py:100–120`
The L'Hôpital threshold `1e-8` is scale-independent; for very fast
absorption (KA > 100 h⁻¹) or very slow systems this tolerance may be
inappropriately tight or loose.  Use `rtol * max(lam_j)`.
Status: `[ ]`

### M-07 · Mixture model EM fitting never unit-tested
**File:** `tests/unit/mixture/test_mixture.py`
Tests cover dataclass construction and `summary()` formatting only; the
EM algorithm itself is not exercised.  Fitting a bimodal synthetic
dataset should be added as a convergence smoke test.
Status: `[ ]`

### M-08 · SAEM lacks external validation vs NONMEM
**File:** `tests/external_validation/`
SAEM is only validated against Monolix on theophylline.  A NONMEM SAEM
reference run with an agreed OFV tolerance would increase confidence.
Status: `[ ]`

### M-09 · Stratified bootstrap resampling not implemented
**File:** `src/openpkpd/inference/bootstrap.py:454–488`
Subjects are resampled uniformly; users cannot stratify by dose group,
study centre, or sex to ensure proportional representation.
Status: `[ ]`

### M-10 · 21 example scripts have no corresponding `.md` documentation
**File:** `docs/examples/index.md`
Only 11 of 32 example scripts appear in the toctree.  At minimum, create
stub pages so every script is reachable from the navigation.
Status: `[ ]`

### M-11 · No estimation method selection decision guide
**File:** `docs/user_guide/estimation_methods.md`
Nine methods are listed with no flowchart or decision tree.  A "Choosing
a method" section with a diagram would substantially reduce user friction.
Status: `[ ]`

### M-12 · `impute_covariates()`, `design()`, and `simulate()` undocumented
**File:** `docs/user_guide/model_builder.md`
Three public `ModelBuilder` / `BuiltModel` methods are absent from the
narrative user guide and only discoverable via autodoc.
Status: `[ ]`

### M-13 · Parse errors lose line-number context in GUI
**File:** `src/openpkpd_gui/services/model_translation_service.py:132–136`
`ParseError` carries structured `line` and `context` fields; the GUI
converts it to a plain string and discards those fields.  Surface line
number and context snippet in the validation issue list.
Status: `[ ]`

### M-14 · IMP importance weight degeneracy not adaptively handled
**File:** `src/openpkpd/estimation/imp.py:482`
Low ESS triggers WARN_006 but does not adaptively increase `isample`.
Implement an ESS-ratio guard that doubles sample count for subjects with
ESS / isample < threshold.
Status: `[ ]`

### M-15 · VPC / NPDE lack bootstrap confidence bands
**File:** `src/openpkpd/simulation/vpc.py`, `npde.py`
Only point-estimate percentile bands are returned; no bootstrap CIs
around the simulated quantile curves.
Status: `[ ]`

---

## P4 — LOW (technical debt, nice-to-have)

### L-01 · Dead code: `_eps_basis_vectors` never used
**File:** `src/openpkpd/model/individual.py:452–454`
Pre-computed one-hot tuples; no call site found in the codebase.  Remove.
Status: `[ ]`

### L-02 · `log_likelihood_normal` in `residuals.py` is dead code
**File:** `src/openpkpd/model/residuals.py:17–25`
Not called from the main likelihood path.  Either wire it in or remove it
to reduce confusion about which formula is authoritative.
Status: `[ ]`

### L-03 · Redundant eigendecomposition in `_eta_penalty_structure`
**File:** `src/openpkpd/model/individual.py:2419`
`repair_pd()` eigen-decomposes Ω, then `np.linalg.inv()` on the repaired
matrix eigen-decomposes again.  Invert directly from the eigenvalues:
`omega_inv = (evecs / evals) @ evecs.T`.
Status: `[ ]`

### L-04 · Parallel backend unit tests missing (Ray, Dask, MPI)
**File:** `tests/unit/parallel/`
Only the basic ProcessPoolExecutor path is tested.  Add smoke tests for
Ray/Dask initialization, timeout, and retry logic.
Status: `[ ]`

### L-05 · TTE and TMDD have no external validation
**File:** `tests/external_validation/`
Both model families have unit tests only.  A reference dataset (even
synthetic) compared against nlmixr2 or NONMEM would establish a baseline.
Status: `[ ]`

### L-06 · Validation-issue rendering duplicated across four workflows
**Files:** `fit_workflow.py`, `data_workflow.py`, `diagnostics_workflow.py`,
`advanced_workflow.py`
Extract to a shared `_render_validation_issues(list_widget, issues)` helper
in a new `openpkpd_gui/widgets/validation_list.py`.
Status: `[ ]`

### L-07 · Magic object-name strings in model workflow event handlers
**File:** `src/openpkpd_gui/workflows/model_workflow.py:1400–1500`
`widget.objectName() == "model-theta-table"` style checks are brittle.
Use tagged callbacks or a central dispatcher instead.
Status: `[ ]`

### L-08 · EVID = 5 not supported or documented
**File:** `src/openpkpd/data/event_processor.py`
NONMEM 7.5+ added EVID = 5.  Either add handling or add an explicit
`ValueError` with a clear message so users know it is unsupported.
Status: `[ ]`

### L-09 · ADVAN9 / ADVAN14 / ADVAN15 (nonlinear absorption/disposition) absent
**File:** `src/openpkpd/pk/`
Users needing saturable absorption must build a custom ODE.  Document this
limitation explicitly and add these as future roadmap items.
Status: `[ ]`

### L-10 · BAYES / NUTS external validation absent
**File:** `tests/external_validation/`
Only the Laplace fallback is tested.  A reference comparison against PyMC
or Stan on a simple normal–normal model would validate the sampling path.
Status: `[ ]`

### L-11 · Visual regression testing absent for plots
**File:** `tests/unit/plots/`
Matplotlib output is only checked by file existence or basic shape; pixel-
level regressions are not caught.  Consider `pytest-image-diff` or similar.
Status: `[ ]`

### L-12 · `build_model_workflow` function is 1 253 lines
**File:** `src/openpkpd_gui/workflows/model_workflow.py:693–1946`
Extract widget-creation helpers (`_build_parameter_tables`,
`_build_code_editors`, `_build_estimation_section`) and an event-handler
class to reduce cognitive load and improve testability.
Status: `[ ]`

---

## Notes

Items marked `[~]` (not necessary / not prudent) will be added inline as
individual findings are investigated and closed.

Priority mapping:
  P1 C-xx → critical correctness; block release
  P2 H-xx → high impact; fix within current sprint
  P3 M-xx → medium; schedule in next sprint
  P4 L-xx → low; address opportunistically or accept as known debt
