---
orphan: true
---

# OpenPKPD Performance Baseline

This document records the performance baseline captured on **2026-03-24** before any
compiled (Rust/Cython) extensions are introduced.  Every future optimisation should be
compared against these numbers to verify improvement and catch regressions.

---

## Environment

| Item | Value |
|------|-------|
| Host CPU | Intel Xeon Gold 6136 @ 3.00 GHz (12 cores / 24 threads) |
| Architecture | x86_64 |
| Python | 3.12.4 (conda-forge, GCC 12.3.0) |
| NumPy | 2.4.3 |
| SciPy | 1.17.1 |
| OpenPKPD commit | `1e0ae438` |

---

## Benchmark scripts

Two profiling scripts exist in `scripts/`.  Both use `cProfile` internally and save
JSON artefacts under `artifacts/profiling/`.

| Script | What it covers | Output file |
|--------|---------------|-------------|
| `scripts/benchmark_estimation.py` | FO, FOCE, FOCEI, SAEM, IMP, IMPMAP, Bayes estimation | `artifacts/profiling/estimation_baseline.json` |
| `scripts/benchmark_estimation.py --workloads bayes_nuts --n-subjects 6 --bayes-samples 12 --bayes-tune 8` | bounded native NUTS diagnostic baseline | `artifacts/profiling/estimation_nuts_bounded_2026-03-29.json` |
| `scripts/profile_pipelines.py` | FOCE (quick), VPC, NPDE, symbolic | `artifacts/profiling/profile_pipelines.json` |
| `scripts/profile_analysis.py` | Diagnostics, NPDE, VPC, NCA | `artifacts/profiling/analysis_baseline_current.json` |

### Reproducing the estimation baseline

```bash
uv run python scripts/benchmark_estimation.py \
    --json-out artifacts/profiling/estimation_baseline.json
```

Default parameters (stored in the JSON `metadata` block):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_subjects` | 12 | 1-cmt oral PK (ADVAN2/TRANS2), 7 obs each |
| `seed` | 42 | |
| `fo_maxeval` | 500 | Runs to convergence |
| `foce_maxeval` | 300 | Runs to convergence |
| `focei_maxeval` | 200 | Runs to convergence |
| `saem_k1` | 150 | Stochastic exploration phase |
| `saem_k2` | 100 | Convergence phase |
| `model` | ADVAN2/TRANS2, proportional error | KA=1.5, CL=2.8, V=32.9 (pop truth) |

### Reproducing the bounded NUTS baseline

```bash
uv run python scripts/benchmark_estimation.py \
    --workloads bayes_nuts \
    --n-subjects 6 \
    --bayes-samples 12 \
    --bayes-tune 8 \
    --json-out artifacts/profiling/estimation_nuts_bounded_2026-03-29.json
```

This bounded run is intentionally small. It is a support-boundary artifact, not
an accuracy benchmark. Its purpose is to capture runtime shape and the NUTS
diagnostic surface (`log_prob_calls`, FOCE call counts/timings, per-chain tree
depth / step size summaries) on a reproducible workload. As of the
2026-03-29 refresh, the benchmark helper uses compiled `ModelBuilder`
callables so the standard oral-PK workload can exercise the cached symbolic
ADVAN2 derivative path when it is available.

Current bounded `bayes_nuts` baseline (`n_subjects=6`, `n_samples=12`,
`tune=8`, `n_chains=2`):

| Metric | Value |
|--------|-------|
| Wall time | 8.53 s |
| Converged | Yes |
| `log_prob_calls` | 327 |
| `foce_inner_calls` | 326 |
| `foce_inner_seconds` | 6.98 s |
| `theta_gradient_calls` | 326 |
| `used_analytic_theta_gradient` | `true` |

This replaces the older finite-difference-backed bounded probe. Both chains now
report `used_fd_gradient=false`.

---

## Estimation benchmark results (2026-03-24)

### Wall-clock summary

| Method | Wall time (s) | Outer evals | Converged | OFV |
|--------|-------------|-------------|-----------|-----|
| FO | 1.76 | 232 | Yes | 160.60 |
| FOCE | 64.17 | 832 | Yes | −399.16 |
| FOCEI | 16.56 | 480 | Yes | −25.36 |
| SAEM | 6.70 | 250 iters | Yes | 165.30 |

### Stage timing breakdown

Time is cumulative wall seconds measured via `timed_patch` instrumentation.

#### FO (1.76 s total)

| Stage | Time (s) | Calls | µs / call |
|-------|---------|-------|-----------|
| `fo.compute_fo_ofv` | 1.70 | 232 | 7,325 |
| `individual.evaluate_observation_model` | 1.06 | 22,272 | 48 |
| `individual._evaluate_predictions` | 0.80 | 22,272 | 36 |

FO is dominated by the Jacobian computation inside `_fo_ofv_individual`
(`autodiff.jacobian` — numerical FD of predictions w.r.t. ETAs to build R_i).

#### FOCE — no interaction (64.17 s total)

| Stage | Time (s) | Calls | µs / call |
|-------|---------|-------|-----------|
| `foce.inner_loop` | 62.92 | 833 | 75,535 |
| `individual.evaluate_observation_model` | 21.99 | 434,122 | 51 |
| `individual._evaluate_predictions` | 16.71 | 434,122 | 38 |
| `individual.obj_eta` | 9.61 | 106,062 | 91 |
| `foce.outer_ofv` | 1.01 | 833 | 1,213 |

The inner loop (per-subject ETA mode-finding via L-BFGS-B) accounts for **98% of total
runtime**.  `individual.obj_eta` is called ~127 times per outer iteration.

#### FOCEI — with interaction (16.56 s total)

| Stage | Time (s) | Calls | µs / call |
|-------|---------|-------|-----------|
| `foce.inner_loop` | 14.67 | 481 | 30,492 |
| `individual.evaluate_observation_model` | 5.65 | 114,092 | 50 |
| `individual._evaluate_predictions` | 4.28 | 114,092 | 38 |
| `individual.obj_eta` | 2.35 | 22,751 | 103 |
| `foce.outer_ofv` | 1.77 | 481 | 3,681 |

FOCEI converged in fewer outer evaluations than FOCE (480 vs. 832), producing a lower
total time despite the extra Woodbury / G_i cost per outer iteration
(`outer_ofv` is ~3× more expensive per call: 3.7 ms vs. 1.2 ms).

#### SAEM (6.70 s total)

| Stage | Time (s) | Calls | µs / call |
|-------|---------|-------|-----------|
| `individual.evaluate_observation_model` | 3.69 | 83,040 | 44 |
| `individual._evaluate_predictions` | 2.80 | 83,040 | 34 |
| `individual.obj_eta` | 0.87 | 9,000 | 97 |

The majority of SAEM time is in the M-step theta optimisation (scipy L-BFGS-B called
260 times for theta/sigma updates), which in turn calls `log_likelihood` 83,040 times.
The E-step MH sampling (`individual.obj_eta` × 9,000) accounts for only ~13% of runtime
at this iteration count; it scales linearly with K1+K2.

---

## Analysis / simulation benchmark results (2026-03-16)

Source: `artifacts/profiling/analysis_baseline_current.json`
Parameters: 6 fit subjects / 24 sim subjects / 500 replicates / 2000 NCA subjects.

| Workload | Wall time (s) | Key hot function | Time in hot fn |
|----------|-------------|-----------------|---------------|
| Diagnostics | 0.015 | `individual.evaluate` | 0.010 s |
| NPDE (500 sim) | 0.636 | `simulation.simulate` | 0.550 s |
| VPC (500 rep) | 1.055 | `simulation.simulate` | 0.714 s |
| NCA (2000 subj) | 0.480 | `nca.compute_subject` | 0.408 s |

---

## Hot path analysis

### Universal hot path: `individual.log_likelihood`

Every estimation and simulation method passes through `IndividualModel.log_likelihood`
(`individual.py:723`).  cProfile cumulative time across methods:

| Method | Calls | Cumulative (s) | µs / call |
|--------|-------|---------------|-----------|
| FOCE | 424,126 | 29.68 | 70 |
| FOCEI | 91,004 | 6.39 | 70 |
| SAEM | 83,040 | 5.20 | 63 |

This function is the **single most important target** for a compiled extension.  It:

1. Calls `evaluate_observation_model` to get `(ipred, obs_mask, f_obs, var)`.
2. Loops over each observation in pure Python:
   - Checks for BLQ.
   - Computes `log_likelihood_normal(dv, mu, sigma2)` using `math.log`.
3. Returns `-2 * sum(ll_i)`.

The per-observation arithmetic is trivial, but the Python loop overhead at ~70 µs/subject
with 7 obs = ~10 µs/observation becomes the bottleneck when called millions of times.

### Call chain (FOCE)

```
FOCEMethod.estimate()
  → _run_single()  [1 call]
    → minimize() [outer L-BFGS-B, 832 iters]
      → objective()  [832 calls]
        → _inner_loop()  [833 calls, 62.9 s]
          → minimize() [inner L-BFGS-B per subject, ~127×/outer iter]
            → obj_eta()  [106,062 calls, 91 µs each]
              → log_likelihood()  [424,126 calls, 70 µs each]
                → evaluate_observation_model()  [434,122 calls, 50 µs each]
                  → _evaluate_predictions()  [→ ADVAN2.solve(), already fast]
                → log_likelihood_normal() × n_obs  [Python loop, ← TARGET]
        → _outer_ofv()  [833 calls, 1.2 ms each]
          → _compute_G_i() × n_eta  [finite-diff G_i, FOCEI only]
          → Woodbury identity  [numpy, already fast]
```

### What to measure when comparing

After a change, re-run:

```bash
uv run python scripts/benchmark_estimation.py \
    --json-out artifacts/profiling/estimation_YYYY-MM-DD-<description>.json
```

The key comparison metrics are:

| Metric | Where | Baseline |
|--------|-------|---------|
| FOCE wall time | `foce.wall_seconds` | 64.17 s |
| FOCE `individual.obj_eta` µs/call | stage_totals | 91 µs |
| FOCE `individual.log_likelihood` µs/call | top_functions (ncalls, cumulative) | 70 µs |
| FOCEI wall time | `focei.wall_seconds` | 16.56 s |
| SAEM wall time | `saem.wall_seconds` | 6.70 s |
| SAEM `individual.obj_eta` µs/call | stage_totals | 97 µs |

---

## Analysis / simulation benchmark (pipeline)

Source: `artifacts/profiling/profile_pipelines.json`
Parameters: 8 subjects (quick FOCE, maxeval=25, partial), 24 sim subjects, 500 replicates.

| Workload | Wall time (s) | Key hot stage | Stage time |
|----------|-------------|-------------|-----------|
| FOCE (8 subj, partial) | 5.43 | `foce.inner_loop` | 4.04 s |
| VPC (24 subj, 500 rep) | 1.94 | `simulation.simulate` | 1.39 s |
| NPDE (24 subj, 500 sim) | 1.71 | `simulation.simulate` | 1.32 s |

---

## What NOT to optimise (already fast)

| Component | Reason |
|-----------|--------|
| `ADVAN2.solve()` / `advan2.py:130` | Already delegated to numpy vectorised ops |
| `ADVAN3/4` analytical solutions | Already vectorised numpy |
| Woodbury matrix identity in FOCEI | numpy matmul + linalg.inv (BLAS/LAPACK) |
| `repair_pd()` in `math/matrix.py` | numpy eigh → LAPACK |
| BLQ `blq_log_likelihood` | scipy.stats.norm (C under the hood) |
| `parser/lexer.py` | One-time startup cost; immaterial |

---

## Optimisation results

### 1. PyO3 `neg2ll_obs_loop` extension — implemented 2026-03-24

**Location:** `rust/src/lib.rs` → `openpkpd._core.neg2ll_obs_loop`
**Build:** `just build-core`

The Rust function replaces the Python per-observation loop inside
`IndividualModel.log_likelihood`.  It handles all BLQ methods (M1–M7) including
censored likelihood (M2/M3/M4) via `libm::erfc` for the normal CDF.

All-NaN LLOQ arrays (the common case) are cached at module level to avoid a heap
allocation per call.

**Measured impact (2026-03-24, seed=42, 12 subjects, 7 obs each):**

| Method | Baseline (s) | After Rust (s) | Speedup | log_lik µs/call |
|--------|-------------|---------------|---------|-----------------|
| FO | 1.76 | 1.62 | 1.09× | — |
| FOCE | 64.17 | 56.92 | **1.13×** | 70.0 → 57.3 |
| FOCEI | 16.56 | ~17.5 | ~1.0× (noise) | 70.2 → 59.5 |
| SAEM | 6.70 | 5.57 | **1.20×** | 62.7 → 49.6 |

**Why the improvement is modest:**

`log_likelihood` = `evaluate_observation_model` (~50 µs/call) + the obs loop.
For 7 observations/subject the Python obs loop was only ~20 µs; Rust brings it to
~7 µs.  The PK solve in `evaluate_observation_model` was not changed and remains the
floor.  For models with more observations per subject (e.g. rich PD sampling with
50+ timepoints), the absolute saving per call scales linearly and the benefit grows
proportionally.

FOCEI shows no reliable improvement because `_outer_ofv` (Woodbury + G_i finite
differences, ~3.7 ms each) dominates its runtime — not `log_likelihood`.

**What to update when comparing:** `foce.wall_seconds`, `saem.wall_seconds`,
and the `log_likelihood` µs/call from `top_functions`.

---

## Further planned optimisations

### 2. NumPy fast-path for standard error models — implemented 2026-03-24

**Location:** `src/openpkpd/model/individual.py` → `IndividualModel._fast_obs_model()`

The existing `_infer_common_error_model()` helper already classifies standard `$ERROR`
patterns at model build time.  A new `_fast_obs_model()` method uses this classification
to evaluate the error model with a single vectorized NumPy expression, bypassing the
per-observation Python loop in `evaluate_observation_model` entirely for the eps=0
estimation path.

Supported patterns: proportional, additive, proportional_theta, additive_theta,
combined_theta, combined_eps.  Any other pattern falls back to the existing loop.

**Measured impact (2026-03-24, seed=42, 12 subjects, 7 obs, proportional error model):**

| Stage | Baseline µs/call | After fast-path µs/call | Speedup |
|-------|-----------------|------------------------|---------|
| `evaluate_observation_model` | ~50 | ~18 | **2.8×** |

**Wall-clock impact on estimation methods:**

| Method | Baseline (s) | After fast-path (s) | Speedup |
|--------|-------------|---------------------|---------|
| FO | 1.76 | ~1.2 | ~1.5× |
| FOCE | 64.17 | ~45 | ~1.4× |
| FOCEI | 16.56 | ~13 | ~1.3× |
| SAEM | 6.70 | ~4.8 | ~1.4× |

The fast-path improvement compounds with the Rust `neg2ll_obs_loop` (optimisation 1):
`evaluate_observation_model` and the obs loop are both on the hot path, so both
reductions accumulate.  With both enabled, `log_likelihood` drops from ~70 µs/call
to roughly ~25 µs/call on this benchmark model.

**Why the speedup is larger here than for the Rust loop:** The Python loop overhead
for 7 observations (~20 µs) was smaller than the Python/NumPy dispatch and indexing
inside `evaluate_observation_model` (~30 µs per call overhead beyond the PK solve).
The NumPy vectorised path eliminates that dispatch overhead entirely.

For custom error models (`$ERROR` blocks with non-standard patterns) the fast path
is not activated and behaviour is identical to baseline.

---

### 3. ODE RHS in Rust (future)

Compile `$DES` blocks to Rust functions rather than Python.  Eliminates the
Python↔C boundary at every ODE step.  Expected 3–20× speedup for ODE-based models
(ADVAN6/8/10); no impact on analytical ADVAN1–4 models.

### 3. SAEM MH loop vectorisation (future)

Batch MH log-likelihood evaluations across chains per subject into a single Rust call.
Currently `obj_eta` is ~13% of SAEM runtime; the M-step theta optimisation dominates.
Becomes relevant at higher `n_chains` or larger subject counts.
