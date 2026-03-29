## Performance analysis and simulation review

Date: 2026-03-15

### Scope and method

This report is based on a static review of the main estimation, analysis, and
simulation paths used by the application. I did not run profilers or large
benchmarks, so the recommendations below are prioritized by code structure,
algorithmic complexity, and per-call overhead visible in the implementation.

For measured profiling and benchmark runs, use:

```bash
just profile-analysis
just benchmark-estimation
just benchmark-estimation --workloads imp impmap
```

When comparing raw `IMP` and `IMPMAP`, treat `IMPMAP` as the costlier but more
robust path because it includes a FOCEI warm start before the IMP outer pass.

and compare with the profiling report in
`docs/user_guide/performance_profiling_report.md`.

### Executive summary

The biggest speed opportunities are concentrated in four places:

1. **FOCE estimation** repeatedly creates worker pools inside the outer
   optimization loop and still relies on finite-difference gradients.
2. **Observation-model evaluation** is heavily Python-bound: per-observation
   loops, repeated list conversions, and `exec`-based compiled callables.
3. **Simulation post-processing** (VPC/NPDE/NPC) does substantial row-wise
   pandas work that can be converted to denser NumPy/pivoted operations.
4. **Bootstrap/SCM/SSE orchestration** rebuilds models and data structures many
   times, with some paths not using the available parallel capacity well.

### Highest-priority opportunities

| Priority | Area | Evidence | Opportunity | Likely effect |
| --- | --- | --- | --- | --- |
| P0 | FOCE inner loop | `src/openpkpd/estimation/foce.py:219-238` creates a new `ProcessPoolExecutor` inside each `_inner_loop()` call; `estimate()` calls `_inner_loop()` once per outer objective evaluation (`131-155`) | Reuse a persistent worker pool across outer iterations, or switch to a backend that stays alive for the whole fit | Large reduction in process-spawn overhead, especially on macOS |
| P0 | FOCE objective evaluation | `foce.py:62-68` and `211-214` use `jac="2-point"`; each eta optimization triggers many extra objective evaluations | Add analytical/autodiff gradients where possible, or at minimum batch/parallelize finite-difference evaluations | Major speedup for fits, bootstrap, SCM, and SSE |
| P0 | Observation model | `src/openpkpd/model/individual.py:223-283` loops over observations; `_call_error_model()` at `299-324` converts arrays to lists every call; `src/openpkpd/parser/code_compiler.py:285-323` executes compiled `$ERROR` via `exec` | Batch error-model evaluation per subject, cache zero-EPS outputs, and avoid repeated list conversion/materialization | Large win across estimation and simulation |
| P1 | Simulation engine | `src/openpkpd/simulation/engine.py:236-299` builds Python dict rows per observation and calls the error model per row; parallelism uses `ThreadPoolExecutor` at `125-142` | Accumulate columns in arrays first, then build DataFrames once; consider process-based or backend-based parallelism for Python-heavy models | Moderate-to-large win for VPC/NPDE/NPC inputs |
| P1 | VPC | `src/openpkpd/simulation/vpc.py:319-410` does two groupby passes and repeated percentile calls | Replace row-group iteration with grouped array aggregation/pivoted quantile computation | Moderate speedup for larger replicate counts |
| P1 | NPDE | `src/openpkpd/simulation/npde.py:170-205`, `252-272`, and `296-312` assemble per-subject matrices row by row and compute PDEs with Python loops | Use joins/pivots to build matrices and vectorized comparisons for PDE calculation | Moderate speedup, especially at `n_replicates >= 500` |

### Detailed findings

#### 1. FOCE estimation is the highest-value target

- `FOCEMethod.estimate()` calls `_inner_loop()` inside the outer optimizer
  objective and again after optimization (`src/openpkpd/estimation/foce.py:131-155`).
- In parallel mode, `_inner_loop()` creates a fresh `ProcessPoolExecutor`
  every time (`219-238`). That means worker startup/teardown repeats across
  many outer iterations.
- Both the serial and worker eta optimizations use `L-BFGS-B` with
  `jac="2-point"` (`62-68`, `211-214`), multiplying the number of expensive
  `obj_eta()` evaluations.

**Recommended next step:** start here. Reusing workers and reducing gradient
cost should improve not only direct fits, but also bootstrap, SCM, and SSE.

#### 2. Observation-model evaluation is Python-bound in the innermost loops

- `IndividualModel.evaluate_observation_model()` evaluates one observation at a
  time (`src/openpkpd/model/individual.py:240-283`).
- For each observation it may call the error model twice (zero-EPS mean and
  nonzero-EPS output), then estimate residual variance.
- `_call_error_model()` converts `theta`, `eta`, and `eps` to Python lists on
  every call (`299-324`).
- The compiled callable itself is still `exec`-driven
  (`src/openpkpd/parser/code_compiler.py:285-323`).

**Opportunity:** move from scalar/per-row evaluation toward subject-level batch
evaluation. Even partial batching or caching of the zero-EPS pass would reduce
overhead materially.

#### 3. Simulation spends a lot of time creating Python objects

- `SimulationEngine._simulate_one_replicate()` loops over every subject and
  observation and appends Python dict rows (`src/openpkpd/simulation/engine.py:236-299`).
- ETA column names are re-materialized into each row; DataFrames are built per
  replicate and concatenated at the end (`144`).
- Replicate parallelism currently uses `ThreadPoolExecutor` (`125-142`), which
  may underutilize CPU cores when the workload is dominated by Python-level work.

**Opportunity:** build column arrays/lists per replicate, then construct a
single DataFrame; precompute stable per-subject values once; evaluate whether a
process/backend strategy outperforms threads for real workloads.

#### 4. VPC, NPDE, and NPC are more pandas-heavy than necessary

- VPC simulated percentiles use nested groupby iteration and repeated quantile
  extraction (`src/openpkpd/simulation/vpc.py:319-410`).
- NPDE builds `Y_sim` matrices via per-row key construction and `iterrows()`
  (`src/openpkpd/simulation/npde.py:252-272`) and then computes PDEs row-wise
  (`296-312`).
- NPC has a costly fallback path that scans the entire lookup dict for near-
  matching times (`src/openpkpd/simulation/npc.py:149-155`).

**Opportunity:**

- normalize/round times once before lookup,
- use pivoted matrices keyed by `(ID, TIME, OBSSEQ, REP)`,
- replace Python loops with vectorized comparisons and batched quantiles.

The NPC fallback is a particularly good low-risk fix because it is both easy and
can avoid pathological slowdowns.

#### 5. NCA has a clear algorithmic optimization path

- `compute_dataset()` filters the full DataFrame once per subject
  (`src/openpkpd/nca/nca.py:380-408`).
- `_compute_lambda_z()` tries every suffix window and runs `linregress()` for
  each candidate (`675-721`).

**Opportunity:**

- pre-group once instead of repeated boolean filtering,
- replace repeated regression setup with suffix-statistic formulas or another
  vectorized regression strategy.

This is a good candidate for a contained optimization that does not require
major architectural changes.

#### 6. Bootstrap, SCM, and SSE repeat heavy rebuild work

- Bootstrap resamples subjects by slicing/copying DataFrames subject by subject
  and rebuilds a new `PopulationModel` per replicate
  (`src/openpkpd/inference/bootstrap.py:469-503`).
- Parallel bootstrap falls back to sequential if pickling the engine fails
  (`555-563`).
- SCM uses `ThreadPoolExecutor` for candidate fits (`src/openpkpd/covariate/scm.py:293-298`) and deep-copies the builder for each trial model (`428-460`).
- SSE exposes `n_jobs` but currently runs re-estimation sequentially in a plain
  `for rep in range(...)` loop (`src/openpkpd/simulation/sse.py:149-220`).

**Opportunity:** reduce payload size for workers, cache immutable build inputs,
and wire SSE to a real parallel backend.

### Application-level integration gaps

- GUI bootstrap defaults to a single worker: `BootstrapConfig.n_jobs = 1`
  (`src/openpkpd_gui/services/bootstrap_service.py:22-29`).
- GUI NPDE goes through `compute_npde()`, which constructs `SimulationEngine`
  without exposing `n_parallel` (`src/openpkpd/plots/diagnostics.py:252-267`).

Even after core optimizations, exposing sensible parallel defaults in these user
paths would improve perceived performance.

### Suggested implementation order

1. Reuse FOCE worker pools and reduce finite-difference overhead.
2. Batch/collapse observation-model work in `IndividualModel`.
3. Optimize simulation output construction in `SimulationEngine`.
4. Fix NPC fallback and vectorize VPC/NPDE aggregation.
5. Optimize NCA grouping and lambda-z regression.
6. Improve orchestration layers: bootstrap payloads, SCM worker model, SSE parallelism.

### Recommendation

If you want measurable results quickly, I would start with **FOCE worker reuse**
and **observation-model batching** first, then benchmark VPC/NPDE on a large
replicate count. Those changes are the most likely to reduce end-user runtime
across multiple workflows rather than only one feature.
