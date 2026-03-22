# Developer performance guide

This page documents the current approach to performance work in OpenPKPD.
It is aimed at contributors who want to speed up fitting, simulation, NCA, or
diagnostics without breaking numerical behavior.

## General practices

Before changing code for speed, follow these rules:

1. **Profile first**
   - Do not optimize from intuition alone.
   - Use the existing profiling scripts and representative workloads before
     deciding where to invest effort.

2. **Prefer narrow, well-guarded optimizations**
   - Fast paths should be activated only when the model shape is clearly
     supported.
   - Preserve a correct fallback for everything else.

3. **Protect numerical behavior first, speed second**
   - Estimation, simulation, and diagnostics are numerically sensitive.
   - A smaller safe speedup is better than a larger but fragile one.

4. **Reuse work instead of recomputing it**
   - Cache repeated parsing, compiled helper generation, repeated plans, and
     last-evaluation bundles where the inputs are stable.
   - Hoist repeated setup/allocation out of inner loops when possible.

5. **Prefer platform-neutral improvements**
   - NumPy/SciPy/Python improvements generally pay off more than solutions that
     depend on optional or harder-to-port runtimes.

6. **Validate every optimization with targeted checks**
   - Run the smallest relevant unit/regression tests first.
   - Re-profile the same workload after the change.

## Existing profiling entry points

Use the repository's existing profiling helpers before introducing new ones:

- `just profile-analysis`
- `uv run python scripts/profile_analysis.py --workloads diagnostics`
- `uv run python scripts/profile_analysis.py --workloads diagnostics_covariate`
- `uv run python scripts/profile_analysis.py --workloads nca`
- `uv run python scripts/profile_pipelines.py`

Useful current helpers:

- `scripts/profile_analysis.py`
  - focused analysis workloads: diagnostics, VPC, NPDE, NCA, covariate-heavy diagnostics
- `scripts/profile_pipelines.py`
  - broader pipeline-level profiling, including representative FOCE workloads
- `artifacts/profiling/`
  - JSON outputs from baseline and follow-up profiling runs

When comparing alternatives, keep the workload, seed, and model shape fixed.

## Patterns that have worked well in this codebase

### 1. Add conservative fast paths for common cases

Examples already present in the codebase include:

- common simulation/error-model fast paths
- cached/default ADVAN2 reuse paths
- symbolic derivative kernels for narrow analytical PK subsets

The common pattern is:

- detect a well-understood subset cheaply
- route to a fast specialized implementation
- fall back to the generic path otherwise

### 2. Cache generated helper code instead of rebuilding it

For symbolic ETA work, the codebase caches **generated NumPy source** to disk
instead of serializing raw SymPy objects. This is preferred because it is easier
to invalidate across library versions and easier to load in new processes.

Use the prewarm helper when you want to avoid first-use cache generation cost:

- `uv run python scripts/prewarm_symbolic_cache.py`

### 3. Reduce repeated allocations and repeated DataFrame work

Many analysis and simulation speedups come from:

- avoiding repeated DataFrame filtering/grouping inside per-subject loops
- reusing precomputed arrays/plans
- skipping materialization of structures that downstream code does not use

These changes are usually safer than algorithmic rewrites.

### 4. Batch work when semantics stay the same

Good batching candidates include:

- evaluating the same objective for many ETA candidates
- aggregating simulation summaries across replicates
- preparing shared subject-level arrays once per workload instead of once per observation

Batching is most useful when it removes Python loop overhead without changing
the optimizer or model semantics.

## Patterns to treat cautiously

### Broad autodiff or runtime replacement

Be cautious about solutions that require replacing the active execution model
for `$PK` / `$ERROR` code or that introduce heavy optional runtime dependencies.

In particular:

- improvements that depend on hard-to-port runtimes may not be worth the
  maintenance cost
- explicit outer-optimizer gradient prototypes should be kept only if profiling
  shows a real win on representative FOCE workloads

### Optimizations that change convergence behavior

Anything that changes optimizer steps, finite-difference behavior, bounds
handling, or initial conditions must be evaluated more carefully than a pure
simulation or reporting optimization.

If a change improves runtime but worsens OFV, convergence, or robustness, do not
keep it without a strong justification.

## Practical workflow for a performance change

1. Reproduce the hotspot with an existing profiling script.
2. Identify the narrowest safe edit.
3. Confirm the exact functions/classes you are changing.
4. Implement the optimization conservatively.
5. Run the smallest relevant tests.
6. Re-run the same profile and compare wall time plus stage totals.
7. Keep the change only if it is measurably faster and behavior remains correct.

## Validation checklist

For most performance PRs, include:

- the command used to profile before the change
- the command used to profile after the change
- the measured wall-time/stage-time difference
- the focused tests that were run
- any remaining limitations or model-shape guards

## Current high-value areas

The active hotspots may shift over time, but recent profiling has repeatedly
pointed developers toward a few shared themes:

- FOCE inner-loop ETA optimization cost
- repeated prediction evaluation work
- simulation engine inner loops
- ADVAN solve/setup overhead
- repeated analysis aggregation work on large subject sets

Contributors should prefer improvements in these shared paths before investing in
more exotic optimizations.