## Analysis and simulation profiling report

Date: 2026-03-16

Raw profile output: `artifacts/profiling/profile_pipelines.json`
Harness: `scripts/profile_pipelines.py`

For the contributor-facing workflow on how to turn profiling output into safe
optimizations, see the {doc}`developer performance guide </developer_performance>`.

### Workloads profiled

- **FOCE fit**: 6-subject oral PK model, FOCEI, `maxeval=12`, serial inner loop
- **VPC**: 24 subjects, 500 replicates, 8 bins, serial simulation
- **NPDE**: 24 subjects, 500 replicates, serial simulation

### Executive summary

Profiling confirms that the recent VPC/NPDE optimizations moved the bottleneck
away from diagnostic post-processing and back into the **shared simulation and
estimation cores**.

Main takeaways:

1. **FOCE is still dominated by inner-loop eta optimization**, but the
   representative workload is now much faster.
2. A **narrow SymPy-generated eta-gradient path** removes most of the prior
   finite-difference cost for the supported `ADVAN2 / TRANS2 / proportional-error`
   subset while preserving a fallback path for everything else.
3. **VPC and NPDE are now mostly dominated by `SimulationEngine`**, not by
   percentile/PDE post-processing.
4. The next shared bottlenecks are now **prediction evaluation**, the ODE solve
   path, and remaining unsupported FOCE models that still use finite differences.

### FOCE profile highlights

- Wall time: **5.43 s**
- `foce.inner_loop`: **4.04 s** (~74% of wall time)
- `foce.outer_ofv`: **1.28 s** (~24%)
- `individual.evaluate_observation_model`: **1.09 s** (nested in inner loop)
- `individual._evaluate_predictions`: **0.21 s**

Top cumulative functions:

- `scipy.optimize.minimize` / `_minimize_lbfgsb`
- SciPy differentiable-function gradient updates (`fun_and_grad`, `_update_grad`)
- `FOCEMethod._inner_loop`
- `IndividualModel.evaluate_observation_model`

#### What this means

FOCE is still driven by repeated eta optimization, but the hot representative
model now uses a **SymPy-generated analytical eta gradient** for a narrow,
conservative subset:

- `ADVAN2`
- `TRANS=2`
- canonical exponential ETA model on `KA`, `CL`, and `V`
- proportional error model `Y = F*(1 + EPS(1))`
- bolus-dose / no-covariate / no-BLQ path

When a model falls outside that subset, FOCE still falls back to the previous
batched finite-difference path.

An alternating A/B benchmark against that fallback baseline showed the symbolic
path was about **88% faster** on the representative workload, and it also
reached a slightly lower OFV on that benchmark/model.

A further safe reduction now skips compartment-amount materialization during
observation-model evaluation when the compiled `$ERROR` block does not use
`A(...)`. This is active for the representative FOCE workload
(`Y = F*(1 + EPS(1))`). The measured gain is modest/noisy, but it removes real
work without changing optimizer semantics.

The practical implication is that the best next FOCE work is likely:

1. extending the symbolic path to more common PK/error-model subsets,
2. reducing observation-model cost further inside the remaining unsupported
   eta-objective path, or
3. revisiting broader analytical/autodiff support only after the compiler/runtime
   path becomes more derivative-friendly.

Broader analytical eta gradients and general autodiff were investigated in this
pass, but they are not yet practical as a universal solution because the active
`$PK` and `$ERROR` execution path is still `exec(...)`-based. The SymPy route is
therefore deliberately narrow and opt-in-by-model-shape rather than generic.

### VPC profile highlights

- Wall time: **1.94 s**
- `simulation.simulate`: **1.39 s** (~72%)
- `vpc.sim_percentiles`: **0.44 s** (~22%)
- `vpc.obs_percentiles`: **0.01 s** (~1%)

Top cumulative functions:

- `SimulationEngine.simulate`
- `SimulationEngine._simulate_one_replicate`
- `IndividualModel._evaluate_predictions`
- `ADVAN2.solve`
- NumPy percentile aggregation in VPC summarization

#### What this means

The VPC aggregation path is no longer the main problem. The remaining runtime is
mostly in **simulation itself**, especially:

- per-replicate work in `_simulate_one_replicate()`
- repeated individual prediction evaluation
- ODE solve work in `ADVAN2.solve`

### NPDE profile highlights

- Wall time: **1.71 s**
- `simulation.simulate`: **1.32 s** (~77%)
- `npde.build_sim_matrix`: **0.18 s** (~10%)
- `npde.compute_pd`: **0.003 s**
- `npde.decorrelate`: **0.011 s**

Top cumulative functions:

- `SimulationEngine.simulate`
- `SimulationEngine._simulate_one_replicate`
- `IndividualModel._evaluate_predictions`
- `ADVAN2.solve`
- NPDE matrix merge/alignment work

#### What this means

The NPDE-specific post-processing is now relatively cheap. Further NPDE speedups
will mostly come from making the **shared simulation layer** faster.

### Cross-pipeline conclusion

After the recent optimization passes, **the common bottlenecks are now more
important than the diagnostic-specific ones**.

The strongest shared targets are:

1. `SimulationEngine._simulate_one_replicate()`
2. `IndividualModel.evaluate()` / `_evaluate_predictions()`
3. `ADVAN2.solve`
4. FOCE eta-objective work for unsupported models / broader symbolic coverage

### Recommended next optimization order

1. **Reduce repeated per-observation work in prediction evaluation**
   - look for safe batching opportunities in `IndividualModel.evaluate()`
   - reduce repeated covariate/amount materialization where possible

2. **Investigate ODE solve cost in `ADVAN2.solve`**
   - look for repeated setup/allocation that can be hoisted
   - verify whether solver inputs can be prepared in a cheaper form

3. **Broaden the FOCE symbolic fast path carefully**
   - target additional common PK/error subsets with strict fallback guards
   - keep reducing unsupported-model eta cost via cheaper batched observation-model evaluation

### Practical recommendation

If the goal is to improve **simulation/diagnostic runtime** next, work on
`IndividualModel._evaluate_predictions()` / `ADVAN2.solve` first.

If the goal is to improve **fitting runtime**, work on **FOCE eta-gradient cost**
first.