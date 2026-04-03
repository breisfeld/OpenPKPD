# Example 29 — Optimal Design with PFIM

This example shows the design API on a simple one-compartment oral model using
the bundled theophylline dataset as the structural template.

Run it with:

```bash
python examples/29_optimal_design.py
```

The script:

- builds a theophylline oral model with fixed population parameters
- evaluates a reference sampling schedule
- optimizes a 4-sample D-optimal schedule over `0.25` to `24` hours
- reports D-efficiency, A-criterion, condition number, and expected SE values

Typical output includes:

- the reference schedule and determinant of its FIM
- the optimized sampling times
- D-efficiency relative to the reference schedule
- expected standard errors from the optimized FIM

This is a good starting point if you want to understand:

- `BuiltModel.design()`
- `PFIMEngine.compute_fim(...)`
- `PFIMEngine.optimize_design(...)`
- `PFIMEngine.efficiency(...)`

Current support boundary:

- The current PFIM implementation assumes a scalar residual variance.
- Multi-endpoint, heteroscedastic, and correlated residual-error structures are not yet implemented in this design path and now fail explicitly instead of being silently approximated.
