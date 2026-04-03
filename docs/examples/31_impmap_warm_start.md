# Example 31 — IMPMAP Warm-Start Diagnostics

**Script:** `examples/31_impmap_warm_start.py`

This example demonstrates the FOCEI warm-start path used by `IMPMAP` on the
bundled warfarin PK dataset. It intentionally runs only a very short IMPMAP
outer optimization so the example stays quick and focused on the warm-start
diagnostics rather than on a full IMP benchmark run.

## What it shows

- `IMPMAP` invokes a short FOCEI pass before the importance-sampling objective.
- Warm-start diagnostics are attached under `result.diagnostics["warm_start"]`.
- The short outer run records an OFV history without needing a long adaptive-ESS
  optimization.

## Key output

The script prints:

- whether the FOCEI warm-start was used,
- whether the warm-start itself converged,
- the warm-start OFV and message,
- the short-run IMPMAP OFV and THETA vector,
- the number of recorded OFV evaluations.

## Notes

- This is a diagnostics-oriented example, not a full validated IMPMAP benchmark.
- For stronger numerical validation of the warfarin IMPMAP path, see the
  empirical reference coverage in `tests/external_validation/test_imp_empirical_reference.py`.
