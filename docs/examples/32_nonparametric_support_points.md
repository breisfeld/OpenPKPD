# Example 32 — Nonparametric Support Points

**Script:** `examples/32_nonparametric_support_points.py`

This example demonstrates nonparametric population estimation on a synthetic
one-compartment oral PK dataset with a single ETA on clearance. The synthetic
dataset keeps runtime short while still producing a non-degenerate support-point
distribution.

## What it shows

- `NONPARAMETRIC` estimation with a deterministic synthetic dataset.
- A support-point approximation to the empirical ETA distribution.
- The highest-weight support points and empirical ETA moments.

## Model

The example uses:

- ADVAN2 / TRANS2
- fixed structural parameters `KA` and `V`
- one random effect on `CL`
- proportional residual error

This narrower setup is deliberate: it is much easier to obtain an informative,
fast support-point example with one ETA than with a heavier multi-ETA real-data
workflow.

## Notes

- The printed `converged` flag comes from the base parametric fit used to seed
  the support points. The support-point distribution itself is the important
  artifact in this example.
- The example is intended as a runnable illustration, not an external
  scientific reference dataset.
