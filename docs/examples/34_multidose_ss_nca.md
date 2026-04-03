# Example 34 — Multi-Dose Steady-State NCA

**Script:** `examples/34_multidose_ss_nca.py`

This example computes steady-state NCA metrics for a repeated oral dosing
regimen by explicitly simulating repeated doses and analyzing the final dosing
interval.

## What it shows

- single-dose `AUC(0-inf)` as the reference exposure,
- last-interval steady-state `AUCtau`,
- `Ctrough`, `Cpeak_ss`, and `Cavg_ss`,
- accumulation ratio `R_ac`,
- percent fluctuation.

## Why the final-interval workflow matters

The example uses an explicit repeated-dose simulation and then slices the last
interval before running steady-state NCA. That is more defensible than relying
on a simplified `SS=1` setup when you want the plotted profile and reported SS
metrics to match the same simulated trajectory.
