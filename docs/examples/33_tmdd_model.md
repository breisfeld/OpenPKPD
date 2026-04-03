# Example 33 — TMDD Model Comparison

**Script:** `examples/33_tmdd_model.py`

This example compares three target-mediated drug disposition formulations under
the same dose scenario:

- full TMDD,
- quasi-steady-state (QSSA),
- Michaelis-Menten approximation.

## What it shows

- peak concentration differences across the three formulations,
- AUC differences over `0–168 h`,
- optional plots for free drug, free target, and drug-target complex.

## Notes

- The Michaelis-Menten approximation is intentionally shown next to the full
  TMDD and QSSA solutions because its exposure can diverge materially.
- The script now uses `np.trapezoid()` for numerical integration, avoiding the
  deprecated `np.trapz()` path.
