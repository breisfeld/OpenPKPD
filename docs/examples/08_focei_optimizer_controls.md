# Example 8 — FOCEI Optimizer Controls from Python

**Script:** `examples/25_focei_optimizer_controls.py`

This example shows how to enable the new FOCEI robustness controls directly
from the Python API and run a complete fit.

## Goal

Use FOCEI with:

- multiple starts
- an explicit outer optimizer
- a fallback polish optimizer
- best-iterate retention
- structured retry after abnormal termination

## Example

```python
from openpkpd.api.model_builder import ModelBuilder

built = (
    ModelBuilder()
    .data("examples/shared_data/theophylline/theophylline.csv")
    .subroutines(advan=2, trans=2)
    .pk(
        """
        KA = THETA(1) * EXP(ETA(1))
        CL = THETA(2) * EXP(ETA(2))
        V  = THETA(3) * EXP(ETA(3))
        """
    )
    .error("Y = F * (1 + EPS(1))")
    .theta([(0.01, 1.5, 20.0), (0.001, 0.08, 5.0), (0.1, 30.0, 500.0)])
    .omega([0.5, 0.3, 0.3])
    .sigma(0.1)
    .estimation(
        method="FOCEI",
        maxeval=40,
        n_starts=2,
        outer_optimizer="L-BFGS-B",
        outer_fallback_optimizer="Powell",
        outer_fallback_maxeval=15,
        retain_best_iterate=True,
        retry_on_abnormal=True,
        retry_omega_scales=(0.5, 0.25),
    )
    .build()
)

result = built.fit()
print(result.summary())
```

## When to use these controls

- `n_starts`: when the fit is sensitive to initials or shows clear local minima
- `outer_fallback_optimizer`: when gradient-based termination is acceptable but
  a short derivative-free polish can recover a slightly better basin
- `retain_best_iterate`: when the terminal iterate is not always the best point visited
- `retry_on_abnormal` and `retry_omega_scales`: when FOCEI fails or terminates abnormally

## Related example

Run it directly with:

```bash
python examples/25_focei_optimizer_controls.py
```

For the control-stream form of the same idea, see Example 9.
