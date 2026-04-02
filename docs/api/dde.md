# Delay Differential Equations (DDE)

ODE solver extension for models with delayed state feedback.

## Class

```{eval-rst}
.. autoclass:: openpkpd.pk.ode.dde.DDESubroutine
   :members:
   :undoc-members: False
```

## Overview

`DDESubroutine` extends ADVAN6 to support delayed compartment values in the
`$DES` right-hand side.  The solver integrates piecewise using scipy's
`solve_ivp` with history interpolation.

```python
from openpkpd.pk.ode.dde import DDESubroutine

dde = DDESubroutine(n_compartments=2, delay_params=["TAU"])
```

The `$DES` block may reference `ALAG(n)` or explicit delay parameters.
See `examples/16_dde_model.py` for a full example.
