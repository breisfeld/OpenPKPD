# Example 22 — PBPK Modelling

**Script:** `examples/22_pbpk_model.py`

Demonstrates the `FiveOrganPBPK` template for physiologically-based
pharmacokinetic (PBPK) modelling.

## Model structure

Five organs with blood-flow-limited distribution:

```
dose → lung ↔ blood ↔ liver (+ CL_liver)
                    ↔ kidney (+ CL_kidney)
                    ↔ gut
                    ↔ muscle
```

Each organ satisfies:

```
dA_organ/dt = Q_organ * (C_blood - C_organ/Kp_organ) [- CL_organ * C_organ]
```

## Usage

```python
from openpkpd.pk.pbpk import FiveOrganPBPK
from openpkpd.data.event_processor import DoseEvent
import numpy as np

model = FiveOrganPBPK()

pk_params = {
    "Q_lung": 350.0, "Q_liver": 90.0, "Q_kidney": 72.0,
    "Q_gut": 60.0,  "Q_muscle": 75.0,
    "V_lung": 0.5,  "V_liver": 1.8,  "V_kidney": 0.3,
    "V_gut": 1.1,   "V_muscle": 35.0, "V_central": 5.0,
    "Kp_lung": 1.2, "Kp_liver": 3.5, "Kp_kidney": 2.8,
    "Kp_gut": 2.1,  "Kp_muscle": 0.9,
    "CL_liver": 15.0, "CL_kidney": 5.0,
}

dose = DoseEvent(time=0.0, amount=100.0, compartment=1)
times = np.linspace(0, 24, 100)
sol = model.solve(pk_params, [dose], times)
```

## Output

`sol.ipred` returns plasma concentration (central compartment / V_central).
`sol.amounts` has shape `(n_times, n_organs+1)` with column order matching
`FiveOrganPBPK.organ_names + ["central"]`.

## Reference

Rowland M et al. (2011). *Physiologically-based pharmacokinetics in drug
development and regulatory science.* Annu Rev Pharmacol Toxicol **51**:45–73.
