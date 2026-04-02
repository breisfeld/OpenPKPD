# PBPK Models

Physiologically-based pharmacokinetic (PBPK) model building blocks.

## Base class

```{eval-rst}
.. autoclass:: openpkpd.pk.pbpk.PBPKModel
   :members:
   :undoc-members: False
```

## Five-organ template

```{eval-rst}
.. autoclass:: openpkpd.pk.pbpk.FiveOrganPBPK
   :members:
   :undoc-members: False
```

## Usage

```python
from openpkpd.pk.pbpk import FiveOrganPBPK

model = FiveOrganPBPK(
    organs=["gut", "liver", "lung", "kidney", "muscle"],
    blood_flow_fractions={"gut": 0.18, "liver": 0.065, "lung": 1.0,
                          "kidney": 0.19, "muscle": 0.16},
    tissue_volumes={"gut": 1.2, "liver": 1.8, "lung": 1.0,
                    "kidney": 0.3, "muscle": 35.0},
    cardiac_output=5.0,  # L/min
)
```

See `examples/22_pbpk_model.py` for a full end-to-end example.
