# Population PD Model

Mixed-effects pharmacodynamic model for population-level PD fitting.

## Class

```{eval-rst}
.. autoclass:: openpkpd.models.population_pd.PopulationPDModel
   :members:
   :undoc-members: False
```

## Overview

`PopulationPDModel` wraps any PD callable (Emax, indirect response, etc.) in a
mixed-effects framework.  It follows the same interface as `PopulationModel` so
it works transparently with all estimation methods (FO, FOCE, SAEM, IMP).

```python
from openpkpd.models.population_pd import PopulationPDModel
from openpkpd.models.pkpd import EmaxModel

pd_model = PopulationPDModel(
    pd_callable=EmaxModel(),
    dataset=dataset,
    omega_specs=omega_specs,
    sigma_specs=sigma_specs,
)
result = FOCEMethod().estimate(pd_model, init_params)
```
