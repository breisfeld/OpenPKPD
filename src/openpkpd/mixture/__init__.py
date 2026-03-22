"""
Mixture models for openpkpd.

Provides discrete K-subpopulation mixture models fitted via the EM algorithm.

Example::

    from openpkpd.mixture import MixtureModel, MixtureResult

    model = MixtureModel(population_model=pop_model, n_subpop=2)
    result = model.fit(init_params=params)
    print(result.summary())
"""

from __future__ import annotations

from openpkpd.mixture.mixture import MixtureModel, MixtureResult

__all__ = ["MixtureModel", "MixtureResult"]
