"""
Prior information for openpkpd.

Provides the $PRIOR record equivalent: augments the OFV with a Gaussian
(quadratic) prior penalty on THETA (and optionally OMEGA).

Example::

    from openpkpd.prior import PriorSpec, PriorAugmentedModel, make_theta_prior

    prior = make_theta_prior(
        theta_mean=[1.5, 0.08, 30.0],
        theta_cv=0.3,
    )
    aug = PriorAugmentedModel(population_model=pop_model, prior=prior)
"""

from __future__ import annotations

from openpkpd.prior.prior import (
    PriorAugmentedModel,
    PriorSpec,
    make_theta_prior,
    prior_from_control_stream,
)

__all__ = ["PriorSpec", "PriorAugmentedModel", "make_theta_prior", "prior_from_control_stream"]
