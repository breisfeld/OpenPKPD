"""
Covariate modeling for openpkpd.

This package provides:
  - CovariateEffect:        Enum of supported effect types.
  - CovariateRelationship:  Defines one covariate-parameter relationship.
  - SCMEngine:              Stepwise covariate model building engine.
  - SCMResult, SCMStep:     Result containers for SCM.

Quick example::

    from openpkpd.covariate import CovariateRelationship, CovariateEffect, SCMEngine

    candidates = [
        CovariateRelationship('CL', 'WT', CovariateEffect.POWER, reference=70.0),
        CovariateRelationship('V',  'WT', CovariateEffect.POWER, reference=70.0),
        CovariateRelationship('CL', 'AGE', CovariateEffect.LINEAR, reference=40.0),
    ]

    engine = SCMEngine(
        base_model_builder=builder,
        base_pk_code=pk_code,
        candidates=candidates,
        forward_pvalue=0.05,
        backward_pvalue=0.001,
    )
    result = engine.run()
    print(result.summary())
"""

from __future__ import annotations

from openpkpd.covariate.effects import (
    CovariateEffect,
    CovariateRelationship,
    categorical_effect,
    exponential_effect,
    linear_effect,
    power_effect,
)
from openpkpd.covariate.scm import (
    SCMEngine,
    SCMResult,
    SCMStep,
)

__all__ = [
    "CovariateEffect",
    "CovariateRelationship",
    "power_effect",
    "linear_effect",
    "exponential_effect",
    "categorical_effect",
    "SCMStep",
    "SCMResult",
    "SCMEngine",
]
