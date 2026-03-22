"""
openpkpd.inference — model comparison and hypothesis testing utilities.

Provides likelihood ratio tests, AIC/BIC comparison tables, and
Akaike weights for population PK/PD model selection.
"""

from __future__ import annotations

from openpkpd.inference.model_comparison import LRTResult, aic_weights, compare_models, lrt

__all__ = [
    "lrt",
    "compare_models",
    "aic_weights",
    "LRTResult",
]
