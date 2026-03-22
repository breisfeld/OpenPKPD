"""TRANS1 — Identity transformation. User provides micro rate constants directly."""

from __future__ import annotations


def apply_trans1(params: dict[str, float]) -> dict[str, float]:
    """
    TRANS1: No transformation. User-defined micro constants are passed through.

    Expected parameters (by ADVAN):
      ADVAN1: K
      ADVAN2: K, KA
      ADVAN3: K, K12, K21
      ADVAN4: K, K12, K21, KA
      ADVAN11: K, K12, K21, K13, K31
      ADVAN12: K, K12, K21, K13, K31, KA
    """
    return dict(params)
