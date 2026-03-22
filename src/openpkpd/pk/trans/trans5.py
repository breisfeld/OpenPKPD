"""TRANS5 — micro rate constants for 3-compartment models (identity-like)."""

from __future__ import annotations


def apply_trans5(params: dict[str, float]) -> dict[str, float]:
    """
    TRANS5: User-provided 3-cmt micro rate constants.

    Parameters: K, K12, K21, K13, K31 (and KA for oral models).
    Pass-through (no transformation needed).
    """
    return dict(params)
