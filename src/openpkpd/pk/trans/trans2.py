"""TRANS2 — CL, V parameterization for 1-compartment models."""

from __future__ import annotations

from openpkpd.utils.errors import PKError


def apply_trans2(params: dict[str, float]) -> dict[str, float]:
    """
    TRANS2: Convert CL, V (and optionally KA) to micro rate constants.

    K = CL / V

    For ADVAN2 (1-cmt oral):
      K  = CL / V
      KA = KA  (pass-through)
    """
    cl = params.get("CL")
    v = params.get("V")

    if cl is None or v is None:
        raise PKError(f"TRANS2 requires CL and V; got params={list(params.keys())}")
    if v <= 0:
        raise PKError(f"TRANS2: V must be > 0, got V={v}")

    result = dict(params)
    result["K"] = cl / v
    result["V"] = v
    return result
