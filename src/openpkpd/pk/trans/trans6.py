"""TRANS6 — CL/V parameterization for 3-compartment models."""

from __future__ import annotations

from openpkpd.utils.errors import PKError


def apply_trans6(params: dict[str, float]) -> dict[str, float]:
    """
    TRANS6: CL, V1, Q2, V2, Q3, V3 → micro rate constants for 3-cmt model.

    K   = CL / V1
    K12 = Q2 / V1
    K21 = Q2 / V2
    K13 = Q3 / V1
    K31 = Q3 / V3
    """
    cl = params.get("CL")
    v1 = params.get("V1")
    if v1 is None:
        v1 = params.get("V")
    q2 = params.get("Q2")
    if q2 is None:
        q2 = params.get("Q")
    v2 = params.get("V2")
    q3 = params.get("Q3")
    v3 = params.get("V3")

    if any(x is None for x in [cl, v1, q2, v2, q3, v3]):
        raise PKError(f"TRANS6 requires CL, V1, Q2, V2, Q3, V3; got params={list(params.keys())}")

    result = dict(params)
    result["K"] = cl / v1  # type: ignore[operator]
    result["K12"] = q2 / v1  # type: ignore[operator]
    result["K21"] = q2 / v2  # type: ignore[operator]
    result["K13"] = q3 / v1  # type: ignore[operator]
    result["K31"] = q3 / v3  # type: ignore[operator]
    return result
