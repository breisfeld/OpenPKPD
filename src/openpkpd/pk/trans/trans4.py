"""TRANS4 — macro-parameter mappings for analytical 2-compartment models."""

from __future__ import annotations

from openpkpd.utils.errors import PKError


def apply_trans4(params: dict[str, float], advan: int | None = None) -> dict[str, float]:
    """
    TRANS4: map macro parameters to micro-rate constants.

    Supported layouts:

    - ADVAN3 / generic 2-cmt IV: ``CL, V1, Q, V2``
      ``K = CL / V1``, ``K12 = Q / V1``, ``K21 = Q / V2``

    - ADVAN4 2-cmt oral: ``KA, CL, V2, Q, V3``
      ``K = CL / V2``, ``K12 = Q / V2``, ``K21 = Q / V3``

    For ADVAN4 (2-cmt oral), KA is passed through.
    """
    cl = params.get("CL")
    q = params.get("Q")

    if cl is None:
        raise PKError(f"TRANS4 requires CL; got params={list(params.keys())}")
    if q is None:
        raise PKError(f"TRANS4 requires Q; got params={list(params.keys())}")

    advan4_layout = (
        advan == 4
        and params.get("V3") is not None
        and params.get("V1") is None
        and params.get("V") is None
    )
    if advan4_layout:
        central_v = params.get("V2")
        peripheral_v = params.get("V3")
        if central_v is None or peripheral_v is None:
            raise PKError(f"TRANS4 for ADVAN4 requires V2 and V3; got params={list(params.keys())}")
        if central_v <= 0:
            raise PKError(f"TRANS4: V2 must be > 0, got V2={central_v}")
        if peripheral_v <= 0:
            raise PKError(f"TRANS4: V3 must be > 0, got V3={peripheral_v}")

        result = dict(params)
        result["K"] = cl / central_v
        result["K12"] = q / central_v
        result["K21"] = q / peripheral_v
        return result

    v1 = params.get("V1")
    if v1 is None:
        v1 = params.get("V")
    v2 = params.get("V2")
    if v1 is None:
        raise PKError(f"TRANS4 requires V1 (or V); got params={list(params.keys())}")
    if v2 is None:
        raise PKError(f"TRANS4 requires V2; got params={list(params.keys())}")

    if v1 <= 0:
        raise PKError(f"TRANS4: V1 must be > 0, got V1={v1}")
    if v2 <= 0:
        raise PKError(f"TRANS4: V2 must be > 0, got V2={v2}")

    result = dict(params)
    result["V1"] = v1
    result["K"] = cl / v1
    result["K12"] = q / v1
    result["K21"] = q / v2
    return result
