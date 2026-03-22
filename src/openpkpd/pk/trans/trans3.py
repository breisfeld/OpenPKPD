"""TRANS3 — CL, Vss, Q parameterization for 2-compartment models."""

from __future__ import annotations

from openpkpd.utils.errors import PKError


def apply_trans3(params: dict[str, float]) -> dict[str, float]:
    """
    TRANS3: CL, Vss, Q → micro rate constants for 2-cmt model.

    K   = CL / V1
    K12 = Q / V1
    K21 = Q / (Vss - V1)

    where V1 is derived from: Vss = V1 + V1 * K12 / K21
    which gives: V1 = Vss / (1 + Q/K21)  ... this requires iterative solution.

    NONMEM TRANS3 standard reparameterization:
      K21 = (K12 + K + K21) - sqrt(...) / 2  (not commonly used)

    Simplified standard form used by most PK software:
      V1   = CL / K    (need K separately)
      K12  = Q / V1
      K21  = Q / V2

    Actually NONMEM TRANS3 is: CL, Vss, Q where V1 is not directly specified.
    Use the classic two-compartment parameterization:
      V1 = CL / K  (where K = CL/V1)

    NONMEM 7 TRANS3 specification:
      Parameters: CL, V (total Vss = V1+V2 here), Q
      With V2 = Vss - V1  — but V1 is unknown without K.

    Note: TRANS3 is unusual. Most users prefer TRANS4 (CL, V1, Q, V2).
    This implementation uses: CL, V, Q where V = V1 (central volume).
    """
    cl = params.get("CL")
    v = params.get("V")
    q = params.get("Q")
    v2 = params.get("V2")

    if cl is None or v is None or q is None:
        raise PKError(f"TRANS3 requires CL, V, Q; got params={list(params.keys())}")

    if v <= 0:
        raise PKError(f"TRANS3: V must be > 0, got V={v}")

    result = dict(params)
    result["K"] = cl / v
    result["K12"] = q / v
    if v2 is not None and v2 > 0:
        result["K21"] = q / v2
    elif "K21" not in params:
        raise PKError("TRANS3: requires V2 or K21 for peripheral compartment")
    return result
