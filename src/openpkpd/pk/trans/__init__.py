"""TRANS parameter transformation routing."""

from __future__ import annotations

from openpkpd.pk.trans.trans1 import apply_trans1
from openpkpd.pk.trans.trans2 import apply_trans2
from openpkpd.pk.trans.trans3 import apply_trans3
from openpkpd.pk.trans.trans4 import apply_trans4
from openpkpd.pk.trans.trans5 import apply_trans5
from openpkpd.pk.trans.trans6 import apply_trans6
from openpkpd.utils.errors import PKError

_TRANS_MAP = {
    1: apply_trans1,
    2: apply_trans2,
    3: apply_trans3,
    4: apply_trans4,
    5: apply_trans5,
    6: apply_trans6,
}


def get_transformer(trans: int, advan: int):
    """Return the transformation callable for a TRANS/ADVAN pair."""
    if trans == 4:
        return lambda params: apply_trans4(params, advan=advan)
    fn = _TRANS_MAP.get(trans)
    if fn is None:
        raise PKError(f"Unsupported TRANS{trans} (ADVAN{advan})")
    return fn


def apply_trans(params: dict[str, float], trans: int, advan: int) -> dict[str, float]:
    """
    Apply the specified TRANS parameterization.

    Args:
        params: Dict of user-specified PK parameters from $PK.
        trans:  TRANS number (1–6).
        advan:  ADVAN number (used for context in error messages).

    Returns:
        Dict with micro rate constants added.
    """
    return get_transformer(trans, advan)(params)
