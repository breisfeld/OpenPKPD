"""
openpkpd.model.individual
=========================
Per-subject model evaluation package.

``IndividualModel`` is the public API.  All other names are re-exported
for backward compatibility with existing imports:

    from openpkpd.model.individual import IndividualModel
    from openpkpd.model.individual import _theta_to_pk_params   # pkpd.py
    from openpkpd.model.individual import _apply_alag           # pfim.py
    from openpkpd.model.individual import _NATIVE_ODE_TEMPLATES # pfim.py
    from openpkpd.model.individual import _NativeOdeTemplate    # tests
    from openpkpd.model.individual import _NAN_LLOQ_CACHE, _NAN_LLOQ_CACHE_LOCK
    from openpkpd.model.individual import _build_lloq_array     # tests
    from openpkpd.model.individual import _try_import           # benchmarks
"""
from openpkpd.model.individual._base import (  # noqa: F401
    _try_import,
    _NativeOdeTemplate,
    _NATIVE_ODE_TEMPLATES,
    _NATIVE_ODE_TEMPLATE_MAP,
    _NATIVE_ELIGIBLE_ADVANS,
    _NATIVE_SUPPORTED_ERROR_MODELS,
    _BLQ_METHOD_CODE,
    _NAN_LLOQ_CACHE,
    _NAN_LLOQ_CACHE_LOCK,
    _apply_alag,
    _build_lloq_array,
    _theta_to_pk_params,
    _RUST_CORE_AVAILABLE,
)
from openpkpd.model.individual._model import IndividualModel  # noqa: F401

__all__ = [
    "IndividualModel",
    "_theta_to_pk_params",
    "_apply_alag",
    "_NATIVE_ODE_TEMPLATES",
    "_NATIVE_ODE_TEMPLATE_MAP",
    "_NativeOdeTemplate",
    "_NATIVE_ELIGIBLE_ADVANS",
    "_NATIVE_SUPPORTED_ERROR_MODELS",
    "_NAN_LLOQ_CACHE",
    "_NAN_LLOQ_CACHE_LOCK",
    "_build_lloq_array",
    "_try_import",
    "_BLQ_METHOD_CODE",
    "_RUST_CORE_AVAILABLE",
]
