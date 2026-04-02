#!/usr/bin/env python3
"""
Split src/openpkpd/model/individual.py into a package.

Run from the repo root:
    python scripts/split_individual.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

SRC = Path("src/openpkpd/model/individual.py")
PKG = Path("src/openpkpd/model/individual")

lines = SRC.read_text().splitlines(keepends=True)


def get_lines(start: int, end: int) -> str:
    """Return lines[start-1 : end] (1-based, inclusive)."""
    return "".join(lines[start - 1 : end])


# ── _base.py ──────────────────────────────────────────────────────────────────
# Lines 1-388: module imports, _NativeOdeTemplate, constants, helpers, regexes
# We also move _theta_to_pk_params here (was lines 2497-2520) so that
# _observation.py can import it without circular imports.
base_header = textwrap.dedent("""\
    \"\"\"
    Module-level singletons for the individual model package.

    Contains: Rust probe imports, _NativeOdeTemplate, native ODE template list
    and associated constants, LLOQ cache, BLQ method codes, helper functions
    (_apply_alag, _build_lloq_array, _try_import), and regex patterns used by
    the error-model inference logic.

    Also exports _theta_to_pk_params (fallback PK param mapping) so that
    _observation.py can import it without creating a circular dependency.
    \"\"\"
""")

base_body = (
    base_header
    + get_lines(1, 388)  # Everything up to (not including) the IndividualModel class
    + "\n\n"
    + get_lines(2497, 2520)  # _theta_to_pk_params
)
(PKG / "_base.py").write_text(base_body)
print(f"Wrote _base.py  ({len(base_body.splitlines())} lines)")

# ── _pk_solution.py ────────────────────────────────────────────────────────────
pk_solution_header = textwrap.dedent("""\
    \"\"\"PKSolutionMixin — native ODE dispatch, template selection, sensitivity probes.\"\"\"
    from __future__ import annotations

    import logging
    from typing import TYPE_CHECKING, Any
    from collections.abc import Callable

    import numpy as np

    from openpkpd.model.individual._base import (
        _NativeOdeTemplate,
        _NATIVE_ODE_TEMPLATES,
        _NATIVE_ELIGIBLE_ADVANS,
        _NATIVE_SUPPORTED_ERROR_MODELS,
        _apply_alag,
    )
    from openpkpd.pk.base import PKSolution

    if TYPE_CHECKING:
        pass

    logger = logging.getLogger(__name__)


    class PKSolutionMixin:
        \"\"\"Mixin providing native ODE dispatch and CVODES sensitivity methods.\"\"\"

""")

# Body: lines 540-1326 (up to @staticmethod before _infer_common_error_model)
# Indent each line by 4 spaces since they're already indented as class methods
pk_body_raw = get_lines(540, 1326)
pk_solution_body = pk_solution_header + pk_body_raw + "\n"
(PKG / "_pk_solution.py").write_text(pk_solution_body)
print(f"Wrote _pk_solution.py  ({len(pk_solution_body.splitlines())} lines)")

# ── _observation.py ────────────────────────────────────────────────────────────
obs_header = textwrap.dedent("""\
    \"\"\"ObservationModelMixin — error model inference, prediction, observation model.\"\"\"
    from __future__ import annotations

    import logging
    import re
    from typing import Any
    from collections.abc import Callable

    import numpy as np

    from openpkpd.model.individual._base import (
        _W_PROP_THETA_RE,
        _W_THETA_RE,
        _W_SQRT_RE,
        _DVID_THEN_RE,
        _MIXED_PKPD_IPRED_RE,
        _MIXED_PKPD_SQRT_RE,
        _MIXED_PKPD_Y_BRANCH_RE,
        _MIXED_PKPD_ALIAS_RE,
        _theta_to_pk_params,
    )
    from openpkpd.utils.errors import PKError

    logger = logging.getLogger(__name__)


    class ObservationModelMixin:
        \"\"\"Mixin providing error model inference and prediction evaluation.\"\"\"

""")

# Body: lines 1327-2040 (from @staticmethod _infer_common_error_model to log_likelihood)
obs_body_raw = get_lines(1327, 2040)
obs_body = obs_header + obs_body_raw + "\n"
(PKG / "_observation.py").write_text(obs_body)
print(f"Wrote _observation.py  ({len(obs_body.splitlines())} lines)")

# ── _likelihood.py ─────────────────────────────────────────────────────────────
ll_header = textwrap.dedent("""\
    \"\"\"LikelihoodMixin — log-likelihood, obj_eta, obj_eta_many.\"\"\"
    from __future__ import annotations

    import logging
    import math

    import numpy as np

    from openpkpd.data.blq import blq_log_likelihood, is_blq
    from openpkpd.model.individual._base import (
        _build_lloq_array,
        _BLQ_METHOD_CODE,
        _RUST_CORE_AVAILABLE,
        _neg2ll_obs_loop_rust,
    )
    from openpkpd.model.residuals import log_likelihood_normal
    from openpkpd.utils.constants import BLQMethod

    logger = logging.getLogger(__name__)


    class LikelihoodMixin:
        \"\"\"Mixin providing log-likelihood and eta-objective evaluation.\"\"\"

""")

# Body: lines 2041-2239
ll_body_raw = get_lines(2041, 2239)
ll_body = ll_header + ll_body_raw + "\n"
(PKG / "_likelihood.py").write_text(ll_body)
print(f"Wrote _likelihood.py  ({len(ll_body.splitlines())} lines)")

# ── _derivatives.py ────────────────────────────────────────────────────────────
deriv_header = textwrap.dedent("""\
    \"\"\"DerivativesMixin — Jacobians, Hessians, supports_* capabilities, penalty.\"\"\"
    from __future__ import annotations

    import logging
    from typing import Any

    import numpy as np

    from openpkpd.math.autodiff import jacobian
    from openpkpd.math.matrix import numerical_gradient, numerical_hessian

    logger = logging.getLogger(__name__)


    class DerivativesMixin:
        \"\"\"Mixin providing gradient, Jacobian, and Hessian methods.\"\"\"

""")

# Body: lines 2241-2495
deriv_body_raw = get_lines(2241, 2495)
deriv_body = deriv_header + deriv_body_raw + "\n"
(PKG / "_derivatives.py").write_text(deriv_body)
print(f"Wrote _derivatives.py  ({len(deriv_body.splitlines())} lines)")

# ── _model.py ──────────────────────────────────────────────────────────────────
model_header = textwrap.dedent("""\
    \"\"\"IndividualModel — composes all four mixins into the public class.\"\"\"
    from __future__ import annotations

    import inspect
    import logging
    from typing import Any
    from collections.abc import Callable

    import numpy as np

    from openpkpd.data.event_processor import SubjectEvents
    from openpkpd.pk.base import PKSubroutine
    from openpkpd.utils.constants import BLQMethod
    from openpkpd.utils.errors import PKError

    from openpkpd.model.individual._pk_solution import PKSolutionMixin
    from openpkpd.model.individual._observation import ObservationModelMixin
    from openpkpd.model.individual._likelihood import LikelihoodMixin
    from openpkpd.model.individual._derivatives import DerivativesMixin

    logger = logging.getLogger(__name__)


    class IndividualModel(
        PKSolutionMixin,
        ObservationModelMixin,
        LikelihoodMixin,
        DerivativesMixin,
    ):
        \"\"\"
        Evaluates the individual log-likelihood and OFV contribution.

        Holds references to the population-level PK model and compiled
        callables so it can be called repeatedly during inner-loop
        optimization (EBE estimation).

        Implemented as a composition of four mixins:
          - PKSolutionMixin      (native ODE dispatch, CVODES sensitivities)
          - ObservationModelMixin (error model inference, predictions)
          - LikelihoodMixin      (log-likelihood, eta objective)
          - DerivativesMixin     (Jacobians, Hessians, supports_* capabilities)
        \"\"\"

""")

# Body: lines 399-539 (from __init__ to end of __setstate__ / _build_observation_dvid)
model_body_raw = get_lines(399, 539)
model_body = model_header + model_body_raw + "\n"
(PKG / "_model.py").write_text(model_body)
print(f"Wrote _model.py  ({len(model_body.splitlines())} lines)")

# ── __init__.py ────────────────────────────────────────────────────────────────
init_content = textwrap.dedent("""\
    \"\"\"
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
    \"\"\"
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
""")
(PKG / "__init__.py").write_text(init_content)
print(f"Wrote __init__.py  ({len(init_content.splitlines())} lines)")

print("\nAll files written.  Next steps:")
print("  1. Verify imports are correct")
print("  2. Delete src/openpkpd/model/individual.py")
print("  3. Run tests")
