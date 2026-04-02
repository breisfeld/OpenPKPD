from __future__ import annotations

import inspect
import logging
import math
import re
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np

from openpkpd.data.blq import blq_log_likelihood, is_blq
from openpkpd.data.event_processor import SubjectEvents
from openpkpd.math.autodiff import jacobian
from openpkpd.math.matrix import numerical_gradient, numerical_hessian
from openpkpd._native import import_core_symbol
from openpkpd.model.residuals import log_likelihood_normal
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.constants import BLQMethod
from openpkpd.utils.errors import PKError

# ---------------------------------------------------------------------------
# Optional Rust-compiled inner-loop extension (openpkpd._core).
# Falls back silently to the pure-Python path if the extension is not built.
# Build:  cd rust && cargo build --release
#         cp target/release/lib_core.so \
#            ../src/openpkpd/_core.cpython-312-x86_64-linux-gnu.so
# ---------------------------------------------------------------------------
_neg2ll_obs_loop_rust = None  # set to callable below if Rust extension is available
try:
    _neg2ll_obs_loop_rust = import_core_symbol("neg2ll_obs_loop")
    _RUST_CORE_AVAILABLE = True
except ImportError:
    _RUST_CORE_AVAILABLE = False

try:
    _native_cvodes_transit_1cmt_pkpd_probe_rust = import_core_symbol(
        "native_cvodes_transit_1cmt_pkpd_probe"
    )
except ImportError:
    _native_cvodes_transit_1cmt_pkpd_probe_rust = None

try:
    _native_cvodes_transit_1cmt_pkpd_probe_multidose_rust = import_core_symbol(
        "native_cvodes_transit_1cmt_pkpd_probe_multidose"
    )
except ImportError:
    _native_cvodes_transit_1cmt_pkpd_probe_multidose_rust = None

try:
    _native_cvodes_transit_1cmt_pkpd_sensitivity_probe_rust = import_core_symbol(
        "native_cvodes_transit_1cmt_pkpd_sensitivity_probe_multidose"
    )
except ImportError:
    _native_cvodes_transit_1cmt_pkpd_sensitivity_probe_rust = None

def _try_import(name: str) -> Any | None:
    try:
        return import_core_symbol(name)
    except ImportError:
        return None

_native_1cmt_iv_probe        = _try_import("native_cvodes_1cmt_iv_probe_multidose")
_native_1cmt_iv_sens_probe   = _try_import("native_cvodes_1cmt_iv_sensitivity_probe_multidose")
_native_1cmt_oral_probe      = _try_import("native_cvodes_1cmt_oral_probe_multidose")
_native_1cmt_oral_sens_probe = _try_import("native_cvodes_1cmt_oral_sensitivity_probe_multidose")
_native_2cmt_iv_probe        = _try_import("native_cvodes_2cmt_iv_probe_multidose")
_native_2cmt_iv_sens_probe   = _try_import("native_cvodes_2cmt_iv_sensitivity_probe_multidose")
_native_2cmt_oral_probe      = _try_import("native_cvodes_2cmt_oral_probe_multidose")
_native_2cmt_oral_sens_probe = _try_import("native_cvodes_2cmt_oral_sensitivity_probe_multidose")
_native_3cmt_iv_probe        = _try_import("native_cvodes_3cmt_iv_probe_multidose")
_native_3cmt_iv_sens_probe   = _try_import("native_cvodes_3cmt_iv_sensitivity_probe_multidose")
_native_3cmt_oral_probe      = _try_import("native_cvodes_3cmt_oral_probe_multidose")
_native_3cmt_oral_sens_probe = _try_import("native_cvodes_3cmt_oral_sensitivity_probe_multidose")
_native_4cmt_iv_probe        = _try_import("native_cvodes_4cmt_iv_probe_multidose")
_native_4cmt_iv_sens_probe   = _try_import("native_cvodes_4cmt_iv_sensitivity_probe_multidose")
_native_4cmt_oral_probe      = _try_import("native_cvodes_4cmt_oral_probe_multidose")
_native_4cmt_oral_sens_probe = _try_import("native_cvodes_4cmt_oral_sensitivity_probe_multidose")
# Infusion-aware probes (IV templates only; rate > 0 triggers dispatch)
_native_1cmt_iv_inf_probe      = _try_import("native_cvodes_1cmt_iv_infusion_probe_multidose")
_native_1cmt_iv_inf_sens_probe = _try_import("native_cvodes_1cmt_iv_infusion_sensitivity_probe_multidose")
_native_2cmt_iv_inf_probe      = _try_import("native_cvodes_2cmt_iv_infusion_probe_multidose")
_native_2cmt_iv_inf_sens_probe = _try_import("native_cvodes_2cmt_iv_infusion_sensitivity_probe_multidose")
_native_3cmt_iv_inf_probe      = _try_import("native_cvodes_3cmt_iv_infusion_probe_multidose")
_native_3cmt_iv_inf_sens_probe = _try_import("native_cvodes_3cmt_iv_infusion_sensitivity_probe_multidose")
_native_4cmt_iv_inf_probe      = _try_import("native_cvodes_4cmt_iv_infusion_probe_multidose")
_native_4cmt_iv_inf_sens_probe = _try_import("native_cvodes_4cmt_iv_infusion_sensitivity_probe_multidose")
# Analytical closed-form probes (ADVAN1/2/3/4 — exact superposition, no ODE integration)
_analytic_1cmt_iv_probe        = _try_import("analytic_1cmt_iv_probe_multidose")
_analytic_1cmt_iv_inf_probe    = _try_import("analytic_1cmt_iv_infusion_probe_multidose")
_analytic_1cmt_oral_probe      = _try_import("analytic_1cmt_oral_probe_multidose")
_analytic_2cmt_iv_probe        = _try_import("analytic_2cmt_iv_probe_multidose")
_analytic_2cmt_iv_inf_probe    = _try_import("analytic_2cmt_iv_infusion_probe_multidose")
_analytic_2cmt_oral_probe      = _try_import("analytic_2cmt_oral_probe_multidose")


class _NativeOdeTemplate:
    """Metadata for one native ODE template.

    ``required_names`` defines BOTH the parameter set that must appear in the
    $PK callable output AND the order in which they are passed to the Rust
    function as ``theta``.  Matching is "all required names present"; templates
    are tried most-specific first so a wider template cannot shadow a narrower
    one when the $PK block only defines the narrower parameter set.
    """

    __slots__ = (
        "name", "required_names", "n_states", "output_cmt_idx",
        "vol_param_name", "is_pkpd", "state_probe_fn", "sens_probe_fn",
        "infusion_state_probe_fn", "infusion_sens_probe_fn",
        "eligible_advans",
    )

    def __init__(
        self,
        name: str,
        required_names: tuple[str, ...],
        n_states: int,
        output_cmt_idx: int,
        vol_param_name: str,
        state_probe_fn: Any,
        sens_probe_fn: Any,
        is_pkpd: bool = False,
        infusion_state_probe_fn: Any = None,
        infusion_sens_probe_fn: Any = None,
        eligible_advans: frozenset[int] = frozenset(),
    ) -> None:
        self.name = name
        self.required_names = required_names
        self.n_states = n_states
        self.output_cmt_idx = output_cmt_idx
        self.vol_param_name = vol_param_name
        self.is_pkpd = is_pkpd
        self.state_probe_fn = state_probe_fn
        self.sens_probe_fn = sens_probe_fn
        self.infusion_state_probe_fn = infusion_state_probe_fn
        self.infusion_sens_probe_fn = infusion_sens_probe_fn
        self.eligible_advans = eligible_advans


# Templates ordered most-specific first (most required_names → least).
# Matching stops at the first template for which ALL required names are
# present in the subject's pk_params dict AND the contract ADVAN is in
# the template's eligible_advans set (or eligible_advans is empty = unrestricted).
#
# Analytical templates come first and carry eligible_advans restrictions so
# they only match ADVAN1/2/3/4.  The CVODES templates that follow are
# unrestricted and serve ADVAN6 (general ODE) models with the same parameter sets.
_NATIVE_ODE_TEMPLATES: list[_NativeOdeTemplate] = [
    # ── Analytical closed-form probes (P1.3) — ADVAN1/2/3/4 only ──────────────
    _NativeOdeTemplate(
        name="analytic_2cmt_oral",  # ADVAN4
        required_names=("KA", "CL", "V2", "Q", "V3"),
        n_states=3, output_cmt_idx=1, vol_param_name="V2",
        state_probe_fn=_analytic_2cmt_oral_probe,
        sens_probe_fn=_native_2cmt_oral_sens_probe,  # CVODES sens remains accurate
        eligible_advans=frozenset({4}),
    ),
    _NativeOdeTemplate(
        name="analytic_2cmt_iv",  # ADVAN3
        required_names=("CL", "V1", "Q", "V2"),
        n_states=2, output_cmt_idx=0, vol_param_name="V1",
        state_probe_fn=_analytic_2cmt_iv_probe,
        sens_probe_fn=_native_2cmt_iv_sens_probe,
        infusion_state_probe_fn=_analytic_2cmt_iv_inf_probe,
        infusion_sens_probe_fn=_native_2cmt_iv_inf_sens_probe,
        eligible_advans=frozenset({3}),
    ),
    _NativeOdeTemplate(
        name="analytic_1cmt_oral",  # ADVAN2
        required_names=("KA", "CL", "V"),
        n_states=2, output_cmt_idx=1, vol_param_name="V",
        state_probe_fn=_analytic_1cmt_oral_probe,
        sens_probe_fn=_native_1cmt_oral_sens_probe,
        eligible_advans=frozenset({2}),
    ),
    _NativeOdeTemplate(
        name="analytic_1cmt_iv",  # ADVAN1
        required_names=("CL", "V"),
        n_states=1, output_cmt_idx=0, vol_param_name="V",
        state_probe_fn=_analytic_1cmt_iv_probe,
        sens_probe_fn=_native_1cmt_iv_sens_probe,
        infusion_state_probe_fn=_analytic_1cmt_iv_inf_probe,
        infusion_sens_probe_fn=_native_1cmt_iv_inf_sens_probe,
        eligible_advans=frozenset({1}),
    ),
    # ── CVODES ODE probes — ADVAN6 (general ODE) and unrestricted ─────────────
    _NativeOdeTemplate(
        name="transit_1cmt_pkpd",
        required_names=("KTR", "KA", "CL", "V", "EMAX", "EC50", "KOUT", "E0"),
        n_states=4, output_cmt_idx=2, vol_param_name="V",
        state_probe_fn=_native_cvodes_transit_1cmt_pkpd_probe_multidose_rust,
        sens_probe_fn=_native_cvodes_transit_1cmt_pkpd_sensitivity_probe_rust,
        is_pkpd=True,
    ),
    _NativeOdeTemplate(
        name="4cmt_oral",
        required_names=("KA", "CL", "V2", "Q3", "V3", "Q4", "V4", "Q5", "V5"),
        n_states=5, output_cmt_idx=1, vol_param_name="V2",
        state_probe_fn=_native_4cmt_oral_probe,
        sens_probe_fn=_native_4cmt_oral_sens_probe,
    ),
    _NativeOdeTemplate(
        name="4cmt_iv",
        required_names=("CL", "V1", "Q2", "V2", "Q3", "V3", "Q4", "V4"),
        n_states=4, output_cmt_idx=0, vol_param_name="V1",
        state_probe_fn=_native_4cmt_iv_probe,
        sens_probe_fn=_native_4cmt_iv_sens_probe,
        infusion_state_probe_fn=_native_4cmt_iv_inf_probe,
        infusion_sens_probe_fn=_native_4cmt_iv_inf_sens_probe,
    ),
    _NativeOdeTemplate(
        name="3cmt_oral",
        required_names=("KA", "CL", "V2", "Q3", "V3", "Q4", "V4"),
        n_states=4, output_cmt_idx=1, vol_param_name="V2",
        state_probe_fn=_native_3cmt_oral_probe,
        sens_probe_fn=_native_3cmt_oral_sens_probe,
    ),
    _NativeOdeTemplate(
        name="3cmt_iv",
        required_names=("CL", "V1", "Q2", "V2", "Q3", "V3"),
        n_states=3, output_cmt_idx=0, vol_param_name="V1",
        state_probe_fn=_native_3cmt_iv_probe,
        sens_probe_fn=_native_3cmt_iv_sens_probe,
        infusion_state_probe_fn=_native_3cmt_iv_inf_probe,
        infusion_sens_probe_fn=_native_3cmt_iv_inf_sens_probe,
    ),
    _NativeOdeTemplate(
        name="2cmt_oral",
        required_names=("KA", "CL", "V2", "Q", "V3"),
        n_states=3, output_cmt_idx=1, vol_param_name="V2",
        state_probe_fn=_native_2cmt_oral_probe,
        sens_probe_fn=_native_2cmt_oral_sens_probe,
    ),
    _NativeOdeTemplate(
        name="2cmt_iv",
        required_names=("CL", "V1", "Q", "V2"),
        n_states=2, output_cmt_idx=0, vol_param_name="V1",
        state_probe_fn=_native_2cmt_iv_probe,
        sens_probe_fn=_native_2cmt_iv_sens_probe,
        infusion_state_probe_fn=_native_2cmt_iv_inf_probe,
        infusion_sens_probe_fn=_native_2cmt_iv_inf_sens_probe,
    ),
    _NativeOdeTemplate(
        name="1cmt_oral",
        required_names=("KA", "CL", "V"),
        n_states=2, output_cmt_idx=1, vol_param_name="V",
        state_probe_fn=_native_1cmt_oral_probe,
        sens_probe_fn=_native_1cmt_oral_sens_probe,
    ),
    _NativeOdeTemplate(
        name="1cmt_iv",
        required_names=("CL", "V"),
        n_states=1, output_cmt_idx=0, vol_param_name="V",
        state_probe_fn=_native_1cmt_iv_probe,
        sens_probe_fn=_native_1cmt_iv_sens_probe,
        infusion_state_probe_fn=_native_1cmt_iv_inf_probe,
        infusion_sens_probe_fn=_native_1cmt_iv_inf_sens_probe,
    ),
]

_NATIVE_ODE_TEMPLATE_MAP: dict[str, _NativeOdeTemplate] = {t.name: t for t in _NATIVE_ODE_TEMPLATES}

# ADVAN6 (general ODE) uses CVODES probes.
# ADVAN1/2/3/4 are routed through exact analytical Rust probes (P1.3):
#   ADVAN1 → analytic_1cmt_iv, ADVAN2 → analytic_1cmt_oral,
#   ADVAN3 → analytic_2cmt_iv, ADVAN4 → analytic_2cmt_oral.
# The eligible_advans field on each template restricts matching so ADVAN6
# models are never accidentally dispatched to an analytical probe.
_NATIVE_ELIGIBLE_ADVANS: frozenset[int] = frozenset({1, 2, 3, 4, 6})

# Error-model patterns for which the native ODE path is supported.
# "mixed_pkpd_dvid_theta" uses dual-DVID routing; the rest are single-output.
_NATIVE_SUPPORTED_ERROR_MODELS: frozenset[str] = frozenset({
    "mixed_pkpd_dvid_theta",
    "proportional",
    "additive",
    "combined_eps",
    "proportional_theta",
    "additive_theta",
    "combined_theta",
})

# BLQMethod string → integer code expected by the Rust function
_BLQ_METHOD_CODE: dict[str | None, int] = {
    None: 0,
    BLQMethod.M1: 1,
    BLQMethod.M2: 2,
    BLQMethod.M3: 3,
    BLQMethod.M4: 4,
    BLQMethod.M5: 5,
    BLQMethod.M6: 6,
    BLQMethod.M7: 7,
}


_NAN_LLOQ_CACHE: dict[int, np.ndarray] = {}
_NAN_LLOQ_CACHE_LOCK = threading.Lock()


def _apply_alag(
    dose_times: list[float],
    dose_compartments: list[int],
    pk_params: dict[str, float],
) -> list[float]:
    """Return dose times shifted by any ALAG{cmt} values present in pk_params.

    Mirrors the adjustment performed by advan6._prepare_doses(): for each dose
    event the lag time for its target compartment is read from pk_params under
    the key ``ALAG{compartment}`` (e.g. ``ALAG1`` for compartment 1).  If no
    lag time exists for a given compartment the dose time is left unchanged.

    This is a pure function with no side-effects; it always returns a new list.
    When no ALAG keys are present in pk_params the original list is returned
    unchanged (no allocation).
    """
    # Fast path: if no ALAG keys exist in pk_params, skip all work.
    if not any(k.startswith("ALAG") for k in pk_params):
        return dose_times

    adjusted: list[float] = []
    for t, cmt in zip(dose_times, dose_compartments):
        lag = pk_params.get(f"ALAG{cmt}", 0.0)
        adjusted.append(t + float(lag))
    return adjusted


def _build_lloq_array(lloq: object, n: int) -> np.ndarray:
    """Return a float64 array of length *n* with per-obs LLOQ values.

    NaN encodes "no LLOQ for this observation" (i.e. normal non-BLQ obs).
    Accepts None (all NaN), a scalar float, or an array.

    The all-NaN case (lloq=None, the common path) reuses a cached array to
    avoid allocating a new one on every log_likelihood call.
    """
    if lloq is None:
        with _NAN_LLOQ_CACHE_LOCK:
            cached = _NAN_LLOQ_CACHE.get(n)
            if cached is None:
                cached = np.full(n, np.nan, dtype=np.float64)
                _NAN_LLOQ_CACHE[n] = cached
        return cached
    out = np.full(n, np.nan, dtype=np.float64)
    if np.ndim(lloq) == 0:
        out[:] = float(lloq)  # type: ignore[arg-type]
    else:
        arr = np.asarray(lloq, dtype=float)
        length = min(len(arr), n)
        out[:length] = arr[:length]
    return out

_W_PROP_THETA_RE = re.compile(r"^w=f\*theta\[(\d+)\]$", re.IGNORECASE)
_W_THETA_RE = re.compile(r"^w=theta\[(\d+)\]$", re.IGNORECASE)
_W_SQRT_RE = re.compile(
    r"^w=math\.sqrt\(theta\[(\d+)\]\*\*2\+\(f\*theta\[(\d+)\]\)\*\*2\)$",
    re.IGNORECASE,
)
_DVID_THEN_RE = re.compile(r"^if\(dvid==(\d+)\):$", re.IGNORECASE)
_MIXED_PKPD_IPRED_RE = re.compile(r"^f=theta\[(\d+)\]\+a\[(\d+)\]$", re.IGNORECASE)
_MIXED_PKPD_SQRT_RE = re.compile(
    r"^w=math\.sqrt\(\(([a-z_][a-z0-9_]*)\*f\)\*\*2\+([a-z_][a-z0-9_]*)\*\*2\)$",
    re.IGNORECASE,
)
_MIXED_PKPD_Y_BRANCH_RE = re.compile(
    r"^y=f\+w\*eps\[(\d+)\]$",
    re.IGNORECASE,
)
_MIXED_PKPD_ALIAS_RE = re.compile(
    r"^([a-z_][a-z0-9_]*)=theta\[(\d+)\]$",
    re.IGNORECASE,
)



def _theta_to_pk_params(
    theta: np.ndarray,
    eta: np.ndarray,
    trans: int,
) -> dict[str, float]:
    """
    Fallback: map theta vector to PK params for simple models.

    For TRANS2: theta[0]=CL, theta[1]=V (or theta[0]=KA, theta[1]=CL, theta[2]=V for ADVAN2)
    """
    if trans == 2:
        if len(theta) >= 3:
            return {
                "KA": float(theta[0]) * math.exp(float(eta[0]) if len(eta) > 0 else 0),
                "CL": float(theta[1]) * math.exp(float(eta[1]) if len(eta) > 1 else 0),
                "V": float(theta[2]) * math.exp(float(eta[2]) if len(eta) > 2 else 0),
            }
        elif len(theta) >= 2:
            return {
                "CL": float(theta[0]) * math.exp(float(eta[0]) if len(eta) > 0 else 0),
                "V": float(theta[1]) * math.exp(float(eta[1]) if len(eta) > 1 else 0),
            }
    return {"K": float(theta[0]), "V": 1.0}
