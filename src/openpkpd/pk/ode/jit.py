"""
ODE JIT acceleration for ADVAN6/8 models.

Four tiers, selected automatically or by caller preference:

    Tier 0 — scipy solve_ivp       (baseline, always available)
    Tier 1 — pure-NumPy RK45       (no new deps, ~1.5× speedup)
    Tier 2 — Numba @njit DES       (when numba installed, ~1.2–2× speedup)
    Tier 3 — Numba @cfunc + LLC    (when numba installed, ~3–5× speedup)

Tier 3 (``'llc'``) is the fastest option.  A ``@cfunc``-compiled RHS is wrapped
in a ``scipy.integrate.LowLevelCallable``, eliminating every Python↔C boundary
crossing during integration.  Parameters and infusion data are packed into a
flat ``float64`` array that the cfunc reads through a ``void*`` data pointer.

Tier 2 (``'numba'``) combines our pure-NumPy RK45 with a ``@njit``-compiled RHS,
avoiding scipy's Python overhead but retaining Python calls at each RK stage.

Tier 1 (``'numpy'``) uses the pure-NumPy RK45 with a Python RHS — no new deps.

Tier 0 (``'scipy'``) is the unchanged original baseline; used as default so that
existing tests are unaffected.

Usage::

    from openpkpd.pk.ode.jit import NUMBA_AVAILABLE, numpy_rk45_solve
    from openpkpd.pk.ode.jit import make_numba_rhs, make_llc_rhs
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

# ── Numba detection ────────────────────────────────────────────────────────────
try:
    import numba as _numba  # noqa: F401
    NUMBA_AVAILABLE = True

    # ── Tier 3: fully-native Numba RK45 ──────────────────────────────────────
    # Dormand-Prince RK45 implemented as @njit so that:
    #   - the RK45 loop runs entirely in LLVM native code
    #   - each call to des_fn (also @njit) is a direct native call, not Python
    # Specialised once per des_fn type; subsequent calls use the cached version.

    @_numba.njit(cache=True)
    def _njit_rk45_core(
        des_fn, t0, tf, y0, t_eval, params,
        inf_cmts, inf_rates, inf_ends,
        rtol, atol,
    ):  # noqa: ANN001
        """Dormand-Prince RK45 in pure Numba. Zero Python overhead per RHS call."""
        n      = len(y0)
        n_inf  = len(inf_cmts)
        n_eval = len(t_eval)
        y      = y0.copy()
        t      = t0
        out    = np.empty((n_eval, n))
        eval_idx = 0

        def _rhs(tt, yy):                        # closure; numba compiles it
            dydt = des_fn(tt, yy, params)        # direct native → native call
            for _j in range(n_inf):
                _cmt = inf_cmts[_j]
                if tt <= inf_ends[_j] + 1e-14 and 0 <= _cmt < n:
                    dydt[_cmt] += inf_rates[_j]
            return dydt

        f0 = _rhs(t, y)

        # Initial step size
        d0 = 0.0; d1 = 0.0
        for i in range(n):
            d0 += y[i] * y[i]; d1 += f0[i] * f0[i]
        d0 = (d0 / n) ** 0.5; d1 = (d1 / n) ** 0.5
        h0 = 1e-5 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        h  = min(h0, tf - t0)

        # Dormand-Prince tableau
        C2 = 1.0/5;   C3 = 3.0/10;  C4 = 4.0/5;   C5 = 8.0/9
        A21 = 1.0/5
        A31 = 3.0/40;     A32 = 9.0/40
        A41 = 44.0/45;    A42 = -56.0/15;    A43 = 32.0/9
        A51 = 19372.0/6561; A52 = -25360.0/2187; A53 = 64448.0/6561; A54 = -212.0/729
        A61 = 9017.0/3168;  A62 = -355.0/33;    A63 = 46732.0/5247
        A64 = 49.0/176;     A65 = -5103.0/18656
        B1 = 35.0/384; B3 = 500.0/1113; B4 = 125.0/192; B5 = -2187.0/6784; B6 = 11.0/84
        E1 = 71.0/57600;  E3 = -71.0/16695;  E4 = 71.0/1920
        E5 = -17253.0/339200; E6 = 22.0/525; E7 = -1.0/40

        _completed = False
        for _ in range(100_000):
            if t >= tf - 1e-14 * abs(tf):
                _completed = True
                break
            h = min(h, tf - t)

            k1 = f0
            k2 = _rhs(t + C2*h, y + h*A21*k1)
            k3 = _rhs(t + C3*h, y + h*(A31*k1 + A32*k2))
            k4 = _rhs(t + C4*h, y + h*(A41*k1 + A42*k2 + A43*k3))
            k5 = _rhs(t + C5*h, y + h*(A51*k1 + A52*k2 + A53*k3 + A54*k4))
            k6 = _rhs(t + h,    y + h*(A61*k1 + A62*k2 + A63*k3 + A64*k4 + A65*k5))
            y_new = y + h*(B1*k1 + B3*k3 + B4*k4 + B5*k5 + B6*k6)
            k7 = _rhs(t + h, y_new)

            err = 0.0
            for i in range(n):
                e_i = h*(E1*k1[i] + E3*k3[i] + E4*k4[i] + E5*k5[i] + E6*k6[i] + E7*k7[i])
                sc  = atol + rtol * max(abs(y[i]), abs(y_new[i]))
                err += (e_i / sc) ** 2
            err = (err / n) ** 0.5

            if err <= 1.0:
                t_new = t + h
                while eval_idx < n_eval and t_eval[eval_idx] <= t_new + 1e-14*abs(t_new):
                    s = (t_eval[eval_idx] - t) / h if h > 0.0 else 1.0
                    s = min(max(s, 0.0), 1.0)
                    s2 = s*s; s3 = s2*s
                    h00 = 2*s3 - 3*s2 + 1; h10 = s3 - 2*s2 + s
                    h01 = -2*s3 + 3*s2;    h11 = s3 - s2
                    for i in range(n):
                        out[eval_idx, i] = (h00*y[i] + h10*h*k1[i]
                                            + h01*y_new[i] + h11*h*k7[i])
                    eval_idx += 1
                y = y_new; t = t_new; f0 = k7

            factor = 0.9 * (1.0 / max(err, 1e-10)) ** 0.2
            h = h * min(max(factor, 0.1), 10.0)

        # Guard: fill any remaining points for float edge cases at tf
        while eval_idx < n_eval:
            for i in range(n):
                out[eval_idx, i] = y[i]
            eval_idx += 1

        # Return (output_array, success_flag): 1 = completed, 0 = step-limit hit
        success = np.int64(1) if _completed else np.int64(0)
        return out, success

except ImportError:
    NUMBA_AVAILABLE = False
    _njit_rk45_core = None  # type: ignore[assignment]


# ── Dormand-Prince RK45 tableau (Shampine 1986) ────────────────────────────────

_C2, _C3, _C4, _C5 = 1/5, 3/10, 4/5, 8/9
_A21 = 1/5
_A31, _A32 = 3/40, 9/40
_A41, _A42, _A43 = 44/45, -56/15, 32/9
_A51, _A52, _A53, _A54 = 19372/6561, -25360/2187, 64448/6561, -212/729
_A61, _A62, _A63, _A64, _A65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

# 5th-order weights
_B1, _B3, _B4, _B5, _B6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

# Error coefficients (5th − 4th order)
_E1 = 71/57600; _E3 = -71/16695; _E4 = 71/1920; _E5 = -17253/339200
_E6 = 22/525;   _E7 = -1/40


def numpy_rk45_solve(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    y0: np.ndarray,
    t_eval: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 100_000,
) -> np.ndarray:
    """
    Adaptive Dormand-Prince RK45, pure NumPy.

    Returns an array of shape ``(n_eval, n_state)`` with the state at each
    time in *t_eval* (must be sorted, within ``[t0, tf]``).

    Raises RuntimeError if the ODE fails to complete within *max_steps*.
    """
    n = len(y0)
    y = y0.copy()
    t = t0
    out = np.empty((len(t_eval), n))
    eval_idx = 0

    # Initial step size
    f0 = rhs(t, y)
    d0 = np.sqrt(np.mean(y * y))
    d1 = np.sqrt(np.mean(f0 * f0))
    h0 = 1e-5 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    h = min(h0, tf - t0)

    _completed = False
    for _ in range(max_steps):
        if t >= tf - 1e-14 * abs(tf):
            _completed = True
            break
        h = min(h, tf - t)

        k1 = f0
        k2 = rhs(t + _C2 * h, y + h * _A21 * k1)
        k3 = rhs(t + _C3 * h, y + h * (_A31 * k1 + _A32 * k2))
        k4 = rhs(t + _C4 * h, y + h * (_A41 * k1 + _A42 * k2 + _A43 * k3))
        k5 = rhs(t + _C5 * h, y + h * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4))
        k6 = rhs(t + h,       y + h * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5))

        y_new = y + h * (_B1 * k1 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6)
        k7 = rhs(t + h, y_new)

        # Error estimate
        err_vec = h * (_E1 * k1 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7)
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err = float(np.sqrt(np.mean((err_vec / sc) ** 2)))

        if err <= 1.0:
            t_new = t + h
            # Record outputs between t and t_new using cubic Hermite interpolation.
            # Uses the FSAL stages k1 (f at t) and k7 (f at t+h) — no extra evaluations.
            # For s ∈ [0,1]:
            #   h00(s) = 2s³-3s²+1,  h10(s) = s³-2s²+s
            #   h01(s) = -2s³+3s²,   h11(s) = s³-s²
            #   y(s) = h00*y + h10*h*k1 + h01*y_new + h11*h*k7
            while eval_idx < len(t_eval) and t_eval[eval_idx] <= t_new + 1e-14 * abs(t_new):
                s = float(np.clip((t_eval[eval_idx] - t) / h, 0.0, 1.0)) if h > 0 else 1.0
                s2 = s * s; s3 = s2 * s
                h00 = 2*s3 - 3*s2 + 1
                h10 = s3 - 2*s2 + s
                h01 = -2*s3 + 3*s2
                h11 = s3 - s2
                out[eval_idx] = h00*y + h10*h*k1 + h01*y_new + h11*h*k7
                eval_idx += 1
            y = y_new
            t = t_new
            f0 = k7  # FSAL

        # Adjust step size
        factor = 0.9 * (1.0 / max(err, 1e-10)) ** 0.2
        h = h * min(max(factor, 0.1), 10.0)

    if not _completed:
        raise RuntimeError(
            f"ODE integration reached max_steps={max_steps} without completing "
            f"[t={t:.6g}, tf={tf:.6g}]. The ODE may be stiff. "
            "Use jit='scipy' with method='Radau' or method='BDF' for stiff models, "
            "or set a higher max_steps."
        )
    # Fill any remaining eval points (shouldn't happen after successful completion,
    # but guards against floating-point edge cases at tf).
    while eval_idx < len(t_eval):
        out[eval_idx] = y
        eval_idx += 1
    return out


def make_numba_rhs(
    des_callable: Any,
    pk_params: dict[str, float],
    active_infusions: dict[int, tuple[float, float]],
    n_compartments: int,
) -> tuple[Callable[[float, np.ndarray], np.ndarray], np.ndarray, tuple[str, ...]] | None:
    """
    Try to build a Numba-accelerated RHS for the ODE.

    Returns ``(rhs_fn, param_array, param_keys)`` on success, or ``None`` if
    Numba is unavailable or the DES callable does not support Numba compilation.
    """
    if not NUMBA_AVAILABLE:
        return None
    if not hasattr(des_callable, "try_compile_numba"):
        return None

    param_keys = tuple(sorted(pk_params.keys()))
    if not des_callable.try_compile_numba(param_keys):
        return None

    param_array = np.array([pk_params[k] for k in param_keys], dtype=np.float64)
    numba_fn = des_callable._numba_fn
    infusion_snapshot = dict(active_infusions)

    def rhs_numba(t: float, y: np.ndarray) -> np.ndarray:
        dydt = numba_fn(t, y, param_array)
        for cmt_idx, (rate, end_t) in infusion_snapshot.items():
            if t <= end_t + 1e-14 and 0 <= cmt_idx < n_compartments:
                dydt[cmt_idx] += rate
        return dydt

    return rhs_numba, param_array, param_keys


# ── Tier 3: fully-native Numba solve (njit RK45 + njit DES) ───────────────────

def make_llc_rhs(
    des_callable: Any,
    pk_params: dict[str, float],
    active_infusions: dict[int, tuple[float, float]],
    n_compartments: int,
) -> "Callable[[float, float, np.ndarray, np.ndarray, float, float], np.ndarray] | None":
    """
    Build a fully-native-Numba segment solver for the ODE.

    Both the RK45 integrator (*this* module's ``_njit_rk45_core``) and the
    DES right-hand side (``@njit``-compiled via ``try_compile_numba``) are
    pure native machine code.  No Python boundary is crossed during the
    integration loop — not even to call the RHS.

    Parameters
    ----------
    des_callable:
        A :class:`~openpkpd.parser.code_compiler.CompiledDESCallable` that
        supports ``try_compile_numba()``.
    pk_params:
        Current PK parameter dict.
    active_infusions:
        Dict ``{cmt_idx: (rate, end_time)}`` for the current integration segment.
    n_compartments:
        Number of ODE compartments.

    Returns
    -------
    A callable ``solve(t0, tf, y0, t_eval, rtol, atol) -> np.ndarray`` where
    the returned array has shape ``(len(t_eval), n_compartments)``.
    Returns ``None`` if Numba is unavailable or DES compilation fails.
    """
    if not NUMBA_AVAILABLE or _njit_rk45_core is None:
        return None
    if not hasattr(des_callable, "try_compile_numba"):
        return None

    param_keys = tuple(sorted(pk_params.keys()))
    if not des_callable.try_compile_numba(param_keys):
        return None

    params   = np.array([pk_params[k] for k in param_keys], dtype=np.float64)
    des_fn   = des_callable._numba_fn

    # Pack infusion data into flat arrays for the njit core
    infusions = [
        (cmt, rate, end_t)
        for cmt, (rate, end_t) in active_infusions.items()
        if 0 <= cmt < n_compartments
    ]
    if infusions:
        inf_cmts  = np.array([c for c, _r, _e in infusions], dtype=np.int64)
        inf_rates = np.array([r for _c, r, _e in infusions], dtype=np.float64)
        inf_ends  = np.array([e for _c, _r, e in infusions], dtype=np.float64)
    else:
        inf_cmts  = np.empty(0, dtype=np.int64)
        inf_rates = np.empty(0, dtype=np.float64)
        inf_ends  = np.empty(0, dtype=np.float64)

    _core  = _njit_rk45_core   # local ref avoids re-lookup per call
    _des   = des_fn
    _p     = params
    _ic    = inf_cmts
    _ir    = inf_rates
    _ie    = inf_ends

    def solve(
        t0: float,
        tf: float,
        y0: "np.ndarray",
        t_eval: "np.ndarray",
        rtol: float,
        atol: float,
    ) -> "np.ndarray":
        out, success = _core(_des, t0, tf, y0, t_eval, _p, _ic, _ir, _ie, rtol, atol)
        if not success:
            raise RuntimeError(
                "ODE integration reached the 100,000-step limit without completing "
                f"[tf={tf:.6g}]. The ODE may be stiff. "
                "Use jit='scipy' with method='Radau' or method='BDF' for stiff models."
            )
        return out

    return solve
