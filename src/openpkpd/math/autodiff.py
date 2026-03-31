"""Numerical gradient/Hessian/Jacobian utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def gradient(
    f: Callable,
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute gradient of f at x.

    Args:
        f:       Scalar-valued function.
        x:       Point at which to evaluate gradient.
        eps:     Finite-difference step size.

    Returns:
        Gradient array of same shape as x.
    """
    from openpkpd.math.matrix import numerical_gradient

    return numerical_gradient(f, x, eps=eps)


def hessian(
    f: Callable,
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute Hessian matrix of f at x.

    Args:
        f:       Scalar-valued function.
        x:       Point at which to evaluate Hessian.
        eps:     Finite-difference step size.

    Returns:
        Hessian matrix of shape (n, n).
    """
    from openpkpd.math.matrix import numerical_hessian

    return numerical_hessian(f, x, eps=eps)


def jacobian(
    f: Callable,
    x: np.ndarray,
    eps: float = 1e-5,
    *,
    f0: np.ndarray | None = None,
    method: str = "central",
) -> np.ndarray:
    """
    Compute Jacobian matrix of f at x.

    Args:
        f:       Vector-valued function returning array of shape (m,).
        x:       Point of shape (n,) at which to evaluate Jacobian.
        eps:     Finite-difference step size.
        f0:      Optional cached value ``f(x)``.
        method:  Finite-difference scheme: ``"central"`` or ``"forward"``.

    Returns:
        Jacobian matrix of shape (m, n).
    """
    if method not in {"central", "forward"}:
        raise ValueError(f"Unsupported jacobian method: {method!r}")

    f0_arr = np.asarray(f(x), dtype=float) if f0 is None else np.asarray(f0, dtype=float)
    m = len(f0_arr)
    n = len(x)
    J = np.zeros((m, n))
    for j in range(n):
        xp = x.copy()
        xp[j] += eps
        if method == "forward":
            J[:, j] = (np.asarray(f(xp), dtype=float) - f0_arr) / eps
            continue
        xm = x.copy()
        xm[j] -= eps
        J[:, j] = (np.asarray(f(xp), dtype=float) - np.asarray(f(xm), dtype=float)) / (2 * eps)
    return J


def value_and_gradient(
    f: Callable,
    x: np.ndarray,
    eps: float = 1e-5,
) -> tuple[float, np.ndarray]:
    """
    Compute both f(x) and grad f(x) in one call.

    More efficient than calling f and gradient separately.
    """
    val = float(f(x))
    from openpkpd.math.matrix import numerical_gradient

    g = numerical_gradient(f, x, eps=eps)
    return val, g
