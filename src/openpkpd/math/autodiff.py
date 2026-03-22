"""
JAX gradient/Hessian/Jacobian utilities.

Provides a unified interface that falls back to numerical differentiation
when JAX is not available or when working with non-JAX callables.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def gradient(
    f: Callable,
    x: np.ndarray,
    use_jax: bool = True,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute gradient of f at x.

    Args:
        f:       Scalar-valued function.
        x:       Point at which to evaluate gradient.
        use_jax: Use JAX autodiff if available; fall back to FD otherwise.
        eps:     Finite-difference step size (used if JAX unavailable).

    Returns:
        Gradient array of same shape as x.
    """
    if use_jax and JAX_AVAILABLE:
        x_jax = jnp.array(x, dtype=jnp.float64)
        g = jax.grad(f)(x_jax)
        return np.asarray(g)
    else:
        from openpkpd.math.matrix import numerical_gradient

        return numerical_gradient(f, x, eps=eps)


def hessian(
    f: Callable,
    x: np.ndarray,
    use_jax: bool = True,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute Hessian matrix of f at x.

    Args:
        f:       Scalar-valued function.
        x:       Point at which to evaluate Hessian.
        use_jax: Use JAX autodiff if available; fall back to FD otherwise.
        eps:     Finite-difference step size (used if JAX unavailable).

    Returns:
        Hessian matrix of shape (n, n).
    """
    if use_jax and JAX_AVAILABLE:
        x_jax = jnp.array(x, dtype=jnp.float64)
        H = jax.hessian(f)(x_jax)
        return np.asarray(H)
    else:
        from openpkpd.math.matrix import numerical_hessian

        return numerical_hessian(f, x, eps=eps)


def jacobian(
    f: Callable,
    x: np.ndarray,
    use_jax: bool = True,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute Jacobian matrix of f at x.

    Args:
        f:       Vector-valued function returning array of shape (m,).
        x:       Point of shape (n,) at which to evaluate Jacobian.
        use_jax: Use JAX autodiff if available; fall back to FD otherwise.
        eps:     Finite-difference step size.

    Returns:
        Jacobian matrix of shape (m, n).
    """
    if use_jax and JAX_AVAILABLE:
        x_jax = jnp.array(x, dtype=jnp.float64)
        J = jax.jacobian(f)(x_jax)
        return np.asarray(J)
    else:
        # Numerical Jacobian via forward differences
        f0 = np.asarray(f(x), dtype=float)
        m = len(f0)
        n = len(x)
        J = np.zeros((m, n))
        for j in range(n):
            xp = x.copy()
            xp[j] += eps
            xm = x.copy()
            xm[j] -= eps
            J[:, j] = (np.asarray(f(xp)) - np.asarray(f(xm))) / (2 * eps)
        return J


def value_and_gradient(
    f: Callable,
    x: np.ndarray,
    use_jax: bool = True,
    eps: float = 1e-5,
) -> tuple[float, np.ndarray]:
    """
    Compute both f(x) and grad f(x) in one call.

    More efficient than calling f and gradient separately.
    """
    if use_jax and JAX_AVAILABLE:
        x_jax = jnp.array(x, dtype=jnp.float64)
        val, g = jax.value_and_grad(f)(x_jax)
        return float(val), np.asarray(g)
    else:
        val = float(f(x))
        from openpkpd.math.matrix import numerical_gradient

        g = numerical_gradient(f, x, eps=eps)
        return val, g
