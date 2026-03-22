"""
Matrix utilities for openpkpd.

Includes:
  - Cholesky decomposition with PD repair
  - LDLT decomposition
  - Block diagonal matrix assembly
  - Log-determinant computation
  - Positive-definiteness testing and repair
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.linalg import LinAlgError

from openpkpd.utils.errors import NumericalError


def cholesky(mat: np.ndarray, min_diag: float = 1e-10) -> np.ndarray:
    """
    Compute the lower-Cholesky factor of a symmetric positive-definite matrix.

    Raises NumericalError if the matrix is not positive-definite and repair fails.
    """
    mat = ensure_symmetric(mat)
    try:
        return np.linalg.cholesky(mat)
    except LinAlgError:
        repaired = repair_pd(mat, epsilon=min_diag)
        try:
            return np.linalg.cholesky(repaired)
        except LinAlgError as exc:
            raise NumericalError(f"Matrix is not positive-definite: {exc}") from exc


def log_det(mat: np.ndarray) -> float:
    """
    Compute log|det(mat)| for a symmetric positive-definite matrix.

    Uses Cholesky: log|det(M)| = 2 * sum(log(diag(L)))
    """
    L = cholesky(mat)
    return 2.0 * float(np.sum(np.log(np.diag(L))))


def log_det_chol(L: np.ndarray) -> float:
    """Compute log|det(M)| from lower-Cholesky factor L."""
    return 2.0 * float(np.sum(np.log(np.diag(L))))


def solve_triangular(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L @ x = b where L is lower-triangular."""
    from scipy.linalg import solve_triangular as _solve

    return _solve(L, b, lower=True)


def quadratic_form(A: np.ndarray, x: np.ndarray) -> float:
    """Compute x^T @ A^{-1} @ x efficiently via Cholesky."""
    L = cholesky(A)
    z = solve_triangular(L, x)
    return float(z @ z)


def ensure_symmetric(mat: np.ndarray) -> np.ndarray:
    """Return (mat + mat.T) / 2."""
    return (mat + mat.T) / 2


def repair_pd(mat: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """
    Repair a nearly-PD matrix by eigenvalue clipping.

    All eigenvalues below epsilon are replaced with epsilon.
    """
    mat = ensure_symmetric(mat)
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def is_pd(mat: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is symmetric positive-definite."""
    mat = ensure_symmetric(mat)
    try:
        L = np.linalg.cholesky(mat)
        return bool(np.all(np.diag(L) > tol))
    except LinAlgError:
        return False


def block_diag(*mats: np.ndarray) -> np.ndarray:
    """Construct block-diagonal matrix from a list of square matrices."""
    from scipy.linalg import block_diag as _block_diag

    return _block_diag(*mats)


def omega_from_lower_triangle(values: list[float], n: int) -> np.ndarray:
    """
    Reconstruct an n×n symmetric matrix from lower-triangle values
    (column-major order, as in NONMEM $OMEGA BLOCK specification).
    """
    mat = np.zeros((n, n))
    idx = 0
    for col in range(n):
        for row in range(col, n):
            mat[row, col] = values[idx]
            mat[col, row] = values[idx]
            idx += 1
    return mat


def lower_triangle_values(mat: np.ndarray) -> list[float]:
    """Extract lower-triangle values (column-major) from a symmetric matrix."""
    n = mat.shape[0]
    vals: list[float] = []
    for col in range(n):
        for row in range(col, n):
            vals.append(float(mat[row, col]))
    return vals


def numerical_hessian(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute the Hessian of f at x using central finite differences.

    H[i,j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*eps^2)
    """
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pp[i] += eps
            x_pp[j] += eps
            x_pm = x.copy()
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp = x.copy()
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm = x.copy()
            x_mm[i] -= eps
            x_mm[j] -= eps
            H[i, j] = H[j, i] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
    return H


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute gradient of f at x using central finite differences."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g
