"""Unit tests for matrix utilities."""

import math

import numpy as np
import pytest

from openpkpd.math.matrix import (
    cholesky,
    is_pd,
    log_det,
    numerical_gradient,
    numerical_hessian,
    repair_pd,
)


@pytest.mark.unit
class TestCholesky:
    def test_basic_pd(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        L = cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_identity(self):
        A = np.eye(3)
        L = cholesky(A)
        np.testing.assert_allclose(L, np.eye(3), rtol=1e-10)

    def test_repair_near_pd(self):
        """Near-PD matrix should be repaired and Cholesky should succeed."""
        A = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
        L = cholesky(A, min_diag=1e-7)  # should not raise
        assert np.all(np.diag(L) > 0)


@pytest.mark.unit
class TestLogDet:
    def test_identity(self):
        A = np.eye(3)
        assert log_det(A) == pytest.approx(0.0)

    def test_diagonal(self):
        A = np.diag([2.0, 3.0, 4.0])
        expected = math.log(2) + math.log(3) + math.log(4)
        assert log_det(A) == pytest.approx(expected)

    def test_2x2(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        # det = 4*3 - 2*2 = 8
        assert log_det(A) == pytest.approx(math.log(8), rel=1e-8)


@pytest.mark.unit
class TestRepairPD:
    def test_already_pd(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        repaired = repair_pd(A)
        np.testing.assert_allclose(repaired, A, rtol=1e-8)

    def test_not_pd_repaired(self):
        # Rank-1 matrix (not PD)
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        repaired = repair_pd(A, epsilon=1e-6)
        assert is_pd(repaired)

    def test_symmetry_preserved(self):
        A = np.array([[2.0, 0.5], [0.5, 1.0]])
        repaired = repair_pd(A)
        np.testing.assert_allclose(repaired, repaired.T, rtol=1e-12)


@pytest.mark.unit
class TestNumericalDerivatives:
    def test_gradient_quadratic(self):
        """Gradient of 0.5*x^T*x = x."""

        def f(x):
            return 0.5 * float(x @ x)

        x = np.array([1.0, 2.0, 3.0])
        g = numerical_gradient(f, x)
        np.testing.assert_allclose(g, x, rtol=1e-4)

    def test_hessian_quadratic(self):
        """Hessian of 0.5*x^T*A*x = A."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])

        def f(x):
            return 0.5 * float(x @ A @ x)

        x = np.array([1.0, -1.0])
        H = numerical_hessian(f, x)
        np.testing.assert_allclose(H, A, rtol=1e-3)


@pytest.mark.unit
def test_is_pd():
    assert is_pd(np.eye(3))
    assert not is_pd(np.zeros((2, 2)))
    assert is_pd(np.diag([1.0, 2.0, 3.0]))
