"""Direct tests for autodiff wrapper numerics."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.math.autodiff import gradient, hessian, jacobian, value_and_gradient


@pytest.mark.unit
class TestAutodiffFallback:
    def test_gradient_matches_polynomial_closed_form(self):
        def f(x: np.ndarray) -> float:
            return float(x[0] ** 3 + x[0] * x[1] + 2.0 * x[1] ** 2)

        x = np.array([1.5, -2.0])
        expected = np.array([3.0 * x[0] ** 2 + x[1], x[0] + 4.0 * x[1]])

        np.testing.assert_allclose(gradient(f, x, use_jax=False), expected, rtol=1e-5, atol=1e-7)

    def test_hessian_matches_quadratic_closed_form(self):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])

        def f(x: np.ndarray) -> float:
            return 0.5 * float(x @ A @ x)

        x = np.array([1.0, -1.5])

        np.testing.assert_allclose(hessian(f, x, use_jax=False), A, rtol=1e-3, atol=1e-5)

    def test_jacobian_matches_vector_valued_closed_form(self):
        def f(x: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    x[0] ** 2 + x[1],
                    x[0] - 2.0 * x[1] ** 3,
                ]
            )

        x = np.array([1.2, -0.5])
        expected = np.array(
            [
                [2.0 * x[0], 1.0],
                [1.0, -6.0 * x[1] ** 2],
            ]
        )

        np.testing.assert_allclose(jacobian(f, x, use_jax=False), expected, rtol=1e-5, atol=1e-7)

    def test_value_and_gradient_matches_function_and_gradient(self):
        def f(x: np.ndarray) -> float:
            return float(x[0] ** 2 + 3.0 * x[0] * x[1] + x[1] ** 2)

        x = np.array([1.0, 2.0])

        value, grad = value_and_gradient(f, x, use_jax=False)

        assert value == pytest.approx(f(x))
        np.testing.assert_allclose(grad, gradient(f, x, use_jax=False), rtol=1e-8, atol=1e-10)
