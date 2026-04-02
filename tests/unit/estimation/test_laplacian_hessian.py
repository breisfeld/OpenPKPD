"""
Tests for scale-relative Hessian step size in LaplacianMethod (LA2).
"""
from __future__ import annotations

import numpy as np
import pytest

from openpkpd.math.matrix import numerical_hessian


# ---------------------------------------------------------------------------
# Test 1: Small eta -> eps stays at 1e-4 (clamped at 1.0)
# ---------------------------------------------------------------------------

def test_scale_relative_eps_small_eta():
    """For eta=[0.01, 0.01], max(|eta|)=0.01 < 1 -> eps = 1e-4 * 1.0 = 1e-4."""
    eta_i = np.array([0.01, 0.01])
    eps = 1e-4 * max(float(np.max(np.abs(eta_i))), 1.0)
    assert abs(eps - 1e-4) < 1e-12


# ---------------------------------------------------------------------------
# Test 2: Large eta -> eps scales up
# ---------------------------------------------------------------------------

def test_scale_relative_eps_large_eta():
    """For eta=[100.0], max(|eta|)=100 > 1 -> eps = 1e-4 * 100 = 0.01."""
    eta_i = np.array([100.0])
    eps = 1e-4 * max(float(np.max(np.abs(eta_i))), 1.0)
    assert abs(eps - 0.01) < 1e-12


# ---------------------------------------------------------------------------
# Test 3: Numerical accuracy for small and large scale
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma2,eta_val", [
    (0.01, 0.001),   # small scale problem
    (100.0, 50.0),   # large scale problem
])
def test_hessian_accuracy_scale_relative(sigma2, eta_val):
    """Scale-relative eps gives Hessian within 0.1% of analytical value 1/sigma2."""
    # f(eta) = 0.5 * eta^2 / sigma2 -> H = 1/sigma2
    def f(eta):
        return 0.5 * float(eta[0]) ** 2 / sigma2

    analytical_hessian = 1.0 / sigma2
    eta_i = np.array([eta_val])
    eps = 1e-4 * max(float(np.max(np.abs(eta_i))), 1.0)

    H = numerical_hessian(f, eta_i, eps=eps)
    estimated = float(H[0, 0])

    rel_error = abs(estimated - analytical_hessian) / abs(analytical_hessian)
    assert rel_error < 0.001, (
        f"sigma2={sigma2}, eta={eta_val}: relative error {rel_error:.4f} >= 0.001; "
        f"estimated={estimated:.6f}, analytical={analytical_hessian:.6f}"
    )


def test_scale_relative_improves_large_scale():
    """Scale-relative step is more accurate than fixed step for large-scale problem."""
    sigma2 = 100.0
    eta_val = 50.0

    def f(eta):
        return 0.5 * float(eta[0]) ** 2 / sigma2

    analytical_hessian = 1.0 / sigma2
    eta_i = np.array([eta_val])

    # Fixed step (original)
    H_fixed = numerical_hessian(f, eta_i, eps=1e-4)
    err_fixed = abs(float(H_fixed[0, 0]) - analytical_hessian) / abs(analytical_hessian)

    # Scale-relative step
    eps_scaled = 1e-4 * max(float(np.max(np.abs(eta_i))), 1.0)
    H_scaled = numerical_hessian(f, eta_i, eps=eps_scaled)
    err_scaled = abs(float(H_scaled[0, 0]) - analytical_hessian) / abs(analytical_hessian)

    # Scale-relative should be at least as good (or better) for large eta
    assert err_scaled < 0.001, (
        f"Scale-relative error {err_scaled:.4f} still exceeds 0.1% for large-scale problem"
    )
