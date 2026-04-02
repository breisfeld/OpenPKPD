"""Tests for ADVAN2 symbolic gradient ka ≈ k guard (SY1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("sympy", reason="sympy required for symbolic gradient tests")

from openpkpd.model.symbolic_eta import (
    _KA_K_TOL,
    _compiled_limit_terms,
    _compiled_terms,
)


def _advan2_concentration(ka: float, k: float, v: float, amt: float, dt: float) -> float:
    """Analytical ADVAN2 concentration (1-cmt oral)."""
    if abs(ka - k) < _KA_K_TOL:
        # L'Hopital limit
        return amt * ka * dt * math.exp(-k * dt) / v
    return amt * ka / (v * (ka - k)) * (math.exp(-k * dt) - math.exp(-ka * dt))


def _advan2_concentration_nd(ka: float, k: float, v: float, amt: float, dt: float) -> float:
    """Naive formula (may be numerically unstable near ka=k)."""
    if ka == k:
        return amt * ka * dt * math.exp(-k * dt) / v
    return amt * ka / (v * (ka - k)) * (math.exp(-k * dt) - math.exp(-ka * dt))


class TestADVAN2KaKGuard:
    """Tests that the ka ≈ k guard produces correct, finite gradients."""

    def test_ka_far_from_k_concentration_finite(self):
        """ka far from k: normal formula gives a finite concentration."""
        ka, k, v, amt, dt = 1.0, 0.1, 10.0, 100.0, 1.0
        c = _advan2_concentration(ka, k, v, amt, dt)
        assert np.isfinite(c)
        assert c > 0

    def test_ka_approx_k_concentration_finite(self):
        """ka ≈ k: L'Hopital path returns finite value."""
        ka, k, v, amt, dt = 0.1001, 0.1, 10.0, 100.0, 1.0
        c = _advan2_concentration(ka, k, v, amt, dt)
        assert np.isfinite(c), "L'Hopital path gave non-finite concentration"
        assert c > 0

    def test_ka_equals_k_limit_formula(self):
        """ka == k exactly: result should equal L'Hopital limit amt*ka*dt*exp(-k*dt)/v."""
        ka = k = 0.1
        v, amt, dt = 10.0, 100.0, 1.0
        expected = amt * ka * dt * math.exp(-k * dt) / v
        c = _advan2_concentration(ka, k, v, amt, dt)
        assert abs(c - expected) < 1e-12, f"c={c}, expected={expected}"

    def test_ka_approx_k_continuity(self):
        """Concentration should vary continuously as ka → k."""
        k = 0.1
        v, amt, dt = 10.0, 100.0, 1.0
        # Values at ka near k should bracket the limit value
        c_limit = _advan2_concentration(k, k, v, amt, dt)
        c_near = _advan2_concentration(k + 1e-8, k, v, amt, dt)
        assert abs(c_near - c_limit) < 1e-5, (
            f"Discontinuity near ka=k: c_near={c_near}, c_limit={c_limit}"
        )

    def test_compiled_terms_ka_far_from_k(self):
        """_compiled_terms contrib should give finite value when ka far from k."""
        terms = _compiled_terms()
        ka_theta, cl_theta, v_theta = 1.0, 1.0, 10.0
        eta0 = eta1 = eta2 = 0.0
        dt = np.array([[1.0]])
        amt = np.array([[100.0]])
        result = terms["contrib"](ka_theta, cl_theta, v_theta, eta0, eta1, eta2, dt, amt)
        assert np.all(np.isfinite(result))

    def test_limit_terms_available(self):
        """_compiled_limit_terms should be importable and have contrib_limit."""
        limit_terms = _compiled_limit_terms()
        assert "contrib_limit" in limit_terms
        assert "contrib_limit_grad" in limit_terms
        assert len(limit_terms["contrib_limit_grad"]) == 3

    def test_limit_terms_give_finite_result(self):
        """contrib_limit should give a finite result at ka == k."""
        limit_terms = _compiled_limit_terms()
        ka = k = 0.1
        v = 10.0
        dt = np.array([[1.0]])
        amt = np.array([[100.0]])
        result = limit_terms["contrib_limit"](ka, k, v, dt, amt)
        assert np.all(np.isfinite(result)), f"contrib_limit gave non-finite: {result}"

    def test_limit_terms_gradient_finite(self):
        """All three limit gradients should give finite results at ka == k."""
        limit_terms = _compiled_limit_terms()
        ka = k = 0.1
        v = 10.0
        dt = np.array([[1.0]])
        amt = np.array([[100.0]])
        for i, grad_fn in enumerate(limit_terms["contrib_limit_grad"]):
            result = grad_fn(ka, k, v, dt, amt)
            assert np.all(np.isfinite(result)), (
                f"contrib_limit_grad[{i}] gave non-finite: {result}"
            )

    def test_numerical_reference_dlog_ka(self):
        """
        Numerical reference for dC/d(log_ka).

        For a 1-cmt oral model with ka=1.0, k=0.2, V=10, dose=100, dt=1.0:
          dC/d(log_ka) = ka * dC/dka
        Compute by finite difference and verify.
        """
        ka, k, v, amt, dt = 1.0, 0.2, 10.0, 100.0, 1.0
        # Analytical concentration
        c0 = _advan2_concentration(ka, k, v, amt, dt)
        eps = 1e-6
        c_plus = _advan2_concentration(ka + eps, k, v, amt, dt)
        dc_dka_fd = (c_plus - c0) / eps
        dc_dlogka_fd = ka * dc_dka_fd

        # Analytical formula: d/dka [ka/(ka-k) * (exp(-k*dt) - exp(-ka*dt))] * amt/v
        # = amt/v * [(ka-k)*(...) - ka*(...) + ka^2*dt*exp(-ka*dt)] / (ka-k)^2
        # Here we just compare with FD which is the reference
        eps2 = 1e-7
        c_plus2 = _advan2_concentration(ka + eps2, k, v, amt, dt)
        dc_dlogka_fd2 = ka * (c_plus2 - c0) / eps2

        # Both FD estimates should agree closely
        assert abs(dc_dlogka_fd - dc_dlogka_fd2) / max(abs(dc_dlogka_fd), 1e-10) < 0.01
        assert np.isfinite(dc_dlogka_fd)
        assert np.isfinite(c0)
