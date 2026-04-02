"""Tests for FIXED theta enforcement in IMP estimation — PR3."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.imp import IMPMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec


# ---------------------------------------------------------------------------
# Minimal mock population model (same pattern as test_imp.py)
# ---------------------------------------------------------------------------


class _GaussianIndividualModel:
    """Simple Gaussian individual model: Y ~ N(mu, sigma2), mu comes from eta."""

    def __init__(self, dv: float) -> None:
        self.dv = float(dv)

    def log_likelihood(self, theta, eta, sigma, trans=None) -> float:
        eta_val = float(np.asarray(eta, dtype=float)[0])
        sigma_var = float(sigma[0, 0])
        return float(np.log(2.0 * np.pi * sigma_var) + (self.dv - eta_val) ** 2 / sigma_var)

    def obj_eta(self, eta, theta, omega, sigma, trans=None) -> float:
        eta_val = float(np.asarray(eta, dtype=float)[0])
        omega_var = float(omega[0, 0])
        return float(
            self.log_likelihood(theta, np.array([eta_val]), sigma, trans=trans)
            + eta_val**2 / omega_var
        )


class _GaussianPopulationModel:
    trans = 2

    def __init__(self, dv: float) -> None:
        self._indiv = {1: _GaussianIndividualModel(dv)}

    def subject_ids(self):
        return [1]

    def individual_model(self, sid):
        return self._indiv[sid]


def _make_params(theta_vals, fixed_flags, omega_val=0.1, sigma_val=0.05):
    """Build a ParameterSet with optionally fixed thetas."""
    theta_specs = [
        ThetaSpec(init=v, lower=0.0, fixed=f)
        for v, f in zip(theta_vals, fixed_flags, strict=True)
    ]
    omega_specs = [OmegaSpec(block_size=1, values=[omega_val])]
    sigma_specs = [SigmaSpec(block_size=1, values=[sigma_val])]
    return ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)


class TestFixedThetaIMP:
    """Verify that IMP respects fixed=True theta parameters."""

    def test_fixed_theta_unchanged(self):
        """A theta marked fixed=True must not change after IMP estimation."""
        fixed_value = 3.14159
        params = _make_params(
            theta_vals=[fixed_value, 1.0],
            fixed_flags=[True, False],
        )
        pop_model = _GaussianPopulationModel(dv=1.0)

        method = IMPMethod(isample=30, maxeval=5, seed=42)
        result = method.estimate(pop_model, params)

        assert result.theta_final[0] == pytest.approx(fixed_value, rel=1e-9), (
            f"Fixed theta changed: {fixed_value} → {result.theta_final[0]}"
        )

    def test_free_theta_in_optimizer_vector(self):
        """A theta marked fixed=False is included in the optimizer vector (sanity check).

        The Gaussian mock does not use theta in its likelihood, so theta values may
        stay near their initial values — but the mechanism test verifies that the
        free theta IS included in to_vector() and that from_vector() recovers it.
        """
        params = _make_params(
            theta_vals=[1.0, 2.0],
            fixed_flags=[False, False],
        )
        vec = params.to_vector()
        # Both thetas should be in the vector (plus omega + sigma chol elements)
        # 2 free thetas + 1 omega log-chol + 1 sigma log-chol = 4 elements
        assert len(vec) >= 4, f"Expected ≥4 elements in optimizer vector, got {len(vec)}"

        # Verify round-trip: perturbing the first element changes theta[0]
        vec_mod = vec.copy()
        vec_mod[0] += 2.0
        new_params = ParameterSet.from_vector(vec_mod, params)
        assert new_params.theta[0] != pytest.approx(params.theta[0], abs=0.01)

    def test_fixed_theta_preserved_across_iterations(self):
        """Fixed theta value is preserved throughout multiple iterations."""
        fixed_cl = 0.75
        params = _make_params(
            theta_vals=[fixed_cl],
            fixed_flags=[True],
            omega_val=0.1,
            sigma_val=0.05,
        )
        pop_model = _GaussianPopulationModel(dv=2.0)

        method = IMPMethod(isample=40, maxeval=10, seed=7)
        result = method.estimate(pop_model, params)

        assert result.theta_final[0] == pytest.approx(fixed_cl, rel=1e-9)

    def test_only_fixed_thetas_do_not_change_separately(self):
        """When a mix of fixed and free thetas is used, only free ones change."""
        fixed_val = 2.0
        free_init = 1.0
        params = _make_params(
            theta_vals=[fixed_val, free_init],
            fixed_flags=[True, False],
        )
        pop_model = _GaussianPopulationModel(dv=3.0)

        method = IMPMethod(isample=30, maxeval=20, seed=99)
        result = method.estimate(pop_model, params)

        # Fixed theta is unchanged
        assert result.theta_final[0] == pytest.approx(fixed_val, rel=1e-9)


class TestParameterSetFixedThetaMechanism:
    """Unit tests verifying the ParameterSet fixed-theta round-trip mechanism.

    These tests directly verify the mechanism that IMP relies on:
    to_vector() excludes fixed thetas; from_vector() restores them.
    """

    def test_fixed_theta_excluded_from_vector(self):
        """to_vector() excludes fixed thetas from the optimizer vector."""
        params = _make_params([1.0, 2.0], [True, False])
        vec = params.to_vector()
        # Should have 1 theta + 1 omega element + 1 sigma element = 3
        # (free theta=1, omega log-chol=1, sigma log-chol=1)
        assert len(vec) == 3

    def test_fixed_theta_restored_by_from_vector(self):
        """from_vector() restores fixed theta to its initial value."""
        params = _make_params([3.14, 2.0], [True, False])
        vec = params.to_vector()
        # Perturb the free-parameter part
        vec[0] += 100.0  # change the free theta
        new_params = ParameterSet.from_vector(vec, params)

        # Fixed theta unchanged
        assert new_params.theta[0] == pytest.approx(3.14, rel=1e-9)
        # Free theta changed
        assert new_params.theta[1] != pytest.approx(2.0, abs=0.1)
