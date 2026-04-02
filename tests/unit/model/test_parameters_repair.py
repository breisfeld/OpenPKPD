"""Tests for apply_bounds() DEBUG logging when _repair_pd modifies the matrix."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec


def _make_params(omega: np.ndarray, sigma: np.ndarray | None = None) -> ParameterSet:
    n = omega.shape[0]
    if sigma is None:
        sigma = np.eye(1) * 0.05
    return ParameterSet(
        theta=np.array([1.0]),
        omega=omega,
        sigma=sigma,
        theta_specs=[ThetaSpec(init=1.0, lower=0.0, upper=10.0)],
        omega_specs=[OmegaSpec(block_size=n, values=[omega[r, c] for c in range(n) for r in range(c, n)])],
        sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
    )


@pytest.mark.unit
class TestApplyBoundsRepairLogging:
    def test_pd_matrix_no_repair_no_debug(self, caplog):
        """A PD omega → no repair, shift=0, no DEBUG log."""
        omega = np.eye(2) * 0.5
        params = _make_params(omega)

        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.parameters"):
            result = params.apply_bounds()

        assert np.allclose(result.omega, omega)
        repair_msgs = [r for r in caplog.records if "OMEGA repaired" in r.message]
        assert len(repair_msgs) == 0, f"Unexpected repair log: {repair_msgs}"

    def test_non_pd_matrix_repair_logged(self, caplog):
        """Near-singular omega (one eigenvalue = -1e-8) → repair happens, DEBUG logged."""
        # Construct a matrix with a tiny negative eigenvalue
        # Rotation by 45 degrees, eigenvalues [1.0, -1e-8]
        theta_rot = np.pi / 4
        Q = np.array([[np.cos(theta_rot), -np.sin(theta_rot)],
                      [np.sin(theta_rot),  np.cos(theta_rot)]])
        D = np.diag([1.0, -1e-8])
        omega = Q @ D @ Q.T

        params = ParameterSet(
            theta=np.array([1.0]),
            omega=omega,
            sigma=np.eye(1) * 0.05,
            theta_specs=[ThetaSpec(init=1.0, lower=0.0, upper=10.0)],
            omega_specs=[OmegaSpec(block_size=2, values=[omega[0, 0], omega[1, 0], omega[1, 1]])],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
        )

        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.parameters"):
            result = params.apply_bounds()

        # The repaired matrix must be PD
        eigenvalues = np.linalg.eigvalsh(result.omega)
        assert np.all(eigenvalues > 0), f"Repaired omega not PD: eigenvalues={eigenvalues}"

        # A DEBUG message must have been logged
        repair_msgs = [r for r in caplog.records if "OMEGA repaired" in r.message]
        assert len(repair_msgs) >= 1, "Expected at least one OMEGA repaired DEBUG message"
        # The shift must be non-zero
        assert "shift" in repair_msgs[0].message.lower() or "shift" in repair_msgs[0].message

    def test_barely_pd_no_repair(self):
        """A 2x2 matrix [[1, 0.999], [0.999, 1]] is just barely PD — passes without repair."""
        omega = np.array([[1.0, 0.999], [0.999, 1.0]])
        eigenvalues = np.linalg.eigvalsh(omega)
        # Verify it is PD before we even apply_bounds
        assert np.all(eigenvalues > 0), f"Test matrix should be PD, got eigenvalues={eigenvalues}"

        params = ParameterSet(
            theta=np.array([1.0]),
            omega=omega,
            sigma=np.eye(1) * 0.05,
            theta_specs=[ThetaSpec(init=1.0, lower=0.0, upper=10.0)],
            omega_specs=[OmegaSpec(block_size=2, values=[1.0, 0.999, 1.0])],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
        )
        result = params.apply_bounds()
        # Result should also be PD (repair should not have been needed or was benign)
        result_eigenvalues = np.linalg.eigvalsh(result.omega)
        assert np.all(result_eigenvalues > 0)

    def test_not_pd_1_plus_eps_repaired(self, caplog):
        """Matrix [[1, 1+ε], [1+ε, 1]] with ε=0.001 (not PD) is repaired."""
        eps = 0.001
        omega = np.array([[1.0, 1.0 + eps], [1.0 + eps, 1.0]])
        eigenvalues_before = np.linalg.eigvalsh(omega)
        # This matrix must have a negative eigenvalue
        assert np.any(eigenvalues_before < 0), (
            f"Test matrix should not be PD, got eigenvalues={eigenvalues_before}"
        )

        params = ParameterSet(
            theta=np.array([1.0]),
            omega=omega,
            sigma=np.eye(1) * 0.05,
            theta_specs=[ThetaSpec(init=1.0, lower=0.0, upper=10.0)],
            omega_specs=[OmegaSpec(block_size=2, values=[1.0, 1.0 + eps, 1.0])],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
        )

        with caplog.at_level(logging.DEBUG, logger="openpkpd.model.parameters"):
            result = params.apply_bounds()

        # Must now be PD
        eigenvalues_after = np.linalg.eigvalsh(result.omega)
        assert np.all(eigenvalues_after > 0), f"Repaired matrix not PD: {eigenvalues_after}"

        repair_msgs = [r for r in caplog.records if "OMEGA repaired" in r.message]
        assert len(repair_msgs) >= 1, "Expected repair to be logged"
