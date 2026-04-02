"""Tests for n_iov_occasions() with various SAME patterns."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec


def _make_params_from_omega_specs(specs: list[OmegaSpec]) -> ParameterSet:
    """Build a ParameterSet from a list of OmegaSpec (theta/sigma are trivial)."""
    # Build the omega matrix from specs (handling SAME by reusing the previous non-SAME spec)
    n_total = sum(s.block_size for s in specs)
    omega = np.zeros((n_total, n_total))
    offset = 0
    last_base_mat = None
    for spec in specs:
        n = spec.block_size
        if spec.same:
            if last_base_mat is not None:
                omega[offset:offset + n, offset:offset + n] = last_base_mat
        else:
            mat = spec.to_matrix()
            omega[offset:offset + n, offset:offset + n] = mat
            last_base_mat = mat
        offset += n

    return ParameterSet(
        theta=np.array([1.0]),
        omega=omega,
        sigma=np.eye(1) * 0.05,
        theta_specs=[ThetaSpec(init=1.0, lower=0.0, upper=10.0)],
        omega_specs=specs,
        sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
    )


@pytest.mark.unit
class TestNIOVOccasions:
    def test_no_same_returns_1(self):
        """No SAME blocks → 1 occasion."""
        specs = [OmegaSpec(block_size=1, values=[0.1])]
        params = _make_params_from_omega_specs(specs)
        assert params.n_iov_occasions() == 1

    def test_base_same_returns_2(self):
        """BASE, SAME → 2 occasions."""
        specs = [
            OmegaSpec(block_size=1, values=[0.1]),
            OmegaSpec(block_size=1, values=[], same=True),
        ]
        params = _make_params_from_omega_specs(specs)
        assert params.n_iov_occasions() == 2

    def test_base_same_same_returns_3(self):
        """BASE, SAME, SAME → 3 occasions."""
        specs = [
            OmegaSpec(block_size=1, values=[0.1]),
            OmegaSpec(block_size=1, values=[], same=True),
            OmegaSpec(block_size=1, values=[], same=True),
        ]
        params = _make_params_from_omega_specs(specs)
        assert params.n_iov_occasions() == 3

    def test_base_same_base_same_returns_2(self):
        """BASE, SAME, BASE, SAME → 2 occasions (only contiguous SAMEs after first BASE)."""
        specs = [
            OmegaSpec(block_size=1, values=[0.1]),
            OmegaSpec(block_size=1, values=[], same=True),
            OmegaSpec(block_size=1, values=[0.2]),
            OmegaSpec(block_size=1, values=[], same=True),
        ]
        params = _make_params_from_omega_specs(specs)
        # Only 1 SAME immediately after the first BASE → 2 occasions
        assert params.n_iov_occasions() == 2

    def test_base_same_same_base_same_returns_3(self):
        """BASE, SAME, SAME, BASE, SAME → 3 occasions (2 SAMEs after first BASE)."""
        specs = [
            OmegaSpec(block_size=1, values=[0.1]),
            OmegaSpec(block_size=1, values=[], same=True),
            OmegaSpec(block_size=1, values=[], same=True),
            OmegaSpec(block_size=1, values=[0.2]),
            OmegaSpec(block_size=1, values=[], same=True),
        ]
        params = _make_params_from_omega_specs(specs)
        # 2 SAMEs immediately after the first BASE → 3 occasions
        assert params.n_iov_occasions() == 3

    def test_empty_omega_specs_returns_1(self):
        """No specs at all → 1 occasion."""
        params = ParameterSet(
            theta=np.array([1.0]),
            omega=np.eye(1) * 0.1,
            sigma=np.eye(1) * 0.05,
            theta_specs=[ThetaSpec(init=1.0, lower=0.0, upper=10.0)],
            omega_specs=[],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
        )
        assert params.n_iov_occasions() == 1

    def test_two_occasion_iov_numerical(self):
        """
        Construct a ParameterSet with BASE + 1 SAME (2-occasion IOV omega);
        verify n_iov_occasions()==2 and that block structure is correct.
        """
        base_var = 0.09  # variance (SD = 0.3)
        specs = [
            OmegaSpec(block_size=1, values=[base_var]),
            OmegaSpec(block_size=1, values=[], same=True),
        ]
        params = _make_params_from_omega_specs(specs)

        assert params.n_iov_occasions() == 2

        # Omega should be 2x2 block diagonal with both blocks = base_var
        assert params.omega.shape == (2, 2)
        assert params.omega[0, 0] == pytest.approx(base_var)
        assert params.omega[1, 1] == pytest.approx(base_var)
        assert params.omega[0, 1] == pytest.approx(0.0)  # off-block is zero
        assert params.omega[1, 0] == pytest.approx(0.0)

    def test_has_iov_true_when_same_present(self):
        """has_iov() returns True when any SAME spec is present."""
        specs = [
            OmegaSpec(block_size=1, values=[0.1]),
            OmegaSpec(block_size=1, values=[], same=True),
        ]
        params = _make_params_from_omega_specs(specs)
        assert params.has_iov() is True

    def test_has_iov_false_when_no_same(self):
        """has_iov() returns False when no SAME specs."""
        specs = [OmegaSpec(block_size=1, values=[0.1])]
        params = _make_params_from_omega_specs(specs)
        assert params.has_iov() is False
