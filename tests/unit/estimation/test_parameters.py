"""Unit tests for ParameterSet and parameter specifications."""

import math

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec


@pytest.mark.unit
class TestThetaSpec:
    def test_basic(self):
        s = ThetaSpec(init=1.5, lower=0.0, upper=10.0)
        assert s.init == 1.5
        assert s.lower == 0.0
        assert s.upper == 10.0
        assert not s.fixed

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            ThetaSpec(init=1.5, lower=5.0, upper=2.0)  # lower > upper

    def test_init_out_of_bounds(self):
        with pytest.raises(ValueError):
            ThetaSpec(init=15.0, lower=0.0, upper=10.0)

    def test_fixed(self):
        s = ThetaSpec(init=1.0, fixed=True)
        assert s.fixed is True

    def test_unbounded(self):
        s = ThetaSpec(init=0.0)
        assert math.isinf(s.lower)
        assert math.isinf(s.upper)


@pytest.mark.unit
class TestOmegaSpec:
    def test_scalar(self):
        s = OmegaSpec(block_size=1, values=[0.5])
        mat = s.to_matrix()
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(0.5)

    def test_2x2_block(self):
        s = OmegaSpec(block_size=2, values=[0.5, 0.01, 0.3])
        mat = s.to_matrix()
        assert mat.shape == (2, 2)
        assert mat[0, 0] == pytest.approx(0.5)
        assert mat[1, 0] == mat[0, 1] == pytest.approx(0.01)  # symmetric
        assert mat[1, 1] == pytest.approx(0.3)

    def test_wrong_n_values(self):
        with pytest.raises(ValueError):
            OmegaSpec(block_size=2, values=[0.5])  # needs 3 values


@pytest.mark.unit
class TestParameterSet:
    def test_from_specs(self, simple_params):
        assert simple_params.n_theta() == 3
        assert simple_params.n_eta() == 3
        assert simple_params.n_eps() == 1

    def test_theta_values(self, simple_params):
        assert simple_params.theta[0] == pytest.approx(1.5)
        assert simple_params.theta[1] == pytest.approx(0.08)
        assert simple_params.theta[2] == pytest.approx(30.0)

    def test_omega_diagonal(self, simple_params):
        omega = simple_params.omega
        assert omega.shape == (3, 3)
        assert omega[0, 0] == pytest.approx(0.5)
        assert omega[1, 1] == pytest.approx(0.3)
        assert omega[2, 2] == pytest.approx(0.3)
        # Off-diagonal should be zero (diagonal OMEGA)
        assert omega[0, 1] == pytest.approx(0.0)

    def test_to_from_vector_roundtrip(self, simple_params):
        """to_vector → from_vector should recover original parameters."""
        vec = simple_params.to_vector()
        recovered = ParameterSet.from_vector(vec, simple_params)
        np.testing.assert_allclose(recovered.theta, simple_params.theta, rtol=1e-6)
        np.testing.assert_allclose(recovered.omega, simple_params.omega, rtol=1e-5)
        np.testing.assert_allclose(recovered.sigma, simple_params.sigma, rtol=1e-5)

    def test_from_vector_clamps_large_covariance_diagonals_with_block_specs(self):
        params = ParameterSet.from_specs(
            [ThetaSpec(init=1.0, fixed=True)],
            [OmegaSpec(block_size=1, values=[0.1])],
            [SigmaSpec(block_size=1, values=[0.2])],
        )

        recovered = ParameterSet.from_vector(np.array([1e6, 1e6]), params)

        assert np.isfinite(recovered.omega[0, 0])
        assert np.isfinite(recovered.sigma[0, 0])
        assert recovered.omega[0, 0] > 0.0
        assert recovered.sigma[0, 0] > 0.0

    def test_from_vector_clamps_large_covariance_diagonals_without_block_specs(self):
        params = ParameterSet(
            theta=np.array([], dtype=np.float64),
            omega=np.array([[0.1]], dtype=np.float64),
            sigma=np.array([[0.2]], dtype=np.float64),
        )

        recovered = ParameterSet.from_vector(np.array([1e6, 1e6]), params)

        assert np.isfinite(recovered.omega[0, 0])
        assert np.isfinite(recovered.sigma[0, 0])
        assert recovered.omega[0, 0] > 0.0
        assert recovered.sigma[0, 0] > 0.0

    def test_apply_bounds(self, simple_params):
        """apply_bounds should clamp theta to valid range."""
        params = simple_params
        params_bounded = params.apply_bounds()
        for i, spec in enumerate(params_bounded.theta_specs):
            val = params_bounded.theta[i]
            if not math.isinf(spec.lower):
                assert val >= spec.lower
            if not math.isinf(spec.upper):
                assert val <= spec.upper

    def test_omega_pd_after_bounds(self, simple_params):
        """OMEGA should be positive-definite after apply_bounds."""
        from openpkpd.math.matrix import is_pd

        bounded = simple_params.apply_bounds()
        assert is_pd(bounded.omega)

    def test_to_vector_packs_only_block_parameters_in_block_order(self):
        """Vectorization should respect OMEGA/SIGMA block structure, not full-matrix shape."""
        params = ParameterSet.from_specs(
            [ThetaSpec(init=1.0, fixed=True)],
            [
                OmegaSpec(block_size=1, values=[0.25]),
                OmegaSpec(block_size=2, values=[0.16, 0.02, 0.09]),
            ],
            [SigmaSpec(block_size=2, values=[0.04, 0.01, 0.09])],
        )

        vec = params.to_vector()

        omega_block_1 = np.linalg.cholesky(np.array([[0.25]]))
        omega_block_2 = np.linalg.cholesky(np.array([[0.16, 0.02], [0.02, 0.09]]))
        sigma_block = np.linalg.cholesky(np.array([[0.04, 0.01], [0.01, 0.09]]))
        expected = np.array(
            [
                math.log(omega_block_1[0, 0]),
                math.log(omega_block_2[0, 0]),
                omega_block_2[1, 0],
                math.log(omega_block_2[1, 1]),
                math.log(sigma_block[0, 0]),
                sigma_block[1, 0],
                math.log(sigma_block[1, 1]),
            ]
        )

        np.testing.assert_allclose(vec, expected, rtol=1e-10, atol=1e-10)

    def test_fixed_omega_sigma_excluded_from_vector_and_n_free(self):
        """Fixed OMEGA/SIGMA blocks should not contribute optimizer dimensions."""
        params = ParameterSet.from_specs(
            [
                ThetaSpec(init=2.0, lower=0.0, upper=10.0),
                ThetaSpec(init=5.0, fixed=True),
            ],
            [
                OmegaSpec(block_size=1, values=[0.25], fixed=True),
                OmegaSpec(block_size=1, values=[0.09]),
            ],
            [
                SigmaSpec(block_size=1, values=[0.04], fixed=True),
                SigmaSpec(block_size=1, values=[0.01]),
            ],
        )

        vec = params.to_vector()
        assert len(vec) == 3
        assert params.n_free() == 3

        trial = vec + np.array([0.2, 0.3, -0.4])
        recovered = ParameterSet.from_vector(trial, params)

        assert recovered.theta[1] == pytest.approx(params.theta[1])
        assert recovered.omega[0, 0] == pytest.approx(params.omega[0, 0])
        assert recovered.sigma[0, 0] == pytest.approx(params.sigma[0, 0])
        assert recovered.omega[0, 1] == pytest.approx(0.0)
        assert recovered.sigma[0, 1] == pytest.approx(0.0)

    def test_expand_omega_iov_repeats_base_blocks_and_marks_occasions(self):
        """IOV expansion should create repeated block-diagonal OMEGA blocks."""
        params = ParameterSet.from_specs(
            [ThetaSpec(init=1.0)],
            [OmegaSpec(block_size=1, values=[0.04], label="eta_cl")],
            [SigmaSpec(block_size=1, values=[0.01])],
        )

        expanded = params.expand_omega_iov(3)

        np.testing.assert_allclose(np.diag(expanded.omega), [0.04, 0.04, 0.04])
        np.testing.assert_allclose(expanded.omega, np.diag([0.04, 0.04, 0.04]))
        assert [spec.same for spec in expanded.omega_specs] == [False, True, True]
        assert [spec.label for spec in expanded.omega_specs] == [
            "eta_cl_occ1",
            "eta_cl_occ2",
            "eta_cl_occ3",
        ]

    def test_n_iov_occasions_counts_same_specs(self):
        """n_iov_occasions should infer occasions from SAME specs."""
        params = ParameterSet(
            theta=np.array([1.0]),
            omega=np.eye(3),
            sigma=np.eye(1),
            theta_specs=[ThetaSpec(init=1.0)],
            omega_specs=[
                OmegaSpec(block_size=1, values=[0.04]),
                OmegaSpec(block_size=1, values=[], same=True),
                OmegaSpec(block_size=1, values=[], same=True),
            ],
            sigma_specs=[SigmaSpec(block_size=1, values=[0.01])],
        )

        assert params.has_iov() is True
        assert params.n_iov_occasions() == 3


@pytest.mark.unit
@given(
    init=st.floats(min_value=0.01, max_value=100.0),
    lower=st.floats(min_value=0.0, max_value=0.005),
    upper=st.floats(min_value=200.0, max_value=1000.0),
)
@settings(max_examples=50)
def test_theta_spec_roundtrip(init, lower, upper):
    """ThetaSpec with valid bounds should round-trip through to_vector/from_vector."""
    spec = ThetaSpec(init=init, lower=lower, upper=upper)
    omega_spec = OmegaSpec(block_size=1, values=[0.1])
    sigma_spec = SigmaSpec(block_size=1, values=[0.1])
    params = ParameterSet.from_specs([spec], [omega_spec], [sigma_spec])
    vec = params.to_vector()
    recovered = ParameterSet.from_vector(vec, params)
    assert recovered.theta[0] == pytest.approx(init, rel=1e-4)
