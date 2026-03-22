"""
Unit tests for PriorSpec, PriorAugmentedModel, and make_theta_prior.

Tests cover:
  - Penalty = 0 at the prior mean
  - Penalty > 0 when theta differs from prior mean
  - Correct quadratic formula
  - OMEGA penalty
  - PriorAugmentedModel attribute delegation
  - make_theta_prior convenience constructor
  - Validation errors
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.prior.prior import (
    PriorAugmentedModel,
    PriorSpec,
    _lower_triangle_vec,
    make_theta_prior,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def simple_prior() -> PriorSpec:
    """2-THETA diagonal prior."""
    return PriorSpec(
        theta_prior=np.array([1.0, 2.0]),
        theta_prior_cov=np.eye(2),
    )


@pytest.fixture()
def tight_prior() -> PriorSpec:
    """1-THETA tight prior with small variance."""
    return PriorSpec(
        theta_prior=np.array([1.0]),
        theta_prior_cov=np.array([[0.1]]),
    )


# ── PriorSpec.penalty() ───────────────────────────────────────────────────────


class TestPriorPenalty:
    """Tests for PriorSpec.penalty()."""

    def test_penalty_zero_at_prior_mean(self, simple_prior: PriorSpec) -> None:
        """Penalty should be exactly 0 when theta equals prior mean."""
        penalty = simple_prior.penalty(np.array([1.0, 2.0]))
        assert penalty == pytest.approx(0.0)

    def test_penalty_positive_when_different(self, simple_prior: PriorSpec) -> None:
        """Penalty should be positive when theta differs from prior mean."""
        penalty = simple_prior.penalty(np.array([1.5, 2.5]))
        assert penalty > 0.0

    def test_penalty_tight_prior_is_large(self, tight_prior: PriorSpec) -> None:
        """Small variance → large penalty for the same deviation."""
        pen_tight = tight_prior.penalty(np.array([2.0]))
        prior_wide = PriorSpec(
            theta_prior=np.array([1.0]),
            theta_prior_cov=np.array([[10.0]]),
        )
        pen_wide = prior_wide.penalty(np.array([2.0]))
        assert pen_tight > pen_wide

    def test_penalty_one_theta_exact_formula(self, tight_prior: PriorSpec) -> None:
        """Verify exact formula: (theta - mu)^T * Sigma^{-1} * (theta - mu)."""
        theta = np.array([2.0])
        mu = tight_prior.theta_prior  # [1.0]
        cov = tight_prior.theta_prior_cov  # [[0.1]]
        prec = np.linalg.inv(cov)
        delta = theta - mu
        expected = float(delta @ prec @ delta)
        assert tight_prior.penalty(theta) == pytest.approx(expected)

    def test_penalty_two_theta_exact_formula(self, simple_prior: PriorSpec) -> None:
        """Identity covariance: penalty = sum of squared deviations."""
        theta = np.array([2.0, 4.0])
        mu = simple_prior.theta_prior
        expected = float(np.sum((theta - mu) ** 2))
        assert simple_prior.penalty(theta) == pytest.approx(expected)

    def test_penalty_symmetric_deviation(self, simple_prior: PriorSpec) -> None:
        """Penalty is the same for +Δ and -Δ (quadratic form)."""
        pen_pos = simple_prior.penalty(np.array([1.5, 2.5]))
        pen_neg = simple_prior.penalty(np.array([0.5, 1.5]))
        assert pen_pos == pytest.approx(pen_neg)

    def test_penalty_with_correlated_covariance(self) -> None:
        """Correlated covariance correctly propagates off-diagonal terms."""
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        prior = PriorSpec(
            theta_prior=np.array([0.0, 0.0]),
            theta_prior_cov=cov,
        )
        theta = np.array([1.0, 0.0])
        pen = prior.penalty(theta)
        prec = np.linalg.inv(cov)
        expected = float(theta @ prec @ theta)
        assert pen == pytest.approx(expected)

    def test_penalty_non_negative(self, simple_prior: PriorSpec) -> None:
        """Penalty must always be >= 0 (it is a quadratic form)."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            theta = rng.normal(size=2)
            assert simple_prior.penalty(theta) >= 0.0


# ── PriorSpec with OMEGA prior ────────────────────────────────────────────────


class TestOmegaPrior:
    """Tests for PriorSpec with omega_prior."""

    @pytest.fixture()
    def omega_prior(self) -> PriorSpec:
        return PriorSpec(
            theta_prior=np.array([1.0]),
            theta_prior_cov=np.eye(1),
            omega_prior=np.array([0.1]),  # prior for OMEGA(1,1)
            omega_prior_cov=np.array([[0.01]]),
        )

    def test_penalty_zero_at_prior_omega(self, omega_prior: PriorSpec) -> None:
        """When theta = prior and omega = prior, total penalty = 0."""
        pen = omega_prior.penalty(
            theta=np.array([1.0]),
            omega=np.array([[0.1]]),
        )
        assert pen == pytest.approx(0.0)

    def test_penalty_positive_omega_deviation(self, omega_prior: PriorSpec) -> None:
        """Deviating from omega prior adds to penalty."""
        pen = omega_prior.penalty(
            theta=np.array([1.0]),  # at prior → theta penalty = 0
            omega=np.array([[0.3]]),  # deviates from 0.1
        )
        assert pen > 0.0

    def test_penalty_omits_omega_when_none(self) -> None:
        """Without omega prior, passing omega is harmless."""
        prior = PriorSpec(
            theta_prior=np.array([1.0]),
            theta_prior_cov=np.eye(1),
        )
        pen = prior.penalty(np.array([1.0]), omega=np.eye(2))
        assert pen == pytest.approx(0.0)

    def test_omega_prior_cov_required_with_omega_prior(self) -> None:
        """Providing omega_prior without omega_prior_cov raises ValueError."""
        with pytest.raises(ValueError, match="omega_prior_cov"):
            PriorSpec(
                theta_prior=np.array([1.0]),
                theta_prior_cov=np.eye(1),
                omega_prior=np.array([0.1]),
            )

    def test_omega_prior_required_with_omega_prior_cov(self) -> None:
        """Providing omega_prior_cov without omega_prior raises ValueError."""
        with pytest.raises(ValueError, match="omega_prior"):
            PriorSpec(
                theta_prior=np.array([1.0]),
                theta_prior_cov=np.eye(1),
                omega_prior_cov=np.array([[0.01]]),
            )

    def test_penalty_multi_dimensional_omega_matches_exact_quadratic_form(self) -> None:
        """OMEGA penalty should use NONMEM lower-triangle column-major ordering."""
        prior = PriorSpec(
            theta_prior=np.array([1.0]),
            theta_prior_cov=np.eye(1),
            omega_prior=np.array([0.10, 0.02, 0.30]),
            omega_prior_cov=np.diag([0.5, 0.25, 0.75]),
        )
        omega = np.array([[0.20, 0.05], [0.05, 0.40]])

        omega_vec = _lower_triangle_vec(omega)
        delta = omega_vec - prior.omega_prior
        expected = float(delta @ np.linalg.inv(prior.omega_prior_cov) @ delta)

        assert prior.penalty(np.array([1.0]), omega=omega) == pytest.approx(expected)


# ── log_prior ─────────────────────────────────────────────────────────────────


class TestLogPrior:
    """Tests for PriorSpec.log_prior()."""

    def test_log_prior_at_mean_is_zero(self, simple_prior: PriorSpec) -> None:
        lp = simple_prior.log_prior(np.array([1.0, 2.0]))
        assert lp == pytest.approx(0.0)

    def test_log_prior_is_negative_elsewhere(self, simple_prior: PriorSpec) -> None:
        lp = simple_prior.log_prior(np.array([0.0, 0.0]))
        assert lp < 0.0

    def test_log_prior_equals_negative_half_penalty(self, simple_prior: PriorSpec) -> None:
        theta = np.array([1.5, 3.0])
        pen = simple_prior.penalty(theta)
        lp = simple_prior.log_prior(theta)
        assert lp == pytest.approx(-0.5 * pen)


# ── PriorSpec validation ──────────────────────────────────────────────────────


class TestPriorSpecValidation:
    """Tests for PriorSpec constructor validation."""

    def test_empty_prior_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            PriorSpec()

    def test_omega_only_prior_is_allowed(self) -> None:
        prior = PriorSpec(
            omega_prior=np.array([0.2]),
            omega_prior_cov=np.array([[0.25]]),
        )
        assert prior.penalty(np.array([123.0]), omega=np.array([[0.2]])) == pytest.approx(0.0)

    def test_cov_shape_mismatch_raises(self) -> None:
        """Covariance shape must match theta_prior length."""
        with pytest.raises(ValueError, match="theta_prior_cov"):
            PriorSpec(
                theta_prior=np.array([1.0, 2.0]),
                theta_prior_cov=np.eye(3),  # wrong size
            )

    def test_singular_covariance_raises_on_precision(self) -> None:
        """Singular covariance raises ValueError on first precision access."""
        prior = PriorSpec(
            theta_prior=np.array([1.0, 2.0]),
            theta_prior_cov=np.zeros((2, 2)),  # singular
        )
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            _ = prior.theta_precision

    def test_repr(self, simple_prior: PriorSpec) -> None:
        r = repr(simple_prior)
        assert "PriorSpec" in r
        assert "n_theta=2" in r


# ── PriorAugmentedModel ───────────────────────────────────────────────────────


class TestPriorAugmentedModel:
    """Tests for PriorAugmentedModel."""

    class _FakePopModel:
        """Minimal mock PopulationModel."""

        def __init__(self) -> None:
            self.my_attr = "hello"
            self._params = None

        def ofv_fo(self, params: object) -> float:
            return 50.0

        def ofv_foce(self, params: object, eta_hat: object) -> float:
            return 48.0

        def subject_ids(self) -> list[int]:
            return [1, 2, 3]

    def _make_aug(self) -> PriorAugmentedModel:
        pop = self._FakePopModel()
        prior = PriorSpec(
            theta_prior=np.array([1.0]),
            theta_prior_cov=np.eye(1),
        )
        return PriorAugmentedModel(pop, prior)

    def test_ofv_at_prior_equals_data_ofv(self) -> None:
        aug = self._make_aug()

        class _FakeParams:
            theta = np.array([1.0])
            omega = np.eye(1) * 0.3

        pen = aug.prior.penalty(_FakeParams.theta, _FakeParams.omega)
        ofv = aug.ofv(_FakeParams)
        assert ofv == pytest.approx(50.0 + pen)

    def test_ofv_exceeds_data_ofv_when_deviated(self) -> None:
        aug = self._make_aug()

        class _FakeParams:
            theta = np.array([5.0])  # far from prior mean 1.0
            omega = np.eye(1) * 0.3

        ofv = aug.ofv(_FakeParams)
        assert ofv > 50.0

    def test_attribute_delegation(self) -> None:
        aug = self._make_aug()
        assert aug.my_attr == "hello"

    def test_subject_ids_delegated(self) -> None:
        aug = self._make_aug()
        assert aug.subject_ids() == [1, 2, 3]

    def test_ofv_fo_delegates_with_penalty(self) -> None:
        aug = self._make_aug()

        class _FakeParams:
            theta = np.array([1.0])
            omega = np.eye(1) * 0.3

        assert aug.ofv_fo(_FakeParams) == pytest.approx(50.0)

    def test_ofv_foce_adds_penalty(self) -> None:
        aug = self._make_aug()

        class _FakeParams:
            theta = np.array([1.0])
            omega = np.eye(1) * 0.3

        ofv = aug.ofv_foce(_FakeParams, eta_hat={})
        assert ofv == pytest.approx(48.0)

    def test_ofv_foce_equals_underlying_ofv_plus_exact_penalty(self) -> None:
        pop = self._FakePopModel()
        prior = PriorSpec(
            theta_prior=np.array([1.0]),
            theta_prior_cov=np.array([[0.5]]),
            omega_prior=np.array([0.2]),
            omega_prior_cov=np.array([[0.25]]),
        )
        aug = PriorAugmentedModel(pop, prior)

        class _FakeParams:
            theta = np.array([1.5])
            omega = np.array([[0.4]])

        penalty = prior.penalty(_FakeParams.theta, _FakeParams.omega)
        assert aug.ofv_foce(_FakeParams, eta_hat={}) == pytest.approx(48.0 + penalty)

    def test_repr_contains_model_info(self) -> None:
        aug = self._make_aug()
        r = repr(aug)
        assert "PriorAugmentedModel" in r


# ── make_theta_prior ──────────────────────────────────────────────────────────


class TestMakeThetaPrior:
    """Tests for make_theta_prior convenience constructor."""

    def test_scalar_cv(self) -> None:
        prior = make_theta_prior([1.0, 2.0], theta_cv=0.3)
        assert prior.theta_prior.tolist() == pytest.approx([1.0, 2.0])
        expected_cov = np.diag([(1.0 * 0.3) ** 2, (2.0 * 0.3) ** 2])
        np.testing.assert_allclose(prior.theta_prior_cov, expected_cov)

    def test_vector_cv(self) -> None:
        prior = make_theta_prior([1.0, 2.0], theta_cv=[0.2, 0.4])
        expected_cov = np.diag([(1.0 * 0.2) ** 2, (2.0 * 0.4) ** 2])
        np.testing.assert_allclose(prior.theta_prior_cov, expected_cov)

    def test_returns_prior_spec(self) -> None:
        prior = make_theta_prior([1.0], theta_cv=0.3)
        assert isinstance(prior, PriorSpec)

    def test_cv_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="theta_cv length"):
            make_theta_prior([1.0, 2.0], theta_cv=[0.3])  # wrong length

    def test_penalty_at_mean_is_zero(self) -> None:
        prior = make_theta_prior([1.5, 0.08, 30.0], theta_cv=0.3)
        pen = prior.penalty(prior.theta_prior)
        assert pen == pytest.approx(0.0)


# ── _lower_triangle_vec ───────────────────────────────────────────────────────


class TestLowerTriangleVec:
    """Tests for _lower_triangle_vec helper."""

    def test_2x2_symmetric_matrix(self) -> None:
        mat = np.array([[1.0, 2.0], [2.0, 3.0]])
        vec = _lower_triangle_vec(mat)
        # column-major lower-triangle: (0,0), (1,0), (1,1)
        assert vec.tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_1x1_matrix(self) -> None:
        mat = np.array([[5.0]])
        vec = _lower_triangle_vec(mat)
        assert vec.tolist() == pytest.approx([5.0])

    def test_output_length(self) -> None:
        n = 3
        mat = np.eye(n)
        vec = _lower_triangle_vec(mat)
        assert len(vec) == n * (n + 1) // 2


# ── Spec-mandated tests ────────────────────────────────────────────────────────


def test_prior_penalty_at_prior_mean() -> None:
    """Spec: penalty should be zero when theta equals prior mean."""
    prior = PriorSpec(
        theta_prior=np.array([1.0, 2.0]),
        theta_prior_cov=np.eye(2),
    )
    penalty = prior.penalty(np.array([1.0, 2.0]))
    assert penalty == pytest.approx(0.0)


def test_prior_penalty_positive() -> None:
    """Spec: penalty should be positive when theta differs from prior mean."""
    prior = PriorSpec(
        theta_prior=np.array([1.0]),
        theta_prior_cov=np.array([[0.1]]),
    )
    penalty = prior.penalty(np.array([2.0]))
    assert penalty > 0.0
