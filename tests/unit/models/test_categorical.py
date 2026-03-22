"""
Unit tests for categorical response models.

Tests cover:
  - ProportionalOddsModel: probability predictions, log-likelihood, fit.
  - DiscreteTimeMarkovModel: transition matrix validity, log-likelihood, fit.
  - ContinuousTimeMarkovModel: transition matrix validity (tested via test_markov.py),
    log-likelihood, fit.
  - CategoricalData dataclass.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from openpkpd.models.categorical import (
    CategoricalData,
    CategoricalResult,
    ContinuousTimeMarkovModel,
    DiscreteTimeMarkovModel,
    ProportionalOddsModel,
)

# ---------------------------------------------------------------------------
# ProportionalOddsModel
# ---------------------------------------------------------------------------


class TestProportionalOddsModel:
    """Tests for ProportionalOddsModel."""

    def test_probs_sum_to_one_reference_subject(self) -> None:
        """Predicted probabilities should sum to 1 (reference covariate = 0)."""
        model = ProportionalOddsModel(n_categories=3)
        # params: 2 raw thresholds + 1 coef
        params = np.array([-1.0, 1.0, 0.5])
        cov = np.array([[0.0]])  # shape (1, 1)
        probs = model.predict_probs(cov, params)
        assert probs.shape == (1, 3)
        assert np.isclose(probs.sum(axis=1), 1.0).all()

    def test_probs_sum_to_one_no_covariates(self) -> None:
        """Probabilities sum to 1 even without covariates."""
        model = ProportionalOddsModel(n_categories=4)
        params = np.array([0.0, 0.0, 0.0])  # 3 raw thresholds, no coef
        cov = np.zeros((5, 0))  # 5 obs, 0 covariates
        probs = model.predict_probs(cov, params)
        assert probs.shape == (5, 4)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_probs_non_negative(self) -> None:
        """All predicted probabilities must be >= 0."""
        model = ProportionalOddsModel(n_categories=3)
        params = np.array([0.5, 1.0, -0.3])
        cov = np.array([[0.0], [1.0], [-1.0]])
        probs = model.predict_probs(cov, params)
        assert (probs >= 0).all()

    def test_predict_probs_match_manual_cumulative_logit_formula(self) -> None:
        """Predicted category probabilities should equal the cumulative-logit formula."""
        model = ProportionalOddsModel(n_categories=3)
        params = np.array([-0.4, np.log(1.5), 0.7])
        cov = np.array([[-1.0], [0.5]])

        probs = model.predict_probs(cov, params)

        alpha = np.array([-0.4, -0.4 + 1.5])
        eta = 0.7 * cov[:, 0]
        cum1 = expit(alpha[0] - eta)
        cum2 = expit(alpha[1] - eta)
        expected = np.column_stack([cum1, cum2 - cum1, 1.0 - cum2])

        np.testing.assert_allclose(probs, expected, atol=1e-10)

    def test_positive_coef_shifts_probability_to_higher_categories(self) -> None:
        """Positive beta increases the linear predictor, shifting probability
        mass toward higher categories (larger logit P(Y <= k))."""
        model = ProportionalOddsModel(n_categories=3)
        # thresholds: alpha_1=0, alpha_2=alpha_1+exp(0)=1; coef=1.0
        params = np.array([0.0, 0.0, 2.0])  # large positive coef
        cov_low = np.array([[0.0]])
        cov_high = np.array([[3.0]])
        probs_low = model.predict_probs(cov_low, params)
        probs_high = model.predict_probs(cov_high, params)
        # Higher covariate value -> higher eta -> lower logit P(Y<=k)
        # -> more probability in higher categories
        # P(Y=2 | high X) > P(Y=2 | low X)
        assert probs_high[0, 2] > probs_low[0, 2]

    def test_log_likelihood_finite(self) -> None:
        """log_likelihood must be finite for valid data and params."""
        model = ProportionalOddsModel(n_categories=3)
        params = np.array([0.0, 0.5, 0.0])
        data = [
            CategoricalData(
                subject_id=1,
                categories=np.array([0, 1, 2, 0, 1]),
                times=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            )
        ]
        ll = model.log_likelihood(data, params)
        assert np.isfinite(ll)

    def test_log_likelihood_is_non_positive(self) -> None:
        """Log-likelihood of probabilities must be <= 0."""
        model = ProportionalOddsModel(n_categories=3)
        params = np.array([0.0, 0.5, 0.0])
        data = [
            CategoricalData(
                subject_id=1,
                categories=np.array([0, 1, 2]),
                times=np.array([0.0, 1.0, 2.0]),
            )
        ]
        ll = model.log_likelihood(data, params)
        assert ll <= 0.0

    def test_log_likelihood_matches_manual_category_probabilities(self) -> None:
        """Total proportional-odds likelihood should equal the exact category-probability sum."""
        model = ProportionalOddsModel(n_categories=3)
        params = np.array([-0.4, np.log(1.5), 0.7])
        covariate = np.array([-1.0, 0.5])
        data = [
            CategoricalData(
                subject_id=1,
                categories=np.array([0, 2]),
                times=np.array([0.0, 1.0]),
                covariates={"x": covariate},
            )
        ]

        alpha = np.array([-0.4, -0.4 + 1.5])
        eta = 0.7 * covariate
        probs = np.column_stack(
            [
                expit(alpha[0] - eta),
                expit(alpha[1] - eta) - expit(alpha[0] - eta),
                1.0 - expit(alpha[1] - eta),
            ]
        )
        expected = float(np.log(probs[0, 0]) + np.log(probs[1, 2]))

        ll = model.log_likelihood(data, params)

        assert ll == pytest.approx(expected, abs=1e-10)

    def test_fit_returns_categorical_result(self) -> None:
        """fit() should return a valid CategoricalResult."""
        rng = np.random.default_rng(5)
        categories = rng.integers(0, 3, size=30)
        data = [
            CategoricalData(
                subject_id=i,
                categories=np.array([categories[i]]),
                times=np.array([0.0]),
            )
            for i in range(30)
        ]
        model = ProportionalOddsModel(n_categories=3)
        result = model.fit(data)
        assert isinstance(result, CategoricalResult)
        assert isinstance(result.thresholds, np.ndarray)
        assert isinstance(result.ofv, float)
        assert isinstance(result.aic, float)
        assert isinstance(result.converged, bool)

    def test_fit_thresholds_ordered(self) -> None:
        """Fitted thresholds must be strictly increasing."""
        rng = np.random.default_rng(99)
        categories = rng.integers(0, 4, size=60)
        data = [
            CategoricalData(
                subject_id=i,
                categories=np.array([categories[i]]),
                times=np.array([0.0]),
            )
            for i in range(60)
        ]
        model = ProportionalOddsModel(n_categories=4)
        result = model.fit(data)
        thresh = result.thresholds
        assert len(thresh) == 3
        assert all(thresh[k] < thresh[k + 1] for k in range(len(thresh) - 1))

    def test_invalid_n_categories(self) -> None:
        """n_categories < 2 should raise ValueError."""
        with pytest.raises(ValueError):
            ProportionalOddsModel(n_categories=1)

    def test_fit_with_covariate(self) -> None:
        """Fitting with a covariate should return non-empty coef."""
        rng = np.random.default_rng(42)
        n = 50
        x = rng.normal(size=n)
        # Higher x -> more likely category 2
        logits = x * 1.5
        probs = np.exp(logits) / (1 + np.exp(logits))
        cats = (probs > 0.5).astype(int)  # binary 0/1 as 2-category
        data = [
            CategoricalData(
                subject_id=i,
                categories=np.array([cats[i]]),
                times=np.array([0.0]),
                covariates={"x": np.array([x[i]])},
            )
            for i in range(n)
        ]
        model = ProportionalOddsModel(n_categories=2)
        result = model.fit(data)
        assert len(result.coef) == 1  # one covariate


# ---------------------------------------------------------------------------
# DiscreteTimeMarkovModel
# ---------------------------------------------------------------------------


class TestDiscreteTimeMarkovModel:
    """Tests for DiscreteTimeMarkovModel."""

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        """T rows must sum to 1."""
        model = DiscreteTimeMarkovModel(n_states=3)
        params = np.zeros(3 * 2)  # 3 states, 2 off-diag per row
        T = model.transition_matrix(params)
        assert T.shape == (3, 3)
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-10)

    def test_transition_matrix_non_negative(self) -> None:
        """All transition probabilities must be >= 0."""
        model = DiscreteTimeMarkovModel(n_states=2)
        params = np.array([1.0, -1.0])  # arbitrary log-odds
        T = model.transition_matrix(params)
        assert (T >= 0).all()

    def test_two_state_transition_symmetry(self) -> None:
        """Zero log-odds -> symmetric T (0.5 for 2 states)."""
        model = DiscreteTimeMarkovModel(n_states=2)
        params = np.zeros(2)  # 2 states, 1 off-diag each
        T = model.transition_matrix(params)
        np.testing.assert_allclose(T, 0.5 * np.ones((2, 2)), atol=1e-10)

    def test_log_likelihood_finite(self) -> None:
        """log_likelihood must be finite for valid data."""
        model = DiscreteTimeMarkovModel(n_states=2)
        params = np.zeros(2)
        data = [
            CategoricalData(
                subject_id=1,
                categories=np.array([0, 1, 0, 1]),
                times=np.array([0.0, 1.0, 2.0, 3.0]),
            )
        ]
        ll = model.log_likelihood(data, params)
        assert np.isfinite(ll)

    def test_log_likelihood_is_non_positive(self) -> None:
        """Log-likelihood of valid transitions should be <= 0."""
        model = DiscreteTimeMarkovModel(n_states=2)
        params = np.zeros(2)
        data = [
            CategoricalData(
                subject_id=1,
                categories=np.array([0, 0, 1]),
                times=np.array([0.0, 1.0, 2.0]),
            )
        ]
        ll = model.log_likelihood(data, params)
        assert ll <= 0.0

    def test_log_likelihood_matches_manual_two_state_sequence_with_concentration(self) -> None:
        """DTMM likelihood should match the hand-computed two-state softmax transitions."""
        model = DiscreteTimeMarkovModel(n_states=2)
        params = np.array([0.4, -0.2, 0.5, -0.3])
        concentrations = np.array([0.0, 1.5, 0.5, 0.0])
        data = [
            CategoricalData(
                subject_id=1,
                categories=np.array([0, 1, 1, 0]),
                times=np.array([0.0, 1.0, 2.0, 3.0]),
                concentrations=concentrations,
            )
        ]

        logit_01 = 0.4 + 0.5 * concentrations[0]
        logit_10_step2 = -0.2 - 0.3 * concentrations[1]
        logit_10_step3 = -0.2 - 0.3 * concentrations[2]
        expected = float(
            np.log(expit(logit_01)) + np.log(expit(-logit_10_step2)) + np.log(expit(logit_10_step3))
        )

        ll = model.log_likelihood(data, params)

        assert ll == pytest.approx(expected, abs=1e-10)

    def test_fit_returns_categorical_result(self) -> None:
        """fit() should return a valid CategoricalResult."""
        rng = np.random.default_rng(33)
        states = rng.integers(0, 2, size=20)
        data = [
            CategoricalData(
                subject_id=1,
                categories=states,
                times=np.arange(20, dtype=float),
            )
        ]
        model = DiscreteTimeMarkovModel(n_states=2)
        result = model.fit(data)
        assert isinstance(result, CategoricalResult)
        assert np.isfinite(result.ofv)

    def test_invalid_n_states(self) -> None:
        """n_states < 2 should raise ValueError."""
        with pytest.raises(ValueError):
            DiscreteTimeMarkovModel(n_states=1)


# ---------------------------------------------------------------------------
# ContinuousTimeMarkovModel
# ---------------------------------------------------------------------------


class TestContinuousTimeMarkovModel:
    """Tests for ContinuousTimeMarkovModel."""

    @staticmethod
    def _two_state_transition(a: float, b: float, t: float) -> np.ndarray:
        decay = np.exp(-(a + b) * t)
        pi_0 = b / (a + b)
        pi_1 = a / (a + b)
        return np.array(
            [
                [pi_0 + (1.0 - pi_0) * decay, pi_1 - pi_1 * decay],
                [pi_0 - pi_0 * decay, pi_1 + (1.0 - pi_1) * decay],
            ]
        )

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        """P(t) rows must sum to 1 for valid probability matrix."""
        model = ContinuousTimeMarkovModel(n_states=2)
        # log-rates for q01 and q10
        params = np.log(np.array([0.1, 0.05]))
        P = model.transition_matrix(t=1.0, params=params)
        assert P.shape == (2, 2)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    def test_transition_matrix_non_negative(self) -> None:
        """All entries in P(t) must be >= 0."""
        model = ContinuousTimeMarkovModel(n_states=3)
        params = np.full(6, -1.0)  # log-rates = -1
        P = model.transition_matrix(t=0.5, params=params)
        assert (P >= 0).all()

    def test_identity_at_t_zero(self) -> None:
        """P(0) must be the identity matrix."""
        model = ContinuousTimeMarkovModel(n_states=3)
        params = np.zeros(6)
        P = model.transition_matrix(t=0.0, params=params)
        np.testing.assert_allclose(P, np.eye(3), atol=1e-10)

    def test_rate_matrix_diagonal_negative(self) -> None:
        """Diagonal of Q must be <= 0 (Q is a valid generator matrix)."""
        model = ContinuousTimeMarkovModel(n_states=2)
        params = np.array([-1.0, -2.0])
        Q = model.rate_matrix(params)
        assert Q[0, 0] <= 0.0
        assert Q[1, 1] <= 0.0

    def test_rate_matrix_rows_sum_to_zero(self) -> None:
        """Q rows must sum to 0 (generator matrix property)."""
        model = ContinuousTimeMarkovModel(n_states=3)
        params = np.array([-1.0, -1.5, -0.8, -1.2, -0.5, -0.9])
        Q = model.rate_matrix(params)
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-10)

    def test_log_likelihood_finite(self) -> None:
        """log_likelihood must be finite for valid data."""
        model = ContinuousTimeMarkovModel(n_states=2)
        params = np.array([-2.0, -2.0])
        data = [
            CategoricalData(
                subject_id=1,
                categories=np.array([0, 1, 0]),
                times=np.array([0.0, 2.0, 5.0]),
            )
        ]
        ll = model.log_likelihood(data, params)
        assert np.isfinite(ll)

    def test_log_likelihood_matches_closed_form_two_state_ctmc(self) -> None:
        """Two-state CTMC log-likelihood should equal the exact sum of log transition probabilities."""
        model = ContinuousTimeMarkovModel(n_states=2)
        a, b = 0.4, 0.2
        params = np.log(np.array([a, b]))
        data = [
            CategoricalData(
                subject_id=1,
                categories=np.array([0, 1, 1, 0]),
                times=np.array([0.0, 0.5, 1.75, 2.5]),
            )
        ]

        P1 = self._two_state_transition(a, b, 0.5)
        P2 = self._two_state_transition(a, b, 1.25)
        P3 = self._two_state_transition(a, b, 0.75)
        expected = np.log(P1[0, 1]) + np.log(P2[1, 1]) + np.log(P3[1, 0])

        ll = model.log_likelihood(data, params)

        assert ll == pytest.approx(expected, abs=1e-10)

    def test_fit_returns_categorical_result(self) -> None:
        """fit() should return a valid CategoricalResult."""
        rng = np.random.default_rng(77)
        states = rng.integers(0, 2, size=15)
        data = [
            CategoricalData(
                subject_id=1,
                categories=states,
                times=np.sort(rng.uniform(0, 20, size=15)),
            )
        ]
        model = ContinuousTimeMarkovModel(n_states=2)
        result = model.fit(data)
        assert isinstance(result, CategoricalResult)
        assert np.isfinite(result.ofv)


# ---------------------------------------------------------------------------
# CategoricalData dataclass
# ---------------------------------------------------------------------------


class TestCategoricalData:
    """Tests for CategoricalData dataclass."""

    def test_basic_construction(self) -> None:
        """CategoricalData stores arrays correctly."""
        data = CategoricalData(
            subject_id=7,
            categories=np.array([0, 1, 2]),
            times=np.array([0.0, 1.0, 2.0]),
        )
        assert data.subject_id == 7
        assert len(data.categories) == 3
        assert data.covariates is None
        assert data.concentrations is None

    def test_with_covariates_and_concentrations(self) -> None:
        """CategoricalData with optional arrays."""
        data = CategoricalData(
            subject_id=2,
            categories=np.array([1, 0]),
            times=np.array([0.0, 5.0]),
            covariates={"age": np.array([45.0, 45.0])},
            concentrations=np.array([0.0, 5.2]),
        )
        assert "age" in data.covariates
        assert len(data.concentrations) == 2
