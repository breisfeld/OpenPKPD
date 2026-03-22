"""
Unit tests for Markov and Hidden Markov Models.

Tests cover:
  - ContinuousTimeMarkovModel transition matrix: row sums to 1, non-negative.
  - ContinuousTimeHMM: forward algorithm, Viterbi, fit.
  - HMMData and HMMResult dataclasses.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from openpkpd.models.categorical import ContinuousTimeMarkovModel
from openpkpd.models.markov import ContinuousTimeHMM, HMMData, HMMResult


def _two_state_ctmc_transition(a: float, b: float, t: float) -> np.ndarray:
    decay = np.exp(-(a + b) * t)
    pi0 = b / (a + b)
    pi1 = a / (a + b)
    return np.array(
        [
            [pi0 + (1.0 - pi0) * decay, pi1 - pi1 * decay],
            [pi0 - pi0 * decay, pi1 + (1.0 - pi1) * decay],
        ]
    )


def _exact_hmm_params() -> np.ndarray:
    log_rates = np.log(np.array([0.4, 0.2]))
    emission = np.array([[0.9, 0.1], [0.2, 0.8]])
    init = np.array([0.6, 0.4])
    return np.concatenate([log_rates, np.log(emission).ravel(), np.log(init)])


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


# ---------------------------------------------------------------------------
# ContinuousTimeMarkovModel transition matrix (primary tests required by spec)
# ---------------------------------------------------------------------------


class TestCTMMTransitionMatrix:
    """Transition matrix validity tests for ContinuousTimeMarkovModel."""

    def test_rows_sum_to_one_2state(self) -> None:
        """P(t) rows should sum to 1 (valid probability matrix) — 2-state."""
        model = ContinuousTimeMarkovModel(n_states=2)
        # Simple 2-state system: q01=0.1, q10=0.05
        params = np.log(np.array([0.1, 0.05]))  # log-parameterised
        P = model.transition_matrix(t=1.0, params=params)
        assert P.shape == (2, 2)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(P >= 0)

    def test_rows_sum_to_one_3state(self) -> None:
        """P(t) rows should sum to 1 for a 3-state model."""
        model = ContinuousTimeMarkovModel(n_states=3)
        params = np.full(6, -1.5)  # 3*(3-1) = 6 log-rates
        P = model.transition_matrix(t=2.0, params=params)
        assert P.shape == (3, 3)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(P >= 0)

    def test_identity_at_zero_time(self) -> None:
        """P(0) = I regardless of parameters."""
        model = ContinuousTimeMarkovModel(n_states=2)
        params = np.array([-1.0, -2.0])
        P = model.transition_matrix(t=0.0, params=params)
        np.testing.assert_allclose(P, np.eye(2), atol=1e-10)

    def test_approaches_stationary_for_large_t(self) -> None:
        """For large t, rows of P(t) converge to the stationary distribution."""
        model = ContinuousTimeMarkovModel(n_states=2)
        params = np.log(np.array([0.3, 0.7]))  # q01=0.3, q10=0.7
        P_large = model.transition_matrix(t=200.0, params=params)
        # Stationary distribution: pi_0 = q10/(q01+q10), pi_1 = q01/(q01+q10)
        pi_0 = 0.7 / (0.3 + 0.7)  # = 0.7
        pi_1 = 0.3 / (0.3 + 0.7)  # = 0.3
        np.testing.assert_allclose(P_large[0, :], [pi_0, pi_1], atol=1e-4)
        np.testing.assert_allclose(P_large[1, :], [pi_0, pi_1], atol=1e-4)

    def test_concentration_effect(self) -> None:
        """Concentration should modify transition rates when betas are included."""
        model = ContinuousTimeMarkovModel(n_states=2)
        # log_rates: [log(0.1), log(0.05)], betas: [1.0, 0.0]
        params_base = np.log(np.array([0.1, 0.05]))
        params_with_conc = np.concatenate([params_base, [2.0, 0.0]])
        P_no_conc = model.transition_matrix(t=1.0, params=params_with_conc, concentration=0.0)
        P_high_conc = model.transition_matrix(t=1.0, params=params_with_conc, concentration=3.0)
        # Higher concentration with positive beta should increase q_01 -> more 0->1 transitions
        assert P_high_conc[0, 1] > P_no_conc[0, 1]

    def test_two_state_transition_matrix_matches_closed_form(self) -> None:
        """Two-state CTMC transition matrix should match the analytic solution."""
        model = ContinuousTimeMarkovModel(n_states=2)
        a, b, t = 0.4, 0.2, 1.75
        params = np.log(np.array([a, b]))

        P = model.transition_matrix(t=t, params=params)
        expected = _two_state_ctmc_transition(a, b, t)

        np.testing.assert_allclose(P, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# ContinuousTimeHMM
# ---------------------------------------------------------------------------


class TestContinuousTimeHMM:
    """Tests for ContinuousTimeHMM."""

    def _make_simple_data(self, seed: int = 0) -> tuple[list[HMMData], np.ndarray]:
        """Generate a small HMM dataset for testing.

        Returns:
            Tuple of (data, init_params).
        """
        rng = np.random.default_rng(seed)
        n_hidden, n_obs = 2, 2
        model = ContinuousTimeHMM(n_hidden, n_obs)
        n_params = model.n_params()
        init_params = rng.normal(0, 0.5, size=n_params)

        observations = rng.integers(0, n_obs, size=10)
        times = np.sort(rng.uniform(0, 10, size=10))
        data = [HMMData(subject_id=1, observations=observations, times=times)]
        return data, init_params

    def test_forward_algorithm_returns_finite(self) -> None:
        """forward_algorithm should return a finite log-likelihood."""
        data, init_params = self._make_simple_data(0)
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        ll = model.forward_algorithm(data[0], init_params)
        assert np.isfinite(ll)

    def test_forward_algorithm_non_positive(self) -> None:
        """Log-likelihood of observations must be <= 0."""
        data, init_params = self._make_simple_data(1)
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        ll = model.forward_algorithm(data[0], init_params)
        assert ll <= 0.0

    def test_emission_matrix_matches_rowwise_softmax(self) -> None:
        """Emission probabilities should be the row-wise softmax of emission logits."""
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=3)
        logits = np.array([0.2, -0.1, 0.4, -0.3, 0.0, 0.8])

        B = model._emission_matrix(logits)
        expected = np.vstack(
            [
                _softmax(np.array([0.2, -0.1, 0.4])),
                _softmax(np.array([-0.3, 0.0, 0.8])),
            ]
        )

        np.testing.assert_allclose(B, expected, atol=1e-10)

    def test_initial_distribution_matches_softmax(self) -> None:
        """Initial hidden-state probabilities should be the softmax of init logits."""
        model = ContinuousTimeHMM(n_hidden_states=3, n_observed_states=2)
        init_logits = np.array([0.3, -0.4, 1.1])

        pi = model._initial_distribution(init_logits)

        np.testing.assert_allclose(pi, _softmax(init_logits), atol=1e-10)

    def test_forward_algorithm_matches_two_step_closed_form(self) -> None:
        """Forward likelihood should match direct enumeration for a tiny 2-state HMM."""
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        params = _exact_hmm_params()
        data = HMMData(
            subject_id=1,
            observations=np.array([0, 1]),
            times=np.array([0.0, 0.75]),
        )

        P = _two_state_ctmc_transition(0.4, 0.2, 0.75)
        B = np.array([[0.9, 0.1], [0.2, 0.8]])
        pi = np.array([0.6, 0.4])
        expected_prob = np.sum(pi * B[:, 0] * (P @ B[:, 1]))

        ll = model.forward_algorithm(data, params)

        assert ll == pytest.approx(np.log(expected_prob), abs=1e-10)

    def test_forward_algorithm_matches_three_step_enumeration(self) -> None:
        """Forward likelihood should match brute-force enumeration on a 3-observation HMM."""
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        params = _exact_hmm_params()
        data = HMMData(
            subject_id=1,
            observations=np.array([0, 1, 0]),
            times=np.array([0.0, 0.7, 1.4]),
        )

        P = _two_state_ctmc_transition(0.4, 0.2, 0.7)
        B = np.array([[0.9, 0.1], [0.2, 0.8]])
        pi = np.array([0.6, 0.4])
        expected_prob = 0.0
        for path in product(range(2), repeat=3):
            expected_prob += (
                pi[path[0]]
                * B[path[0], data.observations[0]]
                * P[path[0], path[1]]
                * B[path[1], data.observations[1]]
                * P[path[1], path[2]]
                * B[path[2], data.observations[2]]
            )

        ll = model.forward_algorithm(data, params)

        assert ll == pytest.approx(np.log(expected_prob), abs=1e-10)

    def test_viterbi_returns_correct_length(self) -> None:
        """Viterbi sequence should have the same length as observations."""
        data, init_params = self._make_simple_data(2)
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        path = model.viterbi(data[0], init_params)
        assert len(path) == len(data[0].observations)

    def test_viterbi_states_in_valid_range(self) -> None:
        """Viterbi states must be in [0, n_hidden)."""
        data, init_params = self._make_simple_data(3)
        model = ContinuousTimeHMM(n_hidden_states=3, n_observed_states=2)
        # Rebuild init_params for 3-state HMM
        n_params = model.n_params()
        rng = np.random.default_rng(3)
        init_params = rng.normal(0, 0.5, size=n_params)
        path = model.viterbi(data[0], init_params)
        assert all(0 <= s < 3 for s in path)

    def test_viterbi_matches_bruteforce_best_path_for_tiny_example(self) -> None:
        """Viterbi path should match brute-force path maximisation on a tiny HMM."""
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        params = _exact_hmm_params()
        data = HMMData(
            subject_id=1,
            observations=np.array([0, 1, 1]),
            times=np.array([0.0, 0.7, 1.4]),
        )

        P = _two_state_ctmc_transition(0.4, 0.2, 0.7)
        B = np.array([[0.9, 0.1], [0.2, 0.8]])
        pi = np.array([0.6, 0.4])

        best_path = None
        best_prob = -1.0
        for path in product(range(2), repeat=3):
            prob = (
                pi[path[0]]
                * B[path[0], data.observations[0]]
                * P[path[0], path[1]]
                * B[path[1], data.observations[1]]
                * P[path[1], path[2]]
                * B[path[2], data.observations[2]]
            )
            if prob > best_prob:
                best_prob = prob
                best_path = np.array(path, dtype=int)

        decoded = model.viterbi(data, params)

        np.testing.assert_array_equal(decoded, best_path)

    def test_fit_returns_hmm_result(self) -> None:
        """fit() should return a valid HMMResult."""
        data, init_params = self._make_simple_data(4)
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        result = model.fit(data, init_params)
        assert isinstance(result, HMMResult)
        assert isinstance(result.ofv, float)
        assert isinstance(result.converged, bool)
        assert isinstance(result.viterbi_states, dict)
        assert isinstance(result.emission_probs, np.ndarray)
        assert isinstance(result.initial_probs, np.ndarray)
        assert isinstance(result.transition_rates, np.ndarray)

    def test_fit_emission_probs_shape(self) -> None:
        """Emission probability matrix should have shape (n_hidden, n_obs)."""
        data, init_params = self._make_simple_data(5)
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        result = model.fit(data, init_params)
        assert result.emission_probs.shape == (2, 2)

    def test_fit_emission_probs_rows_sum_to_one(self) -> None:
        """Emission probability rows must sum to 1."""
        data, init_params = self._make_simple_data(6)
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        result = model.fit(data, init_params)
        np.testing.assert_allclose(result.emission_probs.sum(axis=1), 1.0, atol=1e-6)

    def test_fit_initial_probs_sum_to_one(self) -> None:
        """Initial hidden-state distribution must sum to 1."""
        data, init_params = self._make_simple_data(7)
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        result = model.fit(data, init_params)
        assert result.initial_probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_fit_viterbi_states_keyed_by_subject(self) -> None:
        """viterbi_states dict should be keyed by subject_id."""
        data, init_params = self._make_simple_data(8)
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        result = model.fit(data, init_params)
        assert 1 in result.viterbi_states
        assert len(result.viterbi_states[1]) == len(data[0].observations)

    def test_fit_ofv_finite(self) -> None:
        """Fitted OFV must be finite."""
        data, init_params = self._make_simple_data(9)
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        result = model.fit(data, init_params)
        assert np.isfinite(result.ofv)

    def test_empty_observations(self) -> None:
        """Empty observation sequence should return LL = 0 and empty Viterbi."""
        model = ContinuousTimeHMM(n_hidden_states=2, n_observed_states=2)
        n_params = model.n_params()
        params = np.zeros(n_params)
        empty_data = HMMData(
            subject_id=99,
            observations=np.array([], dtype=int),
            times=np.array([], dtype=float),
        )
        ll = model.forward_algorithm(empty_data, params)
        assert ll == 0.0
        path = model.viterbi(empty_data, params)
        assert len(path) == 0


# ---------------------------------------------------------------------------
# HMMData and HMMResult dataclasses
# ---------------------------------------------------------------------------


class TestHMMData:
    """Tests for HMMData dataclass."""

    def test_basic_construction(self) -> None:
        """HMMData stores observations and times."""
        data = HMMData(
            subject_id=10,
            observations=np.array([0, 1, 0]),
            times=np.array([0.0, 2.0, 5.0]),
        )
        assert data.subject_id == 10
        assert len(data.observations) == 3
        assert data.concentrations is None

    def test_with_concentrations(self) -> None:
        """HMMData with concentration data."""
        data = HMMData(
            subject_id=1,
            observations=np.array([1, 0]),
            times=np.array([0.0, 3.0]),
            concentrations=np.array([5.0, 2.0]),
        )
        assert len(data.concentrations) == 2


class TestHMMResult:
    """Tests for HMMResult dataclass."""

    def test_construction(self) -> None:
        """HMMResult stores all expected fields."""
        result = HMMResult(
            transition_rates=np.array([-1.0, -2.0]),
            emission_probs=np.array([[0.8, 0.2], [0.3, 0.7]]),
            initial_probs=np.array([0.5, 0.5]),
            ofv=123.4,
            converged=True,
            viterbi_states={1: np.array([0, 1, 0])},
        )
        assert result.converged is True
        assert result.ofv == pytest.approx(123.4)
        assert 1 in result.viterbi_states
