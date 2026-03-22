"""
Hidden Markov Models (HMM) for pharmacometric analysis.

Models where the true disease state is hidden (latent) but influences
observable outcomes.  Used for:
  - Seizure dynamics with imperfect detection
  - Immunogenicity modelling
  - Disease progression with noisy state assessment

The continuous-time HMM (CTHMM) implemented here assumes:
  1. Hidden states evolve as a continuous-time Markov chain.
  2. At each observation time the emission probability
     P(observation = o | hidden state = s) is captured by an emission matrix.
  3. Parameters are estimated by maximising the observed-data log-likelihood
     using the forward algorithm (gradient-based MLE), and the most likely
     hidden state sequence is decoded via the Viterbi algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.linalg import expm


@dataclass
class HMMData:
    """Data container for hidden Markov models.

    Attributes:
        subject_id: Unique subject identifier.
        observations: Observed (possibly noisy) state labels, one per time
            point (0-based integers).
        times: Observation times aligned with observations.
        concentrations: Optional drug concentrations at each observation time.
    """

    subject_id: int
    observations: np.ndarray
    times: np.ndarray
    concentrations: np.ndarray | None = None


@dataclass
class HMMResult:
    """Result from HMM maximum-likelihood estimation.

    Attributes:
        transition_rates: Fitted Q-matrix elements (off-diagonal log-rates).
        emission_probs: Fitted emission probability matrix of shape
            (n_hidden, n_observed).
        initial_probs: Fitted initial hidden-state distribution of shape
            (n_hidden,).
        ofv: Objective function value (−2 × log-likelihood).
        converged: Whether the optimiser reported convergence.
        viterbi_states: Dict mapping subject_id to most likely hidden state
            sequence (after fitting).
    """

    transition_rates: np.ndarray
    emission_probs: np.ndarray
    initial_probs: np.ndarray
    ofv: float
    converged: bool
    viterbi_states: dict[int, np.ndarray]


class ContinuousTimeHMM:
    """Continuous-time Hidden Markov Model.

    Hidden states evolve according to a continuous-time Markov chain defined
    by a rate matrix (Q-matrix).  At each observation time, an emission
    probability matrix B maps hidden states to observable states.

    The complete parameter vector has three blocks:
      1. Log off-diagonal Q-matrix rates: length n_hidden * (n_hidden - 1)
      2. Unconstrained emission log-odds (softmax rows): length
         n_hidden * n_observed  (the last category is reference)
      3. Unconstrained initial distribution log-odds: length n_hidden
         (softmax to get pi)

    Attributes:
        n_hidden: Number of hidden states.
        n_observed: Number of observable states.
    """

    def __init__(self, n_hidden_states: int, n_observed_states: int) -> None:
        """Initialise continuous-time HMM.

        Args:
            n_hidden_states: Number of latent (hidden) states.
            n_observed_states: Number of observable state labels.
        """
        if n_hidden_states < 2:
            raise ValueError("n_hidden_states must be >= 2.")
        if n_observed_states < 1:
            raise ValueError("n_observed_states must be >= 1.")
        self.n_hidden = n_hidden_states
        self.n_observed = n_observed_states

    # ------------------------------------------------------------------
    # Parameter layout helpers
    # ------------------------------------------------------------------

    def _n_offdiag(self) -> int:
        """Number of off-diagonal Q-matrix entries."""
        K = self.n_hidden
        return K * (K - 1)

    def _n_emission_params(self) -> int:
        """Number of emission parameters (n_hidden * n_observed)."""
        return self.n_hidden * self.n_observed

    def _n_init_params(self) -> int:
        """Number of initial distribution parameters (n_hidden)."""
        return self.n_hidden

    def n_params(self) -> int:
        """Total number of parameters."""
        return self._n_offdiag() + self._n_emission_params() + self._n_init_params()

    def _split_params(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split combined params into (log_rates, emission_logits, init_logits).

        Args:
            params: Combined parameter vector.

        Returns:
            Tuple of (log_rates, emission_logits, init_logits).
        """
        n_od = self._n_offdiag()
        n_em = self._n_emission_params()
        log_rates = params[:n_od]
        emission_logits = params[n_od : n_od + n_em]
        init_logits = params[n_od + n_em : n_od + n_em + self.n_hidden]
        return log_rates, emission_logits, init_logits

    def _rate_matrix(self, log_rates: np.ndarray, concentration: float = 0.0) -> np.ndarray:
        """Construct Q-matrix from log-rates.

        Args:
            log_rates: Log off-diagonal rates, length n_offdiag.
            concentration: Drug concentration (reserved for future extension).

        Returns:
            Q-matrix of shape (n_hidden, n_hidden).
        """
        K = self.n_hidden
        Q = np.zeros((K, K))
        idx = 0
        for i in range(K):
            for j in range(K):
                if i != j:
                    Q[i, j] = np.exp(float(log_rates[idx]))
                    idx += 1
        for i in range(K):
            Q[i, i] = -Q[i, :].sum()
        return Q

    def _transition_matrix(
        self, delta_t: float, log_rates: np.ndarray, concentration: float = 0.0
    ) -> np.ndarray:
        """Compute P(delta_t) = expm(Q * delta_t).

        Args:
            delta_t: Time interval.
            log_rates: Log off-diagonal rates.
            concentration: Drug concentration.

        Returns:
            Row-stochastic transition matrix (n_hidden, n_hidden).
        """
        if delta_t <= 0.0:
            return np.eye(self.n_hidden)
        Q = self._rate_matrix(log_rates, concentration)
        P = expm(Q * delta_t)
        P = np.clip(P, 0.0, 1.0)
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return P / row_sums

    def _emission_matrix(self, emission_logits: np.ndarray) -> np.ndarray:
        """Convert emission logits to a probability matrix via row-wise softmax.

        Args:
            emission_logits: Unconstrained logits, shape (n_hidden * n_observed,).

        Returns:
            Emission probability matrix B of shape (n_hidden, n_observed).
            B[i, o] = P(observe o | hidden state i).
        """
        logits = emission_logits.reshape(self.n_hidden, self.n_observed)
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        B = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return np.clip(B, 1e-300, 1.0)

    def _initial_distribution(self, init_logits: np.ndarray) -> np.ndarray:
        """Convert logits to initial distribution via softmax.

        Args:
            init_logits: Unconstrained logits of length n_hidden.

        Returns:
            Valid probability vector of length n_hidden.
        """
        if len(init_logits) == 0:
            return np.full(self.n_hidden, 1.0 / self.n_hidden)
        shifted = init_logits - init_logits.max()
        exp_l = np.exp(shifted)
        return exp_l / exp_l.sum()

    # ------------------------------------------------------------------
    # Forward algorithm
    # ------------------------------------------------------------------

    def forward_algorithm(self, data: HMMData, params: np.ndarray) -> float:
        """Forward algorithm to compute log P(observations | params).

        Implements the standard forward pass in log-space to avoid underflow.

        Args:
            data: HMMData for one subject.
            params: Combined parameter vector.

        Returns:
            Log-likelihood for this subject's observation sequence.
        """
        log_rates, emission_logits, init_logits = self._split_params(params)
        B = self._emission_matrix(emission_logits)  # (n_hidden, n_observed)
        pi = self._initial_distribution(init_logits)  # (n_hidden,)

        obs = data.observations.astype(int)
        times = np.asarray(data.times, dtype=float)
        T = len(obs)

        if T == 0:
            return 0.0

        # Initialise log-alpha at t=0
        o0 = obs[0]
        o0 = min(max(o0, 0), self.n_observed - 1)
        log_alpha = np.log(pi + 1e-300) + np.log(B[:, o0] + 1e-300)

        for t in range(1, T):
            delta_t = float(times[t] - times[t - 1])
            conc = float(data.concentrations[t - 1]) if data.concentrations is not None else 0.0
            P = self._transition_matrix(delta_t, log_rates, conc)

            # log-sum-exp over predecessor states
            # log_alpha_new[j] = log sum_i exp(log_alpha[i] + log P[i,j]) + log B[j, o_t]
            log_P = np.log(np.maximum(P, 1e-300))
            # shape broadcast: (n_hidden, n_hidden)
            log_alpha_new = np.logaddexp.reduce(log_alpha[:, np.newaxis] + log_P, axis=0)
            o_t = min(max(obs[t], 0), self.n_observed - 1)
            log_alpha_new += np.log(B[:, o_t] + 1e-300)
            log_alpha = log_alpha_new

        # Total log-likelihood = log sum_i alpha_T[i]
        return float(np.logaddexp.reduce(log_alpha))

    # ------------------------------------------------------------------
    # Viterbi algorithm
    # ------------------------------------------------------------------

    def viterbi(self, data: HMMData, params: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most-probable hidden state sequence.

        Uses log-space dynamic programming.

        Args:
            data: HMMData for one subject.
            params: Combined parameter vector.

        Returns:
            Integer array of most likely hidden states, length T.
        """
        log_rates, emission_logits, init_logits = self._split_params(params)
        B = self._emission_matrix(emission_logits)
        pi = self._initial_distribution(init_logits)

        obs = data.observations.astype(int)
        times = np.asarray(data.times, dtype=float)
        T = len(obs)

        if T == 0:
            return np.array([], dtype=int)

        # Initialise
        o0 = min(max(obs[0], 0), self.n_observed - 1)
        log_delta = np.log(pi + 1e-300) + np.log(B[:, o0] + 1e-300)
        psi = np.zeros((T, self.n_hidden), dtype=int)

        for t in range(1, T):
            delta_t = float(times[t] - times[t - 1])
            conc = float(data.concentrations[t - 1]) if data.concentrations is not None else 0.0
            P = self._transition_matrix(delta_t, log_rates, conc)
            log_P = np.log(np.maximum(P, 1e-300))

            # For each state j, find most likely predecessor
            scores = log_delta[:, np.newaxis] + log_P  # (n_hidden, n_hidden)
            psi[t, :] = np.argmax(scores, axis=0)
            log_delta_new = scores[psi[t, :], np.arange(self.n_hidden)]

            o_t = min(max(obs[t], 0), self.n_observed - 1)
            log_delta = log_delta_new + np.log(B[:, o_t] + 1e-300)

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[T - 1] = int(np.argmax(log_delta))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data: list[HMMData], init_params: np.ndarray) -> HMMResult:
        """Fit CTHMM by maximising the observed-data log-likelihood.

        The observed-data log-likelihood (summed over subjects via the
        forward algorithm) is maximised directly using L-BFGS-B.  This is
        equivalent to the limit of Baum-Welch in that it finds the same MLE,
        and is often faster for moderate-sized problems.

        Args:
            data: Per-subject HMMData records.
            init_params: Initial parameter vector of length :meth:`n_params`.

        Returns:
            HMMResult with fitted parameters, Viterbi states, and diagnostics.
        """

        def neg_ll(params: np.ndarray) -> float:
            try:
                ll = sum(self.forward_algorithm(d, params) for d in data)
                val = -2.0 * ll
                return val if np.isfinite(val) else 1e12
            except Exception:
                return 1e12

        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        ofv = float(result.fun)

        log_rates_hat, emission_logits_hat, init_logits_hat = self._split_params(params_hat)
        B_hat = self._emission_matrix(emission_logits_hat)
        pi_hat = self._initial_distribution(init_logits_hat)

        # Decode Viterbi states for each subject
        viterbi_states: dict[int, np.ndarray] = {}
        for d in data:
            viterbi_states[d.subject_id] = self.viterbi(d, params_hat)

        return HMMResult(
            transition_rates=log_rates_hat,
            emission_probs=B_hat,
            initial_probs=pi_hat,
            ofv=ofv,
            converged=bool(result.success),
            viterbi_states=viterbi_states,
        )
