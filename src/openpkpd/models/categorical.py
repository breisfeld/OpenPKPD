"""
Categorical response models for pharmacometric analysis.

Implements:
  - ProportionalOddsModel: ordered categorical data via cumulative logit links.
  - DiscreteTimeMarkovModel: state transitions at discrete time steps.
  - ContinuousTimeMarkovModel: state transitions via matrix exponential of Q-matrix.

All models support optional drug-concentration dependence and are fitted by
maximum-likelihood estimation using scipy.optimize.minimize.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special
from scipy.linalg import expm


@dataclass
class CategoricalData:
    """Data container for categorical and Markov models.

    Attributes:
        subject_id: Unique subject identifier.
        categories: Integer category/state labels (0-based), one per time point.
        times: Observation times aligned with categories.
        covariates: Optional dict of covariate arrays (one value per observation).
        concentrations: Optional drug concentration at each observation time
            (used by concentration-dependent Markov models).
    """

    subject_id: int
    categories: np.ndarray
    times: np.ndarray
    covariates: dict[str, np.ndarray] | None = None
    concentrations: np.ndarray | None = None


@dataclass
class CategoricalResult:
    """Result from categorical model maximum-likelihood estimation.

    Attributes:
        thresholds: Fitted threshold (cut-point) vector for proportional odds;
            empty array for Markov models.
        coef: Fitted covariate coefficients.
        ofv: Objective function value (−2 × log-likelihood).
        converged: Whether the optimiser reported convergence.
        aic: Akaike Information Criterion.
    """

    thresholds: np.ndarray
    coef: np.ndarray
    ofv: float
    converged: bool
    aic: float


class ProportionalOddsModel:
    """Ordered categorical proportional odds (cumulative logit) model.

    Logit P(Y <= k | X) = alpha_k - beta^T X
    where alpha_1 < alpha_2 < ... < alpha_{K-1} are ordered thresholds.

    P(Y = k | X) = P(Y <= k | X) - P(Y <= k-1 | X)

    Parameter layout in ``params``:
        params[0 : K-1]  = thresholds [alpha_1, ..., alpha_{K-1}]
                           (stored as unconstrained deltas; see below)
        params[K-1 : ]   = covariate coefficients [beta_1, ..., beta_p]

    To enforce alpha_1 < ... < alpha_{K-1} the thresholds are internally
    represented as:
        alpha_1 = raw[0]
        alpha_k = alpha_{k-1} + exp(raw[k-1])   for k >= 2

    Attributes:
        n_categories: Total number of ordered categories K.
    """

    def __init__(self, n_categories: int) -> None:
        """Initialise proportional odds model.

        Args:
            n_categories: Total number of ordered categories K.
        """
        if n_categories < 2:
            raise ValueError("n_categories must be >= 2.")
        self.n_categories = n_categories
        self._thresholds: np.ndarray | None = None
        self._coef: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_thresholds(self, raw: np.ndarray) -> np.ndarray:
        """Convert unconstrained raw params to ordered thresholds.

        Args:
            raw: Raw threshold params of length K-1.

        Returns:
            Ordered thresholds alpha, shape (K-1,).
        """
        n = len(raw)
        alpha = np.empty(n)
        alpha[0] = raw[0]
        for k in range(1, n):
            alpha[k] = alpha[k - 1] + np.exp(raw[k])
        return alpha

    def _split_params(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split combined params into raw thresholds and coef.

        Args:
            params: Combined parameter vector.

        Returns:
            Tuple of (raw_thresholds, coef).
        """
        n_thresh = self.n_categories - 1
        raw_thresh = params[:n_thresh]
        coef = params[n_thresh:]
        return raw_thresh, coef

    def _linear_predictor(self, covariates: np.ndarray, coef: np.ndarray) -> np.ndarray:
        """Compute beta^T X.

        Args:
            covariates: Shape (n_obs, n_cov).
            coef: Shape (n_cov,).

        Returns:
            Linear predictor, shape (n_obs,).
        """
        if coef.size == 0 or covariates.shape[1] == 0:
            return np.zeros(covariates.shape[0])
        return covariates @ coef

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_probs(
        self,
        covariates: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """Predict category probabilities for each observation.

        Args:
            covariates: Covariate matrix, shape (n_obs, n_cov).  Use an
                all-zeros matrix of shape (n_obs, 0) when there are no
                covariates.
            params: Combined parameter vector (raw thresholds + coef).

        Returns:
            Probability matrix of shape (n_obs, n_categories).  Each row
            sums to 1.
        """
        n_obs = covariates.shape[0]
        raw_thresh, coef = self._split_params(params)
        alpha = self._decode_thresholds(raw_thresh)  # (K-1,)
        eta = self._linear_predictor(covariates, coef)  # (n_obs,)

        # Cumulative probabilities P(Y <= k)
        # Shape: (n_obs, K-1)
        cum_logit = alpha[np.newaxis, :] - eta[:, np.newaxis]  # (n_obs, K-1)
        cum_prob = special.expit(cum_logit)  # sigmoid

        # Append 0 at left and 1 at right, then diff to get P(Y=k)
        zeros = np.zeros((n_obs, 1))
        ones = np.ones((n_obs, 1))
        cum_prob_full = np.hstack([zeros, cum_prob, ones])  # (n_obs, K+1)
        probs = np.diff(cum_prob_full, axis=1)  # (n_obs, K)

        # Clip for numerical safety
        probs = np.clip(probs, 1e-300, 1.0)
        # Renormalise rows
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def log_likelihood(self, data: list[CategoricalData], params: np.ndarray) -> float:
        """Compute total log-likelihood.

        Args:
            data: Per-subject CategoricalData records.
            params: Combined parameter vector.

        Returns:
            Total log-likelihood.
        """
        ll = 0.0
        for subj in data:
            n_obs = len(subj.categories)
            # Build covariate matrix
            if subj.covariates is not None:
                keys = sorted(subj.covariates.keys())
                X = np.column_stack([subj.covariates[k] for k in keys]).reshape(n_obs, -1)
            else:
                X = np.zeros((n_obs, 0))

            probs = self.predict_probs(X, params)  # (n_obs, K)
            for i, cat in enumerate(subj.categories):
                cat_idx = int(cat)
                if 0 <= cat_idx < self.n_categories:
                    ll += np.log(probs[i, cat_idx])
                # Out-of-range categories contribute 0 (ignored)
        return ll if np.isfinite(ll) else -1e300

    def fit(
        self,
        data: list[CategoricalData],
        init_params: np.ndarray | None = None,
    ) -> CategoricalResult:
        """Fit proportional odds model via MLE.

        The threshold ordering constraint is handled internally via the
        exponential reparameterisation.

        Args:
            data: Per-subject CategoricalData records.
            init_params: Initial parameter vector.  If ``None``, a sensible
                default is used (equally-spaced thresholds, zero coefs).

        Returns:
            CategoricalResult with fitted parameters.
        """
        n_thresh = self.n_categories - 1

        # Infer number of covariates from data
        n_cov = 0
        for subj in data:
            if subj.covariates is not None:
                n_cov = len(subj.covariates)
                break

        if init_params is None:
            # Equally-spaced thresholds in raw space, zero coef
            raw_thresh0 = np.zeros(n_thresh)
            raw_thresh0[0] = 0.0  # alpha_1 = 0
            if n_thresh > 1:
                raw_thresh0[1:] = 0.0  # deltas = exp(0) = 1 => alpha increments by 1
            init_params = np.concatenate([raw_thresh0, np.zeros(n_cov)])

        def neg_ll(params: np.ndarray) -> float:
            try:
                ll = self.log_likelihood(data, params)
                return -2.0 * ll if np.isfinite(ll) else 1e12
            except Exception:
                return 1e12

        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        raw_thresh_hat, coef_hat = self._split_params(params_hat)
        alpha_hat = self._decode_thresholds(raw_thresh_hat)
        self._thresholds = alpha_hat
        self._coef = coef_hat
        ofv = float(result.fun)
        aic = ofv + 2.0 * len(params_hat)

        return CategoricalResult(
            thresholds=alpha_hat,
            coef=coef_hat,
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
        )


class DiscreteTimeMarkovModel:
    """Discrete-time Markov model (DTMM) for state-transition data.

    Models transitions between disease states (e.g. 0=response, 1=stable,
    2=progression) observed at discrete time points.  The transition
    probability matrix T[i, j] = P(state_{t+1} = j | state_t = i) is
    constrained to have rows summing to 1.

    Parameterisation of T:
        For each row i, the log-odds relative to staying in state i are
        parameterised, and a softmax is applied to obtain valid probabilities.

    Concentration-dependence:
        An optional linear concentration effect on the log-odds is supported.
        Each off-diagonal element (i, j) with i != j has an additive term
        ``gamma_ij * concentration`` in the log-odds.

    Parameter layout (params vector):
        n_states * (n_states - 1) unconstrained log-odds for off-diagonal
        transitions (row-by-row, skipping diagonal).
        Optionally followed by n_states * (n_states - 1) concentration slopes.

    Attributes:
        n_states: Number of Markov states.
    """

    def __init__(self, n_states: int) -> None:
        """Initialise discrete-time Markov model.

        Args:
            n_states: Number of states.
        """
        if n_states < 2:
            raise ValueError("n_states must be >= 2.")
        self.n_states = n_states

    def _n_offdiag(self) -> int:
        """Number of off-diagonal entries per row (= n_states - 1)."""
        return self.n_states - 1

    def transition_matrix(
        self,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> np.ndarray:
        """Compute n_states × n_states transition probability matrix.

        Each row of the transition matrix is obtained by applying a softmax
        over [0, log-odds_i1, log-odds_i2, ...] where 0 is the reference
        log-odds for staying in state i (diagonal element).

        Args:
            params: Parameter vector.  First ``n_states * (n_states-1)``
                elements are baseline log-odds; the next block (if present)
                are concentration slopes.
            concentration: Drug concentration.

        Returns:
            Valid row-stochastic matrix of shape (n_states, n_states).
        """
        K = self.n_states
        n_od = self._n_offdiag()
        n_base = K * n_od

        base_logodds = params[:n_base].reshape(K, n_od)

        # Concentration slopes (optional)
        if len(params) > n_base:
            conc_slopes = params[n_base : n_base + n_od * K].reshape(K, n_od)
        else:
            conc_slopes = np.zeros((K, n_od))

        T = np.zeros((K, K))
        for i in range(K):
            logits = np.zeros(K)
            od_idx = 0
            for j in range(K):
                if j != i:
                    logits[j] = base_logodds[i, od_idx] + conc_slopes[i, od_idx] * concentration
                    od_idx += 1
                # logits[i] = 0 (reference: staying in state i)
            # Softmax to get probabilities
            logits -= logits.max()
            exp_logits = np.exp(logits)
            T[i, :] = exp_logits / exp_logits.sum()
        return T

    def log_likelihood(self, data: list[CategoricalData], params: np.ndarray) -> float:
        """Compute total log-likelihood from observed transition sequences.

        For each consecutive pair (state_t, state_{t+1}) the contribution is
        log T[state_t, state_{t+1}].

        Args:
            data: Per-subject CategoricalData; uses ``categories`` and
                optionally ``concentrations``.
            params: Parameter vector.

        Returns:
            Total log-likelihood.
        """
        ll = 0.0
        for subj in data:
            states = subj.categories.astype(int)
            n = len(states)
            for t in range(n - 1):
                conc = float(subj.concentrations[t]) if subj.concentrations is not None else 0.0
                T = self.transition_matrix(params, conc)
                s_from = states[t]
                s_to = states[t + 1]
                if 0 <= s_from < self.n_states and 0 <= s_to < self.n_states:
                    ll += np.log(max(T[s_from, s_to], 1e-300))
        return ll if np.isfinite(ll) else -1e300

    def fit(
        self,
        data: list[CategoricalData],
        init_params: np.ndarray | None = None,
    ) -> CategoricalResult:
        """Fit discrete-time Markov model via MLE.

        Args:
            data: Per-subject CategoricalData records.
            init_params: Initial log-odds parameter vector.  Defaults to
                zeros (uniform transitions).

        Returns:
            CategoricalResult.
        """
        K = self.n_states
        n_od = self._n_offdiag()
        n_params = K * n_od

        if init_params is None:
            init_params = np.zeros(n_params)

        def neg_ll(params: np.ndarray) -> float:
            try:
                ll = self.log_likelihood(data, params)
                return -2.0 * ll if np.isfinite(ll) else 1e12
            except Exception:
                return 1e12

        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        ofv = float(result.fun)
        aic = ofv + 2.0 * len(params_hat)

        # Expose as thresholds=transition params, coef=empty (no separate coefs)
        return CategoricalResult(
            thresholds=params_hat,
            coef=np.array([]),
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
        )


class ContinuousTimeMarkovModel:
    """Continuous-time Markov model (CTMM) for state-transition data.

    Uses an instantaneous rate matrix (Q-matrix) where:
      - Off-diagonal elements q_ij >= 0 are transition rates from state i to j.
      - Diagonal elements q_ii = -sum_{j != i} q_ij.

    Transition probabilities over interval [t_1, t_2]:
        P(delta_t) = expm(Q * delta_t)

    Parameterisation:
        Off-diagonal rates are log-parameterised to enforce positivity.
        params layout (row-by-row, skipping diagonal):
            params[0 : n_offdiag] = log(q_01, q_02, ..., q_{K-1,K-2})
        Optional concentration slopes follow:
            params[n_offdiag : 2*n_offdiag] = beta_01, beta_02, ...
        where q_ij(C) = q_ij * exp(beta_ij * C).

    Attributes:
        n_states: Number of Markov states.
    """

    def __init__(self, n_states: int) -> None:
        """Initialise continuous-time Markov model.

        Args:
            n_states: Number of states.
        """
        if n_states < 2:
            raise ValueError("n_states must be >= 2.")
        self.n_states = n_states

    def _n_offdiag(self) -> int:
        """Total number of off-diagonal Q-matrix entries."""
        K = self.n_states
        return K * (K - 1)

    def rate_matrix(self, params: np.ndarray, concentration: float = 0.0) -> np.ndarray:
        """Compute the Q-matrix from parameters.

        Off-diagonal rates are recovered via exp(log_rate + beta * C).
        Diagonal elements are set to enforce row-sums = 0.

        Args:
            params: Log-rates and optional concentration slopes.
            concentration: Drug concentration.

        Returns:
            Valid Q-matrix of shape (n_states, n_states).
        """
        K = self.n_states
        n_od = self._n_offdiag()

        log_rates = params[:n_od]
        betas = params[n_od : n_od + n_od] if len(params) > n_od else np.zeros(n_od)

        Q = np.zeros((K, K))
        od_idx = 0
        for i in range(K):
            for j in range(K):
                if i != j:
                    rate = np.exp(float(log_rates[od_idx]) + float(betas[od_idx]) * concentration)
                    Q[i, j] = rate
                    od_idx += 1
        # Set diagonal: q_ii = -sum_{j!=i} q_ij
        for i in range(K):
            Q[i, i] = -np.sum(Q[i, :])
        return Q

    def transition_matrix(
        self,
        t: float,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> np.ndarray:
        """Compute transition probability matrix P(t) = expm(Q * t).

        Args:
            t: Time interval duration.
            params: Parameter vector (log-rates + optional betas).
            concentration: Drug concentration.

        Returns:
            Row-stochastic matrix of shape (n_states, n_states).
        """
        Q = self.rate_matrix(params, concentration)
        if t <= 0.0:
            return np.eye(self.n_states)
        P = expm(Q * t)
        # Clip to [0, 1] and renormalise for numerical safety
        P = np.clip(P, 0.0, 1.0)
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return P / row_sums

    def log_likelihood(self, data: list[CategoricalData], params: np.ndarray) -> float:
        """Compute total log-likelihood from observed transition sequences.

        For each consecutive pair (state at t_i, state at t_{i+1}):
            log P(t_{i+1} - t_i)[state_i, state_{i+1}]

        Args:
            data: Per-subject CategoricalData; uses ``categories``, ``times``,
                and optionally ``concentrations``.
            params: Parameter vector.

        Returns:
            Total log-likelihood.
        """
        ll = 0.0
        for subj in data:
            states = subj.categories.astype(int)
            times = np.asarray(subj.times, dtype=float)
            n = len(states)
            for t_idx in range(n - 1):
                delta_t = float(times[t_idx + 1] - times[t_idx])
                conc = float(subj.concentrations[t_idx]) if subj.concentrations is not None else 0.0
                P = self.transition_matrix(delta_t, params, conc)
                s_from = states[t_idx]
                s_to = states[t_idx + 1]
                if 0 <= s_from < self.n_states and 0 <= s_to < self.n_states:
                    ll += np.log(max(P[s_from, s_to], 1e-300))
        return ll if np.isfinite(ll) else -1e300

    def fit(
        self,
        data: list[CategoricalData],
        init_params: np.ndarray | None = None,
    ) -> CategoricalResult:
        """Fit continuous-time Markov model via MLE.

        Args:
            data: Per-subject CategoricalData records.
            init_params: Initial log-rate parameter vector.  Defaults to
                zeros (all rates = 1).

        Returns:
            CategoricalResult.
        """
        n_od = self._n_offdiag()
        if init_params is None:
            init_params = np.full(n_od, -2.0)  # rates ~ exp(-2) ≈ 0.135

        def neg_ll(params: np.ndarray) -> float:
            try:
                ll = self.log_likelihood(data, params)
                return -2.0 * ll if np.isfinite(ll) else 1e12
            except Exception:
                return 1e12

        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        ofv = float(result.fun)
        aic = ofv + 2.0 * len(params_hat)

        return CategoricalResult(
            thresholds=params_hat,
            coef=np.array([]),
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
        )
