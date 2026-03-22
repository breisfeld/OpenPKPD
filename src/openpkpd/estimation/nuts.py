"""
No-U-Turn Sampler (NUTS) — pure NumPy/SciPy implementation.

Implements the NUTS algorithm with dual averaging step-size adaptation
(Algorithm 6 from Hoffman & Gelman 2014) entirely in NumPy.  No
PyMC, NumPyro, or JAX dependency is required.

The sampler can be used standalone or as the ``'nuts'`` backend for
:class:`~openpkpd.estimation.bayes.BAYESMethod`.

Usage::

    from openpkpd.estimation.nuts import NUTSSampler

    sampler = NUTSSampler(log_prob_fn, grad_log_prob_fn)
    samples = sampler.sample(
        init_theta=np.array([1.5, 0.1, 30.0]),
        n_samples=1000,
        n_warmup=500,
    )
    print(samples.shape)   # (1000, 3)

References
----------
Hoffman, M.D. & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively
    setting path lengths in Hamiltonian Monte Carlo. JMLR 15:1593-1623.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Leapfrog integrator
# ---------------------------------------------------------------------------


def _leapfrog(
    theta: np.ndarray,
    r: np.ndarray,
    grad_log_prob: Callable[[np.ndarray], np.ndarray],
    step_size: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Leapfrog integrator for Hamiltonian dynamics.

    Args:
        theta:         Current position (parameters).
        r:             Current momentum.
        grad_log_prob: Gradient of log p(theta) w.r.t. theta.
        step_size:     Leapfrog step size epsilon.
        n_steps:       Number of leapfrog steps.

    Returns:
        (theta_new, r_new) after *n_steps* steps.
    """
    r = r + 0.5 * step_size * grad_log_prob(theta)
    for _ in range(n_steps - 1):
        theta = theta + step_size * r
        r = r + step_size * grad_log_prob(theta)
    theta = theta + step_size * r
    r = r + 0.5 * step_size * grad_log_prob(theta)
    return theta, r


# ---------------------------------------------------------------------------
# NUTS tree building (Algorithm 3 from Hoffman & Gelman 2014)
# ---------------------------------------------------------------------------


def _build_tree(
    theta: np.ndarray,
    r: np.ndarray,
    log_u: float,
    v: int,
    j: int,
    step_size: float,
    log_prob: Callable[[np.ndarray], float],
    grad_log_prob: Callable[[np.ndarray], np.ndarray],
    joint_log_prob: Callable[[np.ndarray, np.ndarray], float],
    delta_max: float = 1000.0,
) -> tuple:
    """
    Recursively build a NUTS binary tree.

    Returns:
        (theta_minus, r_minus, theta_plus, r_plus,
         theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime)
    """
    if j == 0:
        # Base case: single leapfrog step
        theta_prime, r_prime = _leapfrog(theta, r, grad_log_prob, v * step_size, 1)
        joint = joint_log_prob(theta_prime, r_prime)
        n_prime = int(log_u <= joint)
        s_prime = int(joint > log_u - delta_max)
        alpha_prime = min(1.0, np.exp(joint - joint_log_prob(theta, r)))
        return (
            theta_prime,
            r_prime,
            theta_prime,
            r_prime,
            theta_prime,
            n_prime,
            s_prime,
            alpha_prime,
            1,
        )

    # Recursion
    (
        theta_minus,
        r_minus,
        theta_plus,
        r_plus,
        theta_prime,
        n_prime,
        s_prime,
        alpha_prime,
        n_alpha_prime,
    ) = _build_tree(
        theta,
        r,
        log_u,
        v,
        j - 1,
        step_size,
        log_prob,
        grad_log_prob,
        joint_log_prob,
        delta_max,
    )

    if s_prime:
        if v == -1:
            (
                theta_minus,
                r_minus,
                _,
                _,
                theta_double_prime,
                n_double_prime,
                s_double_prime,
                alpha_double_prime,
                n_alpha_double_prime,
            ) = _build_tree(
                theta_minus,
                r_minus,
                log_u,
                v,
                j - 1,
                step_size,
                log_prob,
                grad_log_prob,
                joint_log_prob,
                delta_max,
            )
        else:
            (
                _,
                _,
                theta_plus,
                r_plus,
                theta_double_prime,
                n_double_prime,
                s_double_prime,
                alpha_double_prime,
                n_alpha_double_prime,
            ) = _build_tree(
                theta_plus,
                r_plus,
                log_u,
                v,
                j - 1,
                step_size,
                log_prob,
                grad_log_prob,
                joint_log_prob,
                delta_max,
            )

        if n_prime > 0:
            accept_prob = n_double_prime / (n_prime + n_double_prime)
        else:
            accept_prob = 1.0 if n_double_prime > 0 else 0.0

        if np.random.uniform() < accept_prob:
            theta_prime = theta_double_prime

        alpha_prime = alpha_prime + alpha_double_prime
        n_alpha_prime = n_alpha_prime + n_alpha_double_prime
        n_prime = n_prime + n_double_prime

        # No-U-Turn condition
        delta_theta = theta_plus - theta_minus
        s_prime = int(
            s_double_prime
            and np.dot(delta_theta, r_minus) >= 0
            and np.dot(delta_theta, r_plus) >= 0
        )

    return (
        theta_minus,
        r_minus,
        theta_plus,
        r_plus,
        theta_prime,
        n_prime,
        s_prime,
        alpha_prime,
        n_alpha_prime,
    )


# ---------------------------------------------------------------------------
# NUTSSampler
# ---------------------------------------------------------------------------


class NUTSSampler:
    """
    No-U-Turn Sampler with dual averaging step-size adaptation.

    Args:
        log_prob_fn:      Function returning the log-probability of theta.
                          Signature: ``(theta: np.ndarray) -> float``.
        grad_log_prob_fn: Function returning the gradient of log p(theta).
                          Signature: ``(theta: np.ndarray) -> np.ndarray``.
                          If None, a finite-difference approximation is used
                          (slower but requires no analytic gradient).
        delta:            Target acceptance probability for dual averaging
                          (default 0.65, NUTS recommended value).
        max_tree_depth:   Maximum binary tree depth (default 10 → 2^10 steps).
        delta_max:        Max energy deviation allowed (default 1000).
        fd_step:          Finite-difference step size (used only when
                          grad_log_prob_fn is None).
        seed:             Random seed for reproducibility.
    """

    def __init__(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        delta: float = 0.65,
        max_tree_depth: int = 10,
        delta_max: float = 1000.0,
        fd_step: float = 1e-5,
        seed: int | None = None,
    ) -> None:
        self._log_prob = log_prob_fn
        self._delta = delta
        self._max_tree_depth = max_tree_depth
        self._delta_max = delta_max
        self._fd_step = fd_step
        if seed is not None:
            np.random.seed(seed)

        if grad_log_prob_fn is not None:
            self._grad_log_prob = grad_log_prob_fn
        else:
            self._grad_log_prob = self._fd_gradient

    def _fd_gradient(self, theta: np.ndarray) -> np.ndarray:
        """Finite-difference gradient approximation."""
        grad = np.zeros_like(theta)
        f0 = self._log_prob(theta)
        for i in range(len(theta)):
            th_p = theta.copy()
            th_p[i] += self._fd_step
            grad[i] = (self._log_prob(th_p) - f0) / self._fd_step
        return grad

    def _joint_log_prob(self, theta: np.ndarray, r: np.ndarray) -> float:
        """Joint log-probability: log p(theta) - 0.5 * r^T r."""
        return float(self._log_prob(theta)) - 0.5 * float(np.dot(r, r))

    def sample(
        self,
        init_theta: np.ndarray,
        n_samples: int = 1000,
        n_warmup: int = 500,
        init_step_size: float = 0.1,
    ) -> np.ndarray:
        """
        Draw samples from the target distribution.

        Warm-up phase uses dual averaging to adapt the step size.
        Post-warm-up samples use the fixed adapted step size.

        Args:
            init_theta:     Initial parameter values, shape (n_params,).
            n_samples:      Number of post-warm-up samples to return.
            n_warmup:       Number of warm-up (adaptation) steps (discarded).
            init_step_size: Initial leapfrog step size.

        Returns:
            Sample array of shape (n_samples, n_params).
        """
        theta = np.asarray(init_theta, dtype=float).copy()
        n_params = len(theta)

        # Dual averaging parameters (Nesterov 2009 / Hoffman & Gelman 2014)
        mu = np.log(10 * init_step_size)
        log_eps_bar = 0.0
        h_bar = 0.0
        gamma = 0.05
        t0 = 10.0
        kappa = 0.75
        epsilon = init_step_size

        samples: list[np.ndarray] = []
        total_steps = n_warmup + n_samples

        for m in range(1, total_steps + 1):
            # Sample fresh momentum
            r0 = np.random.randn(n_params)
            joint0 = self._joint_log_prob(theta, r0)

            # Slice variable
            log_u = joint0 - np.random.exponential(1.0)

            # Build NUTS tree
            theta_minus = theta.copy()
            theta_plus = theta.copy()
            r_minus = r0.copy()
            r_plus = r0.copy()
            n_accepted = 1
            s = 1
            theta_m = theta.copy()
            alpha_sum = 0.0
            n_alpha = 0

            j = 0
            while s and j < self._max_tree_depth:
                v = 1 if np.random.uniform() > 0.5 else -1
                if v == -1:
                    (
                        theta_minus,
                        r_minus,
                        _,
                        _,
                        theta_prime,
                        n_prime,
                        s_prime,
                        alpha_prime,
                        n_alpha_prime,
                    ) = _build_tree(
                        theta_minus,
                        r_minus,
                        log_u,
                        v,
                        j,
                        epsilon,
                        self._log_prob,
                        self._grad_log_prob,
                        self._joint_log_prob,
                        self._delta_max,
                    )
                else:
                    (
                        _,
                        _,
                        theta_plus,
                        r_plus,
                        theta_prime,
                        n_prime,
                        s_prime,
                        alpha_prime,
                        n_alpha_prime,
                    ) = _build_tree(
                        theta_plus,
                        r_plus,
                        log_u,
                        v,
                        j,
                        epsilon,
                        self._log_prob,
                        self._grad_log_prob,
                        self._joint_log_prob,
                        self._delta_max,
                    )

                if s_prime:
                    accept = min(1.0, n_prime / max(n_accepted, 1))
                    if np.random.uniform() < accept:
                        theta_m = theta_prime

                n_accepted += n_prime
                alpha_sum += alpha_prime
                n_alpha += n_alpha_prime

                # No-U-Turn stop
                delta_theta = theta_plus - theta_minus
                s = int(
                    s_prime
                    and np.dot(delta_theta, r_minus) >= 0
                    and np.dot(delta_theta, r_plus) >= 0
                )
                j += 1

            theta = theta_m

            # Dual averaging step-size adaptation during warm-up
            if m <= n_warmup:
                alpha_bar = alpha_sum / max(n_alpha, 1)
                h_bar = (1 - 1.0 / (m + t0)) * h_bar + (self._delta - alpha_bar) / (m + t0)
                log_eps = mu - np.sqrt(m) / gamma * h_bar
                log_eps_bar = m ** (-kappa) * log_eps + (1 - m ** (-kappa)) * log_eps_bar
                epsilon = np.exp(log_eps)
            else:
                epsilon = np.exp(log_eps_bar)
                samples.append(theta.copy())

        return np.array(samples)


# ---------------------------------------------------------------------------
# Integration with BAYESMethod
# ---------------------------------------------------------------------------


def nuts_estimate(
    log_prob_fn: Callable[[np.ndarray], float],
    init_theta: np.ndarray,
    n_samples: int = 1000,
    n_warmup: int = 500,
    delta: float = 0.65,
    seed: int = 42,
) -> dict:
    """
    Run the NUTS sampler and return a summary dict compatible with
    :class:`~openpkpd.estimation.bayes.BayesianResult`.

    Args:
        log_prob_fn:  Log-probability function.
        init_theta:   Initial parameter vector.
        n_samples:    Number of posterior samples to collect.
        n_warmup:     Number of warm-up (adaptation) samples.
        delta:        Target acceptance probability.
        seed:         Random seed.

    Returns:
        Dict with keys: ``samples`` (n_samples × n_params), ``r_hat`` (ones),
        ``n_effective`` (rough estimate), ``backend_used`` = ``'nuts'``.
    """
    sampler = NUTSSampler(log_prob_fn, delta=delta, seed=seed)
    samples = sampler.sample(init_theta, n_samples=n_samples, n_warmup=n_warmup)

    # Rough effective sample size via autocorrelation lag-1 estimate
    n_eff = []
    for k in range(samples.shape[1]):
        col = samples[:, k]
        if col.std() < 1e-10:
            n_eff.append(0)
            continue
        col_norm = col - col.mean()
        ac1 = float(np.corrcoef(col_norm[:-1], col_norm[1:])[0, 1]) if len(col) > 1 else 0.0
        rho = max(min(ac1, 0.9999), -0.9999)
        neff = int(len(col) * (1 - rho) / (1 + rho))
        n_eff.append(max(neff, 1))

    return {
        "samples": samples,
        "r_hat": np.ones(samples.shape[1]),
        "n_effective": np.array(n_eff),
        "backend_used": "nuts",
    }
