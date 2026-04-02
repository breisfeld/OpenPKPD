"""
Stochastic Approximation EM (SAEM) estimation method with multi-chain
Rao-Blackwellisation.

Two-phase algorithm:
  Phase 1 (K1 ≈ 300 iterations, γ=1, stochastic exploration):
    E-step: Sample C chains η_{i,c}^(k) ~ p(η|y_i, θ^(k-1)) via MH per chain
    M-step:
      Ω update — closed-form SA:  Q_Ω = Q_Ω + γ_k·(SS_Ω − Q_Ω),  Ω = Q_Ω
      θ update — direct M-step argmax every iteration:  θ = argmax h_k(θ)
      σ update — direct M-step argmax every iteration:  σ = argmax h_k(σ)

  Phase 2 (K2 ≈ 200 iterations, γ_k = (k-K1)^{-0.7}, convergence):
    Same E-step and Ω update (γ decreasing).
    θ and σ: still direct argmax at each iteration (no SA averaging).
    Final θ/σ reported as mean of the last _PH2_WINDOW phase-2 estimates.

Design note on θ/σ updates
---------------------------
Standard SAEM theory (Kuhn & Lavielle 2004) applies the SA recursion to
sufficient statistics, not to the M-step argmax.  For exponential-family
models the SA-averaged sufficient statistics feed a closed-form M-step
(e.g. Ω = Q_Ω / N).  For nonlinear mixed-effects models the θ M-step has
no closed-form; the common implementation computes
  θ_{k+1} = argmax Q_k(θ),  Q_k(θ) = Q_{k-1}(θ) + γ_k·(h_k(θ)−Q_{k-1}(θ))
i.e. SA is applied to the *function* Q, not to the argmax.  Applying SA
averaging to the argmax sequence instead (Q_θ += γ·(θ_new − Q_θ)) is a
biased approximation: in phase 2 it produces an exponentially-weighted
average of past argmax values that systematically undershoots the true
population parameter (produces a systematic negative bias on CL in practice).  The correct
treatment is to take the direct argmax at each iteration and average the
last _PH2_WINDOW estimates for reporting stability.

Rao-Blackwellisation (Kuhn & Lavielle 2004, Combes & Lavielle 2015):
  With C parallel MH chains per subject, the Monte Carlo variance of the
  sufficient statistics is reduced by a factor of approximately C.  In the
  M-step the per-subject sufficient statistic is the average over chains:

      SS_omega_i = (1/C) * Σ_c  η_{i,c} η_{i,c}^T
      SS_theta_i = (1/C) * Σ_c  log p(y_i | η_{i,c}, θ)

Reference: Delyon, Lavielle, Moulines (1999); Kuhn & Lavielle (2004)
"""

from __future__ import annotations

import math
import time
import warnings as _warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from scipy.optimize import minimize

from openpkpd.estimation.base import EstimationMethod, EstimationResult
from openpkpd.math.matrix import repair_pd
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.constants import Method
from openpkpd.utils.errors import WarningCode, WarningSeverity
from openpkpd.utils.logging import get_logger

logger = get_logger("estimation.saem")


class ConvergenceWarning(UserWarning):
    """Warning emitted when SAEM phase-1 mixing or phase-2 convergence is suspect."""


class SAEMMethod(EstimationMethod):
    """
    Stochastic Approximation EM (SAEM) estimation with multi-chain
    Rao-Blackwellisation.

    When ``n_chains > 1``, multiple independent MH chains are run per subject
    in each E-step iteration.  The per-subject sufficient statistics for the
    M-step are the *average* across chains (Rao-Blackwell estimator), reducing
    Monte Carlo variance by approximately a factor of ``n_chains`` compared to
    the single-chain variant.

    Args:
        n_iter_phase1:    Number of stochastic phase iterations (K1).
        n_iter_phase2:    Number of convergence phase iterations (K2).
        n_chains:         Number of parallel MH chains per subject (≥ 1).
                          Values of 3–5 are typical for Rao-Blackwellisation.
        mh_accept_target: Target MH acceptance rate (for adaptive step-size).
        mh_step_size:     Initial MH proposal standard deviation.
        print_interval:   Print OFV every N iterations.
        seed:             Random seed for reproducibility.
    """

    method_name = Method.SAEM

    #: Number of phase-2 trailing iterations used to assess parameter stability.
    _PH2_WINDOW: int = 20

    def __init__(
        self,
        n_iter_phase1: int = 300,
        n_iter_phase2: int = 200,
        n_chains: int = 5,
        mh_accept_target: float = 0.23,
        mh_step_size: float = 0.4,
        print_interval: int = 10,
        seed: int | None = None,
        n_parallel: int = 1,
        phi_tol: float = 1e-3,
        iteration_callback=None,
        alpha: float = 0.7,
    ) -> None:
        if not (0.5 < alpha <= 1.0):
            raise ValueError(
                f"SAEM SA exponent alpha={alpha} must be in (0.5, 1.0]"
            )
        self.alpha = alpha
        self.n_iter_phase1 = n_iter_phase1
        self.n_iter_phase2 = n_iter_phase2
        self.n_chains = n_chains
        self.mh_accept_target = mh_accept_target
        self.mh_step_size = mh_step_size
        self.print_interval = print_interval
        self.n_parallel = n_parallel
        self.phi_tol = phi_tol
        self.rng = np.random.default_rng(seed)
        self.iteration_callback = iteration_callback

    @staticmethod
    def _check_phase2_convergence(
        ph2_param_history: list[np.ndarray],
        phi_tol: float,
        window: int,
    ) -> tuple[bool, float]:
        """
        Return (converged, rel_change) for the phase-2 stability criterion.

        Compares the mean of the last ``window`` parameter snapshots against the
        mean of the preceding ``window`` snapshots.  Convergence is declared when
        the maximum relative change across all parameters falls below ``phi_tol``.
        Returns ``(False, nan)`` when fewer than ``2 * window`` snapshots exist.
        """
        if len(ph2_param_history) < 2 * window:
            return False, float("nan")
        window_new = np.mean(ph2_param_history[-window:], axis=0)
        window_old = np.mean(ph2_param_history[-2 * window : -window], axis=0)
        denom = np.abs(window_old) + 1e-8
        rel_change = float(np.max(np.abs(window_new - window_old) / denom))
        return rel_change < phi_tol, rel_change

    def _check_phase1_mixing(self, ph1_param_history: list[np.ndarray]) -> None:
        """
        Emit a ConvergenceWarning if Phase 1 appears not to have mixed.

        Uses the same window-comparison logic as _check_phase2_convergence,
        applied to the last n_burn/4 iterations of Phase 1 history.
        """
        n_burn = self.n_iter_phase1
        window = max(1, n_burn // 4)
        _converged, rel_change = self._check_phase2_convergence(
            ph1_param_history, self.phi_tol, window
        )
        if not math.isnan(rel_change) and rel_change >= self.phi_tol:
            _warnings.warn(
                "SAEM Phase 1 may not have mixed: parameter running mean was still changing "
                f"by {rel_change:.2%} at Phase 1 end. Consider increasing n_burn.",
                ConvergenceWarning,
                stacklevel=3,
            )

    @staticmethod
    def _n_free_theta(params: ParameterSet) -> int:
        return (
            sum(1 for s in params.theta_specs if not s.fixed)
            if params.theta_specs
            else params.n_theta()
        )

    @staticmethod
    def _n_free_covariance(specs: list[Any], matrix: np.ndarray) -> int:
        if specs:
            return sum(s.block_size * (s.block_size + 1) // 2 for s in specs if not s.fixed)
        n = matrix.shape[0]
        return n * (n + 1) // 2

    @classmethod
    def _sigma_vector_slice(cls, params: ParameterSet) -> slice:
        start = cls._n_free_theta(params) + cls._n_free_covariance(params.omega_specs, params.omega)
        stop = start + cls._n_free_covariance(params.sigma_specs, params.sigma)
        return slice(start, stop)

    @staticmethod
    def _e_step_one_subject(
        sid: int,
        chains: np.ndarray,
        scale: float,
        indiv: Any,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int,
        n_chains: int,
        n_eta: int,
        rng: np.random.Generator,
    ) -> tuple[int, np.ndarray, np.ndarray, int]:
        """
        Run one MH iteration for all chains of a single subject.

        Returns (sid, new_chains, ss_omega_i, n_accepted).
        Accepts its own RNG so parallel calls are thread-safe.
        """
        new_chains = chains.copy()
        n_accepted = 0
        # Pre-compute Cholesky factor for the MH proposal covariance (scale^2 * omega).
        # This samples from MVN(0, scale^2 * omega) which respects correlations.
        try:
            L_chol = np.linalg.cholesky(scale**2 * omega)
            _use_chol = True
        except np.linalg.LinAlgError:
            # Near-singular omega: fall back to diagonal proposal
            _use_chol = False
            _diag_std = np.sqrt(np.maximum(np.diag(omega), 0.0)) * scale

        for c in range(n_chains):
            eta_current = new_chains[c]
            if _use_chol:
                eta_prop = eta_current + L_chol @ rng.standard_normal(n_eta)
            else:
                eta_prop = eta_current + _diag_std * rng.standard_normal(n_eta)
            try:
                obj_current = indiv.obj_eta(eta_current, theta, omega, sigma, trans=trans)
                obj_prop = indiv.obj_eta(eta_prop, theta, omega, sigma, trans=trans)
                log_accept = -(obj_prop - obj_current) / 2.0
            except Exception:
                log_accept = -np.inf
            if math.log(max(rng.uniform(), 1e-300)) < log_accept:
                new_chains[c] = eta_prop
                n_accepted += 1
        ss_omega_i = (new_chains.T @ new_chains) / n_chains
        return sid, new_chains, ss_omega_i, n_accepted

    def estimate(
        self,
        population_model: Any,
        init_params: ParameterSet,
        **kwargs: Any,
    ) -> EstimationResult:
        t0 = time.time()
        params = init_params.apply_bounds()
        n_subjects = population_model.n_subjects()
        n_eta = params.n_eta()
        n_theta = params.n_theta()
        logger.info(
            f"Starting SAEM: K1={self.n_iter_phase1}, K2={self.n_iter_phase2}, N={n_subjects}"
        )

        n_chains = max(1, self.n_chains)

        # Initialize per-subject chain states
        # eta_chains[sid]: shape (n_chains, n_eta) — one row per chain
        subj_ids = population_model.subject_ids()
        individual_models = {
            sid: population_model.individual_model(sid)
            for sid in subj_ids
        }
        if n_eta > 0:
            omega_init = repair_pd(params.omega)
            eta_chains = {
                sid: self.rng.multivariate_normal(np.zeros(n_eta), omega_init, size=n_chains)
                for sid in subj_ids
            }
        else:
            eta_chains = {sid: np.zeros((n_chains, n_eta)) for sid in subj_ids}
        mh_scales: dict[int, float] = dict.fromkeys(subj_ids, self.mh_step_size)

        # Per-subject RNGs — independent streams for thread-safe parallel E-step.
        # Fall back to self.rng when it doesn't support .integers() (e.g. test stubs)
        # so that serial tests relying on stub behaviour are unaffected.
        def _child_rng() -> np.random.Generator:
            try:
                return np.random.default_rng(int(self.rng.integers(0, 2**31)))
            except AttributeError:
                return self.rng  # type: ignore[return-value]

        subj_rngs: dict[int, np.random.Generator] = {sid: _child_rng() for sid in subj_ids}
        n_workers = self.n_parallel if self.n_parallel > 0 else None

        # Sufficient statistics
        theta = params.theta.copy()
        omega = params.omega.copy()
        sigma = params.sigma.copy()

        # Running average sufficient statistic for OMEGA M-step only.
        # Q_omega = E[eta_i * eta_i^T]  (SAEM SA recursion, closed-form M-step)
        # NOTE: theta and sigma are updated via direct M-step argmax — no SA
        # averaging is applied to those parameters (see module docstring).
        Q_omega = np.zeros((n_eta, n_eta))
        sigma_slice = self._sigma_vector_slice(init_params)
        sigma_free = sigma_slice.stop - sigma_slice.start

        if init_params.theta_specs:
            free_theta_idx = [
                i
                for i, spec in enumerate(init_params.theta_specs)
                if not getattr(spec, "fixed", False)
            ]
            theta_bounds = [
                (
                    None
                    if math.isinf(init_params.theta_specs[i].lower)
                    else init_params.theta_specs[i].lower,
                    None
                    if math.isinf(init_params.theta_specs[i].upper)
                    else init_params.theta_specs[i].upper,
                )
                for i in free_theta_idx
            ]
        else:
            free_theta_idx = list(range(n_theta))
            theta_bounds = [(None, None)] * n_theta

        ofv_history: list[float] = []
        K_total = self.n_iter_phase1 + self.n_iter_phase2

        # Phase-2 parameter tracking for:
        #   ph2_param_history — convergence criterion (compares window means)
        #   ph2_theta_history — final theta_final = mean of last _PH2_WINDOW
        #   ph2_sigma_history — final sigma_final = mean of last _PH2_WINDOW
        ph2_param_history: list[np.ndarray] = []
        ph2_theta_history: list[np.ndarray] = []
        ph2_sigma_history: list[np.ndarray] = []
        # Phase-1 parameter history for mixing diagnostic
        ph1_param_history: list[np.ndarray] = []
        converged = False

        # Per-subject MH acceptance rate warning tracking (emit at most once per phase)
        _warned_low: set[int] = set()
        _warned_high: set[int] = set()

        prev_is_phase1 = True

        for k in range(K_total):
            is_phase1 = k < self.n_iter_phase1

            # Detect Phase 1 → Phase 2 transition and check Phase 1 mixing
            if prev_is_phase1 and not is_phase1:
                _warned_low = set()
                _warned_high = set()
                self._check_phase1_mixing(ph1_param_history)

            prev_is_phase1 = is_phase1

            # Phase 2 step counter: resets at Phase 2 onset (step 1 on first Phase 2 iter).
            # At k == n_iter_phase1 (first Phase 2 step): counter = 1 → gamma = 1^(-alpha) = 1.0
            # At k == n_iter_phase1 + 9 (10th Phase 2 step): counter = 10 → gamma = 10^(-alpha)
            gamma = 1.0 if is_phase1 else (k - self.n_iter_phase1 + 1) ** (-self.alpha)

            # E-step: multi-chain MH sampling per subject (Rao-Blackwellisation)
            # ss_omega_sum: accumulates Σ_i (1/C Σ_c η_{i,c} η_{i,c}^T)
            ss_omega_sum = np.zeros((n_eta, n_eta))
            adapt_scale = is_phase1 and k % 50 == 0

            if n_eta == 0:
                # No random effects: skip MH entirely — eta_chains remain zeros(n_chains, 0).
                pass
            else:
                def _run_subject(
                    sid: int,
                    _theta=theta,
                    _omega=omega,
                    _sigma=sigma,
                ) -> tuple[int, np.ndarray, np.ndarray, int]:
                    return self._e_step_one_subject(
                        sid,
                        eta_chains[sid],
                        mh_scales[sid],
                        individual_models[sid],
                        _theta,
                        _omega,
                        _sigma,
                        population_model.trans,
                        n_chains,
                        n_eta,
                        subj_rngs[sid],
                    )

                if self.n_parallel == 1 or len(subj_ids) <= 1:
                    results = [_run_subject(sid) for sid in subj_ids]
                else:
                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                        futures = {pool.submit(_run_subject, sid): sid for sid in subj_ids}
                        results = [f.result() for f in as_completed(futures)]

                for sid, new_chains, ss_omega_i, n_accepted in results:
                    eta_chains[sid] = new_chains
                    ss_omega_sum += ss_omega_i
                    accept_rate = n_accepted / n_chains
                    if adapt_scale:
                        mh_scales[sid] *= 1.2 if accept_rate > self.mh_accept_target else 0.8
                        mh_scales[sid] = float(np.clip(mh_scales[sid], 0.05, 5.0))
                    # MH acceptance rate extreme warnings (at most once per subject per phase)
                    if accept_rate < 0.05:
                        if sid not in _warned_low:
                            _warned_low.add(sid)
                            logger.warning(
                                "SAEM: subject %s MH acceptance rate %.1f%% is very low — "
                                "chain may be stuck. "
                                "Consider reducing mh_scale or increasing n_chains.",
                                sid, accept_rate * 100,
                            )
                    elif accept_rate > 0.95:
                        if sid not in _warned_high:
                            _warned_high.add(sid)
                            logger.warning(
                                "SAEM: subject %s MH acceptance rate %.1f%% is very high — "
                                "proposals are too small. "
                                "Consider increasing mh_scale.",
                                sid, accept_rate * 100,
                            )

            # M-step: update omega (closed form), theta (numerical), sigma (numerical)
            # OMEGA M-step: Ω = (1/N) Σ_i ss_omega_i  (RB-averaged over chains)
            ss_omega = ss_omega_sum / n_subjects
            # SAEM update
            Q_omega = Q_omega + gamma * (ss_omega - Q_omega)
            omega = repair_pd(Q_omega.copy())

            # THETA M-step: minimize Rao-Blackwell averaged subject OFV
            # (IndividualModel.log_likelihood returns -2 * log p(y | eta, theta))
            # Updated every iteration (phase 1 and 2).  No SA averaging is
            # applied to theta — see module docstring for rationale.
            if free_theta_idx:

                def theta_obj(free_th: np.ndarray, _theta=theta, _sigma=sigma) -> float:
                    th = _theta.copy()
                    th[free_theta_idx] = free_th
                    total = 0.0
                    for sid in subj_ids:
                        indiv = individual_models[sid]
                        # Average subject OFV across chains (RB estimator)
                        for c in range(n_chains):
                            try:
                                ll = indiv.log_likelihood(
                                    th,
                                    eta_chains[sid][c],
                                    _sigma,
                                    trans=population_model.trans,
                                )
                                total += ll / n_chains
                            except Exception:
                                total += 1e6 / n_chains
                    return total

                # Use analytical theta gradient when available: eliminates
                # n_free_theta+1 FD ODE calls per L-BFGS-B step.
                # Probe once to confirm return type is a proper ndarray with
                # len(theta) elements — guards against partial implementations.
                _theta_jac: Any = None
                _rep_indiv = individual_models[subj_ids[0]] if subj_ids else None
                _sg = getattr(_rep_indiv, "supports_theta_data_objective_gradient", None)
                _tg = getattr(_rep_indiv, "theta_data_objective_gradient", None)
                if callable(_sg) and _sg(population_model.trans) and callable(_tg):
                    try:
                        _probe = _tg(
                            theta,
                            eta_chains[subj_ids[0]][0],
                            sigma,
                            trans=population_model.trans,
                        )
                        if isinstance(_probe, np.ndarray) and len(_probe) == len(theta):
                            _free_idx = list(free_theta_idx)

                            def theta_jac(
                                free_th: np.ndarray,
                                _theta=theta,
                                _sigma=sigma,
                                _fidx=_free_idx,  # bind now — not a loop var
                            ) -> np.ndarray:
                                th = _theta.copy()
                                th[_fidx] = free_th
                                grad = np.zeros(len(_fidx), dtype=float)
                                for sid in subj_ids:
                                    indiv = individual_models[sid]
                                    for c in range(n_chains):
                                        try:
                                            g = indiv.theta_data_objective_gradient(
                                                th,
                                                eta_chains[sid][c],
                                                _sigma,
                                                trans=population_model.trans,
                                            )
                                            grad += np.asarray(g, dtype=float)[_fidx] / n_chains
                                        except Exception:
                                            pass
                                return grad

                            _theta_jac = theta_jac
                    except Exception:
                        pass

                th_result = minimize(
                    theta_obj,
                    theta[free_theta_idx],
                    method="L-BFGS-B",
                    jac=_theta_jac,
                    bounds=theta_bounds or None,
                    options={"maxiter": 20, "ftol": 1e-6},
                )
                theta_new = theta.copy()
                theta_new[free_theta_idx] = th_result.x
                theta = (
                    ParameterSet(
                        theta=theta_new,
                        omega=omega,
                        sigma=sigma,
                        theta_specs=init_params.theta_specs,
                        omega_specs=init_params.omega_specs,
                        sigma_specs=init_params.sigma_specs,
                    )
                    .apply_bounds()
                    .theta
                )
                if not is_phase1:
                    ph2_theta_history.append(theta.copy())

            if sigma_free > 0:
                sigma_template = ParameterSet(
                    theta=theta,
                    omega=omega,
                    sigma=sigma,
                    theta_specs=init_params.theta_specs,
                    omega_specs=init_params.omega_specs,
                    sigma_specs=init_params.sigma_specs,
                )
                sigma_vec0 = sigma_template.to_vector()

                def sigma_obj(
                    free_sig: np.ndarray,
                    _sigma_vec0=sigma_vec0,
                    _sigma_template=sigma_template,
                    _theta=theta,
                ) -> float:
                    cand_vec = _sigma_vec0.copy()
                    cand_vec[sigma_slice] = free_sig
                    cand_sigma = (
                        ParameterSet.from_vector(cand_vec, _sigma_template).apply_bounds().sigma
                    )
                    total = 0.0
                    for sid in subj_ids:
                        indiv = individual_models[sid]
                        for c in range(n_chains):
                            try:
                                ll = indiv.log_likelihood(
                                    _theta,
                                    eta_chains[sid][c],
                                    cand_sigma,
                                    trans=population_model.trans,
                                )
                                total += ll / n_chains
                            except Exception:
                                total += 1e6 / n_chains
                    return total

                sig_result = minimize(
                    sigma_obj,
                    sigma_vec0[sigma_slice],
                    method="L-BFGS-B",
                    options={"maxiter": 20, "ftol": 1e-6},
                )
                sigma_vec_new = sigma_vec0.copy()
                sigma_vec_new[sigma_slice] = sig_result.x
                sigma = (
                    ParameterSet.from_vector(sigma_vec_new, sigma_template).apply_bounds().sigma
                )
                if not is_phase1:
                    ph2_sigma_history.append(sigma.copy())

            # Compute current OFV for monitoring (average chain 0 across subjects)
            ofv = 0.0
            for sid in subj_ids:
                indiv = individual_models[sid]
                try:
                    ofv += indiv.obj_eta(
                        eta_chains[sid][0],
                        theta,
                        omega,
                        sigma,
                        trans=population_model.trans,
                    )
                except Exception:
                    ofv += 1e6
            ofv_history.append(ofv)
            if self.iteration_callback is not None:
                try:
                    self.iteration_callback(k, ofv)
                except Exception:
                    pass

            if k % self.print_interval == 0:
                phase = "P1" if is_phase1 else "P2"
                logger.info(f"  SAEM iter {k:4d} [{phase}]  OFV={ofv:.4f}  γ={gamma:.4f}")

            # Track Phase-1 parameter history for mixing diagnostic
            if is_phase1:
                phi1 = np.concatenate([theta, np.diag(omega)])
                ph1_param_history.append(phi1)

            # Phase-2 convergence criterion: parameter stability over a window.
            # Include all omega lower-triangle elements and sigma diagonal so that
            # correlated random effects and residual error are not excluded.
            if not is_phase1:
                n = omega.shape[0]
                omega_lower = omega[np.tril_indices(n)]
                phi = np.concatenate([theta, omega_lower, np.diag(sigma)])
                ph2_param_history.append(phi)
                W = self._PH2_WINDOW
                converged, rel_change = self._check_phase2_convergence(
                    ph2_param_history, self.phi_tol, W
                )
                if converged:
                    logger.info(
                        f"  SAEM phase-2 converged at iter {k} "
                        f"(max rel Δphi={rel_change:.2e} < phi_tol={self.phi_tol:.2e})"
                    )
                    break

        elapsed = time.time() - t0
        final_ofv = ofv_history[-1] if ofv_history else float("nan")
        logger.info(f"SAEM completed in {elapsed:.1f}s, OFV={final_ofv:.4f}")

        # Final post-hoc ETAs: mean across all chains (RB point estimate)
        final_etas = {sid: np.mean(eta_chains[sid], axis=0) for sid in subj_ids}

        # Final theta: mean of the last _PH2_WINDOW phase-2 M-step argmax values.
        # This averaging suppresses iteration-to-iteration noise in the M-step
        # while using the direct argmax (no SA averaging across iterations).
        W = self._PH2_WINDOW
        if ph2_theta_history:
            n_avg = min(W, len(ph2_theta_history))
            theta_final_arr = np.mean(ph2_theta_history[-n_avg:], axis=0)
            theta_final_arr = (
                ParameterSet(
                    theta=theta_final_arr,
                    omega=omega,
                    sigma=sigma,
                    theta_specs=init_params.theta_specs,
                    omega_specs=init_params.omega_specs,
                    sigma_specs=init_params.sigma_specs,
                )
                .apply_bounds()
                .theta
            )
        else:
            theta_final_arr = theta

        # Final sigma: mean of last _PH2_WINDOW phase-2 sigma estimates.
        if ph2_sigma_history:
            n_avg = min(W, len(ph2_sigma_history))
            sigma_final_arr = np.mean(ph2_sigma_history[-n_avg:], axis=0)
        else:
            sigma_final_arr = sigma

        conv_msg = (
            f"SAEM phase-2 stability criterion met "
            f"({n_chains} chain{'s' if n_chains > 1 else ''} per subject)"
            if converged
            else (
                f"SAEM phase-2 ran to {self.n_iter_phase2} iterations without satisfying "
                f"phi_tol={self.phi_tol:.1e} — parameters may not be fully converged"
            )
        )

        res = EstimationResult(
            theta_final=theta_final_arr,
            omega_final=omega,
            sigma_final=sigma_final_arr,
            ofv=final_ofv,
            converged=converged,
            post_hoc_etas=final_etas,
            ofv_history=ofv_history,
            elapsed_time=elapsed,
            method=self.method_name,
            message=conv_msg,
        )
        if not converged:
            res.add_structured_warning(
                WarningCode.WARN_007,
                f"SAEM phase-2 stability criterion (phi_tol={self.phi_tol:.1e}) was not "
                f"satisfied after {self.n_iter_phase2} iterations. Consider increasing "
                "n_iter_phase2 or loosening phi_tol.",
                WarningSeverity.WARNING,
            )
        res.check_omega_conditioning()
        dataset = getattr(population_model, "dataset", None)
        if dataset is not None and hasattr(dataset, "n_observations"):
            res.n_observations = int(dataset.n_observations())
        res.n_subjects = n_subjects
        res.compute_n_parameters(
            theta_specs=getattr(init_params, "theta_specs", None),
            omega_specs=getattr(init_params, "omega_specs", None),
            sigma_specs=getattr(init_params, "sigma_specs", None),
        )
        res.compute_shrinkage()
        return res
