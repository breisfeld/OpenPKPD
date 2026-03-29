"""
Stochastic Approximation EM (SAEM) estimation method with multi-chain
Rao-Blackwellisation.

Two-phase algorithm:
  Phase 1 (K1 ≈ 300 iterations, γ=1, stochastic exploration):
    E-step: Sample C chains η_{i,c}^(k) ~ p(η|y_i, θ^(k-1)) via MH per chain
    M-step: Update θ, Ω from Rao-Blackwell sufficient statistics
    Q-update: Q^(k) = Q^(k-1) + γ_k * (SS_k - Q^(k-1))

  Phase 2 (K2 ≈ 200 iterations, γ_k = (k-K1)^{-0.7}, convergence):
    Same as Phase 1 with decreasing step size → convergence

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
    ) -> None:
        self.n_iter_phase1 = n_iter_phase1
        self.n_iter_phase2 = n_iter_phase2
        self.n_chains = n_chains
        self.mh_accept_target = mh_accept_target
        self.mh_step_size = mh_step_size
        self.print_interval = print_interval
        self.n_parallel = n_parallel
        self.phi_tol = phi_tol
        self.rng = np.random.default_rng(seed)

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
        for c in range(n_chains):
            eta_current = new_chains[c]
            eta_prop = eta_current + scale * rng.standard_normal(n_eta)
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
        if n_eta > 0:
            omega_init = repair_pd(params.omega)
            eta_chains = {
                sid: self.rng.multivariate_normal(np.zeros(n_eta), omega_init, size=n_chains)
                for sid in subj_ids
            }
        else:
            eta_chains = {sid: np.zeros((n_chains, n_eta)) for sid in subj_ids}
        mh_scales: dict[int, float] = dict.fromkeys(subj_ids, self.mh_step_size)

        # Per-subject RNGs — created once so each subject's MH stream is
        # independent and thread-safe across parallel E-step calls.
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

        # Running average sufficient statistics for OMEGA M-step
        # Q_omega = E[eta_i * eta_i^T]
        Q_omega = np.zeros((n_eta, n_eta))
        Q_theta = theta.copy()
        Q_sigma = sigma.copy()
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

        # Phase-2 parameter tracking for convergence criterion
        # Stores the concatenation [theta | diag(omega)] at each phase-2 step.
        ph2_param_history: list[np.ndarray] = []
        converged = False

        for k in range(K_total):
            is_phase1 = k < self.n_iter_phase1
            gamma = 1.0 if is_phase1 else (k - self.n_iter_phase1 + 1) ** (-0.7)

            # E-step: multi-chain MH sampling per subject (Rao-Blackwellisation)
            # ss_omega_sum: accumulates Σ_i (1/C Σ_c η_{i,c} η_{i,c}^T)
            ss_omega_sum = np.zeros((n_eta, n_eta))
            adapt_scale = is_phase1 and k % 50 == 0

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
                    population_model.individual_model(sid),
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
                if adapt_scale:
                    accept_rate = n_accepted / n_chains
                    mh_scales[sid] *= 1.2 if accept_rate > self.mh_accept_target else 0.8
                    mh_scales[sid] = float(np.clip(mh_scales[sid], 0.05, 5.0))

            # M-step: update omega (closed form), theta (numerical), sigma (numerical)
            # OMEGA M-step: Ω = (1/N) Σ_i ss_omega_i  (RB-averaged over chains)
            ss_omega = ss_omega_sum / n_subjects
            # SAEM update
            Q_omega = Q_omega + gamma * (ss_omega - Q_omega)
            omega = repair_pd(Q_omega.copy())

            # THETA M-step: minimize Rao-Blackwell averaged subject OFV
            # (IndividualModel.log_likelihood returns -2 * log p(y | eta, theta))
            # For efficiency, only update theta every 5 iterations in phase 1
            if (k % 5 == 0 or not is_phase1) and free_theta_idx:

                def theta_obj(free_th: np.ndarray, _theta=theta, _sigma=sigma) -> float:
                    th = _theta.copy()
                    th[free_theta_idx] = free_th
                    total = 0.0
                    for sid in subj_ids:
                        indiv = population_model.individual_model(sid)
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

                th_result = minimize(
                    theta_obj,
                    theta[free_theta_idx],
                    method="L-BFGS-B",
                    bounds=theta_bounds or None,
                    options={"maxiter": 20, "ftol": 1e-6},
                )
                theta_new = theta.copy()
                theta_new[free_theta_idx] = th_result.x
                theta_new = (
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
                Q_theta = Q_theta + gamma * (theta_new - Q_theta)
                theta = (
                    ParameterSet(
                        theta=Q_theta.copy(),
                        omega=omega,
                        sigma=sigma,
                        theta_specs=init_params.theta_specs,
                        omega_specs=init_params.omega_specs,
                        sigma_specs=init_params.sigma_specs,
                    )
                    .apply_bounds()
                    .theta
                )

            if (k % 5 == 0 or not is_phase1) and sigma_free > 0:
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
                        indiv = population_model.individual_model(sid)
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
                sigma_new = (
                    ParameterSet.from_vector(sigma_vec_new, sigma_template).apply_bounds().sigma
                )
                Q_sigma = Q_sigma + gamma * (sigma_new - Q_sigma)
                sigma = (
                    ParameterSet(
                        theta=theta,
                        omega=omega,
                        sigma=Q_sigma.copy(),
                        theta_specs=init_params.theta_specs,
                        omega_specs=init_params.omega_specs,
                        sigma_specs=init_params.sigma_specs,
                    )
                    .apply_bounds()
                    .sigma
                )

            # Compute current OFV for monitoring (average chain 0 across subjects)
            ofv = 0.0
            for sid in subj_ids:
                indiv = population_model.individual_model(sid)
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

            if k % self.print_interval == 0:
                phase = "P1" if is_phase1 else "P2"
                logger.info(f"  SAEM iter {k:4d} [{phase}]  OFV={ofv:.4f}  γ={gamma:.4f}")

            # Phase-2 convergence criterion: parameter stability over a window
            if not is_phase1:
                phi = np.concatenate([theta, np.diag(omega)])
                ph2_param_history.append(phi)
                W = self._PH2_WINDOW
                if len(ph2_param_history) >= 2 * W:
                    # Compare mean of last W with mean of preceding W
                    window_new = np.mean(ph2_param_history[-W:], axis=0)
                    window_old = np.mean(ph2_param_history[-2 * W : -W], axis=0)
                    denom = np.abs(window_old) + 1e-8
                    rel_change = float(np.max(np.abs(window_new - window_old) / denom))
                    if rel_change < self.phi_tol:
                        logger.info(
                            f"  SAEM phase-2 converged at iter {k} "
                            f"(max rel Δphi={rel_change:.2e} < phi_tol={self.phi_tol:.2e})"
                        )
                        converged = True
                        break

        elapsed = time.time() - t0
        final_ofv = ofv_history[-1] if ofv_history else float("nan")
        logger.info(f"SAEM completed in {elapsed:.1f}s, OFV={final_ofv:.4f}")

        # Final post-hoc ETAs: mean across all chains (RB point estimate)
        final_etas = {sid: np.mean(eta_chains[sid], axis=0) for sid in subj_ids}

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
            theta_final=theta,
            omega_final=omega,
            sigma_final=sigma,
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
