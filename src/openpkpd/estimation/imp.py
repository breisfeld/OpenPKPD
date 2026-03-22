"""
Importance Sampling (IMP/IMPMAP) estimation method.

IMP computes the exact marginal likelihood:
    p(y_i | theta, Omega) = ∫ p(y_i | eta, theta) * p(eta | Omega) d(eta)

via Monte Carlo importance sampling:
    p(y_i | theta, Omega) ≈ (1/S) Σ_s w_s * p(y_i | eta_s, theta)

where:
    - Proposal: q(eta) = N(eta_hat_i, V_i)  (Laplace approximation at MAP)
    - Weights:  w_s = p(eta_s | Omega) / q(eta_s)
    - ESS monitoring: effective sample size = (Σ w_s)² / Σ w_s²

IMPMAP: Use MAP estimate of eta (like FOCE post-hoc) as proposal center.

Reference: Combes et al. (2011), Lavielle (2014)
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from openpkpd.estimation.base import EstimationMethod, EstimationResult
from openpkpd.math.matrix import numerical_hessian, repair_pd
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.constants import Method
from openpkpd.utils.logging import get_logger

logger = get_logger("estimation.imp")


class IMPMethod(EstimationMethod):
    """
    Importance Sampling (IMP) estimation.

    Args:
        isample:        Number of importance samples per subject.
        maxeval:        Maximum outer-loop evaluations.
        print_interval: Print interval.
        seed:           Random seed.
    """

    method_name = Method.IMP

    def __init__(
        self,
        isample: int = 300,
        maxeval: int = 200,
        print_interval: int = 10,
        seed: int | None = None,
        n_parallel: int = 1,
    ) -> None:
        self.isample = isample
        self.maxeval = maxeval
        self.print_interval = print_interval
        self.n_parallel = n_parallel
        self.rng = np.random.default_rng(seed)
        self._iter = 0
        self._ofv_history: list[float] = []
        self._subj_rngs: dict[int, np.random.Generator] = {}

    def estimate(
        self,
        population_model: Any,
        init_params: ParameterSet,
        **kwargs: Any,
    ) -> EstimationResult:
        t0 = time.time()
        params = init_params.apply_bounds()
        logger.info(f"Starting IMP estimation, isample={self.isample}")

        self._iter = 0
        self._ofv_history = []

        # One RNG per subject — independent streams, thread-safe for parallel eval
        def _child_rng() -> np.random.Generator:
            try:
                return np.random.default_rng(int(self.rng.integers(0, 2**31)))
            except AttributeError:
                return self.rng  # type: ignore[return-value]

        self._subj_rngs = {sid: _child_rng() for sid in population_model.subject_ids()}

        x0 = params.to_vector()

        def objective(x: np.ndarray) -> float:
            p = ParameterSet.from_vector(x, init_params).apply_bounds()
            ofv = self._compute_imp_ofv(population_model, p)
            self._iter += 1
            self._ofv_history.append(ofv)
            if self._iter % self.print_interval == 0:
                logger.info(f"  Iter {self._iter:5d}  OFV={ofv:.4f}")
            return ofv

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": self.maxeval, "ftol": 1e-8},
        )

        final_params = ParameterSet.from_vector(result.x, init_params).apply_bounds()
        final_ofv = self._compute_imp_ofv(population_model, final_params)

        # Post-hoc ETAs via MAP
        eta_hat = self._map_etas(population_model, final_params)

        elapsed = time.time() - t0
        logger.info(f"IMP completed in {elapsed:.1f}s, OFV={final_ofv:.4f}")

        res = EstimationResult(
            theta_final=final_params.theta,
            omega_final=final_params.omega,
            sigma_final=final_params.sigma,
            ofv=final_ofv,
            converged=result.success,
            post_hoc_etas=eta_hat,
            ofv_history=self._ofv_history,
            elapsed_time=elapsed,
            method=self.method_name,
            message=getattr(result, "message", ""),
        )
        res.compute_shrinkage()
        return res

    def _compute_imp_ofv(
        self,
        population_model: Any,
        params: ParameterSet,
    ) -> float:
        """
        IMP OFV = -2 Σ_i log p̂(y_i | theta, Omega)

        where p̂_i is estimated via importance sampling.
        Subjects are evaluated in parallel when n_parallel > 1.
        """
        subject_ids = population_model.subject_ids()

        def _eval_subject(subj_id: int) -> float:
            try:
                return -2.0 * self._importance_sample(population_model, params, subj_id)
            except Exception:
                return 1e10

        if self.n_parallel == 1 or len(subject_ids) <= 1:
            return sum(_eval_subject(sid) for sid in subject_ids)

        n_workers = self.n_parallel if self.n_parallel > 0 else None
        ofv = 0.0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for result in pool.map(_eval_subject, subject_ids):
                ofv += result
        return ofv

    def _importance_sample(
        self,
        population_model: Any,
        params: ParameterSet,
        subj_id: int,
    ) -> float:
        """
        Estimate log p(y_i | theta, Omega) via importance sampling.

        Proposal: q(eta) = N(eta_map, V_map) where eta_map = MAP estimate.
        """
        indiv = population_model.individual_model(subj_id)
        n_eta = params.n_eta()

        # Find MAP (proposal center)
        eta0 = np.zeros(n_eta)

        def neg_log_joint(eta: np.ndarray) -> float:
            return float(
                indiv.obj_eta(
                    eta,
                    params.theta,
                    params.omega,
                    params.sigma,
                    trans=population_model.trans,
                )
            )

        map_result = minimize(
            neg_log_joint,
            eta0,
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-8},
        )
        eta_map = map_result.x

        # Proposal covariance from the MAP Hessian.
        # ``obj_eta`` is an OFV-like quantity (-2 log posterior up to
        # constants), so the local Gaussian covariance is 2 * H^{-1}.
        eta_hessian = getattr(indiv, "eta_objective_hessian", None)
        if callable(eta_hessian):
            H = np.asarray(
                eta_hessian(
                    params.theta,
                    eta_map,
                    params.omega,
                    params.sigma,
                    trans=population_model.trans,
                ),
                dtype=float,
            )
        else:
            H = numerical_hessian(neg_log_joint, eta_map, eps=1e-4)
        H = repair_pd(H, epsilon=1e-6)
        try:
            V_prop = 2.0 * np.linalg.inv(H)
            V_prop = repair_pd(V_prop, epsilon=1e-8)
        except Exception:
            V_prop = params.omega.copy()

        # Draw samples from proposal — use per-subject RNG if available (thread-safe)
        rng = self._subj_rngs.get(subj_id, self.rng)
        samples = rng.multivariate_normal(eta_map, V_prop, size=self.isample)

        # Compute unnormalized importance weights
        omega_pd = repair_pd(params.omega)
        omega_inv = np.linalg.inv(omega_pd)
        log_det_omega = math.log(max(np.linalg.det(omega_pd), 1e-300))
        log_weights = np.zeros(self.isample)
        for s, eta_s in enumerate(samples):
            try:
                # log p(y_i | eta_s, theta) — data likelihood
                ll_data = -0.5 * indiv.log_likelihood(
                    params.theta,
                    eta_s,
                    params.sigma,
                    trans=population_model.trans,
                )
                # log p(eta_s | Omega) — prior
                log_prior = -0.5 * float(eta_s @ omega_inv @ eta_s) - 0.5 * log_det_omega
                # log q(eta_s) — proposal
                log_proposal = multivariate_normal.logpdf(eta_s, mean=eta_map, cov=V_prop)
                log_weights[s] = ll_data + log_prior - log_proposal
            except Exception:
                log_weights[s] = -1e30

        # Log-sum-exp for numerical stability
        log_weights_max = np.max(log_weights)
        log_marg = log_weights_max + math.log(
            np.mean(np.exp(np.clip(log_weights - log_weights_max, -50, 50)))
        )

        # ESS monitoring
        w = np.exp(np.clip(log_weights - np.max(log_weights), -50, 0))
        w /= w.sum()
        ess = 1.0 / float(np.sum(w**2))
        if ess < 0.1 * self.isample:
            logger.debug(f"  Subject {subj_id}: low ESS={ess:.1f}")

        return float(log_marg)

    def _map_etas(self, population_model: Any, params: ParameterSet) -> dict[int, np.ndarray]:
        """Compute MAP estimates of eta for all subjects."""
        eta_hat: dict[int, np.ndarray] = {}
        for sid in population_model.subject_ids():
            indiv = population_model.individual_model(sid)

            def obj(eta: np.ndarray, _indiv=indiv) -> float:
                return float(
                    _indiv.obj_eta(
                        eta, params.theta, params.omega, params.sigma, trans=population_model.trans
                    )
                )

            res = minimize(
                obj, np.zeros(params.n_eta()), method="L-BFGS-B", options={"maxiter": 100}
            )
            eta_hat[sid] = res.x
        return eta_hat
