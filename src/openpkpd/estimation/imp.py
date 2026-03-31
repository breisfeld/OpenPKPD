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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from openpkpd.estimation.base import EstimationMethod, EstimationResult
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.math.matrix import numerical_hessian, repair_pd
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.constants import Method
from openpkpd.utils.errors import WarningCode, WarningSeverity
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

    #: ESS fraction below which a WARN_006 is emitted (fraction of isample).
    ESS_WARN_FRACTION: float = 0.10

    def __init__(
        self,
        isample: int = 300,
        maxeval: int = 200,
        print_interval: int = 10,
        seed: int | None = None,
        n_parallel: int = 1,
        is_map: bool = False,
    ) -> None:
        self.isample = isample
        self.maxeval = maxeval
        self.print_interval = print_interval
        self.n_parallel = n_parallel
        self.is_map = is_map
        self.rng = np.random.default_rng(seed)
        self._iter = 0
        self._ofv_history: list[float] = []
        self._subj_seeds: dict[int, int] = {}
        self._last_ess_by_subject: dict[int, float] = {}
        self._warm_start_diagnostics: dict[str, Any] = {}
        self._proposal_cache: dict[tuple[int, bytes], tuple[np.ndarray, np.ndarray]] = {}
        self._proposal_warm_start: dict[int, np.ndarray] = {}
        self._proposal_cache_lock = threading.Lock()
        # Populated during estimate(): list of (subj_id, ess) for low-ESS subjects
        self._low_ess_subjects: list[tuple[int, float]] = []
        if is_map:
            self.method_name = Method.IMPMAP

    def estimate(
        self,
        population_model: Any,
        init_params: ParameterSet,
        **kwargs: Any,
    ) -> EstimationResult:
        t0 = time.time()
        params = init_params.apply_bounds()
        logger.info(
            f"Starting {'IMPMAP' if self.is_map else 'IMP'} estimation, "
            f"isample={self.isample}"
        )

        self._iter = 0
        self._ofv_history = []
        self._last_ess_by_subject = {}
        self._warm_start_diagnostics = {}
        self._proposal_cache = {}
        self._proposal_warm_start = {}
        self._low_ess_subjects = []

        if self.is_map:
            params = self._warm_start_with_focei(population_model, params)

        # One deterministic seed per subject.  Recreating the generator inside
        # _importance_sample keeps the Monte Carlo objective stable across
        # repeated optimizer evaluations (common random numbers) instead of
        # drifting as RNG state advances.
        def _child_seed() -> int:
            try:
                return int(self.rng.integers(0, 2**31))
            except AttributeError:
                return 42

        self._subj_seeds = {sid: _child_seed() for sid in population_model.subject_ids()}

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
        method_label = "IMPMAP" if self.is_map else "IMP"
        logger.info(f"{method_label} completed in {elapsed:.1f}s, OFV={final_ofv:.4f}")

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
            diagnostics=self._build_diagnostics(result, final_ofv),
            n_function_evals=int(getattr(result, "nfev", 0) or 0),
        )

        # ── WARN_006: low effective sample size ──────────────────────────────
        if self._low_ess_subjects:
            n_low = len(self._low_ess_subjects)
            worst_id, worst_ess = min(self._low_ess_subjects, key=lambda t: t[1])
            threshold = self.ESS_WARN_FRACTION * self.isample
            res.add_structured_warning(
                WarningCode.WARN_006,
                f"{n_low} subject(s) had low effective sample size "
                f"(< {threshold:.0f} = {self.ESS_WARN_FRACTION * 100:.0f}% × isample={self.isample}). "
                f"Worst: subject {worst_id} ESS={worst_ess:.1f}. "
                "Consider increasing isample or tightening the proposal covariance.",
                WarningSeverity.WARNING,
            )

        res.check_omega_conditioning()
        res.compute_shrinkage()
        return res

    def _build_diagnostics(self, result: Any, final_ofv: float) -> dict[str, Any]:
        """Summarize optimizer stop conditions and final IMP sampling quality."""
        ess_values = np.asarray(list(self._last_ess_by_subject.values()), dtype=float)
        ess_threshold = self.ESS_WARN_FRACTION * self.isample
        optimizer_message = str(getattr(result, "message", ""))
        iterations = int(getattr(result, "nit", 0) or 0)
        function_evals = int(getattr(result, "nfev", 0) or 0)
        maxeval_reached = (
            (not bool(getattr(result, "success", False)) and "ITERATIONS REACHED LIMIT" in optimizer_message.upper())
            or iterations >= self.maxeval
        )
        diagnostics: dict[str, Any] = {
            "optimizer": {
                "method": "L-BFGS-B",
                "success": bool(getattr(result, "success", False)),
                "status": int(getattr(result, "status", 0) or 0),
                "message": optimizer_message,
                "iterations": iterations,
                "function_evals": function_evals,
                "maxeval": int(self.maxeval),
                "maxeval_reached": bool(maxeval_reached),
            },
            "objective": {
                "initial_ofv": float(self._ofv_history[0]) if self._ofv_history else None,
                "final_ofv": float(final_ofv),
                "delta_ofv": (
                    float(self._ofv_history[0] - final_ofv) if self._ofv_history else None
                ),
                "history_length": len(self._ofv_history),
            },
            "importance_sampling": {
                "isample": int(self.isample),
                "ess_warning_threshold": float(ess_threshold),
                "final_eval_ess_by_subject": {
                    int(sid): float(ess) for sid, ess in sorted(self._last_ess_by_subject.items())
                },
                "final_eval_min_ess": float(np.min(ess_values)) if ess_values.size else None,
                "final_eval_median_ess": float(np.median(ess_values)) if ess_values.size else None,
                "final_eval_mean_ess": float(np.mean(ess_values)) if ess_values.size else None,
                "final_eval_n_below_warn_threshold": int(np.sum(ess_values < ess_threshold)),
            },
        }
        if self._warm_start_diagnostics:
            diagnostics["warm_start"] = dict(self._warm_start_diagnostics)
        return diagnostics

    def _warm_start_with_focei(
        self,
        population_model: Any,
        init_params: ParameterSet,
    ) -> ParameterSet:
        """Use a short FOCEI pass to seed IMPMAP in a better population basin."""
        warm_maxeval = max(10, min(50, self.maxeval * 4))
        focei = FOCEMethod(
            interaction=True,
            maxeval=warm_maxeval,
            n_parallel=self.n_parallel,
        )
        try:
            warm = focei.estimate(population_model, init_params)
        except Exception as exc:
            self._warm_start_diagnostics = {
                "enabled": True,
                "method": "FOCEI",
                "attempted": True,
                "used": False,
                "maxeval": int(warm_maxeval),
                "error": str(exc),
            }
            logger.warning(f"IMPMAP warm start failed; falling back to direct IMP: {exc}")
            return init_params

        self._warm_start_diagnostics = {
            "enabled": True,
            "method": "FOCEI",
            "attempted": True,
            "used": True,
            "maxeval": int(warm_maxeval),
            "converged": bool(warm.converged),
            "message": str(getattr(warm, "message", "")),
            "ofv": float(getattr(warm, "ofv", np.nan)),
        }
        return ParameterSet(
            theta=np.asarray(warm.theta_final, dtype=float),
            omega=np.asarray(warm.omega_final, dtype=float),
            sigma=np.asarray(warm.sigma_final, dtype=float),
            theta_specs=init_params.theta_specs,
            omega_specs=init_params.omega_specs,
            sigma_specs=init_params.sigma_specs,
        ).apply_bounds()

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
        self._ensure_subject_seeds(population_model)
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

    def _ensure_subject_seeds(self, population_model: Any) -> None:
        """Initialize deterministic per-subject seeds once per subject set."""
        subject_ids = tuple(population_model.subject_ids())
        if set(self._subj_seeds) == set(subject_ids):
            return

        def _child_seed() -> int:
            try:
                return int(self.rng.integers(0, 2**31))
            except AttributeError:
                return 42

        self._subj_seeds = {sid: _child_seed() for sid in subject_ids}

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
        cache_key = (subj_id, self._proposal_cache_key(params))
        with self._proposal_cache_lock:
            cached_proposal = self._proposal_cache.get(cache_key)
            eta0 = np.asarray(
                self._proposal_warm_start.get(subj_id, np.zeros(n_eta, dtype=float)),
                dtype=float,
            ).copy()

        if cached_proposal is not None:
            eta_map = np.asarray(cached_proposal[0], dtype=float).copy()
            V_prop = np.asarray(cached_proposal[1], dtype=float).copy()
        else:
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

            # Use analytical gradient when available: single ODE solve per
            # L-BFGS-B step instead of n_eta+1 finite-difference probes.
            # Probe once to confirm the return type is a proper (float, ndarray)
            # pair — guards against MagicMock or partially-implemented kernels.
            _sg = getattr(indiv, "supports_eta_objective_gradient", None)
            _vg = getattr(indiv, "eta_objective_value_grad", None)
            _native_grad = False
            if callable(_sg) and _sg(population_model.trans) and callable(_vg):
                try:
                    _p = _vg(eta0, params.theta, params.omega, params.sigma,
                             trans=population_model.trans)
                    if isinstance(_p, tuple) and len(_p) == 2:
                        _native_grad = True
                except Exception:
                    pass

            if _native_grad:
                def neg_log_joint_with_grad(
                    eta: np.ndarray,
                ) -> tuple[float, np.ndarray]:
                    val, g = indiv.eta_objective_value_grad(
                        eta,
                        params.theta,
                        params.omega,
                        params.sigma,
                        trans=population_model.trans,
                    )
                    return float(val), np.asarray(g, dtype=float)

                map_result = minimize(
                    neg_log_joint_with_grad,
                    eta0,
                    method="L-BFGS-B",
                    jac=True,
                    options={"maxiter": 100, "ftol": 1e-8},
                )
            else:
                map_result = minimize(
                    neg_log_joint,
                    eta0,
                    method="L-BFGS-B",
                    options={"maxiter": 100, "ftol": 1e-8},
                )
            eta_map = np.asarray(map_result.x, dtype=float)

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

            with self._proposal_cache_lock:
                self._proposal_cache[cache_key] = (
                    eta_map.copy(),
                    np.asarray(V_prop, dtype=float).copy(),
                )
                self._proposal_warm_start[subj_id] = eta_map.copy()

        # Draw samples from a deterministic per-subject stream so that the
        # same parameter vector sees the same Monte Carlo noise across outer
        # objective evaluations.
        rng = np.random.default_rng(self._subj_seeds.get(subj_id, 42))
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
                # log p(eta_s | Omega) — prior (full multivariate-normal formula)
                n_eta = len(eta_s)
                log_prior = (
                    -0.5 * float(eta_s @ omega_inv @ eta_s)
                    - 0.5 * log_det_omega
                    - 0.5 * n_eta * math.log(2.0 * math.pi)
                )
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

        # ESS monitoring — store for WARN_006 emission in estimate()
        w = np.exp(np.clip(log_weights - np.max(log_weights), -50, 0))
        w_sum = float(w.sum())
        if w_sum > 0:
            w_norm = w / w_sum
            ess = 1.0 / float(np.sum(w_norm**2))
        else:
            ess = 0.0
        threshold = self.ESS_WARN_FRACTION * self.isample
        self._last_ess_by_subject[subj_id] = float(ess)
        if ess < threshold:
            logger.debug(f"  Subject {subj_id}: low ESS={ess:.1f} (threshold={threshold:.0f})")
            # Thread-safe append — Python list.append is GIL-protected
            self._low_ess_subjects.append((subj_id, ess))

        return float(log_marg)

    @staticmethod
    def _proposal_cache_key(params: Any) -> bytes:
        to_vector = getattr(params, "to_vector", None)
        if callable(to_vector):
            return np.asarray(to_vector(), dtype=float).tobytes()
        theta = np.asarray(getattr(params, "theta"), dtype=float)
        omega = np.asarray(getattr(params, "omega"), dtype=float)
        sigma = np.asarray(getattr(params, "sigma"), dtype=float)
        return b"|".join((theta.tobytes(), omega.tobytes(), sigma.tobytes()))

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

            # Use analytical gradient when available.
            # Probe once to validate the return type before committing.
            _sg = getattr(indiv, "supports_eta_objective_gradient", None)
            _vg = getattr(indiv, "eta_objective_value_grad", None)
            _eta0 = np.zeros(params.n_eta())
            _native_grad = False
            if callable(_sg) and _sg(population_model.trans) and callable(_vg):
                try:
                    _p = _vg(_eta0, params.theta, params.omega, params.sigma,
                             trans=population_model.trans)
                    if isinstance(_p, tuple) and len(_p) == 2:
                        _native_grad = True
                except Exception:
                    pass

            if _native_grad:
                def obj_with_grad(
                    eta: np.ndarray, _indiv=indiv
                ) -> tuple[float, np.ndarray]:
                    val, g = _indiv.eta_objective_value_grad(
                        eta,
                        params.theta,
                        params.omega,
                        params.sigma,
                        trans=population_model.trans,
                    )
                    return float(val), np.asarray(g, dtype=float)

                res = minimize(
                    obj_with_grad,
                    np.zeros(params.n_eta()),
                    method="L-BFGS-B",
                    jac=True,
                    options={"maxiter": 100},
                )
            else:
                res = minimize(
                    obj, np.zeros(params.n_eta()), method="L-BFGS-B", options={"maxiter": 100}
                )
            eta_hat[sid] = res.x
        return eta_hat
