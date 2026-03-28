"""
FOCE / FOCEI estimation method.

Algorithm:
  Inner loop (per subject, parallelizable):
    η̂_i = argmin_η { -2 log p(y_i|η,θ) + η^T Ω^{-1} η }
    Optimizer: L-BFGS-B via scipy
    Gradient: numerical FD (Stage 1) or JAX (Stage 2)

  Outer loop (population parameters):
    OFV_FOCE = Σ_i [ log|C_i(η̂_i)| + (y-f-Rη̂)^T C_i^{-1} (y-f-Rη̂) + η̂^T Ω^{-1} η̂ ]
    Optimizer: L-BFGS-B via scipy

  FOCEI:
    C_i evaluated at η̂_i (interaction between random effects and residual variance).
    Captures proportional error models correctly.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.optimize import minimize

from openpkpd.estimation.base import EstimationMethod, EstimationResult
from openpkpd.math.matrix import repair_pd
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.constants import LOG2PI, Method
from openpkpd.utils.logging import get_logger

logger = get_logger("estimation.foce")


class _CachedObjEtaEvaluator:
    def __init__(
        self,
        indiv: Any,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int,
    ) -> None:
        self.indiv = indiv
        self.theta = theta
        self.omega = omega
        self.sigma = sigma
        self.trans = trans
        self.cache: dict[bytes, float] = {}
        self.grad_cache: dict[bytes, np.ndarray] = {}
        supports_gradient = getattr(indiv, "supports_eta_objective_gradient", None)
        self._native_value_grad = None
        if callable(supports_gradient) and bool(supports_gradient(trans=trans)):
            value_grad = getattr(indiv, "eta_objective_value_grad", None)
            if callable(value_grad):
                self._native_value_grad = value_grad

    @property
    def has_native_gradient(self) -> bool:
        return self._native_value_grad is not None

    @property
    def has_symbolic_gradient(self) -> bool:
        return self.has_native_gradient

    def _evaluate_with_cache(self, eta: np.ndarray) -> tuple[float, np.ndarray | None]:
        eta_arr = np.asarray(eta, dtype=float)
        key = eta_arr.tobytes()
        cached = self.cache.get(key)
        if cached is not None:
            return cached, self.grad_cache.get(key)
        if self._native_value_grad is not None:
            value, grad = self._native_value_grad(
                eta_arr, self.theta, self.omega, self.sigma, trans=self.trans
            )
            value_f = float(value)
            grad_arr = np.asarray(grad, dtype=float)
            self.cache[key] = value_f
            self.grad_cache[key] = grad_arr
            return value_f, grad_arr
        try:
            value = float(
                self.indiv.obj_eta(eta_arr, self.theta, self.omega, self.sigma, trans=self.trans)
            )
        except Exception:
            value = 1e10
        self.cache[key] = value
        return value, None

    def __call__(self, eta: np.ndarray) -> float:
        value, _grad = self._evaluate_with_cache(eta)
        return value

    def gradient(self, eta: np.ndarray) -> np.ndarray:
        value, grad = self._evaluate_with_cache(eta)
        if grad is None:
            raise RuntimeError(f"native gradient unavailable for eta objective at value {value}")
        return grad

    def evaluate_many(self, eta_batch: np.ndarray) -> np.ndarray:
        eta_arr = np.asarray(eta_batch, dtype=float)
        if eta_arr.ndim == 1:
            eta_arr = eta_arr[None, :]
        if len(eta_arr) == 0:
            return np.array([], dtype=float)

        values = np.empty(len(eta_arr), dtype=float)
        missing_indices: list[int] = []
        missing_rows: list[np.ndarray] = []

        for i, eta in enumerate(eta_arr):
            cached = self.cache.get(eta.tobytes())
            if cached is None:
                missing_indices.append(i)
                missing_rows.append(eta.copy())
            else:
                values[i] = cached

        if missing_rows:
            missing_batch = np.asarray(missing_rows, dtype=float)
            obj_eta_many = getattr(self.indiv, "obj_eta_many", None)
            try:
                if obj_eta_many is not None:
                    missing_values = np.asarray(
                        obj_eta_many(
                            missing_batch, self.theta, self.omega, self.sigma, trans=self.trans
                        ),
                        dtype=float,
                    )
                else:
                    missing_values = np.asarray([self(eta) for eta in missing_batch], dtype=float)
            except Exception:
                missing_values = np.full(len(missing_batch), 1e10, dtype=float)

            for idx, eta, value in zip(
                missing_indices, missing_batch, missing_values, strict=False
            ):
                val = float(value)
                self.cache[eta.tobytes()] = val
                values[idx] = val

        return values

    def workers(self, _fun: Any, iterable: Any) -> list[float]:
        eta_points = [np.asarray(point, dtype=float) for point in iterable]
        if not eta_points:
            return []
        return self.evaluate_many(np.asarray(eta_points, dtype=float)).tolist()


def _make_cached_obj_eta(
    indiv: Any,
    theta: np.ndarray,
    omega: np.ndarray,
    sigma: np.ndarray,
    trans: int,
):
    return _CachedObjEtaEvaluator(indiv, theta, omega, sigma, trans)


def _worker_optimize_eta(
    subject_id: int,
    indiv: Any,
    eta0: np.ndarray,
    theta: np.ndarray,
    omega: np.ndarray,
    sigma: np.ndarray,
    trans: int,
    maxiter: int,
) -> tuple[int, np.ndarray]:
    """
    Module-level worker for ProcessPoolExecutor.

    Receives a picklable IndividualModel (compiled callables recompile lazily
    in the worker process from their stored source strings), runs L-BFGS-B, and
    returns (subject_id, eta_hat).
    """
    obj_eta = _make_cached_obj_eta(indiv, theta, omega, sigma, trans)

    result = minimize(
        obj_eta,
        eta0,
        method="L-BFGS-B",
        jac=obj_eta.gradient if obj_eta.has_native_gradient else "2-point",
        options=(
            {"maxiter": maxiter, "ftol": 1e-10, "eps": 1e-5}
            if obj_eta.has_native_gradient
            else {"maxiter": maxiter, "ftol": 1e-10, "eps": 1e-5, "workers": obj_eta.workers}
        ),
    )
    return subject_id, result.x


def _compute_G_i(
    indiv: Any,
    theta: np.ndarray,
    eta: np.ndarray,
    sigma: np.ndarray,
    trans: int,
    obs_mask: np.ndarray,
    pred0_obs: np.ndarray,
    h: float = 1e-4,
) -> np.ndarray:
    """
    Compute the sensitivity matrix G_i = ∂pred/∂η at η̂_i via forward differences.

    Returns an (n_obs, n_eta) matrix.
    """
    n_eta = len(eta)
    n_obs = len(pred0_obs)
    G = np.zeros((n_obs, n_eta))
    for k in range(n_eta):
        eta_p = eta.copy()
        eta_p[k] += h
        try:
            _, _, _, pred_p, _ = indiv.evaluate_observation_model(theta, eta_p, sigma, trans=trans)
            G[:, k] = (pred_p[obs_mask] - pred0_obs) / h
        except Exception:
            pass  # column stays zero
    return G


class FOCEMethod(EstimationMethod):
    """
    FOCE / FOCEI estimation method.

    Args:
        interaction:    If True, use FOCEI (evaluate C_i at η̂_i, not η=0).
        maxeval:        Maximum outer-loop evaluations per optimization start.
        inner_maxiter:  Maximum inner-loop iterations per subject.
        n_parallel:     Number of parallel processes for inner loop (0=auto).
        sigdig:         Convergence criterion (significant digits).
        gtol:           Gradient norm convergence tolerance for the outer
                        L-BFGS-B optimizer.  Tighten (e.g. 1e-6) for models
                        with weak covariate gradient signal (covariate-rich
                        models, power-law exponents near 0).
        n_starts:       Number of independent optimization starts.  When > 1
                        the optimizer is restarted from ``n_starts - 1``
                        additional random perturbations of the initial values
                        (in the transformed parameter space) and the run with
                        the lowest OFV is returned.  Useful for multi-modal
                        likelihoods (e.g. two-compartment models with V2/Q
                        label-swap local minima).
        perturbation_scale: Standard deviation of Gaussian perturbations
                        applied to the transformed parameter vector for each
                        extra start.  The transformed vector is on a log or
                        logit scale, so a value of 1.0 corresponds roughly
                        to a factor-of-e perturbation in the natural scale.
        seed:           Random seed for perturbation reproducibility.
    """

    method_name = Method.FOCE

    def __init__(
        self,
        interaction: bool = False,
        maxeval: int = 9999,
        inner_maxiter: int = 200,
        n_parallel: int = 1,
        sigdig: int = 3,
        print_interval: int = 5,
        noabort: bool = False,
        gtol: float = 1e-5,
        n_starts: int = 1,
        perturbation_scale: float = 1.0,
        seed: int | None = None,
        outer_optimizer: str = "L-BFGS-B",
        outer_fallback_optimizer: str | None = None,
        outer_fallback_maxeval: int = 40,
        retain_best_iterate: bool = True,
        retry_on_abnormal: bool | None = None,
        retry_omega_scales: tuple[float, ...] = (),
    ) -> None:
        self.interaction = interaction
        self.maxeval = maxeval
        self.inner_maxiter = inner_maxiter
        self.n_parallel = n_parallel
        self.sigdig = sigdig
        self.print_interval = print_interval
        self.noabort = noabort
        self.gtol = gtol
        self.n_starts = n_starts
        self.perturbation_scale = perturbation_scale
        self.seed = seed
        self.outer_optimizer = outer_optimizer
        self.outer_fallback_optimizer = outer_fallback_optimizer
        self.outer_fallback_maxeval = outer_fallback_maxeval
        self.retain_best_iterate = retain_best_iterate
        self.retry_on_abnormal = retry_on_abnormal
        self.retry_omega_scales = retry_omega_scales
        if interaction:
            self.method_name = Method.FOCEI
            if self.outer_fallback_optimizer is None:
                # FOCEI objectives can be locally rough even when L-BFGS-B reports
                # convergence. A short derivative-free polish can recover a better basin.
                self.outer_fallback_optimizer = "Powell"
            if self.retry_on_abnormal is None:
                self.retry_on_abnormal = True
            if not self.retry_omega_scales:
                self.retry_omega_scales = (0.5, 0.25, 0.1)
        elif self.retry_on_abnormal is None:
            self.retry_on_abnormal = False
        self._iter = 0
        self._ofv_history: list[float] = []
        self._current_eta_hat: dict[int, np.ndarray] = {}
        self._inner_loop_pool: ProcessPoolExecutor | None = None
        self._best_outer_x: np.ndarray | None = None
        self._best_outer_ofv: float = np.inf

    def estimate(
        self,
        population_model: Any,
        init_params: ParameterSet,
        **kwargs: Any,
    ) -> EstimationResult:
        t0 = time.time()
        n_starts = max(1, self.n_starts)
        logger.info(
            f"Starting {self.method_name} estimation, "
            f"interaction={self.interaction}, "
            f"n_subjects={population_model.n_subjects()}, "
            f"n_starts={n_starts}"
        )

        params = init_params.apply_bounds()
        x0 = params.to_vector()

        try:
            if self.n_parallel != 1 and population_model.n_subjects() > 1:
                self._create_inner_loop_pool()

            if n_starts == 1:
                result, final_params, final_eta_hat, final_ofv = self._run_single(
                    x0, init_params, population_model
                )
                best_result, best_params, best_eta, best_ofv = (
                    result,
                    final_params,
                    final_eta_hat,
                    final_ofv,
                )
                all_ofv_history = list(self._ofv_history)
            else:
                best_result, best_params, best_eta, best_ofv = None, None, None, np.inf
                all_ofv_history_ms: list[float] = []
                rng = np.random.default_rng(self.seed)
                bounds = params.get_optimizer_bounds()

                for start_idx in range(n_starts):
                    if start_idx == 0:
                        x0_i = x0.copy()
                    else:
                        x0_i = x0 + rng.normal(0.0, self.perturbation_scale, len(x0))
                        # Clamp into bounds (optimizer bounds are in the raw transformed space)
                        for k, (lo, hi) in enumerate(bounds):
                            if lo is not None:
                                x0_i[k] = max(x0_i[k], lo + 1e-8)
                            if hi is not None:
                                x0_i[k] = min(x0_i[k], hi - 1e-8)

                    logger.info(f"  Multi-start {start_idx + 1}/{n_starts}")
                    self._reset_state(init_params, population_model)
                    result_i, params_i, eta_i, ofv_i = self._run_single(
                        x0_i, init_params, population_model
                    )
                    all_ofv_history_ms.extend(self._ofv_history)
                    logger.info(f"  Start {start_idx + 1} OFV={ofv_i:.4f}")

                    if ofv_i < best_ofv:
                        best_result, best_params, best_eta, best_ofv = (
                            result_i,
                            params_i,
                            eta_i,
                            ofv_i,
                        )
                all_ofv_history = all_ofv_history_ms

            result = best_result
            assert best_params is not None, "Optimization produced no parameters"
            assert best_eta is not None, "Optimization produced no eta"
            final_params = best_params
            final_eta_hat = best_eta
            final_ofv = best_ofv
        finally:
            self._shutdown_inner_loop_pool()

        elapsed = time.time() - t0
        converged = result.success
        logger.info(
            f"{self.method_name} completed in {elapsed:.1f}s, "
            f"OFV={final_ofv:.4f}, converged={converged}"
        )

        res = EstimationResult(
            theta_final=final_params.theta,
            omega_final=final_params.omega,
            sigma_final=final_params.sigma,
            ofv=final_ofv,
            converged=converged,
            post_hoc_etas=final_eta_hat,
            ofv_history=all_ofv_history,
            n_function_evals=self._iter,
            elapsed_time=elapsed,
            method=self.method_name,
            message=getattr(result, "message", ""),
        )
        res.compute_shrinkage()
        return res

    def _reset_state(
        self,
        init_params: ParameterSet,
        population_model: Any,
    ) -> None:
        """Reset per-run mutable state before each multi-start attempt."""
        self._iter = 0
        self._ofv_history = []
        self._current_eta_hat = {
            sid: np.zeros(init_params.n_eta()) for sid in population_model.subject_ids()
        }
        self._best_outer_x = None
        self._best_outer_ofv = np.inf

    def _run_single(
        self,
        x0: np.ndarray,
        init_params: ParameterSet,
        population_model: Any,
        *,
        allow_structured_retries: bool = True,
    ) -> tuple[Any, ParameterSet, dict, float]:
        """
        Run one L-BFGS-B optimization from starting point x0.

        Returns (scipy_result, final_params, final_eta_hat, final_ofv).
        """
        self._reset_state(init_params, population_model)

        def objective(x: np.ndarray) -> float:
            p = ParameterSet.from_vector(x, init_params).apply_bounds()
            eta_hat = self._inner_loop(population_model, p)
            self._current_eta_hat = eta_hat
            ofv = self._outer_ofv(population_model, p, eta_hat)
            self._iter += 1
            self._ofv_history.append(ofv)
            if ofv < self._best_outer_ofv:
                self._best_outer_ofv = ofv
                self._best_outer_x = np.asarray(x, dtype=float).copy()
            if self._iter % self.print_interval == 0:
                logger.info(f"  Iter {self._iter:5d}  OFV={ofv:.4f}")
            return ofv

        params = init_params.apply_bounds()
        result = self._run_outer_optimizer(
            objective,
            x0,
            params.get_optimizer_bounds(),
            optimizer=self.outer_optimizer,
            maxeval=self.maxeval,
        )
        final_params = ParameterSet.from_vector(result.x, init_params).apply_bounds()
        final_eta_hat = self._inner_loop(population_model, final_params)
        final_ofv = self._outer_ofv(population_model, final_params, final_eta_hat)
        result, final_params, final_eta_hat, final_ofv = self._maybe_promote_best_iterate(
            result, init_params, population_model, final_params, final_eta_hat, final_ofv
        )

        fallback_optimizer = self.outer_fallback_optimizer
        if fallback_optimizer is not None and fallback_optimizer != self.outer_optimizer:
            fallback_result = self._run_outer_optimizer(
                objective,
                result.x,
                params.get_optimizer_bounds(),
                optimizer=fallback_optimizer,
                maxeval=self.outer_fallback_maxeval,
            )
            fallback_params = ParameterSet.from_vector(fallback_result.x, init_params).apply_bounds()
            fallback_eta_hat = self._inner_loop(population_model, fallback_params)
            fallback_ofv = self._outer_ofv(population_model, fallback_params, fallback_eta_hat)
            if fallback_ofv < final_ofv:
                logger.info(
                    "  Fallback outer optimizer %s improved OFV %.4f -> %.4f",
                    fallback_optimizer,
                    final_ofv,
                    fallback_ofv,
                )
                result = fallback_result
                final_params = fallback_params
                final_eta_hat = fallback_eta_hat
                final_ofv = fallback_ofv
                result, final_params, final_eta_hat, final_ofv = self._maybe_promote_best_iterate(
                    result,
                    init_params,
                    population_model,
                    final_params,
                    final_eta_hat,
                    final_ofv,
                )

        if allow_structured_retries and self._should_run_structured_retries(result):
            for retry_idx, retry_x0 in enumerate(
                self._structured_retry_vectors(init_params, final_params),
                start=1,
            ):
                logger.info("  Structured retry %d/%d", retry_idx, len(self.retry_omega_scales))
                self._reset_state(init_params, population_model)
                retry_result, retry_params, retry_eta_hat, retry_ofv = self._run_single(
                    retry_x0,
                    init_params,
                    population_model,
                    allow_structured_retries=False,
                )
                if retry_ofv < final_ofv:
                    logger.info(
                        "  Structured retry improved OFV %.4f -> %.4f",
                        final_ofv,
                        retry_ofv,
                    )
                    result = retry_result
                    final_params = retry_params
                    final_eta_hat = retry_eta_hat
                    final_ofv = retry_ofv
        return result, final_params, final_eta_hat, final_ofv

    def _maybe_promote_best_iterate(
        self,
        result: Any,
        init_params: ParameterSet,
        population_model: Any,
        final_params: ParameterSet,
        final_eta_hat: dict[int, np.ndarray],
        final_ofv: float,
    ) -> tuple[Any, ParameterSet, dict[int, np.ndarray], float]:
        if not self.retain_best_iterate or self._best_outer_x is None:
            return result, final_params, final_eta_hat, final_ofv
        if self._best_outer_ofv >= final_ofv:
            return result, final_params, final_eta_hat, final_ofv
        best_params = ParameterSet.from_vector(self._best_outer_x, init_params).apply_bounds()
        best_eta_hat = self._inner_loop(population_model, best_params)
        best_ofv = self._outer_ofv(population_model, best_params, best_eta_hat)
        if best_ofv < final_ofv:
            logger.info("  Retaining best iterate OFV %.4f -> %.4f", final_ofv, best_ofv)
            result = SimpleNamespace(
                x=self._best_outer_x.copy(),
                success=getattr(result, "success", False),
                message=f"{getattr(result, 'message', '')} [best-iterate]".strip(),
            )
            return result, best_params, best_eta_hat, best_ofv
        return result, final_params, final_eta_hat, final_ofv

    def _should_run_structured_retries(self, result: Any) -> bool:
        if not self.interaction or not self.retry_on_abnormal or not self.retry_omega_scales:
            return False
        message = str(getattr(result, "message", "") or "").upper()
        return (not bool(getattr(result, "success", False))) or ("ABNORMAL" in message)

    def _structured_retry_vectors(
        self,
        init_params: ParameterSet,
        final_params: ParameterSet,
    ) -> list[np.ndarray]:
        base_theta = np.asarray(final_params.theta, dtype=float)
        base_sigma = np.asarray(final_params.sigma, dtype=float)
        omega_diag = np.diag(final_params.omega).astype(float, copy=True)
        retry_vectors: list[np.ndarray] = []
        for scale in self.retry_omega_scales:
            scaled_diag = np.maximum(omega_diag * float(scale), 1e-8)
            retry_params = ParameterSet(
                theta=base_theta.copy(),
                omega=np.diag(scaled_diag),
                sigma=base_sigma.copy(),
                theta_specs=init_params.theta_specs,
                omega_specs=init_params.omega_specs,
                sigma_specs=init_params.sigma_specs,
            ).apply_bounds()
            retry_vectors.append(retry_params.to_vector())
        return retry_vectors

    def _run_outer_optimizer(
        self,
        objective: Any,
        x0: np.ndarray,
        bounds: list[tuple[float | None, float | None]],
        *,
        optimizer: str,
        maxeval: int,
    ) -> Any:
        method_key = str(optimizer).strip().upper()
        if method_key == "L-BFGS-B":
            method = "L-BFGS-B"
        elif method_key == "POWELL":
            method = "Powell"
        else:
            raise ValueError(f"Unsupported FOCE outer optimizer: {optimizer}")
        options: dict[str, Any]
        if method == "L-BFGS-B":
            options = {"maxiter": maxeval, "ftol": 1e-9, "gtol": self.gtol}
        elif method == "Powell":
            options = {"maxiter": maxeval, "xtol": 1e-3, "ftol": 1e-3}
        return minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            options=options,
        )

    def _inner_loop(
        self,
        population_model: Any,
        params: ParameterSet,
    ) -> dict[int, np.ndarray]:
        """
        Optimize η_i for each subject using L-BFGS-B.

        Returns eta_hat mapping subject_id → η̂_i.

        When n_parallel > 1, subjects are dispatched to a ProcessPoolExecutor.
        Compiled callables are picklable (they store source and recompile lazily
        in the worker), so this achieves true multi-core parallelism with no GIL.
        """
        subject_ids = population_model.subject_ids()
        n_eta = params.n_eta()

        # Serial path (default)
        if self.n_parallel == 1 or len(subject_ids) <= 1:
            eta_hat: dict[int, np.ndarray] = {}
            for sid in subject_ids:
                indiv = population_model.individual_model(sid)
                eta0 = self._current_eta_hat.get(sid, np.zeros(n_eta))
                obj_eta = _make_cached_obj_eta(
                    indiv,
                    params.theta,
                    params.omega,
                    params.sigma,
                    population_model.trans,
                )

                res = minimize(
                    obj_eta,
                    eta0,
                    method="L-BFGS-B",
                    jac=obj_eta.gradient if obj_eta.has_native_gradient else "2-point",
                    options=(
                        {"maxiter": self.inner_maxiter, "ftol": 1e-10, "eps": 1e-5}
                        if obj_eta.has_native_gradient
                        else {
                            "maxiter": self.inner_maxiter,
                            "ftol": 1e-10,
                            "eps": 1e-5,
                            "workers": obj_eta.workers,
                        }
                    ),
                )
                eta_hat[sid] = res.x
            return eta_hat

        # Parallel path — ProcessPoolExecutor for true multi-core (GIL-free)
        pool = self._inner_loop_pool
        created_pool = False
        if pool is None:
            pool = self._create_inner_loop_pool()
            created_pool = True
        futures = {}
        try:
            for sid in subject_ids:
                indiv = population_model.individual_model(sid)
                eta0 = self._current_eta_hat.get(sid, np.zeros(n_eta))
                fut = pool.submit(
                    _worker_optimize_eta,
                    sid,
                    indiv,
                    eta0,
                    params.theta,
                    params.omega,
                    params.sigma,
                    population_model.trans,
                    self.inner_maxiter,
                )
                futures[fut] = sid

            eta_hat = {}
            for future in as_completed(futures):
                sid, eta = future.result()
                eta_hat[sid] = eta
            return eta_hat
        finally:
            if created_pool:
                self._shutdown_inner_loop_pool()

    def _create_inner_loop_pool(self) -> ProcessPoolExecutor:
        n_workers = self.n_parallel if self.n_parallel > 0 else None
        pool = ProcessPoolExecutor(max_workers=n_workers)
        self._inner_loop_pool = pool
        return pool

    def _shutdown_inner_loop_pool(self) -> None:
        if self._inner_loop_pool is not None:
            self._inner_loop_pool.shutdown()
            self._inner_loop_pool = None

    def _outer_ofv(
        self,
        population_model: Any,
        params: ParameterSet,
        eta_hat: dict[int, np.ndarray],
    ) -> float:
        """
        Compute outer FOCE/FOCEI OFV given current η̂.

        NONMEM FOCEI convention (Beal-Sheiner first-order conditional):

          OFV_i = (y_i - f_i)^T C_i^{-1}(y_i - f_i) + log|C_i|
                + η̂_i^T Ω^{-1} η̂_i + log|Ω|
                + (n_i + p)·log(2π)

        where C_i = G_i Ω G_i^T + R_i, G_i = ∂f/∂η|_{η̂_i} (FOCEI),
        or C_i = R_i for FOCE (no interaction).

        C_i^{-1} and log|C_i| are computed via the Woodbury identity /
        matrix determinant lemma so only an (n_eta × n_eta) system is solved:

          M_i = Ω^{-1} + G_i^T R_i^{-1} G_i          (n_eta × n_eta)
          log|C_i| = log|R_i| + log|Ω| + log|M_i|
          r^T C_i^{-1} r = r^T R_i^{-1} r - v^T M_i^{-1} v
                           where v = G_i^T R_i^{-1} r
        """
        ofv = 0.0
        n_eta = params.n_eta()
        try:
            omega_rep = repair_pd(params.omega)
            omega_inv = np.linalg.inv(omega_rep)
            _sign, log_det_omega = np.linalg.slogdet(omega_rep)
            log_det_omega = float(log_det_omega) if _sign > 0 else 0.0
        except np.linalg.LinAlgError:
            omega_inv = np.eye(n_eta)
            log_det_omega = 0.0

        for subj_id in population_model.subject_ids():
            eta_i = eta_hat.get(subj_id, np.zeros(n_eta))
            indiv = population_model.individual_model(subj_id)
            subj_ev = indiv.subject_events
            obs_mask = subj_ev.observation_mask()

            if not np.any(obs_mask):
                continue

            try:
                _, _, _, pred, var = indiv.evaluate_observation_model(
                    params.theta, eta_i, params.sigma, trans=population_model.trans
                )
                dv = subj_ev.obs_dv[obs_mask]
                pred_obs = pred[obs_mask]
                var_obs = np.maximum(var[obs_mask], 1e-10)
                n_obs = len(dv)
                residuals = dv - pred_obs

                if self.interaction and n_eta > 0:
                    # FOCEI: build C_i = G_i Ω G_i^T + R_i via sensitivity matrix
                    G = _compute_G_i(
                        indiv,
                        params.theta,
                        eta_i,
                        params.sigma,
                        population_model.trans,
                        obs_mask,
                        pred_obs,
                    )
                    # Woodbury: M = Ω^{-1} + G^T R^{-1} G  (n_eta × n_eta)
                    G_T_Rinv = G.T / var_obs  # (n_eta, n_obs): G^T R^{-1} (R diagonal)
                    M = omega_inv + G_T_Rinv @ G
                    try:
                        _sm, log_det_M = np.linalg.slogdet(M)
                        log_det_M = float(log_det_M) if _sm > 0 else 0.0
                        quad = float(np.sum(residuals**2 / var_obs))
                        log_det_R = float(np.sum(np.log(var_obs)))
                        # log|C_i| = log|R| + log|Ω| + log|M|
                        log_det_ci = log_det_R + log_det_omega + log_det_M
                    except np.linalg.LinAlgError:
                        # Fallback to no-interaction
                        quad = float(np.sum(residuals**2 / var_obs))
                        log_det_ci = float(np.sum(np.log(var_obs)))
                else:
                    # FOCE (no interaction): C_i = R_i (diagonal)
                    quad = float(np.sum(residuals**2 / var_obs))
                    log_det_ci = float(np.sum(np.log(var_obs)))

                eta_penalty = float(eta_i @ omega_inv @ eta_i)
                ofv_i = n_obs * LOG2PI + log_det_ci + quad + eta_penalty
                if self.interaction:
                    # FOCEI is a conditional Laplace approximation over eta.
                    # The reported marginal objective removes the integrated
                    # Gaussian constant per eta dimension.
                    ofv_i -= n_eta * LOG2PI
                else:
                    ofv_i += log_det_omega
                ofv += ofv_i
            except Exception:
                ofv += 1e10

        # A4: add prior penalty if model is PriorAugmentedModel
        if hasattr(population_model, "prior"):
            ofv += population_model.prior.penalty(params.theta, params.omega)

        return ofv
