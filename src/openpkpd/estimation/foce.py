"""
FOCE / FOCEI estimation method.

Algorithm:
  Inner loop (per subject, parallelizable):
    η̂_i = argmin_η { -2 log p(y_i|η,θ) + η^T Ω^{-1} η }
    Optimizer: L-BFGS-B via scipy
    Gradient: numerical FD or native symbolic derivatives when available

  Outer loop (population parameters):
    OFV_FOCE = Σ_i [ log|C_i(η̂_i)| + (y-f-Rη̂)^T C_i^{-1} (y-f-Rη̂) + η̂^T Ω^{-1} η̂ ]
    Optimizer: L-BFGS-B via scipy

  FOCEI:
    C_i evaluated at η̂_i (interaction between random effects and residual variance).
    Captures proportional error models correctly.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.optimize import minimize

from openpkpd.estimation.base import EstimationMethod, EstimationResult
from openpkpd.math.matrix import repair_pd
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.constants import LOG2PI, Method
from openpkpd.utils.errors import WarningCode, WarningSeverity
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
        # Include subject_id in the cache key so that if two subjects ever share
        # an evaluator instance (defensive), byte-identical ETA arrays for
        # different subjects cannot collide.
        # IndividualModel stores subject_id on subject_events.
        _subj_events = getattr(indiv, "subject_events", None)
        self._subject_id = (
            getattr(_subj_events, "subject_id", None)
            if _subj_events is not None
            else getattr(indiv, "subject_id", None)
        )
        self.cache: dict[tuple, float] = {}
        self.grad_cache: dict[tuple, np.ndarray] = {}
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
        # Include subject_id in the key: defensive guard so that byte-identical
        # ETA arrays from different subjects never collide in a shared cache.
        key = (self._subject_id, eta_arr.tobytes())
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
            cached = self.cache.get((self._subject_id, eta.tobytes()))
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
                self.cache[(self._subject_id, eta.tobytes())] = val
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


def _optimize_eta_lbfgsb(
    obj_eta: _CachedObjEtaEvaluator,
    eta0: np.ndarray,
    maxiter: int,
) -> Any:
    """
    Run a single-subject L-BFGS-B optimisation for the inner η loop.

    Shared by the serial inner loop and the process-pool worker so that
    minimiser options (ftol, eps, parallel workers back-end) stay in sync.
    """
    return minimize(
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


def _can_skip_eta_optimization(params: ParameterSet, *, zero_tol: float = 1e-8) -> bool:
    """
    Return True when η optimisation can be skipped safely.

    This fast path is intentionally narrow: it only applies when every OMEGA
    block is fixed and the full OMEGA matrix is effectively zero. In that case,
    the model is behaving as a no-IIV model and repeatedly optimising per-subject
    η adds substantial cost with negligible mathematical benefit.
    """
    if params.n_eta() == 0:
        return True
    omega_specs = getattr(params, "omega_specs", None)
    if not omega_specs:
        return False
    if any(not spec.fixed for spec in omega_specs):
        return False
    omega = np.asarray(params.omega, dtype=float)
    return bool(np.max(np.abs(omega)) <= zero_tol)


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
    result = _optimize_eta_lbfgsb(obj_eta, eta0, maxiter)
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
    Compute the sensitivity matrix G_i = ∂pred/∂η at η̂_i.

    When the individual model's PK subroutine exposes *solve_with_sensitivity*
    (e.g. ADVAN13 with forward sensitivity equations), the computation is split:
      1. ``solve_with_sensitivity`` gives ∂amounts/∂pk_params (n_t × n_cmt × n_p).
      2. Finite-difference on the *cheap* pk_callable gives ∂pk_params/∂η.
      3. The observation-model Jacobian ∂pred/∂amounts is obtained via one FD
         per ETA dimension, but on the *observation model only* (no ODE re-solve).

    Fallback: standard forward-difference on *evaluate_observation_model* —
    requires one full model evaluation (including ODE) per ETA dimension.

    Returns an (n_obs, n_eta) matrix.
    """
    n_eta = len(eta)
    n_obs = len(pred0_obs)
    G = np.zeros((n_obs, n_eta))

    # ── Try native ADVAN6 CVODES sensitivity path ─────────────────────────────
    # One Rust sensitivity solve gives ∂IPRED/∂η analytically for all η
    # simultaneously, replacing n_eta full ODE probes in the FD fallback.
    # Only activates for non-mixed ADVAN6 models with a valid native contract.
    native_G_fn = getattr(indiv, "native_advan6_prediction_eta_jacobian", None)
    if native_G_fn is not None:
        try:
            result = native_G_fn(theta, eta, obs_mask, n_eta)
            if result is not None:
                return result
        except Exception as _e:
            logger.warning("native_advan6_prediction_eta_jacobian failed: %s; falling through to FD", _e)

    # ── Try sensitivity-assisted gradient (ADVAN13 path) ─────────────────────
    pk_sub = getattr(indiv, "pk_subroutine", None)
    sws = getattr(pk_sub, "solve_with_sensitivity", None) if pk_sub is not None else None

    if sws is not None:
        try:
            return _compute_G_i_via_sensitivity(
                indiv, pk_sub, theta, eta, sigma, trans, obs_mask, pred0_obs, h
            )
        except Exception as _e:
            logger.warning("sensitivity-assisted G_i failed: %s; falling through to FD", _e)

    # ── Standard forward-difference path ─────────────────────────────────────
    for k in range(n_eta):
        h_k = 1e-5 * max(abs(float(eta[k])), 1.0)
        eta_p = eta.copy()
        eta_p[k] += h_k
        try:
            _, _, _, pred_p, _ = indiv.evaluate_observation_model(theta, eta_p, sigma, trans=trans)
            G[:, k] = (pred_p[obs_mask] - pred0_obs) / h_k
        except Exception as _e:
            logger.warning("G_i FD column %d failed: %s; column set to zero", k, _e)
    return G


def _compute_G_i_via_sensitivity(
    indiv: Any,
    pk_sub: Any,
    theta: np.ndarray,
    eta: np.ndarray,
    sigma: np.ndarray,
    trans: int,
    obs_mask: np.ndarray,
    pred0_obs: np.ndarray,
    h: float = 1e-4,
) -> np.ndarray:
    """
    Sensitivity-assisted G_i using ADVAN's solve_with_sensitivity output.

    Algorithm (chain rule):
      G[:, k] ≈ (∂IPRED/∂amounts) × (∂amounts/∂pk_params) × (∂pk_params/∂η[k])

    Step 1 — get ∂amounts/∂pk_params from pk_sub.solve_with_sensitivity().
              shape: (n_obs, n_cmt, n_pk_params)
    Step 2 — FD on the pk_callable (eta → pk_params) to get ∂pk_params/∂η[k].
              n_eta calls, each much cheaper than a full ODE solve.
    Step 3 — combine to get ∂amounts/∂η[k] = sensitivity @ d_pk_k
              shape: (n_obs, n_cmt)
    Step 4 — FD on the observation model (which maps perturbed amounts → IPRED)
              to get ∂IPRED/∂η[k].  Only the observation model runs; ODE is skipped.

    Raises on any failure so the caller can fall back to plain FD.
    """
    n_eta = len(eta)
    n_obs = len(pred0_obs)
    G = np.zeros((n_obs, n_eta))

    # Nominal sensitivity: ∂amounts/∂pk_params at (theta, eta)
    # returns PKSolution with .sensitivity field of shape (n_t, n_cmt, n_pk_params)
    sol0 = pk_sub.solve_with_sensitivity(theta=theta, eta=eta, trans=trans)
    sensitivity = np.asarray(sol0.sensitivity)  # (n_t, n_cmt, n_pk_params)
    amounts0 = np.asarray(sol0.y)               # (n_t, n_cmt) or (n_t,)

    # Base pk_params at nominal eta
    pk_callable = getattr(indiv, "pk_callable", None)
    if pk_callable is None:
        raise RuntimeError("IndividualModel has no pk_callable attribute")
    pk0 = np.asarray(pk_callable(theta, eta, trans=trans), dtype=float)  # (n_pk_params,)
    n_pk = len(pk0)
    n_t, *_ = sensitivity.shape
    if sensitivity.ndim == 2:
        sensitivity = sensitivity[:, :, np.newaxis]  # (n_t, 1, n_pk) → broadcast

    for k in range(n_eta):
        eta_p = eta.copy()
        eta_p[k] += h
        # ∂pk_params/∂η[k] — cheap, no ODE
        pk_p = np.asarray(pk_callable(theta, eta_p, trans=trans), dtype=float)
        d_pk_k = (pk_p - pk0) / h  # (n_pk_params,)

        # ∂amounts/∂η[k] = sensitivity @ d_pk_k  (n_t, n_cmt)
        if sensitivity.ndim == 3:
            d_amounts_k = sensitivity @ d_pk_k  # (n_t, n_cmt)
        else:
            d_amounts_k = sensitivity * d_pk_k[0]

        # ∂IPRED/∂η[k]: perturb amounts by d_amounts_k * h and use FD
        perturbed_amounts = amounts0 + d_amounts_k * h
        obs_from_amounts = getattr(indiv, "_obs_from_amounts", None)
        if obs_from_amounts is not None:
            raw = obs_from_amounts(theta, eta_p, sigma, trans=trans, amounts=perturbed_amounts)
        else:
            raw = indiv.evaluate_observation_model(
                theta, eta_p, sigma, trans=trans, _amounts=perturbed_amounts
            )
        # evaluate_observation_model returns (amounts, obs_mask, dv, pred, var)
        if isinstance(raw, tuple):
            pred_p = np.asarray(raw[3])
        else:
            pred_p = np.asarray(raw)
        if pred_p.shape == pred0_obs.shape:
            G[:, k] = (pred_p - pred0_obs) / h
        elif pred_p.size >= obs_mask.sum():
            G[:, k] = (pred_p[obs_mask] - pred0_obs) / h

    return G


def _estimate_gradient_norm(result: Any) -> float | None:
    """
    Extract gradient norm from a scipy OptimizeResult, or return None.

    L-BFGS-B stores the projected gradient in ``result.jac`` (the gradient
    is the *projected* L-infinity norm for the display, but the full gradient
    array is available when ``jac=True`` or ``jac='2-point'`` is used).
    trust-constr stores the actual gradient in ``result.grad``.
    """
    # trust-constr sets result.grad
    grad = getattr(result, "grad", None)
    if grad is not None:
        arr = np.asarray(grad, dtype=float)
        if arr.size > 0:
            return float(np.max(np.abs(arr)))
    # L-BFGS-B sets result.jac when jac is supplied as array
    jac = getattr(result, "jac", None)
    if jac is not None:
        try:
            arr = np.asarray(jac, dtype=float)
            if arr.size > 0:
                return float(np.max(np.abs(arr)))
        except (TypeError, ValueError):
            pass
    return None


def _extract_inverse_hessian(result: Any) -> np.ndarray | None:
    """
    Extract a dense inverse-Hessian approximation from scipy optimizer output.

    L-BFGS-B exposes ``LbfgsInvHessProduct`` via ``result.hess_inv`` with a
    ``todense()`` method. Other optimizers may expose an array-like object
    directly. When no stable dense approximation is available, return ``None``.
    """
    hess_inv = getattr(result, "hess_inv", None)
    if hess_inv is None:
        return None
    try:
        if hasattr(hess_inv, "todense"):
            arr = np.asarray(hess_inv.todense(), dtype=float)
        else:
            arr = np.asarray(hess_inv, dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or not np.all(np.isfinite(arr)):
        return None
    return arr


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
        iteration_callback=None,
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
        self.iteration_callback = iteration_callback
        self._iter = 0
        self._ofv_history: list[float] = []
        self._current_eta_hat: dict[int, np.ndarray] = {}
        self._inner_loop_pool: ProcessPoolExecutor | None = None
        self._outer_eval_cache: dict[
            tuple[float, ...], tuple[float, dict[int, np.ndarray]]
        ] = {}
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
        inverse_hessian = _extract_inverse_hessian(result)
        if inverse_hessian is not None:
            res.diagnostics.setdefault("optimizer", {})["inverse_hessian"] = inverse_hessian

        # ── Structured estimation warnings ────────────────────────────────────
        # Gradient norm at convergence
        grad_norm = _estimate_gradient_norm(result)
        if grad_norm is not None and grad_norm > 1e-2:
            res.add_structured_warning(
                WarningCode.WARN_003,
                f"Gradient norm at convergence = {grad_norm:.3e} (> 1e-2). "
                "Outer optimisation may not have reached a true minimum; "
                "try tighter gtol or 'trust-constr' outer_optimizer.",
                WarningSeverity.WARNING,
            )

        # Omega conditioning
        res.check_omega_conditioning()

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
        self._outer_eval_cache = {}
        self._best_outer_x = None
        self._best_outer_ofv = np.inf

    def _outer_eval_cache_get(
        self,
        x: np.ndarray,
    ) -> tuple[float, dict[int, np.ndarray]] | None:
        key = tuple(np.asarray(x, dtype=float).tolist())
        cached = self._outer_eval_cache.get(key)
        if cached is None:
            return None
        ofv, eta_hat = cached
        return float(ofv), {
            sid: np.asarray(value, dtype=float).copy() for sid, value in eta_hat.items()
        }

    def _outer_eval_cache_put(
        self,
        x: np.ndarray,
        ofv: float,
        eta_hat: dict[int, np.ndarray],
    ) -> None:
        key = tuple(np.asarray(x, dtype=float).tolist())
        self._outer_eval_cache[key] = (
            float(ofv),
            {sid: np.asarray(value, dtype=float).copy() for sid, value in eta_hat.items()},
        )

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
            cached = self._outer_eval_cache_get(x)
            if cached is not None:
                ofv, eta_hat = cached
                self._current_eta_hat = eta_hat
                return ofv
            p = ParameterSet.from_vector(x, init_params).apply_bounds()
            eta_hat = self._inner_loop(population_model, p)
            self._current_eta_hat = eta_hat
            ofv = self._outer_ofv(population_model, p, eta_hat)
            # If the outer OFV signals a hard failure (≥ 1e9 sentinel), the
            # warm-started η-hat is likely corrupted — typically by the optimizer
            # having explored a near-singular omega region where all ETA values
            # are shrunk to zero.  Retry from a cold (zero) start so that the
            # best-iterate tracker sees an accurate OFV and the next warm-start
            # begins from a clean state rather than the corrupted one.
            if ofv >= 1e9:
                self._current_eta_hat = {
                    sid: np.zeros(init_params.n_eta())
                    for sid in population_model.subject_ids()
                }
                eta_hat = self._inner_loop(population_model, p)
                self._current_eta_hat = eta_hat
                ofv = self._outer_ofv(population_model, p, eta_hat)
            self._outer_eval_cache_put(x, ofv, eta_hat)
            self._iter += 1
            self._ofv_history.append(ofv)
            if self.iteration_callback is not None:
                try:
                    self.iteration_callback(self._iter, ofv)
                except Exception:
                    pass
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
        # Reset to zero warm-start for the final evaluation so that a corrupted
        # η-hat cache (from degenerate omega regions explored during optimization)
        # does not propagate into the reported OFV.
        self._current_eta_hat = {
            sid: np.zeros(init_params.n_eta()) for sid in population_model.subject_ids()
        }
        cached_final = self._outer_eval_cache_get(result.x)
        if cached_final is not None:
            final_ofv, final_eta_hat = cached_final
            self._current_eta_hat = {
                sid: np.asarray(value, dtype=float).copy() for sid, value in final_eta_hat.items()
            }
        else:
            final_eta_hat = self._inner_loop(population_model, final_params)
            final_ofv = self._outer_ofv(population_model, final_params, final_eta_hat)
            self._outer_eval_cache_put(result.x, final_ofv, final_eta_hat)
        result, final_params, final_eta_hat, final_ofv = self._maybe_promote_best_iterate(
            result, init_params, population_model, final_params, final_eta_hat, final_ofv
        )

        result, final_params, final_eta_hat, final_ofv = self._apply_fallback_polish(
            objective, init_params, params, population_model,
            result, final_params, final_eta_hat, final_ofv,
        )
        if allow_structured_retries:
            result, final_params, final_eta_hat, final_ofv = self._apply_structured_retries(
                result, init_params, population_model, final_params, final_eta_hat, final_ofv,
            )
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
        # Re-evaluate at the best iterate using a *cold* (zero) inner warm-start.
        # If the optimizer wandered through a numerically degenerate region
        # (e.g. omega collapsing toward zero), the cached eta_hat can be
        # corrupted, causing _outer_ofv to return the 1e10 penalty sentinel even
        # at a genuinely good parameter point.  A zero-start inner loop is
        # safe and gives the canonical OFV at that point.
        best_params = ParameterSet.from_vector(self._best_outer_x, init_params).apply_bounds()
        n_eta = init_params.n_eta()
        saved_eta_hat = self._current_eta_hat
        self._current_eta_hat = {
            sid: np.zeros(n_eta) for sid in population_model.subject_ids()
        }
        try:
            best_eta_hat = self._inner_loop(population_model, best_params)
            best_ofv = self._outer_ofv(population_model, best_params, best_eta_hat)
        finally:
            self._current_eta_hat = saved_eta_hat
        if best_ofv < final_ofv:
            logger.info("  Retaining best iterate OFV %.4f -> %.4f", final_ofv, best_ofv)
            promoted_fields: dict[str, Any] = {
                "x": self._best_outer_x.copy(),
                "success": getattr(result, "success", False),
                "message": f"{getattr(result, 'message', '')} [best-iterate]".strip(),
            }
            result_x = getattr(result, "x", None)
            result_x_arr = None if result_x is None else np.asarray(result_x, dtype=float)
            same_point = result_x_arr is not None and np.allclose(
                result_x_arr,
                self._best_outer_x,
                rtol=0.0,
                atol=1e-12,
            )
            near_point = result_x_arr is not None and np.allclose(
                result_x_arr,
                self._best_outer_x,
                rtol=1e-4,
                atol=1e-4,
            )
            tiny_ofv_improvement = abs(float(final_ofv) - float(best_ofv)) <= 1e-4
            if same_point or (near_point and tiny_ofv_improvement):
                hess_inv = getattr(result, "hess_inv", None)
                if hess_inv is not None:
                    promoted_fields["hess_inv"] = hess_inv
            result = SimpleNamespace(**promoted_fields)
            return result, best_params, best_eta_hat, best_ofv
        return result, final_params, final_eta_hat, final_ofv

    def _apply_fallback_polish(
        self,
        objective: Any,
        init_params: ParameterSet,
        params: ParameterSet,
        population_model: Any,
        result: Any,
        final_params: ParameterSet,
        final_eta_hat: dict[int, np.ndarray],
        final_ofv: float,
    ) -> tuple[Any, ParameterSet, dict[int, np.ndarray], float]:
        """
        Optionally run a fallback outer optimizer from the current solution.

        If ``outer_fallback_optimizer`` is configured and different from the
        primary optimizer, run a short pass from the current solution.  The
        fallback result replaces the primary only when it improves the OFV.
        ``_maybe_promote_best_iterate`` is applied after the fallback as well.
        """
        fallback_optimizer = self.outer_fallback_optimizer
        if fallback_optimizer is None or fallback_optimizer == self.outer_optimizer:
            return result, final_params, final_eta_hat, final_ofv
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
                result, init_params, population_model, final_params, final_eta_hat, final_ofv
            )
        return result, final_params, final_eta_hat, final_ofv

    def _apply_structured_retries(
        self,
        result: Any,
        init_params: ParameterSet,
        population_model: Any,
        final_params: ParameterSet,
        final_eta_hat: dict[int, np.ndarray],
        final_ofv: float,
    ) -> tuple[Any, ParameterSet, dict[int, np.ndarray], float]:
        """
        Run structured FOCEI retry restarts when abnormal exit is detected.

        Each retry scales the current omega diagonal by a factor from
        ``retry_omega_scales`` and runs ``_run_single`` (without further
        retries) from that new starting point.  The best OFV wins.
        """
        if not self._should_run_structured_retries(result):
            return result, final_params, final_eta_hat, final_ofv
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

    def _outer_gradient_forward(
        self,
        objective: Any,
        x: np.ndarray,
        bounds: list[tuple[float | None, float | None]],
        *,
        rel_step: float = 1e-5,
    ) -> np.ndarray:
        """
        Forward-difference gradient for the outer FOCE objective.

        This explicitly reuses the cached base objective value at ``x`` rather
        than relying on SciPy's generic numerical-difference wrapper. On the
        ODE-heavy mixed-endpoint path this reduces duplicate bookkeeping and
        lets the exact outer-evaluation cache absorb revisits cleanly.
        """
        x_arr = np.asarray(x, dtype=float)
        f0 = float(objective(x_arr))
        grad = np.zeros_like(x_arr)

        for i, value in enumerate(x_arr):
            step = rel_step * max(1.0, abs(float(value)))
            lower, upper = bounds[i]
            if upper is not None and value + step > upper:
                if lower is not None and value - step >= lower:
                    step = -step
                elif lower is not None and lower < value:
                    step = -(value - lower) * 0.5
                elif upper > value:
                    step = (upper - value) * 0.5
            if lower is not None and value + step < lower:
                if upper is not None and value - step <= upper:
                    step = -step
                elif upper is not None and value < upper:
                    step = (upper - value) * 0.5
                elif value > lower:
                    step = -(value - lower) * 0.5
            if step == 0.0:
                step = 1e-8
            x_step = x_arr.copy()
            x_step[i] += step
            f_step = float(objective(x_step))
            grad[i] = (f_step - f0) / step

        return grad

    def _run_outer_optimizer(
        self,
        objective: Any,
        x0: np.ndarray,
        bounds: list[tuple[float | None, float | None]],
        *,
        optimizer: str,
        maxeval: int,
    ) -> Any:
        """
        Run the outer (population-parameter) optimizer.

        Supported methods
        -----------------
        ``L-BFGS-B``    (default) — quasi-Newton with box bounds.
        ``Powell``      — derivative-free, used as fallback for FOCEI.
        ``trust-constr`` — trust-region sequential-QP with box bounds.
                          More robust on ill-conditioned Omega; uses central
                          finite-difference gradients computed internally by
                          scipy when ``jac='2-point'``.
        """
        from scipy.optimize import Bounds as ScipyBounds

        method_key = str(optimizer).strip().upper().replace("-", "").replace("_", "")
        options: dict[str, Any]

        if method_key == "LBFGSB":
            return minimize(
                objective,
                x0,
                method="L-BFGS-B",
                jac=lambda x: self._outer_gradient_forward(objective, x, bounds),
                bounds=bounds,
                options={"maxiter": maxeval, "ftol": 1e-9, "gtol": self.gtol},
            )

        if method_key == "POWELL":
            return minimize(
                objective,
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxiter": maxeval, "xtol": 1e-3, "ftol": 1e-3},
            )

        if method_key == "TRUSTCONSTR":
            # Build scipy Bounds object; None → ±inf
            from scipy.optimize import SR1

            lb = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
            ub = np.array([b[1] if b[1] is not None else +np.inf for b in bounds])
            scipy_bounds = ScipyBounds(lb, ub)
            # scipy trust-constr: when jac is FD ('2-point'), the Hessian must
            # be a quasi-Newton approximation (SR1 or BFGS), not also FD.
            return minimize(
                objective,
                x0,
                method="trust-constr",
                jac="2-point",          # central FD gradient
                hess=SR1(),             # SR1 quasi-Newton Hessian approximation
                bounds=scipy_bounds,
                options={
                    "maxiter": maxeval,
                    "gtol": self.gtol,
                    "xtol": 1e-8,
                    "verbose": 0,
                    "finite_diff_rel_step": 1e-4,
                },
            )

        raise ValueError(
            f"Unsupported FOCE outer optimizer: {optimizer!r}. "
            "Use 'L-BFGS-B', 'Powell', or 'trust-constr'."
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

        if _can_skip_eta_optimization(params):
            zero_eta = np.zeros(n_eta, dtype=float)
            return {sid: zero_eta.copy() for sid in subject_ids}

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

                res = _optimize_eta_lbfgsb(obj_eta, eta0, self.inner_maxiter)
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

    def _sum_outer_subject_terms(
        self,
        subject_ids: list[int],
        term_fn: Any,
    ) -> float:
        if self.n_parallel == 1 or len(subject_ids) <= 1:
            return float(sum(float(term_fn(subj_id)) for subj_id in subject_ids))

        n_workers = self.n_parallel if self.n_parallel > 0 else None
        total = 0.0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for value in pool.map(term_fn, subject_ids):
                total += float(value)
        return float(total)

    def _outer_ofv_subject_term(
        self,
        population_model: Any,
        params: ParameterSet,
        eta_hat: dict[int, np.ndarray],
        subj_id: int,
        n_eta: int,
        omega_inv: np.ndarray,
        log_det_omega: float,
    ) -> float:
        eta_i = eta_hat.get(subj_id, np.zeros(n_eta))
        indiv = population_model.individual_model(subj_id)
        subj_ev = indiv.subject_events
        obs_mask = subj_ev.observation_mask()

        if not np.any(obs_mask):
            return 0.0

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
                G = _compute_G_i(
                    indiv,
                    params.theta,
                    eta_i,
                    params.sigma,
                    population_model.trans,
                    obs_mask,
                    pred_obs,
                )
                G_T_Rinv = G.T / var_obs
                M = omega_inv + G_T_Rinv @ G
                try:
                    _sm, log_det_M = np.linalg.slogdet(M)
                    log_det_M = float(log_det_M) if _sm > 0 else 0.0
                    # Woodbury matrix identity:
                    #   C_i^{-1} = R_i^{-1} - R_i^{-1} G M^{-1} G^T R_i^{-1}
                    # so  r^T C_i^{-1} r  =  r^T R_i^{-1} r  -  (G^T R_i^{-1} r)^T M^{-1} (G^T R_i^{-1} r)
                    G_T_Rinv_r = G_T_Rinv @ residuals  # (n_eta,)
                    quad_rinv = float(np.sum(residuals**2 / var_obs))
                    quad_woodbury = float(G_T_Rinv_r @ np.linalg.solve(M, G_T_Rinv_r))
                    quad = quad_rinv - quad_woodbury
                    log_det_R = float(np.sum(np.log(var_obs)))
                    log_det_ci = log_det_R + log_det_omega + log_det_M
                except np.linalg.LinAlgError:
                    quad = float(np.sum(residuals**2 / var_obs))
                    log_det_ci = float(np.sum(np.log(var_obs)))
            else:
                quad = float(np.sum(residuals**2 / var_obs))
                log_det_ci = float(np.sum(np.log(var_obs)))

            eta_penalty = float(eta_i @ omega_inv @ eta_i)
            ofv_i = n_obs * LOG2PI + log_det_ci + quad + eta_penalty
            if self.interaction:
                ofv_i -= n_eta * LOG2PI
            else:
                ofv_i += log_det_omega
            return float(ofv_i)
        except Exception:
            return 1e10

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
        n_eta = params.n_eta()
        try:
            omega_rep = repair_pd(params.omega)
            omega_inv = np.linalg.inv(omega_rep)
            _sign, log_det_omega = np.linalg.slogdet(omega_rep)
            log_det_omega = float(log_det_omega) if _sign > 0 else 0.0
        except np.linalg.LinAlgError:
            omega_inv = np.eye(n_eta)
            log_det_omega = 0.0
        subject_ids = population_model.subject_ids()
        ofv = self._sum_outer_subject_terms(
            subject_ids,
            lambda subj_id: self._outer_ofv_subject_term(
                population_model,
                params,
                eta_hat,
                subj_id,
                n_eta,
                omega_inv,
                log_det_omega,
            ),
        )

        # A4: add prior penalty if model is PriorAugmentedModel
        if hasattr(population_model, "prior"):
            ofv += population_model.prior.penalty(params.theta, params.omega)

        return ofv
