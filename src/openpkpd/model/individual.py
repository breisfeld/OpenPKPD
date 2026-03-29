"""
Per-subject model evaluation.

IndividualModel evaluates the log-likelihood (and contribution to OFV)
for a single subject given their:
  - Observed data (DV, times, compartments)
  - Dosing events
  - Current ETA (random effects)
  - Population parameters (theta, omega, sigma)
  - Compiled PK and error callables
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from openpkpd.data.blq import blq_log_likelihood, is_blq
from openpkpd.data.event_processor import SubjectEvents
from openpkpd.math.autodiff import jacobian
from openpkpd.math.matrix import numerical_hessian
from openpkpd.model.residuals import log_likelihood_normal
from openpkpd.pk.base import PKSubroutine
from openpkpd.utils.constants import BLQMethod
from openpkpd.utils.errors import PKError

# ---------------------------------------------------------------------------
# Optional Rust-compiled inner-loop extension (openpkpd._core).
# Falls back silently to the pure-Python path if the extension is not built.
# Build:  cd rust && cargo build --release
#         cp target/release/lib_core.so \
#            ../src/openpkpd/_core.cpython-312-x86_64-linux-gnu.so
# ---------------------------------------------------------------------------
try:
    from openpkpd._core import neg2ll_obs_loop as _neg2ll_obs_loop_rust

    _RUST_CORE_AVAILABLE = True
except ImportError:
    _RUST_CORE_AVAILABLE = False

# BLQMethod string → integer code expected by the Rust function
_BLQ_METHOD_CODE: dict[str | None, int] = {
    None: 0,
    BLQMethod.M1: 1,
    BLQMethod.M2: 2,
    BLQMethod.M3: 3,
    BLQMethod.M4: 4,
    BLQMethod.M5: 5,
    BLQMethod.M6: 6,
    BLQMethod.M7: 7,
}


_NAN_LLOQ_CACHE: dict[int, np.ndarray] = {}


def _build_lloq_array(lloq: object, n: int) -> np.ndarray:
    """Return a float64 array of length *n* with per-obs LLOQ values.

    NaN encodes "no LLOQ for this observation" (i.e. normal non-BLQ obs).
    Accepts None (all NaN), a scalar float, or an array.

    The all-NaN case (lloq=None, the common path) reuses a cached array to
    avoid allocating a new one on every log_likelihood call.
    """
    if lloq is None:
        cached = _NAN_LLOQ_CACHE.get(n)
        if cached is None:
            cached = np.full(n, np.nan, dtype=np.float64)
            _NAN_LLOQ_CACHE[n] = cached
        return cached
    out = np.full(n, np.nan, dtype=np.float64)
    if np.ndim(lloq) == 0:
        out[:] = float(lloq)  # type: ignore[arg-type]
    else:
        arr = np.asarray(lloq, dtype=float)
        length = min(len(arr), n)
        out[:length] = arr[:length]
    return out

_W_PROP_THETA_RE = re.compile(r"^w=f\*theta\[(\d+)\]$", re.IGNORECASE)
_W_THETA_RE = re.compile(r"^w=theta\[(\d+)\]$", re.IGNORECASE)
_W_SQRT_RE = re.compile(
    r"^w=math\.sqrt\(theta\[(\d+)\]\*\*2\+\(f\*theta\[(\d+)\]\)\*\*2\)$",
    re.IGNORECASE,
)


class IndividualModel:
    """
    Evaluates the individual log-likelihood and OFV contribution.

    Holds references to the population-level PK model and compiled
    callables so it can be called repeatedly during inner-loop
    optimization (EBE estimation).
    """

    def __init__(
        self,
        subject_events: SubjectEvents,
        pk_subroutine: PKSubroutine,
        pk_callable: Callable | None,
        error_callable: Callable | None,
        n_eps: int = 1,
        blq_method: str = BLQMethod.M1,
        lloq: float | np.ndarray | None = None,
        des_callable: Callable | None = None,
        occasion_indices: np.ndarray | None = None,
    ) -> None:
        self.subject_events = subject_events
        self.pk_subroutine = pk_subroutine
        self.pk_callable = pk_callable
        self.error_callable = error_callable
        self.des_callable = des_callable
        self.n_eps = n_eps
        self.blq_method: str = blq_method
        self.lloq: float | np.ndarray | None = lloq
        self.occasion_indices: np.ndarray | None = occasion_indices  # B1
        self._cached_ipred: np.ndarray | None = None
        self._error_call_mode: str = "unknown"
        self._compiled_error_raw = getattr(error_callable, "_call_raw", None)
        self._error_requires_amounts = self._infer_error_requires_amounts(error_callable)
        self._obs_mask = subject_events.observation_mask()
        self._dose_events = subject_events.dose_events
        self._base_covariates = subject_events.covariate_at(0.0) if pk_callable is not None else {}
        self._has_time_varying_covariates = (
            subject_events.covariate_df is not None
            and pk_callable is not None
            and des_callable is not None
        )
        self._covariate_change_times = (
            subject_events.covariate_change_times()
            if subject_events.covariate_df is not None
            else []
        )
        self._derivative_kernel_cache: dict[int, Any | None] = {}
        self._pk_param_transformers: dict[int, Callable[[dict[str, float]], dict[str, float]]] = {}
        self._observation_covariates = tuple(
            subject_events.observation_covariates_at(i)
            for i in range(len(subject_events.obs_times))
        )
        self._unique_occasions = (
            np.unique(occasion_indices)
            if occasion_indices is not None
            and len(occasion_indices) == len(subject_events.obs_times)
            else None
        )
        solve_signature = inspect.signature(pk_subroutine.solve)
        self._solve_supports_return_amounts = "return_amounts" in solve_signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in solve_signature.parameters.values()
        )
        self._eta_penalty_cache_key: tuple[bytes, tuple[int, ...], int] | None = None
        self._eta_penalty_precision: np.ndarray | None = None
        self._eta_penalty_block_size: int | None = None
        self._eps_basis_vectors: tuple[tuple[float, ...], ...] = tuple(
            tuple(1.0 if i == j else 0.0 for i in range(self.n_eps)) for j in range(self.n_eps)
        )
        self._common_error_model = self._infer_common_error_model(error_callable, self.n_eps)

    def __getstate__(self) -> dict[str, Any]:
        """Drop rebuildable caches so worker-process pickling stays robust."""
        state = self.__dict__.copy()
        state["_pk_param_transformers"] = {}
        state["_derivative_kernel_cache"] = {}
        state["_cached_ipred"] = None
        state["_eta_penalty_cache_key"] = None
        state["_eta_penalty_precision"] = None
        state["_eta_penalty_block_size"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "_pk_param_transformers" not in self.__dict__:
            self._pk_param_transformers = {}
        if "_derivative_kernel_cache" not in self.__dict__:
            self._derivative_kernel_cache = {}

    @staticmethod
    def _infer_error_requires_amounts(error_callable: Callable | None) -> bool:
        if error_callable is None:
            return False

        uses_amounts = getattr(error_callable, "_uses_amounts", None)
        if uses_amounts is not None:
            return bool(uses_amounts)

        try:
            signature = inspect.signature(error_callable)
        except (TypeError, ValueError):
            return True

        parameters = signature.parameters.values()
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
            return True
        return "a" in signature.parameters

    @staticmethod
    def _infer_common_error_model(
        error_callable: Callable | None,
        n_eps: int,
    ) -> tuple[str, tuple[int, ...]] | None:
        source = getattr(error_callable, "_source", None)
        if not isinstance(source, str):
            return None

        lines = ["".join(line.lower().split()) for line in source.splitlines() if line.strip()]
        normalized: list[str] = []
        for line in lines:
            if line == "ipred=f":
                continue
            normalized.append(re.sub(r"\bipred\b", "f", line))
        normalized_lines = tuple(normalized)

        if n_eps == 1:
            if normalized_lines in {("y=f*(1+eps[0])",), ("y=f+f*eps[0]",)}:
                return "proportional", ()
            if normalized_lines == ("y=f+eps[0]",):
                return "additive", ()
            if len(normalized_lines) == 2:
                prop_match = _W_PROP_THETA_RE.fullmatch(normalized_lines[0])
                if prop_match is not None and normalized_lines[1] == "y=f+w*eps[0]":
                    return "proportional_theta", (int(prop_match.group(1)),)
                w_match = _W_THETA_RE.fullmatch(normalized_lines[0])
                if w_match is not None and normalized_lines[1] == "y=f+w*eps[0]":
                    return "additive_theta", (int(w_match.group(1)),)
            if 2 <= len(normalized_lines) <= 4:
                sqrt_match = _W_SQRT_RE.fullmatch(normalized_lines[0])
                if sqrt_match is not None and normalized_lines[1:] in {
                    ("y=f+w*eps[0]",),
                    ("y=f+w*eps[0]", "ires=dv-f", "iwres=ires/w"),
                }:
                    return "combined_theta", (int(sqrt_match.group(1)), int(sqrt_match.group(2)))
        elif n_eps == 2 and normalized_lines == ("y=f+eps[0]+f*eps[1]",):
            return "combined_eps", ()
        return None

    def _fast_obs_model(
        self,
        f: np.ndarray,
        theta: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (pred, var) arrays for common error models without looping.

        Used by ``evaluate_observation_model`` when eps=0 (estimation path).
        pred = f (no shift when eps=0 for all detected patterns).
        var  = per-observation residual variance from sigma and theta.

        Returns None if the pattern is not handled (falls back to Python loop).
        """
        common = self._common_error_model
        if common is None:
            return None
        kind, theta_idx = common
        f_arr = np.asarray(f, dtype=float)
        n = len(f_arr)
        s00 = max(float(sigma[0, 0]) if sigma.size > 0 else 1.0, 1e-10)

        if kind == "proportional":
            # Y = F*(1+EPS)  →  var = F² · σ₀₀
            return f_arr.copy(), np.maximum(f_arr * f_arr * s00, 1e-10)

        if kind == "additive":
            # Y = F+EPS  →  var = σ₀₀ (constant)
            return f_arr.copy(), np.full(n, s00)

        if kind == "proportional_theta":
            # W = F·θ[k],  Y = F+W·EPS  →  var = (θ[k]·F)² · σ₀₀
            w = float(theta[theta_idx[0]])
            return f_arr.copy(), np.maximum((w * f_arr) ** 2 * s00, 1e-10)

        if kind == "additive_theta":
            # W = θ[k],  Y = F+W·EPS  →  var = θ[k]² · σ₀₀
            w = float(theta[theta_idx[0]])
            return f_arr.copy(), np.full(n, max(w * w * s00, 1e-10))

        if kind == "combined_theta":
            # W = √(θ[k1]²+(F·θ[k2])²),  Y = F+W·EPS  →  var = W² · σ₀₀
            add_sd = float(theta[theta_idx[0]])
            prop_sd = float(theta[theta_idx[1]])
            w2 = add_sd * add_sd + (prop_sd * f_arr) ** 2
            return f_arr.copy(), np.maximum(w2 * s00, 1e-10)

        if kind == "combined_eps":
            # Y = F+EPS[0]+F·EPS[1]  →  var = σ₀₀ + F²·σ₁₁
            s11 = max(float(sigma[1, 1]) if sigma.size >= 4 else s00, 1e-10)
            return f_arr.copy(), np.maximum(s00 + f_arr * f_arr * s11, 1e-10)

        return None

    def simulate_error_predictions_fast(
        self,
        theta: np.ndarray,
        f: np.ndarray,
        all_eps: np.ndarray,
    ) -> np.ndarray | None:
        """Return a vectorized simulation path for common compiled $ERROR forms."""
        common = self._common_error_model
        if common is None or self._error_requires_amounts:
            return None

        f_arr = np.asarray(f, dtype=float)
        eps_arr = np.asarray(all_eps, dtype=float)
        if len(f_arr) == 0:
            return np.array([], dtype=float)
        if eps_arr.ndim != 2 or eps_arr.shape[0] != len(f_arr):
            return None

        kind, theta_idx = common
        if kind == "proportional" and eps_arr.shape[1] >= 1:
            return f_arr * (1.0 + eps_arr[:, 0])
        if kind == "additive" and eps_arr.shape[1] >= 1:
            return f_arr + eps_arr[:, 0]
        if kind == "proportional_theta" and eps_arr.shape[1] >= 1:
            w = float(theta[theta_idx[0]])
            return f_arr + (f_arr * w) * eps_arr[:, 0]
        if kind == "additive_theta" and eps_arr.shape[1] >= 1:
            w = float(theta[theta_idx[0]])
            return f_arr + w * eps_arr[:, 0]
        if kind == "combined_theta" and eps_arr.shape[1] >= 1:
            add_sd = float(theta[theta_idx[0]])
            prop_sd = float(theta[theta_idx[1]])
            w_arr = np.sqrt(add_sd * add_sd + (f_arr * prop_sd) ** 2)
            return f_arr + w_arr * eps_arr[:, 0]
        if kind == "combined_eps" and eps_arr.shape[1] >= 2:
            return f_arr + eps_arr[:, 0] + f_arr * eps_arr[:, 1]
        return None

    def _get_pk_param_transformer(
        self,
        trans: int,
    ) -> Callable[[dict[str, float]], dict[str, float]]:
        transformer = self._pk_param_transformers.get(trans)
        if transformer is not None:
            return transformer

        apply_trans = self.pk_subroutine.apply_trans

        def _transform_or_raw(raw_params: dict[str, float]) -> dict[str, float]:
            try:
                return apply_trans(raw_params, trans)
            except Exception:
                return raw_params

        self._pk_param_transformers[trans] = _transform_or_raw
        return _transform_or_raw

    def _evaluate_predictions(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
        *,
        include_amounts: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Internal prediction helper that also returns compartment amounts."""
        obs_times = self.subject_events.obs_times
        if len(obs_times) == 0:
            return np.array([]), np.array([], dtype=bool), np.array([]), None

        obs_mask = self._obs_mask
        solve = self.pk_subroutine.solve
        transform_pk_params = self._get_pk_param_transformer(trans)
        base_solve_kwargs: dict[str, Any] = {}
        if not include_amounts and self._solve_supports_return_amounts:
            base_solve_kwargs["return_amounts"] = False

        if (
            self.occasion_indices is not None
            and self.pk_callable is not None
            and len(self.occasion_indices) == len(obs_times)
        ):
            theta_seq = list(theta)
            eta_seq = list(eta)
            ipred = np.full(len(obs_times), np.nan)
            f_arr = np.full(len(obs_times), np.nan)
            amounts: np.ndarray | None = None if include_amounts else None
            unique_occs = (
                self._unique_occasions
                if self._unique_occasions is not None
                else np.unique(self.occasion_indices)
            )
            for occ in unique_occs:
                occ_mask = self.occasion_indices == occ
                occ_times = obs_times[occ_mask]
                covariates = {"OCC": float(occ)}
                pk_params = self.pk_callable(theta_seq, eta_seq, t=0.0, covariates=covariates)
                micro_params = transform_pk_params(pk_params)
                try:
                    pk_sol = solve(
                        micro_params,
                        self._dose_events,
                        occ_times,
                        pk_callable=None,
                        des_callable=self.des_callable,
                        **base_solve_kwargs,
                    )
                except PKError:
                    raise
                except Exception as exc:
                    raise PKError(
                        f"PK solve failed for subject {self.subject_events.subject_id} "
                        f"occasion {occ}: {exc}"
                    ) from exc
                ipred[occ_mask] = pk_sol.ipred
                f_arr[occ_mask] = pk_sol.f if pk_sol.f is not None else pk_sol.ipred
                if include_amounts and amounts is None:
                    n_comp = pk_sol.amounts.shape[1] if pk_sol.amounts.ndim == 2 else 1
                    amounts = np.full((len(obs_times), n_comp), np.nan)
                if include_amounts and amounts is not None:
                    occ_amounts = pk_sol.amounts
                    if occ_amounts.ndim == 1:
                        occ_amounts = occ_amounts[:, None]
                    amounts[occ_mask, : occ_amounts.shape[1]] = occ_amounts
            return ipred, obs_mask, f_arr, amounts

        if self.pk_callable is not None:
            theta_seq = list(theta)
            eta_seq = list(eta)
            pk_params = self.pk_callable(
                theta_seq,
                eta_seq,
                t=0.0,
                covariates=self._base_covariates,
            )
        else:
            pk_params = _theta_to_pk_params(theta, eta, trans)

        micro_params = transform_pk_params(pk_params)

        solve_kwargs: dict[str, Any] = dict(base_solve_kwargs)
        if self._has_time_varying_covariates and self.pk_callable is not None:
            _pk_callable = self.pk_callable
            _theta_seq = theta_seq
            _eta_seq = eta_seq
            _subj_events = self.subject_events

            def _covariate_fn(t: float) -> dict:
                covs = _subj_events.covariate_at(t)
                raw = _pk_callable(_theta_seq, _eta_seq, t=t, covariates=covs)
                return transform_pk_params(raw)

            solve_kwargs["covariate_fn"] = _covariate_fn
            solve_kwargs["covariate_change_times"] = self._covariate_change_times

        try:
            pk_sol = solve(
                micro_params,
                self._dose_events,
                obs_times,
                pk_callable=None,
                des_callable=self.des_callable,
                **solve_kwargs,
            )
        except PKError:
            raise
        except Exception as exc:
            raise PKError(
                f"PK solve failed for subject {self.subject_events.subject_id}: {exc}"
            ) from exc

        ipred = pk_sol.ipred
        f = pk_sol.f if pk_sol.f is not None else ipred
        return ipred, obs_mask, f, pk_sol.amounts if include_amounts else None

    def evaluate(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate predictions for this subject.

        When ``occasion_indices`` is set (IOV model), the $PK callable is
        evaluated once per unique occasion, with the occasion index passed as
        the covariate ``OCC``.  Results are stitched into a single IPRED array
        ordered by observation time.

        Returns:
            (ipred, obs_mask, f) tuple where:
              - ipred:    Individual predicted values at all obs times
              - obs_mask: Boolean mask for non-missing observations
              - f:        F-scaled prediction (= ipred before $ERROR)
        """
        ipred, obs_mask, f, _amounts = self._evaluate_predictions(
            theta, eta, sigma, trans=trans, include_amounts=False
        )
        return ipred, obs_mask, f

    def evaluate_observation_model(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
        eps_val: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the observation model for this subject.

        Returns:
            (ipred, obs_mask, f, pred, var) tuple where:
              - ipred: PK-side individual predictions at all obs times
              - obs_mask: Boolean mask for non-missing observations
              - f: PK-side F prediction before $ERROR
              - pred: Observation-model mean after $ERROR with EPS fixed
              - var: Per-observation residual variance
        """
        ipred, obs_mask, f, amounts = self._evaluate_predictions(
            theta,
            eta,
            sigma,
            trans=trans,
            include_amounts=self._error_requires_amounts,
        )
        if len(ipred) == 0:
            empty = np.array([])
            return ipred, obs_mask, f, empty, empty

        sigma_diag = float(sigma[0, 0]) if sigma.size > 0 else 1.0
        pred = np.asarray(ipred, dtype=float).copy()
        var = np.full(len(pred), max(sigma_diag, 1e-10), dtype=float)

        if self.error_callable is None:
            return ipred, obs_mask, f, pred, var

        # ── Fast path for standard error models (eps=0 estimation path) ──────
        # When the $ERROR block matches a detected standard pattern and eps is
        # zero (all FOCE/SAEM inner-loop calls), skip the per-observation Python
        # loop entirely and compute pred/var with vectorised NumPy operations.
        # This eliminates ~13 µs/call (FOCE: saves ~5.5M µs across 424k calls).
        if eps_val is None and self._common_error_model is not None and not self._error_requires_amounts:
            _fast = self._fast_obs_model(f, theta, sigma)
            if _fast is not None:
                return ipred, obs_mask, f, _fast[0], _fast[1]

        dv = self.subject_events.obs_dv
        obs_times = self.subject_events.obs_times
        obs_covariates = self._observation_covariates
        eps = np.asarray(eps_val if eps_val is not None else np.zeros(self.n_eps), dtype=float)
        theta_seq = list(theta)
        eta_seq = list(eta)
        zero_eps_seq = [0.0] * self.n_eps
        eps_is_zero = bool(np.allclose(eps, 0.0))
        eps_seq = zero_eps_seq if eps_is_zero else list(eps)
        amount_rows = None
        if amounts is not None:
            amount_array = np.asarray(amounts, dtype=float)
            if amount_array.ndim == 1:
                amount_array = amount_array[:, None]
            amount_rows = amount_array.tolist()

        call_error_model = self._call_error_model_prepared
        extract_prediction = self._extract_error_prediction
        estimate_variance = self._estimate_residual_variance_prepared

        for i in range(len(pred)):
            y_obs = float(dv[i]) if i < len(dv) and np.isfinite(dv[i]) else float("nan")
            a_i = None if amount_rows is None else amount_rows[i]
            f_i = float(f[i])
            ipred_i = float(ipred[i])
            t_i = float(obs_times[i])
            mean_out = call_error_model(
                theta_seq=theta_seq,
                eta_seq=eta_seq,
                eps_seq=zero_eps_seq,
                f_i=f_i,
                ipred_i=ipred_i,
                y_obs=y_obs,
                t_i=t_i,
                a_i=a_i,
                covariates=obs_covariates[i],
                sigma=sigma,
            )
            mean_pred = extract_prediction(mean_out, ipred_i)
            error_out = (
                mean_out
                if eps_is_zero
                else call_error_model(
                    theta_seq=theta_seq,
                    eta_seq=eta_seq,
                    eps_seq=eps_seq,
                    f_i=f_i,
                    ipred_i=ipred_i,
                    y_obs=y_obs,
                    t_i=t_i,
                    a_i=a_i,
                    covariates=obs_covariates[i],
                    sigma=sigma,
                )
            )
            pred[i] = extract_prediction(error_out, ipred_i)
            var[i] = estimate_variance(
                theta_seq=theta_seq,
                eta_seq=eta_seq,
                sigma=sigma,
                f_i=f_i,
                ipred_i=ipred_i,
                y_obs=y_obs,
                t_i=t_i,
                a_i=a_i,
                covariates=obs_covariates[i],
                mean_out=mean_out,
                mean_pred=mean_pred,
                default_var=var[i],
            )

        return ipred, obs_mask, f, pred, var

    @staticmethod
    def _extract_error_prediction(error_out: dict[str, Any], default_pred: float) -> float:
        return float(
            error_out.get(
                "Y",
                error_out.get(
                    "y",
                    error_out.get("IPRED", error_out.get("ipred", default_pred)),
                ),
            )
        )

    def _call_error_model(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        eps: np.ndarray,
        f_i: float,
        ipred_i: float,
        y_obs: float,
        t_i: float,
        a_i: np.ndarray | None,
        covariates: dict[str, Any] | None,
        sigma: np.ndarray,
    ) -> dict[str, Any]:
        return self._call_error_model_prepared(
            theta_seq=list(theta),
            eta_seq=list(eta),
            eps_seq=list(eps),
            f_i=f_i,
            ipred_i=ipred_i,
            y_obs=y_obs,
            t_i=t_i,
            a_i=None if a_i is None else a_i.tolist(),
            covariates=covariates,
            sigma=sigma,
        )

    def _call_error_model_prepared(
        self,
        theta_seq: list[float],
        eta_seq: list[float],
        eps_seq: list[float],
        f_i: float,
        ipred_i: float,
        y_obs: float,
        t_i: float,
        a_i: list[float] | None,
        covariates: dict[str, Any] | None,
        sigma: np.ndarray,
    ) -> dict[str, Any]:
        if self._compiled_error_raw is not None:
            return self._compiled_error_raw(
                theta_seq,
                eta_seq,
                eps_seq,
                f_i,
                ipred=ipred_i,
                dv=y_obs,
                t=t_i,
                a=a_i,
                covariates=covariates,
                sigma=sigma,
            )

        if self._error_call_mode == "full":
            return self.error_callable(  # type: ignore[misc]
                theta_seq,
                eta_seq,
                eps_seq,
                f_i,
                ipred=ipred_i,
                dv=y_obs,
                t=t_i,
                a=a_i,
                covariates=covariates,
                sigma=sigma,
            )

        if self._error_call_mode == "no_sigma":
            return self.error_callable(  # type: ignore[misc]
                theta_seq,
                eta_seq,
                eps_seq,
                f_i,
                ipred=ipred_i,
                dv=y_obs,
                t=t_i,
                a=a_i,
                covariates=covariates,
            )

        if self._error_call_mode == "basic":
            return self.error_callable(  # type: ignore[misc]
                theta_seq,
                eta_seq,
                eps_seq,
                f_i,
                ipred=ipred_i,
                dv=y_obs,
                t=t_i,
            )

        try:
            result = self.error_callable(  # type: ignore[misc]
                theta_seq,
                eta_seq,
                eps_seq,
                f_i,
                ipred=ipred_i,
                dv=y_obs,
                t=t_i,
                a=a_i,
                covariates=covariates,
                sigma=sigma,
            )
            self._error_call_mode = "full"
            return result
        except TypeError as exc:
            if "sigma" in str(exc):
                try:
                    result = self.error_callable(  # type: ignore[misc]
                        theta_seq,
                        eta_seq,
                        eps_seq,
                        f_i,
                        ipred=ipred_i,
                        dv=y_obs,
                        t=t_i,
                        a=a_i,
                        covariates=covariates,
                    )
                    self._error_call_mode = "no_sigma"
                    return result
                except TypeError as inner_exc:
                    if "covariates" not in str(inner_exc) and "a" not in str(inner_exc):
                        raise
            elif "covariates" not in str(exc) and "a" not in str(exc):
                raise
            result = self.error_callable(  # type: ignore[misc]
                theta_seq,
                eta_seq,
                eps_seq,
                f_i,
                ipred=ipred_i,
                dv=y_obs,
                t=t_i,
            )
            self._error_call_mode = "basic"
            self._error_requires_amounts = False
            return result

    def _estimate_residual_variance(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        f_i: float,
        ipred_i: float,
        y_obs: float,
        t_i: float,
        a_i: np.ndarray | None,
        covariates: dict[str, Any] | None,
        mean_out: dict[str, Any],
        mean_pred: float,
        default_var: float,
    ) -> float:
        return self._estimate_residual_variance_prepared(
            theta_seq=list(theta),
            eta_seq=list(eta),
            sigma=sigma,
            f_i=f_i,
            ipred_i=ipred_i,
            y_obs=y_obs,
            t_i=t_i,
            a_i=None if a_i is None else a_i.tolist(),
            covariates=covariates,
            mean_out=mean_out,
            mean_pred=mean_pred,
            default_var=default_var,
        )

    def _estimate_residual_variance_prepared(
        self,
        theta_seq: list[float],
        eta_seq: list[float],
        sigma: np.ndarray,
        f_i: float,
        ipred_i: float,
        y_obs: float,
        t_i: float,
        a_i: list[float] | None,
        covariates: dict[str, Any] | None,
        mean_out: dict[str, Any],
        mean_pred: float,
        default_var: float,
    ) -> float:
        if self.n_eps <= 0 or sigma.size == 0:
            return max(float(default_var), 1e-10)

        w_val = mean_out.get("W", mean_out.get("w"))
        if self.n_eps == 1 and w_val is not None and np.isfinite(w_val):
            return max(float(w_val) ** 2 * float(sigma[0, 0]), 1e-10)

        sensitivities = np.zeros(self.n_eps, dtype=float)
        for j, eps_basis in enumerate(self._eps_basis_vectors):
            error_out = self._call_error_model_prepared(
                theta_seq=theta_seq,
                eta_seq=eta_seq,
                eps_seq=list(eps_basis),
                f_i=f_i,
                ipred_i=ipred_i,
                y_obs=y_obs,
                t_i=t_i,
                a_i=a_i,
                covariates=covariates,
                sigma=sigma,
            )
            sensitivities[j] = self._extract_error_prediction(error_out, ipred_i) - mean_pred

        return max(float(sensitivities @ sigma @ sensitivities), 1e-10)

    def log_likelihood(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
        eps_val: np.ndarray | None = None,
        blq_method: str | None = None,
        lloq: float | np.ndarray | None = None,
    ) -> float:
        """
        Compute -2 * log p(y_i | eta, theta) for this subject.

        For FOCE/FO: eps is set to 0 (linearization around eps=0).

        BLQ handling:
            When ``blq_method`` is not M1 and ``lloq`` is provided (or set on
            the instance), observations where DV < LLOQ receive a censored
            log-likelihood contribution instead of the standard normal one:

            - M1: BLQ observations are excluded (MDV=1 approach). Return 0.0
              for those observations.
            - M2/M3: Censored likelihood P(Y < LLOQ) via log-normal CDF.
            - M4: M3 with truncated-normal normalisation (Y >= 0).
            - M5/M6: Replace DV with LLOQ/2 and use normal likelihood.
            - M7: Replace DV with 0 and use normal likelihood.

        Args:
            theta:      Fixed-effects parameter vector.
            eta:        Individual random-effects vector.
            sigma:      Residual variance-covariance matrix.
            trans:      TRANS parameterisation code (default 2).
            eps_val:    EPS residual vector; zeros if None (for FO/FOCE).
            blq_method: BLQ handling method. If None, uses ``self.blq_method``
                        (default M1).
            lloq:       Lower limit of quantification. Scalar or per-
                        observation array. If None, uses ``self.lloq``.

        Returns:
            -2 * log-likelihood for this subject (scalar).
        """
        ipred, obs_mask, f, pred, var = self.evaluate_observation_model(
            theta,
            eta,
            sigma,
            trans=trans,
            eps_val=eps_val,
        )
        if len(ipred) == 0:
            return 0.0

        # Resolve BLQ settings: method-call arguments override instance attrs
        active_method = blq_method if blq_method is not None else self.blq_method
        active_lloq = lloq if lloq is not None else self.lloq

        dv = self.subject_events.obs_dv
        n = len(obs_mask)

        # ── Rust fast-path ────────────────────────────────────────────────
        # Delegates the entire per-observation loop to the compiled extension.
        # Array conversions use np.asarray (zero-copy when already correct dtype).
        if _RUST_CORE_AVAILABLE:
            lloq_arr = _build_lloq_array(active_lloq, n)
            blq_code = _BLQ_METHOD_CODE.get(active_method, 0)
            dv_arr = np.asarray(dv, dtype=np.float64)
            return _neg2ll_obs_loop_rust(
                dv_arr[:n] if len(dv_arr) > n else dv_arr,
                np.asarray(pred, dtype=np.float64),
                np.asarray(var, dtype=np.float64),
                np.asarray(obs_mask, dtype=bool),
                lloq_arr,
                blq_code,
            )

        # ── Pure-Python fallback ──────────────────────────────────────────
        # Track whether this subject has had its first BLQ (for M6)
        seen_blq_m6: bool = False

        ll = 0.0
        for i, obs in enumerate(obs_mask):
            if not obs:
                continue
            y_obs = float(dv[i])
            if math.isnan(y_obs):
                continue

            mu = float(pred[i])
            var_i = float(var[i])

            # Determine the LLOQ applicable to this observation
            lloq_i: float | None = None
            if active_lloq is not None:
                if np.ndim(active_lloq) == 0:
                    lloq_i = float(active_lloq)  # type: ignore[arg-type]
                else:
                    lloq_arr = np.asarray(active_lloq)
                    lloq_i = float(lloq_arr[i]) if i < len(lloq_arr) else None

            # Check BLQ status and dispatch accordingly
            if lloq_i is not None and not math.isnan(lloq_i) and is_blq(y_obs, lloq_i):
                if active_method == BLQMethod.M1:
                    # Exclude: skip this observation
                    continue
                elif active_method == BLQMethod.M6:
                    if not seen_blq_m6:
                        # First BLQ: use LLOQ/2 imputation
                        seen_blq_m6 = True
                        ll += blq_log_likelihood(y_obs, mu, var_i, lloq_i, BLQMethod.M6)
                    else:
                        # Subsequent BLQ: discard (exclude)
                        continue
                else:
                    ll += blq_log_likelihood(y_obs, mu, var_i, lloq_i, active_method)
            else:
                # Normal (non-BLQ) observation
                ll += log_likelihood_normal(y_obs, mu, var_i)

        return -2.0 * ll

    def obj_eta(
        self,
        eta: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> float:
        """
        Inner-loop objective: -2 * log p(y_i | eta) + eta^T * Omega^{-1} * eta.

        This is minimized over eta in the FOCE inner loop.

        B1 (IOV): When occasion_indices are present and the eta vector is larger
        than omega.shape[0], the extra ETAs are treated as per-occasion random
        effects drawn from the same BSV omega block (diagonal copy).  The eta
        vector layout is [eta_bsv | eta_occ1 | eta_occ2 | ...] and the penalty
        uses the block-diagonal omega formed by stacking omega n_occ+1 times.
        """
        kernel = self.get_subject_derivative_kernel(trans)
        if kernel is not None:
            try:
                value, _grad = self.eta_objective_value_grad(eta, theta, omega, sigma, trans=trans)
                return value
            except Exception:
                pass

        neg2ll_data = self.log_likelihood(theta, eta, sigma, trans=trans)

        omega_inv, block_size = self._eta_penalty_structure(omega, len(eta))
        eta_penalty = self._eta_penalty_value(np.asarray(eta, dtype=float), omega_inv, block_size)
        return neg2ll_data + eta_penalty

    def obj_eta_many(
        self,
        eta_batch: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta_batch, dtype=float)
        if eta_arr.ndim == 1:
            eta_arr = eta_arr[None, :]
        if len(eta_arr) == 0:
            return np.array([], dtype=float)

        omega_inv, block_size = self._eta_penalty_structure(omega, eta_arr.shape[1])

        kernel = self.get_subject_derivative_kernel(trans)
        if kernel is not None:
            try:
                data_values = np.asarray(
                    kernel.eta_data_objective_values(theta, eta_arr, sigma), dtype=float
                )
                penalties = np.empty(len(eta_arr), dtype=float)
                for i, eta in enumerate(eta_arr):
                    penalties[i] = self._eta_penalty_value(eta, omega_inv, block_size)
                return data_values + penalties
            except Exception:
                pass

        values = np.empty(len(eta_arr), dtype=float)
        for i, eta in enumerate(eta_arr):
            try:
                neg2ll_data = self.log_likelihood(theta, eta, sigma, trans=trans)
                values[i] = neg2ll_data + self._eta_penalty_value(eta, omega_inv, block_size)
            except Exception:
                values[i] = 1e10
        return values

    def supports_eta_objective_gradient(self, trans: int = 2) -> bool:
        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        return bool(kernel is not None and getattr(capabilities, "eta_objective_gradient", False))

    def supports_prediction_eta_jacobian(self, trans: int = 2) -> bool:
        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        return bool(kernel is not None and getattr(capabilities, "prediction_eta_jacobian", False))

    def supports_theta_data_objective_gradient(self, trans: int = 2) -> bool:
        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        if kernel is None or not getattr(capabilities, "theta_data_objective_gradient", False):
            return False
        supports = getattr(kernel, "supports_theta_data_objective_gradient", None)
        return bool(callable(supports) and supports())

    def supports_prediction_theta_jacobian(self, trans: int = 2) -> bool:
        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        if kernel is None or not getattr(capabilities, "prediction_theta_jacobian", False):
            return False
        supports = getattr(kernel, "supports_theta_data_objective_gradient", None)
        return bool(callable(supports) and supports())

    def supports_eta_objective_hessian(self, trans: int = 2) -> bool:
        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        return bool(kernel is not None and getattr(capabilities, "eta_objective_hessian", False))

    def eta_objective_value_grad(
        self,
        eta: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> tuple[float, np.ndarray]:
        kernel = self.get_subject_derivative_kernel(trans)
        if kernel is None:
            raise NotImplementedError("eta objective derivative kernel is not available")
        data_value, grad_data = kernel.eta_data_objective_value_grad(
            theta, np.asarray(eta, dtype=float), sigma
        )
        eta_arr = np.asarray(eta, dtype=float)
        omega_inv, block_size = self._eta_penalty_structure(omega, len(eta_arr))
        eta_penalty = self._eta_penalty_value(eta_arr, omega_inv, block_size)
        if block_size is None:
            grad_penalty = 2.0 * (omega_inv @ eta_arr)
        else:
            eta_blocks = eta_arr.reshape(-1, block_size)
            grad_penalty = (2.0 * (eta_blocks @ omega_inv.T)).reshape(-1)
        return data_value + eta_penalty, np.asarray(grad_data, dtype=float) + grad_penalty

    def supports_symbolic_obj_eta(self, trans: int = 2) -> bool:
        return self.supports_eta_objective_gradient(trans)

    def symbolic_obj_eta_value_grad(
        self,
        eta: np.ndarray,
        theta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> tuple[float, np.ndarray]:
        return self.eta_objective_value_grad(eta, theta, omega, sigma, trans=trans)

    def get_subject_derivative_kernel(self, trans: int = 2) -> Any | None:
        if trans in self._derivative_kernel_cache:
            return self._derivative_kernel_cache[trans]
        from openpkpd.model.derivative_kernels import build_subject_derivative_kernel

        kernel = build_subject_derivative_kernel(self, trans)
        self._derivative_kernel_cache[trans] = kernel
        return kernel

    def _get_symbolic_eta_objective(self, trans: int) -> Any | None:
        return self.get_subject_derivative_kernel(trans)

    def eta_objective_hessian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta, dtype=float)
        if len(eta_arr) == 0:
            return np.zeros((0, 0), dtype=float)

        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        if kernel is not None and getattr(capabilities, "eta_objective_hessian", False):
            data_hess = np.asarray(
                kernel.eta_data_objective_hessian(theta, eta_arr, sigma), dtype=float
            )
            omega_inv, block_size = self._eta_penalty_structure(omega, len(eta_arr))
            if block_size is None:
                penalty_hess = 2.0 * omega_inv
            else:
                n_blocks = len(eta_arr) // block_size
                penalty_hess = 2.0 * np.kron(np.eye(n_blocks), omega_inv)
            return data_hess + np.asarray(penalty_hess, dtype=float)

        def obj_eta_local(eta_value: np.ndarray) -> float:
            return float(self.obj_eta(eta_value, theta, omega, sigma, trans=trans))

        return numerical_hessian(obj_eta_local, eta_arr, eps=1e-4)

    def prediction_eta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta, dtype=float)
        obs_mask = self.subject_events.observation_mask()
        n_obs = int(np.sum(obs_mask))
        if len(eta_arr) == 0:
            return np.zeros((n_obs, 0), dtype=float)

        kernel = self.get_subject_derivative_kernel(trans)
        capabilities = getattr(kernel, "capabilities", None)
        if kernel is not None and getattr(capabilities, "prediction_eta_jacobian", False):
            return np.asarray(kernel.prediction_eta_jacobian(theta, eta_arr, sigma), dtype=float)

        def pred_of_eta(eta_value: np.ndarray) -> np.ndarray:
            _, _, _, pred_eta, _ = self.evaluate_observation_model(
                theta, eta_value, sigma, trans=trans
            )
            return pred_eta[obs_mask]

        return jacobian(pred_of_eta, eta_arr, eps=1e-5)

    def theta_data_objective_gradient(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        kernel = self.get_subject_derivative_kernel(trans)
        if kernel is None:
            raise NotImplementedError("theta data-objective derivative kernel is not available")
        return np.asarray(
            kernel.theta_data_objective_gradient(
                np.asarray(theta, dtype=float),
                np.asarray(eta, dtype=float),
                sigma,
            ),
            dtype=float,
        )

    def prediction_theta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        obs_mask = self.subject_events.observation_mask()
        n_obs = int(np.sum(obs_mask))
        kernel = self.get_subject_derivative_kernel(trans)
        if kernel is not None and self.supports_prediction_theta_jacobian(trans):
            return np.asarray(kernel.prediction_theta_jacobian(theta_arr, eta, sigma), dtype=float)
        raise NotImplementedError("prediction theta Jacobian is not available")

    @staticmethod
    def _eta_penalty_value(
        eta: np.ndarray,
        omega_inv: np.ndarray,
        block_size: int | None,
    ) -> float:
        if block_size is None:
            eta_penalty = float(eta @ omega_inv @ eta)
        else:
            eta_blocks = np.asarray(eta, dtype=float).reshape(-1, block_size)
            eta_penalty = float(np.einsum("bi,ij,bj->", eta_blocks, omega_inv, eta_blocks))
        return eta_penalty

    def _eta_penalty_structure(
        self,
        omega: np.ndarray,
        n_eta: int,
    ) -> tuple[np.ndarray, int | None]:
        omega_arr = np.ascontiguousarray(omega, dtype=float)
        cache_key = (omega_arr.tobytes(), omega_arr.shape, n_eta)
        if self._eta_penalty_cache_key == cache_key and self._eta_penalty_precision is not None:
            return self._eta_penalty_precision, self._eta_penalty_block_size

        from openpkpd.math.matrix import repair_pd

        n_bsv = omega.shape[0]
        block_size: int | None = None
        try:
            omega_inv = np.linalg.inv(repair_pd(omega_arr))
        except np.linalg.LinAlgError:
            omega_inv = np.eye(n_bsv)

        if n_eta > n_bsv and self.occasion_indices is not None:
            block_size = n_bsv

        self._eta_penalty_cache_key = cache_key
        self._eta_penalty_precision = omega_inv
        self._eta_penalty_block_size = block_size
        return omega_inv, block_size


def _theta_to_pk_params(
    theta: np.ndarray,
    eta: np.ndarray,
    trans: int,
) -> dict[str, float]:
    """
    Fallback: map theta vector to PK params for simple models.

    For TRANS2: theta[0]=CL, theta[1]=V (or theta[0]=KA, theta[1]=CL, theta[2]=V for ADVAN2)
    """
    if trans == 2:
        if len(theta) >= 3:
            return {
                "KA": float(theta[0]) * math.exp(float(eta[0]) if len(eta) > 0 else 0),
                "CL": float(theta[1]) * math.exp(float(eta[1]) if len(eta) > 1 else 0),
                "V": float(theta[2]) * math.exp(float(eta[2]) if len(eta) > 2 else 0),
            }
        elif len(theta) >= 2:
            return {
                "CL": float(theta[0]) * math.exp(float(eta[0]) if len(eta) > 0 else 0),
                "V": float(theta[1]) * math.exp(float(eta[1]) if len(eta) > 1 else 0),
            }
    return {"K": float(theta[0]), "V": 1.0}
