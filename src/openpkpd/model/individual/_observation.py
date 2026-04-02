"""ObservationModelMixin — error model inference, prediction, observation model."""
from __future__ import annotations

import logging
import re
from typing import Any
from collections.abc import Callable

import numpy as np

from openpkpd.model.individual._base import (
    _W_PROP_THETA_RE,
    _W_THETA_RE,
    _W_SQRT_RE,
    _DVID_THEN_RE,
    _MIXED_PKPD_IPRED_RE,
    _MIXED_PKPD_SQRT_RE,
    _MIXED_PKPD_Y_BRANCH_RE,
    _MIXED_PKPD_ALIAS_RE,
    _theta_to_pk_params,
)
from openpkpd.utils.errors import PKError

logger = logging.getLogger(__name__)


class ObservationModelMixin:
    """Mixin providing error model inference and prediction evaluation."""

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
        elif n_eps == 2:
            if normalized_lines == ("y=f+eps[0]+f*eps[1]",):
                return "combined_eps", ()

            if len(normalized_lines) == 10:
                alias_matches = tuple(_MIXED_PKPD_ALIAS_RE.fullmatch(line) for line in normalized_lines[:3])
                ipred_match = _MIXED_PKPD_IPRED_RE.fullmatch(normalized_lines[3])
                y_pd_match = _MIXED_PKPD_Y_BRANCH_RE.fullmatch(normalized_lines[5])
                dvid_w_match = _DVID_THEN_RE.fullmatch(normalized_lines[6])
                sqrt_match = _MIXED_PKPD_SQRT_RE.fullmatch(normalized_lines[7])
                dvid_y_match = _DVID_THEN_RE.fullmatch(normalized_lines[8])
                y_pk_match = _MIXED_PKPD_Y_BRANCH_RE.fullmatch(normalized_lines[9])
                if all(match is not None for match in alias_matches) and ipred_match is not None:
                    pkprop_name = alias_matches[0].group(1)
                    pkadd_name = alias_matches[1].group(1)
                    pdadd_name = alias_matches[2].group(1)
                    if (
                        normalized_lines[4] == f"w={pdadd_name}"
                        and y_pd_match is not None
                        and int(y_pd_match.group(1)) == 1
                        and dvid_w_match is not None
                        and dvid_y_match is not None
                        and dvid_w_match.group(1) == dvid_y_match.group(1) == "1"
                        and sqrt_match is not None
                        and sqrt_match.group(1) == pkprop_name
                        and sqrt_match.group(2) == pkadd_name
                        and y_pk_match is not None
                        and int(y_pk_match.group(1)) == 0
                    ):
                        return "mixed_pkpd_dvid_theta", (
                            int(ipred_match.group(1)),
                            int(ipred_match.group(2)),
                            int(alias_matches[0].group(2)),
                            int(alias_matches[1].group(2)),
                            int(alias_matches[2].group(2)),
                        )
        return None

    def _fast_obs_model(
        self,
        f: np.ndarray,
        theta: np.ndarray,
        sigma: np.ndarray,
        amounts: np.ndarray | None = None,
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
            # Y = F+EPS[0]+F·EPS[1]  →  var = σ₀₀ + 2·F·σ₀₁ + F²·σ₁₁
            # Off-diagonal term is needed when EPS(1) and EPS(2) are correlated.
            s01 = float(sigma[0, 1]) if sigma.size >= 4 else 0.0
            s11 = float(sigma[1, 1]) if sigma.size >= 4 else s00
            return f_arr.copy(), np.maximum(s00 + 2.0 * s01 * f_arr + s11 * f_arr * f_arr, 1e-10)

        if kind == "mixed_pkpd_dvid_theta":
            if amounts is None:
                return None
            amount_arr = np.asarray(amounts, dtype=float)
            if amount_arr.ndim == 1:
                amount_arr = amount_arr[:, None]
            e0_idx, amount_idx, pk_prop_idx, pk_add_idx, pd_add_idx = theta_idx
            if amount_idx >= amount_arr.shape[1]:
                return None
            dvid = self._observation_dvid
            if dvid is None:
                return None
            if len(dvid) != n:
                return None
            s11 = float(sigma[1, 1]) if sigma.size >= 4 else s00
            pred = f_arr.copy()
            var = np.empty(n, dtype=float)
            pk_mask = dvid == 1.0
            pd_mask = ~pk_mask
            pk_prop = float(theta[pk_prop_idx])
            pk_add = float(theta[pk_add_idx])
            pd_add = float(theta[pd_add_idx])
            if np.any(pk_mask):
                var[pk_mask] = np.maximum(((pk_prop * f_arr[pk_mask]) ** 2 + pk_add * pk_add) * s00, 1e-10)
            if np.any(pd_mask):
                pred[pd_mask] = float(theta[e0_idx]) + amount_arr[pd_mask, amount_idx]
                var[pd_mask] = max(pd_add * pd_add * s11, 1e-10)
            return pred, var

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
            except Exception as _trans_e:
                logger.warning(
                    "IndividualModel %s failed at pk_param transform (TRANS=%d): %s",
                    getattr(self, "subject_id", "?"), trans, _trans_e,
                )
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

        native_pk_sol = self._try_native_pk_backend(micro_params, obs_times)
        if native_pk_sol is not None:
            ipred = native_pk_sol.ipred
            f = native_pk_sol.f if native_pk_sol.f is not None else ipred
            return ipred, obs_mask, f, native_pk_sol.amounts if include_amounts else None

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
        if eps_val is None and self._common_error_model is not None:
            _fast = self._fast_obs_model(f, theta, sigma, amounts=amounts)
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


