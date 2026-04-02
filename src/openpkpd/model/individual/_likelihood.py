"""LikelihoodMixin — log-likelihood, obj_eta, obj_eta_many."""
from __future__ import annotations

import logging
import math

import numpy as np

from openpkpd.data.blq import blq_log_likelihood, is_blq
from openpkpd.model.individual._base import (
    _build_lloq_array,
    _BLQ_METHOD_CODE,
    _RUST_CORE_AVAILABLE,
    _neg2ll_obs_loop_rust,
)
from openpkpd.model.residuals import log_likelihood_normal
from openpkpd.utils.constants import BLQMethod

logger = logging.getLogger(__name__)


class LikelihoodMixin:
    """Mixin providing log-likelihood and eta-objective evaluation."""

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
            pred_arr = np.asarray(pred, dtype=np.float64)
            var_arr = np.asarray(var, dtype=np.float64)
            mask_arr = np.asarray(obs_mask, dtype=bool)
            if not (len(pred_arr) == len(var_arr) == len(mask_arr) == len(lloq_arr) == n):
                raise ValueError(
                    f"neg2ll_obs_loop: array length mismatch — "
                    f"dv={n}, pred={len(pred_arr)}, var={len(var_arr)}, "
                    f"mask={len(mask_arr)}, lloq={len(lloq_arr)}"
                )
            return _neg2ll_obs_loop_rust(
                dv_arr[:n] if len(dv_arr) > n else dv_arr,
                pred_arr,
                var_arr,
                mask_arr,
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
            except Exception as _obj_eta_e:
                logger.warning(
                    "IndividualModel %s failed at obj_eta kernel path: %s",
                    getattr(self, "subject_id", "?"), _obj_eta_e,
                )

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
            except Exception as _many_kernel_e:
                logger.warning(
                    "IndividualModel %s failed at obj_eta_many kernel path: %s",
                    getattr(self, "subject_id", "?"), _many_kernel_e,
                )

        values = np.empty(len(eta_arr), dtype=float)
        for i, eta in enumerate(eta_arr):
            try:
                neg2ll_data = self.log_likelihood(theta, eta, sigma, trans=trans)
                values[i] = neg2ll_data + self._eta_penalty_value(eta, omega_inv, block_size)
            except Exception as _many_ll_e:
                logger.warning(
                    "IndividualModel %s failed at obj_eta_many log_likelihood (row %d): %s",
                    getattr(self, "subject_id", "?"), i, _many_ll_e,
                )
                values[i] = 1e10
        return values

