"""
First-Order (FO) estimation method.

FO linearizes the model at ETA=0, giving a closed-form marginal likelihood:

    p(y_i | theta, Omega, Sigma) ≈ N(y_i; f_i(0), C_i(0))

where:
    f_i(0) = model prediction at eta=0
    C_i(0) = R_i * Omega * R_i^T + Sigma_residual

R_i = df_i/d(eta) at eta=0 (first-order approximation).

FO OFV = Σ_i [ log|C_i| + (y_i - f_i)^T C_i^{-1} (y_i - f_i) ]
       = Σ_i [ n_i*log(2π) + log|C_i| + (y_i - f_i)^T C_i^{-1} (y_i - f_i) ]

Note: FO is biased (uses first-order Taylor expansion) but fast.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from openpkpd.estimation.base import EstimationMethod, EstimationResult
from openpkpd.math.autodiff import jacobian
from openpkpd.math.matrix import repair_pd
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.constants import LOG2PI
from openpkpd.utils.logging import get_logger

logger = get_logger("estimation.fo")


class FOMethod(EstimationMethod):
    """First-Order (FO) estimation method."""

    method_name = "FO"

    def __init__(
        self,
        maxeval: int = 9999,
        sigdig: int = 3,
        print_interval: int = 5,
        noabort: bool = False,
    ) -> None:
        self.maxeval = maxeval
        self.sigdig = sigdig
        self.print_interval = print_interval
        self.noabort = noabort
        self._iter = 0
        self._ofv_history: list[float] = []

    def estimate(
        self,
        population_model: Any,
        init_params: ParameterSet,
        **kwargs: Any,
    ) -> EstimationResult:
        t0 = time.time()
        logger.info(f"Starting FO estimation with {population_model.n_subjects()} subjects")

        # Pack initial parameters into optimizer vector
        x0 = init_params.to_vector()
        self._iter = 0
        self._ofv_history = []

        def objective(x: np.ndarray) -> float:
            p = ParameterSet.from_vector(x, init_params).apply_bounds()
            ofv = self._compute_fo_ofv(population_model, p)
            self._iter += 1
            self._ofv_history.append(ofv)
            if self._iter % self.print_interval == 0:
                logger.info(f"  Iter {self._iter:5d}  OFV={ofv:.4f}")
            return ofv

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": self.maxeval, "ftol": 1e-8, "gtol": 1e-5},
        )

        final_params = ParameterSet.from_vector(result.x, init_params).apply_bounds()
        converged = result.success

        # Compute post-hoc ETAs (= 0 for FO)
        eta_hat = {sid: np.zeros(init_params.n_eta()) for sid in population_model.subject_ids()}
        final_ofv = float(result.fun)

        elapsed = time.time() - t0
        logger.info(
            f"FO estimation completed in {elapsed:.1f}s, OFV={final_ofv:.4f}, converged={converged}"
        )

        res = EstimationResult(
            theta_final=final_params.theta,
            omega_final=final_params.omega,
            sigma_final=final_params.sigma,
            ofv=final_ofv,
            converged=converged,
            post_hoc_etas=eta_hat,
            ofv_history=self._ofv_history,
            n_function_evals=self._iter,
            elapsed_time=elapsed,
            method=self.method_name,
            message=result.message if hasattr(result, "message") else "",
        )
        dataset = getattr(population_model, "dataset", None)
        if dataset is not None and hasattr(dataset, "n_observations"):
            res.n_observations = int(dataset.n_observations())
        res.n_subjects = population_model.n_subjects()
        res.compute_n_parameters(
            theta_specs=getattr(init_params, "theta_specs", None),
            omega_specs=getattr(init_params, "omega_specs", None),
            sigma_specs=getattr(init_params, "sigma_specs", None),
        )
        res.compute_shrinkage()
        return res

    def _compute_fo_ofv(self, population_model: Any, params: ParameterSet) -> float:
        """
        Compute FO OFV = -2 log L_FO.

        For each subject:
          OFV_i = log|C_i| + (y-f)^T C_i^{-1} (y-f) + n_obs * log(2π)

        where C_i = R_i Ω R_i^T + diag(σ_j² * f_ij²) [for proportional]
                  = R_i Ω R_i^T + Σ_eps    [for additive]
        """
        ofv = 0.0
        eta_zero = np.zeros(params.n_eta())

        for subj_id in population_model.subject_ids():
            try:
                ofv_i = self._fo_ofv_individual(population_model, params, subj_id, eta_zero)
                ofv += ofv_i
            except Exception:
                ofv += 1e10

        # A4: add prior penalty if model is PriorAugmentedModel
        if hasattr(population_model, "prior"):
            ofv += population_model.prior.penalty(params.theta, params.omega)

        return ofv

    def _fo_ofv_individual(
        self,
        population_model: Any,
        params: ParameterSet,
        subj_id: int,
        eta_zero: np.ndarray,
    ) -> float:
        """FO OFV contribution for one subject."""
        indiv = population_model.individual_model(subj_id)
        subj_events = indiv.subject_events
        obs_mask = subj_events.observation_mask()
        dv = subj_events.obs_dv[obs_mask]

        if len(dv) == 0:
            return 0.0

        # Observation-model mean and variance at eta=0
        _, _, _, pred, var = indiv.evaluate_observation_model(
            params.theta,
            eta_zero,
            params.sigma,
            trans=population_model.trans,
        )
        pred_obs = pred[obs_mask]
        var_obs = np.maximum(var[obs_mask], 1e-10)
        n_obs = len(dv)

        # FO approximation: R_i ≈ d(pred)/d_eta at eta=0
        R = self._compute_fo_jacobian_R(
            indiv, params, pred_obs, obs_mask, eta_zero, population_model, n_obs
        )

        # C_i = R * Omega * R^T + residual variance
        C_i = R @ params.omega @ R.T + np.diag(var_obs)
        C_i = repair_pd(C_i)

        residuals = dv - pred_obs
        try:
            # Cholesky factor gives log-det and solves the quadratic form in one pass.
            L = np.linalg.cholesky(C_i)
            logdet_val = 2.0 * float(np.sum(np.log(np.diag(L))))
            x = np.linalg.solve(L, residuals)
            quad = float(x @ x)
        except np.linalg.LinAlgError:
            sign, logdet_val = np.linalg.slogdet(C_i)
            if sign <= 0:
                logdet_val = 50.0
            quad = float(np.sum(residuals**2 / var_obs))

        ofv_i = n_obs * LOG2PI + logdet_val + quad
        return ofv_i

    def _compute_fo_jacobian_R(
        self,
        indiv: Any,
        params: ParameterSet,
        pred_obs: np.ndarray,
        obs_mask: np.ndarray,
        eta_zero: np.ndarray,
        population_model: Any,
        n_obs: int,
    ) -> np.ndarray:
        """
        Compute the FO Jacobian R_i = d(pred_i)/d(eta) evaluated at eta=0.

        Tries the native analytical Jacobian first (zero extra ODE calls), then
        falls back to forward finite-differences via ``openpkpd.math.autodiff.jacobian``.
        Returns a zeros matrix of shape ``(n_obs, n_eta)`` on any numerical failure.
        """
        n_eta = params.n_eta()
        if n_eta == 0:
            return np.zeros((n_obs, 0))

        supports_native_jac = getattr(indiv, "supports_prediction_eta_jacobian", None)
        native_jacobian = getattr(indiv, "prediction_eta_jacobian", None)
        if (
            callable(supports_native_jac)
            and bool(supports_native_jac(trans=population_model.trans))
            and callable(native_jacobian)
        ):
            try:
                return np.asarray(
                    native_jacobian(
                        params.theta, eta_zero, params.sigma, trans=population_model.trans
                    ),
                    dtype=float,
                )
            except Exception:
                return np.zeros((n_obs, n_eta))

        def pred_of_eta(eta: np.ndarray) -> np.ndarray:
            _, _, _, pred_eta, _ = indiv.evaluate_observation_model(
                params.theta, eta, params.sigma, trans=population_model.trans
            )
            return pred_eta[obs_mask]

        try:
            return jacobian(pred_of_eta, eta_zero, eps=1e-5, f0=pred_obs, method="forward")
        except Exception:
            return np.zeros((n_obs, n_eta))
