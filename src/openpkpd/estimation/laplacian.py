"""
Laplacian estimation method.

Extends FOCE with a second-order Hessian correction term:

    OFV_Laplace_i = -2 log p(y_i|η̂_i) + η̂_i^T Ω^{-1} η̂_i + log|H_i|

where:
    H_i = d²[-2 log p(y_i,η)] / d(η)²  evaluated at η = η̂_i
        = d²[-2 log p(y_i|η) + η^T Ω^{-1} η] / d(η)²

The Hessian is computed numerically (FD) in Stage 1 and via jax.hessian in Stage 2.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from openpkpd.estimation.foce import FOCEMethod
from openpkpd.math.matrix import numerical_hessian, repair_pd
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.constants import Method
from openpkpd.utils.logging import get_logger

logger = get_logger("estimation.laplacian")


class LaplacianMethod(FOCEMethod):
    """
    Laplacian estimation (FOCE + Hessian correction).

    Inherits the inner/outer loop structure from FOCEMethod.
    Adds log|H_i| to the OFV.
    """

    method_name = Method.LAPLACIAN

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(interaction=kwargs.pop("interaction", True), **kwargs)
        self.method_name = Method.LAPLACIAN

    def _outer_ofv(
        self,
        population_model: Any,
        params: ParameterSet,
        eta_hat: dict[int, np.ndarray],
    ) -> float:
        """
        Laplacian OFV: FOCE OFV + Σ_i log|H_i|.

        H_i is the Hessian of the individual objective at η̂_i.
        """
        ofv = 0.0
        omega_inv = np.linalg.inv(repair_pd(params.omega))

        for subj_id in population_model.subject_ids():
            eta_i = eta_hat.get(subj_id, np.zeros(params.n_eta()))
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

                sigma_diag = float(params.sigma[0, 0]) if params.sigma.size > 0 else 1.0
                if self.interaction and np.allclose(var_obs, sigma_diag, rtol=1e-8, atol=1e-10):
                    var_vec = np.maximum(pred_obs**2 * sigma_diag, 1e-10)
                else:
                    var_vec = var_obs

                # Data log-likelihood
                log_det_ci = float(np.sum(np.log(var_vec)))
                residuals = dv - pred_obs
                quad = float(np.sum(residuals**2 / var_vec))
                eta_penalty = float(eta_i @ omega_inv @ eta_i)

                foce_ofv_i = n_obs * LOG2PI_LOCAL + log_det_ci + quad + eta_penalty

                # Hessian correction: log|H_i|
                def obj_eta(eta: np.ndarray, _indiv=indiv) -> float:
                    return float(
                        _indiv.obj_eta(
                            eta,
                            params.theta,
                            params.omega,
                            params.sigma,
                            trans=population_model.trans,
                        )
                    )

                eta_hessian = getattr(indiv, "eta_objective_hessian", None)
                if callable(eta_hessian):
                    H_i = np.asarray(
                        eta_hessian(
                            params.theta,
                            eta_i,
                            params.omega,
                            params.sigma,
                            trans=population_model.trans,
                        ),
                        dtype=float,
                    )
                else:
                    H_i = numerical_hessian(obj_eta, eta_i, eps=1e-4)
                try:
                    sign, logdet_H = np.linalg.slogdet(H_i)
                    if sign <= 0:
                        logdet_H = 0.0
                except Exception:
                    logdet_H = 0.0

                ofv_i = foce_ofv_i + logdet_H
                ofv += ofv_i
            except Exception:
                ofv += 1e10

        # Add prior penalty if model is PriorAugmentedModel
        if hasattr(population_model, "prior"):
            ofv += population_model.prior.penalty(params.theta, params.omega)

        return ofv


LOG2PI_LOCAL = math.log(2 * math.pi)
