"""
Mixed-effects PD model (PopulationPDModel).

Wraps a PDModel subclass in a hierarchical (population) mixed-effects structure,
adding ETAs on PD parameters such as Emax, EC50, Kin, and Kout.  This enables
fitting of PD data from multiple subjects using FOCE or FO methods.

The mixed-effects model assumes:

    param_i = param_pop * exp(ETA_i)          [log-normal ETA]

where ETA_i ~ N(0, Omega) for each subject.

Usage::

    from openpkpd.models.population_pd import PopulationPDModel
    from openpkpd.models.pkpd import EmaxModel
    from openpkpd.estimation.foce import FOCEMethod

    pop_pd = PopulationPDModel(
        pd_model=EmaxModel(),
        eta_params=["Emax", "EC50"],
        theta_init={"Emax": 100.0, "EC50": 10.0, "Hill": 1.0},
        omega_init=np.diag([0.1, 0.1]),
        sigma2=1.0,
    )
    result = pop_pd.estimate(subjects_data, method=FOCEMethod())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from openpkpd.math.matrix import repair_pd
from openpkpd.models.pkpd import PDData, PDModel

# ── Per-subject data container ───────────────────────────────────────────────


@dataclass
class PopulationPDResult:
    """
    Result from a population PD model estimation.

    Attributes:
        theta:       Fixed-effects parameter estimates.
        omega:       Inter-individual variability matrix (ETA covariance).
        sigma2:      Residual variance estimate.
        ofv:         Final OFV (-2 * log-likelihood).
        converged:   Whether optimiser converged.
        aic:         AIC = OFV + 2 * n_parameters.
        post_hoc_etas: Per-subject ETA estimates {subject_id: eta_vector}.
    """

    theta: dict[str, float]
    omega: np.ndarray
    sigma2: float
    ofv: float
    converged: bool
    aic: float
    post_hoc_etas: dict[Any, np.ndarray] = field(default_factory=dict)


# ── Population PD model ──────────────────────────────────────────────────────


class PopulationPDModel:
    """
    Mixed-effects wrapper for any PDModel subclass.

    Adds ETAs on selected PD parameters and estimates population parameters
    (theta, omega, sigma²) by maximising the FOCE marginal likelihood.

    Args:
        pd_model:   Any PDModel instance (EmaxModel, IDRModel, etc.).
        eta_params: List of parameter names that have ETAs (log-normal).
        theta_init: Initial population (fixed-effects) parameter values.
        omega_init: Initial ETA covariance matrix, shape (len(eta_params),
                    len(eta_params)).
        sigma2:     Initial residual variance (can be estimated or fixed).
        estimate_sigma2: If True, sigma² is estimated; otherwise held fixed.
        maxeval:    Maximum optimiser iterations for the outer loop.
    """

    def __init__(
        self,
        pd_model: PDModel,
        eta_params: list[str],
        theta_init: dict[str, float],
        omega_init: np.ndarray,
        sigma2: float = 1.0,
        estimate_sigma2: bool = True,
        maxeval: int = 500,
    ) -> None:
        self.pd_model = pd_model
        self.eta_params = list(eta_params)
        self.theta_init = dict(theta_init)
        self.omega_init = np.asarray(omega_init, dtype=float)
        self.sigma2 = sigma2
        self.estimate_sigma2 = estimate_sigma2
        self.maxeval = maxeval

        n_eta = len(eta_params)
        if self.omega_init.shape != (n_eta, n_eta):
            raise ValueError(
                f"omega_init must have shape ({n_eta}, {n_eta}), got {self.omega_init.shape}"
            )

    # ── Parameter packing / unpacking ────────────────────────────────────────

    def _pack(
        self,
        theta: dict[str, float],
        omega: np.ndarray,
        sigma2: float,
    ) -> np.ndarray:
        """Pack (theta, omega lower-triangle, sigma2) into a flat vector."""
        theta_vec = np.array([theta[k] for k in self.theta_init])
        n = omega.shape[0]
        omega_lt = [omega[i, j] for i in range(n) for j in range(i + 1)]
        if self.estimate_sigma2:
            return np.concatenate([theta_vec, omega_lt, [sigma2]])
        return np.concatenate([theta_vec, omega_lt])

    def _unpack(self, x: np.ndarray) -> tuple[dict[str, float], np.ndarray, float]:
        """Unpack flat vector back into (theta, omega, sigma2)."""
        n_theta = len(self.theta_init)
        n_eta = len(self.eta_params)
        n_omega_lt = n_eta * (n_eta + 1) // 2

        theta_vec = x[:n_theta]
        theta = {k: max(float(v), 1e-9) for k, v in zip(self.theta_init, theta_vec, strict=False)}

        omega_lt = x[n_theta : n_theta + n_omega_lt]
        omega = np.zeros((n_eta, n_eta))
        idx = 0
        for i in range(n_eta):
            for j in range(i + 1):
                omega[i, j] = omega_lt[idx]
                omega[j, i] = omega_lt[idx]
                idx += 1
        omega = repair_pd(omega)

        sigma2 = max(float(x[n_theta + n_omega_lt]), 1e-6) if self.estimate_sigma2 else self.sigma2

        return theta, omega, sigma2

    # ── Individual OFV (inner FOCE objective) ────────────────────────────────

    def _individual_ofv(
        self,
        eta: np.ndarray,
        data: PDData,
        theta: dict[str, float],
        omega_inv: np.ndarray,
        sigma2: float,
    ) -> float:
        """
        FOCE inner-loop objective for one subject.

        OFV_i = -2*LL_data + eta^T * Omega^{-1} * eta
        """
        # Apply ETAs: param_i = param_pop * exp(eta)
        params_i = dict(theta)
        for k, e in zip(self.eta_params, eta, strict=False):
            params_i[k] = max(theta[k] * float(np.exp(e)), 1e-9)

        try:
            pred = self.pd_model.predict(params_i, data)
            if not np.all(np.isfinite(pred)):
                return 1e10
            resid = data.response - pred
            n = len(resid)
            ll_data = float(np.sum(resid**2) / sigma2 + n * np.log(sigma2))
        except Exception:
            return 1e10

        eta_penalty = float(eta @ omega_inv @ eta)
        return ll_data + eta_penalty

    # ── Post-hoc ETA optimisation ─────────────────────────────────────────────

    def _posthoc_eta(
        self,
        data: PDData,
        theta: dict[str, float],
        omega: np.ndarray,
        sigma2: float,
    ) -> np.ndarray:
        """Optimise ETA for a single subject (L-BFGS-B)."""
        n_eta = len(self.eta_params)
        omega_inv = np.linalg.inv(repair_pd(omega))
        eta0 = np.zeros(n_eta)

        result = minimize(
            self._individual_ofv,
            eta0,
            args=(data, theta, omega_inv, sigma2),
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-10},
        )
        return result.x

    # ── Outer OFV ────────────────────────────────────────────────────────────

    def _outer_ofv(
        self,
        subjects: list[PDData],
        theta: dict[str, float],
        omega: np.ndarray,
        sigma2: float,
    ) -> float:
        """FOCE outer OFV summed over all subjects."""
        omega_inv = np.linalg.inv(repair_pd(omega))
        ofv = 0.0
        for data in subjects:
            eta_hat = self._posthoc_eta(data, theta, omega, sigma2)
            ofv += self._individual_ofv(eta_hat, data, theta, omega_inv, sigma2)
            # Add log-det(Omega) contribution (n_eta terms per subject)
            try:
                sign, logdet = np.linalg.slogdet(omega)
                if sign > 0:
                    ofv += float(logdet)
            except Exception:
                pass
        return ofv

    # ── Main estimation ───────────────────────────────────────────────────────

    def estimate(
        self,
        subjects: list[PDData],
        **kwargs: Any,
    ) -> PopulationPDResult:
        """
        Estimate population PD parameters.

        Args:
            subjects: List of PDData objects, one per subject.
            **kwargs: Passed to scipy.optimize.minimize.

        Returns:
            PopulationPDResult with population and individual estimates.
        """
        x0 = self._pack(self.theta_init, self.omega_init, self.sigma2)

        def objective(x: np.ndarray) -> float:
            theta, omega, sigma2 = self._unpack(x)
            return self._outer_ofv(subjects, theta, omega, sigma2)

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": self.maxeval, "ftol": 1e-9},
        )

        theta_f, omega_f, sigma2_f = self._unpack(result.x)
        ofv = float(result.fun)
        n_params = len(x0)
        aic = ofv + 2.0 * n_params

        # Post-hoc ETAs for all subjects
        post_hoc: dict[Any, np.ndarray] = {}
        for data in subjects:
            post_hoc[data.subject_id] = self._posthoc_eta(data, theta_f, omega_f, sigma2_f)

        return PopulationPDResult(
            theta=theta_f,
            omega=omega_f,
            sigma2=sigma2_f,
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
            post_hoc_etas=post_hoc,
        )


__all__ = ["PopulationPDModel", "PopulationPDResult"]
