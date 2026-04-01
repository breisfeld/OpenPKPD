"""
Population Fisher Information Matrix (PFIM) for optimal design.

Computes the expected Fisher information for a given sampling schedule
and population model parameters. Used to:
  - Evaluate study designs before implementation.
  - Optimise sampling times (D-optimal, A-optimal, E-optimal).
  - Compute expected parameter uncertainty prior to a clinical trial.

The population FIM under the FOCE (first-order conditional estimation)
approximation is:

    FIM(theta, omega, sigma) = sum_i M_i(xi, theta)

where M_i is the individual information matrix for subject i with design
xi (sampling times).

Under the FO approximation (linearisation at ETA = 0), the individual
information matrix simplifies to:

    M_i = (G_i)^T * V_i^{-1} * G_i

where:
    G_i = d f_i / d theta    (Jacobian of predictions w.r.t. fixed effects)
    V_i = Z_i * OMEGA * Z_i^T + SIGMA   (marginal variance of observations)
    Z_i = d f_i / d eta      (Jacobian of predictions w.r.t. random effects)

References:
    Mentré, F., Mallet, A., & Baccar, D. (1997). Optimal design in random-
        effects regression models. Biometrika 84(2):429-442.
    Dumont, C. et al. (2014). PFIM 4.0, an extended R-based program for
        design evaluation and optimisation in nonlinear mixed-effects models.
        Comput. Methods Programs Biomed. 117(2):247-258.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize




@dataclass
class DesignResult:
    """
    Result of an optimal design computation.

    Attributes:
        sampling_times:     Optimal (or evaluated) sampling times, shape (n_samples,).
        information_matrix: Population Fisher Information Matrix, shape (p, p)
                            where p = n_theta.
        d_efficiency:       D-efficiency relative to a reference design,
                            computed as (det(FIM)^(1/p)) / (det(FIM_ref)^(1/p)).
                            Set to NaN when computed for a single design.
        a_efficiency:       A-criterion value = trace(FIM^{-1}).
                            Smaller is better (more informative).
        condition_number:   Condition number of the FIM. Large values indicate
                            near-collinearity between parameters.
        se_theta:           Expected standard errors for each THETA parameter,
                            computed as sqrt(diag(FIM^{-1})), shape (n_theta,).
    """

    sampling_times: np.ndarray
    information_matrix: np.ndarray
    d_efficiency: float
    a_efficiency: float
    condition_number: float
    se_theta: np.ndarray

    def summary(self) -> str:
        """Return a formatted text summary of the design result."""
        p = len(self.sampling_times)
        lines = [
            f"Design Result ({p} sampling times)",
            f"  Times:           {np.round(self.sampling_times, 3).tolist()}",
            f"  D-efficiency:    {self.d_efficiency:.6f}",
            f"  A-criterion:     {self.a_efficiency:.6f}",
            f"  Condition #:     {self.condition_number:.4g}",
            f"  Expected SE(θ):  {np.round(self.se_theta, 4).tolist()}",
        ]
        return "\n".join(lines)


def _pk_callable_jacobian(
    pk_callable: Any,
    theta: list,
    eta: list,
    required: Any,
    base_covariates: dict,
    perturb: str,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Central finite-difference Jacobian of pk_callable outputs.

    Differentiates w.r.t. either ``theta`` or ``eta`` (controlled by
    ``perturb='theta'`` / ``perturb='eta'``).  The pk_callable is called
    at nominal + step and nominal - step for each parameter dimension;
    no ODE integration is involved.

    Args:
        pk_callable:      Callable ``(theta, eta, t, covariates) -> dict``.
        theta:            Nominal theta list.
        eta:              Nominal eta list.
        required:         Ordered sequence of pk_callable output keys to extract.
        base_covariates:  Covariate dict forwarded to pk_callable.
        perturb:          ``'theta'`` or ``'eta'`` — which vector to differentiate.
        eps:              Central-difference step size.

    Returns:
        Jacobian array of shape ``(len(required), len(perturbed_params))``.
    """
    params = theta if perturb == "theta" else eta
    n_params = len(params)
    n_req = len(required)
    J = np.zeros((n_req, n_params))
    for j in range(n_params):
        p_plus = list(params)
        p_minus = list(params)
        p_plus[j] += eps
        p_minus[j] -= eps
        try:
            if perturb == "theta":
                pp = pk_callable(p_plus, eta, t=0.0, covariates=base_covariates)
                pm = pk_callable(p_minus, eta, t=0.0, covariates=base_covariates)
            else:
                pp = pk_callable(theta, p_plus, t=0.0, covariates=base_covariates)
                pm = pk_callable(theta, p_minus, t=0.0, covariates=base_covariates)
        except Exception:
            continue
        for k, name in enumerate(required):
            J[k, j] = (float(pp.get(name, 0.0)) - float(pm.get(name, 0.0))) / (2.0 * eps)
    return J


class PFIMEngine:
    """
    Population Fisher Information Matrix computation and optimal design.

    Evaluates or optimises the expected information content of a sampling
    design for a given population PK/PD model. Supports D-, A-, and
    E-optimal design criteria.

    The FIM is computed using the FO linearisation approach:
      1. Numerical Jacobian G_i = df/dtheta at each sampling time.
      2. Marginal covariance V_i = Z_i * OMEGA * Z_i^T + sigma_diag * I.
      3. Individual information M_i = G_i^T * V_i^{-1} * G_i.
      4. Population FIM = sum_i M_i (summed over n_subjects identical subjects).

    When population_model is None (e.g. in unit tests), only the structural
    methods can be used; compute_fim and optimize_design require a non-None
    population_model.

    Args:
        population_model: Assembled PopulationModel with PK callable.
                          May be None for basic instantiation.
        init_params:      ParameterSet with THETA, OMEGA, SIGMA.
                          May be None for basic instantiation.
        sampling_times:   Default sampling times used when none are specified
                          explicitly in compute_fim / optimize_design.
    """

    def __init__(
        self,
        population_model: Any,
        init_params: Any,
        sampling_times: np.ndarray | None = None,
    ) -> None:
        self.population_model = population_model
        self.init_params = init_params
        self.sampling_times = (
            np.asarray(sampling_times, dtype=float) if sampling_times is not None else np.array([])
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_fim(
        self,
        sampling_times: np.ndarray,
        n_subjects: int = 1,
        omega: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the population Fisher Information Matrix (FIM).

        FIM = n_subjects * M_i(sampling_times)

        where M_i = G_i^T * V_i^{-1} * G_i with:
          G_i = d predictions / d theta  (numerical Jacobian, n_times x n_theta)
          V_i = Z_i * OMEGA * Z_i^T + sigma_diag * I

        Args:
            sampling_times: Array of sampling times, shape (n_samples,).
            n_subjects:     Number of (assumed identical) subjects. The FIM is
                            multiplied by this factor (larger studies → more info).
            omega:          OMEGA matrix override. If None, uses init_params.omega.
            sigma:          SIGMA matrix override. If None, uses init_params.sigma.

        Returns:
            Population FIM, shape (n_theta, n_theta).

        Raises:
            RuntimeError: If population_model or init_params is None.
        """
        if self.population_model is None or self.init_params is None:
            raise RuntimeError("compute_fim requires non-None population_model and init_params.")

        sampling_times = np.asarray(sampling_times, dtype=float)
        theta = self.init_params.theta
        n_theta = len(theta)

        omega_mat = omega if omega is not None else self.init_params.omega
        sigma_mat = sigma if sigma is not None else self.init_params.sigma
        sigma_diag = float(sigma_mat[0, 0]) if sigma_mat.size > 0 else 1.0
        n_times = len(sampling_times)

        if n_times == 0:
            return np.zeros((n_theta, n_theta))

        n_eta = omega_mat.shape[0]

        # Try the native CVODES forward-sensitivity path first.
        # This computes G and Z together in a single ODE solve + cheap pk_callable FDs,
        # replacing 2×(n_theta + n_eta) full ODE solves with finite differences.
        native_GZ = None
        try:
            sid0 = next(iter(self.population_model.subject_ids()))
            indiv0 = self.population_model.individual_model(sid0)
            native_GZ = self._compute_G_and_Z_native(sampling_times, theta, indiv0, n_eta)
        except Exception:
            native_GZ = None

        if native_GZ is not None:
            G, Z = native_GZ
        else:
            # Fall back to central finite-difference loops
            G = self._numerical_gradient_prediction(sampling_times, theta)
            Z = self._numerical_gradient_eta(sampling_times, theta, n_eta)

        # Marginal covariance V_i = Z * OMEGA * Z^T + sigma_diag * I
        V = Z @ omega_mat @ Z.T + sigma_diag * np.eye(n_times)
        # Regularise for numerical stability
        V += 1e-10 * np.eye(n_times)

        try:
            V_inv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            V_inv = np.linalg.pinv(V)

        # Individual FIM: M_i = G^T * V^{-1} * G
        M_i = G.T @ V_inv @ G  # (n_theta, n_theta)

        # Population FIM: sum over subjects (identical subjects assumed)
        fim = n_subjects * M_i
        return fim

    def optimize_design(
        self,
        n_samples: int,
        t_min: float = 0.0,
        t_max: float = 24.0,
        n_subjects: int = 10,
        criterion: str = "D",
        method: str = "differential_evolution",
        n_starts: int = 10,
    ) -> DesignResult:
        """
        Find optimal sampling times for the given criterion.

        Optimisation criteria:
          'D': Maximise det(FIM)^(1/p)  — D-optimal, best overall precision.
          'A': Minimise trace(FIM^{-1}) — A-optimal, minimise avg variance.
          'E': Maximise min eigenvalue  — E-optimal, guard against collinearity.

        Args:
            n_samples:  Number of sampling times to optimise.
            t_min:      Earliest allowed sampling time.
            t_max:      Latest allowed sampling time.
            n_subjects: Number of subjects (scales the FIM).
            criterion:  Optimality criterion: 'D', 'A', or 'E'.
            method:     Optimisation method: 'differential_evolution' or
                        'L-BFGS-B'. Differential evolution is preferred for
                        avoiding local optima.
            n_starts:   Number of random restarts when method='L-BFGS-B'.

        Returns:
            DesignResult with optimal sampling times and FIM properties.
        """
        if self.population_model is None or self.init_params is None:
            raise RuntimeError(
                "optimize_design requires non-None population_model and init_params."
            )

        bounds = [(t_min, t_max)] * n_samples
        criterion_upper = criterion.upper()

        def neg_criterion(times_flat: np.ndarray) -> float:
            times = np.sort(times_flat)
            fim = self.compute_fim(times, n_subjects=n_subjects)
            return -self._criterion_value(fim, criterion_upper)

        if method == "differential_evolution":
            de_result = differential_evolution(
                neg_criterion,
                bounds=bounds,
                seed=42,
                maxiter=300,
                tol=1e-8,
                popsize=15,
                polish=True,
            )
            best_times = np.sort(de_result.x)
            best_val = -float(de_result.fun)
        else:
            # Multi-start L-BFGS-B
            best_val = -np.inf
            best_times = np.linspace(t_min, t_max, n_samples)
            rng = np.random.default_rng(42)
            for _ in range(n_starts):
                t0 = rng.uniform(t_min, t_max, n_samples)
                lbfgs_result = minimize(
                    neg_criterion,
                    t0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-10},
                )
                val = -float(lbfgs_result.fun)
                if val > best_val:
                    best_val = val
                    best_times = np.sort(lbfgs_result.x)

        fim_opt = self.compute_fim(best_times, n_subjects=n_subjects)
        se_theta = self._se_from_fim(fim_opt)
        a_eff = self._a_criterion(fim_opt)
        cond = self._condition_number(fim_opt)

        return DesignResult(
            sampling_times=best_times,
            information_matrix=fim_opt,
            d_efficiency=float("nan"),  # relative only when comparing two designs
            a_efficiency=a_eff,
            condition_number=cond,
            se_theta=se_theta,
        )

    def efficiency(
        self,
        times_test: np.ndarray,
        times_reference: np.ndarray,
        criterion: str = "D",
        n_subjects: int = 1,
    ) -> float:
        """
        Compute relative efficiency of a test design vs a reference design.

        For D-criterion:
            eff = [det(FIM_test) / det(FIM_ref)]^(1/p)

        For A-criterion:
            eff = trace(FIM_ref^{-1}) / trace(FIM_test^{-1})
            (> 1 means test is better than reference)

        Args:
            times_test:      Sampling times for the test design.
            times_reference: Sampling times for the reference design.
            criterion:       'D' or 'A'.
            n_subjects:      Number of subjects for FIM computation.

        Returns:
            Relative efficiency scalar. Values > 1 mean test is better.
        """
        fim_test = self.compute_fim(np.asarray(times_test), n_subjects=n_subjects)
        fim_ref = self.compute_fim(np.asarray(times_reference), n_subjects=n_subjects)

        criterion_upper = criterion.upper()
        val_test = self._criterion_value(fim_test, criterion_upper)
        val_ref = self._criterion_value(fim_ref, criterion_upper)

        if criterion_upper == "D":
            # (det_test / det_ref)^(1/p) = exp((log_det_test - log_det_ref) / p)
            p = fim_test.shape[0]
            sign_t, ld_t = np.linalg.slogdet(fim_test)
            sign_r, ld_r = np.linalg.slogdet(fim_ref)
            if sign_t <= 0 or sign_r <= 0:
                return float("nan")
            return float(np.exp((ld_t - ld_r) / p))

        # A or E: ratio of criterion values
        if val_ref == 0:
            return float("nan")
        return float(val_test / val_ref)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_G_and_Z_native(
        self,
        times: np.ndarray,
        theta: np.ndarray,
        indiv: Any,
        n_eta: int,
        eps: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Compute (G, Z) using a single CVODES forward-sensitivity integration.

        This replaces 2×(n_theta + n_eta) ODE solves (finite-difference loops)
        with **one** ODE solve that simultaneously tracks dA/d(ODE_param_j) for
        all template parameters, followed by cheap finite-difference evaluations
        of only the pk_callable (no ODE integration).

        The chain rule applied is::

            G[t, j] = Σ_k  (dF/d(ODE_param_k)) · (d(ODE_param_k)/d(pop_theta_j))
            Z[t, k] = Σ_m  (dF/d(ODE_param_m)) · (d(ODE_param_m)/d(eta_k))

        where F = A_out / V is the predicted concentration from the matched
        template's output compartment divided by its volume parameter.

        Template matching follows the same rules as
        ``IndividualModel.native_advan6_prediction_eta_jacobian``: the template's
        ``n_states`` must equal the contract's ``n_compartments``, the template
        must have a sensitivity probe, it must not be a mixed PK/PD model, and
        every required parameter name must appear in the pk_callable output.

        Args:
            times:  Design sampling times (need not be sorted).
            theta:  Population THETA vector.
            indiv:  First-subject IndividualModel instance.
            n_eta:  Number of random effects.
            eps:    Step size for pk_callable finite differences.

        Returns:
            (G, Z) tuple with shapes (n_times, n_theta) and (n_times, n_eta),
            or ``None`` if the native path is unavailable for this model.
        """
        contract = getattr(indiv, "_native_ode_contract", None)
        if contract is None:
            return None
        if contract.get("is_pkpd", False):
            return None
        has_infusion = contract.get("has_infusion", False)

        pk_callable = getattr(indiv, "pk_callable", None)
        if pk_callable is None:
            return None

        n_times = len(times)
        n_theta = len(theta)
        theta_list = list(float(t) for t in theta)
        eta_zero = [0.0] * n_eta
        base_covariates = getattr(indiv, "_base_covariates", {})

        # ── Step 1: evaluate pk_callable at nominal (theta, eta=0) ───────────
        try:
            pk_params_0 = pk_callable(theta_list, eta_zero, t=0.0, covariates=base_covariates)
        except Exception:
            return None

        # ── Step 2: template matching ─────────────────────────────────────────
        n_cmt = contract.get("n_compartments", -1)
        template = self._match_native_template(pk_params_0, n_cmt)
        if template is None:
            return None

        required = template.required_names
        n_ode_params = len(required)
        n_states = template.n_states
        out_idx = template.output_cmt_idx
        vol_name = template.vol_param_name
        v_idx = list(required).index(vol_name)
        V = float(pk_params_0[vol_name])
        ode_theta = [float(pk_params_0[name]) for name in required]

        from openpkpd.model.individual import _apply_alag
        dose_times = _apply_alag(
            contract["dose_times"], contract["dose_compartments"], pk_params_0
        )

        # ── Step 3: run sensitivity probe (one ODE solve) ─────────────────────
        # Probe requires sorted observation times.
        order = np.argsort(times, kind="stable")
        sorted_times = times[order]
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(n_times)

        if has_infusion:
            # Infusion sensitivity: needs the infusion-aware sens probe.
            if template.infusion_sens_probe_fn is None:
                return None
        elif template.sens_probe_fn is None:
            return None

        try:
            if has_infusion:
                states_raw, sens_raw = template.infusion_sens_probe_fn(
                    sorted_times.tolist(),
                    dose_times,
                    contract["dose_amts"],
                    contract["dose_rates"],
                    ode_theta,
                )
            else:
                states_raw, sens_raw = template.sens_probe_fn(
                    sorted_times.tolist(),
                    dose_times,
                    contract["dose_amts"],
                    ode_theta,
                )
        except Exception:
            return None

        states = np.array(states_raw, dtype=float)                         # (n_times, n_states)
        # sens[t, j, i] = dA_i / d(ODE_param_j)
        sens = np.array(sens_raw, dtype=float).reshape(n_times, n_ode_params, n_states)

        # ── Step 4: output-function Jacobian dF/d(ODE_param_j) ───────────────
        # F = A_out / V  →  dF/d(ODE_param_j) = sens[:,j,out_idx]/V  for j ≠ vol
        #                                       = sens[:,v_idx,out_idx]/V − A_out/V²  for j == vol
        A_out = states[:, out_idx]                      # (n_times,)
        dF_dODE = sens[:, :, out_idx] / V              # (n_times, n_ode_params)
        dF_dODE[:, v_idx] -= A_out / (V * V)           # quotient rule for volume parameter

        # ── Steps 5 & 6: FD of pk_callable w.r.t. pop theta and eta ─────────
        # d(ODE_param_k)/d(pop_theta_j) and d(ODE_param_k)/d(eta_k) — cheap, no ODE.
        J_pk_theta = _pk_callable_jacobian(
            pk_callable, theta_list, eta_zero, required, base_covariates, "theta", eps
        )
        G = dF_dODE @ J_pk_theta   # (n_times, n_theta)

        J_pk_eta = _pk_callable_jacobian(
            pk_callable, theta_list, eta_zero, required, base_covariates, "eta", eps
        )
        Z = dF_dODE @ J_pk_eta     # (n_times, n_eta)

        # Restore original time ordering
        return G[inverse_order], Z[inverse_order]

    def _match_native_template(self, pk_params_0: dict, n_cmt: int) -> Any:
        """
        Return the first native ODE template that matches this model, or None.

        Matching criteria (identical to IndividualModel.native_advan6_prediction_eta_jacobian):
          - Template has a sensitivity probe (``sens_probe_fn`` is not None).
          - Template is not a mixed PK/PD model (``is_pkpd=False``).
          - Template compartment count equals ``n_cmt``.
          - All of the template's ``required_names`` appear in ``pk_params_0``.
        """
        from openpkpd.model.individual import _NATIVE_ODE_TEMPLATES  # local import avoids circular ref

        for tmpl in _NATIVE_ODE_TEMPLATES:
            if tmpl.sens_probe_fn is None or tmpl.is_pkpd:
                continue
            if tmpl.n_states != n_cmt:
                continue
            if any(name not in pk_params_0 for name in tmpl.required_names):
                continue
            return tmpl
        return None

    def _numerical_gradient_prediction(
        self,
        times: np.ndarray,
        theta: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """
        Compute Jacobian df/dtheta by central finite differences.

        Evaluates the PK model at each sampling time for the current
        population parameters. The model is evaluated via the first subject's
        individual model (as a representative average subject at eta=0).

        Args:
            times: Sampling times, shape (n_times,).
            theta: THETA parameter vector.
            eps:   Finite difference step size.

        Returns:
            Jacobian G, shape (n_times, n_theta).
        """
        n_theta = len(theta)
        n_times = len(times)
        G = np.zeros((n_times, n_theta))

        # Use first subject as representative
        try:
            sid0 = next(iter(self.population_model.subject_ids()))
            indiv = self.population_model.individual_model(sid0)
        except (StopIteration, AttributeError):
            return G

        n_eta = self.init_params.omega.shape[0]
        eta_zero = np.zeros(n_eta)
        sigma = self.init_params.sigma

        def predict_at_theta(th: np.ndarray) -> np.ndarray:
            """Return predictions at all sampling times."""
            try:
                return self._interpolate_predictions(indiv, th, eta_zero, sigma, times)
            except Exception:
                return np.zeros(n_times)

        for j in range(n_theta):
            dth = np.zeros(n_theta)
            dth[j] = eps
            f_plus = predict_at_theta(theta + dth)
            f_minus = predict_at_theta(theta - dth)
            G[:, j] = (f_plus - f_minus) / (2.0 * eps)

        return G

    def _numerical_gradient_eta(
        self,
        times: np.ndarray,
        theta: np.ndarray,
        n_eta: int,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """
        Compute Jacobian df/deta by central finite differences at eta=0.

        Args:
            times: Sampling times, shape (n_times,).
            theta: THETA parameter vector.
            n_eta: Number of random effects.
            eps:   Finite difference step size.

        Returns:
            Jacobian Z, shape (n_times, n_eta).
        """
        n_times = len(times)
        Z = np.zeros((n_times, n_eta))

        if n_eta == 0:
            return Z

        try:
            sid0 = next(iter(self.population_model.subject_ids()))
            indiv = self.population_model.individual_model(sid0)
        except (StopIteration, AttributeError):
            return Z

        sigma = self.init_params.sigma

        for k in range(n_eta):
            deta = np.zeros(n_eta)
            deta[k] = eps
            f_plus = self._interpolate_predictions(indiv, theta, deta, sigma, times)
            f_minus = self._interpolate_predictions(indiv, theta, -deta, sigma, times)
            Z[:, k] = (f_plus - f_minus) / (2.0 * eps)

        return Z

    def _interpolate_predictions(
        self,
        indiv: Any,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the individual model at arbitrary target times.

        When possible, the individual model is re-evaluated directly at the
        requested design times by cloning the subject event structure with the
        new observation grid. If that is not possible, this falls back to the
        legacy interpolation behavior over the subject's original observation
        times.

        Args:
            indiv:        Individual model object.
            theta:        THETA vector.
            eta:          ETA vector.
            sigma:        SIGMA matrix.
            target_times: Requested output times.

        Returns:
            Predicted concentrations at target_times, shape (n_target,).
        """
        target_times = np.asarray(target_times, dtype=float)
        n_target = len(target_times)
        if n_target == 0:
            return np.array([])

        order = np.argsort(target_times, kind="stable")
        sorted_times = target_times[order]
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(n_target)

        try:
            pred_sorted = self._evaluate_predictions_direct(
                indiv,
                theta,
                eta,
                sigma,
                sorted_times,
            )
            if len(pred_sorted) == n_target:
                return np.maximum(pred_sorted, 0.0)[inverse_order]
        except Exception:
            pass

        try:
            _, _, f_full = indiv.evaluate(
                theta,
                eta,
                sigma,
                trans=self.population_model.trans,
            )
            obs_times = np.asarray(indiv.subject_events.obs_times, dtype=float)
            f_full = np.asarray(f_full, dtype=float)
            if len(obs_times) == 0 or len(f_full) == 0:
                return np.zeros(n_target)
            pred_out = np.interp(
                target_times,
                obs_times,
                f_full,
                left=f_full[0],
                right=f_full[-1],
            )
            return np.maximum(pred_out, 0.0)
        except Exception:
            return np.zeros(n_target)

    def _evaluate_predictions_direct(
        self,
        indiv: Any,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """Directly evaluate predictions on a temporary observation grid."""
        subject_events = getattr(indiv, "subject_events", None)
        if subject_events is None:
            raise AttributeError("individual model has no subject_events")

        target_times = np.asarray(target_times, dtype=float)
        source_times = np.asarray(getattr(subject_events, "obs_times", []), dtype=float)
        n_target = len(target_times)

        subject_events_eval = copy(subject_events)
        subject_events_eval.obs_times = target_times.copy()
        subject_events_eval.obs_dv = np.full(n_target, np.nan, dtype=float)
        subject_events_eval.obs_mdv = np.zeros(n_target, dtype=int)

        source_cmt = np.asarray(getattr(subject_events, "obs_cmt", []), dtype=int)
        if len(source_cmt) == len(source_times) and len(source_cmt) > 0:
            subject_events_eval.obs_cmt = self._map_by_time(
                source_times,
                source_cmt,
                target_times,
                default_value=int(source_cmt[0]),
                dtype=int,
            )
        else:
            default_cmt = int(source_cmt[0]) if len(source_cmt) > 0 else 1
            subject_events_eval.obs_cmt = np.full(n_target, default_cmt, dtype=int)

        source_occ = getattr(indiv, "occasion_indices", None)
        if source_occ is not None:
            source_occ_arr = np.asarray(source_occ, dtype=int)
            if len(source_occ_arr) == len(source_times) and len(source_occ_arr) > 0:
                mapped_occ = self._map_by_time(
                    source_times,
                    source_occ_arr,
                    target_times,
                    default_value=int(source_occ_arr[0]),
                    dtype=int,
                )
            else:
                mapped_occ = np.zeros(n_target, dtype=int)
            subject_events_eval.occasion_indices = mapped_occ

        indiv_eval = copy(indiv)
        indiv_eval.subject_events = subject_events_eval
        if hasattr(indiv_eval, "occasion_indices"):
            indiv_eval.occasion_indices = getattr(subject_events_eval, "occasion_indices", None)

        _, _, f_target = indiv_eval.evaluate(
            theta,
            eta,
            sigma,
            trans=self.population_model.trans,
        )
        return np.asarray(f_target, dtype=float)

    def _map_by_time(
        self,
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray,
        default_value: int,
        dtype: type,
    ) -> np.ndarray:
        """Map per-observation metadata to new times using LOCF semantics."""
        if len(source_times) == 0 or len(source_values) == 0:
            return np.full(len(target_times), default_value, dtype=dtype)

        idx = np.searchsorted(source_times, target_times, side="right") - 1
        idx = np.clip(idx, 0, len(source_values) - 1)
        return np.asarray(source_values[idx], dtype=dtype)

    # ------------------------------------------------------------------
    # Criterion helpers
    # ------------------------------------------------------------------

    def _criterion_value(self, fim: np.ndarray, criterion: str) -> float:
        """
        Compute the scalar criterion value for a given FIM.

        Args:
            fim:       Population FIM, shape (p, p).
            criterion: 'D', 'A', or 'E'.

        Returns:
            Scalar criterion value. Higher is always better (D: det, E: min
            eigenvalue; for A, the negative trace of the inverse is used so
            that maximisation still applies during optimisation).
        """
        if criterion == "D":
            sign, logdet = np.linalg.slogdet(fim)
            if sign <= 0:
                return -1e10
            return float(logdet / fim.shape[0])  # (1/p) * log(det)
        elif criterion == "A":
            return -self._a_criterion(fim)  # negate for maximisation
        elif criterion == "E":
            eigvals = np.linalg.eigvalsh(fim)
            return float(eigvals.min())
        else:
            raise ValueError(f"Unknown criterion {criterion!r}. Choose 'D', 'A', or 'E'.")

    def _a_criterion(self, fim: np.ndarray) -> float:
        """
        Compute the A-criterion = trace(FIM^{-1}).

        Returns inf if FIM is singular.
        """
        try:
            fim_inv = np.linalg.inv(fim)
            return float(np.trace(fim_inv))
        except np.linalg.LinAlgError:
            return float("inf")

    def _condition_number(self, fim: np.ndarray) -> float:
        """
        Compute the condition number of FIM (ratio of max to min eigenvalue).

        Returns inf if FIM is singular or min eigenvalue is 0.
        """
        eigvals = np.linalg.eigvalsh(fim)
        min_ev = float(eigvals.min())
        max_ev = float(eigvals.max())
        if min_ev <= 0:
            return float("inf")
        return float(max_ev / min_ev)

    def _se_from_fim(self, fim: np.ndarray) -> np.ndarray:
        """
        Compute expected standard errors from FIM^{-1}.

        SE(theta_j) = sqrt(FIM^{-1}[j, j])

        Returns zeros for singular FIM.
        """
        p = fim.shape[0]
        try:
            fim_inv = np.linalg.inv(fim)
            diag = np.maximum(np.diag(fim_inv), 0.0)
            return np.sqrt(diag)
        except np.linalg.LinAlgError:
            return np.zeros(p)
