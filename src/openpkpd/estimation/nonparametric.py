"""
Nonparametric estimation for population PK/PD models.

Builds a discrete probability distribution over support points derived
from empirical Bayes estimates (from FOCE/FO). Makes no parametric
assumption about the shape of the ETA distribution.

The algorithm is based on the nonparametric maximum likelihood (NPML)
approach of Mallet (1986) extended by Schumitzky (1991) and implemented
similarly to the NONMEM NONPARAMETRIC option.

Algorithm:
  1. Run FO or FOCE to obtain population parameters (THETA, SIGMA) and
     individual empirical Bayes estimates (EBEs) for each subject.
  2. Use EBEs as initial support points for the discrete distribution.
  3. Optimise the probability mass on the support points to maximise the
     marginal log-likelihood using an EM (Expectation-Maximisation) algorithm.
  4. Report the final discrete distribution (support points + weights).

References:
    Mallet, A. (1986). A maximum likelihood estimation method for random
        coefficient regression models. Biometrika 73(3):645-656.
    Schumitzky, A. (1991). Nonparametric EM algorithms for estimating prior
        distributions. Appl. Math. Comput. 45(2-3):143-157.
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpkpd.estimation.base import EstimationMethod, EstimationResult
from openpkpd.utils.logging import get_logger

logger = get_logger("estimation.nonparametric")

# ---------------------------------------------------------------------------
# Extended result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NonparametricResult(EstimationResult):
    """
    Result from nonparametric estimation.

    Attributes:
        support_points:  ETA values at each support point.
                         Shape (n_support, n_eta).
        support_weights: Probability mass at each support point.
                         Shape (n_support,). Sums to 1.
    """

    support_points: np.ndarray = field(default_factory=lambda: np.array([]))
    support_weights: np.ndarray = field(default_factory=lambda: np.array([]))

    def empirical_mean(self) -> np.ndarray:
        """
        Compute the empirical mean ETA across support points.

        Returns:
            Array of shape (n_eta,).
        """
        if self.support_points.ndim < 2 or len(self.support_weights) == 0:
            return np.array([])
        return np.sum(self.support_weights[:, np.newaxis] * self.support_points, axis=0)

    def empirical_variance(self) -> np.ndarray:
        """
        Compute the empirical variance of each ETA across support points.

        Returns:
            Array of shape (n_eta,).
        """
        if self.support_points.ndim < 2 or len(self.support_weights) == 0:
            return np.array([])
        mean = self.empirical_mean()
        var = np.sum(
            self.support_weights[:, np.newaxis] * (self.support_points - mean) ** 2,
            axis=0,
        )
        return var

    def summary(self) -> str:
        """Return a text summary of the nonparametric result."""
        lines = [
            f"Method: {self.method}",
            f"OFV: {self.ofv:.4f}",
            f"Converged: {self.converged}",
            f"n_support_points: {len(self.support_weights)}",
            f"THETA: {self.theta_final}",
        ]
        if len(self.support_weights) > 0:
            lines.append("Support point distribution:")
            for k, (w, sp) in enumerate(
                zip(self.support_weights, self.support_points, strict=False)
            ):
                lines.append(f"  SP{k + 1}: weight={w:.4f}  ETA={sp}")
            mean_eta = self.empirical_mean()
            var_eta = self.empirical_variance()
            lines.append(f"Empirical ETA mean:     {mean_eta}")
            lines.append(f"Empirical ETA variance: {var_eta}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NonparametricMethod
# ---------------------------------------------------------------------------


class NonparametricMethod(EstimationMethod):
    """
    Nonparametric estimation using support point approximation.

    Uses a base parametric method (default: FOCE) to obtain population
    parameters and empirical Bayes estimates (EBEs). The EBEs serve as
    the support points for a discrete probability distribution over the
    ETA space. Weights are optimised by an EM algorithm to maximise the
    marginal log-likelihood.

    This approach is distribution-free: the inter-individual variability
    (IIV) distribution is represented as an arbitrary discrete probability
    mass function rather than a parametric (e.g. normal) distribution.
    The final OMEGA matrix reported is the empirical covariance of the
    support point distribution.

    Args:
        base_method:       Parametric method used in step 1 to obtain EBEs.
                           Must be one of: 'FO', 'FOCE', 'FOCEI'.
        n_support_points:  Number of support points. If None (default), uses
                           the number of subjects (one support point per
                           subject, equal to the initial EBEs).
        max_iter:          Maximum EM iterations for weight optimisation.
        tol:               Convergence tolerance on the change in log-likelihood.
        **kwargs:          Extra keyword arguments passed to the base method.

    Usage::

        method = NonparametricMethod(base_method="FOCE", max_iter=200)
        result = method.estimate(population_model, init_params)
        print(result.summary())
    """

    method_name = "NONPARAMETRIC"

    def __init__(
        self,
        base_method: str = "FOCE",
        n_support_points: int | None = None,
        max_iter: int = 100,
        tol: float = 1e-5,
        n_parallel: int = 1,
        **kwargs: Any,
    ) -> None:
        self.base_method = base_method
        self.n_support_points = n_support_points
        self.max_iter = max_iter
        self.tol = tol
        self.n_parallel = n_parallel
        self.kwargs = kwargs

    # ------------------------------------------------------------------
    # EstimationMethod interface
    # ------------------------------------------------------------------

    def estimate(
        self,
        population_model: Any,
        init_params: Any,
        **kwargs: Any,
    ) -> NonparametricResult:
        """
        Nonparametric estimation in two steps.

        Step 1: Run the base parametric method to obtain THETA, SIGMA,
                and per-subject EBEs (empirical Bayes estimates).
        Step 2: Treat EBEs as support points and optimise their weights
                using an EM algorithm.

        Args:
            population_model: Assembled PopulationModel with dataset.
            init_params:      Initial ParameterSet.
            **kwargs:         Overrides for base method keyword arguments.

        Returns:
            NonparametricResult with support points, weights, and the
            parametric parameters (THETA, SIGMA) from step 1.
        """
        from openpkpd.estimation import get_estimation_method

        t0 = time.time()

        # ---- Step 1: Run base method -----------------------------------
        merged_kwargs = dict(self.kwargs)
        merged_kwargs.update(kwargs)
        base = get_estimation_method(self.base_method, **merged_kwargs)
        base_result = base.estimate(population_model, init_params)

        # Extract EBEs as support points
        ebe_dict = base_result.post_hoc_etas  # {subject_id: eta_vector}
        if not ebe_dict:
            warnings.warn(
                "Base method returned no post-hoc ETAs. Nonparametric step will use zero ETAs.",
                UserWarning,
                stacklevel=2,
            )
            n_eta = init_params.omega.shape[0]
            subject_ids = list(population_model.subject_ids())
            ebe_dict = {sid: np.zeros(n_eta) for sid in subject_ids}

        subject_ids_ordered = list(ebe_dict.keys())
        support_points_init = np.array(
            [ebe_dict[sid] for sid in subject_ids_ordered]
        )  # shape (N, n_eta)

        n_support = (
            self.n_support_points if self.n_support_points is not None else len(subject_ids_ordered)
        )

        # If fewer subjects than requested support points, truncate and log
        if n_support > len(subject_ids_ordered):
            logger.info(
                "Nonparametric: requested %d support points but only %d subjects available; "
                "truncating to %d.",
                n_support, len(subject_ids_ordered), len(subject_ids_ordered),
            )
            n_support = len(subject_ids_ordered)

        support_points = support_points_init[:n_support]

        # ---- Step 2: Optimise weights via EM ---------------------------
        weights = self._optimize_weights(
            support_points=support_points,
            population_model=population_model,
            init_params=init_params,
            base_result=base_result,
        )

        # Compute empirical OMEGA from support distribution
        mean_eta = np.sum(weights[:, np.newaxis] * support_points, axis=0)
        omega_np = np.sum(
            weights[:, np.newaxis, np.newaxis]
            * (
                (support_points - mean_eta)[:, :, np.newaxis]
                * (support_points - mean_eta)[:, np.newaxis, :]
            ),
            axis=0,
        )

        elapsed = time.time() - t0

        return NonparametricResult(
            theta_final=base_result.theta_final,
            omega_final=omega_np,
            sigma_final=base_result.sigma_final,
            ofv=base_result.ofv,
            converged=base_result.converged,
            post_hoc_etas=base_result.post_hoc_etas,
            ofv_history=base_result.ofv_history,
            n_function_evals=base_result.n_function_evals,
            elapsed_time=elapsed,
            method=self.method_name,
            message=base_result.message,
            support_points=support_points,
            support_weights=weights,
        )

    # ------------------------------------------------------------------
    # Internal: EM weight optimisation
    # ------------------------------------------------------------------

    def _optimize_weights(
        self,
        support_points: np.ndarray,
        population_model: Any,
        init_params: Any,
        base_result: EstimationResult,
    ) -> np.ndarray:
        """
        Optimise discrete probability weights over support points to
        maximise the marginal log-likelihood.

        Objective:
            max_{w} sum_i log( sum_k w_k * L(y_i | eta_k, theta) )
            subject to: sum_k w_k = 1,  w_k >= 0

        Uses the EM algorithm:
          E-step: For each subject i and support point k, compute the
                  posterior probability p_ik = w_k * L_ik / sum_j w_j * L_ij.
          M-step: Update weights w_k = mean_i(p_ik).

        Args:
            support_points:   Initial support points (EBEs), shape (K, n_eta).
            population_model: PopulationModel.
            init_params:      ParameterSet at convergence.
            base_result:      EstimationResult from the parametric base method.

        Returns:
            Optimal weights, shape (K,), summing to 1.
        """
        K = support_points.shape[0]

        # Uniform initial weights
        weights = np.ones(K) / K

        # Pre-compute individual likelihoods L_ik for each subject i at
        # each support point k. L_ik = p(y_i | eta_k, theta).
        subject_ids = list(population_model.subject_ids())

        # Build likelihood matrix (N x K)
        L_matrix = self._compute_likelihood_matrix(
            support_points=support_points,
            population_model=population_model,
            init_params=init_params,
            base_result=base_result,
            subject_ids=subject_ids,
        )  # shape (N, K)

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # E-step: posterior responsibilities
            # numerator_ik = w_k * L_ik
            numerator = weights[np.newaxis, :] * L_matrix  # (N, K)
            row_sums = numerator.sum(axis=1, keepdims=True)  # (N, 1)

            # Avoid division by zero
            row_sums = np.maximum(row_sums, 1e-300)
            responsibilities = numerator / row_sums  # (N, K)

            # M-step: update weights
            weights = responsibilities.mean(axis=0)  # (K,)
            # Normalise (should already sum to 1, but enforce numerically)
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            else:
                weights = np.ones(K) / K

            # Compute marginal log-likelihood for convergence check
            marginal = (weights[np.newaxis, :] * L_matrix).sum(axis=1)
            marginal = np.maximum(marginal, 1e-300)
            ll = float(np.sum(np.log(marginal)))

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return weights

    def _compute_likelihood_matrix(
        self,
        support_points: np.ndarray,
        population_model: Any,
        init_params: Any,
        base_result: EstimationResult,
        subject_ids: list[Any],
    ) -> np.ndarray:
        """
        Compute the likelihood matrix L[i, k] = p(y_i | eta_k, theta).

        Uses a Gaussian likelihood with the residual variance from SIGMA.

        Args:
            support_points:   ETA values at each support point, shape (K, n_eta).
            population_model: PopulationModel.
            init_params:      ParameterSet (provides theta, sigma).
            base_result:      FOCE result (provides theta_final, sigma_final).
            subject_ids:      Ordered list of subject identifiers.

        Returns:
            Likelihood matrix of shape (N, K).
        """
        N = len(subject_ids)
        K = support_points.shape[0]
        L_matrix = np.zeros((N, K))

        theta = base_result.theta_final
        sigma = base_result.sigma_final
        sigma_diag = float(sigma[0, 0]) if sigma.size > 0 else 1.0

        def _compute_subject_row(sid: Any) -> np.ndarray:
            row = np.zeros(K, dtype=float)
            try:
                indiv = population_model.individual_model(sid)
                subj_ev = indiv.subject_events
                obs_mask = subj_ev.observation_mask()
                dv = subj_ev.obs_dv[obs_mask]
                if len(dv) == 0:
                    row[:] = 1.0
                    return row

                for k in range(K):
                    eta_k = support_points[k]
                    try:
                        _, _, f = indiv.evaluate(
                            theta,
                            eta_k,
                            sigma,
                            trans=population_model.trans,
                        )
                        f_obs = f[obs_mask]
                        residuals = dv - f_obs
                        # Gaussian log-likelihood (diagonal residual covariance)
                        n_obs = len(dv)
                        log_lik = (
                            -0.5 * n_obs * np.log(2 * np.pi * sigma_diag)
                            - 0.5 * np.sum(residuals**2) / sigma_diag
                        )
                        row[k] = np.exp(np.clip(log_lik, -500, None))
                    except Exception:
                        row[k] = 1e-300
            except Exception:
                row[:] = 1.0 / K
            return row

        if self.n_parallel == 1 or len(subject_ids) <= 1:
            rows = [_compute_subject_row(sid) for sid in subject_ids]
        else:
            n_workers = self.n_parallel if self.n_parallel > 0 else None
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_compute_subject_row, sid): sid for sid in subject_ids}
                rows = []
                for future in futures:
                    sid = futures[future]
                    try:
                        rows.append(future.result())
                    except Exception as e:
                        logger.warning("Nonparametric: worker failed for subject %s: %s", sid, e)
                        rows.append(None)

        for i, row in enumerate(rows):
            if row is None:
                L_matrix[i, :] = 1.0 / K
            else:
                L_matrix[i, :] = row

        # Guard against all-zero rows
        row_sums = L_matrix.sum(axis=1)
        zero_rows = row_sums == 0
        if np.any(zero_rows):
            L_matrix[zero_rows, :] = 1.0 / K

        return L_matrix


# ---------------------------------------------------------------------------
# NPEMMethod — full Nonparametric EM with joint support-point optimisation
# ---------------------------------------------------------------------------


class NPEMMethod(NonparametricMethod):
    """
    Full Nonparametric Expectation-Maximisation (NPEM) with joint
    optimisation of support point **locations** and **weights**.

    Extends :class:`NonparametricMethod` by adding a gradient-based
    M-step that moves each support point to maximise the expected
    complete-data log-likelihood, rather than holding support points fixed
    at the initial EBEs and only optimising the discrete probability
    weights.

    Algorithm (Schumitzky 1991 / Mallet 1986 extended):
      Initialisation: Run base parametric method (FOCE) to get EBEs as
          starting support points, with uniform weights.
      Repeat until convergence:
        E-step: Compute posterior responsibilities r_{ik} =
                w_k * L(y_i | eta_k, theta) / Σ_j w_j * L(y_i | eta_j, theta).
        M-step (weights): w_k = (1/N) Σ_i r_{ik}.
        M-step (locations): For each support point k, update eta_k by
                minimising the weighted negative log-likelihood:
                    eta_k* = argmin_{eta} -Σ_i r_{ik} * log L(y_i | eta, theta)
                using scipy.optimize.minimize (L-BFGS-B).

    Args:
        base_method:        Parametric warm-start method.
        n_support_points:   Number of support points (None = n_subjects).
        max_iter:           Maximum EM iterations.
        max_location_iter:  Maximum scipy optimisation iterations per
                            support-point location update (default 20).
        tol:                Log-likelihood convergence tolerance.
        optimise_locations: If True (default), jointly optimise support
                            point locations.  Set False to recover the
                            behaviour of :class:`NonparametricMethod`.
        **kwargs:           Extra keyword arguments forwarded to the base
                            parametric method.

    Usage::

        from openpkpd.estimation.nonparametric import NPEMMethod

        method = NPEMMethod(base_method="FOCE", n_support_points=20,
                            max_iter=50, optimise_locations=True)
        result = method.estimate(population_model, init_params)
        print(result.summary())
    """

    method_name = "NPEM"

    def __init__(
        self,
        base_method: str = "FOCE",
        n_support_points: int | None = None,
        max_iter: int = 50,
        max_location_iter: int = 20,
        tol: float = 1e-5,
        optimise_locations: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_method=base_method,
            n_support_points=n_support_points,
            max_iter=max_iter,
            tol=tol,
            **kwargs,
        )
        self.max_location_iter = max_location_iter
        self.optimise_locations = optimise_locations

    def estimate(
        self,
        population_model: Any,
        init_params: Any,
        **kwargs: Any,
    ) -> NonparametricResult:
        """
        Run NPEM: joint EM over support point locations and weights.

        Step 1: Warm-start via the base parametric method (FOCE/FO).
        Step 2: Joint E/M iterations over locations and weights.

        Returns:
            :class:`NonparametricResult` with support_points, support_weights,
            and empirical OMEGA.
        """
        from scipy.optimize import minimize

        from openpkpd.estimation import get_estimation_method

        t0 = time.time()

        # ---- Step 1: Warm-start ----------------------------------------
        merged_kwargs = dict(self.kwargs)
        merged_kwargs.update(kwargs)
        base = get_estimation_method(self.base_method, **merged_kwargs)
        base_result = base.estimate(population_model, init_params)

        ebe_dict = base_result.post_hoc_etas or {}
        if not ebe_dict:
            n_eta = init_params.omega.shape[0]
            subject_ids_all = list(population_model.subject_ids())
            ebe_dict = {sid: np.zeros(n_eta) for sid in subject_ids_all}

        subject_ids_ordered = list(ebe_dict.keys())
        support_points_init = np.array([ebe_dict[sid] for sid in subject_ids_ordered])
        N = len(subject_ids_ordered)
        n_support = self.n_support_points if self.n_support_points is not None else N
        n_support = min(n_support, N)
        # Use k-means-like initialisation when n_support < N
        if n_support < N:
            rng = np.random.default_rng(0)
            idx = rng.choice(N, n_support, replace=False)
            support_points = support_points_init[idx].copy()
        else:
            support_points = support_points_init[:n_support].copy()

        K = len(support_points)
        weights = np.ones(K) / K

        theta = base_result.theta_final
        sigma = base_result.sigma_final
        sigma_diag = float(sigma[0, 0]) if sigma.size > 0 else 1.0

        # Cache individual models and observations
        indiv_cache = {}
        dv_cache = {}
        obs_mask_cache = {}
        for sid in subject_ids_ordered:
            try:
                indiv = population_model.individual_model(sid)
                subj_ev = indiv.subject_events
                mask = subj_ev.observation_mask()
                dv = subj_ev.obs_dv[mask]
                indiv_cache[sid] = indiv
                dv_cache[sid] = dv
                obs_mask_cache[sid] = mask
            except Exception:
                pass

        def _subject_log_lik(sid: Any, eta_k: np.ndarray) -> float:
            if sid not in indiv_cache:
                return 0.0
            indiv = indiv_cache[sid]
            dv = dv_cache[sid]
            mask = obs_mask_cache[sid]
            if len(dv) == 0:
                return 0.0
            try:
                _, _, f = indiv.evaluate(theta, eta_k, sigma, trans=population_model.trans)
                f_obs = f[mask]
                residuals = dv - f_obs
                n_obs = len(dv)
                return float(
                    -0.5 * n_obs * np.log(2 * np.pi * sigma_diag)
                    - 0.5 * np.sum(residuals**2) / sigma_diag
                )
            except Exception:
                return -500.0

        def _compute_L(sp: np.ndarray) -> np.ndarray:
            """Compute likelihood matrix (N, K) for current support points."""
            L = np.zeros((N, K))
            for i, sid in enumerate(subject_ids_ordered):
                for k in range(K):
                    ll = _subject_log_lik(sid, sp[k])
                    L[i, k] = np.exp(np.clip(ll, -500, 0))
            row_sums = L.sum(axis=1)
            L[row_sums == 0, :] = 1.0 / K
            return L

        # ---- Step 2: Joint EM iterations --------------------------------
        prev_ll = -np.inf
        for _em_iter in range(self.max_iter):
            L_matrix = _compute_L(support_points)

            # E-step
            numerator = weights[np.newaxis, :] * L_matrix
            row_sums = np.maximum(numerator.sum(axis=1, keepdims=True), 1e-300)
            responsibilities = numerator / row_sums  # (N, K)

            # M-step weights
            weights = responsibilities.mean(axis=0)
            w_sum = weights.sum()
            weights /= w_sum if w_sum > 0 else 1.0

            # M-step locations (optional)
            if self.optimise_locations:
                for k in range(K):
                    r_k = responsibilities[:, k]  # shape (N,)
                    if r_k.sum() < 1e-10:
                        continue

                    def neg_weighted_ll(eta_flat: np.ndarray, _r_k=r_k) -> float:
                        total = 0.0
                        for i, sid in enumerate(subject_ids_ordered):
                            if _r_k[i] < 1e-8:
                                continue
                            total -= _r_k[i] * _subject_log_lik(sid, eta_flat)
                        return total

                    try:
                        res = minimize(
                            neg_weighted_ll,
                            support_points[k],
                            method="L-BFGS-B",
                            options={"maxiter": self.max_location_iter, "ftol": 1e-6},
                        )
                        if res.success or res.fun < neg_weighted_ll(support_points[k]):
                            support_points[k] = res.x
                    except Exception:
                        pass

            # Check convergence
            marginal = (weights[np.newaxis, :] * L_matrix).sum(axis=1)
            marginal = np.maximum(marginal, 1e-300)
            ll = float(np.sum(np.log(marginal)))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        # Compute empirical OMEGA
        mean_eta = np.sum(weights[:, np.newaxis] * support_points, axis=0)
        omega_np = np.sum(
            weights[:, np.newaxis, np.newaxis]
            * (
                (support_points - mean_eta)[:, :, np.newaxis]
                * (support_points - mean_eta)[:, np.newaxis, :]
            ),
            axis=0,
        )

        elapsed = time.time() - t0

        return NonparametricResult(
            theta_final=base_result.theta_final,
            omega_final=omega_np,
            sigma_final=base_result.sigma_final,
            ofv=base_result.ofv,
            converged=base_result.converged,
            post_hoc_etas=base_result.post_hoc_etas,
            ofv_history=base_result.ofv_history,
            n_function_evals=base_result.n_function_evals,
            elapsed_time=elapsed,
            method=self.method_name,
            message=base_result.message,
            support_points=support_points,
            support_weights=weights,
        )
