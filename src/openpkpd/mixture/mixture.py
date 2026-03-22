"""
Mixture models for population heterogeneity.

Implements a discrete mixture of K subpopulations, each with potentially
different PK/PD parameters.  The model is fitted using an EM algorithm:

  * **E-step**: compute the posterior probability that subject *i* belongs to
    subpopulation *k*, proportional to ``mixing_prob_k * L(y_i | params_k)``.
  * **M-step**: update mixing proportions from the posterior sums; update
    each subpopulation's parameters by re-fitting on the data weighted by
    per-subject posterior probabilities.

The log-likelihood of the full mixture model is::

    log L = Σ_i log [ Σ_k π_k * L(y_i | params_k) ]

Usage::

    from openpkpd.mixture.mixture import MixtureModel

    model = MixtureModel(
        population_model=pop_model,
        n_subpop=2,
        max_iter=50,
        tol=1e-4,
    )
    result = model.fit(init_params=base_params)
    print(f"Mixing proportions: {result.mixture_probs}")
    print(f"Subpopulation OFVs: {[r.ofv for r in result.subpop_results]}")
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from openpkpd.estimation.base import EstimationResult

logger = logging.getLogger("openpkpd.mixture.mixture")


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class MixtureResult:
    """
    Result from mixture model estimation.

    Attributes:
        n_subpop:              Number of subpopulations.
        mixture_probs:         Mixing proportions π_k, shape ``(n_subpop,)``.
                               Sum to 1.0.
        subpop_probabilities:  Per-subject posterior probabilities.
                               Dict mapping ``subject_id → array(n_subpop,)``.
        subpop_results:        EstimationResult for each subpopulation.
        ofv:                   Mixture model OFV (-2 * log mixture likelihood).
        converged:             True if EM converged within tolerance.
    """

    n_subpop: int
    mixture_probs: np.ndarray  # (n_subpop,)
    subpop_probabilities: dict[int, np.ndarray]  # {subject_id: (n_subpop,)}
    subpop_results: list[EstimationResult]  # one per subpopulation
    ofv: float
    converged: bool

    def summary(self) -> str:
        """Return a text summary of the mixture model result."""
        lines = [
            "=" * 55,
            "Mixture Model Result",
            "=" * 55,
            f"Number of subpopulations : {self.n_subpop}",
            f"OFV                      : {self.ofv:.4f}",
            f"Converged                : {self.converged}",
            "",
            "Mixing proportions:",
        ]
        for k, pi in enumerate(self.mixture_probs):
            lines.append(f"  Subpop {k + 1}: {pi:.4f}")
        lines.append("")
        lines.append("Subpopulation THETA estimates:")
        for k, res in enumerate(self.subpop_results):
            lines.append(
                f"  Subpop {k + 1}: THETA={np.round(res.theta_final, 4).tolist()}"
                f"  OFV={res.ofv:.4f}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)

    def subject_assignments(self, threshold: float = 0.5) -> dict[int, int]:
        """
        Return the most likely subpopulation for each subject.

        A subject is assigned to subpopulation *k* (1-based) when its
        posterior probability for that subpopulation exceeds *threshold*.
        Subjects below threshold for all subpopulations are assigned to the
        subpopulation with the highest probability.

        Args:
            threshold: Minimum posterior probability for a hard assignment.

        Returns:
            Dict mapping ``subject_id → subpopulation index (1-based)``.
        """
        assignments: dict[int, int] = {}
        for sid, probs in self.subpop_probabilities.items():
            best_k = int(np.argmax(probs))
            assignments[sid] = best_k + 1  # 1-based
        return assignments

    def __repr__(self) -> str:
        return (
            f"MixtureResult("
            f"n_subpop={self.n_subpop}, "
            f"ofv={self.ofv:.4f}, "
            f"converged={self.converged}, "
            f"probs={np.round(self.mixture_probs, 3).tolist()})"
        )


# ── Engine ─────────────────────────────────────────────────────────────────────


class MixtureModel:
    """
    Population mixture model with K subpopulations.

    Each subpopulation has its own THETA vector (and, optionally, its own
    OMEGA/SIGMA if ``shared_variance=False``).  The proportion of subjects in
    each subpopulation is estimated jointly.

    Args:
        population_model:  A configured PopulationModel whose dataset will be
                           used for fitting.
        n_subpop:          Number of discrete subpopulations (default 2).
        max_iter:          Maximum EM iterations (default 100).
        tol:               Convergence tolerance on the change in log-likelihood
                           between consecutive EM iterations (default 1e-4).
        estimation_method: Method name used to fit each subpopulation
                           (default ``'FOCE'``).
        shared_variance:   If True, share OMEGA/SIGMA across subpopulations and
                           only allow THETA to differ.  If False, each
                           subpopulation has its own OMEGA/SIGMA (more flexible
                           but more parameters).  Default True.
        estimation_kwargs: Extra keyword arguments forwarded to the estimation
                           method.
    """

    def __init__(
        self,
        population_model: Any,
        n_subpop: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        estimation_method: str = "FOCE",
        shared_variance: bool = True,
        estimation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if n_subpop < 2:
            raise ValueError(f"n_subpop must be >= 2, got {n_subpop}")
        self.population_model = population_model
        self.n_subpop = n_subpop
        self.max_iter = max_iter
        self.tol = tol
        self.estimation_method = estimation_method
        self.shared_variance = shared_variance
        self.estimation_kwargs: dict[str, Any] = estimation_kwargs or {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, init_params: Any) -> MixtureResult:
        """
        Fit the mixture model via the EM algorithm.

        Args:
            init_params: Initial ParameterSet.  Each subpopulation is
                         initialised from this set (with small perturbations to
                         break symmetry).

        Returns:
            MixtureResult containing mixing proportions, per-subject posteriors,
            per-subpopulation parameter estimates, and the mixture OFV.
        """

        subject_ids = self.population_model.subject_ids()
        n_subj = len(subject_ids)

        # Initialise mixing proportions uniformly
        mixing_probs = np.ones(self.n_subpop) / self.n_subpop

        # Initialise per-subpopulation parameters with small random perturbations
        params_per_subpop = self._init_params(init_params)

        # Initialise posteriors uniformly
        posteriors = np.full((n_subj, self.n_subpop), 1.0 / self.n_subpop)

        prev_ll = -np.inf
        converged = False

        for em_iter in range(self.max_iter):
            # ── E-step ───────────────────────────────────────────────────────
            posteriors = self._e_step(params_per_subpop, mixing_probs)

            # ── M-step ───────────────────────────────────────────────────────
            mixing_probs, params_per_subpop = self._m_step(
                posteriors, params_per_subpop, init_params, subject_ids
            )

            # ── Log-likelihood ───────────────────────────────────────────────
            ll = self._mixture_log_likelihood(params_per_subpop, mixing_probs)

            logger.info(
                "EM iter %d: log-L=%.4f  mixing=%s",
                em_iter + 1,
                ll,
                np.round(mixing_probs, 3).tolist(),
            )

            if abs(ll - prev_ll) < self.tol and em_iter > 0:
                converged = True
                logger.info("EM converged at iteration %d.", em_iter + 1)
                break
            prev_ll = ll

        # Build final per-subpopulation EstimationResult objects
        subpop_results = self._build_subpop_results(params_per_subpop)

        # Map posteriors back to subject_id dict
        subpop_probs_dict = {sid: posteriors[i] for i, sid in enumerate(subject_ids)}

        # Mixture OFV = -2 * log-likelihood
        ofv = -2.0 * self._mixture_log_likelihood(params_per_subpop, mixing_probs)

        return MixtureResult(
            n_subpop=self.n_subpop,
            mixture_probs=mixing_probs,
            subpop_probabilities=subpop_probs_dict,
            subpop_results=subpop_results,
            ofv=ofv,
            converged=converged,
        )

    # ── EM steps ───────────────────────────────────────────────────────────────

    def _e_step(
        self,
        params_per_subpop: list[Any],
        mixing_probs: np.ndarray,
    ) -> np.ndarray:
        """
        E-step: compute posterior probability of each subject per subpopulation.

        For each subject *i* and subpopulation *k*::

            r_ik = π_k * L_k(y_i) / Σ_j π_j * L_j(y_i)

        where ``L_k(y_i)`` is approximated by the FO marginal likelihood under
        subpopulation *k*'s parameters.

        Args:
            params_per_subpop: List of ParameterSet (one per subpopulation).
            mixing_probs:      Current mixing proportions π_k.

        Returns:
            Posterior matrix of shape ``(n_subjects, n_subpop)``.
        """
        subject_ids = self.population_model.subject_ids()
        n_subj = len(subject_ids)
        log_responsibilities = np.zeros((n_subj, self.n_subpop))

        for k, params_k in enumerate(params_per_subpop):
            log_pi_k = np.log(max(mixing_probs[k], 1e-300))
            for i, sid in enumerate(subject_ids):
                ll_i = self._subject_log_likelihood(sid, params_k)
                log_responsibilities[i, k] = log_pi_k + ll_i

        # Normalise in log-space for numerical stability
        log_row_max = log_responsibilities.max(axis=1, keepdims=True)
        log_responsibilities -= log_row_max  # shift
        responsibilities = np.exp(log_responsibilities)
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        return responsibilities / row_sums

    def _m_step(
        self,
        posteriors: np.ndarray,
        params_per_subpop: list[Any],
        init_params: Any,
        subject_ids: list[int],
    ) -> tuple[np.ndarray, list[Any]]:
        """
        M-step: update mixing proportions and per-subpopulation parameters.

        Mixing proportions are updated as the mean posterior probability per
        subpopulation.  Per-subpopulation parameters are re-estimated by
        fitting only on subjects whose posterior probability for that
        subpopulation exceeds a soft threshold, or by using all subjects with
        weighted contributions (via subject-level sample weights).

        In this implementation we use a *hard-assignment* approximation for
        simplicity: each subject is assigned to the subpopulation with the
        highest posterior.  Parameters for each subpopulation are re-estimated
        on that subset.  This is equivalent to the classification EM (CEM)
        algorithm and converges faster at the cost of a cruder approximation.

        Args:
            posteriors:        Shape ``(n_subjects, n_subpop)`` posterior matrix.
            params_per_subpop: Current parameter sets.
            init_params:       Original initial parameters (used as fallback).
            subject_ids:       Ordered list of subject IDs.

        Returns:
            Tuple of (updated mixing_probs, updated params_per_subpop).
        """
        from openpkpd.estimation import get_estimation_method

        # Update mixing proportions
        new_mixing = posteriors.mean(axis=0)
        new_mixing = np.maximum(new_mixing, 1e-6)
        new_mixing /= new_mixing.sum()

        new_params: list[Any] = []
        for k in range(self.n_subpop):
            # Hard assignment: use subjects with highest posterior for this subpop
            assignments = np.argmax(posteriors, axis=1)
            subpop_mask = assignments == k
            subpop_ids = [sid for i, sid in enumerate(subject_ids) if subpop_mask[i]]

            if not subpop_ids:
                # Fallback: keep previous parameters
                logger.warning(
                    "Subpopulation %d has no assigned subjects in M-step; "
                    "keeping previous parameters.",
                    k + 1,
                )
                new_params.append(params_per_subpop[k])
                continue

            # Build a restricted population model with only the assigned subjects
            restricted_model = self._restrict_to_subjects(subpop_ids)

            try:
                est = get_estimation_method(self.estimation_method, **self.estimation_kwargs)
                result = est.estimate(restricted_model, params_per_subpop[k])
                # Reconstruct a ParameterSet from the result
                updated_params = copy.deepcopy(params_per_subpop[k])
                updated_params.theta = result.theta_final.copy()
                if not self.shared_variance:
                    updated_params.omega = result.omega_final.copy()
                    updated_params.sigma = result.sigma_final.copy()
                new_params.append(updated_params)
            except Exception as exc:
                logger.warning(
                    "M-step estimation failed for subpop %d: %s; keeping previous parameters.",
                    k + 1,
                    exc,
                )
                new_params.append(params_per_subpop[k])

        return new_mixing, new_params

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _init_params(self, base_params: Any) -> list[Any]:
        """
        Initialise one ParameterSet per subpopulation.

        All subpopulations start from *base_params* with small multiplicative
        perturbations on THETA to break symmetry.
        """
        params_list: list[Any] = []
        rng = np.random.default_rng(seed=0)
        for k in range(self.n_subpop):
            p = copy.deepcopy(base_params)
            # Add ±10% noise to THETA (except the first subpop which starts unperturbed)
            if k > 0:
                noise = rng.uniform(0.9, 1.1, size=p.theta.shape)
                p.theta = p.theta * noise
            params_list.append(p)
        return params_list

    def _restrict_to_subjects(self, subject_ids: list[int]) -> Any:
        """
        Return a new PopulationModel restricted to the given subject IDs.

        Args:
            subject_ids: Subset of subject IDs to include.

        Returns:
            A new PopulationModel whose dataset contains only those subjects.
        """
        from openpkpd.data.dataset import NONMEMDataset
        from openpkpd.model.population import PopulationModel

        df = self.population_model.dataset.df
        restricted_df = df[df["ID"].isin(subject_ids)].reset_index(drop=True)
        new_dataset = NONMEMDataset(df=restricted_df)

        return PopulationModel(
            dataset=new_dataset,
            pk_subroutine=self.population_model.pk_subroutine,
            params=self.population_model.params,
            pk_callable=self.population_model.pk_callable,
            error_callable=self.population_model.error_callable,
            des_callable=self.population_model.des_callable,
            trans=self.population_model.trans,
            advan=self.population_model.advan,
            covariate_columns=list(self.population_model.covariate_columns),
        )

    def _subject_log_likelihood(self, subject_id: int, params: Any) -> float:
        """
        Compute the approximate individual log-likelihood for a single subject
        under the given parameter set, evaluated at eta=0 (FO approximation).

        Args:
            subject_id:  Subject ID to evaluate.
            params:      ParameterSet to use.

        Returns:
            Approximate log-likelihood (scalar).
        """
        try:
            indiv = self.population_model.individual_model(subject_id)
            eta_zero = np.zeros(params.n_eta())
            ofv_i = indiv.log_likelihood(
                params.theta,
                eta_zero,
                params.sigma,
                trans=self.population_model.trans,
            )
            # ofv_i = -2 * log-likelihood  =>  ll = -ofv_i / 2
            return -ofv_i / 2.0
        except Exception:
            return -1e10  # Very unlikely for this subject under these params

    def _mixture_log_likelihood(
        self,
        params_per_subpop: list[Any],
        mixing_probs: np.ndarray,
    ) -> float:
        """
        Compute the total mixture log-likelihood.

        log L = Σ_i log [ Σ_k π_k * exp(ll_ik) ]

        Uses the log-sum-exp trick for numerical stability.

        Args:
            params_per_subpop: Parameter sets per subpopulation.
            mixing_probs:      Mixing proportions.

        Returns:
            Total log-likelihood (scalar).
        """
        subject_ids = self.population_model.subject_ids()
        n_components = len(params_per_subpop)
        if len(mixing_probs) != n_components:
            raise ValueError("mixing_probs length must match number of parameter sets")
        ll_total = 0.0

        for sid in subject_ids:
            # Per-subpopulation log-likelihoods for this subject
            ll_k = np.array(
                [
                    self._subject_log_likelihood(sid, params_per_subpop[k])
                    for k in range(n_components)
                ]
            )
            # log-sum-exp:  log Σ_k π_k * exp(ll_k)
            log_probs = np.log(np.maximum(mixing_probs, 1e-300)) + ll_k
            log_sum = float(np.max(log_probs)) + np.log(
                np.sum(np.exp(log_probs - np.max(log_probs)))
            )
            ll_total += log_sum

        return float(ll_total)

    def _build_subpop_results(
        self,
        params_per_subpop: list[Any],
    ) -> list[EstimationResult]:
        """
        Build a minimal EstimationResult for each subpopulation from the
        final parameter sets.

        Args:
            params_per_subpop: Final ParameterSet per subpopulation.

        Returns:
            List of EstimationResult, one per subpopulation.
        """
        results: list[EstimationResult] = []
        for k, params_k in enumerate(params_per_subpop):
            # Approximate per-subpopulation OFV using FO marginal likelihood
            ll = self._mixture_log_likelihood([params_k], np.array([1.0]))
            results.append(
                EstimationResult(
                    theta_final=params_k.theta.copy(),
                    omega_final=params_k.omega.copy(),
                    sigma_final=params_k.sigma.copy(),
                    ofv=-2.0 * ll,
                    converged=True,
                    method=self.estimation_method,
                    message=f"Subpopulation {k + 1} final parameters",
                )
            )
        return results


# ── Public exports ─────────────────────────────────────────────────────────────

__all__ = ["MixtureResult", "MixtureModel"]
