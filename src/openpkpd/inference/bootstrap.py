"""
Non-parametric bootstrap confidence intervals for population PK/PD parameters.

Resamples subjects *with replacement* (preserving within-subject correlation)
and re-fits the model on each replicate.  The resulting empirical distribution
of parameter estimates is used to compute percentile-based confidence intervals.

Typical usage::

    from openpkpd.inference.bootstrap import BootstrapEngine

    engine = BootstrapEngine(
        population_model=pop_model,
        initial_params=params,
        estimation_method="FOCE",
        n_boot=200,
        n_jobs=-1,   # use all CPUs
        seed=42,
        ci_level=0.95,
    )
    boot_result = engine.run()
    print(boot_result.summary())
    print("THETA 95% CI:", boot_result.theta_ci)

.. note::
    This module is self-contained and does not modify
    ``src/openpkpd/inference/__init__.py``.  Import directly from
    ``openpkpd.inference.bootstrap``.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

if TYPE_CHECKING:
    from openpkpd.estimation.base import EstimationResult as _EstimationResult

from openpkpd.estimation.base import EstimationResult

logger = logging.getLogger("openpkpd.inference.bootstrap")


# ── BCa confidence interval ────────────────────────────────────────────────────


def bca_ci(
    samples: np.ndarray,
    original_est: float,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """
    Bias-Corrected and Accelerated (BCa) confidence interval for one parameter.

    Implements Efron & Tibshirani (1993) §14.3.  The bias-correction constant
    ``z0`` is derived from the fraction of bootstrap samples below the original
    estimate.  The acceleration constant ``a`` is estimated via leave-one-out
    jackknife on the bootstrap mean (a computationally trivial O(n) operation
    that does not require re-running the model).

    Args:
        samples:      1-D array of bootstrap estimates for one parameter.
        original_est: Point estimate from the original (non-bootstrap) fit.
        ci_level:     Nominal coverage (default 0.95 → 95% CI).

    Returns:
        ``(lo, hi)`` BCa confidence interval bounds.
    """
    n = len(samples)
    alpha = (1.0 - ci_level) / 2.0
    z_alpha = _norm.ppf(alpha)
    z_1ma = _norm.ppf(1.0 - alpha)

    # Bias-correction constant z0
    prop_less = np.mean(samples < original_est)
    prop_less = np.clip(prop_less, 0.5 / n, 1.0 - 0.5 / n)
    z0 = float(_norm.ppf(prop_less))

    # Acceleration constant via LOO jackknife of the bootstrap mean
    total = float(np.sum(samples))
    jack_means = (total - samples) / (n - 1)  # shape (n,), O(n)
    jack_mean = float(np.mean(jack_means))
    diff = jack_mean - jack_means  # positive when jack mean > LOO mean
    denom = float(np.sum(diff**2) ** 1.5)
    a = float(np.sum(diff**3) / (6.0 * denom)) if denom > 0.0 else 0.0

    # Adjusted quantile levels
    def _adj_alpha(z_q: float) -> float:
        inner = z0 + z_q
        div = 1.0 - a * inner
        if div == 0.0:
            return alpha if z_q < 0 else 1.0 - alpha
        return float(_norm.cdf(z0 + inner / div))

    alpha1 = np.clip(_adj_alpha(z_alpha), 0.0, 1.0)
    alpha2 = np.clip(_adj_alpha(z_1ma), 0.0, 1.0)

    lo = float(np.percentile(samples, 100.0 * alpha1))
    hi = float(np.percentile(samples, 100.0 * alpha2))
    return lo, hi


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class BootstrapResult:
    """
    Results from a bootstrap confidence-interval run.

    Attributes:
        n_boot:               Total number of bootstrap replicates attempted.
        n_success:            Number of replicates that converged successfully.
        theta_samples:        THETA estimates from each replicate, shape
                              ``(n_success, n_theta)``.
        omega_diag_samples:   Diagonal of OMEGA from each replicate, shape
                              ``(n_success, n_eta)``.
        sigma_diag_samples:   Diagonal of SIGMA from each replicate, shape
                              ``(n_success, n_eps)``.
        ci_level:             Confidence level (default 0.95 = 95% CI).
    """

    n_boot: int
    n_success: int
    theta_samples: np.ndarray  # shape (n_success, n_theta)
    omega_diag_samples: np.ndarray  # shape (n_success, n_eta)
    sigma_diag_samples: np.ndarray  # shape (n_success, n_eps)
    ci_level: float = 0.95

    @property
    def theta_ci(self) -> np.ndarray:
        """
        Percentile-based CI for each THETA.

        Returns:
            Array of shape ``(n_theta, 2)`` where column 0 is the lower bound
            and column 1 is the upper bound.
        """
        alpha = (1.0 - self.ci_level) / 2.0
        lower = np.percentile(self.theta_samples, 100.0 * alpha, axis=0)
        upper = np.percentile(self.theta_samples, 100.0 * (1.0 - alpha), axis=0)
        return np.column_stack([lower, upper])

    @property
    def omega_diag_ci(self) -> np.ndarray:
        """
        Percentile-based CI for each diagonal element of OMEGA.

        Returns:
            Array of shape ``(n_eta, 2)``.
        """
        alpha = (1.0 - self.ci_level) / 2.0
        lower = np.percentile(self.omega_diag_samples, 100.0 * alpha, axis=0)
        upper = np.percentile(self.omega_diag_samples, 100.0 * (1.0 - alpha), axis=0)
        return np.column_stack([lower, upper])

    @property
    def sigma_diag_ci(self) -> np.ndarray:
        """
        Percentile-based CI for each diagonal element of SIGMA.

        Returns:
            Array of shape ``(n_eps, 2)``.
        """
        alpha = (1.0 - self.ci_level) / 2.0
        lower = np.percentile(self.sigma_diag_samples, 100.0 * alpha, axis=0)
        upper = np.percentile(self.sigma_diag_samples, 100.0 * (1.0 - alpha), axis=0)
        return np.column_stack([lower, upper])

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame with parameter estimates and bootstrap CIs.

        Columns: ``parameter``, ``mean``, ``median``, ``std``,
        ``ci_lower``, ``ci_upper``.

        The rows contain all THETAs followed by OMEGA diagonal elements and
        SIGMA diagonal elements.
        """
        rows: list[dict[str, Any]] = []
        ci_level_pct = int(self.ci_level * 100)

        # THETAs
        theta_ci = self.theta_ci
        for i in range(self.theta_samples.shape[1]):
            col = self.theta_samples[:, i]
            rows.append(
                {
                    "parameter": f"THETA({i + 1})",
                    "mean": float(np.mean(col)),
                    "median": float(np.median(col)),
                    "std": float(np.std(col)),
                    f"ci{ci_level_pct}_lower": float(theta_ci[i, 0]),
                    f"ci{ci_level_pct}_upper": float(theta_ci[i, 1]),
                }
            )

        # OMEGA diagonals
        omega_ci = self.omega_diag_ci
        for i in range(self.omega_diag_samples.shape[1]):
            col = self.omega_diag_samples[:, i]
            rows.append(
                {
                    "parameter": f"OMEGA({i + 1},{i + 1})",
                    "mean": float(np.mean(col)),
                    "median": float(np.median(col)),
                    "std": float(np.std(col)),
                    f"ci{ci_level_pct}_lower": float(omega_ci[i, 0]),
                    f"ci{ci_level_pct}_upper": float(omega_ci[i, 1]),
                }
            )

        # SIGMA diagonals
        sigma_ci = self.sigma_diag_ci
        for i in range(self.sigma_diag_samples.shape[1]):
            col = self.sigma_diag_samples[:, i]
            rows.append(
                {
                    "parameter": f"SIGMA({i + 1},{i + 1})",
                    "mean": float(np.mean(col)),
                    "median": float(np.median(col)),
                    "std": float(np.std(col)),
                    f"ci{ci_level_pct}_lower": float(sigma_ci[i, 0]),
                    f"ci{ci_level_pct}_upper": float(sigma_ci[i, 1]),
                }
            )

        return pd.DataFrame(rows)

    def ci_table(self, original_result: _EstimationResult) -> pd.DataFrame:
        """
        Return a DataFrame with percentile and BCa CIs for all parameters.

        BCa (Bias-Corrected and Accelerated) intervals use the original point
        estimates to correct for sampling bias and skewness in the bootstrap
        distribution, giving better coverage than plain percentile CIs.

        Args:
            original_result: :class:`EstimationResult` from the original
                             (non-bootstrap) model fit.  Provides the point
                             estimates required for BCa bias-correction.

        Returns:
            DataFrame with columns:

            * ``parameter`` — e.g. ``"THETA(1)"``, ``"OMEGA(1,1)"``
            * ``mean``      — bootstrap mean
            * ``se``        — bootstrap standard deviation (≈ standard error)
            * ``p2_5``      — lower percentile CI bound
            * ``p97_5``     — upper percentile CI bound
            * ``bca_lo``    — lower BCa CI bound
            * ``bca_hi``    — upper BCa CI bound

        Notes:
            Column names ``p2_5`` / ``p97_5`` always correspond to the
            ``(1−ci_level)/2`` and ``1−(1−ci_level)/2`` quantiles (2.5th /
            97.5th when ``ci_level=0.95``).
        """
        alpha = (1.0 - self.ci_level) / 2.0
        pct_lo = 100.0 * alpha
        pct_hi = 100.0 * (1.0 - alpha)

        rows: list[dict[str, Any]] = []

        # THETAs
        orig_theta = original_result.theta_final
        for i in range(self.theta_samples.shape[1]):
            col = self.theta_samples[:, i]
            lo, hi = bca_ci(col, float(orig_theta[i]), self.ci_level)
            rows.append(
                {
                    "parameter": f"THETA({i + 1})",
                    "mean": float(np.mean(col)),
                    "se": float(np.std(col)),
                    "p2_5": float(np.percentile(col, pct_lo)),
                    "p97_5": float(np.percentile(col, pct_hi)),
                    "bca_lo": lo,
                    "bca_hi": hi,
                }
            )

        # OMEGA diagonals
        orig_omega_diag = np.diag(original_result.omega_final)
        for i in range(self.omega_diag_samples.shape[1]):
            col = self.omega_diag_samples[:, i]
            lo, hi = bca_ci(col, float(orig_omega_diag[i]), self.ci_level)
            rows.append(
                {
                    "parameter": f"OMEGA({i + 1},{i + 1})",
                    "mean": float(np.mean(col)),
                    "se": float(np.std(col)),
                    "p2_5": float(np.percentile(col, pct_lo)),
                    "p97_5": float(np.percentile(col, pct_hi)),
                    "bca_lo": lo,
                    "bca_hi": hi,
                }
            )

        # SIGMA diagonals
        orig_sigma_diag = np.diag(original_result.sigma_final)
        for i in range(self.sigma_diag_samples.shape[1]):
            col = self.sigma_diag_samples[:, i]
            lo, hi = bca_ci(col, float(orig_sigma_diag[i]), self.ci_level)
            rows.append(
                {
                    "parameter": f"SIGMA({i + 1},{i + 1})",
                    "mean": float(np.mean(col)),
                    "se": float(np.std(col)),
                    "p2_5": float(np.percentile(col, pct_lo)),
                    "p97_5": float(np.percentile(col, pct_hi)),
                    "bca_lo": lo,
                    "bca_hi": hi,
                }
            )

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        pct = int(self.ci_level * 100)
        return (
            f"BootstrapResult("
            f"n_boot={self.n_boot}, "
            f"n_success={self.n_success}, "
            f"n_theta={self.theta_samples.shape[1]}, "
            f"ci_level={pct}%)"
        )


# ── Engine ─────────────────────────────────────────────────────────────────────


class BootstrapEngine:
    """
    Bootstrap confidence interval estimation.

    Resamples the subject-level data (not individual observations) with
    replacement to preserve within-subject correlation structure.  Each
    bootstrap replicate consists of ``n_subjects`` subjects drawn with
    replacement from the original dataset.

    Args:
        population_model:    A fitted (or at least configured) PopulationModel
                             whose ``dataset`` attribute is a NONMEMDataset.
        initial_params:      Starting ParameterSet for each bootstrap re-fit.
                             Typically the final estimates from the original fit.
        estimation_method:   Estimation method name (e.g., ``'FOCE'``).
        n_boot:              Number of bootstrap replicates (default 200).
        n_jobs:              Number of parallel worker processes.
                             ``-1`` uses ``os.cpu_count()``.
                             ``1`` runs sequentially (easier to debug).
        seed:                Base random seed for reproducibility.
        ci_level:            Confidence level for the output CIs (default 0.95).
        **estimation_kwargs: Additional keyword arguments forwarded to the
                             estimation method (e.g., ``maxeval=500``).
    """

    def __init__(
        self,
        population_model: Any,
        initial_params: Any,
        estimation_method: str = "FOCE",
        n_boot: int = 200,
        n_jobs: int = -1,
        seed: int = 42,
        ci_level: float = 0.95,
        **estimation_kwargs: Any,
    ) -> None:
        self.population_model = population_model
        self.initial_params = initial_params
        self.estimation_method = estimation_method
        self.n_boot = n_boot
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(seed)
        self.ci_level = ci_level
        self.estimation_kwargs = estimation_kwargs

        if not (0.0 < ci_level < 1.0):
            raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> BootstrapResult:
        """
        Run the bootstrap by resampling subjects and re-fitting.

        Returns:
            BootstrapResult with parameter samples and CI estimates.
        """
        # Pre-generate seeds for each replicate (reproducible regardless of
        # execution order or parallelism)
        seeds = self.rng.integers(0, 2**31 - 1, size=self.n_boot).tolist()

        n_workers = self._resolve_n_jobs()
        logger.info(
            "Starting bootstrap: n_boot=%d, n_jobs=%d, method=%s",
            self.n_boot,
            n_workers,
            self.estimation_method,
        )

        results: list[EstimationResult | None]
        if n_workers == 1:
            results = [self._fit_one(s) for s in seeds]
        else:
            results = self._run_parallel(seeds, n_workers)

        # Collect successful results
        successful = [r for r in results if r is not None and getattr(r, "converged", False)]
        n_success = len(successful)
        n_failed = self.n_boot - n_success
        if n_failed:
            logger.warning(
                "%d / %d bootstrap replicates failed to converge.",
                n_failed,
                self.n_boot,
            )

        if n_success == 0:
            raise RuntimeError("All bootstrap replicates failed.  Check model specification.")

        n_total = self.n_boot
        if n_success < max(1, int(0.1 * n_total)):
            import warnings
            warnings.warn(
                f"Bootstrap: only {n_success}/{n_total} replicates converged"
                f" ({100 * n_success / n_total:.0f}%). Results may be unreliable.",
                RuntimeWarning,
            )

        theta_samples = np.array([r.theta_final for r in successful])
        omega_diag_samples = np.array([np.diag(r.omega_final) for r in successful])
        sigma_diag_samples = np.array([np.diag(r.sigma_final) for r in successful])

        logger.info("Bootstrap complete: %d/%d successful.", n_success, self.n_boot)

        return BootstrapResult(
            n_boot=self.n_boot,
            n_success=n_success,
            theta_samples=theta_samples,
            omega_diag_samples=omega_diag_samples,
            sigma_diag_samples=sigma_diag_samples,
            ci_level=self.ci_level,
        )

    # ── Subject resampling ─────────────────────────────────────────────────────

    def _resample_subjects(self, rng_seed: int) -> Any:
        """
        Create a new PopulationModel by resampling subjects with replacement.

        Subjects are sampled at the ID level.  Each replicate has the same
        number of subjects as the original dataset, but some subjects may
        appear multiple times (with duplicated IDs renumbered to avoid
        collisions in the estimation engine).

        Args:
            rng_seed: Integer seed for this replicate's RNG.

        Returns:
            A new PopulationModel built from the resampled dataset.
        """

        from openpkpd.data.dataset import NONMEMDataset
        from openpkpd.model.population import PopulationModel

        rng = np.random.default_rng(rng_seed)
        original_ids = self.population_model.dataset.subject_ids()
        n_subj = len(original_ids)

        # Sample IDs with replacement
        sampled_ids = rng.choice(original_ids, size=n_subj, replace=True)

        # Build resampled DataFrame; renumber IDs sequentially
        df = self.population_model.dataset.df
        frames: list[pd.DataFrame] = []
        for new_id, orig_id in enumerate(sampled_ids, start=1):
            subj_df = df[df["ID"] == orig_id].copy()
            subj_df = subj_df.copy()
            subj_df["ID"] = new_id
            frames.append(subj_df)

        import pandas as _pd

        new_df = (
            _pd.concat(frames, ignore_index=True)
            .sort_values(["ID", "TIME"], kind="stable")
            .reset_index(drop=True)
        )

        new_dataset = NONMEMDataset(df=new_df)

        # Clone the population model with the new dataset
        new_pop = PopulationModel(
            dataset=new_dataset,
            pk_subroutine=self.population_model.pk_subroutine,
            params=self.initial_params,
            pk_callable=self.population_model.pk_callable,
            error_callable=self.population_model.error_callable,
            des_callable=self.population_model.des_callable,
            trans=self.population_model.trans,
            advan=self.population_model.advan,
            covariate_columns=list(self.population_model.covariate_columns),
        )
        return new_pop

    # ── Single-replicate fit ────────────────────────────────────────────────────

    def _fit_one(self, resample_seed: int) -> EstimationResult | None:
        """
        Fit on one bootstrap replicate.

        Args:
            resample_seed: Seed used to resample subjects for this replicate.

        Returns:
            EstimationResult on success, None on any failure.
        """
        from openpkpd.estimation import get_estimation_method

        try:
            new_pop = self._resample_subjects(resample_seed)
            est = get_estimation_method(
                self.estimation_method,
                **self.estimation_kwargs,
            )
            result = est.estimate(new_pop, self.initial_params)
            if not getattr(result, "converged", False):
                logger.debug(
                    "Bootstrap replicate (seed=%d) did not converge.",
                    resample_seed,
                )
                return None
            return result
        except Exception as exc:
            logger.debug("Bootstrap replicate (seed=%d) failed: %s", resample_seed, exc)
            return None

    # ── Parallel execution ─────────────────────────────────────────────────────

    def _run_parallel(
        self,
        seeds: list[int],
        n_workers: int,
    ) -> list[EstimationResult | None]:
        """
        Run bootstrap replicates in parallel using ProcessPoolExecutor.

        Args:
            seeds:      Pre-generated random seeds for each replicate.
            n_workers:  Number of worker processes.

        Returns:
            List of EstimationResult (or None for failures), in seed order.
        """
        # We cannot pickle self directly because population_model may contain
        # non-picklable callables.  Fall back to sequential if pickling fails.
        try:
            return self._run_parallel_impl(seeds, n_workers)
        except Exception as exc:
            logger.warning("Parallel bootstrap failed (%s); falling back to sequential.", exc)
            return [self._fit_one(s) for s in seeds]

    def _run_parallel_impl(
        self,
        seeds: list[int],
        n_workers: int,
    ) -> list[EstimationResult | None]:
        """Inner parallel implementation using ProcessPoolExecutor."""
        results: dict[int, EstimationResult | None] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(_fit_one_worker, self, seed): i for i, seed in enumerate(seeds)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.debug("Worker for index %d failed: %s", idx, exc)
                    results[idx] = None

        return [results[i] for i in range(len(seeds))]

    def _resolve_n_jobs(self) -> int:
        """Resolve n_jobs=-1 to the actual CPU count."""
        if self.n_jobs == -1:
            return os.cpu_count() or 1
        return max(1, self.n_jobs)


# ── Module-level worker (must be picklable) ────────────────────────────────────


def _fit_one_worker(engine: BootstrapEngine, seed: int) -> EstimationResult | None:
    """
    Top-level worker function for ProcessPoolExecutor.

    Defined at module level so it is picklable on all platforms.
    """
    return engine._fit_one(seed)


# ── Public exports ─────────────────────────────────────────────────────────────

__all__ = ["BootstrapResult", "BootstrapEngine", "bca_ci"]
